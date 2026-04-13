use std::{
    cmp::Ordering,
    collections::HashMap,
    env,
    fs,
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::{Context, anyhow};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sim_core::{
    CURRENT_SNAPSHOT_SCHEMA_VERSION, Diagnostics, InitialConditions, Particle, SimulationConfig,
    SnapshotManifest, Vec3, validate_particle_count, write_particle_snapshot,
};
use sim_cuda::GpuBackend;
use tokio::sync::{broadcast, mpsc, oneshot};
use tracing::{error, info, warn};
use uuid::Uuid;

#[derive(Clone, Default)]
pub struct SessionRegistry {
    sessions: Arc<RwLock<HashMap<Uuid, SessionHandle>>>,
}

impl SessionRegistry {
    pub fn list(&self) -> Vec<SessionSummary> {
        let mut sessions: Vec<_> = self
            .sessions
            .read()
            .values()
            .map(SessionHandle::summary)
            .collect();
        sessions.sort_by(|left, right| {
            session_rank(left.state)
                .cmp(&session_rank(right.state))
                .then_with(|| {
                    right
                        .sim_time_myr
                        .partial_cmp(&left.sim_time_myr)
                        .unwrap_or(Ordering::Equal)
                })
        });
        sessions
    }

    pub fn get(&self, id: Uuid) -> Option<SessionSummary> {
        self.sessions.read().get(&id).map(SessionHandle::summary)
    }

    pub fn handle(&self, id: Uuid) -> Option<SessionHandle> {
        self.sessions.read().get(&id).cloned()
    }

    pub fn create(&self, params: CreateSessionParams) -> anyhow::Result<SessionSummary> {
        let particle_count = validate_particle_count(&params.config)?;
        ensure_equilibrium_snapshots_for_config(&params.config, &params.preset_id, params.seed)
            .context("failed to prepare equilibrium snapshot cache")?;
        let mut initial_conditions = InitialConditions::generate(&params.config, params.seed)
            .context("failed to generate initial conditions")?;
        if should_relax_initial_conditions(particle_count, &params.config) {
            info!(
                "relaxing initial conditions for preset {} ({} particles)",
                params.preset_id, particle_count
            );
            initial_conditions = relax_initial_conditions_if_needed(&params.config, initial_conditions)
                .context("failed to relax initial conditions")?;
        }

        let id = Uuid::new_v4();
        let preview_budget = params
            .preview_particle_budget
            .unwrap_or(params.config.preview.particle_budget);
        let summary = SessionSummary {
            id,
            preset_id: params.preset_id.clone(),
            name: params.config.name.clone(),
            state: SessionState::Paused,
            particle_count,
            sim_time_myr: 0.0,
            preview_particle_budget: preview_budget,
            latest_snapshot: None,
            diagnostics: Diagnostics {
                particle_count,
                ..Diagnostics::default()
            },
        };

        let (frame_tx, _) = broadcast::channel(16);
        let (command_tx, command_rx) = mpsc::unbounded_channel();
        let latest_frame = Arc::new(RwLock::new(None));
        let summary_arc = Arc::new(RwLock::new(summary.clone()));

        let handle = SessionHandle {
            id,
            summary: Arc::clone(&summary_arc),
            latest_frame: Arc::clone(&latest_frame),
            frame_tx: frame_tx.clone(),
            command_tx: command_tx.clone(),
        };

        self.sessions.write().insert(id, handle.clone());

        tokio::spawn(session_task(
            id,
            params.config,
            preview_budget,
            initial_conditions,
            summary_arc,
            latest_frame,
            frame_tx,
            command_rx,
        ));

        info!("created simulation session {id}");
        Ok(summary)
    }

    pub fn command(&self, id: Uuid, command: SessionCommand) -> anyhow::Result<()> {
        let Some(handle) = self.sessions.read().get(&id).cloned() else {
            return Err(anyhow!("unknown session `{id}`"));
        };
        handle.send_command(command)
    }

    pub async fn command_wait(
        &self,
        id: Uuid,
        command: SessionCommand,
    ) -> anyhow::Result<SessionSummary> {
        let Some(handle) = self.sessions.read().get(&id).cloned() else {
            return Err(anyhow!("unknown session `{id}`"));
        };
        handle.send_command_wait(command).await
    }

    pub async fn stop(&self, id: Uuid) -> anyhow::Result<SessionSummary> {
        let Some(handle) = self.sessions.read().get(&id).cloned() else {
            return Err(anyhow!("unknown session `{id}`"));
        };
        let summary = handle.send_command_wait(SessionCommand::Stop).await?;
        self.sessions.write().remove(&id);
        Ok(summary)
    }
}

#[derive(Clone)]
pub struct SessionHandle {
    pub id: Uuid,
    pub summary: Arc<RwLock<SessionSummary>>,
    pub latest_frame: Arc<RwLock<Option<Vec<u8>>>>,
    pub frame_tx: broadcast::Sender<Vec<u8>>,
    pub command_tx: mpsc::UnboundedSender<SessionRequest>,
}

impl SessionHandle {
    pub fn summary(&self) -> SessionSummary {
        self.summary.read().clone()
    }

    pub fn latest_frame(&self) -> Option<Vec<u8>> {
        self.latest_frame.read().clone()
    }

    pub fn subscribe_frames(&self) -> broadcast::Receiver<Vec<u8>> {
        self.frame_tx.subscribe()
    }

    pub fn send_command(&self, command: SessionCommand) -> anyhow::Result<()> {
        self.command_tx
            .send(SessionRequest { command, ack: None })
            .map_err(|_| anyhow!("session {} is no longer accepting commands", self.id))
    }

    pub async fn send_command_wait(
        &self,
        command: SessionCommand,
    ) -> anyhow::Result<SessionSummary> {
        let (tx, rx) = oneshot::channel();
        self.command_tx
            .send(SessionRequest {
                command,
                ack: Some(tx),
            })
            .map_err(|_| anyhow!("session {} is no longer accepting commands", self.id))?;
        rx.await
            .map_err(|_| anyhow!("session {} command acknowledgement dropped", self.id))?
    }
}

#[derive(Debug, Clone)]
pub struct CreateSessionParams {
    pub config: SimulationConfig,
    pub preset_id: String,
    pub seed: u64,
    pub preview_particle_budget: Option<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSummary {
    pub id: Uuid,
    pub preset_id: String,
    pub name: String,
    pub state: SessionState,
    pub particle_count: u64,
    pub sim_time_myr: f64,
    pub preview_particle_budget: u32,
    pub latest_snapshot: Option<String>,
    pub diagnostics: Diagnostics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Paused,
    Running,
    Failed,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ControlCommand {
    Pause,
    Resume,
    Step { substeps: Option<u32> },
    SetPreviewBudget { particle_budget: u32 },
}

#[derive(Debug, Clone)]
pub enum SessionCommand {
    Pause,
    Resume,
    Step(u32),
    Snapshot,
    Stop,
    Control(ControlCommand),
}

#[derive(Debug)]
pub struct SessionRequest {
    pub command: SessionCommand,
    pub ack: Option<oneshot::Sender<anyhow::Result<SessionSummary>>>,
}

fn ensure_equilibrium_snapshots_for_config(
    config: &SimulationConfig,
    preset_id: &str,
    seed: u64,
) -> anyhow::Result<()> {
    if preset_id == "custom" {
        return Ok(());
    }

    for (galaxy_index, galaxy) in config.galaxies.iter().enumerate() {
        let Some(snapshot_path) = galaxy.equilibrium_snapshot.as_deref() else {
            continue;
        };
        if equilibrium_snapshot_is_current(snapshot_path)? {
            continue;
        }
        generate_equilibrium_snapshot(preset_id, galaxy_index, snapshot_path, seed)
            .with_context(|| format!("generate equilibrium snapshot for {preset_id} galaxy {galaxy_index}"))?;
    }

    Ok(())
}

fn equilibrium_snapshot_is_current(snapshot_path: &str) -> anyhow::Result<bool> {
    let manifest_path = Path::new(snapshot_path);
    if !manifest_path.exists() {
        return Ok(false);
    }
    let bytes = fs::read(manifest_path)
        .with_context(|| format!("failed to read equilibrium manifest {}", manifest_path.display()))?;
    let manifest: SnapshotManifest = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "failed to decode equilibrium manifest {}",
            manifest_path.display()
        )
    })?;
    Ok(manifest.schema_version >= CURRENT_SNAPSHOT_SCHEMA_VERSION)
}

fn generate_equilibrium_snapshot(
    preset_id: &str,
    galaxy_index: usize,
    snapshot_path: &str,
    seed: u64,
) -> anyhow::Result<()> {
    let snapshot_path = PathBuf::from(snapshot_path);
    if let Some(parent) = snapshot_path.parent() {
        fs::create_dir_all(parent)
            .with_context(|| format!("create equilibrium snapshot directory {}", parent.display()))?;
    }

    let mut command = resolve_equilibrate_command();
    command
        .arg("--preset")
        .arg(preset_id)
        .arg("--galaxy")
        .arg(galaxy_index.to_string())
        .arg("--iterations")
        .arg("4")
        .arg("--settle-steps")
        .arg("32")
        .arg("--seed")
        .arg(seed.to_string())
        .arg("--output")
        .arg(&snapshot_path);

    let output = command
        .output()
        .with_context(|| format!("spawn equilibrium generator for {preset_id} galaxy {galaxy_index}"))?;
    if output.status.success() {
        return Ok(());
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    Err(anyhow!(
        "equilibrium generator failed with status {}:\nstdout:\n{}\nstderr:\n{}",
        output.status,
        stdout.trim(),
        stderr.trim()
    ))
}

fn resolve_equilibrate_command() -> Command {
    if let Ok(explicit) = std::env::var("GALAXY_EQUILIBRATE_BIN") {
        return Command::new(explicit);
    }

    if let Ok(current_exe) = std::env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            let sibling = parent.join("equilibrate");
            if sibling.exists() {
                return Command::new(sibling);
            }
        }
    }

    let mut command = Command::new("cargo");
    command
        .arg("run")
        .arg("--release")
        .arg("-p")
        .arg("sim-cuda")
        .arg("--bin")
        .arg("equilibrate")
        .arg("--");
    command.current_dir("/galaxy");
    command
}

fn relax_initial_conditions_if_needed(
    config: &SimulationConfig,
    initial_conditions: InitialConditions,
) -> anyhow::Result<InitialConditions> {
    if config.galaxies.is_empty() {
        return Ok(initial_conditions);
    }

    let relaxation_steps = relaxation_steps_for_config(config);
    if relaxation_steps == 0 {
        return Ok(initial_conditions);
    }

    let mut relaxed_particles = Vec::with_capacity(initial_conditions.particles.len());
    for (galaxy_index, galaxy) in config.galaxies.iter().enumerate() {
        let target_origin = Vec3::new(
            galaxy.position_kpc[0],
            galaxy.position_kpc[1],
            galaxy.position_kpc[2],
        );
        let target_velocity = Vec3::new(
            galaxy.velocity_kms[0],
            galaxy.velocity_kms[1],
            galaxy.velocity_kms[2],
        );
        let mut isolated_particles: Vec<Particle> = initial_conditions
            .particles
            .iter()
            .filter(|particle| particle.galaxy_index == galaxy_index as u32)
            .cloned()
            .map(|mut particle| {
                particle.position_kpc -= target_origin;
                particle.velocity_kms -= target_velocity;
                particle.galaxy_index = 0;
                particle
            })
            .collect();

        let isolated_mass = isolated_particles
            .iter()
            .map(|particle| particle.mass_msun)
            .sum::<f64>();
        if isolated_particles.is_empty() {
            continue;
        }

        let mut isolated_config = config.clone();
        isolated_config.name = format!("{}-relax-{galaxy_index}", config.name);
        isolated_config.galaxies = vec![{
            let mut galaxy = galaxy.clone();
            galaxy.position_kpc = [0.0, 0.0, 0.0];
            galaxy.velocity_kms = [0.0, 0.0, 0.0];
            galaxy
        }];
        isolated_config.preview.particle_budget = isolated_config.preview.particle_budget.min(64);

        let isolated_initial_conditions = InitialConditions {
            seed: initial_conditions.seed
                ^ (galaxy_index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            particles: std::mem::take(&mut isolated_particles),
            total_mass_msun: isolated_mass,
        };
        let mut backend = GpuBackend::new(&isolated_config, &isolated_initial_conditions)
            .with_context(|| format!("failed to initialize isolated relaxation backend for galaxy {galaxy_index}"))?;
        backend
            .advance(relaxation_steps)
            .with_context(|| format!("failed to relax galaxy {galaxy_index}"))?;
        let mut particles = backend
            .download_particles()
            .with_context(|| format!("failed to download relaxed particles for galaxy {galaxy_index}"))?;
        recenter_particles(&mut particles, galaxy_index as u32, target_origin, target_velocity);
        relaxed_particles.extend(particles.into_iter().map(|mut particle| {
            particle.galaxy_index = galaxy_index as u32;
            particle
        }));
    }

    Ok(InitialConditions {
        seed: initial_conditions.seed,
        particles: relaxed_particles,
        total_mass_msun: initial_conditions.total_mass_msun,
    })
}

fn relaxation_steps_for_config(config: &SimulationConfig) -> u32 {
    let particle_count = config
        .galaxies
        .iter()
        .map(|galaxy| {
            u64::from(galaxy.halo_particle_count)
                + u64::from(galaxy.disk_particle_count)
                + u64::from(galaxy.bulge_particle_count)
                + 1
        })
        .sum::<u64>()
        .max(1);
    let target_particle_updates = 320_000_000_u64;
    let raw_steps = (target_particle_updates / particle_count).clamp(16, 96);
    raw_steps as u32
}

fn should_relax_initial_conditions(particle_count: u64, config: &SimulationConfig) -> bool {
    let _ = particle_count;
    let _ = config;
    false
}

fn recenter_particles(
    particles: &mut [Particle],
    galaxy_index: u32,
    origin: Vec3,
    bulk_velocity: Vec3,
) {
    if particles.is_empty() {
        return;
    }

    let mut total_mass = 0.0;
    let mut center_of_mass = Vec3::ZERO;
    let mut center_velocity = Vec3::ZERO;
    for particle in particles.iter() {
        total_mass += particle.mass_msun;
        center_of_mass += particle.position_kpc * particle.mass_msun;
        center_velocity += particle.velocity_kms * particle.mass_msun;
    }
    if total_mass > 0.0 {
        center_of_mass = center_of_mass / total_mass;
        center_velocity = center_velocity / total_mass;
    }

    let position_shift = origin - center_of_mass;
    let velocity_shift = bulk_velocity - center_velocity;
    for particle in particles.iter_mut() {
        particle.position_kpc += position_shift;
        particle.velocity_kms += velocity_shift;
        particle.galaxy_index = galaxy_index;
    }
    if let Some(smbh) = particles
        .iter_mut()
        .find(|particle| matches!(particle.component, sim_core::ParticleComponent::Smbh))
    {
        smbh.position_kpc = origin;
        smbh.velocity_kms = bulk_velocity;
        smbh.galaxy_index = galaxy_index;
    }
}

async fn session_task(
    id: Uuid,
    config: SimulationConfig,
    preview_budget: u32,
    initial_conditions: InitialConditions,
    summary: Arc<RwLock<SessionSummary>>,
    latest_frame: Arc<RwLock<Option<Vec<u8>>>>,
    frame_tx: broadcast::Sender<Vec<u8>>,
    mut command_rx: mpsc::UnboundedReceiver<SessionRequest>,
) {
    let mut preview_budget = preview_budget;
    let mut running = false;
    let mut frame_dirty = false;
    let initial_conditions = initial_conditions;
    let particle_count = initial_conditions.particles.len() as u64;
    let mesh_cells = config
        .gravity
        .mesh_resolution
        .into_iter()
        .map(u64::from)
        .product::<u64>()
        .max(1);
    let target_particle_updates_per_tick = 512_000_000_u64;
    let target_mesh_cells_per_tick = 160_000_000_u64;
    let particle_limited_steps =
        (target_particle_updates_per_tick / particle_count.max(1)).max(1);
    let mesh_limited_steps = (target_mesh_cells_per_tick / mesh_cells).max(1);
    let max_steps_per_tick = particle_limited_steps
        .min(mesh_limited_steps)
        .clamp(1, 64) as u32;
    let target_tick_wall = Duration::from_millis(900);
    let mut steps_per_tick = max_steps_per_tick.min(4).max(1);
    let mut per_step_wall_ema_s: Option<f64> = None;
    let mut backend = match GpuBackend::new(&config, &initial_conditions) {
        Ok(backend) => backend,
        Err(error) => {
            error!("failed to initialize CUDA backend for session {id}: {error:#}");
            summary.write().state = SessionState::Failed;
            return;
        }
    };

    let frame_interval = Duration::from_secs_f64(1.0 / f64::from(config.preview.target_fps.max(1)));
    let mut frame_ticker = tokio::time::interval(frame_interval);
    frame_ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    if let Err(error) = publish_frame(
        &mut backend,
        preview_budget,
        &summary,
        &latest_frame,
        &frame_tx,
    ) {
        warn!("failed to publish initial frame for session {id}: {error:#}");
    }

    loop {
        if !running {
            let Some(request) = command_rx.recv().await else {
                return;
            };
            let SessionRequest { command, ack } = request;

            if matches!(command, SessionCommand::Stop) {
                summary.write().state = SessionState::Paused;
                if let Some(ack) = ack {
                    let _ = ack.send(Ok(summary.read().clone()));
                }
                return;
            }

            let result = match command {
                SessionCommand::Pause => {
                    running = false;
                    summary.write().state = SessionState::Paused;
                    Ok(())
                }
                SessionCommand::Resume => {
                    running = true;
                    summary.write().state = SessionState::Running;
                    Ok(())
                }
                SessionCommand::Step(substeps) => step_and_publish(
                    &mut backend,
                    substeps.max(1),
                    preview_budget,
                    &summary,
                    &latest_frame,
                    &frame_tx,
                ),
                SessionCommand::Snapshot => match run_backend_blocking(|| backend.download_particles()) {
                    Ok(particles) => {
                        let snapshot_summary = summary.read().clone();
                        match write_snapshot(id, &config, &snapshot_summary, &particles) {
                            Ok(path) => {
                                summary.write().latest_snapshot = Some(path);
                                Ok(())
                            }
                            Err(error) => Err(error),
                        }
                    }
                    Err(error) => {
                        Err(error.context("failed to download particles for snapshot"))
                    }
                },
                SessionCommand::Stop => unreachable!("stop is handled before command dispatch"),
                SessionCommand::Control(control) => match control {
                    ControlCommand::Pause => {
                        running = false;
                        summary.write().state = SessionState::Paused;
                        Ok(())
                    }
                    ControlCommand::Resume => {
                        running = true;
                        summary.write().state = SessionState::Running;
                        Ok(())
                    }
                    ControlCommand::Step { substeps } => step_and_publish(
                        &mut backend,
                        substeps.unwrap_or(1).max(1),
                        preview_budget,
                        &summary,
                        &latest_frame,
                        &frame_tx,
                    ),
                    ControlCommand::SetPreviewBudget { particle_budget } => {
                        preview_budget = particle_budget.max(1);
                        summary.write().preview_particle_budget = preview_budget;
                        let result = publish_frame(
                            &mut backend,
                            preview_budget,
                            &summary,
                            &latest_frame,
                            &frame_tx,
                        );
                        if result.is_ok() {
                            frame_dirty = false;
                        }
                        result
                    }
                },
            };

            match result {
                Ok(()) => {
                    if let Some(ack) = ack {
                        let _ = ack.send(Ok(summary.read().clone()));
                    }
                }
                Err(error) => {
                    error!("session {id} command failed: {error:#}");
                    summary.write().state = SessionState::Failed;
                    if let Some(ack) = ack {
                        let _ = ack.send(Err(error));
                    }
                    return;
                }
            }
            continue;
        }

        tokio::select! {
            biased;
            maybe_command = command_rx.recv() => {
                let Some(request) = maybe_command else {
                    return;
                };
                let SessionRequest { command, ack } = request;

                if matches!(command, SessionCommand::Stop) {
                    summary.write().state = SessionState::Paused;
                    if let Some(ack) = ack {
                        let _ = ack.send(Ok(summary.read().clone()));
                    }
                    return;
                }

                let result = match command {
                    SessionCommand::Pause => {
                        running = false;
                        summary.write().state = SessionState::Paused;
                        Ok(())
                    }
                    SessionCommand::Resume => {
                        running = true;
                        summary.write().state = SessionState::Running;
                        Ok(())
                    }
                    SessionCommand::Step(substeps) => step_and_publish(
                        &mut backend,
                        substeps.max(1),
                        preview_budget,
                        &summary,
                        &latest_frame,
                        &frame_tx,
                    ),
                    SessionCommand::Snapshot => match run_backend_blocking(|| backend.download_particles()) {
                        Ok(particles) => {
                            let snapshot_summary = summary.read().clone();
                            match write_snapshot(id, &config, &snapshot_summary, &particles) {
                                Ok(path) => {
                                    summary.write().latest_snapshot = Some(path);
                                    Ok(())
                                }
                                Err(error) => Err(error),
                            }
                        }
                        Err(error) => {
                            Err(error.context("failed to download particles for snapshot"))
                        }
                    },
                    SessionCommand::Stop => unreachable!("stop is handled before command dispatch"),
                    SessionCommand::Control(control) => match control {
                        ControlCommand::Pause => {
                            running = false;
                            summary.write().state = SessionState::Paused;
                            Ok(())
                        }
                        ControlCommand::Resume => {
                            running = true;
                            summary.write().state = SessionState::Running;
                            Ok(())
                        }
                        ControlCommand::Step { substeps } => step_and_publish(
                            &mut backend,
                            substeps.unwrap_or(1).max(1),
                            preview_budget,
                            &summary,
                            &latest_frame,
                            &frame_tx,
                        ),
                        ControlCommand::SetPreviewBudget { particle_budget } => {
                            preview_budget = particle_budget.max(1);
                            summary.write().preview_particle_budget = preview_budget;
                            let result = publish_frame(
                                &mut backend,
                                preview_budget,
                                &summary,
                                &latest_frame,
                                &frame_tx,
                            );
                            if result.is_ok() {
                                frame_dirty = false;
                            }
                            result
                        }
                    },
                };

                match result {
                    Ok(()) => {
                        if let Some(ack) = ack {
                            let _ = ack.send(Ok(summary.read().clone()));
                        }
                    }
                    Err(error) => {
                        error!("session {id} command failed: {error:#}");
                        summary.write().state = SessionState::Failed;
                        if let Some(ack) = ack {
                            let _ = ack.send(Err(error));
                        }
                        return;
                    }
                }
            }
            _ = frame_ticker.tick(), if frame_dirty => {
                if let Err(error) = publish_frame(
                    &mut backend,
                    preview_budget,
                    &summary,
                    &latest_frame,
                    &frame_tx,
                ) {
                    error!("session {id} failed while publishing frame: {error:#}");
                    summary.write().state = SessionState::Failed;
                    return;
                }
                frame_dirty = false;
            }
            _ = tokio::task::yield_now() => {
                let advance_start = Instant::now();
                if let Err(error) = advance_without_publish(
                    &mut backend,
                    steps_per_tick,
                    &summary,
                ) {
                    error!("session {id} failed during run loop: {error:#}");
                    summary.write().state = SessionState::Failed;
                    return;
                }
                let wall_seconds = advance_start.elapsed().as_secs_f64();
                let per_step_wall_s = wall_seconds / f64::from(steps_per_tick.max(1));
                per_step_wall_ema_s = Some(match per_step_wall_ema_s {
                    Some(previous) => previous * 0.7 + per_step_wall_s * 0.3,
                    None => per_step_wall_s,
                });
                if let Some(per_step_wall_ema_s) = per_step_wall_ema_s {
                    let target_steps = (target_tick_wall.as_secs_f64()
                        / per_step_wall_ema_s.max(1.0e-6))
                        .round()
                        .max(1.0) as u32;
                    steps_per_tick = target_steps.clamp(1, max_steps_per_tick.max(1));
                }
                frame_dirty = true;
            }
        }
    }
}

fn session_rank(state: SessionState) -> u8 {
    match state {
        SessionState::Running => 0,
        SessionState::Paused => 1,
        SessionState::Failed => 2,
    }
}

fn profile_session_loop() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        env::var("SIM_SERVER_PROFILE_SESSION_LOOP")
            .map(|value| value != "0")
            .unwrap_or(false)
    })
}

fn advance_without_publish(
    backend: &mut GpuBackend,
    steps: u32,
    summary: &Arc<RwLock<SessionSummary>>,
) -> anyhow::Result<()> {
    let start = Instant::now();
    let diagnostics = run_backend_blocking(|| backend.advance(steps.max(1)))?;
    if profile_session_loop() {
        info!(
            "session advance steps={} wall_ms={:.3} sim_time_myr={:.3}",
            steps.max(1),
            start.elapsed().as_secs_f64() * 1000.0,
            diagnostics.sim_time_myr
        );
    }
    let mut summary = summary.write();
    summary.sim_time_myr = diagnostics.sim_time_myr;
    summary.diagnostics = diagnostics;
    Ok(())
}

fn publish_frame(
    backend: &mut GpuBackend,
    preview_budget: u32,
    summary: &Arc<RwLock<SessionSummary>>,
    latest_frame: &Arc<RwLock<Option<Vec<u8>>>>,
    frame_tx: &broadcast::Sender<Vec<u8>>,
) -> anyhow::Result<()> {
    let start = Instant::now();
    let preview = run_backend_blocking(|| backend.preview_frame(preview_budget))?;
    let payload = bincode::serialize(&preview).context("failed to serialize preview frame")?;
    if profile_session_loop() {
        info!(
            "session publish preview_budget={} preview_count={} wall_ms={:.3} bytes={}",
            preview_budget,
            preview.diagnostics.preview_count,
            start.elapsed().as_secs_f64() * 1000.0,
            payload.len()
        );
    }
    *latest_frame.write() = Some(payload.clone());
    summary.write().diagnostics = preview.diagnostics.clone();
    let _ = frame_tx.send(payload);
    Ok(())
}

fn step_and_publish(
    backend: &mut GpuBackend,
    substeps: u32,
    preview_budget: u32,
    summary: &Arc<RwLock<SessionSummary>>,
    latest_frame: &Arc<RwLock<Option<Vec<u8>>>>,
    frame_tx: &broadcast::Sender<Vec<u8>>,
) -> anyhow::Result<()> {
    let diagnostics = run_backend_blocking(|| backend.step(substeps))?;
    {
        let mut summary = summary.write();
        summary.sim_time_myr = diagnostics.sim_time_myr;
        summary.diagnostics = diagnostics;
    }
    publish_frame(backend, preview_budget, summary, latest_frame, frame_tx)
}

fn run_backend_blocking<T>(work: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if matches!(handle.runtime_flavor(), tokio::runtime::RuntimeFlavor::MultiThread) => {
            tokio::task::block_in_place(work)
        }
        _ => work(),
    }
}

fn write_snapshot(
    id: Uuid,
    config: &SimulationConfig,
    summary: &SessionSummary,
    particles: &[Particle],
) -> anyhow::Result<String> {
    let directory = PathBuf::from(&config.snapshots.directory).join(id.to_string());
    fs::create_dir_all(&directory).context("failed to create snapshot directory")?;

    let manifest_path = directory.join("manifest.json");
    write_particle_snapshot(&manifest_path, &config.name, summary.sim_time_myr, particles)
        .with_context(|| format!("failed to write snapshot manifest {}", manifest_path.display()))?;

    info!(
        "wrote snapshot manifest for session {id} to {}",
        manifest_path.display()
    );
    Ok(manifest_path.display().to_string())
}
