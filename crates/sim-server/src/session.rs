use std::{cmp::Ordering, collections::HashMap, fs, path::PathBuf, sync::Arc, time::Duration};

use anyhow::{Context, anyhow};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sim_core::{
    Diagnostics, InitialConditions, Particle, SimulationConfig, SnapshotChunk, SnapshotManifest,
    validate_particle_count,
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
        let initial_conditions = InitialConditions::generate(&params.config, params.seed)
            .context("failed to generate initial conditions")?;

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
    let particle_count = initial_conditions.particles.len() as u64;
    // Keep the browser fed with frames even for multi-million-particle presets
    // by scaling the simulation work budget to the particle count.
    let target_particle_updates_per_tick = 96_000_000_u64;
    let steps_per_tick = ((target_particle_updates_per_tick / particle_count.max(1)) as u32)
        .clamp(4, config.integration.max_substeps.saturating_mul(4).max(1));
    let mut backend = match GpuBackend::new(&config, &initial_conditions) {
        Ok(backend) => backend,
        Err(error) => {
            error!("failed to initialize CUDA backend for session {id}: {error:#}");
            summary.write().state = SessionState::Failed;
            return;
        }
    };

    let frame_interval = Duration::from_secs_f64(1.0 / f64::from(config.preview.target_fps.max(1)));
    let mut ticker = tokio::time::interval(frame_interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

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
        tokio::select! {
            _ = ticker.tick() => {
                if !running {
                    continue;
                }
                if let Err(error) = advance_and_publish(
                    &mut backend,
                    steps_per_tick,
                    preview_budget,
                    &summary,
                    &latest_frame,
                    &frame_tx,
                ) {
                    error!("session {id} failed during run loop: {error:#}");
                    summary.write().state = SessionState::Failed;
                    return;
                }
            }
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
                    SessionCommand::Snapshot => match backend.download_particles() {
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
                            publish_frame(
                                &mut backend,
                                preview_budget,
                                &summary,
                                &latest_frame,
                                &frame_tx,
                            )
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

fn advance_and_publish(
    backend: &mut GpuBackend,
    steps: u32,
    preview_budget: u32,
    summary: &Arc<RwLock<SessionSummary>>,
    latest_frame: &Arc<RwLock<Option<Vec<u8>>>>,
    frame_tx: &broadcast::Sender<Vec<u8>>,
) -> anyhow::Result<()> {
    let diagnostics = backend.advance(steps.max(1))?;
    {
        let mut summary = summary.write();
        summary.sim_time_myr = diagnostics.sim_time_myr;
        summary.diagnostics = diagnostics;
    }
    publish_frame(backend, preview_budget, summary, latest_frame, frame_tx)
}

fn publish_frame(
    backend: &mut GpuBackend,
    preview_budget: u32,
    summary: &Arc<RwLock<SessionSummary>>,
    latest_frame: &Arc<RwLock<Option<Vec<u8>>>>,
    frame_tx: &broadcast::Sender<Vec<u8>>,
) -> anyhow::Result<()> {
    let preview = backend.preview_frame(preview_budget)?;
    let payload = bincode::serialize(&preview).context("failed to serialize preview frame")?;
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
    let diagnostics = backend.step(substeps)?;
    {
        let mut summary = summary.write();
        summary.sim_time_myr = diagnostics.sim_time_myr;
        summary.diagnostics = diagnostics;
    }
    publish_frame(backend, preview_budget, summary, latest_frame, frame_tx)
}

fn write_snapshot(
    id: Uuid,
    config: &SimulationConfig,
    summary: &SessionSummary,
    particles: &[Particle],
) -> anyhow::Result<String> {
    let directory = PathBuf::from(&config.snapshots.directory).join(id.to_string());
    fs::create_dir_all(&directory).context("failed to create snapshot directory")?;

    let manifest = SnapshotManifest {
        schema_version: 1,
        simulation_name: config.name.clone(),
        sim_time_myr: summary.sim_time_myr,
        particle_count: summary.particle_count,
        chunk_files: vec![SnapshotChunk {
            path: "particles.bin".to_string(),
            particle_offset: 0,
            particle_count: summary.particle_count,
        }],
    };

    let manifest_path = directory.join("manifest.json");
    fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).context("failed to encode snapshot manifest")?,
    )
    .with_context(|| {
        format!(
            "failed to write snapshot manifest to {}",
            manifest_path.display()
        )
    })?;

    let chunk_path = directory.join("particles.bin");
    fs::write(
        &chunk_path,
        bincode::serialize(particles).context("failed to encode particle chunk")?,
    )
    .with_context(|| format!("failed to write particle chunk {}", chunk_path.display()))?;

    info!(
        "wrote snapshot manifest for session {id} to {}",
        manifest_path.display()
    );
    Ok(manifest_path.display().to_string())
}
