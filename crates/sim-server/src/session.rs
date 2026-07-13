use std::{
    cmp::Ordering,
    collections::HashMap,
    env, fs,
    path::{Component, Path, PathBuf},
    sync::Arc,
    time::{Duration, Instant},
};

use anyhow::Context;
use bytes::Bytes;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sim_core::{
    Diagnostics, InitialConditions, Particle, SimulationConfig, validate_particle_count,
    write_particle_snapshot,
};
use sim_cuda::GpuBackend;
use thiserror::Error;
use tokio::sync::{broadcast, mpsc, oneshot};
use tracing::{error, info, warn};
use uuid::Uuid;

/// Upper bound on client-requested substeps for a single step command; the GPU
/// is uninterruptible while a step runs, so an unbounded value would let one
/// request pin the session indefinitely.
pub const MAX_STEP_SUBSTEPS: u32 = 64;

/// Command backlog per session; fire-and-forget senders drop on overflow so a
/// spamming websocket cannot grow memory while the GPU is busy.
const COMMAND_CHANNEL_CAPACITY: usize = 64;

#[derive(Debug, Error)]
pub enum SessionError {
    #[error("unknown session `{0}`")]
    NotFound(Uuid),
    #[error(transparent)]
    Failed(#[from] anyhow::Error),
}

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

    /// Creates a session and spawns its driver task. The returned summary is
    /// `Paused`; GPU initialization happens asynchronously in the task and
    /// flips the session to `Failed` if it cannot come up.
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
        let (command_tx, command_rx) = mpsc::channel(COMMAND_CHANNEL_CAPACITY);
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

    pub async fn command_wait(
        &self,
        id: Uuid,
        command: SessionCommand,
    ) -> Result<SessionSummary, SessionError> {
        let Some(handle) = self.handle(id) else {
            return Err(SessionError::NotFound(id));
        };
        handle.send_command_wait(command).await.map_err(Into::into)
    }

    pub async fn stop(&self, id: Uuid) -> Result<SessionSummary, SessionError> {
        let Some(handle) = self.handle(id) else {
            return Err(SessionError::NotFound(id));
        };
        // A session whose task already exited (failed backend init, prior
        // stop, command error) can no longer acknowledge anything; it must
        // still be removable, or it lingers in the registry forever.
        let summary = match handle.send_command_wait(SessionCommand::Stop).await {
            Ok(summary) => summary,
            Err(error) => {
                warn!("stopping unresponsive session {id}: {error:#}");
                handle.summary()
            }
        };
        self.sessions.write().remove(&id);
        Ok(summary)
    }
}

#[derive(Clone)]
pub struct SessionHandle {
    pub id: Uuid,
    pub summary: Arc<RwLock<SessionSummary>>,
    pub latest_frame: Arc<RwLock<Option<Bytes>>>,
    pub frame_tx: broadcast::Sender<Bytes>,
    pub command_tx: mpsc::Sender<SessionRequest>,
}

impl SessionHandle {
    pub fn summary(&self) -> SessionSummary {
        self.summary.read().clone()
    }

    pub fn latest_frame(&self) -> Option<Bytes> {
        self.latest_frame.read().clone()
    }

    pub fn subscribe_frames(&self) -> broadcast::Receiver<Bytes> {
        self.frame_tx.subscribe()
    }

    /// Fire-and-forget command dispatch; drops the command (with a warning)
    /// when the backlog is full rather than growing without bound.
    pub fn send_command(&self, command: SessionCommand) -> anyhow::Result<()> {
        match self.command_tx.try_send(SessionRequest { command, ack: None }) {
            Ok(()) => Ok(()),
            Err(mpsc::error::TrySendError::Full(request)) => {
                warn!(
                    "session {} command backlog full; dropping {:?}",
                    self.id, request.command
                );
                Ok(())
            }
            Err(mpsc::error::TrySendError::Closed(_)) => Err(anyhow::anyhow!(
                "session {} is no longer accepting commands",
                self.id
            )),
        }
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
            .await
            .map_err(|_| anyhow::anyhow!("session {} is no longer accepting commands", self.id))?;
        rx.await
            .map_err(|_| anyhow::anyhow!("session {} command acknowledgement dropped", self.id))?
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

/// Wire format for the control websocket; folded into [`SessionCommand`] at
/// the parse boundary.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ControlCommand {
    Pause,
    Resume,
    Step { substeps: Option<u32> },
    SetPreviewBudget { particle_budget: u32 },
}

impl From<ControlCommand> for SessionCommand {
    fn from(command: ControlCommand) -> Self {
        match command {
            ControlCommand::Pause => SessionCommand::Pause,
            ControlCommand::Resume => SessionCommand::Resume,
            ControlCommand::Step { substeps } => {
                SessionCommand::Step(substeps.unwrap_or(1).clamp(1, MAX_STEP_SUBSTEPS))
            }
            ControlCommand::SetPreviewBudget { particle_budget } => {
                SessionCommand::SetPreviewBudget(particle_budget)
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum SessionCommand {
    Pause,
    Resume,
    Step(u32),
    Snapshot,
    SetPreviewBudget(u32),
    Stop,
}

#[derive(Debug)]
pub struct SessionRequest {
    pub command: SessionCommand,
    pub ack: Option<oneshot::Sender<anyhow::Result<SessionSummary>>>,
}

struct SessionContext {
    id: Uuid,
    config: SimulationConfig,
    backend: GpuBackend,
    summary: Arc<RwLock<SessionSummary>>,
    latest_frame: Arc<RwLock<Option<Bytes>>>,
    frame_tx: broadcast::Sender<Bytes>,
    running: bool,
    preview_budget: u32,
    preview_pending: bool,
}

enum RequestOutcome {
    Continue,
    Exit,
}

#[allow(clippy::too_many_arguments)]
async fn session_task(
    id: Uuid,
    config: SimulationConfig,
    preview_budget: u32,
    initial_conditions: InitialConditions,
    summary: Arc<RwLock<SessionSummary>>,
    latest_frame: Arc<RwLock<Option<Bytes>>>,
    frame_tx: broadcast::Sender<Bytes>,
    mut command_rx: mpsc::Receiver<SessionRequest>,
) {
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
    let target_tick_wall = Duration::from_millis(125);
    let mut steps_per_tick = 1_u32;
    let mut per_step_wall_ema_s: Option<f64> = None;

    let backend = match GpuBackend::new(&config, &initial_conditions) {
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

    let mut ctx = SessionContext {
        id,
        config,
        backend,
        summary,
        latest_frame,
        frame_tx,
        running: false,
        preview_budget,
        preview_pending: false,
    };

    if let Err(error) = publish_frame(&mut ctx) {
        warn!("failed to publish initial frame for session {id}: {error:#}");
    }

    loop {
        if !ctx.running {
            let Some(request) = command_rx.recv().await else {
                return;
            };
            if matches!(process_request(&mut ctx, request), RequestOutcome::Exit) {
                return;
            }
            continue;
        }

        tokio::select! {
            biased;
            maybe_command = command_rx.recv() => {
                let Some(request) = maybe_command else {
                    return;
                };
                if matches!(process_request(&mut ctx, request), RequestOutcome::Exit) {
                    return;
                }
            }
            _ = frame_ticker.tick() => {
                // Preview capture runs on its own CUDA stream: collect the
                // previous frame if it is ready and queue the next one, all
                // without stalling the advance branch below.
                if ctx.preview_pending {
                    match try_publish_ready_frame(&mut ctx) {
                        Ok(published) => {
                            if published {
                                ctx.preview_pending = false;
                            }
                        }
                        Err(error) => {
                            error!("session {id} failed while publishing frame: {error:#}");
                            ctx.summary.write().state = SessionState::Failed;
                            return;
                        }
                    }
                }
                if !ctx.preview_pending {
                    if let Err(error) = request_preview_capture(&mut ctx) {
                        error!("session {id} failed while queueing preview frame: {error:#}");
                        ctx.summary.write().state = SessionState::Failed;
                        return;
                    }
                    ctx.preview_pending = true;
                }
            }
            _ = tokio::task::yield_now() => {
                let advance_start = Instant::now();
                if let Err(error) = advance_without_publish(&mut ctx, steps_per_tick) {
                    error!("session {id} failed during run loop: {error:#}");
                    ctx.summary.write().state = SessionState::Failed;
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
                    steps_per_tick = target_steps.clamp(1, max_steps_per_tick);
                }
            }
        }
    }
}

fn process_request(ctx: &mut SessionContext, request: SessionRequest) -> RequestOutcome {
    let SessionRequest { command, ack } = request;

    if matches!(command, SessionCommand::Stop) {
        ctx.summary.write().state = SessionState::Paused;
        if let Some(ack) = ack {
            let _ = ack.send(Ok(ctx.summary.read().clone()));
        }
        return RequestOutcome::Exit;
    }

    match apply_command(ctx, command) {
        Ok(()) => {
            if let Some(ack) = ack {
                let _ = ack.send(Ok(ctx.summary.read().clone()));
            }
            RequestOutcome::Continue
        }
        Err(error) => {
            error!("session {} command failed: {error:#}", ctx.id);
            ctx.summary.write().state = SessionState::Failed;
            if let Some(ack) = ack {
                let _ = ack.send(Err(error));
            }
            RequestOutcome::Exit
        }
    }
}

fn apply_command(ctx: &mut SessionContext, command: SessionCommand) -> anyhow::Result<()> {
    match command {
        SessionCommand::Pause => {
            ctx.running = false;
            ctx.summary.write().state = SessionState::Paused;
            let result = publish_frame(ctx);
            ctx.preview_pending = false;
            if let Err(error) = result {
                warn!("session {} failed to publish pause frame: {error:#}", ctx.id);
            }
            Ok(())
        }
        SessionCommand::Resume => {
            ctx.running = true;
            ctx.summary.write().state = SessionState::Running;
            Ok(())
        }
        SessionCommand::Step(substeps) => {
            ctx.preview_pending = false;
            let substeps = substeps.clamp(1, MAX_STEP_SUBSTEPS);
            let diagnostics = run_backend_blocking(|| ctx.backend.step(substeps))?;
            {
                let mut summary = ctx.summary.write();
                summary.sim_time_myr = diagnostics.sim_time_myr;
                summary.diagnostics = diagnostics;
            }
            publish_frame(ctx)
        }
        SessionCommand::Snapshot => {
            let id = ctx.id;
            let snapshot_summary = ctx.summary.read().clone();
            let backend = &mut ctx.backend;
            let config = &ctx.config;
            // Both the particle download and the multi-hundred-MB file write
            // belong off the async runtime.
            let path = run_backend_blocking(|| {
                let particles = backend
                    .download_particles()
                    .context("failed to download particles for snapshot")?;
                write_snapshot(id, config, &snapshot_summary, &particles)
            })?;
            ctx.summary.write().latest_snapshot = Some(path);
            Ok(())
        }
        SessionCommand::SetPreviewBudget(particle_budget) => {
            ctx.preview_budget = particle_budget.max(1);
            ctx.summary.write().preview_particle_budget = ctx.preview_budget;
            let result = publish_frame(ctx);
            ctx.preview_pending = false;
            result
        }
        SessionCommand::Stop => unreachable!("stop is handled before command dispatch"),
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

fn advance_without_publish(ctx: &mut SessionContext, steps: u32) -> anyhow::Result<()> {
    let start = Instant::now();
    let backend = &mut ctx.backend;
    let diagnostics = run_backend_blocking(|| backend.advance(steps.max(1)))?;
    if profile_session_loop() {
        info!(
            "session advance steps={} wall_ms={:.3} sim_time_myr={:.3}",
            steps.max(1),
            start.elapsed().as_secs_f64() * 1000.0,
            diagnostics.sim_time_myr
        );
    }
    let mut summary = ctx.summary.write();
    summary.sim_time_myr = diagnostics.sim_time_myr;
    summary.diagnostics = diagnostics;
    Ok(())
}

fn publish_frame(ctx: &mut SessionContext) -> anyhow::Result<()> {
    let start = Instant::now();
    let preview_budget = ctx.preview_budget;
    let backend = &mut ctx.backend;
    let (payload, diagnostics) =
        run_backend_blocking(|| backend.preview_packet_bytes(preview_budget))?;
    if profile_session_loop() {
        info!(
            "session publish preview_budget={} preview_count={} wall_ms={:.3} bytes={}",
            preview_budget,
            diagnostics.preview_count,
            start.elapsed().as_secs_f64() * 1000.0,
            payload.len()
        );
    }
    let payload = Bytes::from(payload);
    *ctx.latest_frame.write() = Some(payload.clone());
    ctx.summary.write().diagnostics = diagnostics;
    let _ = ctx.frame_tx.send(payload);
    Ok(())
}

fn request_preview_capture(ctx: &mut SessionContext) -> anyhow::Result<()> {
    let preview_budget = ctx.preview_budget;
    let backend = &mut ctx.backend;
    let diagnostics = run_backend_blocking(|| backend.request_preview(preview_budget))?;
    ctx.summary.write().diagnostics = diagnostics;
    Ok(())
}

fn try_publish_ready_frame(ctx: &mut SessionContext) -> anyhow::Result<bool> {
    let start = Instant::now();
    let backend = &mut ctx.backend;
    let Some((payload, diagnostics)) =
        run_backend_blocking(|| backend.try_collect_preview_packet_bytes())?
    else {
        return Ok(false);
    };
    if profile_session_loop() {
        info!(
            "session publish_ready preview_count={} wall_ms={:.3} bytes={}",
            diagnostics.preview_count,
            start.elapsed().as_secs_f64() * 1000.0,
            payload.len()
        );
    }
    let payload = Bytes::from(payload);
    *ctx.latest_frame.write() = Some(payload.clone());
    ctx.summary.write().diagnostics = diagnostics;
    let _ = ctx.frame_tx.send(payload);
    Ok(true)
}

fn run_backend_blocking<T>(work: impl FnOnce() -> anyhow::Result<T>) -> anyhow::Result<T> {
    match tokio::runtime::Handle::try_current() {
        Ok(handle) if matches!(handle.runtime_flavor(), tokio::runtime::RuntimeFlavor::MultiThread) => {
            tokio::task::block_in_place(work)
        }
        _ => work(),
    }
}

/// Resolves the snapshot root, refusing client-supplied absolute paths or
/// parent-directory escapes.
fn snapshot_root(config: &SimulationConfig) -> PathBuf {
    let configured = Path::new(&config.snapshots.directory);
    let is_safe = configured.is_relative()
        && configured
            .components()
            .all(|component| matches!(component, Component::Normal(_) | Component::CurDir));
    if is_safe && !config.snapshots.directory.is_empty() {
        configured.to_path_buf()
    } else {
        warn!(
            "ignoring unsafe snapshot directory {:?}; using output/snapshots",
            config.snapshots.directory
        );
        PathBuf::from("output/snapshots")
    }
}

fn write_snapshot(
    id: Uuid,
    config: &SimulationConfig,
    summary: &SessionSummary,
    particles: &[Particle],
) -> anyhow::Result<String> {
    let directory = snapshot_root(config).join(id.to_string());
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
