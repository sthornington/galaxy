use std::{path::PathBuf, sync::Arc};

use axum::{
    Json, Router,
    extract::{
        Path, Query, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use futures_util::StreamExt;
use serde::{Deserialize, Serialize};
use sim_core::{MergerPreset, SimulationConfig, built_in_presets, decode_preview_packet};
use tokio::sync::broadcast::error::{RecvError, TryRecvError};
use tower_http::{cors::CorsLayer, services::ServeDir, trace::TraceLayer};
use tracing::{error, warn};
use uuid::Uuid;

use crate::session::{
    ControlCommand, CreateSessionParams, MAX_STEP_SUBSTEPS, SessionCommand, SessionError,
    SessionRegistry, SessionSummary,
};

#[derive(Clone)]
pub struct AppState {
    presets: Vec<MergerPreset>,
    sessions: SessionRegistry,
    static_dir: PathBuf,
}

impl AppState {
    pub fn new() -> Self {
        Self {
            presets: built_in_presets(),
            sessions: SessionRegistry::default(),
            static_dir: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("static"),
        }
    }
}

async fn client_log(body: String) -> axum::http::StatusCode {
    let mut trimmed = body;
    trimmed.truncate(2000);
    warn!(target: "client", "browser report: {trimmed}");
    axum::http::StatusCode::NO_CONTENT
}

pub fn router(state: Arc<AppState>) -> Router {
    let static_dir = state.static_dir.clone();
    Router::new()
        .route("/api/presets", get(list_presets))
        .route("/api/sessions", get(list_sessions))
        .route("/api/session", post(create_session))
        .route("/api/session/{id}", get(get_session))
        .route("/api/session/{id}/pause", post(pause_session))
        .route("/api/session/{id}/resume", post(resume_session))
        .route("/api/session/{id}/stop", post(stop_session))
        .route("/api/session/{id}/step", post(step_session))
        .route("/api/session/{id}/snapshot", post(snapshot_session))
        .route("/ws/frames/{id}", get(ws_frames))
        .route("/ws/control/{id}", get(ws_control))
        .route("/api/client-log", post(client_log))
        .fallback_service(ServeDir::new(static_dir).append_index_html_on_directories(true))
        .layer(tower_http::set_header::SetResponseHeaderLayer::overriding(
            axum::http::header::CACHE_CONTROL,
            axum::http::HeaderValue::from_static("no-cache"),
        ))
        .layer(CorsLayer::permissive())
        .layer(TraceLayer::new_for_http())
        .with_state(state)
}

#[derive(Debug, Serialize)]
struct PresetResponse {
    id: &'static str,
    title: &'static str,
    summary: &'static str,
    config: SimulationConfig,
}

async fn list_presets(State(state): State<Arc<AppState>>) -> Json<Vec<PresetResponse>> {
    Json(
        state
            .presets
            .iter()
            .map(|preset| PresetResponse {
                id: preset.id,
                title: preset.title,
                summary: preset.summary,
                config: preset.config.clone(),
            })
            .collect(),
    )
}

async fn list_sessions(State(state): State<Arc<AppState>>) -> Json<Vec<SessionSummary>> {
    Json(state.sessions.list())
}

#[derive(Debug, Deserialize)]
struct CreateSessionRequest {
    preset_id: Option<String>,
    seed: Option<u64>,
    config: Option<SimulationConfig>,
    preview_particle_budget: Option<u32>,
}

async fn create_session(
    State(state): State<Arc<AppState>>,
    Json(request): Json<CreateSessionRequest>,
) -> Result<Json<SessionSummary>, AppError> {
    let config = if let Some(config) = request.config {
        config
    } else if let Some(preset_id) = request.preset_id.as_deref() {
        state
            .presets
            .iter()
            .find(|preset| preset.id == preset_id)
            .map(|preset| preset.config.clone())
            .ok_or_else(|| AppError::NotFound(format!("unknown preset `{preset_id}`")))?
    } else {
        return Err(AppError::BadRequest(
            "either preset_id or config must be provided".to_string(),
        ));
    };

    let params = CreateSessionParams {
        config,
        preset_id: request.preset_id.unwrap_or_else(|| "custom".to_string()),
        seed: request.seed.unwrap_or(42),
        preview_particle_budget: request.preview_particle_budget,
    };
    let sessions = state.sessions.clone();
    let summary = tokio::task::spawn_blocking(move || sessions.create(params))
        .await
        .map_err(|error| AppError::Internal(anyhow::anyhow!("session creation task failed to join: {error}")))??;

    Ok(Json(summary))
}

async fn get_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<SessionSummary>, AppError> {
    state
        .sessions
        .get(id)
        .map(Json)
        .ok_or_else(|| AppError::NotFound(format!("unknown session `{id}`")))
}

async fn pause_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<SessionSummary>, AppError> {
    Ok(Json(
        state
            .sessions
            .command_wait(id, SessionCommand::Pause)
            .await?,
    ))
}

async fn resume_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<SessionSummary>, AppError> {
    Ok(Json(
        state
            .sessions
            .command_wait(id, SessionCommand::Resume)
            .await?,
    ))
}

async fn stop_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<SessionSummary>, AppError> {
    Ok(Json(state.sessions.stop(id).await?))
}

#[derive(Debug, Deserialize)]
struct StepRequest {
    substeps: Option<u32>,
}

async fn step_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Json(request): Json<StepRequest>,
) -> Result<Json<SessionSummary>, AppError> {
    let substeps = request.substeps.unwrap_or(1).clamp(1, MAX_STEP_SUBSTEPS);
    Ok(Json(
        state
            .sessions
            .command_wait(id, SessionCommand::Step(substeps))
            .await?,
    ))
}

async fn snapshot_session(
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<Json<SessionSummary>, AppError> {
    Ok(Json(
        state
            .sessions
            .command_wait(id, SessionCommand::Snapshot)
            .await?,
    ))
}

#[derive(Debug, Deserialize)]
struct FrameQuery {
    format: Option<String>,
}

async fn ws_frames(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
    Query(query): Query<FrameQuery>,
) -> Result<impl IntoResponse, AppError> {
    let session = state
        .sessions
        .handle(id)
        .ok_or_else(|| AppError::NotFound(format!("unknown session `{id}`")))?;
    let as_json = query.format.as_deref() == Some("json");
    Ok(ws.on_upgrade(move |socket| frame_socket(socket, session, as_json)))
}

async fn ws_control(
    ws: WebSocketUpgrade,
    State(state): State<Arc<AppState>>,
    Path(id): Path<Uuid>,
) -> Result<impl IntoResponse, AppError> {
    let session = state
        .sessions
        .handle(id)
        .ok_or_else(|| AppError::NotFound(format!("unknown session `{id}`")))?;
    Ok(ws.on_upgrade(move |socket| control_socket(socket, session)))
}

/// Sends one frame in the negotiated encoding. Returns Ok(false) when the
/// frame could not be decoded (skippable), Err when the socket is gone.
async fn send_frame(
    socket: &mut WebSocket,
    frame: &bytes::Bytes,
    as_json: bool,
) -> Result<bool, axum::Error> {
    if as_json {
        match decode_preview_packet(frame) {
            Ok(decoded) => {
                socket
                    .send(Message::Text(
                        serde_json::to_string(&decoded)
                            .unwrap_or_else(|_| "{}".to_string())
                            .into(),
                    ))
                    .await?;
                Ok(true)
            }
            Err(error) => {
                warn!("failed to decode preview frame for JSON websocket: {error}");
                Ok(false)
            }
        }
    } else {
        socket.send(Message::Binary(frame.clone())).await?;
        Ok(true)
    }
}

async fn frame_socket(
    mut socket: WebSocket,
    session: crate::session::SessionHandle,
    as_json: bool,
) {
    // Subscribe before sending the cached frame so a frame published in
    // between is not lost.
    let mut receiver = session.subscribe_frames();
    if let Some(frame) = session.latest_frame() {
        if send_frame(&mut socket, &frame, as_json).await.is_err() {
            return;
        }
    }

    loop {
        tokio::select! {
            // Drive the read side too: without it, client close frames are
            // never observed and disconnected viewers leak tasks while the
            // session is paused (no broadcasts to fail on).
            inbound = socket.recv() => {
                match inbound {
                    Some(Ok(Message::Close(_))) | None => return,
                    Some(Ok(_)) => {}
                    Some(Err(error)) => {
                        warn!("frame websocket receive failed: {error}");
                        return;
                    }
                }
            }
            frame = receiver.recv() => {
                match frame {
                    Ok(frame) => {
                        if send_frame(&mut socket, &frame, as_json).await.is_err() {
                            return;
                        }
                    }
                    Err(RecvError::Lagged(skipped)) => {
                        warn!("frame websocket lagged by {skipped}; resynchronizing to latest frame");
                        if let Some(frame) = session.latest_frame() {
                            if send_frame(&mut socket, &frame, as_json).await.is_err() {
                                return;
                            }
                        }
                        loop {
                            match receiver.try_recv() {
                                Ok(_) => continue,
                                Err(TryRecvError::Lagged(_)) => continue,
                                Err(TryRecvError::Empty) => break,
                                Err(TryRecvError::Closed) => return,
                            }
                        }
                    }
                    Err(RecvError::Closed) => return,
                }
            }
        }
    }
}

async fn control_socket(mut socket: WebSocket, session: crate::session::SessionHandle) {
    while let Some(message) = socket.next().await {
        let Ok(message) = message else {
            return;
        };

        match message {
            Message::Text(text) => match serde_json::from_str::<ControlCommand>(&text) {
                Ok(command) => {
                    if let Err(error) = session.send_command(SessionCommand::from(command)) {
                        error!("control websocket command failed: {error}");
                        let body =
                            serde_json::json!({ "error": error.to_string() }).to_string();
                        let _ = socket.send(Message::Text(body.into())).await;
                        return;
                    }
                }
                Err(error) => {
                    let body =
                        serde_json::json!({ "error": format!("invalid control command: {error}") })
                            .to_string();
                    let _ = socket.send(Message::Text(body.into())).await;
                }
            },
            Message::Close(_) => return,
            _ => {}
        }
    }
}

#[derive(Debug)]
enum AppError {
    BadRequest(String),
    NotFound(String),
    Internal(anyhow::Error),
}

impl From<anyhow::Error> for AppError {
    fn from(value: anyhow::Error) -> Self {
        Self::Internal(value)
    }
}

impl From<SessionError> for AppError {
    fn from(value: SessionError) -> Self {
        match value {
            SessionError::NotFound(id) => Self::NotFound(format!("unknown session `{id}`")),
            SessionError::Failed(error) => Self::Internal(error),
        }
    }
}

impl IntoResponse for AppError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            Self::BadRequest(message) => (StatusCode::BAD_REQUEST, message),
            Self::NotFound(message) => (StatusCode::NOT_FOUND, message),
            Self::Internal(error) => {
                error!("{error:#}");
                (StatusCode::INTERNAL_SERVER_ERROR, error.to_string())
            }
        };
        (status, Json(serde_json::json!({ "error": message }))).into_response()
    }
}
