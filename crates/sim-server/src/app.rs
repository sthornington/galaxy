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
use tower_http::{cors::CorsLayer, services::ServeDir, trace::TraceLayer};
use tracing::{error, warn};
use uuid::Uuid;

use crate::session::{
    ControlCommand, CreateSessionParams, SessionCommand, SessionRegistry, SessionSummary,
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
        .fallback_service(ServeDir::new(static_dir).append_index_html_on_directories(true))
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
    Ok(Json(
        state
            .sessions
            .command_wait(id, SessionCommand::Step(request.substeps.unwrap_or(1)))
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

async fn frame_socket(
    mut socket: WebSocket,
    session: crate::session::SessionHandle,
    as_json: bool,
) {
    if let Some(frame) = session.latest_frame() {
        let send_result = if as_json {
            match decode_preview_packet(&frame) {
                Ok(decoded) => {
                    socket
                        .send(Message::Text(
                            serde_json::to_string(&decoded)
                                .unwrap_or_else(|_| "{}".to_string())
                                .into(),
                        ))
                        .await
                }
                Err(error) => {
                    warn!("failed to decode cached frame for JSON websocket: {error}");
                    Ok(())
                }
            }
        } else {
            socket.send(Message::Binary(frame.into())).await
        };
        if let Err(error) = send_result {
            warn!("failed to send cached frame to websocket client: {error}");
            return;
        }
    }

    let mut receiver = session.subscribe_frames();
    loop {
        match receiver.recv().await {
            Ok(frame) => {
                let result = if as_json {
                    match decode_preview_packet(&frame) {
                        Ok(decoded) => {
                            socket
                                .send(Message::Text(
                                    serde_json::to_string(&decoded)
                                        .unwrap_or_else(|_| "{}".to_string())
                                        .into(),
                                ))
                                .await
                        }
                        Err(error) => {
                            warn!("failed to decode preview frame for JSON websocket: {error}");
                            continue;
                        }
                    }
                } else {
                    socket.send(Message::Binary(frame.into())).await
                };
                if let Err(error) = result {
                    warn!("frame websocket send failed: {error}");
                    return;
                }
            }
            Err(error) => {
                warn!("frame broadcast channel closed: {error}");
                return;
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
                    if let Err(error) = session.send_command(SessionCommand::Control(command)) {
                        error!("control websocket command failed: {error}");
                        let _ = socket
                            .send(Message::Text(format!("{{\"error\":\"{error}\"}}").into()))
                            .await;
                        return;
                    }
                }
                Err(error) => {
                    let _ = socket
                        .send(Message::Text(
                            format!("{{\"error\":\"invalid control command: {error}\"}}").into(),
                        ))
                        .await;
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
