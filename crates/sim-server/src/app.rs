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

const PREVIEW_PACKET_PREFIX_BYTES: usize = 80 + 56;
const PREVIEW_PARTICLE_BYTES: usize = 16;
const PREVIEW_DELTA_HEADER_BYTES: usize = 32;
const PREVIEW_DELTA_VERSION: u32 = 2;
const PREVIEW_DELTA_KEYFRAME_INTERVAL: u32 = 64;

#[derive(Default)]
struct PreviewDeltaEncoder {
    previous: Option<bytes::Bytes>,
    frames_since_keyframe: u32,
}

impl PreviewDeltaEncoder {
    /// Drops the delta reference so the next frame goes out as a keyframe;
    /// the client requests this when it cannot apply a delta.
    fn reset(&mut self) {
        self.previous = None;
        self.frames_since_keyframe = 0;
    }

    fn encode(&mut self, frame: &bytes::Bytes) -> bytes::Bytes {
        let encoded = self
            .previous
            .as_ref()
            .filter(|_| self.frames_since_keyframe < PREVIEW_DELTA_KEYFRAME_INTERVAL)
            .and_then(|previous| encode_preview_delta(previous, frame));

        self.previous = Some(frame.clone());
        if let Some(encoded) = encoded.filter(|encoded| encoded.len() < frame.len()) {
            self.frames_since_keyframe += 1;
            bytes::Bytes::from(encoded)
        } else {
            self.frames_since_keyframe = 0;
            frame.clone()
        }
    }
}

fn preview_frame_layout(frame: &[u8]) -> Option<(usize, f64)> {
    if frame.len() < PREVIEW_PACKET_PREFIX_BYTES
        || &frame[..4] != b"GPKT"
        || u32::from_le_bytes(frame[4..8].try_into().ok()?) != 2
    {
        return None;
    }
    let count = u32::from_le_bytes(frame[16..20].try_into().ok()?) as usize;
    let expected =
        PREVIEW_PACKET_PREFIX_BYTES.checked_add(count.checked_mul(PREVIEW_PARTICLE_BYTES)?)?;
    if frame.len() != expected {
        return None;
    }
    let sim_time = f64::from_le_bytes(frame[24..32].try_into().ok()?);
    Some((count, sim_time))
}

fn signed_bits_required(max_magnitude: u32) -> u8 {
    if max_magnitude == 0 {
        1
    } else {
        (u32::BITS - max_magnitude.leading_zeros() + 1) as u8
    }
}

fn append_signed_bits(
    out: &mut Vec<u8>,
    bit_buffer: &mut u64,
    buffered_bits: &mut u8,
    value: i32,
    bits: u8,
) {
    let mask = (1_u64 << bits) - 1;
    *bit_buffer |= ((i64::from(value) as u64) & mask) << *buffered_bits;
    *buffered_bits += bits;
    while *buffered_bits >= 8 {
        out.push(*bit_buffer as u8);
        *bit_buffer >>= 8;
        *buffered_bits -= 8;
    }
}

fn encode_preview_delta(previous: &[u8], current: &[u8]) -> Option<Vec<u8>> {
    let (previous_count, previous_sim_time) = preview_frame_layout(previous)?;
    let (current_count, _) = preview_frame_layout(current)?;
    if previous_count != current_count {
        return None;
    }

    let mut max_position_delta = 0_u32;
    let mut max_velocity_delta = 0_u32;
    let mut max_mass_delta = 0_u32;
    for particle in 0..current_count {
        let offset = PREVIEW_PACKET_PREFIX_BYTES + particle * PREVIEW_PARTICLE_BYTES;
        // Component and reserved byte must match; the quantized mass word is
        // delta-coded like the kinematic fields because gas particles stream
        // their evolving SPH density through it.
        if previous[offset + 14..offset + 16] != current[offset + 14..offset + 16] {
            return None;
        }
        for field in 0..7 {
            let field_offset = offset + field * 2;
            let previous_value =
                u16::from_le_bytes(previous[field_offset..field_offset + 2].try_into().ok()?);
            let current_value =
                u16::from_le_bytes(current[field_offset..field_offset + 2].try_into().ok()?);
            let magnitude = (i32::from(current_value) - i32::from(previous_value)).unsigned_abs();
            if field < 3 {
                max_position_delta = max_position_delta.max(magnitude);
            } else if field < 6 {
                max_velocity_delta = max_velocity_delta.max(magnitude);
            } else {
                max_mass_delta = max_mass_delta.max(magnitude);
            }
        }
    }
    let position_bits = signed_bits_required(max_position_delta);
    let velocity_bits = signed_bits_required(max_velocity_delta);
    let mass_bits = signed_bits_required(max_mass_delta);
    if position_bits > 17 || velocity_bits > 17 || mass_bits > 17 {
        return None;
    }

    let payload_bits = current_count.checked_mul(
        3 * (usize::from(position_bits) + usize::from(velocity_bits)) + usize::from(mass_bits),
    )?;
    let payload_bytes = payload_bits.div_ceil(8);
    let mut out = Vec::with_capacity(
        PREVIEW_DELTA_HEADER_BYTES + PREVIEW_PACKET_PREFIX_BYTES + payload_bytes,
    );
    out.extend_from_slice(b"GPDL");
    out.extend_from_slice(&PREVIEW_DELTA_VERSION.to_le_bytes());
    out.extend_from_slice(&(current_count as u32).to_le_bytes());
    out.push(position_bits);
    out.push(velocity_bits);
    out.push(mass_bits);
    out.push(0_u8);
    out.extend_from_slice(&previous_sim_time.to_le_bytes());
    out.extend_from_slice(&(current.len() as u32).to_le_bytes());
    out.extend_from_slice(&(payload_bytes as u32).to_le_bytes());
    out.extend_from_slice(&current[..PREVIEW_PACKET_PREFIX_BYTES]);

    let mut bit_buffer = 0_u64;
    let mut buffered_bits = 0_u8;
    for particle in 0..current_count {
        let offset = PREVIEW_PACKET_PREFIX_BYTES + particle * PREVIEW_PARTICLE_BYTES;
        for field in 0..7 {
            let field_offset = offset + field * 2;
            let previous_value =
                u16::from_le_bytes(previous[field_offset..field_offset + 2].try_into().ok()?);
            let current_value =
                u16::from_le_bytes(current[field_offset..field_offset + 2].try_into().ok()?);
            append_signed_bits(
                &mut out,
                &mut bit_buffer,
                &mut buffered_bits,
                i32::from(current_value) - i32::from(previous_value),
                if field < 3 {
                    position_bits
                } else if field < 6 {
                    velocity_bits
                } else {
                    mass_bits
                },
            );
        }
    }
    if buffered_bits > 0 {
        out.push(bit_buffer as u8);
    }
    debug_assert_eq!(
        out.len(),
        PREVIEW_DELTA_HEADER_BYTES + PREVIEW_PACKET_PREFIX_BYTES + payload_bytes
    );
    Some(out)
}

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
    delta: Option<u8>,
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
    let use_delta = !as_json && query.delta == Some(1);
    Ok(ws.on_upgrade(move |socket| frame_socket(socket, session, as_json, use_delta)))
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
    use_delta: bool,
    delta_encoder: &mut PreviewDeltaEncoder,
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
        let wire_frame = if use_delta {
            delta_encoder.encode(frame)
        } else {
            frame.clone()
        };
        socket.send(Message::Binary(wire_frame)).await?;
        Ok(true)
    }
}

async fn frame_socket(
    mut socket: WebSocket,
    session: crate::session::SessionHandle,
    as_json: bool,
    use_delta: bool,
) {
    // Subscribe before sending the cached frame so a frame published in
    // between is not lost.
    let mut receiver = session.subscribe_frames();
    let mut delta_encoder = PreviewDeltaEncoder::default();
    let mut client_ready = true;
    if let Some(frame) = session.latest_frame() {
        if send_frame(&mut socket, &frame, as_json, use_delta, &mut delta_encoder)
            .await
            .is_err()
        {
            return;
        }
        client_ready = !use_delta;
    }

    loop {
        tokio::select! {
            // Drive the read side too: without it, client close frames are
            // never observed and disconnected viewers leak tasks while the
            // session is paused (no broadcasts to fail on).
            inbound = socket.recv() => {
                match inbound {
                    Some(Ok(Message::Close(_))) | None => return,
                    Some(Ok(Message::Text(message))) if use_delta => {
                        match message.as_str() {
                            "ready" => client_ready = true,
                            "resync" => {
                                delta_encoder.reset();
                                client_ready = true;
                            }
                            _ => {}
                        }
                    }
                    Some(Ok(_)) => {}
                    Some(Err(error)) => {
                        warn!("frame websocket receive failed: {error}");
                        return;
                    }
                }
            }
            frame = receiver.recv(), if !use_delta || client_ready => {
                match frame {
                    Ok(mut frame) => {
                        // A preview is a live view, not a recording. If socket
                        // backpressure accumulated frames while the previous
                        // send was in flight, discard the stale intermediates
                        // and transmit only the newest queued state.
                        loop {
                            match receiver.try_recv() {
                                Ok(newer) => frame = newer,
                                Err(TryRecvError::Lagged(_)) => continue,
                                Err(TryRecvError::Empty) => break,
                                Err(TryRecvError::Closed) => return,
                            }
                        }
                        if send_frame(
                            &mut socket,
                            &frame,
                            as_json,
                            use_delta,
                            &mut delta_encoder,
                        )
                        .await
                        .is_err()
                        {
                            return;
                        }
                        client_ready = !use_delta;
                    }
                    Err(RecvError::Lagged(skipped)) => {
                        warn!("frame websocket lagged by {skipped}; resynchronizing to latest frame");
                        let mut latest = session.latest_frame();
                        loop {
                            match receiver.try_recv() {
                                Ok(newer) => latest = Some(newer),
                                Err(TryRecvError::Lagged(_)) => continue,
                                Err(TryRecvError::Empty) => break,
                                Err(TryRecvError::Closed) => return,
                            }
                        }
                        if let Some(frame) = latest {
                            if send_frame(
                                &mut socket,
                                &frame,
                                as_json,
                                use_delta,
                                &mut delta_encoder,
                            )
                            .await
                            .is_err()
                            {
                                return;
                            }
                            client_ready = !use_delta;
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

#[cfg(test)]
mod preview_delta_tests {
    use super::*;

    fn preview_frame(sim_time: f64, count: usize, step: i32) -> bytes::Bytes {
        let mut frame = vec![0_u8; PREVIEW_PACKET_PREFIX_BYTES + count * PREVIEW_PARTICLE_BYTES];
        frame[0..4].copy_from_slice(b"GPKT");
        frame[4..8].copy_from_slice(&2_u32.to_le_bytes());
        frame[16..20].copy_from_slice(&(count as u32).to_le_bytes());
        frame[24..32].copy_from_slice(&sim_time.to_le_bytes());

        for particle in 0..count {
            let offset = PREVIEW_PACKET_PREFIX_BYTES + particle * PREVIEW_PARTICLE_BYTES;
            let base = 10_000 + particle as i32 * 8;
            let values = [
                base + step * 3,
                base + 1 - step * 2,
                base + 2 + step,
                base + 3 + step * 11,
                base + 4 - step * 7,
                base + 5 + step * 5,
            ];
            for (field, value) in values.into_iter().enumerate() {
                frame[offset + field * 2..offset + field * 2 + 2]
                    .copy_from_slice(&(value as u16).to_le_bytes());
            }
            frame[offset + 12..offset + 14]
                .copy_from_slice(&((particle as i32 + 100 + step * 2) as u16).to_le_bytes());
            frame[offset + 14] = (particle % 4) as u8;
        }
        bytes::Bytes::from(frame)
    }

    fn decode_delta(previous: &[u8], delta: &[u8]) -> Vec<u8> {
        assert_eq!(&delta[..4], b"GPDL");
        let count = u32::from_le_bytes(delta[8..12].try_into().unwrap()) as usize;
        let position_bits = delta[12];
        let velocity_bits = delta[13];
        let mass_bits = delta[14];
        let full_frame_bytes = u32::from_le_bytes(delta[24..28].try_into().unwrap()) as usize;
        let mut output = vec![0_u8; full_frame_bytes];
        output[..PREVIEW_PACKET_PREFIX_BYTES].copy_from_slice(
            &delta[PREVIEW_DELTA_HEADER_BYTES
                ..PREVIEW_DELTA_HEADER_BYTES + PREVIEW_PACKET_PREFIX_BYTES],
        );

        let payload = &delta[PREVIEW_DELTA_HEADER_BYTES + PREVIEW_PACKET_PREFIX_BYTES..];
        let mut bit_offset = 0_usize;
        for particle in 0..count {
            let record_offset = PREVIEW_PACKET_PREFIX_BYTES + particle * PREVIEW_PARTICLE_BYTES;
            for field in 0..7 {
                let bits = if field < 3 {
                    position_bits
                } else if field < 6 {
                    velocity_bits
                } else {
                    mass_bits
                };
                let mut encoded = 0_u32;
                for bit in 0..usize::from(bits) {
                    let source_bit = bit_offset + bit;
                    encoded |= u32::from((payload[source_bit / 8] >> (source_bit % 8)) & 1) << bit;
                }
                bit_offset += usize::from(bits);
                let sign = 1_u32 << (bits - 1);
                let value = if encoded & sign == 0 {
                    encoded as i32
                } else {
                    encoded as i32 - (1_i32 << bits)
                };
                let field_offset = record_offset + field * 2;
                let previous_value = u16::from_le_bytes(
                    previous[field_offset..field_offset + 2].try_into().unwrap(),
                );
                output[field_offset..field_offset + 2]
                    .copy_from_slice(&((i32::from(previous_value) + value) as u16).to_le_bytes());
            }
            output[record_offset + 14..record_offset + 16]
                .copy_from_slice(&previous[record_offset + 14..record_offset + 16]);
        }
        output
    }

    #[test]
    fn preview_delta_round_trips_exactly() {
        let first = preview_frame(1.0, 64, 0);
        let second = preview_frame(2.0, 64, 1);
        let mut encoder = PreviewDeltaEncoder::default();

        assert_eq!(encoder.encode(&first), first);
        let encoded = encoder.encode(&second);
        assert_eq!(&encoded[..4], b"GPDL");
        assert!(encoded.len() < second.len());
        assert_eq!(decode_delta(&first, &encoded), second.as_ref());
    }

    #[test]
    fn preview_delta_keyframes_on_record_layout_changes() {
        let first = preview_frame(1.0, 64, 0);
        let mut changed_component = preview_frame(2.0, 64, 1).to_vec();
        changed_component[PREVIEW_PACKET_PREFIX_BYTES + 14] ^= 1;
        let changed_component = bytes::Bytes::from(changed_component);
        let changed_count = preview_frame(3.0, 32, 2);
        let mut encoder = PreviewDeltaEncoder::default();

        assert_eq!(&encoder.encode(&first)[..4], b"GPKT");
        assert_eq!(&encoder.encode(&changed_component)[..4], b"GPKT");
        assert_eq!(&encoder.encode(&changed_count)[..4], b"GPKT");
    }

    #[test]
    fn preview_delta_carries_mass_changes() {
        let first = preview_frame(1.0, 64, 0);
        let mut changed_mass = preview_frame(2.0, 64, 1).to_vec();
        changed_mass[PREVIEW_PACKET_PREFIX_BYTES + 12] ^= 1;
        let changed_mass = bytes::Bytes::from(changed_mass);
        let mut encoder = PreviewDeltaEncoder::default();

        assert_eq!(&encoder.encode(&first)[..4], b"GPKT");
        let encoded = encoder.encode(&changed_mass);
        assert_eq!(&encoded[..4], b"GPDL");
        assert_eq!(decode_delta(&first, &encoded), changed_mass.as_ref());
    }

    #[test]
    fn preview_delta_emits_periodic_keyframes() {
        let mut encoder = PreviewDeltaEncoder::default();
        assert_eq!(&encoder.encode(&preview_frame(0.0, 64, 0))[..4], b"GPKT");
        for step in 1..=PREVIEW_DELTA_KEYFRAME_INTERVAL {
            assert_eq!(
                &encoder.encode(&preview_frame(f64::from(step), 64, step as i32))[..4],
                b"GPDL"
            );
        }
        assert_eq!(
            &encoder.encode(&preview_frame(
                f64::from(PREVIEW_DELTA_KEYFRAME_INTERVAL + 1),
                64,
                PREVIEW_DELTA_KEYFRAME_INTERVAL as i32 + 1,
            ))[..4],
            b"GPKT"
        );
    }
}
