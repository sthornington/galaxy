use std::{net::SocketAddr, sync::Arc};

use anyhow::Context;
use sim_server::app::{self, AppState};
use tracing::info;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            std::env::var("RUST_LOG").unwrap_or_else(|_| "sim_server=info,tower_http=info".into()),
        )
        .with_target(false)
        .init();

    let state = Arc::new(AppState::new());
    let app = app::router(state);
    let addr: SocketAddr = std::env::var("SIM_SERVER_ADDR")
        .unwrap_or_else(|_| "0.0.0.0:8080".to_string())
        .parse()
        .context("invalid SIM_SERVER_ADDR")?;

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .context("failed to bind HTTP listener")?;
    info!("sim-server listening on http://{addr}");
    axum::serve(listener, app)
        .await
        .context("HTTP server failed")
}
