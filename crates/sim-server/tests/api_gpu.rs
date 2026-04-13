use std::{path::PathBuf, sync::Arc, time::Duration};

use axum::{
    Router,
    body::{Body, to_bytes},
    http::{Request, StatusCode, header},
};
use serde_json::{Value, json};
use sim_core::{SimulationConfig, built_in_presets};
use sim_server::{
    app::{AppState, router},
    session::{SessionState, SessionSummary},
};
use tower::util::ServiceExt;
use uuid::Uuid;

#[tokio::test]
#[ignore = "requires NVIDIA GPU"]
async fn gpu_http_session_lifecycle_smoke() {
    let snapshot_root = std::env::temp_dir().join(format!("galaxy-server-test-{}", Uuid::new_v4()));
    let app = router(Arc::new(AppState::new()));
    let config = small_test_config(&snapshot_root);

    eprintln!("create session");
    let create = request_json(
        app.clone(),
        Request::post("/api/session")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(
                json!({
                    "config": config,
                    "seed": 7_u64,
                    "preview_particle_budget": 96_u32
                })
                .to_string(),
            ))
            .unwrap(),
    )
    .await;
    assert_eq!(create.status, StatusCode::OK);
    let created: SessionSummary = serde_json::from_value(create.body).unwrap();
    assert_eq!(created.state, SessionState::Paused);
    assert!(created.particle_count > 0);
    assert_eq!(created.sim_time_myr, 0.0);

    eprintln!("step session");
    let step = request_json(
        app.clone(),
        Request::post(format!("/api/session/{}/step", created.id))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(json!({ "substeps": 2_u32 }).to_string()))
            .unwrap(),
    )
    .await;
    assert_eq!(step.status, StatusCode::OK);
    let stepped: SessionSummary = serde_json::from_value(step.body).unwrap();
    assert_eq!(stepped.state, SessionState::Paused);
    assert!(stepped.sim_time_myr > 0.0);
    assert!(stepped.diagnostics.preview_count > 0);
    assert!(stepped.diagnostics.preview_count <= stepped.preview_particle_budget);

    eprintln!("snapshot session");
    let snapshot = request_json(
        app.clone(),
        Request::post(format!("/api/session/{}/snapshot", created.id))
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from("{}"))
            .unwrap(),
    )
    .await;
    assert_eq!(snapshot.status, StatusCode::OK);
    let snapped: SessionSummary = serde_json::from_value(snapshot.body).unwrap();
    let manifest_path = PathBuf::from(snapped.latest_snapshot.as_ref().unwrap());
    assert!(manifest_path.exists());
    assert!(
        manifest_path
            .parent()
            .unwrap()
            .join("particles.bin")
            .exists()
    );

    eprintln!("fetch index");
    let index = request_text(app.clone(), Request::get("/").body(Body::empty()).unwrap()).await;
    assert_eq!(index.status, StatusCode::OK);
    assert!(index.body.contains("/fallback.mjs"));

    std::fs::remove_dir_all(snapshot_root).ok();
}

struct JsonResponse {
    status: StatusCode,
    body: Value,
}

struct TextResponse {
    status: StatusCode,
    body: String,
}

async fn request_json(app: Router, request: Request<Body>) -> JsonResponse {
    let response = tokio::time::timeout(Duration::from_secs(10), app.oneshot(request))
        .await
        .expect("request timed out")
        .unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    JsonResponse {
        status,
        body: serde_json::from_slice(&body).unwrap(),
    }
}

async fn request_text(app: Router, request: Request<Body>) -> TextResponse {
    let response = tokio::time::timeout(Duration::from_secs(10), app.oneshot(request))
        .await
        .expect("request timed out")
        .unwrap();
    let status = response.status();
    let body = to_bytes(response.into_body(), usize::MAX).await.unwrap();
    TextResponse {
        status,
        body: String::from_utf8(body.to_vec()).unwrap(),
    }
}

fn small_test_config(snapshot_root: &std::path::Path) -> SimulationConfig {
    let mut config = built_in_presets()
        .into_iter()
        .find(|preset| preset.id == "minor-merger")
        .unwrap()
        .config;
    config.name = "gpu-http-smoke".to_string();
    config.preview.particle_budget = 96;
    config.snapshots.directory = snapshot_root.display().to_string();
    for galaxy in &mut config.galaxies {
        galaxy.halo_particle_count = 128;
        galaxy.disk_particle_count = 64;
        galaxy.bulge_particle_count = 16;
        galaxy.equilibrium_snapshot = None;
    }
    config
}
