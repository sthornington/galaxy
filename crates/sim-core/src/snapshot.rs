use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotManifest {
    pub schema_version: u32,
    pub simulation_name: String,
    pub sim_time_myr: f64,
    pub particle_count: u64,
    pub chunk_files: Vec<SnapshotChunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotChunk {
    pub path: String,
    pub particle_offset: u64,
    pub particle_count: u64,
}
