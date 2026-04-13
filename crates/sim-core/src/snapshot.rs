use std::{fs, path::{Path, PathBuf}};

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

use crate::Particle;

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

pub fn write_particle_snapshot(
    manifest_path: impl AsRef<Path>,
    simulation_name: &str,
    sim_time_myr: f64,
    particles: &[Particle],
) -> anyhow::Result<SnapshotManifest> {
    let manifest_path = manifest_path.as_ref();
    let directory = manifest_path
        .parent()
        .context("snapshot manifest path must have a parent directory")?;
    fs::create_dir_all(directory)
        .with_context(|| format!("failed to create snapshot directory {}", directory.display()))?;

    let chunk_name = "particles.bin";
    let chunk_path = directory.join(chunk_name);
    fs::write(
        &chunk_path,
        bincode::serialize(particles).context("failed to encode particle snapshot")?,
    )
    .with_context(|| format!("failed to write particle snapshot {}", chunk_path.display()))?;

    let manifest = SnapshotManifest {
        schema_version: 1,
        simulation_name: simulation_name.to_string(),
        sim_time_myr,
        particle_count: particles.len() as u64,
        chunk_files: vec![SnapshotChunk {
            path: chunk_name.to_string(),
            particle_offset: 0,
            particle_count: particles.len() as u64,
        }],
    };

    fs::write(
        manifest_path,
        serde_json::to_vec_pretty(&manifest).context("failed to encode snapshot manifest")?,
    )
    .with_context(|| format!("failed to write manifest {}", manifest_path.display()))?;

    Ok(manifest)
}

pub fn load_particle_snapshot(
    manifest_path: impl AsRef<Path>,
) -> anyhow::Result<(SnapshotManifest, Vec<Particle>)> {
    let manifest_path = manifest_path.as_ref();
    let manifest: SnapshotManifest = serde_json::from_slice(
        &fs::read(manifest_path)
            .with_context(|| format!("failed to read manifest {}", manifest_path.display()))?,
    )
    .with_context(|| format!("failed to decode manifest {}", manifest_path.display()))?;

    let base_dir = manifest_path
        .parent()
        .context("snapshot manifest path must have a parent directory")?;
    let mut particles = Vec::with_capacity(manifest.particle_count as usize);
    let mut expected_offset = 0_u64;

    for chunk in &manifest.chunk_files {
        if chunk.particle_offset != expected_offset {
            bail!(
                "snapshot chunks are not contiguous: expected offset {}, got {}",
                expected_offset,
                chunk.particle_offset
            );
        }
        let chunk_path = resolve_chunk_path(base_dir, &chunk.path);
        let mut chunk_particles: Vec<Particle> = bincode::deserialize(
            &fs::read(&chunk_path)
                .with_context(|| format!("failed to read chunk {}", chunk_path.display()))?,
        )
        .with_context(|| format!("failed to decode particle chunk {}", chunk_path.display()))?;
        if chunk_particles.len() as u64 != chunk.particle_count {
            bail!(
                "particle chunk {} has {} particles, manifest expected {}",
                chunk_path.display(),
                chunk_particles.len(),
                chunk.particle_count
            );
        }
        expected_offset += chunk.particle_count;
        particles.append(&mut chunk_particles);
    }

    if particles.len() as u64 != manifest.particle_count {
        bail!(
            "snapshot manifest expected {} particles but loaded {}",
            manifest.particle_count,
            particles.len()
        );
    }

    Ok((manifest, particles))
}

fn resolve_chunk_path(base_dir: &Path, chunk_path: &str) -> PathBuf {
    let chunk_path = Path::new(chunk_path);
    if chunk_path.is_absolute() {
        chunk_path.to_path_buf()
    } else {
        base_dir.join(chunk_path)
    }
}
