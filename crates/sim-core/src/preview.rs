use serde::{Deserialize, Serialize};

use crate::Vec3;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewFrame {
    pub sim_time_myr: f64,
    pub diagnostics: Diagnostics,
    pub particles: Vec<PreviewParticle>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewParticle {
    pub position_kpc: [f32; 3],
    pub velocity_kms: [f32; 3],
    pub color_rgba: [f32; 4],
    pub intensity: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Diagnostics {
    pub particle_count: u64,
    pub preview_count: u32,
    pub sim_time_myr: f64,
    pub dt_myr: f64,
    pub kinetic_energy: f64,
    pub estimated_potential_energy: f64,
    pub total_momentum: Vec3,
}
