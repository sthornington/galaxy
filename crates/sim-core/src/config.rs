use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub name: String,
    pub gravity: GravityConfig,
    pub relativity: RelativityConfig,
    pub preview: PreviewConfig,
    pub snapshots: SnapshotConfig,
    pub integration: TimeIntegrationConfig,
    pub galaxies: Vec<GalaxyConfig>,
    pub initial_separation_kpc: f64,
    pub initial_relative_velocity_kms: f64,
    pub output_directory: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GravityConfig {
    pub grav_const_kpc_kms2_per_msun: f64,
    pub halo_softening_kpc: f64,
    pub disk_softening_kpc: f64,
    pub bulge_softening_kpc: f64,
    pub opening_angle: f64,
    pub mesh_resolution: [u32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelativityConfig {
    pub enable_weak_field_mesh: bool,
    pub enable_smbh_post_newtonian: bool,
    pub observer_effects: ObserverEffectsConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObserverEffectsConfig {
    pub doppler_boosting: bool,
    pub gravitational_redshift: bool,
    pub weak_lensing: bool,
    pub time_delay: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreviewConfig {
    pub particle_budget: u32,
    pub density_grid: [u32; 2],
    pub target_fps: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotConfig {
    pub directory: String,
    pub cadence_steps: u32,
    pub compress: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeIntegrationConfig {
    pub base_timestep_myr: f64,
    pub max_substeps: u32,
    pub cfl_safety_factor: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GalaxyConfig {
    pub label: String,
    #[serde(default)]
    pub equilibrium_snapshot: Option<String>,
    #[serde(default)]
    pub initial_profile: GalaxyInitialProfile,
    pub halo_mass_msun: f64,
    pub halo_scale_radius_kpc: f64,
    pub halo_particle_count: u32,
    pub disk_mass_msun: f64,
    pub disk_scale_radius_kpc: f64,
    pub disk_scale_height_kpc: f64,
    pub disk_particle_count: u32,
    pub bulge_mass_msun: f64,
    pub bulge_scale_radius_kpc: f64,
    pub bulge_particle_count: u32,
    pub smbh: SmbhConfig,
    pub position_kpc: [f64; 3],
    pub velocity_kms: [f64; 3],
    pub disk_tilt_deg: [f64; 3],
    pub color_rgba: [f32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum GalaxyInitialProfile {
    #[default]
    AnalyticGalaxy,
    UniformSphere {
        radius_kpc: f64,
        #[serde(default)]
        velocity_dispersion_kms: f64,
        #[serde(default)]
        edge_rotation_speed_kms: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmbhConfig {
    pub mass_msun: f64,
    pub softening_kpc: f64,
    pub substeps: u32,
}
