use crate::config::{
    GalaxyConfig, GravityConfig, ObserverEffectsConfig, PreviewConfig, RelativityConfig,
    SimulationConfig, SmbhConfig, SnapshotConfig, TimeIntegrationConfig,
};

const GRAV_CONST_KPC_KMS2_PER_MSUN: f64 = 4.300_91e-6;

#[derive(Debug, Clone)]
pub struct MergerPreset {
    pub id: &'static str,
    pub title: &'static str,
    pub summary: &'static str,
    pub config: SimulationConfig,
}

pub fn built_in_presets() -> Vec<MergerPreset> {
    vec![major_merger(), polar_flyby(), minor_merger()]
}

fn gravity_defaults() -> GravityConfig {
    GravityConfig {
        grav_const_kpc_kms2_per_msun: 4.300_91e-6,
        halo_softening_kpc: 0.08,
        disk_softening_kpc: 0.03,
        bulge_softening_kpc: 0.02,
        opening_angle: 0.55,
        mesh_resolution: [256, 256, 128],
    }
}

fn relativity_defaults() -> RelativityConfig {
    RelativityConfig {
        enable_weak_field_mesh: true,
        enable_smbh_post_newtonian: true,
        observer_effects: ObserverEffectsConfig {
            doppler_boosting: true,
            gravitational_redshift: true,
            weak_lensing: true,
            time_delay: true,
        },
    }
}

fn preview_defaults() -> PreviewConfig {
    PreviewConfig {
        particle_budget: 500_000,
        density_grid: [960, 540],
        target_fps: 120,
    }
}

fn snapshot_defaults() -> SnapshotConfig {
    SnapshotConfig {
        directory: "output/snapshots".to_string(),
        cadence_steps: 120,
        compress: false,
    }
}

fn integration_defaults() -> TimeIntegrationConfig {
    TimeIntegrationConfig {
        base_timestep_myr: 0.05,
        max_substeps: 8,
        cfl_safety_factor: 0.35,
    }
}

fn total_galaxy_mass_msun(galaxy: &GalaxyConfig) -> f64 {
    galaxy.halo_mass_msun + galaxy.disk_mass_msun + galaxy.bulge_mass_msun + galaxy.smbh.mass_msun
}

fn set_bound_pair_orbit(
    primary: &mut GalaxyConfig,
    secondary: &mut GalaxyConfig,
    separation_kpc: f64,
    pericenter_kpc: f64,
) -> f64 {
    let primary_mass = total_galaxy_mass_msun(primary);
    let secondary_mass = total_galaxy_mass_msun(secondary);
    let total_mass = primary_mass + secondary_mass;
    let apocenter = separation_kpc.max(pericenter_kpc * 1.05);
    let semi_major_axis = 0.5 * (apocenter + pericenter_kpc);
    let relative_speed =
        (GRAV_CONST_KPC_KMS2_PER_MSUN * total_mass * (2.0 / apocenter - 1.0 / semi_major_axis))
            .max(0.0)
            .sqrt();

    let primary_offset = -apocenter * secondary_mass / total_mass;
    let secondary_offset = apocenter * primary_mass / total_mass;
    let primary_speed = -relative_speed * secondary_mass / total_mass;
    let secondary_speed = relative_speed * primary_mass / total_mass;

    primary.position_kpc = [primary_offset, 0.0, 0.0];
    secondary.position_kpc = [secondary_offset, 0.0, 0.0];
    primary.velocity_kms = [0.0, primary_speed, 0.0];
    secondary.velocity_kms = [0.0, secondary_speed, 0.0];
    relative_speed
}

fn major_merger() -> MergerPreset {
    let mut primary = GalaxyConfig {
        label: "Primary".to_string(),
        halo_mass_msun: 1.2e12,
        halo_scale_radius_kpc: 18.0,
        halo_particle_count: 2_500_000,
        disk_mass_msun: 6.0e10,
        disk_scale_radius_kpc: 3.8,
        disk_scale_height_kpc: 0.35,
        disk_particle_count: 1_250_000,
        bulge_mass_msun: 9.0e9,
        bulge_scale_radius_kpc: 0.8,
        bulge_particle_count: 200_000,
        smbh: SmbhConfig {
            mass_msun: 4.3e6,
            softening_kpc: 0.002,
            substeps: 16,
        },
        position_kpc: [0.0, 0.0, 0.0],
        velocity_kms: [0.0, 0.0, 0.0],
        disk_tilt_deg: [10.0, 25.0, 0.0],
        color_rgba: [0.96, 0.76, 0.42, 1.0],
    };
    let mut secondary = GalaxyConfig {
        label: "Secondary".to_string(),
        halo_mass_msun: 1.05e12,
        halo_scale_radius_kpc: 16.0,
        halo_particle_count: 2_000_000,
        disk_mass_msun: 5.5e10,
        disk_scale_radius_kpc: 3.2,
        disk_scale_height_kpc: 0.32,
        disk_particle_count: 1_150_000,
        bulge_mass_msun: 8.0e9,
        bulge_scale_radius_kpc: 0.7,
        bulge_particle_count: 180_000,
        smbh: SmbhConfig {
            mass_msun: 3.7e6,
            softening_kpc: 0.002,
            substeps: 16,
        },
        position_kpc: [0.0, 0.0, 0.0],
        velocity_kms: [0.0, 0.0, 0.0],
        disk_tilt_deg: [-35.0, 70.0, 18.0],
        color_rgba: [0.45, 0.76, 1.0, 1.0],
    };
    let relative_speed = set_bound_pair_orbit(&mut primary, &mut secondary, 100.0, 14.0);

    MergerPreset {
        id: "major-merger",
        title: "Equal-Mass Major Merger",
        summary: "Two Milky-Way-like galaxies on a bound orbit with tilted stellar disks.",
        config: SimulationConfig {
            name: "major-merger".to_string(),
            gravity: gravity_defaults(),
            relativity: relativity_defaults(),
            preview: preview_defaults(),
            snapshots: snapshot_defaults(),
            integration: integration_defaults(),
            initial_separation_kpc: 100.0,
            initial_relative_velocity_kms: relative_speed,
            output_directory: "output/major-merger".to_string(),
            galaxies: vec![primary, secondary],
        },
    }
}

fn polar_flyby() -> MergerPreset {
    let mut config = major_merger().config;
    config.name = "polar-flyby".to_string();
    config.output_directory = "output/polar-flyby".to_string();
    config.initial_separation_kpc = 140.0;
    let (primary, secondary) = config.galaxies.split_at_mut(1);
    config.initial_relative_velocity_kms =
        set_bound_pair_orbit(&mut primary[0], &mut secondary[0], 140.0, 28.0);
    config.galaxies[1].disk_tilt_deg = [88.0, 10.0, 12.0];
    config.galaxies[1].color_rgba = [0.78, 0.54, 1.0, 1.0];
    MergerPreset {
        id: "polar-flyby",
        title: "Polar Fly-By",
        summary: "A high-inclination encounter optimized for prominent tidal tails.",
        config,
    }
}

fn minor_merger() -> MergerPreset {
    let mut config = major_merger().config;
    config.name = "minor-merger".to_string();
    config.output_directory = "output/minor-merger".to_string();
    config.initial_separation_kpc = 140.0;
    config.galaxies[1].halo_mass_msun = 2.8e11;
    config.galaxies[1].halo_particle_count = 950_000;
    config.galaxies[1].disk_mass_msun = 1.4e10;
    config.galaxies[1].disk_particle_count = 360_000;
    config.galaxies[1].bulge_mass_msun = 1.2e9;
    config.galaxies[1].bulge_particle_count = 48_000;
    config.galaxies[1].smbh.mass_msun = 7.5e5;
    config.galaxies[1].color_rgba = [0.59, 1.0, 0.74, 1.0];
    let (primary, secondary) = config.galaxies.split_at_mut(1);
    config.initial_relative_velocity_kms =
        set_bound_pair_orbit(&mut primary[0], &mut secondary[0], 140.0, 18.0);
    MergerPreset {
        id: "minor-merger",
        title: "Minor Merger",
        summary: "A lower-mass companion spirals into a massive disk galaxy.",
        config,
    }
}
