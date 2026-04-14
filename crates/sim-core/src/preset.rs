use crate::config::{
    GalaxyConfig, GalaxyInitialProfile, GravityConfig, ObserverEffectsConfig, PreviewConfig,
    RelativityConfig, SimulationConfig, SmbhConfig, SnapshotConfig, TimeIntegrationConfig,
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
    vec![
        uniform_sphere_collapse(),
        major_merger_debug(),
        major_merger(),
        polar_flyby(),
        minor_merger(),
    ]
}

fn generated_equilibrium_snapshot_path(preset_id: &str, galaxy_index: usize) -> String {
    format!("/galaxy/output/equilibrium/{preset_id}/galaxy-{galaxy_index}/manifest.json")
}

fn gravity_defaults() -> GravityConfig {
    GravityConfig {
        grav_const_kpc_kms2_per_msun: 4.300_91e-6,
        halo_softening_kpc: 0.08,
        disk_softening_kpc: 0.03,
        bulge_softening_kpc: 0.02,
        opening_angle: 0.55,
        mesh_resolution: [192, 192, 96],
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
        base_timestep_myr: 0.2,
        max_substeps: 8,
        cfl_safety_factor: 0.35,
    }
}

fn total_galaxy_mass_msun(galaxy: &GalaxyConfig) -> f64 {
    galaxy.halo_mass_msun + galaxy.disk_mass_msun + galaxy.bulge_mass_msun + galaxy.smbh.mass_msun
}

fn circular_speed_kms(total_mass_msun: f64, radius_kpc: f64) -> f64 {
    (GRAV_CONST_KPC_KMS2_PER_MSUN * total_mass_msun / radius_kpc.max(1.0e-6)).sqrt()
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
    let current_radius = separation_kpc.max(pericenter_kpc * 1.01);

    // Start on the inbound leg of a bound ellipse instead of at apocenter so the galaxies
    // are already moving toward each other when the viewer opens.
    let start_true_anomaly = (-130.0_f64).to_radians();
    let cos_f = start_true_anomaly.cos();
    let sin_f = start_true_anomaly.sin();
    let eccentricity =
        ((pericenter_kpc - current_radius) / (current_radius * cos_f - pericenter_kpc))
            .clamp(0.05, 0.92);
    let semi_latus_rectum = pericenter_kpc * (1.0 + eccentricity);
    let orbital_speed_scale =
        (GRAV_CONST_KPC_KMS2_PER_MSUN * total_mass / semi_latus_rectum.max(1.0e-6)).sqrt();
    let relative_radial_speed = orbital_speed_scale * eccentricity * sin_f;
    let relative_tangential_speed = orbital_speed_scale * (1.0 + eccentricity * cos_f);

    let primary_offset = -current_radius * secondary_mass / total_mass;
    let secondary_offset = current_radius * primary_mass / total_mass;
    let relative_velocity = [relative_radial_speed, relative_tangential_speed, 0.0];

    primary.position_kpc = [primary_offset, 0.0, 0.0];
    secondary.position_kpc = [secondary_offset, 0.0, 0.0];
    primary.velocity_kms = [
        -relative_velocity[0] * secondary_mass / total_mass,
        -relative_velocity[1] * secondary_mass / total_mass,
        0.0,
    ];
    secondary.velocity_kms = [
        relative_velocity[0] * primary_mass / total_mass,
        relative_velocity[1] * primary_mass / total_mass,
        0.0,
    ];
    (relative_radial_speed * relative_radial_speed
        + relative_tangential_speed * relative_tangential_speed)
        .sqrt()
}

fn set_direct_infall_pair_orbit(
    primary: &mut GalaxyConfig,
    secondary: &mut GalaxyConfig,
    separation_kpc: f64,
    speed_fraction_of_escape: f64,
    tangential_fraction_of_total: f64,
) -> f64 {
    let primary_mass = total_galaxy_mass_msun(primary);
    let secondary_mass = total_galaxy_mass_msun(secondary);
    let total_mass = primary_mass + secondary_mass;
    let separation = separation_kpc.max(1.0e-6);
    let escape_speed =
        (2.0 * GRAV_CONST_KPC_KMS2_PER_MSUN * total_mass / separation).sqrt();
    let relative_speed = escape_speed * speed_fraction_of_escape.clamp(0.0, 0.999);
    let tangential_speed = relative_speed * tangential_fraction_of_total.clamp(0.0, 0.95);
    let radial_speed = -(relative_speed * relative_speed - tangential_speed * tangential_speed)
        .max(0.0)
        .sqrt();

    let primary_offset = -separation * secondary_mass / total_mass;
    let secondary_offset = separation * primary_mass / total_mass;

    primary.position_kpc = [primary_offset, 0.0, 0.0];
    secondary.position_kpc = [secondary_offset, 0.0, 0.0];
    primary.velocity_kms = [
        -radial_speed * secondary_mass / total_mass,
        -tangential_speed * secondary_mass / total_mass,
        0.0,
    ];
    secondary.velocity_kms = [
        radial_speed * primary_mass / total_mass,
        tangential_speed * primary_mass / total_mass,
        0.0,
    ];
    relative_speed
}

fn major_merger() -> MergerPreset {
    let mut primary = GalaxyConfig {
        label: "Primary".to_string(),
        equilibrium_snapshot: Some(generated_equilibrium_snapshot_path("major-merger", 0)),
        initial_profile: GalaxyInitialProfile::AnalyticGalaxy,
        halo_mass_msun: 4.2e12,
        halo_scale_radius_kpc: 14.0,
        halo_particle_count: 800_000,
        disk_mass_msun: 1.35e11,
        disk_scale_radius_kpc: 2.8,
        disk_scale_height_kpc: 0.3,
        disk_particle_count: 320_000,
        bulge_mass_msun: 3.6e10,
        bulge_scale_radius_kpc: 0.55,
        bulge_particle_count: 64_000,
        smbh: SmbhConfig {
            mass_msun: 1.0e7,
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
        equilibrium_snapshot: Some(generated_equilibrium_snapshot_path("major-merger", 1)),
        initial_profile: GalaxyInitialProfile::AnalyticGalaxy,
        halo_mass_msun: 3.8e12,
        halo_scale_radius_kpc: 12.0,
        halo_particle_count: 720_000,
        disk_mass_msun: 1.2e11,
        disk_scale_radius_kpc: 2.4,
        disk_scale_height_kpc: 0.28,
        disk_particle_count: 280_000,
        bulge_mass_msun: 3.0e10,
        bulge_scale_radius_kpc: 0.5,
        bulge_particle_count: 56_000,
        smbh: SmbhConfig {
            mass_msun: 8.5e6,
            softening_kpc: 0.002,
            substeps: 16,
        },
        position_kpc: [0.0, 0.0, 0.0],
        velocity_kms: [0.0, 0.0, 0.0],
        disk_tilt_deg: [-35.0, 70.0, 18.0],
        color_rgba: [0.45, 0.76, 1.0, 1.0],
    };
    let relative_speed =
        set_direct_infall_pair_orbit(&mut primary, &mut secondary, 24.0, 0.28, 0.16);

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
            initial_separation_kpc: 24.0,
            initial_relative_velocity_kms: relative_speed,
            output_directory: "output/major-merger".to_string(),
            galaxies: vec![primary, secondary],
        },
    }
}

fn uniform_sphere_collapse() -> MergerPreset {
    let sphere_radius_kpc = 3.0;
    let sphere_total_mass_msun = 1.75e11 + 3.25e11;
    let sphere_edge_circular_speed_kms =
        circular_speed_kms(sphere_total_mass_msun, sphere_radius_kpc);
    MergerPreset {
        id: "uniform-sphere-collapse",
        title: "Rotating Uniform Sphere Collapse",
        summary: "A dense protogalactic halo+stellar sphere with differential rotation about +Z, for sanity-checking rotating collapse into a disk-like remnant.",
        config: SimulationConfig {
            name: "uniform-sphere-collapse".to_string(),
            gravity: GravityConfig {
                mesh_resolution: [128, 128, 128],
                ..gravity_defaults()
            },
            relativity: relativity_defaults(),
            preview: PreviewConfig {
                particle_budget: 131_072,
                density_grid: [960, 540],
                target_fps: 120,
            },
            snapshots: snapshot_defaults(),
            integration: TimeIntegrationConfig {
                base_timestep_myr: 0.05,
                max_substeps: 16,
                cfl_safety_factor: 0.2,
            },
            initial_separation_kpc: 0.0,
            initial_relative_velocity_kms: 0.0,
            output_directory: "output/uniform-sphere-collapse".to_string(),
            galaxies: vec![GalaxyConfig {
                label: "Cold Sphere".to_string(),
                equilibrium_snapshot: None,
                initial_profile: GalaxyInitialProfile::UniformSphere {
                    radius_kpc: sphere_radius_kpc,
                    velocity_dispersion_kms: sphere_edge_circular_speed_kms * 0.025,
                    edge_rotation_speed_kms: sphere_edge_circular_speed_kms * 0.9,
                },
                halo_mass_msun: 1.75e11,
                halo_scale_radius_kpc: 1.0,
                halo_particle_count: 80_000,
                disk_mass_msun: 0.0,
                disk_scale_radius_kpc: 1.0,
                disk_scale_height_kpc: 0.2,
                disk_particle_count: 0,
                bulge_mass_msun: 3.25e11,
                bulge_scale_radius_kpc: 1.0,
                bulge_particle_count: 120_000,
                smbh: SmbhConfig {
                    mass_msun: 0.0,
                    softening_kpc: 0.01,
                    substeps: 8,
                },
                position_kpc: [0.0, 0.0, 0.0],
                velocity_kms: [0.0, 0.0, 0.0],
                disk_tilt_deg: [0.0, 0.0, 0.0],
                color_rgba: [1.0, 0.94, 0.86, 1.0],
            }],
        },
    }
}

fn major_merger_debug() -> MergerPreset {
    let mut config = major_merger().config;
    config.name = "major-merger-debug".to_string();
    config.output_directory = "output/major-merger-debug".to_string();
    config.integration.base_timestep_myr = 0.2;
    config.preview.particle_budget = 16_384;
    config.galaxies[0].halo_particle_count = 80_000;
    config.galaxies[0].disk_particle_count = 32_000;
    config.galaxies[0].bulge_particle_count = 6_400;
    config.galaxies[1].halo_particle_count = 72_000;
    config.galaxies[1].disk_particle_count = 28_000;
    config.galaxies[1].bulge_particle_count = 5_600;
    config.galaxies[0].equilibrium_snapshot =
        Some(generated_equilibrium_snapshot_path("major-merger-debug", 0));
    config.galaxies[1].equilibrium_snapshot =
        Some(generated_equilibrium_snapshot_path("major-merger-debug", 1));

    MergerPreset {
        id: "major-merger-debug",
        title: "Major Merger Debug",
        summary: "Same equal-mass encounter at roughly 10x fewer particles for fast solver debugging.",
        config,
    }
}

fn polar_flyby() -> MergerPreset {
    let mut config = major_merger().config;
    config.name = "polar-flyby".to_string();
    config.output_directory = "output/polar-flyby".to_string();
    config.initial_separation_kpc = 30.0;
    let (primary, secondary) = config.galaxies.split_at_mut(1);
    config.initial_relative_velocity_kms =
        set_bound_pair_orbit(&mut primary[0], &mut secondary[0], 30.0, 14.0);
    config.galaxies[1].disk_tilt_deg = [88.0, 10.0, 12.0];
    config.galaxies[1].color_rgba = [0.78, 0.54, 1.0, 1.0];
    config.galaxies[1].equilibrium_snapshot =
        Some(generated_equilibrium_snapshot_path("polar-flyby", 1));
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
    config.initial_separation_kpc = 32.0;
    config.galaxies[1].halo_mass_msun = 1.0e12;
    config.galaxies[1].halo_particle_count = 950_000;
    config.galaxies[1].disk_mass_msun = 4.6e10;
    config.galaxies[1].disk_particle_count = 360_000;
    config.galaxies[1].bulge_mass_msun = 7.5e9;
    config.galaxies[1].bulge_particle_count = 48_000;
    config.galaxies[1].smbh.mass_msun = 3.0e6;
    config.galaxies[1].color_rgba = [0.59, 1.0, 0.74, 1.0];
    config.galaxies[1].equilibrium_snapshot =
        Some(generated_equilibrium_snapshot_path("minor-merger", 1));
    let (primary, secondary) = config.galaxies.split_at_mut(1);
    config.initial_relative_velocity_kms =
        set_direct_infall_pair_orbit(&mut primary[0], &mut secondary[0], 32.0, 0.30, 0.20);
    MergerPreset {
        id: "minor-merger",
        title: "Minor Merger",
        summary: "A lower-mass companion spirals into a massive disk galaxy.",
        config,
    }
}
