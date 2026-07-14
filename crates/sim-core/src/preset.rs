use crate::config::{
    GalaxyConfig, GalaxyInitialProfile, GravityConfig, ObserverEffectsConfig, PreviewConfig,
    RelativityConfig, SimulationConfig, SmbhConfig, SnapshotConfig, TimeIntegrationConfig,
};
use crate::math::GRAV_CONST_KPC_KMS2_PER_MSUN;

#[derive(Debug, Clone)]
pub struct MergerPreset {
    pub id: &'static str,
    pub title: &'static str,
    pub summary: &'static str,
    pub config: SimulationConfig,
}

pub fn built_in_presets() -> Vec<MergerPreset> {
    vec![
        isolated_spiral(),
        uniform_sphere_collapse(),
        triple_sphere_orbit_debug(),
        major_merger_debug(),
        major_merger(),
        major_merger_distant(),
        grand_merger(),
        smbh_playground(),
        polar_flyby(),
        minor_merger(),
    ]
}

fn gravity_defaults() -> GravityConfig {
    GravityConfig {
        grav_const_kpc_kms2_per_msun: GRAV_CONST_KPC_KMS2_PER_MSUN,
        halo_softening_kpc: 0.08,
        disk_softening_kpc: 0.03,
        bulge_softening_kpc: 0.02,
        opening_angle: 0.55,
        // Merger halos are close to spherical, so keep the mesh cubic; a
        // coarser z-axis would inflate the TreePM split radius (and with it
        // the short-range cutoff) for every axis.
        mesh_resolution: [192, 192, 192],
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
        target_fps: 120,
    }
}

fn snapshot_defaults() -> SnapshotConfig {
    SnapshotConfig {
        directory: "output/snapshots".to_string(),
        cadence_steps: 120,
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
    // If the eccentricity clamp engaged, the requested radius no longer lies on the
    // conic; recompute the starting separation from the clamped orbit so positions
    // and velocities describe the same ellipse.
    let current_radius = semi_latus_rectum / (1.0 + eccentricity * cos_f);
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

fn set_equal_triple_ring_orbit(galaxies: &mut [GalaxyConfig], orbital_radius_kpc: f64) -> f64 {
    debug_assert!(galaxies.len() == 3);
    let per_galaxy_mass = total_galaxy_mass_msun(&galaxies[0]);
    let orbital_speed =
        (GRAV_CONST_KPC_KMS2_PER_MSUN * per_galaxy_mass * (3.0_f64).sqrt() / (3.0 * orbital_radius_kpc.max(1.0e-6)))
            .sqrt();

    for (index, galaxy) in galaxies.iter_mut().enumerate() {
        let angle = index as f64 * (2.0 * std::f64::consts::PI / 3.0);
        let cos_angle = angle.cos();
        let sin_angle = angle.sin();
        galaxy.position_kpc = [orbital_radius_kpc * cos_angle, orbital_radius_kpc * sin_angle, 0.0];
        galaxy.velocity_kms = [-orbital_speed * sin_angle, orbital_speed * cos_angle, 0.0];
    }

    orbital_speed
}

fn major_merger() -> MergerPreset {
    let mut primary = GalaxyConfig {
        label: "Primary".to_string(),
        equilibrium_snapshot: None,
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
        equilibrium_snapshot: None,
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

/// The major merger's primary galaxy alone: a Milky-Way-like disk left to
/// evolve in isolation so spiral structure can develop undisturbed.
fn isolated_spiral() -> MergerPreset {
    let mut config = major_merger().config;
    config.name = "isolated-spiral".to_string();
    config.output_directory = "output/isolated-spiral".to_string();
    config.galaxies.truncate(1);
    config.galaxies[0].position_kpc = [0.0, 0.0, 0.0];
    config.galaxies[0].velocity_kms = [0.0, 0.0, 0.0];
    config.initial_separation_kpc = 0.0;
    config.initial_relative_velocity_kms = 0.0;

    MergerPreset {
        id: "isolated-spiral",
        title: "Isolated Spiral Galaxy",
        summary: "A single Milky-Way-like disk galaxy in equilibrium — watch swing-amplified spiral arms wind up undisturbed.",
        config,
    }
}

/// The same equal-mass pair as `major-merger`, but starting five times
/// further out on a slow infall: both disks get a few hundred Myr to develop
/// spiral structure before the first passage disturbs them.
fn major_merger_distant() -> MergerPreset {
    let mut config = major_merger().config;
    config.name = "major-merger-distant".to_string();
    config.output_directory = "output/major-merger-distant".to_string();
    config.initial_separation_kpc = 120.0;
    let (primary, secondary) = config.galaxies.split_at_mut(1);
    config.initial_relative_velocity_kms =
        set_direct_infall_pair_orbit(&mut primary[0], &mut secondary[0], 120.0, 0.35, 0.35);

    MergerPreset {
        id: "major-merger-distant",
        title: "Distant Major Merger",
        summary: "The equal-mass encounter started 120 kpc apart: spiral arms wind up in both disks long before the collision.",
        config,
    }
}

/// The showcase run: the distant equal-mass encounter rebuilt at 5.5M
/// particles with the luminous fraction cranked (2.46M star particles).
/// Halo-to-disk particle mass ratio stays ~20x, low enough that collisional
/// disk heating is negligible over the merger timescale.
fn grand_merger() -> MergerPreset {
    let mut config = major_merger().config;
    config.name = "grand-merger".to_string();
    config.output_directory = "output/grand-merger".to_string();
    // A finer PM mesh shrinks the force-split radius, moving work from the
    // (per-particle, superlinear) short-range tree onto the FFT, which is
    // nearly free at this size. Tuned empirically: 986 ms/step at 192^3.
    config.gravity.mesh_resolution = [256, 256, 256];
    // Star particles dominate both the picture and the short-range cost, so
    // spend the budget there; the halo is sampled coarser (heavier particles,
    // extra softening to keep disk heating negligible) since it only needs to
    // supply a smooth potential.
    config.gravity.halo_softening_kpc = 0.12;
    {
        let primary = &mut config.galaxies[0];
        primary.halo_particle_count = 850_000;
        primary.disk_particle_count = 1_100_000;
        primary.bulge_particle_count = 220_000;
    }
    {
        let secondary = &mut config.galaxies[1];
        secondary.halo_particle_count = 750_000;
        secondary.disk_particle_count = 950_000;
        secondary.bulge_particle_count = 190_000;
    }
    config.initial_separation_kpc = 60.0;
    let (primary, secondary) = config.galaxies.split_at_mut(1);
    config.initial_relative_velocity_kms =
        set_direct_infall_pair_orbit(&mut primary[0], &mut secondary[0], 60.0, 0.35, 0.35);

    MergerPreset {
        id: "grand-merger",
        title: "Grand Merger (5.5M particles)",
        summary: "The distant equal-mass collision at full resolution: 2.5 million star particles develop spiral arms, collide, and relax into a remnant. IC generation takes a few minutes.",
        config,
    }
}

/// A dense spiral disk with three massive "marauder" SMBHs on inclined
/// crossing orbits. Each marauder is a bare-SMBH galaxy (all component
/// counts zero); at 2.5-5e8 Msun their dynamical influence radii are
/// 50-120 pc — well above the force softening — so they visibly carve
/// dynamical-friction wakes and gaps through the stellar disk before
/// sinking toward the center and interacting with each other.
fn smbh_playground() -> MergerPreset {
    let bare_smbh = |label: &str,
                     mass_msun: f64,
                     position_kpc: [f64; 3],
                     velocity_kms: [f64; 3]| GalaxyConfig {
        label: label.to_string(),
        equilibrium_snapshot: None,
        initial_profile: GalaxyInitialProfile::UniformSphere {
            radius_kpc: 0.1,
            velocity_dispersion_kms: 0.0,
            edge_rotation_speed_kms: 0.0,
        },
        halo_mass_msun: 0.0,
        halo_scale_radius_kpc: 1.0,
        halo_particle_count: 0,
        disk_mass_msun: 0.0,
        disk_scale_radius_kpc: 1.0,
        disk_scale_height_kpc: 0.1,
        disk_particle_count: 0,
        bulge_mass_msun: 0.0,
        bulge_scale_radius_kpc: 0.5,
        bulge_particle_count: 0,
        smbh: SmbhConfig {
            mass_msun,
            softening_kpc: 0.008,
            substeps: 16,
        },
        position_kpc,
        velocity_kms,
        disk_tilt_deg: [0.0, 0.0, 0.0],
        color_rgba: [0.4, 0.95, 1.0, 1.0],
    };

    let mut config = major_merger().config;
    config.name = "smbh-playground".to_string();
    config.output_directory = "output/smbh-playground".to_string();
    config.gravity.mesh_resolution = [256, 256, 256];
    config.gravity.halo_softening_kpc = 0.12;
    config.galaxies.truncate(1);
    {
        let host = &mut config.galaxies[0];
        host.label = "Host".to_string();
        host.position_kpc = [0.0, 0.0, 0.0];
        host.velocity_kms = [0.0, 0.0, 0.0];
        host.disk_tilt_deg = [0.0, 0.0, 0.0];
        // Dense stellar sampling so the wakes read clearly; coarser halo with
        // matched softening (same trade as grand-merger).
        host.disk_particle_count = 1_200_000;
        host.bulge_particle_count = 240_000;
        host.halo_particle_count = 850_000;
        host.smbh.mass_msun = 5.0e7;
    }
    // Rotation speeds ~200-230 km/s at these radii for this galaxy; modest
    // deviations just make the orbits eccentric, which is the point.
    config.galaxies.push(bare_smbh(
        "Marauder A",
        2.5e8,
        [4.5, 0.0, 1.0],
        [0.0, 205.0, 35.0],
    ));
    config.galaxies.push(bare_smbh(
        "Marauder B",
        3.5e8,
        [0.0, 7.0, 3.0],
        [-185.0, 0.0, -55.0],
    ));
    config.galaxies.push(bare_smbh(
        "Marauder C",
        5.0e8,
        [-9.5, 0.0, 0.5],
        [0.0, -215.0, 0.0],
    ));
    config.initial_separation_kpc = 0.0;
    config.initial_relative_velocity_kms = 0.0;

    MergerPreset {
        id: "smbh-playground",
        title: "SMBH Marauders",
        summary: "Three massive black holes (marked in cyan) on crossing orbits through a dense spiral disk — watch dynamical-friction wakes carve the stars as they spiral inward.",
        config,
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

fn triple_sphere_orbit_debug() -> MergerPreset {
    let sphere_radius_kpc = 2.2;
    let orbital_radius_kpc = 6.5;
    let halo_mass_msun = 8.5e10;
    let stellar_mass_msun = 1.65e11;
    let smbh_mass_msun = 2.5e8;
    let sphere_total_mass_msun = halo_mass_msun + stellar_mass_msun + smbh_mass_msun;
    let sphere_edge_circular_speed_kms =
        circular_speed_kms(sphere_total_mass_msun, sphere_radius_kpc);

    let make_galaxy = |label: &str, color_rgba: [f32; 4]| GalaxyConfig {
        label: label.to_string(),
        equilibrium_snapshot: None,
        initial_profile: GalaxyInitialProfile::UniformSphere {
            radius_kpc: sphere_radius_kpc,
            velocity_dispersion_kms: sphere_edge_circular_speed_kms * 0.03,
            edge_rotation_speed_kms: sphere_edge_circular_speed_kms * 0.92,
        },
        halo_mass_msun,
        halo_scale_radius_kpc: 0.9,
        halo_particle_count: 20_000,
        disk_mass_msun: 0.0,
        disk_scale_radius_kpc: 1.0,
        disk_scale_height_kpc: 0.2,
        disk_particle_count: 0,
        bulge_mass_msun: stellar_mass_msun,
        bulge_scale_radius_kpc: 0.9,
        bulge_particle_count: 80_000,
        smbh: SmbhConfig {
            mass_msun: smbh_mass_msun,
            softening_kpc: 0.002,
            substeps: 16,
        },
        position_kpc: [0.0, 0.0, 0.0],
        velocity_kms: [0.0, 0.0, 0.0],
        disk_tilt_deg: [0.0, 0.0, 0.0],
        color_rgba,
    };

    let mut galaxies = vec![
        make_galaxy("Sphere A", [1.0, 0.88, 0.76, 1.0]),
        make_galaxy("Sphere B", [0.72, 0.84, 1.0, 1.0]),
        make_galaxy("Sphere C", [0.92, 0.74, 1.0, 1.0]),
    ];
    let orbital_speed = set_equal_triple_ring_orbit(&mut galaxies, orbital_radius_kpc);

    MergerPreset {
        id: "triple-sphere-orbit-debug",
        title: "Triple Sphere Orbit Debug",
        summary: "Three close rotating stellar spheres with central SMBHs, orbiting the origin in a symmetric three-body ring.",
        config: SimulationConfig {
            name: "triple-sphere-orbit-debug".to_string(),
            gravity: GravityConfig {
                mesh_resolution: [160, 160, 128],
                ..gravity_defaults()
            },
            relativity: relativity_defaults(),
            preview: PreviewConfig {
                particle_budget: 98_304,
                target_fps: 120,
            },
            snapshots: snapshot_defaults(),
            integration: TimeIntegrationConfig {
                base_timestep_myr: 0.05,
                max_substeps: 16,
                cfl_safety_factor: 0.2,
            },
            initial_separation_kpc: orbital_radius_kpc * 2.0,
            initial_relative_velocity_kms: orbital_speed,
            output_directory: "output/triple-sphere-orbit-debug".to_string(),
            galaxies,
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
    // Match the primary's halo mass-per-particle (~5.25e6 Msun) so the overlapping
    // halos don't two-body heat each other; halo particles are invisible in the
    // viewer, so extra resolution here buys nothing visually.
    config.galaxies[1].halo_particle_count = 190_000;
    config.galaxies[1].disk_mass_msun = 4.6e10;
    config.galaxies[1].disk_particle_count = 360_000;
    config.galaxies[1].bulge_mass_msun = 7.5e9;
    config.galaxies[1].bulge_particle_count = 48_000;
    config.galaxies[1].smbh.mass_msun = 3.0e6;
    config.galaxies[1].color_rgba = [0.59, 1.0, 0.74, 1.0];
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
