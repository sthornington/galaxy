pub mod config;
pub mod init;
pub mod math;
pub mod preset;
pub mod preview;
pub mod snapshot;

pub use config::{
    GalaxyConfig, GalaxyInitialProfile, GravityConfig, ObserverEffectsConfig, PreviewConfig,
    RelativityConfig, SimulationConfig, SmbhConfig, SnapshotConfig, TimeIntegrationConfig,
};
pub use init::{
    InitialConditionError, InitialConditions, Particle, ParticleComponent, generate_analytic_galaxy,
    validate_particle_count,
};
pub use math::Vec3;
pub use preset::{MergerPreset, built_in_presets};
pub use preview::{
    Diagnostics, PreviewFrame, PreviewPacketHeader, PreviewPacketParticle, PreviewParticle,
    decode_preview_packet, encode_preview_packet_into,
};
pub use snapshot::{
    CURRENT_SNAPSHOT_SCHEMA_VERSION, SnapshotChunk, SnapshotManifest, load_particle_snapshot,
    write_particle_snapshot,
};

#[cfg(test)]
mod tests {
    use std::{
        fs,
        time::{SystemTime, UNIX_EPOCH},
    };

    use super::{
        Diagnostics, InitialConditions, MergerPreset, Particle, ParticleComponent,
        PreviewPacketParticle, Vec3, built_in_presets, decode_preview_packet,
        encode_preview_packet_into, load_particle_snapshot, validate_particle_count,
        write_particle_snapshot,
    };

    fn strip_equilibrium_snapshots(preset: &mut MergerPreset) {
        for galaxy in &mut preset.config.galaxies {
            galaxy.equilibrium_snapshot = None;
        }
    }

    #[test]
    fn built_in_presets_have_unique_ids() {
        let presets = built_in_presets();
        let mut ids = std::collections::BTreeSet::new();
        for preset in &presets {
            assert!(ids.insert(preset.id), "duplicate preset id: {}", preset.id);
        }
    }

    #[test]
    fn initial_conditions_match_configured_particle_count() {
        let mut preset = built_in_presets().remove(0);
        strip_equilibrium_snapshots(&mut preset);
        let expected = validate_particle_count(&preset.config).unwrap();
        let initial_conditions = InitialConditions::generate(&preset.config, 42).unwrap();
        assert_eq!(initial_conditions.particles.len() as u64, expected);
        assert!(initial_conditions.total_mass_msun > 0.0);
    }

    #[test]
    fn minor_merger_disks_have_internal_angular_momentum() {
        let mut preset = built_in_presets()
            .into_iter()
            .find(|preset| preset.id == "minor-merger")
            .unwrap();
        strip_equilibrium_snapshots(&mut preset);
        let initial_conditions = InitialConditions::generate(&preset.config, 42).unwrap();

        for (galaxy_index, galaxy) in preset.config.galaxies.iter().enumerate() {
            let origin = Vec3::new(
                galaxy.position_kpc[0],
                galaxy.position_kpc[1],
                galaxy.position_kpc[2],
            );
            let bulk_velocity = Vec3::new(
                galaxy.velocity_kms[0],
                galaxy.velocity_kms[1],
                galaxy.velocity_kms[2],
            );

            let mut total_angular_momentum = Vec3::ZERO;
            let mut disk_particles = 0_u64;
            for particle in initial_conditions
                .particles
                .iter()
                .filter(|particle| particle.galaxy_index == galaxy_index as u32)
            {
                if !matches!(particle.component, ParticleComponent::Disk) {
                    continue;
                }

                let relative_position = particle.position_kpc - origin;
                let relative_velocity = particle.velocity_kms - bulk_velocity;
                let angular_momentum = Vec3::new(
                    relative_position.y * relative_velocity.z
                        - relative_position.z * relative_velocity.y,
                    relative_position.z * relative_velocity.x
                        - relative_position.x * relative_velocity.z,
                    relative_position.x * relative_velocity.y
                        - relative_position.y * relative_velocity.x,
                ) * particle.mass_msun;

                total_angular_momentum += angular_momentum;
                disk_particles += 1;
            }

            assert!(disk_particles > 0);
            assert!(
                total_angular_momentum.length() > 0.0,
                "galaxy {galaxy_index} disk angular momentum should be non-zero"
            );
        }
    }

    #[test]
    fn major_merger_disks_have_visible_vertical_structure() {
        let mut preset = built_in_presets()
            .into_iter()
            .find(|preset| preset.id == "major-merger")
            .unwrap();
        strip_equilibrium_snapshots(&mut preset);
        let initial_conditions = InitialConditions::generate(&preset.config, 42).unwrap();

        for (galaxy_index, galaxy) in preset.config.galaxies.iter().enumerate() {
            let origin = Vec3::new(
                galaxy.position_kpc[0],
                galaxy.position_kpc[1],
                galaxy.position_kpc[2],
            );
            let bulk_velocity = Vec3::new(
                galaxy.velocity_kms[0],
                galaxy.velocity_kms[1],
                galaxy.velocity_kms[2],
            );

            let disk_particles: Vec<_> = initial_conditions
                .particles
                .iter()
                .filter(|particle| {
                    particle.galaxy_index == galaxy_index as u32
                        && matches!(particle.component, ParticleComponent::Disk)
                })
                .collect();

            assert!(!disk_particles.is_empty());

            let mut total_angular_momentum = Vec3::ZERO;
            for particle in &disk_particles {
                let relative_position = particle.position_kpc - origin;
                let relative_velocity = particle.velocity_kms - bulk_velocity;
                let angular_momentum = Vec3::new(
                    relative_position.y * relative_velocity.z
                        - relative_position.z * relative_velocity.y,
                    relative_position.z * relative_velocity.x
                        - relative_position.x * relative_velocity.z,
                    relative_position.x * relative_velocity.y
                        - relative_position.y * relative_velocity.x,
                ) * particle.mass_msun;
                total_angular_momentum += angular_momentum;
            }

            let disk_normal = total_angular_momentum.normalized();
            let rms_height = (disk_particles
                .iter()
                .map(|particle| {
                    let relative_position = particle.position_kpc - origin;
                    let height = relative_position.dot(disk_normal);
                    height * height
                })
                .sum::<f64>()
                / disk_particles.len() as f64)
                .sqrt();

            assert!(
                rms_height > galaxy.disk_scale_height_kpc * 0.35,
                "galaxy {galaxy_index} disk should not collapse into an unrealistically thin sheet"
            );
        }
    }

    #[test]
    fn major_merger_disks_have_exponential_radial_scale() {
        let mut preset = built_in_presets()
            .into_iter()
            .find(|preset| preset.id == "major-merger")
            .unwrap();
        strip_equilibrium_snapshots(&mut preset);
        let initial_conditions = InitialConditions::generate(&preset.config, 42).unwrap();

        for (galaxy_index, galaxy) in preset.config.galaxies.iter().enumerate() {
            let origin = Vec3::new(
                galaxy.position_kpc[0],
                galaxy.position_kpc[1],
                galaxy.position_kpc[2],
            );
            let bulk_velocity = Vec3::new(
                galaxy.velocity_kms[0],
                galaxy.velocity_kms[1],
                galaxy.velocity_kms[2],
            );

            let mut total_angular_momentum = Vec3::ZERO;
            let disk_particles: Vec<_> = initial_conditions
                .particles
                .iter()
                .filter(|particle| {
                    particle.galaxy_index == galaxy_index as u32
                        && matches!(particle.component, ParticleComponent::Disk)
                })
                .collect();
            assert!(!disk_particles.is_empty());

            for particle in &disk_particles {
                let relative_position = particle.position_kpc - origin;
                let relative_velocity = particle.velocity_kms - bulk_velocity;
                total_angular_momentum += Vec3::new(
                    relative_position.y * relative_velocity.z
                        - relative_position.z * relative_velocity.y,
                    relative_position.z * relative_velocity.x
                        - relative_position.x * relative_velocity.z,
                    relative_position.x * relative_velocity.y
                        - relative_position.y * relative_velocity.x,
                ) * particle.mass_msun;
            }
            let disk_normal = total_angular_momentum.normalized();

            let mean_cylindrical_radius = disk_particles
                .iter()
                .map(|particle| {
                    let relative_position = particle.position_kpc - origin;
                    let height = relative_position.dot(disk_normal);
                    let in_plane = relative_position - disk_normal * height;
                    in_plane.length()
                })
                .sum::<f64>()
                / disk_particles.len() as f64;
            let scale_ratio = mean_cylindrical_radius / galaxy.disk_scale_radius_kpc.max(1.0e-6);

            assert!(
                (1.6..=2.4).contains(&scale_ratio),
                "galaxy {galaxy_index} disk radius sampler drifted from exponential expectation: mean_R/Rd={scale_ratio:.3}"
            );
        }
    }

    #[test]
    fn galaxies_are_recentred_to_configured_origin_and_bulk_velocity() {
        let mut preset = built_in_presets()
            .into_iter()
            .find(|preset| preset.id == "major-merger")
            .unwrap();
        strip_equilibrium_snapshots(&mut preset);
        let initial_conditions = InitialConditions::generate(&preset.config, 42).unwrap();

        for (galaxy_index, galaxy) in preset.config.galaxies.iter().enumerate() {
            let mut total_mass = 0.0;
            let mut center_of_mass = Vec3::ZERO;
            let mut center_velocity = Vec3::ZERO;

            for particle in initial_conditions
                .particles
                .iter()
                .filter(|particle| particle.galaxy_index == galaxy_index as u32)
            {
                total_mass += particle.mass_msun;
                center_of_mass += particle.position_kpc * particle.mass_msun;
                center_velocity += particle.velocity_kms * particle.mass_msun;
            }

            center_of_mass = center_of_mass / total_mass.max(1.0);
            center_velocity = center_velocity / total_mass.max(1.0);
            let target_position =
                Vec3::new(galaxy.position_kpc[0], galaxy.position_kpc[1], galaxy.position_kpc[2]);
            let target_velocity =
                Vec3::new(galaxy.velocity_kms[0], galaxy.velocity_kms[1], galaxy.velocity_kms[2]);

            assert!(
                (center_of_mass - target_position).length() < 1.0e-2,
                "galaxy {galaxy_index} COM drifted from configured origin"
            );
            assert!(
                (center_velocity - target_velocity).length() < 1.0e-2,
                "galaxy {galaxy_index} bulk velocity drifted from configured orbit"
            );
        }
    }

    #[test]
    fn major_merger_orbit_is_bound_and_mass_weighted() {
        let mut preset = built_in_presets()
            .into_iter()
            .find(|preset| preset.id == "major-merger")
            .unwrap();
        strip_equilibrium_snapshots(&mut preset);
        let galaxies = &preset.config.galaxies;
        let m1 = galaxies[0].halo_mass_msun
            + galaxies[0].disk_mass_msun
            + galaxies[0].bulge_mass_msun
            + galaxies[0].smbh.mass_msun;
        let m2 = galaxies[1].halo_mass_msun
            + galaxies[1].disk_mass_msun
            + galaxies[1].bulge_mass_msun
            + galaxies[1].smbh.mass_msun;
        let total_mass = m1 + m2;

        let position1 =
            Vec3::new(galaxies[0].position_kpc[0], galaxies[0].position_kpc[1], galaxies[0].position_kpc[2]);
        let position2 =
            Vec3::new(galaxies[1].position_kpc[0], galaxies[1].position_kpc[1], galaxies[1].position_kpc[2]);
        let velocity1 =
            Vec3::new(galaxies[0].velocity_kms[0], galaxies[0].velocity_kms[1], galaxies[0].velocity_kms[2]);
        let velocity2 =
            Vec3::new(galaxies[1].velocity_kms[0], galaxies[1].velocity_kms[1], galaxies[1].velocity_kms[2]);

        let barycenter = (position1 * m1 + position2 * m2) / total_mass;
        let baryvelocity = (velocity1 * m1 + velocity2 * m2) / total_mass;
        let separation = (position2 - position1).length();
        let relative_speed = (velocity2 - velocity1).length();
        let specific_orbital_energy =
            0.5 * relative_speed * relative_speed - 4.300_91e-6 * total_mass / separation;

        assert!(barycenter.length() < 1.0e-6, "orbit should be centered on the barycenter");
        assert!(
            baryvelocity.length() < 1.0e-6,
            "orbit should start with zero net linear momentum"
        );
        assert!(specific_orbital_energy < 0.0, "major merger should start on a bound orbit");
    }

    #[test]
    fn initial_conditions_can_load_equilibrium_snapshot_per_galaxy() {
        let mut preset = built_in_presets()
            .into_iter()
            .find(|preset| preset.id == "major-merger")
            .unwrap();
        preset.config.galaxies.truncate(1);
        preset.config.galaxies[0].equilibrium_snapshot = None;

        let timestamp = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let temp_root = std::env::temp_dir().join(format!("galaxy-snapshot-test-{timestamp}"));
        let manifest_path = temp_root.join("manifest.json");

        let snapshot_particles = vec![
            Particle {
                galaxy_index: 0,
                component: ParticleComponent::Disk,
                position_kpc: Vec3::new(1.0, 0.0, 0.5),
                velocity_kms: Vec3::new(0.0, 10.0, 0.0),
                mass_msun: 2.0,
                softening_kpc: 0.01,
                color_rgba: [0.1, 0.2, 0.3, 1.0],
            },
            Particle {
                galaxy_index: 0,
                component: ParticleComponent::Smbh,
                position_kpc: Vec3::ZERO,
                velocity_kms: Vec3::ZERO,
                mass_msun: 1.0,
                softening_kpc: 0.001,
                color_rgba: [1.0, 1.0, 1.0, 1.0],
            },
        ];
        write_particle_snapshot(&manifest_path, "test", 0.0, &snapshot_particles).unwrap();
        let (_manifest, loaded_snapshot) = load_particle_snapshot(&manifest_path).unwrap();
        assert_eq!(loaded_snapshot.len(), snapshot_particles.len());

        preset.config.galaxies[0].equilibrium_snapshot =
            Some(manifest_path.display().to_string());
        let initial_conditions = InitialConditions::generate(&preset.config, 42).unwrap();
        assert_eq!(initial_conditions.particles.len(), snapshot_particles.len());

        let smbh = initial_conditions
            .particles
            .iter()
            .find(|particle| matches!(particle.component, ParticleComponent::Smbh))
            .unwrap();
        let target_origin = Vec3::new(
            preset.config.galaxies[0].position_kpc[0],
            preset.config.galaxies[0].position_kpc[1],
            preset.config.galaxies[0].position_kpc[2],
        );
        let target_velocity = Vec3::new(
            preset.config.galaxies[0].velocity_kms[0],
            preset.config.galaxies[0].velocity_kms[1],
            preset.config.galaxies[0].velocity_kms[2],
        );
        assert_eq!(smbh.position_kpc, target_origin);
        assert_eq!(smbh.velocity_kms, target_velocity);

        let disk = initial_conditions
            .particles
            .iter()
            .find(|particle| matches!(particle.component, ParticleComponent::Disk))
            .unwrap();
        assert!(disk.position_kpc.length() > 0.0);
        assert_eq!(disk.color_rgba, preset.config.galaxies[0].color_rgba);

        fs::remove_dir_all(temp_root).ok();
    }

    #[test]
    fn preview_packet_round_trip_preserves_particles() {
        let diagnostics = Diagnostics {
            particle_count: 42,
            preview_count: 2,
            sim_time_myr: 1.25,
            dt_myr: 0.05,
            kinetic_energy: 12.5,
            estimated_potential_energy: -27.0,
            total_momentum: Vec3::new(1.0, -2.0, 3.5),
        };
        let particles = vec![
            PreviewPacketParticle {
                position_kpc: [1.0, 2.0, 3.0],
                velocity_kms: [4.0, 5.0, 6.0],
                mass_msun: 7.0,
                component: 1,
            },
            PreviewPacketParticle {
                position_kpc: [-1.0, -2.0, -3.0],
                velocity_kms: [-4.0, -5.0, -6.0],
                mass_msun: 8.0,
                component: 2,
            },
        ];

        let mut bytes = Vec::new();
        encode_preview_packet_into(&mut bytes, &diagnostics, &particles);
        let decoded = decode_preview_packet(&bytes).unwrap();

        assert_eq!(decoded.sim_time_myr, diagnostics.sim_time_myr);
        assert_eq!(decoded.diagnostics.particle_count, diagnostics.particle_count);
        assert_eq!(decoded.diagnostics.preview_count, diagnostics.preview_count);
        assert_eq!(decoded.particles.len(), particles.len());
        assert_eq!(decoded.particles[0].position_kpc, particles[0].position_kpc);
        assert_eq!(decoded.particles[1].velocity_kms, particles[1].velocity_kms);
        assert_eq!(decoded.particles[1].component, particles[1].component);
    }
}
