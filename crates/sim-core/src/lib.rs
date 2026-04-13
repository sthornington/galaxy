pub mod config;
pub mod init;
pub mod math;
pub mod preset;
pub mod preview;
pub mod snapshot;

pub use config::{
    GalaxyConfig, GravityConfig, ObserverEffectsConfig, PreviewConfig, RelativityConfig,
    SimulationConfig, SmbhConfig, SnapshotConfig, TimeIntegrationConfig,
};
pub use init::{
    InitialConditionError, InitialConditions, Particle, ParticleComponent, validate_particle_count,
};
pub use math::Vec3;
pub use preset::{MergerPreset, built_in_presets};
pub use preview::{Diagnostics, PreviewFrame, PreviewParticle};
pub use snapshot::{SnapshotChunk, SnapshotManifest};

#[cfg(test)]
mod tests {
    use super::{InitialConditions, ParticleComponent, Vec3, built_in_presets, validate_particle_count};

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
        let preset = built_in_presets().remove(0);
        let expected = validate_particle_count(&preset.config).unwrap();
        let initial_conditions = InitialConditions::generate(&preset.config, 42).unwrap();
        assert_eq!(initial_conditions.particles.len() as u64, expected);
        assert!(initial_conditions.total_mass_msun > 0.0);
    }

    #[test]
    fn minor_merger_disks_have_internal_angular_momentum() {
        let preset = built_in_presets()
            .into_iter()
            .find(|preset| preset.id == "minor-merger")
            .unwrap();
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
        let preset = built_in_presets()
            .into_iter()
            .find(|preset| preset.id == "major-merger")
            .unwrap();
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
        let preset = built_in_presets()
            .into_iter()
            .find(|preset| preset.id == "major-merger")
            .unwrap();
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
        let preset = built_in_presets()
            .into_iter()
            .find(|preset| preset.id == "major-merger")
            .unwrap();
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
        let preset = built_in_presets()
            .into_iter()
            .find(|preset| preset.id == "major-merger")
            .unwrap();
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
}
