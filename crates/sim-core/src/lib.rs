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
}
