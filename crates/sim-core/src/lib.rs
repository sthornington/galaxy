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
    use super::{InitialConditions, built_in_presets, validate_particle_count};

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
}
