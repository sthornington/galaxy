//! Force-accuracy harness: zeroes all velocities, advances one tiny KDK step so
//! the measured velocity change is proportional to the acceleration, and
//! compares the solver's accelerations against a brute-force direct sum on a
//! random sample of particles.

use anyhow::{Context, Result};
use sim_core::{InitialConditions, Particle, Vec3, built_in_presets};
use sim_cuda::GpuBackend;

/// Minimal splitmix64 so the sample selection is reproducible without pulling
/// a rand dependency into this crate.
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_index(&mut self, bound: usize) -> usize {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^= z >> 31;
        (z % bound as u64) as usize
    }
}

const KPC_PER_KM_PER_MYR: f64 = 0.001022712165045695;

fn main() -> Result<()> {
    let mut preset_id = "major-merger-debug".to_string();
    let mut sample_count = 256_usize;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--preset" => preset_id = it.next().context("missing --preset value")?,
            "--samples" => sample_count = it.next().context("missing --samples value")?.parse()?,
            other => anyhow::bail!("unknown argument `{other}`"),
        }
    }

    let mut preset = built_in_presets()
        .into_iter()
        .find(|preset| preset.id == preset_id)
        .with_context(|| format!("unknown preset `{preset_id}`"))?;
    // Analytic ICs only: this measures force accuracy, not equilibrium quality.
    for galaxy in &mut preset.config.galaxies {
        galaxy.equilibrium_snapshot = None;
    }

    // This harness pairs solver output with the reference by array index, so
    // the backend's cell-order compaction must stay off.
    unsafe {
        std::env::set_var("SIM_CUDA_DISABLE_REORDER", "1");
        std::env::set_var("SIM_CUDA_PM_SUBSTEP_INTERVAL", "1");
    }

    let mut initial_conditions = InitialConditions::generate(&preset.config, 7)?;
    for particle in &mut initial_conditions.particles {
        particle.velocity_kms = Vec3::ZERO;
    }
    let particles = initial_conditions.particles.clone();

    let dt_myr = 1.0e-4;
    let mut config = preset.config.clone();
    config.integration.base_timestep_myr = dt_myr;
    config.integration.max_substeps = 1;
    // 1PN needs velocities; irrelevant here.
    config.relativity.enable_smbh_post_newtonian = false;

    let mut backend = GpuBackend::new(&config, &initial_conditions)?;
    backend.step(1)?;
    let kicked = backend.download_particles()?;
    anyhow::ensure!(kicked.len() == particles.len());

    let grav = config.gravity.grav_const_kpc_kms2_per_msun;
    let mut rng = SplitMix64(99);
    let mut ratios = Vec::new();
    println!("component ratio_along |a_solver| |a_direct| rel_err radius_kpc");
    for _ in 0..sample_count {
        let target_index = rng.next_index(particles.len());
        let target = &particles[target_index];
        let mut direct = Vec3::ZERO;
        for (source_index, source) in particles.iter().enumerate() {
            if source_index == target_index {
                continue;
            }
            let delta = source.position_kpc - target.position_kpc;
            let softening = target.softening_kpc.max(source.softening_kpc);
            let r2 = delta.length_squared() + softening * softening;
            let inv_r = 1.0 / r2.sqrt();
            direct += delta * (grav * source.mass_msun * inv_r * inv_r * inv_r);
        }

        // One base step = two half-kicks of dt/2 around a drift; with zero
        // initial velocities the drift is negligible, so dv = a * dt.
        let solver = (kicked[target_index].velocity_kms - target.velocity_kms)
            / (dt_myr * KPC_PER_KM_PER_MYR);

        let direct_mag = direct.length().max(1.0e-12);
        let solver_mag = solver.length();
        let rel_err = (solver - direct).length() / direct_mag;
        let along = solver.dot(direct) / (direct_mag * direct_mag);
        let radius = radius_from_galaxy(target, &config);
        ratios.push(rel_err);
        println!(
            "{:?} {:.4} {:.4e} {:.4e} {:.4} {:.2}",
            target.component, along, solver_mag, direct_mag, rel_err, radius
        );
    }

    ratios.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let quantile = |q: f64| ratios[((ratios.len() - 1) as f64 * q) as usize];
    println!(
        "\nrel force error: p50={:.4} p90={:.4} p99={:.4} max={:.4}",
        quantile(0.5),
        quantile(0.9),
        quantile(0.99),
        quantile(1.0)
    );
    Ok(())
}

fn radius_from_galaxy(particle: &Particle, config: &sim_core::SimulationConfig) -> f64 {
    let galaxy = &config.galaxies[particle.galaxy_index as usize];
    let origin = Vec3::new(
        galaxy.position_kpc[0],
        galaxy.position_kpc[1],
        galaxy.position_kpc[2],
    );
    (particle.position_kpc - origin).length()
}
