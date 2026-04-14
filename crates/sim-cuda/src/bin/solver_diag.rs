use std::{
    env,
    path::{Path, PathBuf},
    process::Command,
    time::Instant,
};

use anyhow::{Context, Result, bail};
use sim_core::{
    CURRENT_SNAPSHOT_SCHEMA_VERSION, InitialConditions, Particle, ParticleComponent,
    SimulationConfig, SnapshotManifest, Vec3, built_in_presets,
};
use sim_cuda::GpuBackend;

#[derive(Clone, Debug)]
struct DiagArgs {
    preset_id: String,
    steps: u32,
    batch: u32,
    relax_steps: u32,
    seed: u64,
    isolate_galaxy: Option<usize>,
    base_timestep_myr: Option<f64>,
}

#[derive(Clone, Debug)]
struct GalaxyMetrics {
    disk_r50_kpc: f64,
    disk_r90_kpc: f64,
    disk_rms_height_kpc: f64,
    disk_spin: f64,
    disk_mean_radial_velocity_kms: f64,
    disk_count: usize,
}

#[derive(Clone, Debug)]
struct SystemMetrics {
    center_of_mass_kpc: Vec3,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let preset = built_in_presets()
        .into_iter()
        .find(|preset| preset.id == args.preset_id)
        .with_context(|| format!("unknown preset `{}`", args.preset_id))?;
    ensure_equilibrium_snapshots_for_config(&preset.config, &args.preset_id, args.seed)
        .context("prepare equilibrium snapshot cache")?;

    let mut initial_conditions =
        InitialConditions::generate(&preset.config, args.seed).context("generate ICs")?;
    let mut config = preset.config.clone();
    if let Some(base_timestep_myr) = args.base_timestep_myr {
        config.integration.base_timestep_myr = base_timestep_myr;
    }
    if let Some(galaxy_index) = args.isolate_galaxy {
        let isolated = isolate_galaxy_state(&config, &initial_conditions, galaxy_index)
            .with_context(|| format!("isolate galaxy {galaxy_index}"))?;
        config = isolated.0;
        initial_conditions = isolated.1;
    }
    if args.relax_steps > 0 {
        initial_conditions = relax_initial_conditions(&config, initial_conditions, args.relax_steps)
            .context("relax ICs")?;
    }

    let initial_metrics = compute_all_metrics(&initial_conditions.particles, &config);
    let initial_system = compute_system_metrics(&initial_conditions.particles);
    print_metrics(
        "initial",
        0.0,
        0.0,
        0.0,
        1.0,
        Vec3::ZERO,
        &initial_system,
        &initial_metrics,
        &initial_metrics,
    );

    let mut backend = GpuBackend::new(&config, &initial_conditions).context("create backend")?;
    let mut completed = 0_u32;
    let mut baseline_total_energy = None;
    while completed < args.steps {
        let run_steps = args.batch.min(args.steps - completed).max(1);
        let wall_start = Instant::now();
        let diagnostics = backend.advance(run_steps).context("advance backend")?;
        let wall_seconds = wall_start.elapsed().as_secs_f64();
        let total_energy = diagnostics.kinetic_energy + diagnostics.estimated_potential_energy;
        let baseline_energy = *baseline_total_energy.get_or_insert(total_energy);
        let energy_ratio = if baseline_energy.abs() > 0.0 {
            total_energy / baseline_energy
        } else {
            1.0
        };
        let particles = backend.download_particles().context("download particles")?;
        let metrics = compute_all_metrics(&particles, &config);
        let system = compute_system_metrics(&particles);
        print_metrics(
            &format!("step+{run_steps}"),
            diagnostics.sim_time_myr,
            wall_seconds,
            total_energy,
            energy_ratio,
            diagnostics.total_momentum,
            &system,
            &metrics,
            &initial_metrics,
        );
        completed += run_steps;
    }

    Ok(())
}

fn parse_args() -> Result<DiagArgs> {
    let mut args = DiagArgs {
        preset_id: "major-merger-debug".to_string(),
        steps: 16,
        batch: 1,
        relax_steps: 16,
        seed: 42,
        isolate_galaxy: None,
        base_timestep_myr: None,
    };

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--preset" => args.preset_id = next_value(&mut it, "--preset")?,
            "--steps" => args.steps = next_value(&mut it, "--steps")?.parse()?,
            "--batch" => args.batch = next_value(&mut it, "--batch")?.parse()?,
            "--relax-steps" => args.relax_steps = next_value(&mut it, "--relax-steps")?.parse()?,
            "--seed" => args.seed = next_value(&mut it, "--seed")?.parse()?,
            "--base-timestep-myr" => {
                args.base_timestep_myr = Some(next_value(&mut it, "--base-timestep-myr")?.parse()?)
            }
            "--isolate-galaxy" => {
                args.isolate_galaxy = Some(next_value(&mut it, "--isolate-galaxy")?.parse()?)
            }
            "--no-relax" => args.relax_steps = 0,
            "--help" | "-h" => {
                println!(
                    "usage: cargo run -p sim-cuda --bin solver_diag -- [--preset ID] [--steps N] [--batch N] [--relax-steps N] [--seed N] [--base-timestep-myr DT] [--isolate-galaxy INDEX]"
                );
                std::process::exit(0);
            }
            other => bail!("unknown argument `{other}`"),
        }
    }

    Ok(args)
}

fn next_value(it: &mut impl Iterator<Item = String>, flag: &str) -> Result<String> {
    it.next().with_context(|| format!("missing value for {flag}"))
}

fn ensure_equilibrium_snapshots_for_config(
    config: &SimulationConfig,
    preset_id: &str,
    seed: u64,
) -> Result<()> {
    for (galaxy_index, galaxy) in config.galaxies.iter().enumerate() {
        let Some(snapshot_path) = galaxy.equilibrium_snapshot.as_deref() else {
            continue;
        };
        if equilibrium_snapshot_is_current(snapshot_path)? {
            continue;
        }
        generate_equilibrium_snapshot(preset_id, galaxy_index, snapshot_path, seed)
            .with_context(|| format!("generate equilibrium snapshot for {preset_id} galaxy {galaxy_index}"))?;
    }
    Ok(())
}

fn equilibrium_snapshot_is_current(snapshot_path: &str) -> Result<bool> {
    let manifest_path = Path::new(snapshot_path);
    if !manifest_path.exists() {
        return Ok(false);
    }
    let bytes = std::fs::read(manifest_path)
        .with_context(|| format!("failed to read equilibrium manifest {}", manifest_path.display()))?;
    let manifest: SnapshotManifest = serde_json::from_slice(&bytes).with_context(|| {
        format!(
            "failed to decode equilibrium manifest {}",
            manifest_path.display()
        )
    })?;
    Ok(manifest.schema_version >= CURRENT_SNAPSHOT_SCHEMA_VERSION)
}

fn generate_equilibrium_snapshot(
    preset_id: &str,
    galaxy_index: usize,
    snapshot_path: &str,
    seed: u64,
) -> Result<()> {
    let snapshot_path = PathBuf::from(snapshot_path);
    if let Some(parent) = snapshot_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create equilibrium snapshot directory {}", parent.display()))?;
    }

    let mut command = resolve_equilibrate_command();
    command
        .arg("--preset")
        .arg(preset_id)
        .arg("--galaxy")
        .arg(galaxy_index.to_string())
        .arg("--iterations")
        .arg("4")
        .arg("--settle-steps")
        .arg("12")
        .arg("--seed")
        .arg(seed.to_string())
        .arg("--output")
        .arg(&snapshot_path);

    let output = command
        .output()
        .with_context(|| format!("spawn equilibrium generator for {preset_id} galaxy {galaxy_index}"))?;
    if output.status.success() {
        return Ok(());
    }

    bail!(
        "equilibrium generator failed with status {}:\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout).trim(),
        String::from_utf8_lossy(&output.stderr).trim()
    )
}

fn resolve_equilibrate_command() -> Command {
    if let Ok(explicit) = env::var("GALAXY_EQUILIBRATE_BIN") {
        return Command::new(explicit);
    }

    if let Ok(current_exe) = env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            let sibling = parent.join("equilibrate");
            let sibling_is_current = sibling.exists()
                && match (
                    sibling.metadata().and_then(|meta| meta.modified()),
                    current_exe.metadata().and_then(|meta| meta.modified()),
                ) {
                    (Ok(sibling_time), Ok(current_time)) => sibling_time >= current_time,
                    _ => false,
                };
            if sibling_is_current {
                return Command::new(sibling);
            }
        }
    }

    let mut command = Command::new("cargo");
    command
        .arg("run")
        .arg("--release")
        .arg("-p")
        .arg("sim-cuda")
        .arg("--bin")
        .arg("equilibrate")
        .arg("--");
    command.current_dir("/galaxy");
    command
}

fn relax_initial_conditions(
    config: &SimulationConfig,
    initial_conditions: InitialConditions,
    relaxation_steps: u32,
) -> Result<InitialConditions> {
    if relaxation_steps == 0 || config.galaxies.is_empty() {
        return Ok(initial_conditions);
    }

    let mut relaxed_particles = Vec::with_capacity(initial_conditions.particles.len());
    for (galaxy_index, galaxy) in config.galaxies.iter().enumerate() {
        let target_origin = Vec3::new(
            galaxy.position_kpc[0],
            galaxy.position_kpc[1],
            galaxy.position_kpc[2],
        );
        let target_velocity = Vec3::new(
            galaxy.velocity_kms[0],
            galaxy.velocity_kms[1],
            galaxy.velocity_kms[2],
        );

        let isolated_particles: Vec<Particle> = initial_conditions
            .particles
            .iter()
            .filter(|particle| particle.galaxy_index == galaxy_index as u32)
            .cloned()
            .map(|mut particle| {
                particle.position_kpc -= target_origin;
                particle.velocity_kms -= target_velocity;
                particle.galaxy_index = 0;
                particle
            })
            .collect();
        if isolated_particles.is_empty() {
            continue;
        }

        let isolated_mass = isolated_particles.iter().map(|particle| particle.mass_msun).sum::<f64>();
        let mut isolated_config = config.clone();
        isolated_config.name = format!("{}-diag-relax-{galaxy_index}", config.name);
        isolated_config.galaxies = vec![{
            let mut isolated = galaxy.clone();
            isolated.position_kpc = [0.0, 0.0, 0.0];
            isolated.velocity_kms = [0.0, 0.0, 0.0];
            isolated
        }];
        let isolated_ic = InitialConditions {
            seed: initial_conditions.seed
                ^ (galaxy_index as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15),
            particles: isolated_particles,
            total_mass_msun: isolated_mass,
        };
        let mut backend = GpuBackend::new(&isolated_config, &isolated_ic)
            .with_context(|| format!("create isolated backend for galaxy {galaxy_index}"))?;
        backend
            .advance(relaxation_steps)
            .with_context(|| format!("relax galaxy {galaxy_index}"))?;
        let mut particles = backend
            .download_particles()
            .with_context(|| format!("download relaxed galaxy {galaxy_index}"))?;
        recenter_particles(&mut particles, galaxy_index as u32, target_origin, target_velocity);
        relaxed_particles.extend(particles.into_iter().map(|mut particle| {
            particle.galaxy_index = galaxy_index as u32;
            particle
        }));
    }

    Ok(InitialConditions {
        seed: initial_conditions.seed,
        particles: relaxed_particles,
        total_mass_msun: initial_conditions.total_mass_msun,
    })
}

fn isolate_galaxy_state(
    config: &SimulationConfig,
    initial_conditions: &InitialConditions,
    galaxy_index: usize,
) -> Result<(SimulationConfig, InitialConditions)> {
    let galaxy = config
        .galaxies
        .get(galaxy_index)
        .with_context(|| format!("galaxy index {galaxy_index} out of range"))?
        .clone();
    let target_origin = Vec3::new(
        galaxy.position_kpc[0],
        galaxy.position_kpc[1],
        galaxy.position_kpc[2],
    );
    let target_velocity = Vec3::new(
        galaxy.velocity_kms[0],
        galaxy.velocity_kms[1],
        galaxy.velocity_kms[2],
    );

    let mut particles: Vec<Particle> = initial_conditions
        .particles
        .iter()
        .filter(|particle| particle.galaxy_index == galaxy_index as u32)
        .cloned()
        .collect();
    if particles.is_empty() {
        bail!("galaxy {galaxy_index} has no particles");
    }
    for particle in &mut particles {
        particle.position_kpc -= target_origin;
        particle.velocity_kms -= target_velocity;
        particle.galaxy_index = 0;
    }

    let total_mass_msun = particles.iter().map(|particle| particle.mass_msun).sum::<f64>();
    let mut isolated_config = config.clone();
    isolated_config.name = format!("{}-isolated-{galaxy_index}", config.name);
    isolated_config.output_directory =
        format!("{}/isolated-{galaxy_index}", config.output_directory);
    isolated_config.initial_separation_kpc = 0.0;
    isolated_config.initial_relative_velocity_kms = 0.0;
    isolated_config.galaxies = vec![{
        let mut isolated = galaxy;
        isolated.position_kpc = [0.0, 0.0, 0.0];
        isolated.velocity_kms = [0.0, 0.0, 0.0];
        isolated
    }];

    Ok((
        isolated_config,
        InitialConditions {
            seed: initial_conditions.seed,
            particles,
            total_mass_msun,
        },
    ))
}

fn recenter_particles(
    particles: &mut [Particle],
    galaxy_index: u32,
    origin: Vec3,
    bulk_velocity: Vec3,
) {
    if particles.is_empty() {
        return;
    }

    let mut total_mass = 0.0;
    let mut center_of_mass = Vec3::ZERO;
    let mut center_velocity = Vec3::ZERO;
    for particle in particles.iter() {
        total_mass += particle.mass_msun;
        center_of_mass += particle.position_kpc * particle.mass_msun;
        center_velocity += particle.velocity_kms * particle.mass_msun;
    }
    if total_mass > 0.0 {
        center_of_mass = center_of_mass / total_mass;
        center_velocity = center_velocity / total_mass;
    }

    let position_shift = origin - center_of_mass;
    let velocity_shift = bulk_velocity - center_velocity;
    for particle in particles.iter_mut() {
        particle.position_kpc += position_shift;
        particle.velocity_kms += velocity_shift;
    }

    if let Some(smbh) = particles.iter_mut().find(|particle| {
        particle.galaxy_index == galaxy_index && matches!(particle.component, ParticleComponent::Smbh)
    }) {
        smbh.position_kpc = origin;
        smbh.velocity_kms = bulk_velocity;
    }
}

fn compute_all_metrics(particles: &[Particle], config: &SimulationConfig) -> Vec<GalaxyMetrics> {
    config
        .galaxies
        .iter()
        .enumerate()
        .map(|(galaxy_index, _)| compute_galaxy_metrics(particles, galaxy_index as u32))
        .collect()
}

fn compute_galaxy_metrics(particles: &[Particle], galaxy_index: u32) -> GalaxyMetrics {
    let mut center = Vec3::ZERO;
    let mut bulk_velocity = Vec3::ZERO;
    if let Some(smbh) = particles.iter().find(|particle| {
        particle.galaxy_index == galaxy_index && matches!(particle.component, ParticleComponent::Smbh)
    }) {
        center = smbh.position_kpc;
        bulk_velocity = smbh.velocity_kms;
    }

    let mut tracer_particles: Vec<&Particle> = particles
        .iter()
        .filter(|particle| {
            particle.galaxy_index == galaxy_index && matches!(particle.component, ParticleComponent::Disk)
        })
        .collect();
    if tracer_particles.is_empty() {
        tracer_particles = particles
            .iter()
            .filter(|particle| {
                particle.galaxy_index == galaxy_index
                    && !matches!(particle.component, ParticleComponent::Halo | ParticleComponent::Smbh)
            })
            .collect();
    }
    if tracer_particles.is_empty() {
        return GalaxyMetrics {
            disk_r50_kpc: 0.0,
            disk_r90_kpc: 0.0,
            disk_rms_height_kpc: 0.0,
            disk_spin: 0.0,
            disk_mean_radial_velocity_kms: 0.0,
            disk_count: 0,
        };
    }

    let total_angular_momentum = tracer_particles.iter().fold(Vec3::ZERO, |accum, particle| {
        let r = particle.position_kpc - center;
        let v = particle.velocity_kms - bulk_velocity;
        accum
            + Vec3::new(
                r.y * v.z - r.z * v.y,
                r.z * v.x - r.x * v.z,
                r.x * v.y - r.y * v.x,
            ) * particle.mass_msun
    });
    let disk_normal = total_angular_momentum.normalized();
    let spin = total_angular_momentum.length();

    let mut cylindrical_radii = Vec::with_capacity(tracer_particles.len());
    let mut height_sq_sum = 0.0;
    let mut radial_velocity_sum = 0.0;
    for particle in &tracer_particles {
        let r = particle.position_kpc - center;
        let v = particle.velocity_kms - bulk_velocity;
        let height = r.dot(disk_normal);
        let in_plane = r - disk_normal * height;
        let cylindrical_radius = in_plane.length();
        cylindrical_radii.push(cylindrical_radius);
        height_sq_sum += height * height;
        if cylindrical_radius > 1.0e-6 {
            radial_velocity_sum += in_plane.dot(v) / cylindrical_radius;
        }
    }
    cylindrical_radii.sort_by(|a, b| a.partial_cmp(b).unwrap());

    GalaxyMetrics {
        disk_r50_kpc: quantile(&cylindrical_radii, 0.5),
        disk_r90_kpc: quantile(&cylindrical_radii, 0.9),
        disk_rms_height_kpc: (height_sq_sum / tracer_particles.len() as f64).sqrt(),
        disk_spin: spin,
        disk_mean_radial_velocity_kms: radial_velocity_sum / tracer_particles.len() as f64,
        disk_count: tracer_particles.len(),
    }
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}

fn compute_system_metrics(particles: &[Particle]) -> SystemMetrics {
    let mut total_mass = 0.0;
    let mut weighted_position = Vec3::ZERO;
    for particle in particles {
        total_mass += particle.mass_msun;
        weighted_position += particle.position_kpc * particle.mass_msun;
    }
    let center_of_mass_kpc = if total_mass > 0.0 {
        weighted_position / total_mass
    } else {
        Vec3::ZERO
    };
    SystemMetrics { center_of_mass_kpc }
}

fn print_metrics(
    label: &str,
    sim_time_myr: f64,
    wall_seconds: f64,
    total_energy: f64,
    energy_ratio: f64,
    total_momentum: Vec3,
    system: &SystemMetrics,
    metrics: &[GalaxyMetrics],
    baseline: &[GalaxyMetrics],
) {
    let momentum_mag = total_momentum.length();
    println!(
        "label={label} sim_time_myr={sim_time_myr:.3} wall_s={wall_seconds:.3} total_energy={total_energy:.6e} total_energy_ratio={energy_ratio:.6} momentum_mag={momentum_mag:.6e} com=({:.4},{:.4},{:.4})",
        system.center_of_mass_kpc.x,
        system.center_of_mass_kpc.y,
        system.center_of_mass_kpc.z,
    );
    for (index, (current, initial)) in metrics.iter().zip(baseline.iter()).enumerate() {
        let r50_ratio = current.disk_r50_kpc / initial.disk_r50_kpc.max(1.0e-6);
        let r90_ratio = current.disk_r90_kpc / initial.disk_r90_kpc.max(1.0e-6);
        let z_ratio = current.disk_rms_height_kpc / initial.disk_rms_height_kpc.max(1.0e-6);
        let spin_ratio = current.disk_spin / initial.disk_spin.max(1.0e-6);
        println!(
            " galaxy={index} disk_count={} r50_kpc={:.3} r50_ratio={:.3} r90_kpc={:.3} r90_ratio={:.3} z_rms_kpc={:.3} z_ratio={:.3} spin_ratio={:.3} mean_vr_kms={:.3}",
            current.disk_count,
            current.disk_r50_kpc,
            r50_ratio,
            current.disk_r90_kpc,
            r90_ratio,
            current.disk_rms_height_kpc,
            z_ratio,
            spin_ratio,
            current.disk_mean_radial_velocity_kms
        );
    }
}
