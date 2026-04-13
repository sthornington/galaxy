use std::{cmp::Ordering, env, path::PathBuf, time::Instant};

use anyhow::{Context, Result, bail};
use sim_core::{
    InitialConditions, Particle, ParticleComponent, SimulationConfig, Vec3, built_in_presets,
    write_particle_snapshot,
};
use sim_cuda::GpuBackend;

#[derive(Clone, Debug)]
struct Args {
    preset_id: String,
    galaxy_index: usize,
    iterations: u32,
    settle_steps: u32,
    seed: u64,
    output: PathBuf,
}

#[derive(Clone, Debug)]
struct GalaxyMetrics {
    disk_r50_kpc: f64,
    disk_rms_height_kpc: f64,
    disk_spin: f64,
    disk_mean_radial_velocity_kms: f64,
    disk_count: usize,
}

#[derive(Clone, Debug)]
struct DiskVelocitySample {
    radius: f64,
    abs_height: f64,
    v_r: f64,
    v_phi: f64,
    v_z: f64,
}

#[derive(Clone, Debug)]
struct SphericalVelocitySample {
    radius: f64,
    position: Vec3,
    velocity: Vec3,
}

#[derive(Clone, Copy, Debug)]
struct Frame3 {
    center: Vec3,
    bulk_velocity: Vec3,
    normal: Vec3,
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let preset = built_in_presets()
        .into_iter()
        .find(|preset| preset.id == args.preset_id)
        .with_context(|| format!("unknown preset `{}`", args.preset_id))?;
    let isolated_config =
        build_isolated_config(&preset.config, args.galaxy_index).context("build isolated config")?;

    let mut current_ic = InitialConditions::generate(&isolated_config, args.seed)
        .context("generate initial isolated galaxy")?;
    print_metrics("initial", &current_ic.particles);

    for iteration in 0..args.iterations {
        let started = Instant::now();
        let mut backend = GpuBackend::new(&isolated_config, &current_ic)
            .with_context(|| format!("create backend for iteration {}", iteration + 1))?;
        let diagnostics = backend
            .advance(args.settle_steps)
            .with_context(|| format!("advance iteration {}", iteration + 1))?;
        let evolved_particles = backend
            .download_particles()
            .with_context(|| format!("download particles for iteration {}", iteration + 1))?;
        let wall_seconds = started.elapsed().as_secs_f64();
        print_metrics(
            &format!(
                "iteration-{} sim_time={:.3}Myr wall={:.3}s",
                iteration + 1,
                diagnostics.sim_time_myr,
                wall_seconds
            ),
            &evolved_particles,
        );

        let fresh_seed = args.seed ^ ((iteration as u64 + 1).wrapping_mul(0x9e37_79b9_7f4a_7c15));
        let fresh_ic = InitialConditions::generate(&isolated_config, fresh_seed)
            .with_context(|| format!("generate target positions for iteration {}", iteration + 1))?;
        let remapped_particles = remap_velocity_distribution(&evolved_particles, fresh_ic.particles)
            .with_context(|| format!("remap velocities for iteration {}", iteration + 1))?;
        current_ic = InitialConditions {
            seed: fresh_seed,
            total_mass_msun: remapped_particles.iter().map(|particle| particle.mass_msun).sum(),
            particles: remapped_particles,
        };
        print_metrics(&format!("remapped-{}", iteration + 1), &current_ic.particles);

        if iteration + 1 == args.iterations {
            break;
        }
    }

    write_particle_snapshot(
        &args.output,
        &format!("{}-galaxy-{}-equilibrated", args.preset_id, args.galaxy_index),
        0.0,
        &current_ic.particles,
    )
    .with_context(|| format!("write equilibrated snapshot {}", args.output.display()))?;

    println!("wrote {}", args.output.display());
    Ok(())
}

fn parse_args() -> Result<Args> {
    let mut args = Args {
        preset_id: "major-merger-debug".to_string(),
        galaxy_index: 0,
        iterations: 6,
        settle_steps: 20,
        seed: 42,
        output: PathBuf::from("output/equilibrium/major-merger-debug/galaxy-0/manifest.json"),
    };

    let mut it = env::args().skip(1);
    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--preset" => args.preset_id = next_value(&mut it, "--preset")?,
            "--galaxy" => args.galaxy_index = next_value(&mut it, "--galaxy")?.parse()?,
            "--iterations" => args.iterations = next_value(&mut it, "--iterations")?.parse()?,
            "--settle-steps" => {
                args.settle_steps = next_value(&mut it, "--settle-steps")?.parse()?
            }
            "--seed" => args.seed = next_value(&mut it, "--seed")?.parse()?,
            "--output" => args.output = PathBuf::from(next_value(&mut it, "--output")?),
            "--help" | "-h" => {
                println!(
                    "usage: cargo run -p sim-cuda --bin equilibrate -- [--preset ID] [--galaxy INDEX] [--iterations N] [--settle-steps N] [--seed N] [--output PATH]"
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

fn build_isolated_config(config: &SimulationConfig, galaxy_index: usize) -> Result<SimulationConfig> {
    let galaxy = config
        .galaxies
        .get(galaxy_index)
        .with_context(|| format!("galaxy index {galaxy_index} out of range"))?
        .clone();
    let mut isolated_config = config.clone();
    isolated_config.name = format!("{}-isolated-{galaxy_index}", config.name);
    isolated_config.output_directory = format!("{}/isolated-{galaxy_index}", config.output_directory);
    isolated_config.initial_separation_kpc = 0.0;
    isolated_config.initial_relative_velocity_kms = 0.0;
    isolated_config.preview.particle_budget = isolated_config.preview.particle_budget.min(64);
    isolated_config.galaxies = vec![{
        let mut galaxy = galaxy;
        galaxy.position_kpc = [0.0, 0.0, 0.0];
        galaxy.velocity_kms = [0.0, 0.0, 0.0];
        galaxy.equilibrium_snapshot = None;
        galaxy
    }];
    Ok(isolated_config)
}

fn remap_velocity_distribution(
    evolved_particles: &[Particle],
    mut target_particles: Vec<Particle>,
) -> Result<Vec<Particle>> {
    let frame = disk_frame(evolved_particles)?;
    let halo_samples = spherical_samples(evolved_particles, ParticleComponent::Halo, frame);
    let bulge_samples = spherical_samples(evolved_particles, ParticleComponent::Bulge, frame);
    let disk_samples = disk_samples(evolved_particles, frame)?;

    remap_spherical_component(
        &halo_samples,
        &mut target_particles,
        ParticleComponent::Halo,
        frame.bulk_velocity,
    );
    remap_disk_component(&disk_samples, &mut target_particles, frame);
    remap_spherical_component(
        &bulge_samples,
        &mut target_particles,
        ParticleComponent::Bulge,
        frame.bulk_velocity,
    );

    if let Some(smbh) = target_particles
        .iter_mut()
        .find(|particle| matches!(particle.component, ParticleComponent::Smbh))
    {
        smbh.position_kpc = Vec3::ZERO;
        smbh.velocity_kms = Vec3::ZERO;
    }

    recenter_local_particles(&mut target_particles);
    Ok(target_particles)
}

fn spherical_samples(
    particles: &[Particle],
    component: ParticleComponent,
    frame: Frame3,
) -> Vec<SphericalVelocitySample> {
    let mut samples = particles
        .iter()
        .filter(|particle| particle.component == component)
        .map(|particle| {
            let position = particle.position_kpc - frame.center;
            let velocity = particle.velocity_kms - frame.bulk_velocity;
            SphericalVelocitySample {
                radius: position.length(),
                position,
                velocity,
            }
        })
        .collect::<Vec<_>>();
    samples.sort_by(|lhs, rhs| lhs.radius.partial_cmp(&rhs.radius).unwrap_or(Ordering::Equal));
    samples
}

fn disk_samples(particles: &[Particle], frame: Frame3) -> Result<Vec<DiskVelocitySample>> {
    let mut samples = Vec::new();
    for particle in particles
        .iter()
        .filter(|particle| matches!(particle.component, ParticleComponent::Disk))
    {
        let relative_position = particle.position_kpc - frame.center;
        let relative_velocity = particle.velocity_kms - frame.bulk_velocity;
        let height = relative_position.dot(frame.normal);
        let in_plane = relative_position - frame.normal * height;
        let radius = in_plane.length();
        let radial_dir = if radius > 1.0e-8 {
            in_plane / radius
        } else {
            orthonormal_perpendicular(frame.normal)
        };
        let phi_dir = frame.normal.cross(radial_dir).normalized();
        samples.push(DiskVelocitySample {
            radius,
            abs_height: height.abs(),
            v_r: relative_velocity.dot(radial_dir),
            v_phi: relative_velocity.dot(phi_dir),
            v_z: relative_velocity.dot(frame.normal),
        });
    }
    if samples.is_empty() {
        bail!("disk component has no particles");
    }
    samples.sort_by(compare_disk_samples);
    neutralize_disk_odd_moments(&mut samples);
    Ok(samples)
}

fn neutralize_disk_odd_moments(samples: &mut [DiskVelocitySample]) {
    if samples.is_empty() {
        return;
    }

    let bin_count = samples.len().min(32);
    for bin_index in 0..bin_count {
        let start = bin_index * samples.len() / bin_count;
        let end = ((bin_index + 1) * samples.len() / bin_count).min(samples.len());
        if end <= start {
            continue;
        }
        let span = &samples[start..end];
        let mean_v_r = span.iter().map(|sample| sample.v_r).sum::<f64>() / span.len() as f64;
        let mean_v_z = span.iter().map(|sample| sample.v_z).sum::<f64>() / span.len() as f64;
        for sample in &mut samples[start..end] {
            sample.v_r -= mean_v_r;
            sample.v_z -= mean_v_z;
        }
    }
}

fn remap_disk_component(samples: &[DiskVelocitySample], particles: &mut [Particle], frame: Frame3) {
    let mut targets = particles
        .iter_mut()
        .filter(|particle| matches!(particle.component, ParticleComponent::Disk))
        .map(|particle| {
            let relative_position = particle.position_kpc - frame.center;
            let height = relative_position.dot(frame.normal);
            let in_plane = relative_position - frame.normal * height;
            let radius = in_plane.length();
            (radius, height.abs(), particle)
        })
        .collect::<Vec<_>>();
    targets.sort_by(|lhs, rhs| {
        lhs.0
            .partial_cmp(&rhs.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| lhs.1.partial_cmp(&rhs.1).unwrap_or(Ordering::Equal))
    });

    for (sample, (_, _, particle)) in samples.iter().zip(targets.into_iter()) {
        let relative_position = particle.position_kpc - frame.center;
        let height = relative_position.dot(frame.normal);
        let in_plane = relative_position - frame.normal * height;
        let radius = in_plane.length();
        let radial_dir = if radius > 1.0e-8 {
            in_plane / radius
        } else {
            orthonormal_perpendicular(frame.normal)
        };
        let phi_dir = frame.normal.cross(radial_dir).normalized();
        particle.velocity_kms = frame.bulk_velocity
            + radial_dir * sample.v_r
            + phi_dir * sample.v_phi
            + frame.normal * sample.v_z;
    }
}

fn remap_spherical_component(
    samples: &[SphericalVelocitySample],
    particles: &mut [Particle],
    component: ParticleComponent,
    bulk_velocity: Vec3,
) {
    let mut targets = particles
        .iter_mut()
        .filter(|particle| particle.component == component)
        .map(|particle| ((particle.position_kpc.length()), particle))
        .collect::<Vec<_>>();
    targets.sort_by(|lhs, rhs| lhs.0.partial_cmp(&rhs.0).unwrap_or(Ordering::Equal));

    for (sample, (_, particle)) in samples.iter().zip(targets.into_iter()) {
        let target_position = particle.position_kpc;
        let rotated_velocity = rotate_vector_between_directions(sample.velocity, sample.position, target_position);
        particle.velocity_kms = bulk_velocity + rotated_velocity;
    }
}

fn disk_frame(particles: &[Particle]) -> Result<Frame3> {
    let mut center = Vec3::ZERO;
    let mut bulk_velocity = Vec3::ZERO;
    if let Some(smbh) = particles
        .iter()
        .find(|particle| matches!(particle.component, ParticleComponent::Smbh))
    {
        center = smbh.position_kpc;
        bulk_velocity = smbh.velocity_kms;
    }

    let mut angular_momentum = Vec3::ZERO;
    for particle in particles
        .iter()
        .filter(|particle| matches!(particle.component, ParticleComponent::Disk))
    {
        let r = particle.position_kpc - center;
        let v = particle.velocity_kms - bulk_velocity;
        angular_momentum += Vec3::new(
            r.y * v.z - r.z * v.y,
            r.z * v.x - r.x * v.z,
            r.x * v.y - r.y * v.x,
        ) * particle.mass_msun;
    }
    let normal = angular_momentum.normalized();
    if normal.length() <= 1.0e-8 {
        bail!("disk angular momentum vanished during equilibration");
    }
    Ok(Frame3 {
        center,
        bulk_velocity,
        normal,
    })
}

fn recenter_local_particles(particles: &mut [Particle]) {
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
    if total_mass <= 0.0 {
        return;
    }

    center_of_mass = center_of_mass / total_mass;
    center_velocity = center_velocity / total_mass;
    for particle in particles.iter_mut() {
        particle.position_kpc -= center_of_mass;
        particle.velocity_kms -= center_velocity;
    }

    if let Some(smbh) = particles
        .iter_mut()
        .find(|particle| matches!(particle.component, ParticleComponent::Smbh))
    {
        smbh.position_kpc = Vec3::ZERO;
        smbh.velocity_kms = Vec3::ZERO;
    }
}

fn rotate_vector_between_directions(vector: Vec3, from: Vec3, to: Vec3) -> Vec3 {
    let from_n = from.normalized();
    let to_n = to.normalized();
    if from_n.length() <= 1.0e-8 || to_n.length() <= 1.0e-8 {
        return vector;
    }
    let cross = from_n.cross(to_n);
    let sin_theta = cross.length();
    let cos_theta = from_n.dot(to_n).clamp(-1.0, 1.0);
    if sin_theta <= 1.0e-8 {
        if cos_theta >= 0.0 {
            return vector;
        }
        let axis = orthonormal_perpendicular(from_n);
        return rotate_around_axis(vector, axis, std::f64::consts::PI);
    }
    let axis = cross / sin_theta;
    let angle = sin_theta.atan2(cos_theta);
    rotate_around_axis(vector, axis, angle)
}

fn rotate_around_axis(vector: Vec3, axis: Vec3, angle: f64) -> Vec3 {
    let axis = axis.normalized();
    let (sin_angle, cos_angle) = angle.sin_cos();
    vector * cos_angle
        + axis.cross(vector) * sin_angle
        + axis * axis.dot(vector) * (1.0 - cos_angle)
}

fn orthonormal_perpendicular(vector: Vec3) -> Vec3 {
    let reference = if vector.x.abs() < 0.8 {
        Vec3::new(1.0, 0.0, 0.0)
    } else {
        Vec3::new(0.0, 1.0, 0.0)
    };
    vector.cross(reference).normalized()
}

fn compare_disk_samples(lhs: &DiskVelocitySample, rhs: &DiskVelocitySample) -> Ordering {
    lhs.radius
        .partial_cmp(&rhs.radius)
        .unwrap_or(Ordering::Equal)
        .then_with(|| lhs.abs_height.partial_cmp(&rhs.abs_height).unwrap_or(Ordering::Equal))
}

fn print_metrics(label: &str, particles: &[Particle]) {
    match compute_galaxy_metrics(particles) {
        Some(metrics) => {
            println!(
                "{label}: disk_count={} r50_kpc={:.3} z_rms_kpc={:.3} spin={:.6e} mean_vr_kms={:.3}",
                metrics.disk_count,
                metrics.disk_r50_kpc,
                metrics.disk_rms_height_kpc,
                metrics.disk_spin,
                metrics.disk_mean_radial_velocity_kms
            );
        }
        None => {
            println!("{label}: no disk particles");
        }
    }
}

fn compute_galaxy_metrics(particles: &[Particle]) -> Option<GalaxyMetrics> {
    let frame = disk_frame(particles).ok()?;
    let disk_particles = particles
        .iter()
        .filter(|particle| matches!(particle.component, ParticleComponent::Disk))
        .collect::<Vec<_>>();
    if disk_particles.is_empty() {
        return None;
    }

    let mut cylindrical_radii = Vec::with_capacity(disk_particles.len());
    let mut height_sq_sum = 0.0;
    let mut radial_velocity_sum = 0.0;
    let mut spin = 0.0;
    for particle in disk_particles {
        let relative_position = particle.position_kpc - frame.center;
        let relative_velocity = particle.velocity_kms - frame.bulk_velocity;
        let height = relative_position.dot(frame.normal);
        let in_plane = relative_position - frame.normal * height;
        let radius = in_plane.length();
        let radial_dir = if radius > 1.0e-8 {
            in_plane / radius
        } else {
            orthonormal_perpendicular(frame.normal)
        };
        cylindrical_radii.push(radius);
        height_sq_sum += height * height;
        radial_velocity_sum += relative_velocity.dot(radial_dir);
        spin += frame
            .normal
            .dot(relative_position.cross(relative_velocity) * particle.mass_msun);
    }
    cylindrical_radii.sort_by(|lhs, rhs| lhs.partial_cmp(rhs).unwrap_or(Ordering::Equal));

    Some(GalaxyMetrics {
        disk_r50_kpc: quantile(&cylindrical_radii, 0.5),
        disk_rms_height_kpc: (height_sq_sum / cylindrical_radii.len() as f64).sqrt(),
        disk_spin: spin,
        disk_mean_radial_velocity_kms: radial_velocity_sum / cylindrical_radii.len() as f64,
        disk_count: cylindrical_radii.len(),
    })
}

fn quantile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * q.clamp(0.0, 1.0)).round() as usize;
    sorted[index]
}
