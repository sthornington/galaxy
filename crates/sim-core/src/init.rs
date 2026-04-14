use anyhow::bail;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{
    GalaxyInitialProfile, GravityConfig, SimulationConfig, Vec3, load_particle_snapshot,
};

const GRAV_CONST_KPC_KMS2_PER_MSUN: f64 = 4.300_91e-6;
const NFW_CONCENTRATION: f64 = 12.0;
const TOOMRE_Q_TARGET: f64 = 1.2;
const EQUILIBRIUM_SAMPLES: usize = 320;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParticleComponent {
    Halo,
    Disk,
    Bulge,
    Smbh,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Particle {
    pub galaxy_index: u32,
    pub component: ParticleComponent,
    pub position_kpc: Vec3,
    pub velocity_kms: Vec3,
    pub mass_msun: f64,
    pub softening_kpc: f64,
    pub color_rgba: [f32; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InitialConditions {
    pub seed: u64,
    pub particles: Vec<Particle>,
    pub total_mass_msun: f64,
}

#[derive(Debug, Error)]
pub enum InitialConditionError {
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    #[error("snapshot load failed: {0}")]
    SnapshotLoad(String),
}

#[derive(Debug, Clone, Copy)]
struct GalaxyPotential {
    halo_mass_msun: f64,
    halo_scale_radius_kpc: f64,
    disk_mass_msun: f64,
    disk_scale_radius_kpc: f64,
    disk_scale_height_kpc: f64,
    bulge_mass_msun: f64,
    bulge_scale_radius_kpc: f64,
    smbh_mass_msun: f64,
}

#[derive(Debug, Clone, Copy)]
struct PotentialSoftening {
    halo_kpc: f64,
    disk_kpc: f64,
    bulge_kpc: f64,
    smbh_kpc: f64,
}

struct EquilibriumTable {
    radii_kpc: Vec<f64>,
    halo_sigma_sq: Vec<f64>,
    bulge_sigma_sq: Vec<f64>,
    disk_sigma_r_sq: Vec<f64>,
    disk_sigma_phi_sq: Vec<f64>,
    disk_sigma_z_sq: Vec<f64>,
    disk_streaming_speed_sq: Vec<f64>,
}

impl InitialConditions {
    pub fn generate(config: &SimulationConfig, seed: u64) -> Result<Self, InitialConditionError> {
        if config.galaxies.is_empty() {
            return Err(InitialConditionError::InvalidConfig(
                "at least one galaxy is required".to_string(),
            ));
        }

        let mut rng = SmallRng::seed_from_u64(seed);
        let mut particles = Vec::new();
        let mut total_mass_msun = 0.0;

        for (galaxy_index, galaxy) in config.galaxies.iter().enumerate() {
            let galaxy_particles = if let Some(snapshot_path) = &galaxy.equilibrium_snapshot {
                load_galaxy_from_snapshot(config, galaxy_index as u32, snapshot_path)?
            } else {
                generate_analytic_galaxy(config, galaxy_index as u32, galaxy, &mut rng)?
            };
            total_mass_msun += galaxy_particles.iter().map(|particle| particle.mass_msun).sum::<f64>();
            particles.extend(galaxy_particles);
        }

        Ok(Self {
            seed,
            particles,
            total_mass_msun,
        })
    }
}

pub fn generate_analytic_galaxy(
    config: &SimulationConfig,
    galaxy_index: u32,
    galaxy: &crate::GalaxyConfig,
    rng: &mut SmallRng,
) -> Result<Vec<Particle>, InitialConditionError> {
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

    if let GalaxyInitialProfile::UniformSphere {
        radius_kpc,
        velocity_dispersion_kms,
        edge_rotation_speed_kms,
    } = galaxy.initial_profile
    {
        return generate_uniform_sphere_galaxy(
            config,
            galaxy_index,
            galaxy,
            rng,
            origin,
            bulk_velocity,
            radius_kpc,
            velocity_dispersion_kms,
            edge_rotation_speed_kms,
        );
    }

    let rotation = rotation_from_euler_deg(galaxy.disk_tilt_deg);
    let potential = GalaxyPotential {
        halo_mass_msun: galaxy.halo_mass_msun,
        halo_scale_radius_kpc: galaxy.halo_scale_radius_kpc,
        disk_mass_msun: galaxy.disk_mass_msun,
        disk_scale_radius_kpc: galaxy.disk_scale_radius_kpc,
        disk_scale_height_kpc: galaxy.disk_scale_height_kpc,
        bulge_mass_msun: galaxy.bulge_mass_msun,
        bulge_scale_radius_kpc: galaxy.bulge_scale_radius_kpc,
        smbh_mass_msun: galaxy.smbh.mass_msun,
    };
    let equilibrium = build_equilibrium_table(galaxy, &config.gravity, potential);
    let mut particles = Vec::new();

    extend_nfw_halo(
        &mut particles,
        rng,
        galaxy_index,
        galaxy.halo_particle_count,
        galaxy.halo_mass_msun,
        galaxy.halo_scale_radius_kpc,
        config.gravity.halo_softening_kpc,
        galaxy.color_rgba,
        origin,
        bulk_velocity,
        &equilibrium,
    )?;

    extend_exponential_disk(
        &mut particles,
        rng,
        galaxy_index,
        galaxy.disk_particle_count,
        galaxy.disk_mass_msun,
        galaxy.disk_scale_radius_kpc,
        galaxy.disk_scale_height_kpc,
        config.gravity.disk_softening_kpc,
        galaxy.color_rgba,
        origin,
        bulk_velocity,
        rotation,
        &equilibrium,
    )?;

    extend_hernquist_bulge(
        &mut particles,
        rng,
        galaxy_index,
        galaxy.bulge_particle_count,
        galaxy.bulge_mass_msun,
        galaxy.bulge_scale_radius_kpc,
        config.gravity.bulge_softening_kpc,
        galaxy.color_rgba,
        origin,
        bulk_velocity,
        &equilibrium,
    )?;

    particles.push(Particle {
        galaxy_index,
        component: ParticleComponent::Smbh,
        position_kpc: origin,
        velocity_kms: bulk_velocity,
        mass_msun: galaxy.smbh.mass_msun,
        softening_kpc: galaxy.smbh.softening_kpc,
        color_rgba: [1.0, 1.0, 1.0, 1.0],
    });

    recenter_galaxy(&mut particles, origin, bulk_velocity, galaxy_index);
    Ok(particles)
}

fn generate_uniform_sphere_galaxy(
    config: &SimulationConfig,
    galaxy_index: u32,
    galaxy: &crate::GalaxyConfig,
    rng: &mut SmallRng,
    origin: Vec3,
    bulk_velocity: Vec3,
    radius_kpc: f64,
    velocity_dispersion_kms: f64,
    edge_rotation_speed_kms: f64,
) -> Result<Vec<Particle>, InitialConditionError> {
    if !(radius_kpc > 0.0) {
        return Err(InitialConditionError::InvalidConfig(
            "uniform sphere radius must be positive".to_string(),
        ));
    }
    if edge_rotation_speed_kms < 0.0 {
        return Err(InitialConditionError::InvalidConfig(
            "uniform sphere edge rotation speed must be non-negative".to_string(),
        ));
    }

    let mut particles = Vec::new();
    extend_uniform_sphere_component(
        &mut particles,
        rng,
        galaxy_index,
        ParticleComponent::Halo,
        galaxy.halo_particle_count,
        galaxy.halo_mass_msun,
        radius_kpc,
        config.gravity.halo_softening_kpc,
        galaxy.color_rgba,
        origin,
        bulk_velocity,
        velocity_dispersion_kms,
        edge_rotation_speed_kms,
    )?;
    extend_uniform_sphere_component(
        &mut particles,
        rng,
        galaxy_index,
        ParticleComponent::Disk,
        galaxy.disk_particle_count,
        galaxy.disk_mass_msun,
        radius_kpc,
        config.gravity.disk_softening_kpc,
        galaxy.color_rgba,
        origin,
        bulk_velocity,
        velocity_dispersion_kms,
        edge_rotation_speed_kms,
    )?;
    extend_uniform_sphere_component(
        &mut particles,
        rng,
        galaxy_index,
        ParticleComponent::Bulge,
        galaxy.bulge_particle_count,
        galaxy.bulge_mass_msun,
        radius_kpc,
        config.gravity.bulge_softening_kpc,
        galaxy.color_rgba,
        origin,
        bulk_velocity,
        velocity_dispersion_kms,
        edge_rotation_speed_kms,
    )?;

    particles.push(Particle {
        galaxy_index,
        component: ParticleComponent::Smbh,
        position_kpc: origin,
        velocity_kms: bulk_velocity,
        mass_msun: galaxy.smbh.mass_msun,
        softening_kpc: galaxy.smbh.softening_kpc,
        color_rgba: [1.0, 1.0, 1.0, 1.0],
    });

    recenter_galaxy(&mut particles, origin, bulk_velocity, galaxy_index);
    Ok(particles)
}

fn load_galaxy_from_snapshot(
    config: &SimulationConfig,
    galaxy_index: u32,
    snapshot_path: &str,
) -> Result<Vec<Particle>, InitialConditionError> {
    let galaxy = &config.galaxies[galaxy_index as usize];
    let (_, mut particles) = load_particle_snapshot(snapshot_path)
        .map_err(|error| InitialConditionError::SnapshotLoad(error.to_string()))?;
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

    for particle in &mut particles {
        particle.galaxy_index = galaxy_index;
        particle.position_kpc += origin;
        particle.velocity_kms += bulk_velocity;
        particle.color_rgba = match particle.component {
            ParticleComponent::Smbh => [1.0, 1.0, 1.0, 1.0],
            _ => galaxy.color_rgba,
        };
    }

    recenter_galaxy(&mut particles, origin, bulk_velocity, galaxy_index);
    Ok(particles)
}

fn extend_nfw_halo(
    particles: &mut Vec<Particle>,
    rng: &mut SmallRng,
    galaxy_index: u32,
    count: u32,
    total_mass_msun: f64,
    scale_radius_kpc: f64,
    softening_kpc: f64,
    color_rgba: [f32; 4],
    origin: Vec3,
    bulk_velocity: Vec3,
    equilibrium: &EquilibriumTable,
) -> Result<(), InitialConditionError> {
    if count == 0 {
        return Ok(());
    }
    if !(total_mass_msun > 0.0 && scale_radius_kpc > 0.0) {
        return Err(InitialConditionError::InvalidConfig(
            "halo mass and scale radius must be positive".to_string(),
        ));
    }
    let particle_mass = total_mass_msun / count as f64;
    let virial_radius = scale_radius_kpc * 12.0;

    for _ in 0..count {
        let r = sample_nfw_radius(rng, scale_radius_kpc, virial_radius);
        let direction = sample_unit_vector(rng);
        let position = origin + direction * r;
        let dispersion = interpolate_log_grid(&equilibrium.radii_kpc, &equilibrium.halo_sigma_sq, r)
            .max(0.0)
            .sqrt()
            .max(4.0);
        let velocity = bulk_velocity + sample_gaussian3(rng, dispersion);

        particles.push(Particle {
            galaxy_index,
            component: ParticleComponent::Halo,
            position_kpc: position,
            velocity_kms: velocity,
            mass_msun: particle_mass,
            softening_kpc,
            color_rgba,
        });
    }

    Ok(())
}

fn extend_uniform_sphere_component(
    particles: &mut Vec<Particle>,
    rng: &mut SmallRng,
    galaxy_index: u32,
    component: ParticleComponent,
    count: u32,
    total_mass_msun: f64,
    radius_kpc: f64,
    softening_kpc: f64,
    color_rgba: [f32; 4],
    origin: Vec3,
    bulk_velocity: Vec3,
    velocity_dispersion_kms: f64,
    edge_rotation_speed_kms: f64,
) -> Result<(), InitialConditionError> {
    if count == 0 {
        return Ok(());
    }
    if !(total_mass_msun > 0.0) {
        return Err(InitialConditionError::InvalidConfig(format!(
            "uniform sphere component {:?} needs positive mass when particle_count > 0",
            component
        )));
    }

    let particle_mass = total_mass_msun / count as f64;
    let angular_speed_kms_per_kpc = edge_rotation_speed_kms / radius_kpc.max(1.0e-6);
    for _ in 0..count {
        let radius = radius_kpc * rng.random::<f64>().cbrt();
        let direction = sample_unit_vector(rng);
        let local_position = direction * radius;
        let cylindrical_radius = (local_position.x * local_position.x + local_position.y * local_position.y)
            .sqrt();
        let rotational_velocity = if cylindrical_radius > 1.0e-6 {
            let tangential_speed = angular_speed_kms_per_kpc * cylindrical_radius;
            Vec3::new(
                -local_position.y / cylindrical_radius * tangential_speed,
                local_position.x / cylindrical_radius * tangential_speed,
                0.0,
            )
        } else {
            Vec3::ZERO
        };
        particles.push(Particle {
            galaxy_index,
            component,
            position_kpc: origin + local_position,
            velocity_kms: bulk_velocity
                + rotational_velocity
                + sample_gaussian3(rng, velocity_dispersion_kms),
            mass_msun: particle_mass,
            softening_kpc,
            color_rgba,
        });
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn extend_exponential_disk(
    particles: &mut Vec<Particle>,
    rng: &mut SmallRng,
    galaxy_index: u32,
    count: u32,
    total_mass_msun: f64,
    scale_radius_kpc: f64,
    scale_height_kpc: f64,
    softening_kpc: f64,
    color_rgba: [f32; 4],
    origin: Vec3,
    bulk_velocity: Vec3,
    rotation: [[f64; 3]; 3],
    equilibrium: &EquilibriumTable,
) -> Result<(), InitialConditionError> {
    if count == 0 {
        return Ok(());
    }
    if !(total_mass_msun > 0.0 && scale_radius_kpc > 0.0 && scale_height_kpc > 0.0) {
        return Err(InitialConditionError::InvalidConfig(
            "disk mass and scale lengths must be positive".to_string(),
        ));
    }
    let particle_mass = total_mass_msun / count as f64;

    for _ in 0..count {
        // For a 2D exponential disk, p(R) dR ∝ R exp(-R/Rd) dR, i.e. a Gamma(k=2, theta=Rd).
        // Sampling a plain exponential overconcentrates the disk and destabilizes the live system.
        let u1: f64 = rng.random::<f64>().clamp(1.0e-8, 1.0 - 1.0e-8);
        let u2: f64 = rng.random::<f64>().clamp(1.0e-8, 1.0 - 1.0e-8);
        let radius = -scale_radius_kpc * (u1 * u2).ln();
        let phi = rng.random::<f64>() * std::f64::consts::TAU;
        let z = sample_sech2_height(rng, scale_height_kpc);
        let local = Vec3::new(radius * phi.cos(), radius * phi.sin(), z);
        let rotated = rotate(rotation, local);
        let position = origin + rotated;
        let sigma_r = interpolate_log_grid(&equilibrium.radii_kpc, &equilibrium.disk_sigma_r_sq, radius)
            .max(0.0)
            .sqrt();
        let sigma_phi = interpolate_log_grid(
            &equilibrium.radii_kpc,
            &equilibrium.disk_sigma_phi_sq,
            radius,
        )
        .max(0.0)
        .sqrt();
        let sigma_z = interpolate_log_grid(&equilibrium.radii_kpc, &equilibrium.disk_sigma_z_sq, radius)
            .max(0.0)
            .sqrt();
        let streaming_speed = interpolate_log_grid(
            &equilibrium.radii_kpc,
            &equilibrium.disk_streaming_speed_sq,
            radius,
        )
        .max(0.0)
        .sqrt();
        let radial_direction = rotate(rotation, Vec3::new(phi.cos(), phi.sin(), 0.0));
        let tangent_direction = rotate(rotation, Vec3::new(-phi.sin(), phi.cos(), 0.0));
        let normal_direction = rotate(rotation, Vec3::new(0.0, 0.0, 1.0));
        let v_phi = streaming_speed + sample_gaussian(rng, sigma_phi);
        let velocity = bulk_velocity
            + radial_direction * sample_gaussian(rng, sigma_r)
            + tangent_direction * v_phi
            + normal_direction * sample_gaussian(rng, sigma_z);

        particles.push(Particle {
            galaxy_index,
            component: ParticleComponent::Disk,
            position_kpc: position,
            velocity_kms: velocity,
            mass_msun: particle_mass,
            softening_kpc,
            color_rgba,
        });
    }

    Ok(())
}

fn extend_hernquist_bulge(
    particles: &mut Vec<Particle>,
    rng: &mut SmallRng,
    galaxy_index: u32,
    count: u32,
    total_mass_msun: f64,
    scale_radius_kpc: f64,
    softening_kpc: f64,
    color_rgba: [f32; 4],
    origin: Vec3,
    bulk_velocity: Vec3,
    equilibrium: &EquilibriumTable,
) -> Result<(), InitialConditionError> {
    if count == 0 {
        return Ok(());
    }
    if !(total_mass_msun > 0.0 && scale_radius_kpc > 0.0) {
        return Err(InitialConditionError::InvalidConfig(
            "bulge mass and scale radius must be positive".to_string(),
        ));
    }
    let particle_mass = total_mass_msun / count as f64;
    let max_radius = scale_radius_kpc * 40.0;
    let max_cumulative = (max_radius / (max_radius + scale_radius_kpc)).powi(2);

    for _ in 0..count {
        let u = (rng.random::<f64>() * max_cumulative).clamp(1.0e-8, max_cumulative);
        let root_u = u.sqrt();
        let radius = scale_radius_kpc * root_u / (1.0 - root_u);
        let direction = sample_unit_vector(rng);
        let position = origin + direction * radius;
        let dispersion =
            interpolate_log_grid(&equilibrium.radii_kpc, &equilibrium.bulge_sigma_sq, radius)
                .max(0.0)
                .sqrt()
                .max(6.0);
        let velocity = bulk_velocity + sample_gaussian3(rng, dispersion);

        particles.push(Particle {
            galaxy_index,
            component: ParticleComponent::Bulge,
            position_kpc: position,
            velocity_kms: velocity,
            mass_msun: particle_mass,
            softening_kpc,
            color_rgba,
        });
    }

    Ok(())
}

fn sample_nfw_radius(rng: &mut SmallRng, scale_radius_kpc: f64, virial_radius_kpc: f64) -> f64 {
    let max_x = virial_radius_kpc / scale_radius_kpc;
    let max_pdf = max_x / (1.0 + max_x).powi(2);
    loop {
        let x = rng.random::<f64>() * max_x;
        let y = rng.random::<f64>() * max_pdf;
        let pdf = x / (1.0 + x).powi(2);
        if y <= pdf {
            return x * scale_radius_kpc;
        }
    }
}

fn sample_unit_vector(rng: &mut SmallRng) -> Vec3 {
    let z = 2.0 * rng.random::<f64>() - 1.0;
    let phi = rng.random::<f64>() * std::f64::consts::TAU;
    let r = (1.0 - z * z).sqrt();
    Vec3::new(r * phi.cos(), r * phi.sin(), z)
}

fn sample_gaussian(rng: &mut SmallRng, sigma: f64) -> f64 {
    if sigma <= f64::EPSILON {
        return 0.0;
    }
    let n: f64 = StandardNormal.sample(rng);
    n * sigma
}

fn sample_gaussian3(rng: &mut SmallRng, sigma: f64) -> Vec3 {
    if sigma <= f64::EPSILON {
        return Vec3::ZERO;
    }
    let nx: f64 = StandardNormal.sample(rng);
    let ny: f64 = StandardNormal.sample(rng);
    let nz: f64 = StandardNormal.sample(rng);
    Vec3::new(nx * sigma, ny * sigma, nz * sigma)
}

fn sample_sech2_height(rng: &mut SmallRng, scale_height_kpc: f64) -> f64 {
    let u = rng.random::<f64>().clamp(1.0e-6, 1.0 - 1.0e-6);
    0.5 * scale_height_kpc * (u / (1.0 - u)).ln()
}

fn build_equilibrium_table(
    galaxy: &crate::GalaxyConfig,
    gravity: &GravityConfig,
    potential: GalaxyPotential,
) -> EquilibriumTable {
    let softening = PotentialSoftening {
        halo_kpc: gravity.halo_softening_kpc.max(1.0e-3),
        disk_kpc: gravity.disk_softening_kpc.max(1.0e-3),
        bulge_kpc: gravity.bulge_softening_kpc.max(1.0e-3),
        smbh_kpc: galaxy.smbh.softening_kpc.max(1.0e-4),
    };
    let max_radius = nfw_virial_radius_kpc(galaxy.halo_scale_radius_kpc)
        .max(galaxy.disk_scale_radius_kpc * 20.0)
        .max(galaxy.bulge_scale_radius_kpc * 40.0)
        .max(galaxy.disk_scale_height_kpc * 50.0)
        .max(8.0);
    let min_radius = gravity
        .bulge_softening_kpc
        .min(gravity.disk_softening_kpc)
        .min(gravity.halo_softening_kpc)
        .min(galaxy.smbh.softening_kpc)
        .max(1.0e-3);

    let mut radii_kpc = Vec::with_capacity(EQUILIBRIUM_SAMPLES);
    for i in 0..EQUILIBRIUM_SAMPLES {
        let t = i as f64 / (EQUILIBRIUM_SAMPLES.saturating_sub(1)) as f64;
        radii_kpc.push(min_radius * (max_radius / min_radius).powf(t));
    }

    let mut halo_density = Vec::with_capacity(EQUILIBRIUM_SAMPLES);
    let mut bulge_density = Vec::with_capacity(EQUILIBRIUM_SAMPLES);
    let mut enclosed_mass = Vec::with_capacity(EQUILIBRIUM_SAMPLES);
    let mut circular_speed_sq = Vec::with_capacity(EQUILIBRIUM_SAMPLES);
    let mut surface_density = Vec::with_capacity(EQUILIBRIUM_SAMPLES);

    for &radius in &radii_kpc {
        halo_density.push(nfw_density_msun_per_kpc3(
            potential.halo_mass_msun,
            potential.halo_scale_radius_kpc,
            radius,
        ));
        bulge_density.push(hernquist_density_msun_per_kpc3(
            potential.bulge_mass_msun,
            potential.bulge_scale_radius_kpc,
            radius,
        ));
        enclosed_mass.push(
            nfw_enclosed_mass_msun(
                potential.halo_mass_msun,
                potential.halo_scale_radius_kpc,
                effective_softened_radius(radius, softening.halo_kpc),
            ) + hernquist_enclosed_mass_msun(
                potential.bulge_mass_msun,
                potential.bulge_scale_radius_kpc,
                effective_softened_radius(radius, softening.bulge_kpc),
            ) + exponential_disk_enclosed_mass_msun(
                potential.disk_mass_msun,
                potential.disk_scale_radius_kpc,
                effective_softened_radius(radius, softening.disk_kpc),
            ) + softened_point_mass_enclosed_mass_msun(
                potential.smbh_mass_msun,
                radius,
                softening.smbh_kpc,
            ),
        );
        circular_speed_sq.push(
            softened_spherical_circular_velocity_sq(
                potential.halo_mass_msun,
                potential.halo_scale_radius_kpc,
                radius,
                softening.halo_kpc,
                nfw_enclosed_mass_msun,
            ) + softened_spherical_circular_velocity_sq(
                potential.bulge_mass_msun,
                potential.bulge_scale_radius_kpc,
                radius,
                softening.bulge_kpc,
                hernquist_enclosed_mass_msun,
            ) + softened_miyamoto_nagai_circular_velocity_sq(
                potential.disk_mass_msun,
                potential.disk_scale_radius_kpc,
                potential.disk_scale_height_kpc,
                radius,
                softening.disk_kpc,
            ) + softened_point_mass_circular_velocity_sq(
                potential.smbh_mass_msun,
                radius,
                softening.smbh_kpc,
            ),
        );
        surface_density.push(exponential_disk_surface_density_msun_per_kpc2(
            potential.disk_mass_msun,
            potential.disk_scale_radius_kpc,
            radius,
        ));
    }

    let halo_sigma_sq = isotropic_jeans_sigma_sq(
        &radii_kpc,
        &halo_density,
        &enclosed_mass,
        gravity.halo_softening_kpc,
    );
    let bulge_sigma_sq = isotropic_jeans_sigma_sq(
        &radii_kpc,
        &bulge_density,
        &enclosed_mass,
        gravity.bulge_softening_kpc,
    );

    let mut omega_sq = vec![0.0; EQUILIBRIUM_SAMPLES];
    let mut epicyclic_kappa_sq = vec![0.0; EQUILIBRIUM_SAMPLES];
    for i in 0..EQUILIBRIUM_SAMPLES {
        let radius = radii_kpc[i].max(1.0e-4);
        omega_sq[i] = circular_speed_sq[i].max(0.0) / (radius * radius);
    }
    for i in 0..EQUILIBRIUM_SAMPLES {
        let derivative = derivative_on_grid(&radii_kpc, &omega_sq, i);
        epicyclic_kappa_sq[i] = (radii_kpc[i] * derivative + 4.0 * omega_sq[i]).max(1.0e-6);
    }

    let mut disk_sigma_r_sq = vec![0.0; EQUILIBRIUM_SAMPLES];
    let mut disk_sigma_phi_sq = vec![0.0; EQUILIBRIUM_SAMPLES];
    let mut disk_sigma_z_sq = vec![0.0; EQUILIBRIUM_SAMPLES];
    let mut disk_streaming_speed_sq = vec![0.0; EQUILIBRIUM_SAMPLES];
    let mut tracer_pressure = vec![0.0; EQUILIBRIUM_SAMPLES];

    for i in 0..EQUILIBRIUM_SAMPLES {
        let radius = radii_kpc[i];
        let omega = omega_sq[i].sqrt();
        let kappa = epicyclic_kappa_sq[i].sqrt();
        let sigma_r = if surface_density[i] > 0.0 {
            (TOOMRE_Q_TARGET * 3.36 * GRAV_CONST_KPC_KMS2_PER_MSUN * surface_density[i]
                / kappa.max(1.0e-4))
                .max(12.0)
                .min(circular_speed_sq[i].max(0.0).sqrt() * 0.6)
        } else {
            0.0
        };
        let sigma_phi =
            (sigma_r * kappa / (2.0 * omega.max(1.0e-4))).max(0.35 * sigma_r).min(0.95 * sigma_r);
        let sigma_z = (std::f64::consts::PI
            * GRAV_CONST_KPC_KMS2_PER_MSUN
            * surface_density[i]
            * potential.disk_scale_height_kpc.max(1.0e-3))
        .sqrt()
        .max(8.0)
        .min(circular_speed_sq[i].max(0.0).sqrt() * 0.45);

        disk_sigma_r_sq[i] = sigma_r * sigma_r;
        disk_sigma_phi_sq[i] = sigma_phi * sigma_phi;
        disk_sigma_z_sq[i] = sigma_z * sigma_z;
        tracer_pressure[i] = (surface_density[i] * disk_sigma_r_sq[i]).max(1.0e-12);

        let _ = radius;
    }

    for i in 0..EQUILIBRIUM_SAMPLES {
        let dln_pressure_dln_r = derivative_ln_on_grid(&radii_kpc, &tracer_pressure, i);
        let streaming_sq = circular_speed_sq[i]
            - disk_sigma_phi_sq[i]
            + disk_sigma_r_sq[i] * (1.0 + dln_pressure_dln_r);
        disk_streaming_speed_sq[i] = streaming_sq.max(0.0);
    }

    EquilibriumTable {
        radii_kpc,
        halo_sigma_sq,
        bulge_sigma_sq,
        disk_sigma_r_sq,
        disk_sigma_phi_sq,
        disk_sigma_z_sq,
        disk_streaming_speed_sq,
    }
}

fn isotropic_jeans_sigma_sq(
    radii_kpc: &[f64],
    density_profile: &[f64],
    enclosed_mass_msun: &[f64],
    softening_kpc: f64,
) -> Vec<f64> {
    let mut sigma_sq = vec![0.0; radii_kpc.len()];
    let mut running_integral = 0.0;

    for i in (0..radii_kpc.len().saturating_sub(1)).rev() {
        let r0 = radii_kpc[i].max(softening_kpc);
        let r1 = radii_kpc[i + 1].max(softening_kpc);
        let integrand0 = density_profile[i] * GRAV_CONST_KPC_KMS2_PER_MSUN * enclosed_mass_msun[i]
            / (r0 * r0);
        let integrand1 = density_profile[i + 1]
            * GRAV_CONST_KPC_KMS2_PER_MSUN
            * enclosed_mass_msun[i + 1]
            / (r1 * r1);
        running_integral += 0.5 * (r1 - r0) * (integrand0 + integrand1);
        sigma_sq[i] = if density_profile[i] > 1.0e-18 {
            (running_integral / density_profile[i]).max(0.0)
        } else {
            0.0
        };
    }

    if sigma_sq.len() > 1 {
        let last = sigma_sq.len() - 1;
        sigma_sq[last] = sigma_sq[last - 1];
    }

    sigma_sq
}

fn derivative_on_grid(radii_kpc: &[f64], values: &[f64], index: usize) -> f64 {
    if values.len() <= 1 {
        return 0.0;
    }
    if index == 0 {
        return (values[1] - values[0]) / (radii_kpc[1] - radii_kpc[0]);
    }
    if index + 1 == values.len() {
        return (values[index] - values[index - 1]) / (radii_kpc[index] - radii_kpc[index - 1]);
    }
    (values[index + 1] - values[index - 1]) / (radii_kpc[index + 1] - radii_kpc[index - 1])
}

fn derivative_ln_on_grid(radii_kpc: &[f64], values: &[f64], index: usize) -> f64 {
    let radius = radii_kpc[index].max(1.0e-6);
    let value = values[index].max(1.0e-18);
    radius * derivative_on_grid(radii_kpc, values, index) / value
}

fn interpolate_log_grid(radii_kpc: &[f64], values: &[f64], radius_kpc: f64) -> f64 {
    if radii_kpc.is_empty() || values.is_empty() {
        return 0.0;
    }
    let radius = radius_kpc.max(radii_kpc[0]);
    if radius <= radii_kpc[0] {
        return values[0];
    }
    let last = radii_kpc.len() - 1;
    if radius >= radii_kpc[last] {
        return values[last];
    }

    let upper = radii_kpc.partition_point(|sample| *sample < radius);
    let lower = upper.saturating_sub(1);
    let x0 = radii_kpc[lower].ln();
    let x1 = radii_kpc[upper].ln();
    let t = (radius.ln() - x0) / (x1 - x0);
    values[lower] + (values[upper] - values[lower]) * t
}

fn recenter_galaxy(
    particles: &mut [Particle],
    origin: Vec3,
    bulk_velocity: Vec3,
    galaxy_index: u32,
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

fn nfw_virial_radius_kpc(scale_radius_kpc: f64) -> f64 {
    scale_radius_kpc * NFW_CONCENTRATION
}

fn effective_softened_radius(radius_kpc: f64, softening_kpc: f64) -> f64 {
    (radius_kpc * radius_kpc + softening_kpc * softening_kpc).sqrt()
}

fn softened_point_mass_enclosed_mass_msun(
    mass_msun: f64,
    radius_kpc: f64,
    softening_kpc: f64,
) -> f64 {
    if !(mass_msun > 0.0 && radius_kpc >= 0.0 && softening_kpc > 0.0) {
        return mass_msun.max(0.0);
    }
    let r2 = radius_kpc * radius_kpc;
    let eps2 = softening_kpc * softening_kpc;
    mass_msun * r2.powf(1.5) / (r2 + eps2).powf(1.5)
}

fn softened_point_mass_circular_velocity_sq(
    mass_msun: f64,
    radius_kpc: f64,
    softening_kpc: f64,
) -> f64 {
    if !(mass_msun > 0.0 && radius_kpc > 0.0 && softening_kpc > 0.0) {
        return 0.0;
    }
    let r2 = radius_kpc * radius_kpc;
    let eps2 = softening_kpc * softening_kpc;
    GRAV_CONST_KPC_KMS2_PER_MSUN * mass_msun * r2 / (r2 + eps2).powf(1.5)
}

fn softened_spherical_circular_velocity_sq(
    mass_msun: f64,
    scale_radius_kpc: f64,
    radius_kpc: f64,
    softening_kpc: f64,
    enclosed_mass_fn: fn(f64, f64, f64) -> f64,
) -> f64 {
    if !(mass_msun > 0.0 && scale_radius_kpc > 0.0 && radius_kpc > 0.0) {
        return 0.0;
    }
    let effective_radius = effective_softened_radius(radius_kpc, softening_kpc);
    let enclosed_mass = enclosed_mass_fn(mass_msun, scale_radius_kpc, effective_radius);
    GRAV_CONST_KPC_KMS2_PER_MSUN * enclosed_mass * radius_kpc * radius_kpc
        / effective_radius.powi(3)
}

fn softened_miyamoto_nagai_circular_velocity_sq(
    mass_msun: f64,
    scale_radius_kpc: f64,
    scale_height_kpc: f64,
    cylindrical_radius_kpc: f64,
    softening_kpc: f64,
) -> f64 {
    miyamoto_nagai_circular_velocity_sq(
        mass_msun,
        scale_radius_kpc,
        (scale_height_kpc * scale_height_kpc + softening_kpc * softening_kpc).sqrt(),
        cylindrical_radius_kpc,
        0.0,
    )
}

fn nfw_enclosed_mass_msun(total_mass_msun: f64, scale_radius_kpc: f64, radius_kpc: f64) -> f64 {
    if !(total_mass_msun > 0.0 && scale_radius_kpc > 0.0 && radius_kpc > 0.0) {
        return 0.0;
    }
    let concentration = NFW_CONCENTRATION;
    let norm = (1.0 + concentration).ln() - concentration / (1.0 + concentration);
    let x = (radius_kpc / scale_radius_kpc).clamp(0.0, concentration);
    total_mass_msun * (((1.0 + x).ln() - x / (1.0 + x)) / norm)
}

fn nfw_density_msun_per_kpc3(total_mass_msun: f64, scale_radius_kpc: f64, radius_kpc: f64) -> f64 {
    if !(total_mass_msun > 0.0 && scale_radius_kpc > 0.0 && radius_kpc > 0.0) {
        return 0.0;
    }
    let concentration = NFW_CONCENTRATION;
    let norm = (1.0 + concentration).ln() - concentration / (1.0 + concentration);
    let rho0 = total_mass_msun / (4.0 * std::f64::consts::PI * scale_radius_kpc.powi(3) * norm);
    let x = radius_kpc / scale_radius_kpc;
    rho0 / (x * (1.0 + x).powi(2))
}

fn nfw_circular_velocity_sq(mass_msun: f64, scale_radius_kpc: f64, radius_kpc: f64) -> f64 {
    if !(mass_msun > 0.0 && scale_radius_kpc > 0.0 && radius_kpc > 0.0) {
        return 0.0;
    }
    GRAV_CONST_KPC_KMS2_PER_MSUN * nfw_enclosed_mass_msun(mass_msun, scale_radius_kpc, radius_kpc)
        / radius_kpc
}

fn hernquist_enclosed_mass_msun(mass_msun: f64, scale_radius_kpc: f64, radius_kpc: f64) -> f64 {
    if !(mass_msun > 0.0 && scale_radius_kpc > 0.0 && radius_kpc > 0.0) {
        return 0.0;
    }
    mass_msun * (radius_kpc / (radius_kpc + scale_radius_kpc)).powi(2)
}

fn hernquist_density_msun_per_kpc3(
    mass_msun: f64,
    scale_radius_kpc: f64,
    radius_kpc: f64,
) -> f64 {
    if !(mass_msun > 0.0 && scale_radius_kpc > 0.0 && radius_kpc > 0.0) {
        return 0.0;
    }
    mass_msun * scale_radius_kpc
        / (2.0 * std::f64::consts::PI * radius_kpc * (radius_kpc + scale_radius_kpc).powi(3))
}

fn exponential_disk_surface_density_msun_per_kpc2(
    mass_msun: f64,
    scale_radius_kpc: f64,
    radius_kpc: f64,
) -> f64 {
    if !(mass_msun > 0.0 && scale_radius_kpc > 0.0 && radius_kpc >= 0.0) {
        return 0.0;
    }
    mass_msun / (2.0 * std::f64::consts::PI * scale_radius_kpc.powi(2))
        * (-radius_kpc / scale_radius_kpc).exp()
}

fn exponential_disk_enclosed_mass_msun(
    mass_msun: f64,
    scale_radius_kpc: f64,
    radius_kpc: f64,
) -> f64 {
    if !(mass_msun > 0.0 && scale_radius_kpc > 0.0 && radius_kpc >= 0.0) {
        return 0.0;
    }
    let x = radius_kpc / scale_radius_kpc;
    mass_msun * (1.0 - (-x).exp() * (1.0 + x))
}

fn hernquist_circular_velocity_sq(mass_msun: f64, scale_radius_kpc: f64, radius_kpc: f64) -> f64 {
    if !(mass_msun > 0.0 && scale_radius_kpc > 0.0 && radius_kpc > 0.0) {
        return 0.0;
    }

    GRAV_CONST_KPC_KMS2_PER_MSUN
        * hernquist_enclosed_mass_msun(mass_msun, scale_radius_kpc, radius_kpc)
        / radius_kpc
}

fn miyamoto_nagai_circular_velocity_sq(
    mass_msun: f64,
    scale_radius_kpc: f64,
    scale_height_kpc: f64,
    cylindrical_radius_kpc: f64,
    z_kpc: f64,
) -> f64 {
    if !(mass_msun > 0.0
        && scale_radius_kpc > 0.0
        && scale_height_kpc > 0.0
        && cylindrical_radius_kpc > 0.0)
    {
        return 0.0;
    }

    let b = scale_height_kpc.max(1.0e-4);
    let b_term = (z_kpc * z_kpc + b * b).sqrt();
    let sum = scale_radius_kpc.max(1.0e-4) + b_term;
    let denom = (cylindrical_radius_kpc * cylindrical_radius_kpc + sum * sum).powf(1.5);

    GRAV_CONST_KPC_KMS2_PER_MSUN * mass_msun * cylindrical_radius_kpc * cylindrical_radius_kpc
        / denom
}

fn point_mass_circular_velocity_sq(mass_msun: f64, radius_kpc: f64) -> f64 {
    if !(mass_msun > 0.0 && radius_kpc > 0.0) {
        return 0.0;
    }

    GRAV_CONST_KPC_KMS2_PER_MSUN * mass_msun / radius_kpc
}

fn rotation_from_euler_deg(angles_deg: [f64; 3]) -> [[f64; 3]; 3] {
    let [ax, ay, az] = angles_deg.map(f64::to_radians);
    let (sx, cx) = ax.sin_cos();
    let (sy, cy) = ay.sin_cos();
    let (sz, cz) = az.sin_cos();

    [
        [cy * cz, cz * sx * sy - cx * sz, sx * sz + cx * cz * sy],
        [cy * sz, cx * cz + sx * sy * sz, cx * sy * sz - cz * sx],
        [-sy, cy * sx, cx * cy],
    ]
}

fn rotate(matrix: [[f64; 3]; 3], v: Vec3) -> Vec3 {
    Vec3::new(
        matrix[0][0] * v.x + matrix[0][1] * v.y + matrix[0][2] * v.z,
        matrix[1][0] * v.x + matrix[1][1] * v.y + matrix[1][2] * v.z,
        matrix[2][0] * v.x + matrix[2][1] * v.y + matrix[2][2] * v.z,
    )
}

pub fn validate_particle_count(config: &SimulationConfig) -> anyhow::Result<u64> {
    let mut total = 0_u64;
    for galaxy in &config.galaxies {
        total += u64::from(galaxy.halo_particle_count)
            + u64::from(galaxy.disk_particle_count)
            + u64::from(galaxy.bulge_particle_count)
            + 1;
    }
    if total == 0 {
        bail!("configuration produced zero particles");
    }
    Ok(total)
}
