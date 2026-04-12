use anyhow::bail;
use rand::{Rng, SeedableRng, rngs::SmallRng};
use rand_distr::{Distribution, StandardNormal};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::{SimulationConfig, Vec3};

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
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

            let rotation = rotation_from_euler_deg(galaxy.disk_tilt_deg);

            extend_nfw_halo(
                &mut particles,
                &mut rng,
                galaxy_index as u32,
                galaxy.halo_particle_count,
                galaxy.halo_mass_msun,
                galaxy.halo_scale_radius_kpc,
                config.gravity.halo_softening_kpc,
                galaxy.color_rgba,
                origin,
                bulk_velocity,
            )?;

            extend_exponential_disk(
                &mut particles,
                &mut rng,
                galaxy_index as u32,
                galaxy.disk_particle_count,
                galaxy.disk_mass_msun,
                galaxy.disk_scale_radius_kpc,
                galaxy.disk_scale_height_kpc,
                config.gravity.disk_softening_kpc,
                galaxy.color_rgba,
                origin,
                bulk_velocity,
                rotation,
                galaxy.halo_mass_msun + galaxy.disk_mass_msun + galaxy.bulge_mass_msun,
            )?;

            extend_hernquist_bulge(
                &mut particles,
                &mut rng,
                galaxy_index as u32,
                galaxy.bulge_particle_count,
                galaxy.bulge_mass_msun,
                galaxy.bulge_scale_radius_kpc,
                config.gravity.bulge_softening_kpc,
                galaxy.color_rgba,
                origin,
                bulk_velocity,
            )?;

            particles.push(Particle {
                galaxy_index: galaxy_index as u32,
                component: ParticleComponent::Smbh,
                position_kpc: origin,
                velocity_kms: bulk_velocity,
                mass_msun: galaxy.smbh.mass_msun,
                softening_kpc: galaxy.smbh.softening_kpc,
                color_rgba: [1.0, 1.0, 1.0, 1.0],
            });

            total_mass_msun += galaxy.halo_mass_msun
                + galaxy.disk_mass_msun
                + galaxy.bulge_mass_msun
                + galaxy.smbh.mass_msun;
        }

        Ok(Self {
            seed,
            particles,
            total_mass_msun,
        })
    }
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
        let enclosed_mass = total_mass_msun * (r / (r + scale_radius_kpc)).powi(2);
        let dispersion =
            ((4.300_91e-6 * enclosed_mass / (3.0 * r.max(softening_kpc))).max(0.0)).sqrt();
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
    enclosed_mass_reference: f64,
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
        let u: f64 = rng.random::<f64>().clamp(1.0e-8, 1.0 - 1.0e-8);
        let radius = -scale_radius_kpc * (1.0 - u).ln();
        let phi = rng.random::<f64>() * std::f64::consts::TAU;
        let z = 0.5 * scale_height_kpc * ((rng.random::<f64>() - 0.5) * 1.8).tanh();
        let local = Vec3::new(radius * phi.cos(), radius * phi.sin(), z);
        let rotated = rotate(rotation, local);
        let position = origin + rotated;

        let enclosed_mass = enclosed_mass_reference * radius / (radius + scale_radius_kpc);
        let v_c =
            ((4.300_91e-6 * enclosed_mass / radius.max(scale_radius_kpc * 0.25)).max(0.0)).sqrt();
        let tangent = rotate(rotation, Vec3::new(-phi.sin(), phi.cos(), 0.0));
        let dispersion = 0.08 * v_c;
        let velocity = bulk_velocity + tangent * v_c + sample_gaussian3(rng, dispersion);

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

    for _ in 0..count {
        let u = rng.random::<f64>().clamp(1.0e-8, 1.0 - 1.0e-8);
        let root_u = u.sqrt();
        let radius = scale_radius_kpc * root_u / (1.0 - root_u);
        let direction = sample_unit_vector(rng);
        let position = origin + direction * radius;
        let dispersion = ((4.300_91e-6 * total_mass_msun / (6.0 * radius.max(scale_radius_kpc)))
            .max(0.0))
        .sqrt();
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

fn sample_gaussian3(rng: &mut SmallRng, sigma: f64) -> Vec3 {
    if sigma <= f64::EPSILON {
        return Vec3::ZERO;
    }
    let nx: f64 = StandardNormal.sample(rng);
    let ny: f64 = StandardNormal.sample(rng);
    let nz: f64 = StandardNormal.sample(rng);
    Vec3::new(nx * sigma, ny * sigma, nz * sigma)
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
