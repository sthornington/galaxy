use std::{ffi::c_void, mem::MaybeUninit};

use anyhow::{Context, anyhow};
use sim_core::{
    Diagnostics, InitialConditions, Particle, ParticleComponent, PreviewFrame, PreviewParticle,
    SimulationConfig, Vec3,
};

#[repr(C)]
#[derive(Clone, Copy)]
struct FfiParticle {
    position_kpc: [f64; 3],
    velocity_kms: [f64; 3],
    mass_msun: f64,
    softening_kpc: f64,
    galaxy_index: u32,
    component: u32,
    color_rgba: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FfiPreviewParticle {
    position_kpc: [f32; 3],
    velocity_kms: [f32; 3],
    mass_msun: f32,
    galaxy_index: u32,
    component: u32,
    color_rgba: [f32; 4],
    intensity: f32,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FfiGalaxy {
    halo_mass_msun: f64,
    halo_scale_radius_kpc: f64,
    disk_mass_msun: f64,
    disk_scale_radius_kpc: f64,
    disk_scale_height_kpc: f64,
    bulge_mass_msun: f64,
    bulge_scale_radius_kpc: f64,
    disk_rotation: [f64; 9],
}

#[repr(C)]
#[derive(Clone, Copy)]
struct FfiDiagnostics {
    particle_count: u64,
    preview_count: u32,
    sim_time_myr: f64,
    dt_myr: f64,
    kinetic_energy: f64,
    estimated_potential_energy: f64,
    total_momentum: [f64; 3],
}

#[repr(C)]
struct FfiCreateParams {
    particle_count: u64,
    galaxy_count: u32,
    grav_const_kpc_kms2_per_msun: f64,
    base_timestep_myr: f64,
    max_substeps: u32,
    cfl_safety_factor: f64,
    opening_angle: f64,
    mesh_resolution: [u32; 3],
    enable_smbh_post_newtonian: u32,
}

unsafe extern "C" {
    fn sim_cuda_create(
        params: *const FfiCreateParams,
        particles: *const FfiParticle,
        galaxies: *const FfiGalaxy,
        out_handle: *mut *mut c_void,
        error_buffer: *mut i8,
        error_buffer_len: usize,
    ) -> i32;
    fn sim_cuda_destroy(handle: *mut c_void) -> i32;
    fn sim_cuda_step(
        handle: *mut c_void,
        requested_substeps: u32,
        diagnostics: *mut FfiDiagnostics,
        error_buffer: *mut i8,
        error_buffer_len: usize,
    ) -> i32;
    fn sim_cuda_advance(
        handle: *mut c_void,
        steps: u32,
        diagnostics: *mut FfiDiagnostics,
        error_buffer: *mut i8,
        error_buffer_len: usize,
    ) -> i32;
    fn sim_cuda_fill_preview(
        handle: *mut c_void,
        max_particles: u32,
        out_particles: *mut FfiPreviewParticle,
        out_count: *mut u32,
        error_buffer: *mut i8,
        error_buffer_len: usize,
    ) -> i32;
    fn sim_cuda_copy_particles(
        handle: *mut c_void,
        out_particles: *mut FfiParticle,
        particle_capacity: u64,
        error_buffer: *mut i8,
        error_buffer_len: usize,
    ) -> i32;
}

pub struct GpuBackend {
    handle: *mut c_void,
    particle_count: u64,
    last_diagnostics: Diagnostics,
    preview_scratch: Vec<FfiPreviewParticle>,
}

unsafe impl Send for GpuBackend {}

impl GpuBackend {
    pub fn new(
        config: &SimulationConfig,
        initial_conditions: &InitialConditions,
    ) -> anyhow::Result<Self> {
        let params = FfiCreateParams {
            particle_count: initial_conditions.particles.len() as u64,
            galaxy_count: config.galaxies.len() as u32,
            grav_const_kpc_kms2_per_msun: config.gravity.grav_const_kpc_kms2_per_msun,
            base_timestep_myr: config.integration.base_timestep_myr,
            max_substeps: config.integration.max_substeps,
            cfl_safety_factor: config.integration.cfl_safety_factor,
            opening_angle: config.gravity.opening_angle,
            mesh_resolution: config.gravity.mesh_resolution,
            enable_smbh_post_newtonian: u32::from(
                config.relativity.enable_smbh_post_newtonian,
            ),
        };
        let ffi_particles: Vec<FfiParticle> = initial_conditions
            .particles
            .iter()
            .map(ffi_particle_from_particle)
            .collect();
        let ffi_galaxies: Vec<FfiGalaxy> = config
            .galaxies
            .iter()
            .map(ffi_galaxy_from_config)
            .collect();

        let mut handle = std::ptr::null_mut();
        let mut error_buffer = [0_i8; 512];
        let code = unsafe {
            sim_cuda_create(
                &params,
                ffi_particles.as_ptr(),
                ffi_galaxies.as_ptr(),
                &mut handle,
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        if code != 0 {
            return Err(anyhow!(decode_error(&error_buffer)))
                .context("failed to initialize CUDA simulation backend");
        }

        Ok(Self {
            handle,
            particle_count: initial_conditions.particles.len() as u64,
            last_diagnostics: Diagnostics {
                particle_count: initial_conditions.particles.len() as u64,
                ..Diagnostics::default()
            },
            preview_scratch: Vec::new(),
        })
    }

    pub fn step(&mut self, requested_substeps: u32) -> anyhow::Result<Diagnostics> {
        let mut ffi = MaybeUninit::<FfiDiagnostics>::uninit();
        let mut error_buffer = [0_i8; 512];
        let code = unsafe {
            sim_cuda_step(
                self.handle,
                requested_substeps,
                ffi.as_mut_ptr(),
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        if code != 0 {
            return Err(anyhow!(decode_error(&error_buffer))).context("CUDA step failed");
        }

        let diagnostics = diagnostics_from_ffi(unsafe { ffi.assume_init() });
        self.last_diagnostics = diagnostics.clone();
        Ok(diagnostics)
    }

    pub fn advance(&mut self, steps: u32) -> anyhow::Result<Diagnostics> {
        let mut ffi = MaybeUninit::<FfiDiagnostics>::uninit();
        let mut error_buffer = [0_i8; 512];
        let code = unsafe {
            sim_cuda_advance(
                self.handle,
                steps,
                ffi.as_mut_ptr(),
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        if code != 0 {
            return Err(anyhow!(decode_error(&error_buffer))).context("CUDA advance failed");
        }

        let diagnostics = diagnostics_from_ffi(unsafe { ffi.assume_init() });
        self.last_diagnostics = diagnostics.clone();
        Ok(diagnostics)
    }

    pub fn preview_frame(&mut self, budget: u32) -> anyhow::Result<PreviewFrame> {
        let max_particles = budget.min(self.particle_count.min(u64::from(u32::MAX)) as u32);
        self.preview_scratch.resize(
            max_particles as usize,
            FfiPreviewParticle {
                position_kpc: [0.0; 3],
                velocity_kms: [0.0; 3],
                mass_msun: 0.0,
                galaxy_index: 0,
                component: 0,
                color_rgba: [0.0; 4],
                intensity: 0.0,
            },
        );
        let mut out_count = 0_u32;
        let mut error_buffer = [0_i8; 512];
        let code = unsafe {
            sim_cuda_fill_preview(
                self.handle,
                max_particles,
                self.preview_scratch.as_mut_ptr(),
                &mut out_count,
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        if code != 0 {
            return Err(anyhow!(decode_error(&error_buffer))).context("preview extraction failed");
        }

        Ok(PreviewFrame {
            sim_time_myr: self.last_diagnostics.sim_time_myr,
            diagnostics: Diagnostics {
                preview_count: out_count,
                ..self.last_diagnostics.clone()
            },
            particles: self.preview_scratch[..out_count as usize]
                .into_iter()
                .map(|particle| PreviewParticle {
                    position_kpc: particle.position_kpc,
                    velocity_kms: particle.velocity_kms,
                    mass_msun: particle.mass_msun,
                    galaxy_index: particle.galaxy_index,
                    component: particle.component,
                    color_rgba: particle.color_rgba,
                    intensity: particle.intensity,
                })
                .collect(),
        })
    }

    pub fn download_particles(&mut self) -> anyhow::Result<Vec<Particle>> {
        let mut particles = vec![
            FfiParticle {
                position_kpc: [0.0; 3],
                velocity_kms: [0.0; 3],
                mass_msun: 0.0,
                softening_kpc: 0.0,
                galaxy_index: 0,
                component: 0,
                color_rgba: [0.0; 4],
            };
            self.particle_count as usize
        ];
        let mut error_buffer = [0_i8; 512];
        let code = unsafe {
            sim_cuda_copy_particles(
                self.handle,
                particles.as_mut_ptr(),
                self.particle_count,
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        if code != 0 {
            return Err(anyhow!(decode_error(&error_buffer)))
                .context("failed to download particles");
        }

        Ok(particles.into_iter().map(particle_from_ffi).collect())
    }
}

impl Drop for GpuBackend {
    fn drop(&mut self) {
        if !self.handle.is_null() {
            let _ = unsafe { sim_cuda_destroy(self.handle) };
        }
    }
}

fn ffi_particle_from_particle(particle: &Particle) -> FfiParticle {
    FfiParticle {
        position_kpc: [
            particle.position_kpc.x,
            particle.position_kpc.y,
            particle.position_kpc.z,
        ],
        velocity_kms: [
            particle.velocity_kms.x,
            particle.velocity_kms.y,
            particle.velocity_kms.z,
        ],
        mass_msun: particle.mass_msun,
        softening_kpc: particle.softening_kpc,
        galaxy_index: particle.galaxy_index,
        component: match particle.component {
            ParticleComponent::Halo => 0,
            ParticleComponent::Disk => 1,
            ParticleComponent::Bulge => 2,
            ParticleComponent::Smbh => 3,
        },
        color_rgba: particle.color_rgba,
    }
}

fn ffi_galaxy_from_config(galaxy: &sim_core::GalaxyConfig) -> FfiGalaxy {
    let rotation = rotation_from_euler_deg(galaxy.disk_tilt_deg);
    FfiGalaxy {
        halo_mass_msun: galaxy.halo_mass_msun,
        halo_scale_radius_kpc: galaxy.halo_scale_radius_kpc,
        disk_mass_msun: galaxy.disk_mass_msun,
        disk_scale_radius_kpc: galaxy.disk_scale_radius_kpc,
        disk_scale_height_kpc: galaxy.disk_scale_height_kpc,
        bulge_mass_msun: galaxy.bulge_mass_msun,
        bulge_scale_radius_kpc: galaxy.bulge_scale_radius_kpc,
        disk_rotation: [
            rotation[0][0],
            rotation[0][1],
            rotation[0][2],
            rotation[1][0],
            rotation[1][1],
            rotation[1][2],
            rotation[2][0],
            rotation[2][1],
            rotation[2][2],
        ],
    }
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

fn diagnostics_from_ffi(ffi: FfiDiagnostics) -> Diagnostics {
    Diagnostics {
        particle_count: ffi.particle_count,
        preview_count: ffi.preview_count,
        sim_time_myr: ffi.sim_time_myr,
        dt_myr: ffi.dt_myr,
        kinetic_energy: ffi.kinetic_energy,
        estimated_potential_energy: ffi.estimated_potential_energy,
        total_momentum: Vec3::new(
            ffi.total_momentum[0],
            ffi.total_momentum[1],
            ffi.total_momentum[2],
        ),
    }
}

fn particle_from_ffi(particle: FfiParticle) -> Particle {
    Particle {
        galaxy_index: particle.galaxy_index,
        component: match particle.component {
            0 => ParticleComponent::Halo,
            1 => ParticleComponent::Disk,
            2 => ParticleComponent::Bulge,
            3 => ParticleComponent::Smbh,
            _ => ParticleComponent::Halo,
        },
        position_kpc: Vec3::new(
            particle.position_kpc[0],
            particle.position_kpc[1],
            particle.position_kpc[2],
        ),
        velocity_kms: Vec3::new(
            particle.velocity_kms[0],
            particle.velocity_kms[1],
            particle.velocity_kms[2],
        ),
        mass_msun: particle.mass_msun,
        softening_kpc: particle.softening_kpc,
        color_rgba: particle.color_rgba,
    }
}

fn decode_error(bytes: &[i8]) -> String {
    let bytes: Vec<u8> = bytes
        .iter()
        .copied()
        .take_while(|byte| *byte != 0)
        .map(|byte| byte as u8)
        .collect();
    String::from_utf8_lossy(&bytes).trim().to_string()
}

#[cfg(test)]
mod tests {
    use sim_core::{
        Diagnostics, GalaxyConfig, InitialConditions, ObserverEffectsConfig, Particle,
        ParticleComponent, PreviewConfig, RelativityConfig, SimulationConfig, SmbhConfig,
        SnapshotConfig, TimeIntegrationConfig, Vec3, built_in_presets,
    };
    use uuid::Uuid;

    use super::GpuBackend;

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn gpu_backend_steps_emits_preview_and_downloads_particles() {
        let config = small_test_config();
        let initial_conditions = InitialConditions::generate(&config, 7).unwrap();
        let before_positions: Vec<Vec3> = initial_conditions
            .particles
            .iter()
            .map(|particle| particle.position_kpc)
            .collect();

        let mut backend = GpuBackend::new(&config, &initial_conditions).unwrap();
        let initial_preview = backend.preview_frame(64).unwrap();
        assert!(!initial_preview.particles.is_empty());
        assert!(initial_preview.particles.len() <= 64);

        let diagnostics = backend.step(2).unwrap();
        assert!(diagnostics.sim_time_myr > 0.0);
        assert_eq!(
            diagnostics.particle_count,
            initial_conditions.particles.len() as u64
        );

        let stepped_preview = backend.preview_frame(64).unwrap();
        assert!(stepped_preview.diagnostics.preview_count > 0);

        let particles = backend.download_particles().unwrap();
        assert_eq!(particles.len(), initial_conditions.particles.len());
        assert!(
            particles
                .iter()
                .zip(before_positions.iter())
                .any(|(particle, before)| {
                    (particle.position_kpc - *before).length_squared() > 0.0
                })
        );
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn isolated_primary_galaxy_stays_compact_over_short_horizon() {
        let mut config = small_test_config();
        config.galaxies.truncate(1);
        config.name = "isolated-primary-stability".to_string();

        let initial_conditions = InitialConditions::generate(&config, 13).unwrap();
        let initial_disk_radius = mean_disk_radius(&initial_conditions.particles);

        let mut backend = GpuBackend::new(&config, &initial_conditions).unwrap();
        let diagnostics = backend.advance(256).unwrap();
        assert!(diagnostics.sim_time_myr > 0.0);

        let particles = backend.download_particles().unwrap();
        let final_disk_radius = mean_disk_radius(&particles);
        let ratio = final_disk_radius / initial_disk_radius;

        assert!(
            (0.6..=1.4).contains(&ratio),
            "disk radius drifted too far: initial={initial_disk_radius:.3}, final={final_disk_radius:.3}, ratio={ratio:.3}"
        );
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn isolated_primary_galaxy_retains_core_and_spin() {
        let mut config = small_test_config();
        config.galaxies.truncate(1);
        config.name = "isolated-primary-core-spin".to_string();
        config.gravity.mesh_resolution = [128, 128, 64];

        let initial_conditions = InitialConditions::generate(&config, 29).unwrap();
        let initial_spin = disk_angular_momentum_magnitude(&initial_conditions.particles);
        let initial_inner_radius = disk_quantile_radius(&initial_conditions.particles, 0.25);

        let mut backend = GpuBackend::new(&config, &initial_conditions).unwrap();
        let diagnostics = backend.advance(192).unwrap();
        assert!(diagnostics.sim_time_myr > 0.0);

        let particles = backend.download_particles().unwrap();
        let final_spin = disk_angular_momentum_magnitude(&particles);
        let final_inner_radius = disk_quantile_radius(&particles, 0.25);

        assert!(
            final_spin > initial_spin * 0.55,
            "disk lost too much angular momentum: initial={initial_spin:.3e}, final={final_spin:.3e}"
        );
        let inner_radius_ratio = final_inner_radius / initial_inner_radius.max(1.0e-6);
        assert!(
            (0.65..=1.45).contains(&inner_radius_ratio),
            "disk inner radius drifted too far: initial={initial_inner_radius:.3}, final={final_inner_radius:.3}, ratio={inner_radius_ratio:.3}"
        );
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn self_gravity_moves_particles_without_analytic_galaxy_masses() {
        let config = self_gravity_only_config();
        let initial_conditions = InitialConditions {
            seed: 11,
            total_mass_msun: 2.0e8,
            particles: vec![
                Particle {
                    galaxy_index: 0,
                    component: ParticleComponent::Disk,
                    position_kpc: Vec3::new(-8.0, 0.0, 0.0),
                    velocity_kms: Vec3::ZERO,
                    mass_msun: 1.0e8,
                    softening_kpc: 0.2,
                    color_rgba: [1.0, 1.0, 1.0, 1.0],
                },
                Particle {
                    galaxy_index: 0,
                    component: ParticleComponent::Disk,
                    position_kpc: Vec3::new(8.0, 0.0, 0.0),
                    velocity_kms: Vec3::ZERO,
                    mass_msun: 1.0e8,
                    softening_kpc: 0.2,
                    color_rgba: [1.0, 1.0, 1.0, 1.0],
                },
                Particle {
                    galaxy_index: 0,
                    component: ParticleComponent::Smbh,
                    position_kpc: Vec3::new(0.0, 0.0, 0.0),
                    velocity_kms: Vec3::ZERO,
                    mass_msun: 0.0,
                    softening_kpc: 0.01,
                    color_rgba: [1.0, 1.0, 1.0, 1.0],
                },
            ],
        };

        let mut backend = GpuBackend::new(&config, &initial_conditions).unwrap();
        let Diagnostics {
            sim_time_myr, ..
        } = backend.advance(32).unwrap();
        assert!(sim_time_myr > 0.0);

        let particles = backend.download_particles().unwrap();
        let left = particles
            .iter()
            .find(|particle| {
                matches!(particle.component, ParticleComponent::Disk)
                    && particle.position_kpc.x < 0.0
            })
            .unwrap();
        let right = particles
            .iter()
            .find(|particle| {
                matches!(particle.component, ParticleComponent::Disk)
                    && particle.position_kpc.x > 0.0
            })
            .unwrap();

        let final_separation = (right.position_kpc - left.position_kpc).length();
        assert!(
            final_separation < 16.0,
            "self-gravity should pull the particles together, got separation {final_separation:.3}"
        );
    }

    #[derive(Clone, Copy, Debug)]
    struct GalaxyDiskMetrics {
        r50_kpc: f64,
        rms_height_kpc: f64,
        spin: f64,
    }

    #[test]
    #[ignore = "requires NVIDIA GPU"]
    fn major_merger_debug_stays_structurally_bound_over_first_2_myr() {
        let config = built_in_presets()
            .into_iter()
            .find(|preset| preset.id == "major-merger-debug")
            .unwrap()
            .config;
        let initial_conditions = InitialConditions::generate(&config, 42).unwrap();
        let initial_metrics = [
            galaxy_disk_metrics(&initial_conditions.particles, 0),
            galaxy_disk_metrics(&initial_conditions.particles, 1),
        ];

        let mut backend = GpuBackend::new(&config, &initial_conditions).unwrap();
        let diagnostics = backend.advance(20).unwrap();
        assert!(diagnostics.sim_time_myr >= 2.0 - 1.0e-6);

        let particles = backend.download_particles().unwrap();
        for (galaxy_index, initial) in initial_metrics.iter().enumerate() {
            let final_metrics = galaxy_disk_metrics(&particles, galaxy_index as u32);
            let r50_ratio = final_metrics.r50_kpc / initial.r50_kpc.max(1.0e-6);
            let z_ratio = final_metrics.rms_height_kpc / initial.rms_height_kpc.max(1.0e-6);
            let spin_ratio = final_metrics.spin / initial.spin.max(1.0e-6);

            assert!(
                r50_ratio <= 1.03,
                "galaxy {galaxy_index} disk expanded too quickly over first 2 Myr: ratio={r50_ratio:.3}"
            );
            assert!(
                z_ratio <= 1.05,
                "galaxy {galaxy_index} disk thickened too quickly over first 2 Myr: ratio={z_ratio:.3}"
            );
            assert!(
                spin_ratio >= 0.995,
                "galaxy {galaxy_index} disk lost too much spin over first 2 Myr: ratio={spin_ratio:.6}"
            );
        }
    }

    fn mean_disk_radius(particles: &[sim_core::Particle]) -> f64 {
        let center = particles
            .iter()
            .find(|particle| matches!(particle.component, ParticleComponent::Smbh))
            .map(|particle| particle.position_kpc)
            .unwrap_or(Vec3::ZERO);

        let mut total = 0.0;
        let mut count = 0_u64;
        for particle in particles {
            if !matches!(particle.component, ParticleComponent::Disk) {
                continue;
            }
            total += (particle.position_kpc - center).length();
            count += 1;
        }

        total / count.max(1) as f64
    }

    fn disk_quantile_radius(particles: &[sim_core::Particle], quantile: f64) -> f64 {
        let center = particles
            .iter()
            .find(|particle| matches!(particle.component, ParticleComponent::Smbh))
            .map(|particle| particle.position_kpc)
            .unwrap_or(Vec3::ZERO);
        let mut radii: Vec<f64> = particles
            .iter()
            .filter(|particle| matches!(particle.component, ParticleComponent::Disk))
            .map(|particle| (particle.position_kpc - center).length())
            .collect();
        if radii.is_empty() {
            return 0.0;
        }
        radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let index = ((radii.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
        radii[index]
    }

    fn disk_angular_momentum_magnitude(particles: &[sim_core::Particle]) -> f64 {
        let center = particles
            .iter()
            .find(|particle| matches!(particle.component, ParticleComponent::Smbh))
            .map(|particle| particle.position_kpc)
            .unwrap_or(Vec3::ZERO);
        let bulk_velocity = particles
            .iter()
            .find(|particle| matches!(particle.component, ParticleComponent::Smbh))
            .map(|particle| particle.velocity_kms)
            .unwrap_or(Vec3::ZERO);
        particles
            .iter()
            .filter(|particle| matches!(particle.component, ParticleComponent::Disk))
            .fold(Vec3::ZERO, |accum, particle| {
                let radius = particle.position_kpc - center;
                let velocity = particle.velocity_kms - bulk_velocity;
                let angular_momentum = Vec3::new(
                    radius.y * velocity.z - radius.z * velocity.y,
                    radius.z * velocity.x - radius.x * velocity.z,
                    radius.x * velocity.y - radius.y * velocity.x,
                ) * particle.mass_msun;
                accum + angular_momentum
            })
            .length()
    }

    fn galaxy_disk_metrics(particles: &[sim_core::Particle], galaxy_index: u32) -> GalaxyDiskMetrics {
        let center = particles
            .iter()
            .find(|particle| {
                particle.galaxy_index == galaxy_index
                    && matches!(particle.component, ParticleComponent::Smbh)
            })
            .map(|particle| particle.position_kpc)
            .unwrap_or(Vec3::ZERO);
        let bulk_velocity = particles
            .iter()
            .find(|particle| {
                particle.galaxy_index == galaxy_index
                    && matches!(particle.component, ParticleComponent::Smbh)
            })
            .map(|particle| particle.velocity_kms)
            .unwrap_or(Vec3::ZERO);

        let disk_particles: Vec<_> = particles
            .iter()
            .filter(|particle| {
                particle.galaxy_index == galaxy_index
                    && matches!(particle.component, ParticleComponent::Disk)
            })
            .collect();
        let spin_vector = disk_particles.iter().fold(Vec3::ZERO, |accum, particle| {
            let radius = particle.position_kpc - center;
            let velocity = particle.velocity_kms - bulk_velocity;
            accum
                + Vec3::new(
                    radius.y * velocity.z - radius.z * velocity.y,
                    radius.z * velocity.x - radius.x * velocity.z,
                    radius.x * velocity.y - radius.y * velocity.x,
                ) * particle.mass_msun
        });
        let disk_normal = spin_vector.normalized();

        let mut cylindrical_radii: Vec<f64> = disk_particles
            .iter()
            .map(|particle| {
                let relative = particle.position_kpc - center;
                let height = relative.dot(disk_normal);
                let in_plane = relative - disk_normal * height;
                in_plane.length()
            })
            .collect();
        cylindrical_radii.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let rms_height_kpc = (disk_particles
            .iter()
            .map(|particle| {
                let relative = particle.position_kpc - center;
                let height = relative.dot(disk_normal);
                height * height
            })
            .sum::<f64>()
            / disk_particles.len().max(1) as f64)
            .sqrt();

        GalaxyDiskMetrics {
            r50_kpc: quantile_radius(&cylindrical_radii, 0.5),
            rms_height_kpc,
            spin: spin_vector.length(),
        }
    }

    fn quantile_radius(sorted_radii: &[f64], quantile: f64) -> f64 {
        if sorted_radii.is_empty() {
            return 0.0;
        }
        let index =
            ((sorted_radii.len() - 1) as f64 * quantile.clamp(0.0, 1.0)).round() as usize;
        sorted_radii[index]
    }

    fn small_test_config() -> SimulationConfig {
        let mut config = built_in_presets()
            .into_iter()
            .find(|preset| preset.id == "minor-merger")
            .unwrap()
            .config;
        config.name = "gpu-backend-smoke".to_string();
        config.output_directory = format!("/tmp/galaxy-cuda-smoke-{}", Uuid::new_v4());
        config.snapshots.directory = config.output_directory.clone();
        config.preview.particle_budget = 64;
        config.gravity.mesh_resolution = [64, 64, 32];
        for galaxy in &mut config.galaxies {
            galaxy.halo_particle_count = 96;
            galaxy.disk_particle_count = 48;
            galaxy.bulge_particle_count = 12;
        }
        config
    }

    fn self_gravity_only_config() -> SimulationConfig {
        SimulationConfig {
            name: "self-gravity-only".to_string(),
            gravity: sim_core::GravityConfig {
                grav_const_kpc_kms2_per_msun: 4.300_91e-6,
                halo_softening_kpc: 0.2,
                disk_softening_kpc: 0.2,
                bulge_softening_kpc: 0.2,
                opening_angle: 0.55,
                mesh_resolution: [64, 64, 32],
            },
            relativity: RelativityConfig {
                enable_weak_field_mesh: true,
                enable_smbh_post_newtonian: true,
                observer_effects: ObserverEffectsConfig {
                    doppler_boosting: true,
                    gravitational_redshift: true,
                    weak_lensing: true,
                    time_delay: true,
                },
            },
            preview: PreviewConfig {
                particle_budget: 16,
                density_grid: [64, 64],
                target_fps: 60,
            },
            snapshots: SnapshotConfig {
                directory: format!("/tmp/galaxy-self-gravity-{}", Uuid::new_v4()),
                cadence_steps: 120,
                compress: false,
            },
            integration: TimeIntegrationConfig {
                base_timestep_myr: 0.05,
                max_substeps: 4,
                cfl_safety_factor: 0.35,
            },
            galaxies: vec![GalaxyConfig {
                label: "Test".to_string(),
                halo_mass_msun: 0.0,
                halo_scale_radius_kpc: 10.0,
                halo_particle_count: 0,
                disk_mass_msun: 0.0,
                disk_scale_radius_kpc: 2.0,
                disk_scale_height_kpc: 0.4,
                disk_particle_count: 0,
                bulge_mass_msun: 0.0,
                bulge_scale_radius_kpc: 1.0,
                bulge_particle_count: 0,
                smbh: SmbhConfig {
                    mass_msun: 0.0,
                    softening_kpc: 0.01,
                    substeps: 8,
                },
                position_kpc: [0.0, 0.0, 0.0],
                velocity_kms: [0.0, 0.0, 0.0],
                disk_tilt_deg: [0.0, 0.0, 0.0],
                color_rgba: [1.0, 1.0, 1.0, 1.0],
            }],
            initial_separation_kpc: 0.0,
            initial_relative_velocity_kms: 0.0,
            output_directory: format!("/tmp/galaxy-self-gravity-{}", Uuid::new_v4()),
        }
    }
}
