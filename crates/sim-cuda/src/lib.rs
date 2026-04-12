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
    color_rgba: [f32; 4],
    intensity: f32,
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
    grav_const_kpc_kms2_per_msun: f64,
    base_timestep_myr: f64,
}

unsafe extern "C" {
    fn sim_cuda_create(
        params: *const FfiCreateParams,
        particles: *const FfiParticle,
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
}

unsafe impl Send for GpuBackend {}

impl GpuBackend {
    pub fn new(
        config: &SimulationConfig,
        initial_conditions: &InitialConditions,
    ) -> anyhow::Result<Self> {
        let params = FfiCreateParams {
            particle_count: initial_conditions.particles.len() as u64,
            grav_const_kpc_kms2_per_msun: config.gravity.grav_const_kpc_kms2_per_msun,
            base_timestep_myr: config.integration.base_timestep_myr,
        };
        let ffi_particles: Vec<FfiParticle> = initial_conditions
            .particles
            .iter()
            .map(ffi_particle_from_particle)
            .collect();

        let mut handle = std::ptr::null_mut();
        let mut error_buffer = [0_i8; 512];
        let code = unsafe {
            sim_cuda_create(
                &params,
                ffi_particles.as_ptr(),
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

    pub fn preview_frame(&mut self, budget: u32) -> anyhow::Result<PreviewFrame> {
        let max_particles = budget.min(self.particle_count.min(u64::from(u32::MAX)) as u32);
        let mut particles = vec![
            FfiPreviewParticle {
                position_kpc: [0.0; 3],
                velocity_kms: [0.0; 3],
                color_rgba: [0.0; 4],
                intensity: 0.0,
            };
            max_particles as usize
        ];
        let mut out_count = 0_u32;
        let mut error_buffer = [0_i8; 512];
        let code = unsafe {
            sim_cuda_fill_preview(
                self.handle,
                max_particles,
                particles.as_mut_ptr(),
                &mut out_count,
                error_buffer.as_mut_ptr(),
                error_buffer.len(),
            )
        };
        if code != 0 {
            return Err(anyhow!(decode_error(&error_buffer))).context("preview extraction failed");
        }

        particles.truncate(out_count as usize);
        Ok(PreviewFrame {
            sim_time_myr: self.last_diagnostics.sim_time_myr,
            diagnostics: Diagnostics {
                preview_count: out_count,
                ..self.last_diagnostics.clone()
            },
            particles: particles
                .into_iter()
                .map(|particle| PreviewParticle {
                    position_kpc: particle.position_kpc,
                    velocity_kms: particle.velocity_kms,
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
