#pragma once

#include <cstddef>
#include <cstdint>

extern "C" {

struct SimCudaParticle {
  double position_kpc[3];
  double velocity_kms[3];
  double mass_msun;
  double softening_kpc;
  std::uint32_t galaxy_index;
  std::uint32_t component;
  float color_rgba[4];
};

struct SimCudaPreviewParticle {
  float position_kpc[3];
  float velocity_kms[3];
  float mass_msun;
  std::uint32_t component;
};

struct SimCudaGalaxy {
  double halo_mass_msun;
  double halo_scale_radius_kpc;
  double disk_mass_msun;
  double disk_scale_radius_kpc;
  double disk_scale_height_kpc;
  double bulge_mass_msun;
  double bulge_scale_radius_kpc;
  double disk_rotation[9];
};

struct SimCudaDiagnostics {
  std::uint64_t particle_count;
  std::uint32_t preview_count;
  double sim_time_myr;
  double dt_myr;
  double kinetic_energy;
  double estimated_potential_energy;
  double total_momentum[3];
};

struct SimCudaCreateParams {
  std::uint64_t particle_count;
  std::uint32_t galaxy_count;
  double grav_const_kpc_kms2_per_msun;
  double base_timestep_myr;
  std::uint32_t max_substeps;
  double cfl_safety_factor;
  double opening_angle;
  std::uint32_t mesh_resolution[3];
  std::uint32_t enable_smbh_post_newtonian;
};

int sim_cuda_create(const SimCudaCreateParams* params,
                    const SimCudaParticle* particles,
                    const SimCudaGalaxy* galaxies,
                    void** out_handle,
                    char* error_buffer,
                    std::size_t error_buffer_len);

int sim_cuda_destroy(void* handle);

int sim_cuda_step(void* handle,
                  std::uint32_t requested_substeps,
                  SimCudaDiagnostics* diagnostics,
                  char* error_buffer,
                  std::size_t error_buffer_len);

int sim_cuda_advance(void* handle,
                     std::uint32_t steps,
                     SimCudaDiagnostics* diagnostics,
                     char* error_buffer,
                     std::size_t error_buffer_len);

int sim_cuda_fill_preview(void* handle,
                          std::uint32_t max_particles,
                          SimCudaPreviewParticle* out_particles,
                          std::uint32_t* out_count,
                          char* error_buffer,
                          std::size_t error_buffer_len);

int sim_cuda_request_preview(void* handle,
                             std::uint32_t max_particles,
                             std::uint32_t* out_count,
                             char* error_buffer,
                             std::size_t error_buffer_len);

int sim_cuda_collect_preview(void* handle,
                             SimCudaPreviewParticle* out_particles,
                             std::uint32_t particle_capacity,
                             std::uint32_t* out_count,
                             int* out_ready,
                             char* error_buffer,
                             std::size_t error_buffer_len);

int sim_cuda_copy_particles(void* handle,
                            SimCudaParticle* out_particles,
                            std::uint64_t particle_capacity,
                            char* error_buffer,
                            std::size_t error_buffer_len);

}
