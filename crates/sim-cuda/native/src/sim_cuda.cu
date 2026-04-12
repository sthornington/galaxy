#include "sim_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr double kKpcPerKmPerMyr = 0.001022712165045695;

struct DeviceState {
  std::uint64_t particle_count = 0;
  std::uint32_t galaxy_count = 0;
  double grav_const = 0.0;
  double base_timestep_myr = 0.0;
  double sim_time_myr = 0.0;

  SimCudaParticle* particles = nullptr;
  SimCudaGalaxy* galaxies = nullptr;
  int* galaxy_smbh_indices = nullptr;
  std::vector<int> galaxy_smbh_indices_host;
  SimCudaPreviewParticle* preview_particles = nullptr;
  std::uint32_t preview_capacity = 0;
};

void fill_error(char* buffer, std::size_t len, const char* message) {
  if (buffer == nullptr || len == 0) {
    return;
  }
  std::snprintf(buffer, len, "%s", message);
}

void fill_cuda_error(char* buffer, std::size_t len, const char* context, cudaError_t err) {
  if (buffer == nullptr || len == 0) {
    return;
  }
  std::snprintf(buffer, len, "%s: %s", context, cudaGetErrorString(err));
}

void fill_basic_diagnostics(const DeviceState& state,
                            const double dt_myr,
                            SimCudaDiagnostics* diagnostics) {
  diagnostics->particle_count = state.particle_count;
  diagnostics->preview_count = 0;
  diagnostics->sim_time_myr = state.sim_time_myr;
  diagnostics->dt_myr = dt_myr;
  diagnostics->kinetic_energy = 0.0;
  diagnostics->estimated_potential_energy = 0.0;
  diagnostics->total_momentum[0] = 0.0;
  diagnostics->total_momentum[1] = 0.0;
  diagnostics->total_momentum[2] = 0.0;
}

int ensure_preview_capacity(DeviceState* state,
                            const std::uint32_t count,
                            char* error_buffer,
                            const std::size_t error_buffer_len) {
  if (count <= state->preview_capacity) {
    return 0;
  }

  if (state->preview_particles != nullptr) {
    cudaFree(state->preview_particles);
    state->preview_particles = nullptr;
    state->preview_capacity = 0;
  }

  cudaError_t cuda_status = cudaMalloc(
      reinterpret_cast<void**>(&state->preview_particles),
      sizeof(SimCudaPreviewParticle) * count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for preview buffer failed", cuda_status);
    return 1;
  }

  state->preview_capacity = count;
  return 0;
}

__device__ __forceinline__ double softened_inv_r3(const double dx,
                                                  const double dy,
                                                  const double dz,
                                                  const double softening) {
  const double r2 = dx * dx + dy * dy + dz * dz + softening * softening;
  const double inv_r = rsqrt(r2);
  return inv_r * inv_r * inv_r;
}

__device__ __forceinline__ void add_point_mass_acceleration(double& ax,
                                                            double& ay,
                                                            double& az,
                                                            const double dx,
                                                            const double dy,
                                                            const double dz,
                                                            const double softening,
                                                            const double grav_const,
                                                            const double mass_msun) {
  if (mass_msun <= 0.0) {
    return;
  }

  const double accel_scale = grav_const * mass_msun * softened_inv_r3(dx, dy, dz, softening);
  ax += accel_scale * dx;
  ay += accel_scale * dy;
  az += accel_scale * dz;
}

__device__ __forceinline__ void add_hernquist_acceleration(double& ax,
                                                           double& ay,
                                                           double& az,
                                                           const double dx,
                                                           const double dy,
                                                           const double dz,
                                                           const double softening,
                                                           const double grav_const,
                                                           const double mass_msun,
                                                           const double scale_radius_kpc) {
  if (mass_msun <= 0.0 || scale_radius_kpc <= 0.0) {
    return;
  }

  const double r = sqrt(dx * dx + dy * dy + dz * dz + softening * softening);
  if (r <= 1.0e-9) {
    return;
  }

  const double denom = r * (r + scale_radius_kpc) * (r + scale_radius_kpc);
  const double accel_scale = grav_const * mass_msun / denom;
  ax += accel_scale * dx;
  ay += accel_scale * dy;
  az += accel_scale * dz;
}

__device__ __forceinline__ void world_to_disk(const double* rotation,
                                              const double wx,
                                              const double wy,
                                              const double wz,
                                              double& lx,
                                              double& ly,
                                              double& lz) {
  lx = rotation[0] * wx + rotation[3] * wy + rotation[6] * wz;
  ly = rotation[1] * wx + rotation[4] * wy + rotation[7] * wz;
  lz = rotation[2] * wx + rotation[5] * wy + rotation[8] * wz;
}

__device__ __forceinline__ void disk_to_world(const double* rotation,
                                              const double lx,
                                              const double ly,
                                              const double lz,
                                              double& wx,
                                              double& wy,
                                              double& wz) {
  wx = rotation[0] * lx + rotation[1] * ly + rotation[2] * lz;
  wy = rotation[3] * lx + rotation[4] * ly + rotation[5] * lz;
  wz = rotation[6] * lx + rotation[7] * ly + rotation[8] * lz;
}

__device__ __forceinline__ void add_miyamoto_nagai_acceleration(double& ax,
                                                                double& ay,
                                                                double& az,
                                                                const double* rotation,
                                                                const double world_rel_x,
                                                                const double world_rel_y,
                                                                const double world_rel_z,
                                                                const double softening,
                                                                const double grav_const,
                                                                const double mass_msun,
                                                                const double scale_radius_kpc,
                                                                const double scale_height_kpc) {
  if (mass_msun <= 0.0 || scale_radius_kpc <= 0.0 || scale_height_kpc <= 0.0) {
    return;
  }

  double lx = 0.0;
  double ly = 0.0;
  double lz = 0.0;
  world_to_disk(rotation, world_rel_x, world_rel_y, world_rel_z, lx, ly, lz);

  const double b = fmax(scale_height_kpc, 1.0e-4);
  const double B = sqrt(lz * lz + b * b);
  const double a = fmax(scale_radius_kpc, 1.0e-4);
  const double sum = a + B;
  const double denom2 = lx * lx + ly * ly + sum * sum + softening * softening;
  const double denom = sqrt(denom2);
  const double accel_scale = grav_const * mass_msun / (denom2 * denom);

  const double local_ax = -accel_scale * lx;
  const double local_ay = -accel_scale * ly;
  const double local_az = -accel_scale * sum * lz / fmax(B, 1.0e-6);

  double world_ax = 0.0;
  double world_ay = 0.0;
  double world_az = 0.0;
  disk_to_world(rotation, local_ax, local_ay, local_az, world_ax, world_ay, world_az);

  ax += world_ax;
  ay += world_ay;
  az += world_az;
}

__global__ void step_particles(SimCudaParticle* particles,
                               const SimCudaGalaxy* galaxies,
                               const int* galaxy_smbh_indices,
                               const std::uint32_t galaxy_count,
                               const std::uint64_t particle_count,
                               const double grav_const,
                               const double dt_myr) {
  const std::uint64_t index = static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= particle_count) {
    return;
  }

  SimCudaParticle& particle = particles[index];
  double ax = 0.0;
  double ay = 0.0;
  double az = 0.0;

  for (std::uint32_t galaxy_index = 0; galaxy_index < galaxy_count; ++galaxy_index) {
    const int center_index = galaxy_smbh_indices[galaxy_index];
    if (center_index < 0) {
      continue;
    }

    const SimCudaParticle center = particles[center_index];
    const SimCudaGalaxy galaxy = galaxies[galaxy_index];

    const double world_rel_x = particle.position_kpc[0] - center.position_kpc[0];
    const double world_rel_y = particle.position_kpc[1] - center.position_kpc[1];
    const double world_rel_z = particle.position_kpc[2] - center.position_kpc[2];
    const double to_center_x = -world_rel_x;
    const double to_center_y = -world_rel_y;
    const double to_center_z = -world_rel_z;
    const double softening = fmax(particle.softening_kpc, center.softening_kpc);

    add_hernquist_acceleration(
        ax,
        ay,
        az,
        to_center_x,
        to_center_y,
        to_center_z,
        softening,
        grav_const,
        galaxy.halo_mass_msun,
        galaxy.halo_scale_radius_kpc);
    add_hernquist_acceleration(
        ax,
        ay,
        az,
        to_center_x,
        to_center_y,
        to_center_z,
        softening,
        grav_const,
        galaxy.bulge_mass_msun,
        galaxy.bulge_scale_radius_kpc);
    add_miyamoto_nagai_acceleration(
        ax,
        ay,
        az,
        galaxy.disk_rotation,
        world_rel_x,
        world_rel_y,
        world_rel_z,
        softening,
        grav_const,
        galaxy.disk_mass_msun,
        galaxy.disk_scale_radius_kpc,
        galaxy.disk_scale_height_kpc);

    if (static_cast<int>(index) != center_index) {
      add_point_mass_acceleration(
          ax,
          ay,
          az,
          to_center_x,
          to_center_y,
          to_center_z,
          softening,
          grav_const,
          center.mass_msun);
    }
  }

  particle.velocity_kms[0] += ax * dt_myr * kKpcPerKmPerMyr;
  particle.velocity_kms[1] += ay * dt_myr * kKpcPerKmPerMyr;
  particle.velocity_kms[2] += az * dt_myr * kKpcPerKmPerMyr;

  particle.position_kpc[0] += particle.velocity_kms[0] * dt_myr * kKpcPerKmPerMyr;
  particle.position_kpc[1] += particle.velocity_kms[1] * dt_myr * kKpcPerKmPerMyr;
  particle.position_kpc[2] += particle.velocity_kms[2] * dt_myr * kKpcPerKmPerMyr;
}

__global__ void sample_preview(const SimCudaParticle* particles,
                               const std::uint64_t particle_count,
                               const std::uint32_t preview_count,
                               const std::uint64_t stride,
                               SimCudaPreviewParticle* out_particles) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= preview_count) {
    return;
  }

  const std::uint64_t source_index =
      min(static_cast<std::uint64_t>(index) * stride, particle_count - 1);
  const SimCudaParticle& particle = particles[source_index];
  const float speed = static_cast<float>(sqrt(
      particle.velocity_kms[0] * particle.velocity_kms[0] +
      particle.velocity_kms[1] * particle.velocity_kms[1] +
      particle.velocity_kms[2] * particle.velocity_kms[2]));

  out_particles[index].position_kpc[0] = static_cast<float>(particle.position_kpc[0]);
  out_particles[index].position_kpc[1] = static_cast<float>(particle.position_kpc[1]);
  out_particles[index].position_kpc[2] = static_cast<float>(particle.position_kpc[2]);
  out_particles[index].velocity_kms[0] = static_cast<float>(particle.velocity_kms[0]);
  out_particles[index].velocity_kms[1] = static_cast<float>(particle.velocity_kms[1]);
  out_particles[index].velocity_kms[2] = static_cast<float>(particle.velocity_kms[2]);
  out_particles[index].mass_msun = static_cast<float>(particle.mass_msun);
  out_particles[index].galaxy_index = particle.galaxy_index;
  out_particles[index].component = particle.component;
  out_particles[index].color_rgba[0] = particle.color_rgba[0];
  out_particles[index].color_rgba[1] = particle.color_rgba[1];
  out_particles[index].color_rgba[2] = particle.color_rgba[2];
  out_particles[index].color_rgba[3] = particle.color_rgba[3];
  out_particles[index].intensity = 0.35f + fminf(1.0f, speed / 320.0f);
}

SimCudaPreviewParticle preview_particle_from_host_particle(const SimCudaParticle& particle,
                                                           const float intensity) {
  SimCudaPreviewParticle preview{};
  preview.position_kpc[0] = static_cast<float>(particle.position_kpc[0]);
  preview.position_kpc[1] = static_cast<float>(particle.position_kpc[1]);
  preview.position_kpc[2] = static_cast<float>(particle.position_kpc[2]);
  preview.velocity_kms[0] = static_cast<float>(particle.velocity_kms[0]);
  preview.velocity_kms[1] = static_cast<float>(particle.velocity_kms[1]);
  preview.velocity_kms[2] = static_cast<float>(particle.velocity_kms[2]);
  preview.mass_msun = static_cast<float>(particle.mass_msun);
  preview.galaxy_index = particle.galaxy_index;
  preview.component = particle.component;
  preview.color_rgba[0] = particle.color_rgba[0];
  preview.color_rgba[1] = particle.color_rgba[1];
  preview.color_rgba[2] = particle.color_rgba[2];
  preview.color_rgba[3] = particle.color_rgba[3];
  preview.intensity = intensity;
  return preview;
}

int run_steps(DeviceState* state,
              const std::uint32_t step_count,
              const double dt_myr,
              const double advanced_myr,
              SimCudaDiagnostics* diagnostics,
              char* error_buffer,
              const std::size_t error_buffer_len) {
  if (diagnostics == nullptr) {
    fill_error(error_buffer, error_buffer_len, "null diagnostics pointer");
    return 1;
  }

  const int threads_per_block = 256;
  const int blocks =
      static_cast<int>((state->particle_count + threads_per_block - 1) / threads_per_block);

  for (std::uint32_t step = 0; step < step_count; ++step) {
    step_particles<<<blocks, threads_per_block>>>(
        state->particles,
        state->galaxies,
        state->galaxy_smbh_indices,
        state->galaxy_count,
        state->particle_count,
        state->grav_const,
        dt_myr);
    const cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "kernel launch failed", cuda_status);
      return 1;
    }
  }

  const cudaError_t cuda_status = cudaDeviceSynchronize();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "device synchronization failed", cuda_status);
    return 1;
  }

  state->sim_time_myr += advanced_myr;
  fill_basic_diagnostics(*state, advanced_myr, diagnostics);
  return 0;
}

}  // namespace

extern "C" int sim_cuda_create(const SimCudaCreateParams* params,
                               const SimCudaParticle* particles,
                               const SimCudaGalaxy* galaxies,
                               void** out_handle,
                               char* error_buffer,
                               std::size_t error_buffer_len) {
  if (params == nullptr || particles == nullptr || galaxies == nullptr || out_handle == nullptr) {
    fill_error(error_buffer, error_buffer_len, "invalid create parameters");
    return 1;
  }

  auto state = std::make_unique<DeviceState>();
  state->particle_count = params->particle_count;
  state->galaxy_count = params->galaxy_count;
  state->grav_const = params->grav_const_kpc_kms2_per_msun;
  state->base_timestep_myr = params->base_timestep_myr;

  const auto particle_bytes = sizeof(SimCudaParticle) * params->particle_count;
  cudaError_t cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->particles), particle_bytes);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for particles failed", cuda_status);
    return 1;
  }

  cuda_status = cudaMemcpy(state->particles, particles, particle_bytes, cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMemcpy for particles failed", cuda_status);
    return 1;
  }

  const auto galaxy_bytes = sizeof(SimCudaGalaxy) * params->galaxy_count;
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->galaxies), galaxy_bytes);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for galaxies failed", cuda_status);
    return 1;
  }

  cuda_status = cudaMemcpy(state->galaxies, galaxies, galaxy_bytes, cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMemcpy for galaxies failed", cuda_status);
    return 1;
  }

  std::vector<int> galaxy_smbh_indices(params->galaxy_count, -1);
  for (std::uint64_t i = 0; i < params->particle_count; ++i) {
    if (particles[i].component == 3u && particles[i].galaxy_index < params->galaxy_count) {
      galaxy_smbh_indices[particles[i].galaxy_index] = static_cast<int>(i);
    }
  }

  if (std::any_of(galaxy_smbh_indices.begin(), galaxy_smbh_indices.end(), [](const int index) {
        return index < 0;
      })) {
    fill_error(error_buffer, error_buffer_len, "every galaxy needs an SMBH anchor particle");
    return 1;
  }

  state->galaxy_smbh_indices_host = galaxy_smbh_indices;
  cuda_status = cudaMalloc(
      reinterpret_cast<void**>(&state->galaxy_smbh_indices),
      sizeof(int) * params->galaxy_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for SMBH anchors failed", cuda_status);
    return 1;
  }

  cuda_status = cudaMemcpy(
      state->galaxy_smbh_indices,
      galaxy_smbh_indices.data(),
      sizeof(int) * params->galaxy_count,
      cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMemcpy for SMBH anchors failed", cuda_status);
    return 1;
  }

  *out_handle = state.release();
  return 0;
}

extern "C" int sim_cuda_destroy(void* handle) {
  if (handle == nullptr) {
    return 0;
  }

  auto* state = static_cast<DeviceState*>(handle);
  cudaFree(state->preview_particles);
  cudaFree(state->galaxy_smbh_indices);
  cudaFree(state->galaxies);
  cudaFree(state->particles);
  delete state;
  return 0;
}

extern "C" int sim_cuda_step(void* handle,
                             std::uint32_t requested_substeps,
                             SimCudaDiagnostics* diagnostics,
                             char* error_buffer,
                             std::size_t error_buffer_len) {
  if (handle == nullptr) {
    fill_error(error_buffer, error_buffer_len, "null simulation handle");
    return 1;
  }

  auto* state = static_cast<DeviceState*>(handle);
  const std::uint32_t substeps = std::max(1u, requested_substeps);
  const double dt_myr = state->base_timestep_myr / static_cast<double>(substeps);
  return run_steps(
      state,
      substeps,
      dt_myr,
      state->base_timestep_myr,
      diagnostics,
      error_buffer,
      error_buffer_len);
}

extern "C" int sim_cuda_advance(void* handle,
                                std::uint32_t steps,
                                SimCudaDiagnostics* diagnostics,
                                char* error_buffer,
                                std::size_t error_buffer_len) {
  if (handle == nullptr) {
    fill_error(error_buffer, error_buffer_len, "null simulation handle");
    return 1;
  }

  auto* state = static_cast<DeviceState*>(handle);
  const std::uint32_t step_count = std::max(1u, steps);
  return run_steps(
      state,
      step_count,
      state->base_timestep_myr,
      state->base_timestep_myr * static_cast<double>(step_count),
      diagnostics,
      error_buffer,
      error_buffer_len);
}

extern "C" int sim_cuda_fill_preview(void* handle,
                                     std::uint32_t max_particles,
                                     SimCudaPreviewParticle* out_particles,
                                     std::uint32_t* out_count,
                                     char* error_buffer,
                                     std::size_t error_buffer_len) {
  if (handle == nullptr || out_particles == nullptr || out_count == nullptr) {
    fill_error(error_buffer, error_buffer_len, "invalid preview parameters");
    return 1;
  }

  auto* state = static_cast<DeviceState*>(handle);
  if (state->particle_count == 0 || max_particles == 0) {
    *out_count = 0;
    return 0;
  }

  const std::uint32_t count = std::min<std::uint32_t>(
      max_particles,
      static_cast<std::uint32_t>(std::min<std::uint64_t>(
          state->particle_count,
          static_cast<std::uint64_t>(std::numeric_limits<std::uint32_t>::max()))));
  const std::uint32_t anchor_count =
      std::min<std::uint32_t>(count, static_cast<std::uint32_t>(state->galaxy_smbh_indices_host.size()));
  const std::uint32_t sampled_count = count - anchor_count;
  if (ensure_preview_capacity(state, count, error_buffer, error_buffer_len) != 0) {
    return 1;
  }

  cudaError_t cuda_status = cudaSuccess;
  if (sampled_count > 0) {
    const std::uint64_t stride =
        std::max<std::uint64_t>(1, state->particle_count / std::max<std::uint32_t>(1, sampled_count));
    const int threads_per_block = 256;
    const int blocks = static_cast<int>((sampled_count + threads_per_block - 1) / threads_per_block);

    sample_preview<<<blocks, threads_per_block>>>(
        state->particles,
        state->particle_count,
        sampled_count,
        stride,
        state->preview_particles + anchor_count);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "preview kernel launch failed", cuda_status);
      return 1;
    }

    cuda_status = cudaMemcpy(
        out_particles + anchor_count,
        state->preview_particles + anchor_count,
        sizeof(SimCudaPreviewParticle) * sampled_count,
        cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "preview download failed", cuda_status);
      return 1;
    }
  }

  for (std::uint32_t i = 0; i < anchor_count; ++i) {
    SimCudaParticle particle{};
    cuda_status = cudaMemcpy(
        &particle,
        state->particles + state->galaxy_smbh_indices_host[i],
        sizeof(SimCudaParticle),
        cudaMemcpyDeviceToHost);
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "SMBH preview download failed", cuda_status);
      return 1;
    }
    out_particles[i] = preview_particle_from_host_particle(particle, 1.35f);
  }

  *out_count = count;
  return 0;
}

extern "C" int sim_cuda_copy_particles(void* handle,
                                       SimCudaParticle* out_particles,
                                       std::uint64_t particle_capacity,
                                       char* error_buffer,
                                       std::size_t error_buffer_len) {
  if (handle == nullptr || out_particles == nullptr) {
    fill_error(error_buffer, error_buffer_len, "invalid particle copy parameters");
    return 1;
  }

  auto* state = static_cast<DeviceState*>(handle);
  if (particle_capacity < state->particle_count) {
    fill_error(error_buffer, error_buffer_len, "destination buffer is smaller than particle count");
    return 1;
  }

  const auto bytes = sizeof(SimCudaParticle) * state->particle_count;
  const cudaError_t cuda_status =
      cudaMemcpy(out_particles, state->particles, bytes, cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "particle download failed", cuda_status);
    return 1;
  }

  return 0;
}
