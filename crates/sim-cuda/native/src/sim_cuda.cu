#include "sim_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

#include <cuda_runtime.h>

namespace {

constexpr double kKpcPerKmPerMyr = 0.001022712165045695;

struct DeviceState {
  std::uint64_t particle_count = 0;
  double grav_const = 0.0;
  double base_timestep_myr = 0.0;
  double sim_time_myr = 0.0;

  SimCudaParticle* particles = nullptr;
  std::vector<int> smbh_indices;
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

__device__ double softened_inv_r3(const double dx, const double dy, const double dz, const double softening) {
  const double r2 = dx * dx + dy * dy + dz * dz + softening * softening;
  const double inv_r = rsqrt(r2);
  return inv_r * inv_r * inv_r;
}

__global__ void step_particles(SimCudaParticle* particles,
                               const int* smbh_indices,
                               const int smbh_count,
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

  if (particle.component != 3u) {
    for (int smbh_i = 0; smbh_i < smbh_count; ++smbh_i) {
      const SimCudaParticle smbh = particles[smbh_indices[smbh_i]];
      const double dx = smbh.position_kpc[0] - particle.position_kpc[0];
      const double dy = smbh.position_kpc[1] - particle.position_kpc[1];
      const double dz = smbh.position_kpc[2] - particle.position_kpc[2];
      const double softening =
          particle.softening_kpc > smbh.softening_kpc ? particle.softening_kpc : smbh.softening_kpc;
      const double inv_r3 = softened_inv_r3(dx, dy, dz, softening);
      const double accel_scale = grav_const * smbh.mass_msun * inv_r3;
      ax += accel_scale * dx;
      ay += accel_scale * dy;
      az += accel_scale * dz;
    }
  }

  particle.velocity_kms[0] += ax * dt_myr * kKpcPerKmPerMyr;
  particle.velocity_kms[1] += ay * dt_myr * kKpcPerKmPerMyr;
  particle.velocity_kms[2] += az * dt_myr * kKpcPerKmPerMyr;

  particle.position_kpc[0] += particle.velocity_kms[0] * dt_myr * kKpcPerKmPerMyr;
  particle.position_kpc[1] += particle.velocity_kms[1] * dt_myr * kKpcPerKmPerMyr;
  particle.position_kpc[2] += particle.velocity_kms[2] * dt_myr * kKpcPerKmPerMyr;
}

int compute_diagnostics(const DeviceState& state, SimCudaDiagnostics* diagnostics) {
  if (diagnostics == nullptr) {
    return 1;
  }

  diagnostics->particle_count = state.particle_count;
  diagnostics->preview_count = 0;
  diagnostics->sim_time_myr = state.sim_time_myr;
  diagnostics->dt_myr = state.base_timestep_myr;

  double kinetic_energy = 0.0;
  double potential_energy = 0.0;
  double momentum[3] = {0.0, 0.0, 0.0};

  for (std::uint64_t i = 0; i < state.particle_count; ++i) {
    const SimCudaParticle& particle = state.particles[i];
    const double v2 = particle.velocity_kms[0] * particle.velocity_kms[0] +
                      particle.velocity_kms[1] * particle.velocity_kms[1] +
                      particle.velocity_kms[2] * particle.velocity_kms[2];
    kinetic_energy += 0.5 * particle.mass_msun * v2;
    momentum[0] += particle.mass_msun * particle.velocity_kms[0];
    momentum[1] += particle.mass_msun * particle.velocity_kms[1];
    momentum[2] += particle.mass_msun * particle.velocity_kms[2];

    if (particle.component != 3u) {
      for (const int smbh_index : state.smbh_indices) {
        const SimCudaParticle& smbh = state.particles[smbh_index];
        const double dx = smbh.position_kpc[0] - particle.position_kpc[0];
        const double dy = smbh.position_kpc[1] - particle.position_kpc[1];
        const double dz = smbh.position_kpc[2] - particle.position_kpc[2];
        const double radius = std::sqrt(dx * dx + dy * dy + dz * dz + particle.softening_kpc * particle.softening_kpc);
        potential_energy -= state.grav_const * particle.mass_msun * smbh.mass_msun / radius;
      }
    }
  }

  diagnostics->kinetic_energy = kinetic_energy;
  diagnostics->estimated_potential_energy = potential_energy;
  diagnostics->total_momentum[0] = momentum[0];
  diagnostics->total_momentum[1] = momentum[1];
  diagnostics->total_momentum[2] = momentum[2];
  return 0;
}

}  // namespace

extern "C" int sim_cuda_create(const SimCudaCreateParams* params,
                               const SimCudaParticle* particles,
                               void** out_handle,
                               char* error_buffer,
                               std::size_t error_buffer_len) {
  if (params == nullptr || particles == nullptr || out_handle == nullptr) {
    fill_error(error_buffer, error_buffer_len, "invalid create parameters");
    return 1;
  }

  auto state = std::make_unique<DeviceState>();
  state->particle_count = params->particle_count;
  state->grav_const = params->grav_const_kpc_kms2_per_msun;
  state->base_timestep_myr = params->base_timestep_myr;

  const auto bytes = sizeof(SimCudaParticle) * params->particle_count;
  cudaError_t cuda_status =
      cudaMallocManaged(reinterpret_cast<void**>(&state->particles), bytes);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMallocManaged failed", cuda_status);
    return 1;
  }

  std::memcpy(state->particles, particles, bytes);
  for (std::uint64_t i = 0; i < params->particle_count; ++i) {
    if (state->particles[i].component == 3u) {
      state->smbh_indices.push_back(static_cast<int>(i));
    }
  }

  *out_handle = state.release();
  return 0;
}

extern "C" int sim_cuda_destroy(void* handle) {
  if (handle == nullptr) {
    return 0;
  }
  auto* state = static_cast<DeviceState*>(handle);
  if (state->particles != nullptr) {
    cudaFree(state->particles);
  }
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
  const int threads_per_block = 256;
  const int blocks = static_cast<int>((state->particle_count + threads_per_block - 1) / threads_per_block);

  int* device_smbh_indices = nullptr;
  if (!state->smbh_indices.empty()) {
    const auto bytes = sizeof(int) * state->smbh_indices.size();
    cudaError_t cuda_status =
        cudaMalloc(reinterpret_cast<void**>(&device_smbh_indices), bytes);
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for SMBH index buffer failed", cuda_status);
      return 1;
    }
    cuda_status = cudaMemcpy(
        device_smbh_indices,
        state->smbh_indices.data(),
        bytes,
        cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
      cudaFree(device_smbh_indices);
      fill_cuda_error(error_buffer, error_buffer_len, "cudaMemcpy for SMBH index buffer failed", cuda_status);
      return 1;
    }
  }

  for (std::uint32_t step = 0; step < substeps; ++step) {
    step_particles<<<blocks, threads_per_block>>>(
        state->particles,
        device_smbh_indices,
        static_cast<int>(state->smbh_indices.size()),
        state->particle_count,
        state->grav_const,
        dt_myr);
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
      if (device_smbh_indices != nullptr) {
        cudaFree(device_smbh_indices);
      }
      fill_cuda_error(error_buffer, error_buffer_len, "kernel launch failed", cuda_status);
      return 1;
    }
  }

  cudaError_t cuda_status = cudaDeviceSynchronize();
  if (device_smbh_indices != nullptr) {
    cudaFree(device_smbh_indices);
  }
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "device synchronization failed", cuda_status);
    return 1;
  }

  state->sim_time_myr += state->base_timestep_myr;
  return compute_diagnostics(*state, diagnostics);
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
  if (state->particle_count == 0) {
    *out_count = 0;
    return 0;
  }

  const std::uint32_t count = std::min<std::uint32_t>(
      max_particles,
      static_cast<std::uint32_t>(std::min<std::uint64_t>(state->particle_count, static_cast<std::uint64_t>(UINT32_MAX))));
  const std::uint64_t stride = std::max<std::uint64_t>(1, state->particle_count / std::max<std::uint32_t>(1, count));

  std::uint32_t cursor = 0;
  for (std::uint64_t i = 0; i < state->particle_count && cursor < count; i += stride) {
    const SimCudaParticle& particle = state->particles[i];
    const float speed = static_cast<float>(std::sqrt(
        particle.velocity_kms[0] * particle.velocity_kms[0] +
        particle.velocity_kms[1] * particle.velocity_kms[1] +
        particle.velocity_kms[2] * particle.velocity_kms[2]));

    out_particles[cursor].position_kpc[0] = static_cast<float>(particle.position_kpc[0]);
    out_particles[cursor].position_kpc[1] = static_cast<float>(particle.position_kpc[1]);
    out_particles[cursor].position_kpc[2] = static_cast<float>(particle.position_kpc[2]);
    out_particles[cursor].velocity_kms[0] = static_cast<float>(particle.velocity_kms[0]);
    out_particles[cursor].velocity_kms[1] = static_cast<float>(particle.velocity_kms[1]);
    out_particles[cursor].velocity_kms[2] = static_cast<float>(particle.velocity_kms[2]);
    std::memcpy(out_particles[cursor].color_rgba, particle.color_rgba, sizeof(float) * 4);
    out_particles[cursor].intensity = 0.35f + std::min(1.0f, speed / 320.0f);
    ++cursor;
  }

  *out_count = cursor;
  (void)error_buffer;
  (void)error_buffer_len;
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

  cudaError_t cuda_status = cudaDeviceSynchronize();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "device synchronization failed", cuda_status);
    return 1;
  }

  std::memcpy(out_particles, state->particles, sizeof(SimCudaParticle) * state->particle_count);
  return 0;
}
