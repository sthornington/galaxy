#include "sim_cuda.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>

namespace {

constexpr double kKpcPerKmPerMyr = 0.001022712165045695;
constexpr double kSpeedOfLightKms = 299792.458;
constexpr double kFourPi = 12.566370614359172;
constexpr double kPi = 3.14159265358979323846;
constexpr double kGlobalDomainPadding = 2.25;
constexpr double kMinGlobalBoxLengthKpc = 64.0;
constexpr int kLocalFineNx = 128;
constexpr int kLocalFineNy = 128;
constexpr int kLocalFineNz = 64;
constexpr int kLocalCoarseNx = 32;
constexpr int kLocalCoarseNy = 32;
constexpr int kLocalCoarseNz = 16;
constexpr double kLocalCorrectionBlendExtent = 0.78;
constexpr int kMaxRefinementPatches = 8;
constexpr int kMaxDensityTreeDepth = 5;
constexpr int kMinRefineCellsXY = 8;
constexpr int kMinRefineCellsZ = 4;
constexpr double kRefineSplitDensityRatio = 0.12;
constexpr double kRefineLeafDensityRatio = 0.045;
constexpr double kRefineMinMassFraction = 2.0e-4;

struct MeshBuffers {
  int nx = 0;
  int ny = 0;
  int nz = 0;
  int nz_complex = 0;
  std::size_t real_count = 0;
  std::size_t complex_count = 0;

  cufftReal* density_grid = nullptr;
  cufftComplex* density_k = nullptr;
  cufftComplex* force_k = nullptr;
  cufftReal* force_x = nullptr;
  cufftReal* force_y = nullptr;
  cufftReal* force_z = nullptr;

  cufftHandle forward_plan = 0;
  cufftHandle inverse_plan = 0;
};

struct RefinementPatchState {
  bool active = false;
  int grid_min[3] = {0, 0, 0};
  int grid_max[3] = {0, 0, 0};
  double box_length[3] = {1.0, 1.0, 1.0};
  double domain_origin[3] = {0.0, 0.0, 0.0};
  double cell_size_fine[3] = {1.0, 1.0, 1.0};
  double cell_size_coarse[3] = {1.0, 1.0, 1.0};
  MeshBuffers fine;
  MeshBuffers coarse;
};

struct DensityRegionSummary {
  double mass_msun = 0.0;
  double max_density = 0.0;
};

struct DensityTreeCandidate {
  int grid_min[3] = {0, 0, 0};
  int grid_max[3] = {0, 0, 0};
  double mass_msun = 0.0;
  double max_density = 0.0;
  int depth = 0;
  double score = 0.0;
};

struct DeviceState {
  std::uint64_t particle_count = 0;
  std::uint32_t galaxy_count = 0;
  double grav_const = 0.0;
  double base_timestep_myr = 0.0;
  double sim_time_myr = 0.0;
  bool enable_smbh_post_newtonian = false;

  int nx = 0;
  int ny = 0;
  int nz = 0;
  int nz_complex = 0;
  std::size_t real_count = 0;
  std::size_t complex_count = 0;

  double domain_origin[3] = {0.0, 0.0, 0.0};
  double box_length[3] = {1.0, 1.0, 1.0};
  double cell_size[3] = {1.0, 1.0, 1.0};
  double cell_volume = 1.0;

  SimCudaParticle* particles = nullptr;
  int* galaxy_smbh_indices = nullptr;
  std::vector<int> galaxy_smbh_indices_host;
  std::vector<cufftReal> density_host;
  std::vector<RefinementPatchState> refinement_patches;

  cufftReal* density_grid = nullptr;
  cufftComplex* density_k = nullptr;
  cufftComplex* force_k = nullptr;
  cufftReal* force_x = nullptr;
  cufftReal* force_y = nullptr;
  cufftReal* force_z = nullptr;

  cufftHandle forward_plan = 0;
  cufftHandle inverse_plan = 0;

  SimCudaPreviewParticle* preview_particles = nullptr;
  std::uint32_t preview_capacity = 0;
};

const char* cufft_error_string(const cufftResult status) {
  switch (status) {
    case CUFFT_SUCCESS:
      return "CUFFT_SUCCESS";
    case CUFFT_INVALID_PLAN:
      return "CUFFT_INVALID_PLAN";
    case CUFFT_ALLOC_FAILED:
      return "CUFFT_ALLOC_FAILED";
    case CUFFT_INVALID_TYPE:
      return "CUFFT_INVALID_TYPE";
    case CUFFT_INVALID_VALUE:
      return "CUFFT_INVALID_VALUE";
    case CUFFT_INTERNAL_ERROR:
      return "CUFFT_INTERNAL_ERROR";
    case CUFFT_EXEC_FAILED:
      return "CUFFT_EXEC_FAILED";
    case CUFFT_SETUP_FAILED:
      return "CUFFT_SETUP_FAILED";
    case CUFFT_INVALID_SIZE:
      return "CUFFT_INVALID_SIZE";
    case CUFFT_UNALIGNED_DATA:
      return "CUFFT_UNALIGNED_DATA";
    case CUFFT_INVALID_DEVICE:
      return "CUFFT_INVALID_DEVICE";
    case CUFFT_NO_WORKSPACE:
      return "CUFFT_NO_WORKSPACE";
    case CUFFT_NOT_IMPLEMENTED:
      return "CUFFT_NOT_IMPLEMENTED";
    case CUFFT_NOT_SUPPORTED:
      return "CUFFT_NOT_SUPPORTED";
    default:
      return "CUFFT_UNKNOWN_ERROR";
  }
}

void fill_error(char* buffer, const std::size_t len, const char* message) {
  if (buffer == nullptr || len == 0) {
    return;
  }
  std::snprintf(buffer, len, "%s", message);
}

void fill_cuda_error(char* buffer,
                     const std::size_t len,
                     const char* context,
                     const cudaError_t err) {
  if (buffer == nullptr || len == 0) {
    return;
  }
  std::snprintf(buffer, len, "%s: %s", context, cudaGetErrorString(err));
}

void fill_cufft_error(char* buffer,
                      const std::size_t len,
                      const char* context,
                      const cufftResult err) {
  if (buffer == nullptr || len == 0) {
    return;
  }
  std::snprintf(buffer, len, "%s: %s", context, cufft_error_string(err));
}

void destroy_mesh_buffers(MeshBuffers& mesh) {
  if (mesh.forward_plan != 0) {
    cufftDestroy(mesh.forward_plan);
    mesh.forward_plan = 0;
  }
  if (mesh.inverse_plan != 0) {
    cufftDestroy(mesh.inverse_plan);
    mesh.inverse_plan = 0;
  }
  cudaFree(mesh.force_z);
  cudaFree(mesh.force_y);
  cudaFree(mesh.force_x);
  cudaFree(mesh.force_k);
  cudaFree(mesh.density_k);
  cudaFree(mesh.density_grid);
  mesh.force_z = nullptr;
  mesh.force_y = nullptr;
  mesh.force_x = nullptr;
  mesh.force_k = nullptr;
  mesh.density_k = nullptr;
  mesh.density_grid = nullptr;
}

int create_mesh_buffers(MeshBuffers& mesh,
                        const int nx,
                        const int ny,
                        const int nz,
                        char* error_buffer,
                        const std::size_t error_buffer_len) {
  mesh.nx = nx;
  mesh.ny = ny;
  mesh.nz = nz;
  mesh.nz_complex = nz / 2 + 1;
  mesh.real_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                    static_cast<std::size_t>(nz);
  mesh.complex_count = static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
                       static_cast<std::size_t>(mesh.nz_complex);

  cudaError_t cuda_status =
      cudaMalloc(reinterpret_cast<void**>(&mesh.density_grid), sizeof(cufftReal) * mesh.real_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for local density grid failed", cuda_status);
    return 1;
  }

  cuda_status =
      cudaMalloc(reinterpret_cast<void**>(&mesh.density_k), sizeof(cufftComplex) * mesh.complex_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for local density spectrum failed", cuda_status);
    return 1;
  }

  cuda_status =
      cudaMalloc(reinterpret_cast<void**>(&mesh.force_k), sizeof(cufftComplex) * mesh.complex_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for local force spectrum failed", cuda_status);
    return 1;
  }

  cuda_status = cudaMalloc(reinterpret_cast<void**>(&mesh.force_x), sizeof(cufftReal) * mesh.real_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for local force_x failed", cuda_status);
    return 1;
  }

  cuda_status = cudaMalloc(reinterpret_cast<void**>(&mesh.force_y), sizeof(cufftReal) * mesh.real_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for local force_y failed", cuda_status);
    return 1;
  }

  cuda_status = cudaMalloc(reinterpret_cast<void**>(&mesh.force_z), sizeof(cufftReal) * mesh.real_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for local force_z failed", cuda_status);
    return 1;
  }

  cufftResult fft_status = cufftPlan3d(&mesh.forward_plan, nx, ny, nz, CUFFT_R2C);
  if (fft_status != CUFFT_SUCCESS) {
    fill_cufft_error(error_buffer, error_buffer_len, "local forward FFT plan creation failed", fft_status);
    return 1;
  }

  fft_status = cufftPlan3d(&mesh.inverse_plan, nx, ny, nz, CUFFT_C2R);
  if (fft_status != CUFFT_SUCCESS) {
    fill_cufft_error(error_buffer, error_buffer_len, "local inverse FFT plan creation failed", fft_status);
    return 1;
  }

  return 0;
}

void destroy_state(DeviceState* state) {
  if (state == nullptr) {
    return;
  }

  for (auto& patch : state->refinement_patches) {
    destroy_mesh_buffers(patch.fine);
    destroy_mesh_buffers(patch.coarse);
  }
  if (state->forward_plan != 0) {
    cufftDestroy(state->forward_plan);
  }
  if (state->inverse_plan != 0) {
    cufftDestroy(state->inverse_plan);
  }
  cudaFree(state->preview_particles);
  cudaFree(state->force_z);
  cudaFree(state->force_y);
  cudaFree(state->force_x);
  cudaFree(state->force_k);
  cudaFree(state->density_k);
  cudaFree(state->density_grid);
  cudaFree(state->galaxy_smbh_indices);
  cudaFree(state->particles);
  delete state;
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

  const cudaError_t cuda_status = cudaMalloc(
      reinterpret_cast<void**>(&state->preview_particles),
      sizeof(SimCudaPreviewParticle) * count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(
        error_buffer, error_buffer_len, "cudaMalloc for preview buffer failed", cuda_status);
    return 1;
  }

  state->preview_capacity = count;
  return 0;
}

void compute_simulation_domain(DeviceState* state, const SimCudaParticle* particles) {
  double min_pos[3] = {
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
  };
  double max_pos[3] = {
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
  };

  for (std::uint64_t i = 0; i < state->particle_count; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      min_pos[axis] = std::min(min_pos[axis], particles[i].position_kpc[axis]);
      max_pos[axis] = std::max(max_pos[axis], particles[i].position_kpc[axis]);
    }
  }

  for (int axis = 0; axis < 3; ++axis) {
    const double range = std::max(1.0, max_pos[axis] - min_pos[axis]);
    const double center = 0.5 * (min_pos[axis] + max_pos[axis]);
    const double padded_length = std::max(kMinGlobalBoxLengthKpc, range * kGlobalDomainPadding);
    state->box_length[axis] = padded_length;
    state->domain_origin[axis] = center - 0.5 * padded_length;
  }

  state->cell_size[0] = state->box_length[0] / static_cast<double>(state->nx);
  state->cell_size[1] = state->box_length[1] / static_cast<double>(state->ny);
  state->cell_size[2] = state->box_length[2] / static_cast<double>(state->nz);
  state->cell_volume = state->cell_size[0] * state->cell_size[1] * state->cell_size[2];
}

__device__ __forceinline__ int wrap_index(int index, const int dim) {
  index %= dim;
  if (index < 0) {
    index += dim;
  }
  return index;
}

__device__ __forceinline__ int clamp_index(int index, const int dim) {
  return max(0, min(dim - 1, index));
}

__host__ __device__ __forceinline__ std::size_t real_grid_index(
    const int x, const int y, const int z, const int ny, const int nz) {
  return (static_cast<std::size_t>(x) * static_cast<std::size_t>(ny) +
          static_cast<std::size_t>(y)) *
             static_cast<std::size_t>(nz) +
         static_cast<std::size_t>(z);
}

__host__ __device__ __forceinline__ std::size_t complex_grid_index(
    const int x, const int y, const int z, const int ny, const int nz_complex) {
  return (static_cast<std::size_t>(x) * static_cast<std::size_t>(ny) +
          static_cast<std::size_t>(y)) *
             static_cast<std::size_t>(nz_complex) +
         static_cast<std::size_t>(z);
}

__device__ __forceinline__ double wrap_position(
    const double value, const double origin, const double length) {
  double shifted = value - origin;
  shifted -= floor(shifted / length) * length;
  return shifted;
}

bool region_has_refinement_extent(const int grid_min[3], const int grid_max[3]) {
  return (grid_max[0] - grid_min[0] > kMinRefineCellsXY) &&
         (grid_max[1] - grid_min[1] > kMinRefineCellsXY) &&
         (grid_max[2] - grid_min[2] > kMinRefineCellsZ);
}

DensityRegionSummary summarize_density_region(const DeviceState& state,
                                              const int grid_min[3],
                                              const int grid_max[3]) {
  DensityRegionSummary summary;
  for (int ix = grid_min[0]; ix < grid_max[0]; ++ix) {
    for (int iy = grid_min[1]; iy < grid_max[1]; ++iy) {
      for (int iz = grid_min[2]; iz < grid_max[2]; ++iz) {
        const double density = state.density_host[real_grid_index(ix, iy, iz, state.ny, state.nz)];
        if (density <= 0.0) {
          continue;
        }
        summary.mass_msun += density * state.cell_volume;
        summary.max_density = std::max(summary.max_density, density);
      }
    }
  }
  return summary;
}

void collect_density_tree_candidates(const DeviceState& state,
                                     const int grid_min[3],
                                     const int grid_max[3],
                                     const DensityRegionSummary& summary,
                                     const double root_mass_msun,
                                     const double root_max_density,
                                     const int depth,
                                     std::vector<DensityTreeCandidate>& out_candidates) {
  if (summary.mass_msun <= root_mass_msun * kRefineMinMassFraction ||
      summary.max_density <= root_max_density * kRefineLeafDensityRatio) {
    return;
  }

  const bool can_split = depth < kMaxDensityTreeDepth && region_has_refinement_extent(grid_min, grid_max) &&
                         summary.max_density > root_max_density * kRefineSplitDensityRatio;
  if (!can_split) {
    if (depth == 0) {
      return;
    }
    DensityTreeCandidate candidate;
    for (int axis = 0; axis < 3; ++axis) {
      candidate.grid_min[axis] = grid_min[axis];
      candidate.grid_max[axis] = grid_max[axis];
    }
    candidate.mass_msun = summary.mass_msun;
    candidate.max_density = summary.max_density;
    candidate.depth = depth;
    candidate.score = candidate.max_density * std::sqrt(std::max(candidate.mass_msun, 1.0));
    out_candidates.push_back(candidate);
    return;
  }

  const int mid[3] = {
      (grid_min[0] + grid_max[0]) / 2,
      (grid_min[1] + grid_max[1]) / 2,
      (grid_min[2] + grid_max[2]) / 2,
  };
  bool emitted_child = false;
  for (int octant = 0; octant < 8; ++octant) {
    const int child_min[3] = {
        (octant & 1) == 0 ? grid_min[0] : mid[0],
        (octant & 2) == 0 ? grid_min[1] : mid[1],
        (octant & 4) == 0 ? grid_min[2] : mid[2],
    };
    const int child_max[3] = {
        (octant & 1) == 0 ? mid[0] : grid_max[0],
        (octant & 2) == 0 ? mid[1] : grid_max[1],
        (octant & 4) == 0 ? mid[2] : grid_max[2],
    };
    if (child_min[0] >= child_max[0] || child_min[1] >= child_max[1] || child_min[2] >= child_max[2]) {
      continue;
    }
    const DensityRegionSummary child_summary = summarize_density_region(state, child_min, child_max);
    if (child_summary.mass_msun <= 0.0) {
      continue;
    }
    emitted_child = true;
    collect_density_tree_candidates(
        state,
        child_min,
        child_max,
        child_summary,
        root_mass_msun,
        root_max_density,
        depth + 1,
        out_candidates);
  }

  if (!emitted_child && depth > 0) {
    DensityTreeCandidate candidate;
    for (int axis = 0; axis < 3; ++axis) {
      candidate.grid_min[axis] = grid_min[axis];
      candidate.grid_max[axis] = grid_max[axis];
    }
    candidate.mass_msun = summary.mass_msun;
    candidate.max_density = summary.max_density;
    candidate.depth = depth;
    candidate.score = candidate.max_density * std::sqrt(std::max(candidate.mass_msun, 1.0));
    out_candidates.push_back(candidate);
  }
}

bool grid_regions_overlap(const int a_min[3], const int a_max[3], const int b_min[3], const int b_max[3]) {
  for (int axis = 0; axis < 3; ++axis) {
    if (a_max[axis] <= b_min[axis] || b_max[axis] <= a_min[axis]) {
      return false;
    }
  }
  return true;
}

void initialize_refinement_patch_geometry(const DeviceState& state,
                                          RefinementPatchState& patch,
                                          const DensityTreeCandidate& candidate) {
  patch.active = true;
  for (int axis = 0; axis < 3; ++axis) {
    const int width = candidate.grid_max[axis] - candidate.grid_min[axis];
    const int pad = std::max(2, width / 2);
    patch.grid_min[axis] = std::max(0, candidate.grid_min[axis] - pad);
    const int limit = axis == 0 ? state.nx : (axis == 1 ? state.ny : state.nz);
    patch.grid_max[axis] = std::min(limit, candidate.grid_max[axis] + pad);
    patch.domain_origin[axis] =
        state.domain_origin[axis] + static_cast<double>(patch.grid_min[axis]) * state.cell_size[axis];
    patch.box_length[axis] =
        static_cast<double>(patch.grid_max[axis] - patch.grid_min[axis]) * state.cell_size[axis];
  }
  patch.cell_size_fine[0] = patch.box_length[0] / static_cast<double>(patch.fine.nx);
  patch.cell_size_fine[1] = patch.box_length[1] / static_cast<double>(patch.fine.ny);
  patch.cell_size_fine[2] = patch.box_length[2] / static_cast<double>(patch.fine.nz);
  patch.cell_size_coarse[0] = patch.box_length[0] / static_cast<double>(patch.coarse.nx);
  patch.cell_size_coarse[1] = patch.box_length[1] / static_cast<double>(patch.coarse.ny);
  patch.cell_size_coarse[2] = patch.box_length[2] / static_cast<double>(patch.coarse.nz);
}

int update_refinement_patches(DeviceState* state,
                              char* error_buffer,
                              const std::size_t error_buffer_len) {
  if (state->density_host.size() != state->real_count) {
    state->density_host.resize(state->real_count);
  }
  const cudaError_t cuda_status = cudaMemcpy(state->density_host.data(),
                                             state->density_grid,
                                             sizeof(cufftReal) * state->real_count,
                                             cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "density download for refinement failed", cuda_status);
    return 1;
  }

  const int root_min[3] = {0, 0, 0};
  const int root_max[3] = {state->nx, state->ny, state->nz};
  const DensityRegionSummary root_summary = summarize_density_region(*state, root_min, root_max);

  for (auto& patch : state->refinement_patches) {
    patch.active = false;
  }
  if (root_summary.mass_msun <= 0.0 || root_summary.max_density <= 0.0) {
    return 0;
  }

  std::vector<DensityTreeCandidate> candidates;
  candidates.reserve(64);
  collect_density_tree_candidates(
      *state,
      root_min,
      root_max,
      root_summary,
      root_summary.mass_msun,
      root_summary.max_density,
      0,
      candidates);
  std::sort(candidates.begin(), candidates.end(), [](const DensityTreeCandidate& lhs, const DensityTreeCandidate& rhs) {
    if (lhs.depth != rhs.depth) {
      return lhs.depth > rhs.depth;
    }
    return lhs.score > rhs.score;
  });

  std::vector<DensityTreeCandidate> selected;
  selected.reserve(state->refinement_patches.size());
  for (const auto& candidate : candidates) {
    int padded_min[3];
    int padded_max[3];
    for (int axis = 0; axis < 3; ++axis) {
      const int width = candidate.grid_max[axis] - candidate.grid_min[axis];
      const int pad = std::max(2, width / 2);
      const int limit = axis == 0 ? state->nx : (axis == 1 ? state->ny : state->nz);
      padded_min[axis] = std::max(0, candidate.grid_min[axis] - pad);
      padded_max[axis] = std::min(limit, candidate.grid_max[axis] + pad);
    }
    bool overlaps = false;
    for (const auto& prior : selected) {
      int prior_min[3];
      int prior_max[3];
      for (int axis = 0; axis < 3; ++axis) {
        const int width = prior.grid_max[axis] - prior.grid_min[axis];
        const int pad = std::max(2, width / 2);
        const int limit = axis == 0 ? state->nx : (axis == 1 ? state->ny : state->nz);
        prior_min[axis] = std::max(0, prior.grid_min[axis] - pad);
        prior_max[axis] = std::min(limit, prior.grid_max[axis] + pad);
      }
      if (grid_regions_overlap(padded_min, padded_max, prior_min, prior_max)) {
        overlaps = true;
        break;
      }
    }
    if (overlaps) {
      continue;
    }
    selected.push_back(candidate);
    if (selected.size() >= state->refinement_patches.size()) {
      break;
    }
  }

  for (std::size_t i = 0; i < selected.size(); ++i) {
    initialize_refinement_patch_geometry(*state, state->refinement_patches[i], selected[i]);
  }
  return 0;
}

__device__ __forceinline__ float sinc_pi(const float x) {
  if (fabsf(x) <= 1.0e-5f) {
    return 1.0f;
  }
  return sinf(x) / x;
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

__device__ __forceinline__ void add_smbh_1pn_acceleration(double& ax,
                                                          double& ay,
                                                          double& az,
                                                          const SimCudaParticle& particle,
                                                          const SimCudaParticle& source,
                                                          const double grav_const) {
  if (source.mass_msun <= 0.0) {
    return;
  }

  const double rx = particle.position_kpc[0] - source.position_kpc[0];
  const double ry = particle.position_kpc[1] - source.position_kpc[1];
  const double rz = particle.position_kpc[2] - source.position_kpc[2];
  const double softening = fmax(particle.softening_kpc, source.softening_kpc);
  const double r2 = rx * rx + ry * ry + rz * rz + softening * softening;
  const double r = sqrt(r2);
  if (r <= 1.0e-6) {
    return;
  }

  const double vx = particle.velocity_kms[0] - source.velocity_kms[0];
  const double vy = particle.velocity_kms[1] - source.velocity_kms[1];
  const double vz = particle.velocity_kms[2] - source.velocity_kms[2];
  const double v2 = vx * vx + vy * vy + vz * vz;
  const double rv = rx * vx + ry * vy + rz * vz;
  const double gm = grav_const * source.mass_msun;
  const double coeff = gm / (kSpeedOfLightKms * kSpeedOfLightKms * r * r * r);
  const double radial_scale = 4.0 * gm / r - v2;

  ax += coeff * (-radial_scale * rx + 4.0 * rv * vx);
  ay += coeff * (-radial_scale * ry + 4.0 * rv * vy);
  az += coeff * (-radial_scale * rz + 4.0 * rv * vz);
}

__device__ __forceinline__ double sample_grid_trilinear(const cufftReal* grid,
                                                        const int nx,
                                                        const int ny,
                                                        const int nz,
                                                        const double origin_x,
                                                        const double origin_y,
                                                        const double origin_z,
                                                        const double length_x,
                                                        const double length_y,
                                                        const double length_z,
                                                        const double cell_x,
                                                        const double cell_y,
                                                        const double cell_z,
                                                        const double px,
                                                        const double py,
                                                        const double pz) {
  const double wrapped_x = wrap_position(px, origin_x, length_x);
  const double wrapped_y = wrap_position(py, origin_y, length_y);
  const double wrapped_z = wrap_position(pz, origin_z, length_z);

  const double gx = wrapped_x / cell_x;
  const double gy = wrapped_y / cell_y;
  const double gz = wrapped_z / cell_z;

  const int i0 = static_cast<int>(floor(gx));
  const int j0 = static_cast<int>(floor(gy));
  const int k0 = static_cast<int>(floor(gz));
  const int i1 = wrap_index(i0 + 1, nx);
  const int j1 = wrap_index(j0 + 1, ny);
  const int k1 = wrap_index(k0 + 1, nz);
  const int wi0 = wrap_index(i0, nx);
  const int wj0 = wrap_index(j0, ny);
  const int wk0 = wrap_index(k0, nz);

  const double tx = gx - floor(gx);
  const double ty = gy - floor(gy);
  const double tz = gz - floor(gz);
  const double wx0 = 1.0 - tx;
  const double wy0 = 1.0 - ty;
  const double wz0 = 1.0 - tz;
  const double wx1 = tx;
  const double wy1 = ty;
  const double wz1 = tz;

  const double c000 =
      grid[real_grid_index(wi0, wj0, wk0, ny, nz)] * wx0 * wy0 * wz0;
  const double c001 =
      grid[real_grid_index(wi0, wj0, k1, ny, nz)] * wx0 * wy0 * wz1;
  const double c010 =
      grid[real_grid_index(wi0, j1, wk0, ny, nz)] * wx0 * wy1 * wz0;
  const double c011 =
      grid[real_grid_index(wi0, j1, k1, ny, nz)] * wx0 * wy1 * wz1;
  const double c100 =
      grid[real_grid_index(i1, wj0, wk0, ny, nz)] * wx1 * wy0 * wz0;
  const double c101 =
      grid[real_grid_index(i1, wj0, k1, ny, nz)] * wx1 * wy0 * wz1;
  const double c110 =
      grid[real_grid_index(i1, j1, wk0, ny, nz)] * wx1 * wy1 * wz0;
  const double c111 =
      grid[real_grid_index(i1, j1, k1, ny, nz)] * wx1 * wy1 * wz1;

  return c000 + c001 + c010 + c011 + c100 + c101 + c110 + c111;
}

__device__ __forceinline__ double sample_grid_trilinear_local(const cufftReal* grid,
                                                              const int nx,
                                                              const int ny,
                                                              const int nz,
                                                              const double origin_x,
                                                              const double origin_y,
                                                              const double origin_z,
                                                              const double cell_x,
                                                              const double cell_y,
                                                              const double cell_z,
                                                              const double px,
                                                              const double py,
                                                              const double pz) {
  const double gx = (px - origin_x) / cell_x;
  const double gy = (py - origin_y) / cell_y;
  const double gz = (pz - origin_z) / cell_z;
  if (gx < 0.0 || gy < 0.0 || gz < 0.0 ||
      gx >= static_cast<double>(nx - 1) ||
      gy >= static_cast<double>(ny - 1) ||
      gz >= static_cast<double>(nz - 1)) {
    return 0.0;
  }

  const int i0 = static_cast<int>(floor(gx));
  const int j0 = static_cast<int>(floor(gy));
  const int k0 = static_cast<int>(floor(gz));
  const int i1 = i0 + 1;
  const int j1 = j0 + 1;
  const int k1 = k0 + 1;

  const double tx = gx - floor(gx);
  const double ty = gy - floor(gy);
  const double tz = gz - floor(gz);
  const double wx0 = 1.0 - tx;
  const double wy0 = 1.0 - ty;
  const double wz0 = 1.0 - tz;
  const double wx1 = tx;
  const double wy1 = ty;
  const double wz1 = tz;

  const double c000 =
      grid[real_grid_index(i0, j0, k0, ny, nz)] * wx0 * wy0 * wz0;
  const double c001 =
      grid[real_grid_index(i0, j0, k1, ny, nz)] * wx0 * wy0 * wz1;
  const double c010 =
      grid[real_grid_index(i0, j1, k0, ny, nz)] * wx0 * wy1 * wz0;
  const double c011 =
      grid[real_grid_index(i0, j1, k1, ny, nz)] * wx0 * wy1 * wz1;
  const double c100 =
      grid[real_grid_index(i1, j0, k0, ny, nz)] * wx1 * wy0 * wz0;
  const double c101 =
      grid[real_grid_index(i1, j0, k1, ny, nz)] * wx1 * wy0 * wz1;
  const double c110 =
      grid[real_grid_index(i1, j1, k0, ny, nz)] * wx1 * wy1 * wz0;
  const double c111 =
      grid[real_grid_index(i1, j1, k1, ny, nz)] * wx1 * wy1 * wz1;

  return c000 + c001 + c010 + c011 + c100 + c101 + c110 + c111;
}

__global__ void deposit_mass_cic(const SimCudaParticle* particles,
                                 const std::uint64_t particle_count,
                                 cufftReal* density_grid,
                                 const int nx,
                                 const int ny,
                                 const int nz,
                                 const double origin_x,
                                 const double origin_y,
                                 const double origin_z,
                                 const double length_x,
                                 const double length_y,
                                 const double length_z,
                                 const double cell_x,
                                 const double cell_y,
                                 const double cell_z,
                                 const float inv_cell_volume) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= particle_count) {
    return;
  }

  const SimCudaParticle& particle = particles[index];
  if (particle.component == 3u || particle.mass_msun <= 0.0) {
    return;
  }

  const double wrapped_x = wrap_position(particle.position_kpc[0], origin_x, length_x);
  const double wrapped_y = wrap_position(particle.position_kpc[1], origin_y, length_y);
  const double wrapped_z = wrap_position(particle.position_kpc[2], origin_z, length_z);

  const double gx = wrapped_x / cell_x;
  const double gy = wrapped_y / cell_y;
  const double gz = wrapped_z / cell_z;

  const int i0 = static_cast<int>(floor(gx));
  const int j0 = static_cast<int>(floor(gy));
  const int k0 = static_cast<int>(floor(gz));
  const int i1 = wrap_index(i0 + 1, nx);
  const int j1 = wrap_index(j0 + 1, ny);
  const int k1 = wrap_index(k0 + 1, nz);
  const int wi0 = wrap_index(i0, nx);
  const int wj0 = wrap_index(j0, ny);
  const int wk0 = wrap_index(k0, nz);

  const float tx = static_cast<float>(gx - floor(gx));
  const float ty = static_cast<float>(gy - floor(gy));
  const float tz = static_cast<float>(gz - floor(gz));
  const float wx0 = 1.0f - tx;
  const float wy0 = 1.0f - ty;
  const float wz0 = 1.0f - tz;
  const float wx1 = tx;
  const float wy1 = ty;
  const float wz1 = tz;
  const float mass_density = static_cast<float>(particle.mass_msun) * inv_cell_volume;

  atomicAdd(
      &density_grid[real_grid_index(wi0, wj0, wk0, ny, nz)],
      mass_density * wx0 * wy0 * wz0);
  atomicAdd(
      &density_grid[real_grid_index(wi0, wj0, k1, ny, nz)],
      mass_density * wx0 * wy0 * wz1);
  atomicAdd(
      &density_grid[real_grid_index(wi0, j1, wk0, ny, nz)],
      mass_density * wx0 * wy1 * wz0);
  atomicAdd(
      &density_grid[real_grid_index(wi0, j1, k1, ny, nz)],
      mass_density * wx0 * wy1 * wz1);
  atomicAdd(
      &density_grid[real_grid_index(i1, wj0, wk0, ny, nz)],
      mass_density * wx1 * wy0 * wz0);
  atomicAdd(
      &density_grid[real_grid_index(i1, wj0, k1, ny, nz)],
      mass_density * wx1 * wy0 * wz1);
  atomicAdd(
      &density_grid[real_grid_index(i1, j1, wk0, ny, nz)],
      mass_density * wx1 * wy1 * wz0);
  atomicAdd(
      &density_grid[real_grid_index(i1, j1, k1, ny, nz)],
      mass_density * wx1 * wy1 * wz1);
}

__global__ void deposit_mass_cic_local(const SimCudaParticle* particles,
                                       const std::uint64_t particle_count,
                                       cufftReal* density_grid,
                                       const int nx,
                                       const int ny,
                                       const int nz,
                                       const double origin_x,
                                       const double origin_y,
                                       const double origin_z,
                                       const double cell_x,
                                       const double cell_y,
                                       const double cell_z,
                                       const float inv_cell_volume) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= particle_count) {
    return;
  }

  const SimCudaParticle& particle = particles[index];
  if (particle.component == 3u || particle.mass_msun <= 0.0) {
    return;
  }

  const double gx = (particle.position_kpc[0] - origin_x) / cell_x;
  const double gy = (particle.position_kpc[1] - origin_y) / cell_y;
  const double gz = (particle.position_kpc[2] - origin_z) / cell_z;
  if (gx < 0.0 || gy < 0.0 || gz < 0.0 ||
      gx >= static_cast<double>(nx - 1) ||
      gy >= static_cast<double>(ny - 1) ||
      gz >= static_cast<double>(nz - 1)) {
    return;
  }

  const int i0 = static_cast<int>(floor(gx));
  const int j0 = static_cast<int>(floor(gy));
  const int k0 = static_cast<int>(floor(gz));
  const int i1 = i0 + 1;
  const int j1 = j0 + 1;
  const int k1 = k0 + 1;

  const float tx = static_cast<float>(gx - floor(gx));
  const float ty = static_cast<float>(gy - floor(gy));
  const float tz = static_cast<float>(gz - floor(gz));
  const float wx0 = 1.0f - tx;
  const float wy0 = 1.0f - ty;
  const float wz0 = 1.0f - tz;
  const float wx1 = tx;
  const float wy1 = ty;
  const float wz1 = tz;
  const float mass_density = static_cast<float>(particle.mass_msun) * inv_cell_volume;

  atomicAdd(
      &density_grid[real_grid_index(i0, j0, k0, ny, nz)],
      mass_density * wx0 * wy0 * wz0);
  atomicAdd(
      &density_grid[real_grid_index(i0, j0, k1, ny, nz)],
      mass_density * wx0 * wy0 * wz1);
  atomicAdd(
      &density_grid[real_grid_index(i0, j1, k0, ny, nz)],
      mass_density * wx0 * wy1 * wz0);
  atomicAdd(
      &density_grid[real_grid_index(i0, j1, k1, ny, nz)],
      mass_density * wx0 * wy1 * wz1);
  atomicAdd(
      &density_grid[real_grid_index(i1, j0, k0, ny, nz)],
      mass_density * wx1 * wy0 * wz0);
  atomicAdd(
      &density_grid[real_grid_index(i1, j0, k1, ny, nz)],
      mass_density * wx1 * wy0 * wz1);
  atomicAdd(
      &density_grid[real_grid_index(i1, j1, k0, ny, nz)],
      mass_density * wx1 * wy1 * wz0);
  atomicAdd(
      &density_grid[real_grid_index(i1, j1, k1, ny, nz)],
      mass_density * wx1 * wy1 * wz1);
}

__global__ void apply_potential_spectrum(const cufftComplex* density_k,
                                         cufftComplex* potential_k,
                                         const int nx,
                                         const int ny,
                                         const int nz,
                                         const int nz_complex,
                                         const double length_x,
                                         const double length_y,
                                         const double length_z,
                                         const double cell_x,
                                         const double cell_y,
                                         const double cell_z,
                                         const float grav_const) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t complex_count =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) *
      static_cast<std::size_t>(nz_complex);
  if (index >= complex_count) {
    return;
  }

  const int iz = static_cast<int>(index % static_cast<std::size_t>(nz_complex));
  const std::size_t xy = index / static_cast<std::size_t>(nz_complex);
  const int iy = static_cast<int>(xy % static_cast<std::size_t>(ny));
  const int ix = static_cast<int>(xy / static_cast<std::size_t>(ny));

  const int kx_index = (ix <= nx / 2) ? ix : ix - nx;
  const int ky_index = (iy <= ny / 2) ? iy : iy - ny;
  const int kz_index = iz;

  const float kx = static_cast<float>(2.0 * kPi * static_cast<double>(kx_index) / length_x);
  const float ky = static_cast<float>(2.0 * kPi * static_cast<double>(ky_index) / length_y);
  const float kz = static_cast<float>(2.0 * kPi * static_cast<double>(kz_index) / length_z);
  const float k_squared = kx * kx + ky * ky + kz * kz;

  if (k_squared <= 1.0e-12f) {
    potential_k[index].x = 0.0f;
    potential_k[index].y = 0.0f;
    return;
  }

  const float wx = sinc_pi(0.5f * kx * static_cast<float>(cell_x));
  const float wy = sinc_pi(0.5f * ky * static_cast<float>(cell_y));
  const float wz = sinc_pi(0.5f * kz * static_cast<float>(cell_z));
  const float window = wx * wy * wz;
  const float deconvolution = fminf(6.0f, 1.0f / fmaxf(window * window, 0.18f));
  const float scale = -static_cast<float>(kFourPi) * grav_const * deconvolution / k_squared;

  const cufftComplex rho = density_k[index];
  potential_k[index].x = scale * rho.x;
  potential_k[index].y = scale * rho.y;
}

__global__ void compute_force_from_potential(const cufftReal* potential_grid,
                                             cufftReal* force_x,
                                             cufftReal* force_y,
                                             cufftReal* force_z,
                                             const int nx,
                                             const int ny,
                                             const int nz,
                                             const double cell_x,
                                             const double cell_y,
                                             const double cell_z) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t real_count =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
  if (index >= real_count) {
    return;
  }

  const int iz = static_cast<int>(index % static_cast<std::size_t>(nz));
  const std::size_t xy = index / static_cast<std::size_t>(nz);
  const int iy = static_cast<int>(xy % static_cast<std::size_t>(ny));
  const int ix = static_cast<int>(xy / static_cast<std::size_t>(ny));

  const int ixm = wrap_index(ix - 1, nx);
  const int ixp = wrap_index(ix + 1, nx);
  const int iym = wrap_index(iy - 1, ny);
  const int iyp = wrap_index(iy + 1, ny);
  const int izm = wrap_index(iz - 1, nz);
  const int izp = wrap_index(iz + 1, nz);

  const float phi_xm = potential_grid[real_grid_index(ixm, iy, iz, ny, nz)];
  const float phi_xp = potential_grid[real_grid_index(ixp, iy, iz, ny, nz)];
  const float phi_ym = potential_grid[real_grid_index(ix, iym, iz, ny, nz)];
  const float phi_yp = potential_grid[real_grid_index(ix, iyp, iz, ny, nz)];
  const float phi_zm = potential_grid[real_grid_index(ix, iy, izm, ny, nz)];
  const float phi_zp = potential_grid[real_grid_index(ix, iy, izp, ny, nz)];

  force_x[index] = -(phi_xp - phi_xm) / static_cast<float>(2.0 * cell_x);
  force_y[index] = -(phi_yp - phi_ym) / static_cast<float>(2.0 * cell_y);
  force_z[index] = -(phi_zp - phi_zm) / static_cast<float>(2.0 * cell_z);
}

__global__ void compute_force_from_potential_clamped(const cufftReal* potential_grid,
                                                     cufftReal* force_x,
                                                     cufftReal* force_y,
                                                     cufftReal* force_z,
                                                     const int nx,
                                                     const int ny,
                                                     const int nz,
                                                     const double cell_x,
                                                     const double cell_y,
                                                     const double cell_z) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t real_count =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
  if (index >= real_count) {
    return;
  }

  const int iz = static_cast<int>(index % static_cast<std::size_t>(nz));
  const std::size_t xy = index / static_cast<std::size_t>(nz);
  const int iy = static_cast<int>(xy % static_cast<std::size_t>(ny));
  const int ix = static_cast<int>(xy / static_cast<std::size_t>(ny));

  const int ixm = clamp_index(ix - 1, nx);
  const int ixp = clamp_index(ix + 1, nx);
  const int iym = clamp_index(iy - 1, ny);
  const int iyp = clamp_index(iy + 1, ny);
  const int izm = clamp_index(iz - 1, nz);
  const int izp = clamp_index(iz + 1, nz);

  const float phi_xm = potential_grid[real_grid_index(ixm, iy, iz, ny, nz)];
  const float phi_xp = potential_grid[real_grid_index(ixp, iy, iz, ny, nz)];
  const float phi_ym = potential_grid[real_grid_index(ix, iym, iz, ny, nz)];
  const float phi_yp = potential_grid[real_grid_index(ix, iyp, iz, ny, nz)];
  const float phi_zm = potential_grid[real_grid_index(ix, iy, izm, ny, nz)];
  const float phi_zp = potential_grid[real_grid_index(ix, iy, izp, ny, nz)];

  const float dx = static_cast<float>((ixp == ixm) ? cell_x : (ixp - ixm) * cell_x);
  const float dy = static_cast<float>((iyp == iym) ? cell_y : (iyp - iym) * cell_y);
  const float dz = static_cast<float>((izp == izm) ? cell_z : (izp - izm) * cell_z);
  force_x[index] = -(phi_xp - phi_xm) / fmaxf(dx, 1.0e-6f);
  force_y[index] = -(phi_yp - phi_ym) / fmaxf(dy, 1.0e-6f);
  force_z[index] = -(phi_zp - phi_zm) / fmaxf(dz, 1.0e-6f);
}

__global__ void kick_particles_global(SimCudaParticle* particles,
                                      const std::uint64_t particle_count,
                                      const cufftReal* force_x,
                                      const cufftReal* force_y,
                                      const cufftReal* force_z,
                                      const int nx,
                                      const int ny,
                                      const int nz,
                                      const double origin_x,
                                      const double origin_y,
                                      const double origin_z,
                                      const double length_x,
                                      const double length_y,
                                      const double length_z,
                                      const double cell_x,
                                      const double cell_y,
                                      const double cell_z,
                                      const double inv_fft_cells,
                                      const int* galaxy_smbh_indices,
                                      const std::uint32_t galaxy_count,
                                      const double grav_const,
                                      const int enable_smbh_post_newtonian,
                                      const double dt_myr) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= particle_count) {
    return;
  }

  SimCudaParticle& particle = particles[index];

  double ax = sample_grid_trilinear(
                  force_x,
                  nx,
                  ny,
                  nz,
                  origin_x,
                  origin_y,
                  origin_z,
                  length_x,
                  length_y,
                  length_z,
                  cell_x,
                  cell_y,
                  cell_z,
                  particle.position_kpc[0],
                  particle.position_kpc[1],
                  particle.position_kpc[2]) *
              inv_fft_cells;
  double ay = sample_grid_trilinear(
                  force_y,
                  nx,
                  ny,
                  nz,
                  origin_x,
                  origin_y,
                  origin_z,
                  length_x,
                  length_y,
                  length_z,
                  cell_x,
                  cell_y,
                  cell_z,
                  particle.position_kpc[0],
                  particle.position_kpc[1],
                  particle.position_kpc[2]) *
              inv_fft_cells;
  double az = sample_grid_trilinear(
                  force_z,
                  nx,
                  ny,
                  nz,
                  origin_x,
                  origin_y,
                  origin_z,
                  length_x,
                  length_y,
                  length_z,
                  cell_x,
                  cell_y,
                  cell_z,
                  particle.position_kpc[0],
                  particle.position_kpc[1],
                  particle.position_kpc[2]) *
              inv_fft_cells;

  for (std::uint32_t galaxy_index = 0; galaxy_index < galaxy_count; ++galaxy_index) {
    const int source_index = galaxy_smbh_indices[galaxy_index];
    if (source_index < 0 || static_cast<std::uint64_t>(source_index) >= particle_count) {
      continue;
    }
    if (static_cast<int>(index) == source_index) {
      continue;
    }

    const SimCudaParticle source = particles[source_index];
    const double dx = source.position_kpc[0] - particle.position_kpc[0];
    const double dy = source.position_kpc[1] - particle.position_kpc[1];
    const double dz = source.position_kpc[2] - particle.position_kpc[2];
    const double softening = fmax(particle.softening_kpc, source.softening_kpc);
    add_point_mass_acceleration(ax, ay, az, dx, dy, dz, softening, grav_const, source.mass_msun);

    if (enable_smbh_post_newtonian != 0 && particle.component == 3u) {
      add_smbh_1pn_acceleration(ax, ay, az, particle, source, grav_const);
    }
  }

  particle.velocity_kms[0] += ax * dt_myr * kKpcPerKmPerMyr;
  particle.velocity_kms[1] += ay * dt_myr * kKpcPerKmPerMyr;
  particle.velocity_kms[2] += az * dt_myr * kKpcPerKmPerMyr;
}

__global__ void apply_local_mesh_correction(SimCudaParticle* particles,
                                            const std::uint64_t particle_count,
                                            const cufftReal* fine_force_x,
                                            const cufftReal* fine_force_y,
                                            const cufftReal* fine_force_z,
                                            const int fine_nx,
                                            const int fine_ny,
                                            const int fine_nz,
                                            const double fine_origin_x,
                                            const double fine_origin_y,
                                            const double fine_origin_z,
                                            const double fine_cell_x,
                                            const double fine_cell_y,
                                            const double fine_cell_z,
                                            const double fine_inv_fft_cells,
                                            const cufftReal* coarse_force_x,
                                            const cufftReal* coarse_force_y,
                                            const cufftReal* coarse_force_z,
                                            const int coarse_nx,
                                            const int coarse_ny,
                                            const int coarse_nz,
                                            const double coarse_origin_x,
                                            const double coarse_origin_y,
                                            const double coarse_origin_z,
                                            const double coarse_cell_x,
                                            const double coarse_cell_y,
                                            const double coarse_cell_z,
                                            const double coarse_inv_fft_cells,
                                            const double center_x,
                                            const double center_y,
                                            const double center_z,
                                            const double half_x,
                                            const double half_y,
                                            const double half_z,
                                            const double dt_myr) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= particle_count) {
    return;
  }

  SimCudaParticle& particle = particles[index];
  if (particle.component == 3u) {
    return;
  }

  const double dx = particle.position_kpc[0] - center_x;
  const double dy = particle.position_kpc[1] - center_y;
  const double dz = particle.position_kpc[2] - center_z;
  const double qx = fabs(dx) / fmax(half_x, 1.0e-6);
  const double qy = fabs(dy) / fmax(half_y, 1.0e-6);
  const double qz = fabs(dz) / fmax(half_z, 1.0e-6);
  const double q = fmax(qx, fmax(qy, qz));
  if (q >= 1.0) {
    return;
  }

  const double t = fmin(1.0, q / kLocalCorrectionBlendExtent);
  const double weight = 1.0 - t * t * (3.0 - 2.0 * t);

  const double ax_fine = sample_grid_trilinear_local(
                             fine_force_x,
                             fine_nx,
                             fine_ny,
                             fine_nz,
                             fine_origin_x,
                             fine_origin_y,
                             fine_origin_z,
                             fine_cell_x,
                             fine_cell_y,
                             fine_cell_z,
                             particle.position_kpc[0],
                             particle.position_kpc[1],
                             particle.position_kpc[2]) *
                         fine_inv_fft_cells;
  const double ay_fine = sample_grid_trilinear_local(
                             fine_force_y,
                             fine_nx,
                             fine_ny,
                             fine_nz,
                             fine_origin_x,
                             fine_origin_y,
                             fine_origin_z,
                             fine_cell_x,
                             fine_cell_y,
                             fine_cell_z,
                             particle.position_kpc[0],
                             particle.position_kpc[1],
                             particle.position_kpc[2]) *
                         fine_inv_fft_cells;
  const double az_fine = sample_grid_trilinear_local(
                             fine_force_z,
                             fine_nx,
                             fine_ny,
                             fine_nz,
                             fine_origin_x,
                             fine_origin_y,
                             fine_origin_z,
                             fine_cell_x,
                             fine_cell_y,
                             fine_cell_z,
                             particle.position_kpc[0],
                             particle.position_kpc[1],
                             particle.position_kpc[2]) *
                         fine_inv_fft_cells;
  const double ax_coarse = sample_grid_trilinear_local(
                               coarse_force_x,
                               coarse_nx,
                               coarse_ny,
                               coarse_nz,
                               coarse_origin_x,
                               coarse_origin_y,
                               coarse_origin_z,
                               coarse_cell_x,
                               coarse_cell_y,
                               coarse_cell_z,
                               particle.position_kpc[0],
                               particle.position_kpc[1],
                               particle.position_kpc[2]) *
                           coarse_inv_fft_cells;
  const double ay_coarse = sample_grid_trilinear_local(
                               coarse_force_y,
                               coarse_nx,
                               coarse_ny,
                               coarse_nz,
                               coarse_origin_x,
                               coarse_origin_y,
                               coarse_origin_z,
                               coarse_cell_x,
                               coarse_cell_y,
                               coarse_cell_z,
                               particle.position_kpc[0],
                               particle.position_kpc[1],
                               particle.position_kpc[2]) *
                           coarse_inv_fft_cells;
  const double az_coarse = sample_grid_trilinear_local(
                               coarse_force_z,
                               coarse_nx,
                               coarse_ny,
                               coarse_nz,
                               coarse_origin_x,
                               coarse_origin_y,
                               coarse_origin_z,
                               coarse_cell_x,
                               coarse_cell_y,
                               coarse_cell_z,
                               particle.position_kpc[0],
                               particle.position_kpc[1],
                               particle.position_kpc[2]) *
                           coarse_inv_fft_cells;

  particle.velocity_kms[0] += (ax_fine - ax_coarse) * weight * dt_myr * kKpcPerKmPerMyr;
  particle.velocity_kms[1] += (ay_fine - ay_coarse) * weight * dt_myr * kKpcPerKmPerMyr;
  particle.velocity_kms[2] += (az_fine - az_coarse) * weight * dt_myr * kKpcPerKmPerMyr;
}

__global__ void drift_particles(SimCudaParticle* particles,
                                const std::uint64_t particle_count,
                                const double dt_myr) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= particle_count) {
    return;
  }

  SimCudaParticle& particle = particles[index];
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
  const float speed = static_cast<float>(
      sqrt(particle.velocity_kms[0] * particle.velocity_kms[0] +
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

int build_force_mesh(DeviceState* state, char* error_buffer, const std::size_t error_buffer_len) {
  const cudaError_t clear_status =
      cudaMemset(state->density_grid, 0, sizeof(cufftReal) * state->real_count);
  if (clear_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "density grid clear failed", clear_status);
    return 1;
  }

  const int threads_per_block = 256;
  const int particle_blocks =
      static_cast<int>((state->particle_count + threads_per_block - 1) / threads_per_block);
  deposit_mass_cic<<<particle_blocks, threads_per_block>>>(
      state->particles,
      state->particle_count,
      state->density_grid,
      state->nx,
      state->ny,
      state->nz,
      state->domain_origin[0],
      state->domain_origin[1],
      state->domain_origin[2],
      state->box_length[0],
      state->box_length[1],
      state->box_length[2],
      state->cell_size[0],
      state->cell_size[1],
      state->cell_size[2],
      static_cast<float>(1.0 / state->cell_volume));
  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "mass deposit kernel failed", cuda_status);
    return 1;
  }

  if (!state->refinement_patches.empty() &&
      update_refinement_patches(state, error_buffer, error_buffer_len) != 0) {
    return 1;
  }

  const cufftResult forward_status =
      cufftExecR2C(state->forward_plan, state->density_grid, state->density_k);
  if (forward_status != CUFFT_SUCCESS) {
    fill_cufft_error(error_buffer, error_buffer_len, "forward density FFT failed", forward_status);
    return 1;
  }

  const int complex_blocks =
      static_cast<int>((state->complex_count + threads_per_block - 1) / threads_per_block);
  apply_potential_spectrum<<<complex_blocks, threads_per_block>>>(
      state->density_k,
      state->force_k,
      state->nx,
      state->ny,
      state->nz,
      state->nz_complex,
      state->box_length[0],
      state->box_length[1],
      state->box_length[2],
      state->cell_size[0],
      state->cell_size[1],
      state->cell_size[2],
      static_cast<float>(state->grav_const));
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "potential spectrum kernel failed", cuda_status);
    return 1;
  }

  const cufftResult inverse_status =
      cufftExecC2R(state->inverse_plan, state->force_k, state->density_grid);
  if (inverse_status != CUFFT_SUCCESS) {
    fill_cufft_error(
        error_buffer, error_buffer_len, "inverse potential FFT failed", inverse_status);
    return 1;
  }

  const int real_blocks =
      static_cast<int>((state->real_count + threads_per_block - 1) / threads_per_block);
  compute_force_from_potential<<<real_blocks, threads_per_block>>>(
      state->density_grid,
      state->force_x,
      state->force_y,
      state->force_z,
      state->nx,
      state->ny,
      state->nz,
      state->cell_size[0],
      state->cell_size[1],
      state->cell_size[2]);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "force gradient kernel failed", cuda_status);
    return 1;
  }

  return 0;
}

int build_local_force_mesh(DeviceState* state,
                           RefinementPatchState& patch,
                           char* error_buffer,
                           const std::size_t error_buffer_len) {
  if (!patch.active) {
    return 0;
  }
  const int threads_per_block = 256;
  const int particle_blocks =
      static_cast<int>((state->particle_count + threads_per_block - 1) / threads_per_block);

  auto build_mesh = [&](MeshBuffers& mesh, const double cell_size[3]) -> int {
    cudaError_t cuda_status = cudaMemset(mesh.density_grid, 0, sizeof(cufftReal) * mesh.real_count);
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "local density grid clear failed", cuda_status);
      return 1;
    }

    deposit_mass_cic_local<<<particle_blocks, threads_per_block>>>(
        state->particles,
        state->particle_count,
        mesh.density_grid,
        mesh.nx,
        mesh.ny,
        mesh.nz,
        patch.domain_origin[0],
        patch.domain_origin[1],
        patch.domain_origin[2],
        cell_size[0],
        cell_size[1],
        cell_size[2],
        static_cast<float>(1.0 / (cell_size[0] * cell_size[1] * cell_size[2])));
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "local mass deposit kernel failed", cuda_status);
      return 1;
    }

    const cufftResult forward_status = cufftExecR2C(mesh.forward_plan, mesh.density_grid, mesh.density_k);
    if (forward_status != CUFFT_SUCCESS) {
      fill_cufft_error(error_buffer, error_buffer_len, "local forward density FFT failed", forward_status);
      return 1;
    }

    const int complex_blocks =
        static_cast<int>((mesh.complex_count + threads_per_block - 1) / threads_per_block);
    apply_potential_spectrum<<<complex_blocks, threads_per_block>>>(
        mesh.density_k,
        mesh.force_k,
        mesh.nx,
        mesh.ny,
        mesh.nz,
        mesh.nz_complex,
        patch.box_length[0],
        patch.box_length[1],
        patch.box_length[2],
        cell_size[0],
        cell_size[1],
        cell_size[2],
        static_cast<float>(state->grav_const));
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "local potential spectrum kernel failed", cuda_status);
      return 1;
    }

    const cufftResult inverse_status = cufftExecC2R(mesh.inverse_plan, mesh.force_k, mesh.density_grid);
    if (inverse_status != CUFFT_SUCCESS) {
      fill_cufft_error(error_buffer, error_buffer_len, "local inverse potential FFT failed", inverse_status);
      return 1;
    }

    const int real_blocks =
        static_cast<int>((mesh.real_count + threads_per_block - 1) / threads_per_block);
    compute_force_from_potential_clamped<<<real_blocks, threads_per_block>>>(
        mesh.density_grid,
        mesh.force_x,
        mesh.force_y,
        mesh.force_z,
        mesh.nx,
        mesh.ny,
        mesh.nz,
        cell_size[0],
        cell_size[1],
        cell_size[2]);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "local force gradient kernel failed", cuda_status);
      return 1;
    }

    return 0;
  };

  if (build_mesh(patch.fine, patch.cell_size_fine) != 0) {
    return 1;
  }
  if (build_mesh(patch.coarse, patch.cell_size_coarse) != 0) {
    return 1;
  }
  return 0;
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
  const int particle_blocks =
      static_cast<int>((state->particle_count + threads_per_block - 1) / threads_per_block);
  const double inv_fft_cells = 1.0 / static_cast<double>(state->real_count);

  for (std::uint32_t step = 0; step < step_count; ++step) {
    if (build_force_mesh(state, error_buffer, error_buffer_len) != 0) {
      return 1;
    }

    for (auto& patch : state->refinement_patches) {
      if (build_local_force_mesh(state, patch, error_buffer, error_buffer_len) != 0) {
        return 1;
      }
    }

    kick_particles_global<<<particle_blocks, threads_per_block>>>(
        state->particles,
        state->particle_count,
        state->force_x,
        state->force_y,
        state->force_z,
        state->nx,
        state->ny,
        state->nz,
        state->domain_origin[0],
        state->domain_origin[1],
        state->domain_origin[2],
        state->box_length[0],
        state->box_length[1],
        state->box_length[2],
        state->cell_size[0],
        state->cell_size[1],
        state->cell_size[2],
        inv_fft_cells,
        state->galaxy_smbh_indices,
        state->galaxy_count,
        state->grav_const,
        state->enable_smbh_post_newtonian ? 1 : 0,
        dt_myr);

    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "global particle kick kernel failed", cuda_status);
      return 1;
    }

    for (const auto& patch : state->refinement_patches) {
      if (!patch.active) {
        continue;
      }
      apply_local_mesh_correction<<<particle_blocks, threads_per_block>>>(
          state->particles,
          state->particle_count,
          patch.fine.force_x,
          patch.fine.force_y,
          patch.fine.force_z,
          patch.fine.nx,
          patch.fine.ny,
          patch.fine.nz,
          patch.domain_origin[0],
          patch.domain_origin[1],
          patch.domain_origin[2],
          patch.cell_size_fine[0],
          patch.cell_size_fine[1],
          patch.cell_size_fine[2],
          1.0 / static_cast<double>(patch.fine.real_count),
          patch.coarse.force_x,
          patch.coarse.force_y,
          patch.coarse.force_z,
          patch.coarse.nx,
          patch.coarse.ny,
          patch.coarse.nz,
          patch.domain_origin[0],
          patch.domain_origin[1],
          patch.domain_origin[2],
          patch.cell_size_coarse[0],
          patch.cell_size_coarse[1],
          patch.cell_size_coarse[2],
          1.0 / static_cast<double>(patch.coarse.real_count),
          patch.domain_origin[0] + 0.5 * patch.box_length[0],
          patch.domain_origin[1] + 0.5 * patch.box_length[1],
          patch.domain_origin[2] + 0.5 * patch.box_length[2],
          0.5 * patch.box_length[0],
          0.5 * patch.box_length[1],
          0.5 * patch.box_length[2],
          dt_myr);
      cuda_status = cudaGetLastError();
      if (cuda_status != cudaSuccess) {
        fill_cuda_error(error_buffer, error_buffer_len, "local correction kernel failed", cuda_status);
        return 1;
      }
    }

    drift_particles<<<particle_blocks, threads_per_block>>>(state->particles, state->particle_count, dt_myr);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "particle drift kernel failed", cuda_status);
      return 1;
    }
  }

  const cudaError_t sync_status = cudaDeviceSynchronize();
  if (sync_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "device synchronization failed", sync_status);
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

  if (params->particle_count == 0 || params->galaxy_count == 0) {
    fill_error(error_buffer, error_buffer_len, "simulation needs particles and galaxies");
    return 1;
  }
  if (params->mesh_resolution[0] == 0 || params->mesh_resolution[1] == 0 ||
      params->mesh_resolution[2] == 0) {
    fill_error(error_buffer, error_buffer_len, "mesh resolution must be positive");
    return 1;
  }

  auto* state = new DeviceState();
  state->particle_count = params->particle_count;
  state->galaxy_count = params->galaxy_count;
  state->grav_const = params->grav_const_kpc_kms2_per_msun;
  state->base_timestep_myr = params->base_timestep_myr;
  state->enable_smbh_post_newtonian = params->enable_smbh_post_newtonian != 0;
  state->nx = static_cast<int>(params->mesh_resolution[0]);
  state->ny = static_cast<int>(params->mesh_resolution[1]);
  state->nz = static_cast<int>(params->mesh_resolution[2]);
  state->nz_complex = state->nz / 2 + 1;
  state->real_count = static_cast<std::size_t>(state->nx) * static_cast<std::size_t>(state->ny) *
                      static_cast<std::size_t>(state->nz);
  state->complex_count = static_cast<std::size_t>(state->nx) * static_cast<std::size_t>(state->ny) *
                         static_cast<std::size_t>(state->nz_complex);

  compute_simulation_domain(state, particles);

  const auto particle_bytes = sizeof(SimCudaParticle) * params->particle_count;
  cudaError_t cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->particles), particle_bytes);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for particles failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  cuda_status = cudaMemcpy(state->particles, particles, particle_bytes, cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMemcpy for particles failed", cuda_status);
    destroy_state(state);
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
    destroy_state(state);
    return 1;
  }

  state->galaxy_smbh_indices_host = galaxy_smbh_indices;
  state->refinement_patches.resize(kMaxRefinementPatches);
  for (auto& patch : state->refinement_patches) {
    if (create_mesh_buffers(patch.fine,
                            kLocalFineNx,
                            kLocalFineNy,
                            kLocalFineNz,
                            error_buffer,
                            error_buffer_len) != 0) {
      destroy_state(state);
      return 1;
    }
    if (create_mesh_buffers(patch.coarse,
                            kLocalCoarseNx,
                            kLocalCoarseNy,
                            kLocalCoarseNz,
                            error_buffer,
                            error_buffer_len) != 0) {
      destroy_state(state);
      return 1;
    }
  }

  cuda_status = cudaMalloc(
      reinterpret_cast<void**>(&state->galaxy_smbh_indices),
      sizeof(int) * params->galaxy_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(
        error_buffer, error_buffer_len, "cudaMalloc for SMBH index buffer failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  cuda_status = cudaMemcpy(state->galaxy_smbh_indices,
                           galaxy_smbh_indices.data(),
                           sizeof(int) * params->galaxy_count,
                           cudaMemcpyHostToDevice);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(
        error_buffer, error_buffer_len, "cudaMemcpy for SMBH index buffer failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  cuda_status = cudaMalloc(
      reinterpret_cast<void**>(&state->density_grid), sizeof(cufftReal) * state->real_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for density grid failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  cuda_status = cudaMalloc(
      reinterpret_cast<void**>(&state->density_k), sizeof(cufftComplex) * state->complex_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for density spectrum failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  cuda_status = cudaMalloc(
      reinterpret_cast<void**>(&state->force_k), sizeof(cufftComplex) * state->complex_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for force spectrum failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  cuda_status = cudaMalloc(
      reinterpret_cast<void**>(&state->force_x), sizeof(cufftReal) * state->real_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for force_x failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  cuda_status = cudaMalloc(
      reinterpret_cast<void**>(&state->force_y), sizeof(cufftReal) * state->real_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for force_y failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  cuda_status = cudaMalloc(
      reinterpret_cast<void**>(&state->force_z), sizeof(cufftReal) * state->real_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for force_z failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  cufftResult fft_status = cufftPlan3d(&state->forward_plan, state->nx, state->ny, state->nz, CUFFT_R2C);
  if (fft_status != CUFFT_SUCCESS) {
    fill_cufft_error(error_buffer, error_buffer_len, "forward FFT plan creation failed", fft_status);
    destroy_state(state);
    return 1;
  }

  fft_status = cufftPlan3d(&state->inverse_plan, state->nx, state->ny, state->nz, CUFFT_C2R);
  if (fft_status != CUFFT_SUCCESS) {
    fill_cufft_error(error_buffer, error_buffer_len, "inverse FFT plan creation failed", fft_status);
    destroy_state(state);
    return 1;
  }

  *out_handle = state;
  return 0;
}

extern "C" int sim_cuda_destroy(void* handle) {
  destroy_state(static_cast<DeviceState*>(handle));
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
      state, substeps, dt_myr, state->base_timestep_myr, diagnostics, error_buffer, error_buffer_len);
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
  return run_steps(state,
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

    cuda_status = cudaMemcpy(out_particles + anchor_count,
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
    cuda_status = cudaMemcpy(&particle,
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
    fill_error(error_buffer, error_buffer_len, "particle capacity smaller than simulation state");
    return 1;
  }

  const cudaError_t cuda_status = cudaMemcpy(out_particles,
                                             state->particles,
                                             sizeof(SimCudaParticle) * state->particle_count,
                                             cudaMemcpyDeviceToHost);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "particle download failed", cuda_status);
    return 1;
  }

  return 0;
}
