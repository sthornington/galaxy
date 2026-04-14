#include "sim_cuda.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

namespace {

constexpr double kKpcPerKmPerMyr = 0.001022712165045695;
constexpr double kSpeedOfLightKms = 299792.458;
constexpr double kFourPi = 12.566370614359172;
constexpr double kPi = 3.14159265358979323846;
constexpr double kSqrtPi = 1.77245385090551602729;
constexpr double kGlobalDomainPadding = 4.0;
constexpr double kShortRangeDomainPadding = 1.15;
constexpr double kMinGlobalBoxLengthKpc = 64.0;
constexpr double kMinShortRangeBoxLengthKpc = 8.0;
constexpr double kMinShortRangeCellSizeKpc = 0.35;
constexpr double kMaxShortRangeCellSizeKpc = 3.0;
constexpr double kShortRangeTargetOccupancy = 8.0;
constexpr std::uint32_t kShortRangeForceFactorLutSize = 4096;
constexpr std::size_t kMaxShortRangeCells = 4u * 1024u * 1024u;
constexpr int kMaxShortRangeAxisCells = 1024;
constexpr int kShortRangeDirectCellThreshold = 64;
// Shipped presets should all exercise the same solver path.
// The host-built particle tree remains available as an opt-in diagnostic path via
// SIM_CUDA_PARTICLE_TREE_THRESHOLD, but the default runtime stays on the large-run cell tree.
constexpr std::uint32_t kDefaultParticleTreeThreshold = 0;
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

std::uint32_t particle_tree_threshold() {
  static const std::uint32_t threshold = []() {
    const char* raw_value = std::getenv("SIM_CUDA_PARTICLE_TREE_THRESHOLD");
    if (raw_value == nullptr || raw_value[0] == '\0') {
      return kDefaultParticleTreeThreshold;
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(raw_value, &end, 10);
    if (end == raw_value || (end != nullptr && *end != '\0')) {
      return kDefaultParticleTreeThreshold;
    }
    return static_cast<std::uint32_t>(
        std::min<unsigned long long>(parsed, std::numeric_limits<std::uint32_t>::max()));
  }();
  return threshold;
}

bool short_range_baryons_only() {
  static const bool enabled = []() {
    const char* raw_value = std::getenv("SIM_CUDA_SHORT_RANGE_BARYONS_ONLY");
    if (raw_value == nullptr || raw_value[0] == '\0') {
      return false;
    }
    return std::strcmp(raw_value, "0") != 0;
  }();
  return enabled;
}

bool short_range_target_baryons_only() {
  static const bool enabled = []() {
    const char* raw_value = std::getenv("SIM_CUDA_SHORT_RANGE_TARGET_BARYONS_ONLY");
    if (raw_value == nullptr || raw_value[0] == '\0') {
      return false;
    }
    return std::strcmp(raw_value, "0") != 0;
  }();
  return enabled;
}

double short_range_target_occupancy(const std::uint64_t particle_count) {
  static const double override = []() {
    const char* raw_value = std::getenv("SIM_CUDA_SHORT_RANGE_TARGET_OCCUPANCY");
    if (raw_value == nullptr || raw_value[0] == '\0') {
      return 0.0;
    }
    char* end = nullptr;
    const double parsed = std::strtod(raw_value, &end);
    if (end == raw_value || (end != nullptr && *end != '\0') || !(parsed > 0.0)) {
      return 0.0;
    }
    return parsed;
  }();
  if (override > 0.0) {
    return override;
  }
  // Low-particle interactive runs were over-refining the short-range grid,
  // which inflated the host-built cell tree/interactions far more than it
  // improved force quality. Use coarser target occupancy at low N and a
  // still-moderate occupancy at larger N.
  if (particle_count <= 300'000u) {
    return 64.0;
  }
  if (particle_count <= 1'000'000u) {
    return 32.0;
  }
  if (particle_count >= 2'000'000u) {
    return 32.0;
  }
  return 24.0;
}

bool profile_force_stages() {
  static const bool enabled = []() {
    const char* raw_value = std::getenv("SIM_CUDA_PROFILE_FORCE_STAGES");
    if (raw_value == nullptr || raw_value[0] == '\0') {
      return false;
    }
    return std::strcmp(raw_value, "0") != 0;
  }();
  return enabled;
}

__host__ __device__ __forceinline__ double treepm_short_range_force_factor(
    const double r, const double split_radius_kpc);

void flush_profile_stage(const char* label,
                         const std::chrono::steady_clock::time_point start,
                         const std::chrono::steady_clock::time_point end) {
  std::fprintf(stderr,
               "[sim-cuda] stage=%s wall_ms=%.3f\n",
               label,
               std::chrono::duration<double, std::milli>(end - start).count());
}

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

struct ShortRangeTreeNode {
  double mass = 0.0;
  double com[3] = {0.0, 0.0, 0.0};
  double center[3] = {0.0, 0.0, 0.0};
  double half_size = 0.0;
  double softening_kpc = 0.0;
  int child[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
  std::uint8_t child_mask = 0;
  int cell_id = -1;
  int particle_index = -1;
};

struct ShortRangeSourceCell {
  int cell_id = -1;
  double mass = 0.0;
  double com[3] = {0.0, 0.0, 0.0};
  double softening_kpc = 0.0;
};

struct ShortRangeParticleSource {
  int particle_index = -1;
  double mass = 0.0;
  double position[3] = {0.0, 0.0, 0.0};
  double softening_kpc = 0.0;
};

struct ShortRangeInteractionSource {
  double mass = 0.0;
  double com[3] = {0.0, 0.0, 0.0};
  double softening_kpc = 0.0;
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
  std::uint32_t max_substeps = 1;
  double cfl_safety_factor = 0.35;
  double max_softening_kpc = 0.05;
  double opening_angle = 0.55;
  bool short_tree_particle_mode = false;
  bool short_range_target_baryons_only = false;
  bool force_state_valid = false;

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
  double tight_domain_origin[3] = {0.0, 0.0, 0.0};
  double tight_box_length[3] = {1.0, 1.0, 1.0};

  int short_nx = 0;
  int short_ny = 0;
  int short_nz = 0;
  std::size_t short_cell_count = 0;
  std::size_t short_cell_capacity = 0;
  std::size_t short_cell_interaction_cell_capacity = 0;
  std::size_t short_cell_interaction_capacity = 0;
  double short_domain_origin[3] = {0.0, 0.0, 0.0};
  double short_box_length[3] = {1.0, 1.0, 1.0};
  double short_cell_size[3] = {1.0, 1.0, 1.0};
  double short_cutoff_kpc = 0.0;
  double short_pm_softening_kpc = 0.0;
  double short_force_factor_lut_scale = 0.0;

  SimCudaParticle* particles = nullptr;
  int* galaxy_smbh_indices = nullptr;
  int* preview_visible_particle_indices = nullptr;
  int* short_source_particle_indices = nullptr;
  int* short_sorted_cell_ids = nullptr;
  int* short_sorted_particle_indices = nullptr;
  int* short_cell_start = nullptr;
  int* short_cell_end = nullptr;
  int* short_cell_interaction_start = nullptr;
  double* short_cell_mass = nullptr;
  double* short_cell_com_x = nullptr;
  double* short_cell_com_y = nullptr;
  double* short_cell_com_z = nullptr;
  double* short_cell_octant_mass = nullptr;
  double* short_cell_octant_com_x = nullptr;
  double* short_cell_octant_com_y = nullptr;
  double* short_cell_octant_com_z = nullptr;
  float* short_force_factor_lut = nullptr;
  ShortRangeInteractionSource* short_cell_interactions = nullptr;
  ShortRangeTreeNode* short_tree_nodes = nullptr;
  std::uint32_t short_source_particle_count = 0;
  std::uint32_t preview_visible_particle_count = 0;
  std::uint32_t short_tree_node_count = 0;
  std::size_t short_tree_node_capacity = 0;
  int short_tree_root = -1;
  std::vector<int> galaxy_smbh_indices_host;
  std::vector<cufftReal> density_host;
  std::vector<RefinementPatchState> refinement_patches;
  std::vector<double> short_cell_mass_host;
  std::vector<double> short_cell_com_x_host;
  std::vector<double> short_cell_com_y_host;
  std::vector<double> short_cell_com_z_host;
  std::vector<int> short_cell_interaction_start_host;
  std::vector<ShortRangeInteractionSource> short_cell_interactions_host;
  std::vector<float> short_force_factor_lut_host;
  std::vector<ShortRangeTreeNode> short_tree_host;
  std::vector<SimCudaParticle> particles_host;

  cufftReal* density_grid = nullptr;
  cufftComplex* density_k = nullptr;
  cufftComplex* force_k = nullptr;
  cufftReal* force_x = nullptr;
  cufftReal* force_y = nullptr;
  cufftReal* force_z = nullptr;

  cufftHandle forward_plan = 0;
  cufftHandle inverse_plan = 0;
  cudaStream_t compute_stream = nullptr;
  cudaStream_t preview_stream = nullptr;
  cudaEvent_t preview_sample_event = nullptr;
  cudaEvent_t preview_copy_done_event = nullptr;

  SimCudaPreviewParticle* preview_particles = nullptr;
  SimCudaPreviewParticle* preview_host_particles = nullptr;
  std::uint32_t preview_capacity = 0;
  std::uint32_t preview_in_flight_count = 0;
  std::uint32_t preview_sample_offset = 0;
  bool preview_in_flight = false;
};

__global__ void sample_preview(const SimCudaParticle* particles,
                               const int* visible_particle_indices,
                               const std::uint32_t visible_particle_count,
                               const int* anchor_indices,
                               const std::uint32_t anchor_count,
                               const std::uint32_t sampled_count,
                               const std::uint32_t sample_offset,
                               const std::uint64_t stride,
                               SimCudaPreviewParticle* out_particles);

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
  if (state->preview_copy_done_event != nullptr) {
    cudaEventDestroy(state->preview_copy_done_event);
  }
  if (state->preview_sample_event != nullptr) {
    cudaEventDestroy(state->preview_sample_event);
  }
  if (state->preview_stream != nullptr) {
    cudaStreamDestroy(state->preview_stream);
  }
  if (state->compute_stream != nullptr) {
    cudaStreamDestroy(state->compute_stream);
  }
  if (state->forward_plan != 0) {
    cufftDestroy(state->forward_plan);
  }
  if (state->inverse_plan != 0) {
    cufftDestroy(state->inverse_plan);
  }
  cudaFree(state->short_cell_com_z);
  cudaFree(state->short_cell_com_y);
  cudaFree(state->short_cell_com_x);
  cudaFree(state->short_cell_mass);
  cudaFree(state->short_cell_interactions);
  cudaFree(state->short_cell_interaction_start);
  cudaFree(state->short_cell_octant_com_z);
  cudaFree(state->short_cell_octant_com_y);
  cudaFree(state->short_cell_octant_com_x);
  cudaFree(state->short_cell_octant_mass);
  cudaFree(state->short_force_factor_lut);
  cudaFree(state->short_tree_nodes);
  cudaFree(state->short_cell_end);
  cudaFree(state->short_cell_start);
  cudaFree(state->short_sorted_particle_indices);
  cudaFree(state->short_sorted_cell_ids);
  cudaFree(state->short_source_particle_indices);
  cudaFree(state->preview_visible_particle_indices);
  cudaFreeHost(state->preview_host_particles);
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
  }
  if (state->preview_host_particles != nullptr) {
    cudaFreeHost(state->preview_host_particles);
    state->preview_host_particles = nullptr;
  }
  state->preview_capacity = 0;
  state->preview_in_flight = false;
  state->preview_in_flight_count = 0;

  cudaError_t cuda_status = cudaMalloc(
      reinterpret_cast<void**>(&state->preview_particles),
      sizeof(SimCudaPreviewParticle) * count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(
        error_buffer, error_buffer_len, "cudaMalloc for preview buffer failed", cuda_status);
    return 1;
  }
  cuda_status = cudaMallocHost(
      reinterpret_cast<void**>(&state->preview_host_particles),
      sizeof(SimCudaPreviewParticle) * count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(
        error_buffer, error_buffer_len, "cudaMallocHost for preview buffer failed", cuda_status);
    cudaFree(state->preview_particles);
    state->preview_particles = nullptr;
    return 1;
  }

  state->preview_capacity = count;
  return 0;
}

int wait_for_in_flight_preview(DeviceState* state,
                               char* error_buffer,
                               const std::size_t error_buffer_len) {
  if (!state->preview_in_flight) {
    return 0;
  }

  const cudaError_t cuda_status = cudaEventSynchronize(state->preview_copy_done_event);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(
        error_buffer, error_buffer_len, "preview synchronization failed", cuda_status);
    return 1;
  }

  state->preview_in_flight = false;
  state->preview_in_flight_count = 0;
  return 0;
}

int schedule_preview_capture(DeviceState* state,
                             const std::uint32_t max_particles,
                             std::uint32_t* out_count,
                             char* error_buffer,
                             const std::size_t error_buffer_len) {
  if ((state->preview_visible_particle_count == 0 &&
       state->galaxy_smbh_indices_host.empty()) ||
      max_particles == 0) {
    if (out_count != nullptr) {
      *out_count = 0;
    }
    state->preview_in_flight = false;
    state->preview_in_flight_count = 0;
    return 0;
  }

  if (state->preview_in_flight) {
    fill_error(error_buffer, error_buffer_len, "preview capture already in flight");
    return 1;
  }

  const std::uint32_t visible_count = state->preview_visible_particle_count;
  const std::uint32_t max_anchor_count =
      static_cast<std::uint32_t>(state->galaxy_smbh_indices_host.size());
  const std::uint32_t anchor_count = std::min<std::uint32_t>(max_particles, max_anchor_count);
  const std::uint32_t sample_budget = max_particles - anchor_count;
  const std::uint32_t sampled_count = std::min<std::uint32_t>(sample_budget, visible_count);
  const std::uint32_t count = anchor_count + sampled_count;
  const std::uint64_t stride = sampled_count == 0
                                   ? 1
                                   : std::max<std::uint64_t>(
                                         1,
                                         static_cast<std::uint64_t>(visible_count) /
                                             static_cast<std::uint64_t>(sampled_count));
  const std::uint32_t sample_offset =
      visible_count == 0 ? 0 : (state->preview_sample_offset % visible_count);
  if (visible_count > 0) {
    const std::uint32_t hop = std::max<std::uint32_t>(1, visible_count / 17);
    state->preview_sample_offset = (sample_offset + hop) % visible_count;
  }
  if (ensure_preview_capacity(state, count, error_buffer, error_buffer_len) != 0) {
    return 1;
  }
  const int threads_per_block = 256;
  const int blocks = static_cast<int>((count + threads_per_block - 1) / threads_per_block);

  sample_preview<<<blocks, threads_per_block, 0, state->compute_stream>>>(
      state->particles,
      state->preview_visible_particle_indices,
      state->preview_visible_particle_count,
      state->galaxy_smbh_indices,
      anchor_count,
      sampled_count,
      sample_offset,
      stride,
      state->preview_particles);
  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "preview kernel launch failed", cuda_status);
    return 1;
  }

  cuda_status = cudaEventRecord(state->preview_sample_event, state->compute_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "preview sample event record failed", cuda_status);
    return 1;
  }

  cuda_status = cudaStreamWaitEvent(state->preview_stream, state->preview_sample_event, 0);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "preview stream wait failed", cuda_status);
    return 1;
  }

  cuda_status = cudaMemcpyAsync(state->preview_host_particles,
                                state->preview_particles,
                                sizeof(SimCudaPreviewParticle) * count,
                                cudaMemcpyDeviceToHost,
                                state->preview_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "preview async download failed", cuda_status);
    return 1;
  }

  cuda_status = cudaEventRecord(state->preview_copy_done_event, state->preview_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "preview copy event record failed", cuda_status);
    return 1;
  }

  state->preview_in_flight = true;
  state->preview_in_flight_count = count;
  if (out_count != nullptr) {
    *out_count = count;
  }
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
  double min_short_pos[3] = {
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
      std::numeric_limits<double>::infinity(),
  };
  double max_short_pos[3] = {
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
      -std::numeric_limits<double>::infinity(),
  };

  for (std::uint64_t i = 0; i < state->particle_count; ++i) {
    for (int axis = 0; axis < 3; ++axis) {
      min_pos[axis] = std::min(min_pos[axis], particles[i].position_kpc[axis]);
      max_pos[axis] = std::max(max_pos[axis], particles[i].position_kpc[axis]);
      if (particles[i].component != 0u) {
        min_short_pos[axis] = std::min(min_short_pos[axis], particles[i].position_kpc[axis]);
        max_short_pos[axis] = std::max(max_short_pos[axis], particles[i].position_kpc[axis]);
      }
    }
  }

  for (int axis = 0; axis < 3; ++axis) {
    const double range = std::max(1.0, max_pos[axis] - min_pos[axis]);
    const double center = 0.5 * (min_pos[axis] + max_pos[axis]);
    const bool has_short_bounds = min_short_pos[axis] <= max_short_pos[axis];
    const double short_min = has_short_bounds ? min_short_pos[axis] : min_pos[axis];
    const double short_max = has_short_bounds ? max_short_pos[axis] : max_pos[axis];
    const double short_range = std::max(1.0, short_max - short_min);
    const double short_center = 0.5 * (short_min + short_max);
    const double tight_length =
        std::max(kMinShortRangeBoxLengthKpc, short_range * kShortRangeDomainPadding);
    const double padded_length = std::max(kMinGlobalBoxLengthKpc, range * kGlobalDomainPadding);
    state->tight_box_length[axis] = tight_length;
    state->tight_domain_origin[axis] = short_center - 0.5 * tight_length;
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

struct DomainBounds {
  double min_pos[3];
  double max_pos[3];
  double short_min_pos[3];
  double short_max_pos[3];
};

__host__ __device__ DomainBounds empty_domain_bounds() {
  DomainBounds bounds{};
  for (int axis = 0; axis < 3; ++axis) {
    bounds.min_pos[axis] = std::numeric_limits<double>::infinity();
    bounds.max_pos[axis] = -std::numeric_limits<double>::infinity();
    bounds.short_min_pos[axis] = std::numeric_limits<double>::infinity();
    bounds.short_max_pos[axis] = -std::numeric_limits<double>::infinity();
  }
  return bounds;
}

struct DomainBoundsAccessor {
  __host__ __device__ DomainBounds operator()(const SimCudaParticle& particle) const {
    DomainBounds bounds = empty_domain_bounds();
    for (int axis = 0; axis < 3; ++axis) {
      const double position = particle.position_kpc[axis];
      bounds.min_pos[axis] = position;
      bounds.max_pos[axis] = position;
      if (particle.component != 3u) {
        bounds.short_min_pos[axis] = position;
        bounds.short_max_pos[axis] = position;
      }
    }
    return bounds;
  }
};

struct MergeDomainBounds {
  __host__ __device__ DomainBounds operator()(const DomainBounds& lhs, const DomainBounds& rhs) const {
    DomainBounds merged{};
    for (int axis = 0; axis < 3; ++axis) {
      merged.min_pos[axis] = fmin(lhs.min_pos[axis], rhs.min_pos[axis]);
      merged.max_pos[axis] = fmax(lhs.max_pos[axis], rhs.max_pos[axis]);
      merged.short_min_pos[axis] = fmin(lhs.short_min_pos[axis], rhs.short_min_pos[axis]);
      merged.short_max_pos[axis] = fmax(lhs.short_max_pos[axis], rhs.short_max_pos[axis]);
    }
    return merged;
  }
};

struct SpeedAccessor {
  __host__ __device__ double operator()(const SimCudaParticle& particle) const {
    return sqrt(particle.velocity_kms[0] * particle.velocity_kms[0] +
                particle.velocity_kms[1] * particle.velocity_kms[1] +
                particle.velocity_kms[2] * particle.velocity_kms[2]);
  }
};

int update_simulation_domain_from_device(DeviceState* state,
                                         char* error_buffer,
                                         const std::size_t error_buffer_len) {
  try {
    thrust::device_ptr<SimCudaParticle> begin(state->particles);
    thrust::device_ptr<SimCudaParticle> end = begin + state->particle_count;
    const DomainBounds bounds = thrust::transform_reduce(
        thrust::cuda::par.on(state->compute_stream),
        begin,
        end,
        DomainBoundsAccessor{},
        empty_domain_bounds(),
        MergeDomainBounds{});
    for (int axis = 0; axis < 3; ++axis) {
      const double min_value = bounds.min_pos[axis];
      const double max_value = bounds.max_pos[axis];
      const double short_min_value = bounds.short_min_pos[axis];
      const double short_max_value = bounds.short_max_pos[axis];
      const double range = std::max(1.0, max_value - min_value);
      const double center = 0.5 * (min_value + max_value);
      const bool has_short_bounds = short_min_value <= short_max_value;
      const double short_range = std::max(
          1.0, (has_short_bounds ? short_max_value : max_value) - (has_short_bounds ? short_min_value : min_value));
      const double short_center = 0.5 *
                                  ((has_short_bounds ? short_min_value : min_value) +
                                   (has_short_bounds ? short_max_value : max_value));
      const double required_tight_length =
          std::max(kMinShortRangeBoxLengthKpc, short_range * kShortRangeDomainPadding);
      const double required_padded_length =
          std::max(kMinGlobalBoxLengthKpc, range * kGlobalDomainPadding);

      // Keep the global PM domain fixed unless particles genuinely outgrow it.
      // Re-centering/rescaling the FFT mesh every half-kick changes the live force law and
      // injects numerical heating into otherwise stable disks.
      const double global_center = state->domain_origin[axis] + 0.5 * state->box_length[axis];
      const double next_global_length = std::max(state->box_length[axis], required_padded_length);
      state->box_length[axis] = next_global_length;
      state->domain_origin[axis] = global_center - 0.5 * next_global_length;

      // The short-range grid can follow the active region, but only grow, not shrink,
      // so the short-range PM subtraction softening does not oscillate frame-to-frame.
      const double next_tight_length = std::max(state->tight_box_length[axis], required_tight_length);
      state->tight_box_length[axis] = next_tight_length;
      state->tight_domain_origin[axis] = short_center - 0.5 * next_tight_length;
    }
    state->cell_size[0] = state->box_length[0] / static_cast<double>(state->nx);
    state->cell_size[1] = state->box_length[1] / static_cast<double>(state->ny);
    state->cell_size[2] = state->box_length[2] / static_cast<double>(state->nz);
    state->cell_volume = state->cell_size[0] * state->cell_size[1] * state->cell_size[2];
  } catch (const std::exception& error) {
    fill_error(error_buffer, error_buffer_len, error.what());
    return 1;
  }
  return 0;
}

int ensure_short_range_cell_storage(DeviceState* state,
                                    const std::size_t required_cell_count,
                                    char* error_buffer,
                                    const std::size_t error_buffer_len) {
  if (required_cell_count == 0) {
    return 0;
  }
  if (required_cell_count <= state->short_cell_capacity &&
      state->short_cell_start != nullptr &&
      state->short_cell_end != nullptr) {
    return 0;
  }

  cudaFree(state->short_cell_start);
  cudaFree(state->short_cell_end);
  cudaFree(state->short_cell_mass);
  cudaFree(state->short_cell_com_x);
  cudaFree(state->short_cell_com_y);
  cudaFree(state->short_cell_com_z);
  cudaFree(state->short_cell_octant_mass);
  cudaFree(state->short_cell_octant_com_x);
  cudaFree(state->short_cell_octant_com_y);
  cudaFree(state->short_cell_octant_com_z);
  state->short_cell_start = nullptr;
  state->short_cell_end = nullptr;
  state->short_cell_mass = nullptr;
  state->short_cell_com_x = nullptr;
  state->short_cell_com_y = nullptr;
  state->short_cell_com_z = nullptr;
  state->short_cell_octant_mass = nullptr;
  state->short_cell_octant_com_x = nullptr;
  state->short_cell_octant_com_y = nullptr;
  state->short_cell_octant_com_z = nullptr;
  state->short_cell_capacity = 0;

  cudaError_t cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->short_cell_start),
                                       sizeof(int) * required_cell_count);
  if (cuda_status != cudaSuccess) {
    if (error_buffer != nullptr && error_buffer_len > 0) {
      std::snprintf(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range cell starts failed (%zu cells): %s",
                    required_cell_count,
                    cudaGetErrorString(cuda_status));
    }
    return 1;
  }

  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->short_cell_end),
                           sizeof(int) * required_cell_count);
  if (cuda_status != cudaSuccess) {
    if (error_buffer != nullptr && error_buffer_len > 0) {
      std::snprintf(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range cell ends failed (%zu cells): %s",
                    required_cell_count,
                    cudaGetErrorString(cuda_status));
    }
    return 1;
  }

  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->short_cell_mass),
                           sizeof(double) * required_cell_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range cell masses failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->short_cell_com_x),
                           sizeof(double) * required_cell_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range cell com_x failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->short_cell_com_y),
                           sizeof(double) * required_cell_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range cell com_y failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->short_cell_com_z),
                           sizeof(double) * required_cell_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range cell com_z failed",
                    cuda_status);
    return 1;
  }

  const std::size_t octant_count = required_cell_count * 8u;
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->short_cell_octant_mass),
                           sizeof(double) * octant_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range cell octant masses failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->short_cell_octant_com_x),
                           sizeof(double) * octant_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range cell octant com_x failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->short_cell_octant_com_y),
                           sizeof(double) * octant_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range cell octant com_y failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->short_cell_octant_com_z),
                           sizeof(double) * octant_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range cell octant com_z failed",
                    cuda_status);
    return 1;
  }

  state->short_cell_capacity = required_cell_count;
  return 0;
}

int ensure_short_range_tree_storage(DeviceState* state,
                                    const std::size_t required_node_count,
                                    char* error_buffer,
                                    const std::size_t error_buffer_len) {
  if (required_node_count == 0) {
    return 0;
  }
  if (required_node_count <= state->short_tree_node_capacity &&
      state->short_tree_nodes != nullptr) {
    return 0;
  }

  cudaFree(state->short_tree_nodes);
  state->short_tree_nodes = nullptr;
  state->short_tree_node_capacity = 0;

  const cudaError_t cuda_status =
      cudaMalloc(reinterpret_cast<void**>(&state->short_tree_nodes),
                 sizeof(ShortRangeTreeNode) * required_node_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range tree nodes failed",
                    cuda_status);
    return 1;
  }
  state->short_tree_node_capacity = required_node_count;
  return 0;
}

int ensure_short_range_interaction_storage(DeviceState* state,
                                           const std::size_t required_cell_count,
                                           const std::size_t required_interaction_count,
                                           char* error_buffer,
                                           const std::size_t error_buffer_len) {
  if (required_cell_count > state->short_cell_interaction_cell_capacity ||
      state->short_cell_interaction_start == nullptr) {
    cudaFree(state->short_cell_interaction_start);
    state->short_cell_interaction_start = nullptr;
    state->short_cell_interaction_cell_capacity = 0;

    const cudaError_t cuda_status =
        cudaMalloc(reinterpret_cast<void**>(&state->short_cell_interaction_start),
                   sizeof(int) * (required_cell_count + 1));
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer,
                      error_buffer_len,
                      "cudaMalloc for short-range interaction starts failed",
                      cuda_status);
      return 1;
    }
    state->short_cell_interaction_cell_capacity = required_cell_count;
  }

  if (required_interaction_count == 0) {
    cudaFree(state->short_cell_interactions);
    state->short_cell_interactions = nullptr;
    state->short_cell_interaction_capacity = 0;
    return 0;
  }

  if (required_interaction_count <= state->short_cell_interaction_capacity &&
      state->short_cell_interactions != nullptr) {
    return 0;
  }

  cudaFree(state->short_cell_interactions);
  state->short_cell_interactions = nullptr;
  state->short_cell_interaction_capacity = 0;

  const cudaError_t cuda_status =
      cudaMalloc(reinterpret_cast<void**>(&state->short_cell_interactions),
                 sizeof(ShortRangeInteractionSource) * required_interaction_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range interaction sources failed",
                    cuda_status);
    return 1;
  }
  state->short_cell_interaction_capacity = required_interaction_count;
  return 0;
}

void accumulate_tree_node_properties(const std::vector<ShortRangeSourceCell>& sources,
                                     const std::vector<int>& indices,
                                     ShortRangeTreeNode& node) {
  double mass = 0.0;
  double com_x = 0.0;
  double com_y = 0.0;
  double com_z = 0.0;
  double softening = 0.0;
  for (const int index : indices) {
    const ShortRangeSourceCell& source = sources[index];
    mass += source.mass;
    com_x += source.mass * source.com[0];
    com_y += source.mass * source.com[1];
    com_z += source.mass * source.com[2];
    softening = std::max(softening, source.softening_kpc);
  }
  if (mass > 0.0) {
    com_x /= mass;
    com_y /= mass;
    com_z /= mass;
  }
  node.mass = mass;
  node.com[0] = com_x;
  node.com[1] = com_y;
  node.com[2] = com_z;
  node.softening_kpc = softening;
}

int build_short_range_tree_recursive(const std::vector<ShortRangeSourceCell>& sources,
                                     const std::vector<int>& indices,
                                     const double center[3],
                                     const double half_size,
                                     const int depth,
                                     std::vector<ShortRangeTreeNode>& nodes) {
  const int node_index = static_cast<int>(nodes.size());
  nodes.emplace_back();
  ShortRangeTreeNode& node = nodes[node_index];
  node.center[0] = center[0];
  node.center[1] = center[1];
  node.center[2] = center[2];
  node.half_size = half_size;
  accumulate_tree_node_properties(sources, indices, node);

  if (indices.size() == 1 || depth >= 24 || half_size <= 1.0e-3) {
    if (indices.size() == 1) {
      const ShortRangeSourceCell& source = sources[indices.front()];
      node.cell_id = source.cell_id;
      node.softening_kpc = source.softening_kpc;
    }
    return node_index;
  }

  std::vector<int> buckets[8];
  for (const int index : indices) {
    const ShortRangeSourceCell& source = sources[index];
    int octant = 0;
    if (source.com[0] >= center[0]) {
      octant |= 1;
    }
    if (source.com[1] >= center[1]) {
      octant |= 2;
    }
    if (source.com[2] >= center[2]) {
      octant |= 4;
    }
    buckets[octant].push_back(index);
  }

  const double child_half = half_size * 0.5;
  for (int octant = 0; octant < 8; ++octant) {
    if (buckets[octant].empty()) {
      continue;
    }
    double child_center[3] = {
        center[0] + ((octant & 1) ? child_half : -child_half),
        center[1] + ((octant & 2) ? child_half : -child_half),
        center[2] + ((octant & 4) ? child_half : -child_half),
    };
    const int child_index = build_short_range_tree_recursive(
        sources, buckets[octant], child_center, child_half, depth + 1, nodes);
    nodes[node_index].child[octant] = child_index;
    nodes[node_index].child_mask |= static_cast<std::uint8_t>(1u << octant);
  }
  nodes[node_index].softening_kpc = std::max(nodes[node_index].softening_kpc, half_size * 0.5);
  return node_index;
}

void accumulate_tree_node_properties(const std::vector<ShortRangeParticleSource>& sources,
                                     const std::vector<int>& indices,
                                     ShortRangeTreeNode& node) {
  double mass = 0.0;
  double com_x = 0.0;
  double com_y = 0.0;
  double com_z = 0.0;
  double softening = 0.0;
  for (const int index : indices) {
    const ShortRangeParticleSource& source = sources[index];
    mass += source.mass;
    com_x += source.mass * source.position[0];
    com_y += source.mass * source.position[1];
    com_z += source.mass * source.position[2];
    softening = std::max(softening, source.softening_kpc);
  }
  if (mass > 0.0) {
    com_x /= mass;
    com_y /= mass;
    com_z /= mass;
  }
  node.mass = mass;
  node.com[0] = com_x;
  node.com[1] = com_y;
  node.com[2] = com_z;
  node.softening_kpc = softening;
}

int build_short_range_particle_tree_recursive(const std::vector<ShortRangeParticleSource>& sources,
                                              const std::vector<int>& indices,
                                              const double center[3],
                                              const double half_size,
                                              const int depth,
                                              std::vector<ShortRangeTreeNode>& nodes) {
  const int node_index = static_cast<int>(nodes.size());
  nodes.emplace_back();
  ShortRangeTreeNode& node = nodes[node_index];
  node.center[0] = center[0];
  node.center[1] = center[1];
  node.center[2] = center[2];
  node.half_size = half_size;
  accumulate_tree_node_properties(sources, indices, node);

  if (indices.size() == 1 || depth >= 32 || half_size <= 1.0e-4) {
    if (indices.size() == 1) {
      const ShortRangeParticleSource& source = sources[indices.front()];
      node.particle_index = source.particle_index;
      node.softening_kpc = source.softening_kpc;
    }
    return node_index;
  }

  std::vector<int> buckets[8];
  for (const int index : indices) {
    const ShortRangeParticleSource& source = sources[index];
    int octant = 0;
    if (source.position[0] >= center[0]) {
      octant |= 1;
    }
    if (source.position[1] >= center[1]) {
      octant |= 2;
    }
    if (source.position[2] >= center[2]) {
      octant |= 4;
    }
    buckets[octant].push_back(index);
  }

  const double child_half = half_size * 0.5;
  for (int octant = 0; octant < 8; ++octant) {
    if (buckets[octant].empty()) {
      continue;
    }
    double child_center[3] = {
        center[0] + ((octant & 1) ? child_half : -child_half),
        center[1] + ((octant & 2) ? child_half : -child_half),
        center[2] + ((octant & 4) ? child_half : -child_half),
    };
    const int child_index = build_short_range_particle_tree_recursive(
        sources, buckets[octant], child_center, child_half, depth + 1, nodes);
    nodes[node_index].child[octant] = child_index;
    nodes[node_index].child_mask |= static_cast<std::uint8_t>(1u << octant);
  }
  nodes[node_index].softening_kpc = std::max(nodes[node_index].softening_kpc, half_size * 0.25);
  return node_index;
}

void append_short_range_cell_interactions(const DeviceState* state,
                                          const int cell_id,
                                          std::vector<ShortRangeInteractionSource>& out) {
  if (state->short_tree_root < 0 || state->short_tree_host.empty()) {
    return;
  }

  const int cell_plane = state->short_ny * state->short_nz;
  const int ix = cell_id / cell_plane;
  const int rem = cell_id - ix * cell_plane;
  const int iy = rem / state->short_nz;
  const int iz = rem - iy * state->short_nz;
  const double half_x = 0.5 * state->short_cell_size[0];
  const double half_y = 0.5 * state->short_cell_size[1];
  const double half_z = 0.5 * state->short_cell_size[2];
  const double center_x =
      state->short_domain_origin[0] + (static_cast<double>(ix) + 0.5) * state->short_cell_size[0];
  const double center_y =
      state->short_domain_origin[1] + (static_cast<double>(iy) + 0.5) * state->short_cell_size[1];
  const double center_z =
      state->short_domain_origin[2] + (static_cast<double>(iz) + 0.5) * state->short_cell_size[2];
  const double target_half_diag = std::sqrt(half_x * half_x + half_y * half_y + half_z * half_z);
  const double short_cutoff_sq = state->short_cutoff_kpc * state->short_cutoff_kpc;
  const double theta_sq = state->opening_angle * state->opening_angle;

  std::vector<int> stack;
  stack.reserve(128);
  stack.push_back(state->short_tree_root);

  while (!stack.empty()) {
    const int node_index = stack.back();
    stack.pop_back();
    if (node_index < 0 || static_cast<std::size_t>(node_index) >= state->short_tree_host.size()) {
      continue;
    }

    const ShortRangeTreeNode& node = state->short_tree_host[static_cast<std::size_t>(node_index)];
    if (!(node.mass > 0.0)) {
      continue;
    }

    const bool is_leaf = node.child_mask == 0;
    if (is_leaf && node.cell_id == cell_id) {
      continue;
    }

    const double dx_aabb = std::max(std::fabs(center_x - node.center[0]) - (node.half_size + half_x), 0.0);
    const double dy_aabb = std::max(std::fabs(center_y - node.center[1]) - (node.half_size + half_y), 0.0);
    const double dz_aabb = std::max(std::fabs(center_z - node.center[2]) - (node.half_size + half_z), 0.0);
    if (dx_aabb * dx_aabb + dy_aabb * dy_aabb + dz_aabb * dz_aabb > short_cutoff_sq) {
      continue;
    }

    const double dx = node.com[0] - center_x;
    const double dy = node.com[1] - center_y;
    const double dz = node.com[2] - center_z;
    const double r2 = dx * dx + dy * dy + dz * dz;
    const double combined_half = node.half_size + target_half_diag;
    const double size_over_r_sq =
        r2 > 1.0e-12 ? (4.0 * combined_half * combined_half) / r2
                     : std::numeric_limits<double>::infinity();
    if (!is_leaf && size_over_r_sq > theta_sq) {
      for (int child = 0; child < 8; ++child) {
        if ((node.child_mask & static_cast<std::uint8_t>(1u << child)) != 0) {
          stack.push_back(node.child[child]);
        }
      }
      continue;
    }

    if (r2 <= 1.0e-12 || r2 > short_cutoff_sq) {
      continue;
    }

    ShortRangeInteractionSource interaction{};
    interaction.mass = node.mass;
    interaction.com[0] = node.com[0];
    interaction.com[1] = node.com[1];
    interaction.com[2] = node.com[2];
    interaction.softening_kpc = node.softening_kpc;
    out.push_back(interaction);
  }
}

int build_short_range_cell_interactions(DeviceState* state,
                                        char* error_buffer,
                                        const std::size_t error_buffer_len) {
  state->short_cell_interaction_start_host.assign(state->short_cell_count + 1, 0);
  state->short_cell_interactions_host.clear();
  if (state->short_cell_count == 0 || state->short_tree_root < 0 || state->short_tree_host.empty()) {
    return ensure_short_range_interaction_storage(
        state, state->short_cell_count, 0, error_buffer, error_buffer_len);
  }

  state->short_cell_interactions_host.reserve(state->short_tree_host.size() * 8u);
  int interaction_cursor = 0;
  for (std::size_t cell_id = 0; cell_id < state->short_cell_count; ++cell_id) {
    state->short_cell_interaction_start_host[cell_id] = interaction_cursor;
    if (!(state->short_cell_mass_host[cell_id] > 0.0)) {
      continue;
    }
    append_short_range_cell_interactions(
        state, static_cast<int>(cell_id), state->short_cell_interactions_host);
    interaction_cursor = static_cast<int>(state->short_cell_interactions_host.size());
  }
  state->short_cell_interaction_start_host[state->short_cell_count] = interaction_cursor;

  if (ensure_short_range_interaction_storage(state,
                                             state->short_cell_count,
                                             state->short_cell_interactions_host.size(),
                                             error_buffer,
                                             error_buffer_len) != 0) {
    return 1;
  }

  cudaError_t cuda_status =
      cudaMemcpyAsync(state->short_cell_interaction_start,
                      state->short_cell_interaction_start_host.data(),
                      sizeof(int) * (state->short_cell_count + 1),
                      cudaMemcpyHostToDevice,
                      state->compute_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(
        error_buffer, error_buffer_len, "short-range interaction starts upload failed", cuda_status);
    return 1;
  }

  if (!state->short_cell_interactions_host.empty()) {
    cuda_status = cudaMemcpyAsync(state->short_cell_interactions,
                                  state->short_cell_interactions_host.data(),
                                  sizeof(ShortRangeInteractionSource) *
                                      state->short_cell_interactions_host.size(),
                                  cudaMemcpyHostToDevice,
                                  state->compute_stream);
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer,
                      error_buffer_len,
                      "short-range interaction sources upload failed",
                      cuda_status);
      return 1;
    }
  }
  return 0;
}

int build_short_range_tree(DeviceState* state,
                           char* error_buffer,
                           const std::size_t error_buffer_len) {
  state->short_tree_root = -1;
  state->short_tree_node_count = 0;
  state->short_tree_particle_mode = false;
  state->short_tree_host.clear();
  if (state->short_cell_count == 0) {
    return 0;
  }

  if (state->short_source_particle_count <= particle_tree_threshold()) {
    state->particles_host.resize(state->particle_count);
    cudaError_t particle_status =
        cudaMemcpyAsync(state->particles_host.data(),
                        state->particles,
                        sizeof(SimCudaParticle) * state->particle_count,
                        cudaMemcpyDeviceToHost,
                        state->compute_stream);
    if (particle_status == cudaSuccess) {
      particle_status = cudaStreamSynchronize(state->compute_stream);
    }
    if (particle_status != cudaSuccess) {
      fill_cuda_error(error_buffer,
                      error_buffer_len,
                      "particle download for short-range tree failed",
                      particle_status);
      return 1;
    }

    std::vector<ShortRangeParticleSource> sources;
    sources.reserve(state->short_source_particle_count);
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
    for (std::uint64_t particle_index = 0; particle_index < state->particle_count; ++particle_index) {
      const SimCudaParticle& particle = state->particles_host[particle_index];
      if (particle.component == 3u || !(particle.mass_msun > 0.0)) {
        continue;
      }
      ShortRangeParticleSource source;
      source.particle_index = static_cast<int>(particle_index);
      source.mass = particle.mass_msun;
      source.position[0] = particle.position_kpc[0];
      source.position[1] = particle.position_kpc[1];
      source.position[2] = particle.position_kpc[2];
      source.softening_kpc = particle.softening_kpc;
      sources.push_back(source);
      for (int axis = 0; axis < 3; ++axis) {
        min_pos[axis] = std::min(min_pos[axis], source.position[axis]);
        max_pos[axis] = std::max(max_pos[axis], source.position[axis]);
      }
    }

    if (!sources.empty()) {
      double center[3] = {0.0, 0.0, 0.0};
      double span = 0.0;
      for (int axis = 0; axis < 3; ++axis) {
        center[axis] = 0.5 * (min_pos[axis] + max_pos[axis]);
        span = std::max(span, max_pos[axis] - min_pos[axis]);
      }
      const double half_size = std::max(
          0.5 * span + std::max(state->short_cell_size[0],
                                std::max(state->short_cell_size[1], state->short_cell_size[2])),
          2.0 * state->max_softening_kpc);

      std::vector<int> indices(sources.size());
      for (std::size_t i = 0; i < sources.size(); ++i) {
        indices[i] = static_cast<int>(i);
      }
      state->short_tree_host.reserve(sources.size() * 2);
      state->short_tree_root = build_short_range_particle_tree_recursive(
          sources, indices, center, half_size, 0, state->short_tree_host);
      state->short_tree_node_count = static_cast<std::uint32_t>(state->short_tree_host.size());
      state->short_tree_particle_mode = true;
    }
  }

  if (state->short_tree_particle_mode) {
    if (ensure_short_range_tree_storage(state,
                                        state->short_tree_node_count,
                                        error_buffer,
                                        error_buffer_len) != 0) {
      return 1;
    }
    const cudaError_t cuda_status =
        cudaMemcpyAsync(state->short_tree_nodes,
                        state->short_tree_host.data(),
                        sizeof(ShortRangeTreeNode) * state->short_tree_node_count,
                        cudaMemcpyHostToDevice,
                        state->compute_stream);
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "short-range particle tree upload failed", cuda_status);
      return 1;
    }
    return 0;
  }
  return ensure_short_range_interaction_storage(
      state, state->short_cell_count, 0, error_buffer, error_buffer_len);
}

int update_short_range_grid(DeviceState* state,
                            char* error_buffer,
                            const std::size_t error_buffer_len) {
  for (int axis = 0; axis < 3; ++axis) {
    if (!std::isfinite(state->tight_box_length[axis]) || state->tight_box_length[axis] <= 0.0) {
      if (error_buffer != nullptr && error_buffer_len > 0) {
        std::snprintf(error_buffer,
                      error_buffer_len,
                      "invalid short-range box length on axis %d: %.6e",
                      axis,
                      state->tight_box_length[axis]);
      }
      return 1;
    }
  }
  const double volume = std::max(1.0,
                                 state->tight_box_length[0] *
                                     state->tight_box_length[1] *
                                     state->tight_box_length[2]);
  const double target_cells = std::clamp(
      static_cast<double>(state->particle_count) /
          short_range_target_occupancy(state->particle_count),
      1024.0,
      static_cast<double>(kMaxShortRangeCells));
  double target_cell_size = cbrt(volume / std::max(1.0, target_cells));
  target_cell_size = std::clamp(target_cell_size,
                                std::max(kMinShortRangeCellSizeKpc, state->max_softening_kpc * 4.0),
                                kMaxShortRangeCellSizeKpc);
  if (!std::isfinite(target_cell_size) || target_cell_size <= 0.0) {
    if (error_buffer != nullptr && error_buffer_len > 0) {
      std::snprintf(error_buffer,
                    error_buffer_len,
                    "invalid short-range cell size from volume %.6e and target_cells %.6e",
                    volume,
                    target_cells);
    }
    return 1;
  }

  int short_nx = 0;
  int short_ny = 0;
  int short_nz = 0;
  std::size_t short_cell_count = 0;
  double cell_size = target_cell_size;
  for (int axis = 0; axis < 3; ++axis) {
    cell_size = std::max(cell_size, state->tight_box_length[axis] / static_cast<double>(kMaxShortRangeAxisCells));
  }
  int guard_iterations = 0;
  while (true) {
    short_nx = std::max(4, static_cast<int>(ceil(state->tight_box_length[0] / cell_size)));
    short_ny = std::max(4, static_cast<int>(ceil(state->tight_box_length[1] / cell_size)));
    short_nz = std::max(4, static_cast<int>(ceil(state->tight_box_length[2] / cell_size)));
    short_cell_count = static_cast<std::size_t>(short_nx) * static_cast<std::size_t>(short_ny) *
                       static_cast<std::size_t>(short_nz);
    if (short_cell_count <= kMaxShortRangeCells) {
      break;
    }
    cell_size *= 1.1;
    if (++guard_iterations > 64 || !std::isfinite(cell_size)) {
      if (error_buffer != nullptr && error_buffer_len > 0) {
        std::snprintf(error_buffer,
                      error_buffer_len,
                      "failed to size short-range grid: lengths=(%.6e, %.6e, %.6e) cell_size=%.6e cells=(%d,%d,%d)",
                      state->tight_box_length[0],
                      state->tight_box_length[1],
                      state->tight_box_length[2],
                      cell_size,
                      short_nx,
                      short_ny,
                      short_nz);
      }
      return 1;
    }
  }

  if (ensure_short_range_cell_storage(state, short_cell_count + 1, error_buffer, error_buffer_len) != 0) {
    return 1;
  }

  state->short_nx = short_nx;
  state->short_ny = short_ny;
  state->short_nz = short_nz;
  state->short_cell_count = short_cell_count;
  for (int axis = 0; axis < 3; ++axis) {
    state->short_domain_origin[axis] = state->tight_domain_origin[axis];
    state->short_box_length[axis] = state->tight_box_length[axis];
  }
  state->short_cell_size[0] = state->short_box_length[0] / static_cast<double>(state->short_nx);
  state->short_cell_size[1] = state->short_box_length[1] / static_cast<double>(state->short_ny);
  state->short_cell_size[2] = state->short_box_length[2] / static_cast<double>(state->short_nz);
  const double max_short_cell = std::max(state->short_cell_size[0],
                                         std::max(state->short_cell_size[1], state->short_cell_size[2]));
  const double max_pm_cell =
      std::max(state->cell_size[0], std::max(state->cell_size[1], state->cell_size[2]));
  state->short_pm_softening_kpc =
      std::max(max_pm_cell * 1.25, 2.0 * state->max_softening_kpc);
  state->short_cutoff_kpc =
      std::max(4.5 * state->short_pm_softening_kpc, 1.5 * max_short_cell);

  if (state->short_force_factor_lut == nullptr) {
    const cudaError_t cuda_status =
        cudaMalloc(reinterpret_cast<void**>(&state->short_force_factor_lut),
                   sizeof(float) * kShortRangeForceFactorLutSize);
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(
          error_buffer, error_buffer_len, "cudaMalloc for short-range factor LUT failed", cuda_status);
      return 1;
    }
    state->short_force_factor_lut_host.resize(kShortRangeForceFactorLutSize);
  }

  if (state->short_force_factor_lut_host.size() != kShortRangeForceFactorLutSize) {
    state->short_force_factor_lut_host.resize(kShortRangeForceFactorLutSize);
  }

  const double short_cutoff_sq = state->short_cutoff_kpc * state->short_cutoff_kpc;
  state->short_force_factor_lut_scale =
      short_cutoff_sq > 0.0
          ? static_cast<double>(kShortRangeForceFactorLutSize - 1) / short_cutoff_sq
          : 0.0;
  for (std::uint32_t i = 0; i < kShortRangeForceFactorLutSize; ++i) {
    const double t = static_cast<double>(i) /
                     static_cast<double>(std::max<std::uint32_t>(1, kShortRangeForceFactorLutSize - 1));
    const double r = sqrt(t * short_cutoff_sq);
    state->short_force_factor_lut_host[i] = static_cast<float>(
        treepm_short_range_force_factor(r, state->short_pm_softening_kpc));
  }

  const cudaError_t lut_copy_status =
      cudaMemcpyAsync(state->short_force_factor_lut,
                      state->short_force_factor_lut_host.data(),
                      sizeof(float) * kShortRangeForceFactorLutSize,
                      cudaMemcpyHostToDevice,
                      state->compute_stream);
  if (lut_copy_status != cudaSuccess) {
    fill_cuda_error(
        error_buffer, error_buffer_len, "short-range factor LUT upload failed", lut_copy_status);
    return 1;
  }
  return 0;
}

std::uint32_t estimate_substeps_for_step(DeviceState* state,
                                         const double dt_myr,
                                         char* error_buffer,
                                         const std::size_t error_buffer_len) {
  try {
    thrust::device_ptr<SimCudaParticle> begin(state->particles);
    thrust::device_ptr<SimCudaParticle> end = begin + state->particle_count;
    const double max_speed_kms = thrust::transform_reduce(
        thrust::cuda::par.on(state->compute_stream),
        begin,
        end,
        SpeedAccessor{},
        0.0,
        thrust::maximum<double>());
    const double min_global_cell = std::max(
        1.0e-4, std::min(state->cell_size[0], std::min(state->cell_size[1], state->cell_size[2])));
    double integration_scale = min_global_cell;
    if (state->short_cell_count > 0) {
      const double min_short_cell =
          std::max(1.0e-4,
                   std::min(state->short_cell_size[0],
                            std::min(state->short_cell_size[1], state->short_cell_size[2])));
      // The near-field correction varies on a much smaller scale than the global PM mesh.
      // Keep substeps small enough that particles traverse at most about a quarter of a
      // short-range cell per kick, otherwise dense regions heat numerically.
      integration_scale = std::min(integration_scale, 0.25 * min_short_cell);
    }
    const double allowed_displacement =
        state->cfl_safety_factor * std::max(1.0e-4, integration_scale);
    const double predicted_displacement = max_speed_kms * dt_myr * kKpcPerKmPerMyr;
    const double raw_substeps = predicted_displacement / std::max(allowed_displacement, 1.0e-6);
    const std::uint32_t substeps = static_cast<std::uint32_t>(
        std::clamp(std::ceil(raw_substeps), 1.0, static_cast<double>(std::max(1u, state->max_substeps))));
    return std::max(1u, substeps);
  } catch (const std::exception& error) {
    fill_error(error_buffer, error_buffer_len, error.what());
    return std::max(1u, state->max_substeps);
  }
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

__host__ __device__ __forceinline__ double treepm_short_range_force_factor(
    const double r, const double split_radius_kpc) {
  if (!(split_radius_kpc > 0.0) || !(r > 0.0)) {
    return 1.0;
  }
  const double x = 0.5 * r / split_radius_kpc;
  return erfc(x) + (r / (kSqrtPi * split_radius_kpc)) * exp(-(x * x));
}

__device__ __forceinline__ double treepm_short_range_force_factor_lookup(
    const float* lut,
    const std::uint32_t lut_size,
    const double lut_scale,
    const double r2,
    const double split_radius_kpc) {
  if (lut == nullptr || lut_size < 2 || !(lut_scale > 0.0)) {
    return treepm_short_range_force_factor(sqrt(r2), split_radius_kpc);
  }

  const double scaled = fmin(r2 * lut_scale, static_cast<double>(lut_size - 1));
  const std::uint32_t index0 = static_cast<std::uint32_t>(scaled);
  const std::uint32_t index1 = min(index0 + 1u, lut_size - 1);
  const double frac = scaled - static_cast<double>(index0);
  const double value0 = static_cast<double>(lut[index0]);
  const double value1 = static_cast<double>(lut[index1]);
  return value0 + (value1 - value0) * frac;
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

__device__ __forceinline__ void sample_grid_trilinear_vector(const cufftReal* grid_x,
                                                             const cufftReal* grid_y,
                                                             const cufftReal* grid_z,
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
                                                             const double pz,
                                                             double& out_x,
                                                             double& out_y,
                                                             double& out_z) {
  const double gx = fmin(fmax((px - origin_x) / cell_x, 0.0), static_cast<double>(nx) - 1.000001);
  const double gy = fmin(fmax((py - origin_y) / cell_y, 0.0), static_cast<double>(ny) - 1.000001);
  const double gz = fmin(fmax((pz - origin_z) / cell_z, 0.0), static_cast<double>(nz) - 1.000001);

  const int i0 = static_cast<int>(floor(gx));
  const int j0 = static_cast<int>(floor(gy));
  const int k0 = static_cast<int>(floor(gz));
  const int i1 = clamp_index(i0 + 1, nx);
  const int j1 = clamp_index(j0 + 1, ny);
  const int k1 = clamp_index(k0 + 1, nz);

  const double tx = gx - floor(gx);
  const double ty = gy - floor(gy);
  const double tz = gz - floor(gz);
  const double wx0 = 1.0 - tx;
  const double wy0 = 1.0 - ty;
  const double wz0 = 1.0 - tz;
  const double wx1 = tx;
  const double wy1 = ty;
  const double wz1 = tz;

  const std::size_t idx000 = real_grid_index(i0, j0, k0, ny, nz);
  const std::size_t idx001 = real_grid_index(i0, j0, k1, ny, nz);
  const std::size_t idx010 = real_grid_index(i0, j1, k0, ny, nz);
  const std::size_t idx011 = real_grid_index(i0, j1, k1, ny, nz);
  const std::size_t idx100 = real_grid_index(i1, j0, k0, ny, nz);
  const std::size_t idx101 = real_grid_index(i1, j0, k1, ny, nz);
  const std::size_t idx110 = real_grid_index(i1, j1, k0, ny, nz);
  const std::size_t idx111 = real_grid_index(i1, j1, k1, ny, nz);

  const double w000 = wx0 * wy0 * wz0;
  const double w001 = wx0 * wy0 * wz1;
  const double w010 = wx0 * wy1 * wz0;
  const double w011 = wx0 * wy1 * wz1;
  const double w100 = wx1 * wy0 * wz0;
  const double w101 = wx1 * wy0 * wz1;
  const double w110 = wx1 * wy1 * wz0;
  const double w111 = wx1 * wy1 * wz1;

  out_x = grid_x[idx000] * w000 + grid_x[idx001] * w001 + grid_x[idx010] * w010 +
          grid_x[idx011] * w011 + grid_x[idx100] * w100 + grid_x[idx101] * w101 +
          grid_x[idx110] * w110 + grid_x[idx111] * w111;
  out_y = grid_y[idx000] * w000 + grid_y[idx001] * w001 + grid_y[idx010] * w010 +
          grid_y[idx011] * w011 + grid_y[idx100] * w100 + grid_y[idx101] * w101 +
          grid_y[idx110] * w110 + grid_y[idx111] * w111;
  out_z = grid_z[idx000] * w000 + grid_z[idx001] * w001 + grid_z[idx010] * w010 +
          grid_z[idx011] * w011 + grid_z[idx100] * w100 + grid_z[idx101] * w101 +
          grid_z[idx110] * w110 + grid_z[idx111] * w111;
}

__device__ __forceinline__ bool sample_grid_trilinear_vector_local(const cufftReal* grid_x,
                                                                   const cufftReal* grid_y,
                                                                   const cufftReal* grid_z,
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
                                                                   const double pz,
                                                                   double& out_x,
                                                                   double& out_y,
                                                                   double& out_z) {
  const double gx = (px - origin_x) / cell_x;
  const double gy = (py - origin_y) / cell_y;
  const double gz = (pz - origin_z) / cell_z;
  if (gx < 0.0 || gy < 0.0 || gz < 0.0 ||
      gx >= static_cast<double>(nx - 1) ||
      gy >= static_cast<double>(ny - 1) ||
      gz >= static_cast<double>(nz - 1)) {
    out_x = 0.0;
    out_y = 0.0;
    out_z = 0.0;
    return false;
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

  const std::size_t idx000 = real_grid_index(i0, j0, k0, ny, nz);
  const std::size_t idx001 = real_grid_index(i0, j0, k1, ny, nz);
  const std::size_t idx010 = real_grid_index(i0, j1, k0, ny, nz);
  const std::size_t idx011 = real_grid_index(i0, j1, k1, ny, nz);
  const std::size_t idx100 = real_grid_index(i1, j0, k0, ny, nz);
  const std::size_t idx101 = real_grid_index(i1, j0, k1, ny, nz);
  const std::size_t idx110 = real_grid_index(i1, j1, k0, ny, nz);
  const std::size_t idx111 = real_grid_index(i1, j1, k1, ny, nz);

  const double w000 = wx0 * wy0 * wz0;
  const double w001 = wx0 * wy0 * wz1;
  const double w010 = wx0 * wy1 * wz0;
  const double w011 = wx0 * wy1 * wz1;
  const double w100 = wx1 * wy0 * wz0;
  const double w101 = wx1 * wy0 * wz1;
  const double w110 = wx1 * wy1 * wz0;
  const double w111 = wx1 * wy1 * wz1;

  out_x = grid_x[idx000] * w000 + grid_x[idx001] * w001 + grid_x[idx010] * w010 +
          grid_x[idx011] * w011 + grid_x[idx100] * w100 + grid_x[idx101] * w101 +
          grid_x[idx110] * w110 + grid_x[idx111] * w111;
  out_y = grid_y[idx000] * w000 + grid_y[idx001] * w001 + grid_y[idx010] * w010 +
          grid_y[idx011] * w011 + grid_y[idx100] * w100 + grid_y[idx101] * w101 +
          grid_y[idx110] * w110 + grid_y[idx111] * w111;
  out_z = grid_z[idx000] * w000 + grid_z[idx001] * w001 + grid_z[idx010] * w010 +
          grid_z[idx011] * w011 + grid_z[idx100] * w100 + grid_z[idx101] * w101 +
          grid_z[idx110] * w110 + grid_z[idx111] * w111;
  return true;
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

  const double gx = fmin(
      fmax((particle.position_kpc[0] - origin_x) / cell_x, 0.0),
      static_cast<double>(nx) - 1.000001);
  const double gy = fmin(
      fmax((particle.position_kpc[1] - origin_y) / cell_y, 0.0),
      static_cast<double>(ny) - 1.000001);
  const double gz = fmin(
      fmax((particle.position_kpc[2] - origin_z) / cell_z, 0.0),
      static_cast<double>(nz) - 1.000001);

  const int i0 = static_cast<int>(floor(gx));
  const int j0 = static_cast<int>(floor(gy));
  const int k0 = static_cast<int>(floor(gz));
  const int i1 = clamp_index(i0 + 1, nx);
  const int j1 = clamp_index(j0 + 1, ny);
  const int k1 = clamp_index(k0 + 1, nz);

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

__host__ __device__ __forceinline__ int short_cell_linear_index(const int ix,
                                                                const int iy,
                                                                const int iz,
                                                                const int ny,
                                                                const int nz) {
  return (ix * ny + iy) * nz + iz;
}

__global__ void compute_short_range_cells(const SimCudaParticle* particles,
                                          const int* source_particle_indices,
                                          const std::uint32_t source_particle_count,
                                          int* sorted_cell_ids,
                                          int* sorted_particle_indices,
                                          double* cell_mass,
                                          double* cell_com_x,
                                          double* cell_com_y,
                                          double* cell_com_z,
                                          double* octant_mass,
                                          double* octant_com_x,
                                          double* octant_com_y,
                                          double* octant_com_z,
                                          const int nx,
                                          const int ny,
                                          const int nz,
                                          const double origin_x,
                                          const double origin_y,
                                          const double origin_z,
                                          const double cell_x,
                                          const double cell_y,
                                          const double cell_z) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= source_particle_count) {
    return;
  }

  const int particle_index = source_particle_indices[index];
  const SimCudaParticle& particle = particles[particle_index];
  const double gx = fmin(
      fmax((particle.position_kpc[0] - origin_x) / cell_x, 0.0),
      static_cast<double>(nx) - 1.000001);
  const double gy = fmin(
      fmax((particle.position_kpc[1] - origin_y) / cell_y, 0.0),
      static_cast<double>(ny) - 1.000001);
  const double gz = fmin(
      fmax((particle.position_kpc[2] - origin_z) / cell_z, 0.0),
      static_cast<double>(nz) - 1.000001);
  const int ix = static_cast<int>(floor(gx));
  const int iy = static_cast<int>(floor(gy));
  const int iz = static_cast<int>(floor(gz));

  const int cell_id = short_cell_linear_index(ix, iy, iz, ny, nz);
  sorted_cell_ids[index] = cell_id;
  sorted_particle_indices[index] = particle_index;
  atomicAdd(&cell_mass[cell_id], particle.mass_msun);
  atomicAdd(&cell_com_x[cell_id], particle.mass_msun * particle.position_kpc[0]);
  atomicAdd(&cell_com_y[cell_id], particle.mass_msun * particle.position_kpc[1]);
  atomicAdd(&cell_com_z[cell_id], particle.mass_msun * particle.position_kpc[2]);

  const double center_x = origin_x + (static_cast<double>(ix) + 0.5) * cell_x;
  const double center_y = origin_y + (static_cast<double>(iy) + 0.5) * cell_y;
  const double center_z = origin_z + (static_cast<double>(iz) + 0.5) * cell_z;
  const int octant = (particle.position_kpc[0] >= center_x ? 1 : 0) |
                     (particle.position_kpc[1] >= center_y ? 2 : 0) |
                     (particle.position_kpc[2] >= center_z ? 4 : 0);
  const std::size_t slot =
      static_cast<std::size_t>(cell_id) * 8u + static_cast<std::size_t>(octant);
  atomicAdd(&octant_mass[slot], particle.mass_msun);
  atomicAdd(&octant_com_x[slot], particle.mass_msun * particle.position_kpc[0]);
  atomicAdd(&octant_com_y[slot], particle.mass_msun * particle.position_kpc[1]);
  atomicAdd(&octant_com_z[slot], particle.mass_msun * particle.position_kpc[2]);
}

__global__ void build_short_range_cell_ranges(const int* sorted_cell_ids,
                                              const std::uint64_t particle_count,
                                              int* cell_start,
                                              int* cell_end) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= particle_count) {
    return;
  }

  const int cell_id = sorted_cell_ids[index];
  if (index == 0 || sorted_cell_ids[index - 1] != cell_id) {
    cell_start[cell_id] = static_cast<int>(index);
  }
  if (index + 1 == particle_count || sorted_cell_ids[index + 1] != cell_id) {
    cell_end[cell_id] = static_cast<int>(index + 1);
  }
}

__global__ void normalize_short_range_cell_moments(const std::size_t cell_count,
                                                   double* cell_mass,
                                                   double* cell_com_x,
                                                   double* cell_com_y,
                                                   double* cell_com_z) {
  const std::size_t cell_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (cell_index >= cell_count) {
    return;
  }

  const double mass = cell_mass[cell_index];

  if (mass > 0.0) {
    cell_com_x[cell_index] /= mass;
    cell_com_y[cell_index] /= mass;
    cell_com_z[cell_index] /= mass;
  } else {
    cell_com_x[cell_index] = 0.0;
    cell_com_y[cell_index] = 0.0;
    cell_com_z[cell_index] = 0.0;
  }
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
                                         const double split_radius_kpc,
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
  const float window_sq = window * window;
  const float deconvolution = 1.0f / fmaxf(window_sq * window_sq, 1.0e-4f);
  const float split = static_cast<float>(split_radius_kpc);
  const float long_range_filter = expf(-(k_squared * split * split));
  const float scale =
      -static_cast<float>(kFourPi) * grav_const * deconvolution * long_range_filter / k_squared;

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

  const int ixmm = wrap_index(ix - 2, nx);
  const int ixpp = wrap_index(ix + 2, nx);
  const int iymm = wrap_index(iy - 2, ny);
  const int iypp = wrap_index(iy + 2, ny);
  const int izmm = wrap_index(iz - 2, nz);
  const int izpp = wrap_index(iz + 2, nz);

  const float phi_xmm = potential_grid[real_grid_index(ixmm, iy, iz, ny, nz)];
  const float phi_xm = potential_grid[real_grid_index(ixm, iy, iz, ny, nz)];
  const float phi_xp = potential_grid[real_grid_index(ixp, iy, iz, ny, nz)];
  const float phi_xpp = potential_grid[real_grid_index(ixpp, iy, iz, ny, nz)];
  const float phi_ymm = potential_grid[real_grid_index(ix, iymm, iz, ny, nz)];
  const float phi_ym = potential_grid[real_grid_index(ix, iym, iz, ny, nz)];
  const float phi_yp = potential_grid[real_grid_index(ix, iyp, iz, ny, nz)];
  const float phi_ypp = potential_grid[real_grid_index(ix, iypp, iz, ny, nz)];
  const float phi_zmm = potential_grid[real_grid_index(ix, iy, izmm, ny, nz)];
  const float phi_zm = potential_grid[real_grid_index(ix, iy, izm, ny, nz)];
  const float phi_zp = potential_grid[real_grid_index(ix, iy, izp, ny, nz)];
  const float phi_zpp = potential_grid[real_grid_index(ix, iy, izpp, ny, nz)];

  force_x[index] =
      -((2.0f / 3.0f) * (phi_xp - phi_xm) - (1.0f / 12.0f) * (phi_xpp - phi_xmm)) /
      static_cast<float>(cell_x);
  force_y[index] =
      -((2.0f / 3.0f) * (phi_yp - phi_ym) - (1.0f / 12.0f) * (phi_ypp - phi_ymm)) /
      static_cast<float>(cell_y);
  force_z[index] =
      -((2.0f / 3.0f) * (phi_zp - phi_zm) - (1.0f / 12.0f) * (phi_zpp - phi_zmm)) /
      static_cast<float>(cell_z);
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

__global__ void kick_particles_global(SimCudaParticle* __restrict__ particles,
                                      const std::uint64_t particle_count,
                                      const cufftReal* __restrict__ force_x,
                                      const cufftReal* __restrict__ force_y,
                                      const cufftReal* __restrict__ force_z,
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
                                      const int* __restrict__ short_sorted_particle_indices,
                                      const int* __restrict__ short_cell_start,
                                      const int* __restrict__ short_cell_end,
                                      const int* __restrict__ short_cell_interaction_start,
                                      const double* __restrict__ short_cell_mass,
                                      const double* __restrict__ short_cell_com_x,
                                      const double* __restrict__ short_cell_com_y,
                                      const double* __restrict__ short_cell_com_z,
                                      const double* __restrict__ short_cell_octant_mass,
                                      const double* __restrict__ short_cell_octant_com_x,
                                      const double* __restrict__ short_cell_octant_com_y,
                                      const double* __restrict__ short_cell_octant_com_z,
                                      const ShortRangeInteractionSource* __restrict__ short_cell_interactions,
                                      const float* __restrict__ short_force_factor_lut,
                                      const std::uint32_t short_force_factor_lut_size,
                                      const double short_force_factor_lut_scale,
                                      const ShortRangeTreeNode* __restrict__ short_tree_nodes,
                                      const int short_tree_root,
                                      const std::uint32_t short_tree_node_count,
                                      const int short_tree_particle_mode,
                                      const int short_nx,
                                      const int short_ny,
                                      const int short_nz,
                                      const double short_origin_x,
                                      const double short_origin_y,
                                      const double short_origin_z,
                                      const double short_cell_x,
                                      const double short_cell_y,
                                      const double short_cell_z,
                                      const double short_cutoff_kpc,
                                      const double short_pm_softening_kpc,
                                      const double opening_angle,
                                      const int short_range_target_baryons_only,
                                      const int* __restrict__ galaxy_smbh_indices,
                                      const std::uint32_t galaxy_count,
                                      const double grav_const,
                                      const int enable_smbh_post_newtonian,
                                      const double dt_myr) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= particle_count) {
    return;
  }

  (void)length_x;
  (void)length_y;
  (void)length_z;

  SimCudaParticle& particle = particles[index];
  const double particle_pos_x = particle.position_kpc[0];
  const double particle_pos_y = particle.position_kpc[1];
  const double particle_pos_z = particle.position_kpc[2];
  const double particle_vel_x = particle.velocity_kms[0];
  const double particle_vel_y = particle.velocity_kms[1];
  const double particle_vel_z = particle.velocity_kms[2];
  const double particle_mass = particle.mass_msun;
  const double particle_softening = particle.softening_kpc;
  const std::uint32_t particle_component = particle.component;
  const double short_cutoff_sq = short_cutoff_kpc * short_cutoff_kpc;

  double ax = 0.0;
  double ay = 0.0;
  double az = 0.0;
  sample_grid_trilinear_vector(force_x,
                               force_y,
                               force_z,
                               nx,
                               ny,
                               nz,
                               origin_x,
                               origin_y,
                               origin_z,
                               cell_x,
                               cell_y,
                               cell_z,
                               particle_pos_x,
                               particle_pos_y,
                               particle_pos_z,
                               ax,
                               ay,
                               az);
  ax *= inv_fft_cells;
  ay *= inv_fft_cells;
  az *= inv_fft_cells;

  if (particle_component != 3u &&
      !(short_range_target_baryons_only != 0 && particle_component == 0u)) {
    int particle_cell = -1;
    int particle_cell_x = -1;
    int particle_cell_y = -1;
    int particle_cell_z = -1;
    if (short_tree_particle_mode == 0) {
      const double gx = fmin(
          fmax((particle_pos_x - short_origin_x) / short_cell_x, 0.0),
          static_cast<double>(short_nx) - 1.000001);
      const double gy = fmin(
          fmax((particle_pos_y - short_origin_y) / short_cell_y, 0.0),
          static_cast<double>(short_ny) - 1.000001);
      const double gz = fmin(
          fmax((particle_pos_z - short_origin_z) / short_cell_z, 0.0),
          static_cast<double>(short_nz) - 1.000001);
      particle_cell_x = static_cast<int>(floor(gx));
      particle_cell_y = static_cast<int>(floor(gy));
      particle_cell_z = static_cast<int>(floor(gz));
      particle_cell = short_cell_linear_index(particle_cell_x,
                                              particle_cell_y,
                                              particle_cell_z,
                                              short_ny,
                                              short_nz);
      const int start = short_cell_start[particle_cell];
      if (start >= 0) {
        const int end = short_cell_end[particle_cell];
        const int occupancy = end - start;
        if (occupancy <= kShortRangeDirectCellThreshold) {
          for (int sorted_index = start; sorted_index < end; ++sorted_index) {
            const int source_index = short_sorted_particle_indices[sorted_index];
            if (source_index < 0 ||
                static_cast<std::uint64_t>(source_index) >= particle_count ||
                source_index == static_cast<int>(index)) {
              continue;
            }

            const SimCudaParticle source = particles[source_index];
            const double dx = source.position_kpc[0] - particle_pos_x;
            const double dy = source.position_kpc[1] - particle_pos_y;
            const double dz = source.position_kpc[2] - particle_pos_z;
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= 1.0e-12 || r2 > short_cutoff_sq) {
              continue;
            }

            const double short_softening = fmax(particle_softening, source.softening_kpc);
            const double direct_scale =
                grav_const * source.mass_msun * softened_inv_r3(dx, dy, dz, short_softening);
            const double correction_scale =
                direct_scale * treepm_short_range_force_factor_lookup(short_force_factor_lut,
                                                                      short_force_factor_lut_size,
                                                                      short_force_factor_lut_scale,
                                                                      r2,
                                                                      short_pm_softening_kpc);
            ax += correction_scale * dx;
            ay += correction_scale * dy;
            az += correction_scale * dz;
          }
        } else {
          const int cell_plane = short_ny * short_nz;
          const int cell_x_index = particle_cell / cell_plane;
          const int rem = particle_cell - cell_x_index * cell_plane;
          const int cell_y_index = rem / short_nz;
          const int cell_z_index = rem - cell_y_index * short_nz;
          const double center_x =
              short_origin_x + (static_cast<double>(cell_x_index) + 0.5) * short_cell_x;
          const double center_y =
              short_origin_y + (static_cast<double>(cell_y_index) + 0.5) * short_cell_y;
          const double center_z =
              short_origin_z + (static_cast<double>(cell_z_index) + 0.5) * short_cell_z;
          const int particle_octant =
              (particle_pos_x >= center_x ? 1 : 0) |
              (particle_pos_y >= center_y ? 2 : 0) |
              (particle_pos_z >= center_z ? 4 : 0);
          const double cell_softening =
              fmax(particle_softening, 0.5 * fmax(short_cell_x, fmax(short_cell_y, short_cell_z)));
          const std::size_t cell_base = static_cast<std::size_t>(particle_cell) * 8u;
          for (int octant = 0; octant < 8; ++octant) {
            const std::size_t slot = cell_base + static_cast<std::size_t>(octant);
            double mass = short_cell_octant_mass[slot];
            if (!(mass > 0.0)) {
              continue;
            }
            double sum_x = short_cell_octant_com_x[slot];
            double sum_y = short_cell_octant_com_y[slot];
            double sum_z = short_cell_octant_com_z[slot];
            if (octant == particle_octant) {
              mass -= particle_mass;
              if (!(mass > 0.0)) {
                continue;
              }
              sum_x -= particle_mass * particle_pos_x;
              sum_y -= particle_mass * particle_pos_y;
              sum_z -= particle_mass * particle_pos_z;
            }
            const double com_x = sum_x / mass;
            const double com_y = sum_y / mass;
            const double com_z = sum_z / mass;
            const double dx = com_x - particle_pos_x;
            const double dy = com_y - particle_pos_y;
            const double dz = com_z - particle_pos_z;
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= 1.0e-12 || r2 > short_cutoff_sq) {
              continue;
            }

            const double direct_scale =
                grav_const * mass * softened_inv_r3(dx, dy, dz, cell_softening);
            const double correction_scale =
                direct_scale * treepm_short_range_force_factor_lookup(short_force_factor_lut,
                                                                      short_force_factor_lut_size,
                                                                      short_force_factor_lut_scale,
                                                                      r2,
                                                                      short_pm_softening_kpc);
            ax += correction_scale * dx;
            ay += correction_scale * dy;
            az += correction_scale * dz;
          }
        }
      }
    }

    if (short_tree_particle_mode == 0 && particle_cell >= 0) {
      const int neighbor_radius_x =
          min(1, max(1, static_cast<int>(ceil(short_cutoff_kpc / fmax(short_cell_x, 1.0e-6)))));
      const int neighbor_radius_y =
          min(1, max(1, static_cast<int>(ceil(short_cutoff_kpc / fmax(short_cell_y, 1.0e-6)))));
      const int neighbor_radius_z =
          min(1, max(1, static_cast<int>(ceil(short_cutoff_kpc / fmax(short_cell_z, 1.0e-6)))));
      const double cell_half_x = 0.5 * short_cell_x;
      const double cell_half_y = 0.5 * short_cell_y;
      const double cell_half_z = 0.5 * short_cell_z;
      const double cell_softening =
          fmax(particle_softening, 0.5 * fmax(short_cell_x, fmax(short_cell_y, short_cell_z)));

      for (int neighbor_x = max(0, particle_cell_x - neighbor_radius_x);
           neighbor_x <= min(short_nx - 1, particle_cell_x + neighbor_radius_x);
           ++neighbor_x) {
        for (int neighbor_y = max(0, particle_cell_y - neighbor_radius_y);
             neighbor_y <= min(short_ny - 1, particle_cell_y + neighbor_radius_y);
             ++neighbor_y) {
          for (int neighbor_z = max(0, particle_cell_z - neighbor_radius_z);
               neighbor_z <= min(short_nz - 1, particle_cell_z + neighbor_radius_z);
               ++neighbor_z) {
            const int neighbor_cell =
                short_cell_linear_index(neighbor_x, neighbor_y, neighbor_z, short_ny, short_nz);
            if (neighbor_cell == particle_cell) {
              continue;
            }

            const double center_x =
                short_origin_x + (static_cast<double>(neighbor_x) + 0.5) * short_cell_x;
            const double center_y =
                short_origin_y + (static_cast<double>(neighbor_y) + 0.5) * short_cell_y;
            const double center_z =
                short_origin_z + (static_cast<double>(neighbor_z) + 0.5) * short_cell_z;
            const double dx_aabb =
                fmax(fabs(particle_pos_x - center_x) - cell_half_x, 0.0);
            const double dy_aabb =
                fmax(fabs(particle_pos_y - center_y) - cell_half_y, 0.0);
            const double dz_aabb =
                fmax(fabs(particle_pos_z - center_z) - cell_half_z, 0.0);
            if (dx_aabb * dx_aabb + dy_aabb * dy_aabb + dz_aabb * dz_aabb > short_cutoff_sq) {
              continue;
            }

            const int start = short_cell_start[neighbor_cell];
            if (start < 0) {
              continue;
            }
            const double mass = short_cell_mass[neighbor_cell];
            if (!(mass > 0.0)) {
              continue;
            }
            const double dx = short_cell_com_x[neighbor_cell] - particle_pos_x;
            const double dy = short_cell_com_y[neighbor_cell] - particle_pos_y;
            const double dz = short_cell_com_z[neighbor_cell] - particle_pos_z;
            const double r2 = dx * dx + dy * dy + dz * dz;
            if (r2 <= 1.0e-12 || r2 > short_cutoff_sq) {
              continue;
            }

            const double direct_scale =
                grav_const * mass * softened_inv_r3(dx, dy, dz, cell_softening);
            const double correction_scale =
                direct_scale * treepm_short_range_force_factor_lookup(short_force_factor_lut,
                                                                      short_force_factor_lut_size,
                                                                      short_force_factor_lut_scale,
                                                                      r2,
                                                                      short_pm_softening_kpc);
            ax += correction_scale * dx;
            ay += correction_scale * dy;
            az += correction_scale * dz;
          }
        }
      }
    } else if (short_tree_nodes != nullptr && short_tree_root >= 0 && short_tree_node_count > 0) {
      int stack[64];
      int stack_size = 0;
      stack[stack_size++] = short_tree_root;
      const double theta_sq = opening_angle * opening_angle;
      while (stack_size > 0) {
        const int node_index = stack[--stack_size];
        if (node_index < 0 || static_cast<std::uint32_t>(node_index) >= short_tree_node_count) {
          continue;
        }

        const ShortRangeTreeNode node = short_tree_nodes[node_index];
        if (node.mass <= 0.0) {
          continue;
        }

        const bool is_leaf = node.child_mask == 0;
        if (short_tree_particle_mode == 0 && is_leaf && node.cell_id == particle_cell) {
          continue;
        }

        const double dx = node.com[0] - particle_pos_x;
        const double dy = node.com[1] - particle_pos_y;
        const double dz = node.com[2] - particle_pos_z;
        const double dx_aabb =
            fmax(fabs(particle_pos_x - node.center[0]) - node.half_size, 0.0);
        const double dy_aabb =
            fmax(fabs(particle_pos_y - node.center[1]) - node.half_size, 0.0);
        const double dz_aabb =
            fmax(fabs(particle_pos_z - node.center[2]) - node.half_size, 0.0);
        if (dx_aabb * dx_aabb + dy_aabb * dy_aabb + dz_aabb * dz_aabb > short_cutoff_sq) {
          continue;
        }
        const double r2 = dx * dx + dy * dy + dz * dz;
        if (r2 <= 1.0e-12) {
          continue;
        }

        if (short_tree_particle_mode != 0) {
          const bool contains_particle =
              fabs(particle_pos_x - node.center[0]) <= node.half_size &&
              fabs(particle_pos_y - node.center[1]) <= node.half_size &&
              fabs(particle_pos_z - node.center[2]) <= node.half_size;
          if (contains_particle && !is_leaf && stack_size <= 56) {
            for (int child = 0; child < 8; ++child) {
              if ((node.child_mask & static_cast<std::uint8_t>(1u << child)) != 0) {
                stack[stack_size++] = node.child[child];
              }
            }
            continue;
          }
          if (is_leaf && node.particle_index == static_cast<int>(index)) {
            continue;
          }
        }

        const double size_over_r_sq = (4.0 * node.half_size * node.half_size) / r2;
        if (!is_leaf && size_over_r_sq > theta_sq && stack_size <= 56) {
          for (int child = 0; child < 8; ++child) {
            if ((node.child_mask & static_cast<std::uint8_t>(1u << child)) != 0) {
              stack[stack_size++] = node.child[child];
            }
          }
          continue;
        }

        const double node_softening = fmax(particle_softening, node.softening_kpc);
        const double direct_scale =
            grav_const * node.mass * softened_inv_r3(dx, dy, dz, node_softening);
        const double correction_scale =
            direct_scale * treepm_short_range_force_factor_lookup(short_force_factor_lut,
                                                                  short_force_factor_lut_size,
                                                                  short_force_factor_lut_scale,
                                                                  r2,
                                                                  short_pm_softening_kpc);
        ax += correction_scale * dx;
        ay += correction_scale * dy;
        az += correction_scale * dz;
      }
    }
  }

  for (std::uint32_t galaxy_index = 0; galaxy_index < galaxy_count; ++galaxy_index) {
    const int source_index = galaxy_smbh_indices[galaxy_index];
    if (source_index < 0 || static_cast<std::uint64_t>(source_index) >= particle_count) {
      continue;
    }
    if (static_cast<int>(index) == source_index) {
      continue;
    }

    const SimCudaParticle source = particles[source_index];
    const double dx = source.position_kpc[0] - particle_pos_x;
    const double dy = source.position_kpc[1] - particle_pos_y;
    const double dz = source.position_kpc[2] - particle_pos_z;
    const double softening = fmax(particle_softening, source.softening_kpc);
    add_point_mass_acceleration(ax, ay, az, dx, dy, dz, softening, grav_const, source.mass_msun);

    if (enable_smbh_post_newtonian != 0 && particle_component == 3u) {
      add_smbh_1pn_acceleration(ax, ay, az, particle, source, grav_const);
    }
  }

  particle.velocity_kms[0] = particle_vel_x + ax * dt_myr * kKpcPerKmPerMyr;
  particle.velocity_kms[1] = particle_vel_y + ay * dt_myr * kKpcPerKmPerMyr;
  particle.velocity_kms[2] = particle_vel_z + az * dt_myr * kKpcPerKmPerMyr;
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

  double ax_fine = 0.0;
  double ay_fine = 0.0;
  double az_fine = 0.0;
  sample_grid_trilinear_vector_local(fine_force_x,
                                     fine_force_y,
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
                                     particle.position_kpc[2],
                                     ax_fine,
                                     ay_fine,
                                     az_fine);
  ax_fine *= fine_inv_fft_cells;
  ay_fine *= fine_inv_fft_cells;
  az_fine *= fine_inv_fft_cells;

  double ax_coarse = 0.0;
  double ay_coarse = 0.0;
  double az_coarse = 0.0;
  sample_grid_trilinear_vector_local(coarse_force_x,
                                     coarse_force_y,
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
                                     particle.position_kpc[2],
                                     ax_coarse,
                                     ay_coarse,
                                     az_coarse);
  ax_coarse *= coarse_inv_fft_cells;
  ay_coarse *= coarse_inv_fft_cells;
  az_coarse *= coarse_inv_fft_cells;

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
                               const int* visible_particle_indices,
                               const std::uint32_t visible_particle_count,
                               const int* anchor_indices,
                               const std::uint32_t anchor_count,
                               const std::uint32_t sampled_count,
                               const std::uint32_t sample_offset,
                               const std::uint64_t stride,
                               SimCudaPreviewParticle* out_particles) {
  const std::uint32_t preview_count = anchor_count + sampled_count;
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= preview_count) {
    return;
  }

  std::uint64_t source_index = 0;
  if (index < anchor_count) {
    source_index = static_cast<std::uint64_t>(anchor_indices[index] < 0 ? 0 : anchor_indices[index]);
  } else {
    const std::uint64_t sample_slot =
        min((static_cast<std::uint64_t>(sample_offset) +
             static_cast<std::uint64_t>(index - anchor_count) * stride) %
                static_cast<std::uint64_t>(visible_particle_count),
            static_cast<std::uint64_t>(visible_particle_count - 1));
    source_index = static_cast<std::uint64_t>(visible_particle_indices[sample_slot]);
  }
  const SimCudaParticle& particle = particles[source_index];
  out_particles[index].position_kpc[0] = static_cast<float>(particle.position_kpc[0]);
  out_particles[index].position_kpc[1] = static_cast<float>(particle.position_kpc[1]);
  out_particles[index].position_kpc[2] = static_cast<float>(particle.position_kpc[2]);
  out_particles[index].velocity_kms[0] = static_cast<float>(particle.velocity_kms[0]);
  out_particles[index].velocity_kms[1] = static_cast<float>(particle.velocity_kms[1]);
  out_particles[index].velocity_kms[2] = static_cast<float>(particle.velocity_kms[2]);
  out_particles[index].mass_msun = static_cast<float>(particle.mass_msun);
  out_particles[index].component = particle.component;
}

SimCudaPreviewParticle preview_particle_from_host_particle(const SimCudaParticle& particle) {
  SimCudaPreviewParticle preview{};
  preview.position_kpc[0] = static_cast<float>(particle.position_kpc[0]);
  preview.position_kpc[1] = static_cast<float>(particle.position_kpc[1]);
  preview.position_kpc[2] = static_cast<float>(particle.position_kpc[2]);
  preview.velocity_kms[0] = static_cast<float>(particle.velocity_kms[0]);
  preview.velocity_kms[1] = static_cast<float>(particle.velocity_kms[1]);
  preview.velocity_kms[2] = static_cast<float>(particle.velocity_kms[2]);
  preview.mass_msun = static_cast<float>(particle.mass_msun);
  preview.component = particle.component;
  return preview;
}

int build_force_mesh(DeviceState* state, char* error_buffer, const std::size_t error_buffer_len) {
  const cudaError_t clear_status =
      cudaMemsetAsync(state->density_grid,
                      0,
                      sizeof(cufftReal) * state->real_count,
                      state->compute_stream);
  if (clear_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "density grid clear failed", clear_status);
    return 1;
  }

  const int threads_per_block = 256;
  const int particle_blocks =
      static_cast<int>((state->particle_count + threads_per_block - 1) / threads_per_block);
  deposit_mass_cic<<<particle_blocks, threads_per_block, 0, state->compute_stream>>>(
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

  const cufftResult forward_status =
      cufftExecR2C(state->forward_plan, state->density_grid, state->density_k);
  if (forward_status != CUFFT_SUCCESS) {
    fill_cufft_error(error_buffer, error_buffer_len, "forward density FFT failed", forward_status);
    return 1;
  }

  const int complex_blocks =
      static_cast<int>((state->complex_count + threads_per_block - 1) / threads_per_block);
  apply_potential_spectrum<<<complex_blocks, threads_per_block, 0, state->compute_stream>>>(
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
      state->short_pm_softening_kpc,
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
  compute_force_from_potential<<<real_blocks, threads_per_block, 0, state->compute_stream>>>(
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

int build_short_range_structure(DeviceState* state,
                                char* error_buffer,
                                const std::size_t error_buffer_len) {
  if (state->short_cell_count == 0 || state->short_source_particle_count == 0) {
    return 0;
  }

  const bool profile_stages = profile_force_stages();
  auto finish_stage = [&](const char* label, const std::chrono::steady_clock::time_point start) -> int {
    if (!profile_stages) {
      return 0;
    }
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, label, sync_status);
      return 1;
    }
    flush_profile_stage(label, start, std::chrono::steady_clock::now());
    return 0;
  };

  if (state->short_source_particle_count <= particle_tree_threshold()) {
    return build_short_range_tree(state, error_buffer, error_buffer_len);
  }

  const int threads_per_block = 256;
  const int particle_blocks =
      static_cast<int>((state->short_source_particle_count + threads_per_block - 1) / threads_per_block);
  auto stage_start = std::chrono::steady_clock::now();
  cudaError_t cuda_status = cudaMemsetAsync(state->short_cell_start,
                                            0xff,
                                            sizeof(int) * (state->short_cell_count + 1),
                                            state->compute_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "short-range cell start clear failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMemsetAsync(state->short_cell_end,
                                0xff,
                                sizeof(int) * (state->short_cell_count + 1),
                                state->compute_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "short-range cell end clear failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMemsetAsync(state->short_cell_mass,
                                0,
                                sizeof(double) * state->short_cell_count,
                                state->compute_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "short-range cell mass clear failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMemsetAsync(state->short_cell_com_x,
                                0,
                                sizeof(double) * state->short_cell_count,
                                state->compute_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "short-range cell com_x clear failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMemsetAsync(state->short_cell_com_y,
                                0,
                                sizeof(double) * state->short_cell_count,
                                state->compute_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "short-range cell com_y clear failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMemsetAsync(state->short_cell_com_z,
                                0,
                                sizeof(double) * state->short_cell_count,
                                state->compute_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "short-range cell com_z clear failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMemsetAsync(state->short_cell_octant_mass,
                                0,
                                sizeof(double) * state->short_cell_count * 8u,
                                state->compute_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "short-range octant mass clear failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMemsetAsync(state->short_cell_octant_com_x,
                                0,
                                sizeof(double) * state->short_cell_count * 8u,
                                state->compute_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "short-range octant com_x clear failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMemsetAsync(state->short_cell_octant_com_y,
                                0,
                                sizeof(double) * state->short_cell_count * 8u,
                                state->compute_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "short-range octant com_y clear failed",
                    cuda_status);
    return 1;
  }
  cuda_status = cudaMemsetAsync(state->short_cell_octant_com_z,
                                0,
                                sizeof(double) * state->short_cell_count * 8u,
                                state->compute_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "short-range octant com_z clear failed",
                    cuda_status);
    return 1;
  }
  if (finish_stage("short_range.clear_buffers", stage_start) != 0) {
    return 1;
  }

  stage_start = std::chrono::steady_clock::now();
  compute_short_range_cells<<<particle_blocks, threads_per_block, 0, state->compute_stream>>>(
      state->particles,
      state->short_source_particle_indices,
      state->short_source_particle_count,
      state->short_sorted_cell_ids,
      state->short_sorted_particle_indices,
      state->short_cell_mass,
      state->short_cell_com_x,
      state->short_cell_com_y,
      state->short_cell_com_z,
      state->short_cell_octant_mass,
      state->short_cell_octant_com_x,
      state->short_cell_octant_com_y,
      state->short_cell_octant_com_z,
      state->short_nx,
      state->short_ny,
      state->short_nz,
      state->short_domain_origin[0],
      state->short_domain_origin[1],
      state->short_domain_origin[2],
      state->short_cell_size[0],
      state->short_cell_size[1],
      state->short_cell_size[2]);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "short-range cell kernel failed", cuda_status);
    return 1;
  }
  if (finish_stage("short_range.compute_cells", stage_start) != 0) {
    return 1;
  }

  try {
    stage_start = std::chrono::steady_clock::now();
    thrust::device_ptr<int> key_begin(state->short_sorted_cell_ids);
    thrust::device_ptr<int> key_end = key_begin + state->short_source_particle_count;
    thrust::device_ptr<int> value_begin(state->short_sorted_particle_indices);
    thrust::sort_by_key(
        thrust::cuda::par.on(state->compute_stream), key_begin, key_end, value_begin);
    if (finish_stage("short_range.sort_by_cell", stage_start) != 0) {
      return 1;
    }
  } catch (const std::exception& error) {
    fill_error(error_buffer, error_buffer_len, error.what());
    return 1;
  }

  stage_start = std::chrono::steady_clock::now();
  build_short_range_cell_ranges<<<particle_blocks, threads_per_block, 0, state->compute_stream>>>(
      state->short_sorted_cell_ids,
      state->short_source_particle_count,
      state->short_cell_start,
      state->short_cell_end);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "short-range range kernel failed", cuda_status);
    return 1;
  }

  const int cell_blocks =
      static_cast<int>((state->short_cell_count + threads_per_block - 1) / threads_per_block);
  normalize_short_range_cell_moments<<<cell_blocks, threads_per_block, 0, state->compute_stream>>>(
      state->short_cell_count,
      state->short_cell_mass,
      state->short_cell_com_x,
      state->short_cell_com_y,
      state->short_cell_com_z);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "short-range moments kernel failed", cuda_status);
    return 1;
  }
  if (finish_stage("short_range.build_cell_ranges_and_moments", stage_start) != 0) {
    return 1;
  }

  stage_start = std::chrono::steady_clock::now();
  if (build_short_range_tree(state, error_buffer, error_buffer_len) != 0) {
    return 1;
  }
  if (finish_stage("short_range.build_tree", stage_start) != 0) {
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
    cudaError_t cuda_status = cudaMemsetAsync(
        mesh.density_grid, 0, sizeof(cufftReal) * mesh.real_count, state->compute_stream);
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "local density grid clear failed", cuda_status);
      return 1;
    }

    deposit_mass_cic_local<<<particle_blocks, threads_per_block, 0, state->compute_stream>>>(
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
    apply_potential_spectrum<<<complex_blocks, threads_per_block, 0, state->compute_stream>>>(
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
        0.0,
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
    compute_force_from_potential_clamped<<<real_blocks, threads_per_block, 0, state->compute_stream>>>(
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

int build_force_state(DeviceState* state,
                      char* error_buffer,
                      const std::size_t error_buffer_len) {
  const bool profile_stages = profile_force_stages();
  const auto total_start = std::chrono::steady_clock::now();

  auto finish_stage = [&](const char* label, const std::chrono::steady_clock::time_point start) -> int {
    if (!profile_stages) {
      return 0;
    }
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, label, sync_status);
      return 1;
    }
    flush_profile_stage(label, start, std::chrono::steady_clock::now());
    return 0;
  };

  auto stage_start = std::chrono::steady_clock::now();
  if (update_simulation_domain_from_device(state, error_buffer, error_buffer_len) != 0) {
    return 1;
  }
  if (finish_stage("update_simulation_domain_from_device", stage_start) != 0) {
    return 1;
  }

  stage_start = std::chrono::steady_clock::now();
  if (update_short_range_grid(state, error_buffer, error_buffer_len) != 0) {
    return 1;
  }
  if (finish_stage("update_short_range_grid", stage_start) != 0) {
    return 1;
  }

  stage_start = std::chrono::steady_clock::now();
  if (build_force_mesh(state, error_buffer, error_buffer_len) != 0) {
    return 1;
  }
  if (finish_stage("build_force_mesh", stage_start) != 0) {
    return 1;
  }

  stage_start = std::chrono::steady_clock::now();
  if (build_short_range_structure(state, error_buffer, error_buffer_len) != 0) {
    return 1;
  }
  if (finish_stage("build_short_range_structure", stage_start) != 0) {
    return 1;
  }

  state->force_state_valid = true;
  if (profile_stages) {
    flush_profile_stage("build_force_state_total", total_start, std::chrono::steady_clock::now());
  }
  return 0;
}

int apply_force_kick_from_state(DeviceState* state,
                                const int particle_blocks,
                                const int threads_per_block,
                                const double inv_fft_cells,
                                const double dt_myr,
                                char* error_buffer,
                                const std::size_t error_buffer_len) {
  kick_particles_global<<<particle_blocks, threads_per_block, 0, state->compute_stream>>>(
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
      state->short_sorted_particle_indices,
      state->short_cell_start,
      state->short_cell_end,
      state->short_cell_interaction_start,
      state->short_cell_mass,
      state->short_cell_com_x,
      state->short_cell_com_y,
      state->short_cell_com_z,
      state->short_cell_octant_mass,
      state->short_cell_octant_com_x,
      state->short_cell_octant_com_y,
      state->short_cell_octant_com_z,
      state->short_cell_interactions,
      state->short_force_factor_lut,
      kShortRangeForceFactorLutSize,
      state->short_force_factor_lut_scale,
      state->short_tree_nodes,
      state->short_tree_root,
      state->short_tree_node_count,
      state->short_tree_particle_mode ? 1 : 0,
      state->short_nx,
      state->short_ny,
      state->short_nz,
      state->short_domain_origin[0],
      state->short_domain_origin[1],
      state->short_domain_origin[2],
      state->short_cell_size[0],
      state->short_cell_size[1],
      state->short_cell_size[2],
      state->short_cutoff_kpc,
      state->short_pm_softening_kpc,
      state->opening_angle,
      state->short_range_target_baryons_only ? 1 : 0,
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
  const bool profile_stages = profile_force_stages();

  auto finish_profile_stage = [&](const char* label,
                                  const std::chrono::steady_clock::time_point start) -> int {
    if (!profile_stages) {
      return 0;
    }
    const cudaError_t sync_status = cudaDeviceSynchronize();
    if (sync_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, label, sync_status);
      return 1;
    }
    flush_profile_stage(label, start, std::chrono::steady_clock::now());
    return 0;
  };

  for (std::uint32_t step = 0; step < step_count; ++step) {
    const std::uint32_t substeps =
        estimate_substeps_for_step(state, dt_myr, error_buffer, error_buffer_len);
    const double substep_dt_myr = dt_myr / static_cast<double>(std::max(1u, substeps));
    const double half_substep_dt_myr = 0.5 * substep_dt_myr;

    for (std::uint32_t substep = 0; substep < substeps; ++substep) {
      auto stage_start = std::chrono::steady_clock::now();
      if (!state->force_state_valid &&
          build_force_state(state, error_buffer, error_buffer_len) != 0) {
        return 1;
      }
      if (finish_profile_stage("run_steps.build_force_state_a", stage_start) != 0) {
        return 1;
      }

      stage_start = std::chrono::steady_clock::now();
      if (apply_force_kick_from_state(state,
                                      particle_blocks,
                                      threads_per_block,
                                      inv_fft_cells,
                                      half_substep_dt_myr,
                                      error_buffer,
                                      error_buffer_len) != 0) {
        return 1;
      }
      if (finish_profile_stage("run_steps.kick_a", stage_start) != 0) {
        return 1;
      }

      stage_start = std::chrono::steady_clock::now();
      drift_particles<<<particle_blocks, threads_per_block, 0, state->compute_stream>>>(
          state->particles, state->particle_count, substep_dt_myr);
      cudaError_t cuda_status = cudaGetLastError();
      if (cuda_status != cudaSuccess) {
        fill_cuda_error(error_buffer, error_buffer_len, "particle drift kernel failed", cuda_status);
        return 1;
      }
      state->force_state_valid = false;
      if (finish_profile_stage("run_steps.drift", stage_start) != 0) {
        return 1;
      }

      stage_start = std::chrono::steady_clock::now();
      if (build_force_state(state, error_buffer, error_buffer_len) != 0) {
        return 1;
      }
      if (finish_profile_stage("run_steps.build_force_state_b", stage_start) != 0) {
        return 1;
      }

      stage_start = std::chrono::steady_clock::now();
      if (apply_force_kick_from_state(state,
                                      particle_blocks,
                                      threads_per_block,
                                      inv_fft_cells,
                                      half_substep_dt_myr,
                                      error_buffer,
                                      error_buffer_len) != 0) {
        return 1;
      }
      if (finish_profile_stage("run_steps.kick_b", stage_start) != 0) {
        return 1;
      }
    }
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
  state->max_substeps = std::max(1u, params->max_substeps);
  state->cfl_safety_factor = std::max(0.05, params->cfl_safety_factor);
  state->opening_angle = std::max(0.2, params->opening_angle);
  state->nx = static_cast<int>(params->mesh_resolution[0]);
  state->ny = static_cast<int>(params->mesh_resolution[1]);
  state->nz = static_cast<int>(params->mesh_resolution[2]);
  state->nz_complex = state->nz / 2 + 1;
  state->real_count = static_cast<std::size_t>(state->nx) * static_cast<std::size_t>(state->ny) *
                      static_cast<std::size_t>(state->nz);
  state->complex_count = static_cast<std::size_t>(state->nx) * static_cast<std::size_t>(state->ny) *
                         static_cast<std::size_t>(state->nz_complex);
  for (std::uint64_t i = 0; i < params->particle_count; ++i) {
    state->max_softening_kpc = std::max(state->max_softening_kpc, particles[i].softening_kpc);
  }

  cudaError_t cuda_status =
      cudaStreamCreateWithFlags(&state->compute_stream, cudaStreamNonBlocking);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "compute stream creation failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  cuda_status = cudaStreamCreateWithFlags(&state->preview_stream, cudaStreamNonBlocking);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "preview stream creation failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  cuda_status = cudaEventCreateWithFlags(&state->preview_sample_event, cudaEventDisableTiming);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "preview sample event creation failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  cuda_status = cudaEventCreateWithFlags(&state->preview_copy_done_event, cudaEventDisableTiming);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "preview copy event creation failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  compute_simulation_domain(state, particles);

  const auto particle_bytes = sizeof(SimCudaParticle) * params->particle_count;
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->particles), particle_bytes);
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
  std::vector<int> short_source_particle_indices_host;
  std::vector<int> preview_visible_particle_indices_host;
  short_source_particle_indices_host.reserve(params->particle_count / 3);
  preview_visible_particle_indices_host.reserve(params->particle_count / 8);
  const bool baryons_only = short_range_baryons_only();
  state->short_range_target_baryons_only = short_range_target_baryons_only();
  for (std::uint64_t i = 0; i < params->particle_count; ++i) {
    if (particles[i].component != 0u && particles[i].component != 3u && particles[i].mass_msun > 0.0) {
      preview_visible_particle_indices_host.push_back(static_cast<int>(i));
    }
    if (particles[i].component == 3u || !(particles[i].mass_msun > 0.0)) {
      continue;
    }
    if (baryons_only && particles[i].component == 0u) {
      continue;
    }
    short_source_particle_indices_host.push_back(static_cast<int>(i));
  }
  state->short_source_particle_count =
      static_cast<std::uint32_t>(short_source_particle_indices_host.size());
  state->preview_visible_particle_count =
      static_cast<std::uint32_t>(preview_visible_particle_indices_host.size());
  if (state->short_source_particle_count > 0) {
    cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->short_source_particle_indices),
                             sizeof(int) * state->short_source_particle_count);
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer,
                      error_buffer_len,
                      "cudaMalloc for short-range source indices failed",
                      cuda_status);
      destroy_state(state);
      return 1;
    }
    cuda_status = cudaMemcpy(state->short_source_particle_indices,
                             short_source_particle_indices_host.data(),
                             sizeof(int) * state->short_source_particle_count,
                             cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer,
                      error_buffer_len,
                      "cudaMemcpy for short-range source indices failed",
                      cuda_status);
      destroy_state(state);
      return 1;
    }
  }
  if (state->preview_visible_particle_count > 0) {
    cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->preview_visible_particle_indices),
                             sizeof(int) * state->preview_visible_particle_count);
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer,
                      error_buffer_len,
                      "cudaMalloc for preview visible indices failed",
                      cuda_status);
      destroy_state(state);
      return 1;
    }
    cuda_status = cudaMemcpy(state->preview_visible_particle_indices,
                             preview_visible_particle_indices_host.data(),
                             sizeof(int) * state->preview_visible_particle_count,
                             cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer,
                      error_buffer_len,
                      "cudaMemcpy for preview visible indices failed",
                      cuda_status);
      destroy_state(state);
      return 1;
    }
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->short_sorted_cell_ids),
                           sizeof(int) * std::max<std::uint32_t>(1, state->short_source_particle_count));
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range cell ids failed",
                    cuda_status);
    destroy_state(state);
    return 1;
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->short_sorted_particle_indices),
                           sizeof(int) * std::max<std::uint32_t>(1, state->short_source_particle_count));
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range particle indices failed",
                    cuda_status);
    destroy_state(state);
    return 1;
  }
  if (update_short_range_grid(state, error_buffer, error_buffer_len) != 0) {
    destroy_state(state);
    return 1;
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
  fft_status = cufftSetStream(state->forward_plan, state->compute_stream);
  if (fft_status != CUFFT_SUCCESS) {
    fill_cufft_error(error_buffer, error_buffer_len, "forward FFT stream binding failed", fft_status);
    destroy_state(state);
    return 1;
  }

  fft_status = cufftPlan3d(&state->inverse_plan, state->nx, state->ny, state->nz, CUFFT_C2R);
  if (fft_status != CUFFT_SUCCESS) {
    fill_cufft_error(error_buffer, error_buffer_len, "inverse FFT plan creation failed", fft_status);
    destroy_state(state);
    return 1;
  }
  fft_status = cufftSetStream(state->inverse_plan, state->compute_stream);
  if (fft_status != CUFFT_SUCCESS) {
    fill_cufft_error(error_buffer, error_buffer_len, "inverse FFT stream binding failed", fft_status);
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
  if (wait_for_in_flight_preview(state, error_buffer, error_buffer_len) != 0) {
    return 1;
  }
  if (schedule_preview_capture(state, max_particles, out_count, error_buffer, error_buffer_len) != 0) {
    return 1;
  }

  const cudaError_t cuda_status = cudaEventSynchronize(state->preview_copy_done_event);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "preview download failed", cuda_status);
    state->preview_in_flight = false;
    state->preview_in_flight_count = 0;
    return 1;
  }

  if (*out_count > 0) {
    std::memcpy(out_particles,
                state->preview_host_particles,
                sizeof(SimCudaPreviewParticle) * (*out_count));
  }
  state->preview_in_flight = false;
  state->preview_in_flight_count = 0;
  return 0;
}

extern "C" int sim_cuda_request_preview(void* handle,
                                        std::uint32_t max_particles,
                                        std::uint32_t* out_count,
                                        char* error_buffer,
                                        std::size_t error_buffer_len) {
  if (handle == nullptr || out_count == nullptr) {
    fill_error(error_buffer, error_buffer_len, "invalid preview request parameters");
    return 1;
  }

  auto* state = static_cast<DeviceState*>(handle);
  return schedule_preview_capture(state, max_particles, out_count, error_buffer, error_buffer_len);
}

extern "C" int sim_cuda_collect_preview(void* handle,
                                        SimCudaPreviewParticle* out_particles,
                                        std::uint32_t particle_capacity,
                                        std::uint32_t* out_count,
                                        int* out_ready,
                                        char* error_buffer,
                                        std::size_t error_buffer_len) {
  if (handle == nullptr || out_particles == nullptr || out_count == nullptr || out_ready == nullptr) {
    fill_error(error_buffer, error_buffer_len, "invalid preview collect parameters");
    return 1;
  }

  auto* state = static_cast<DeviceState*>(handle);
  *out_ready = 0;
  *out_count = 0;
  if (!state->preview_in_flight) {
    return 0;
  }
  if (particle_capacity < state->preview_in_flight_count) {
    fill_error(error_buffer, error_buffer_len, "preview capacity smaller than in-flight frame");
    return 1;
  }

  const cudaError_t query_status = cudaEventQuery(state->preview_copy_done_event);
  if (query_status == cudaErrorNotReady) {
    return 0;
  }
  if (query_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "preview readiness query failed", query_status);
    return 1;
  }

  if (state->preview_in_flight_count > 0) {
    std::memcpy(out_particles,
                state->preview_host_particles,
                sizeof(SimCudaPreviewParticle) * state->preview_in_flight_count);
  }
  *out_count = state->preview_in_flight_count;
  *out_ready = 1;
  state->preview_in_flight = false;
  state->preview_in_flight_count = 0;
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
