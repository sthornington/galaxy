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
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>
#include <cufft.h>
#include <thrust/device_ptr.h>
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sort.h>
#include <thrust/transform_reduce.h>

namespace {

constexpr double kKpcPerKmPerMyr = 0.001022712165045695;
constexpr double kSpeedOfLightKms = 299792.458;
constexpr double kPi = 3.14159265358979323846;
// The global PM mesh solves with a free-space (isolated) Green's function, so
// the box only needs to comfortably exceed twice the particle extent per axis
// for the circular convolution to be exact for every in-extent pair.
constexpr double kGlobalDomainPadding = 2.15;
constexpr double kGlobalDomainGuardFraction = 0.03;
constexpr double kShortRangeDomainPadding = 1.15;
constexpr double kMinGlobalBoxLengthKpc = 64.0;
constexpr double kMinShortRangeBoxLengthKpc = 8.0;
constexpr double kMinShortRangeCellSizeKpc = 0.35;
constexpr double kMaxShortRangeCellSizeKpc = 3.0;
constexpr std::size_t kMaxShortRangeCells = 4u * 1024u * 1024u;
constexpr int kMaxShortRangeAxisCells = 1024;
constexpr int kShortRangeDirectNeighborhoodThresholdLowN = 768;
constexpr int kShortRangeDirectNeighborhoodThresholdDefault = 256;
// Coarsened mass/COM pyramid over the short-range grid. Level 0 is the fine
// grid itself; each level halves the resolution until the top level is at most
// 3 cells per axis so the per-particle interaction-list walk covers the whole
// grid without any radius truncation.
constexpr int kMaxPyramidLevels = 12;

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

// Diagnostics that pair particles by array index (force_check) disable the
// cell-order compaction, which otherwise permutes the particle array every
// force build.
bool particle_reorder_disabled() {
  static const bool disabled = []() {
    const char* raw_value = std::getenv("SIM_CUDA_DISABLE_REORDER");
    if (raw_value == nullptr || raw_value[0] == '\0') {
      return false;
    }
    return std::strcmp(raw_value, "0") != 0;
  }();
  return disabled;
}

int pm_substep_interval() {
  static const int interval = []() {
    const char* raw_value = std::getenv("SIM_CUDA_PM_SUBSTEP_INTERVAL");
    if (raw_value == nullptr || raw_value[0] == '\0') {
      return 2;
    }
    char* end = nullptr;
    const long parsed = std::strtol(raw_value, &end, 10);
    if (end == raw_value || parsed < 1 || parsed > 8) {
      return 2;
    }
    return static_cast<int>(parsed);
  }();
  return interval;
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
  // The short grid only spans the dense baryonic region now, so aim for a few
  // particles per cell: most neighborhoods stay on the exact direct-sum path
  // and the octant fallback only fires in cluster cores.
  if (particle_count <= 300'000u) {
    return 4.0;
  }
  if (particle_count <= 1'000'000u) {
    return 6.0;
  }
  return 8.0;
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

void flush_profile_stage(const char* label,
                         const std::chrono::steady_clock::time_point start,
                         const std::chrono::steady_clock::time_point end) {
  std::fprintf(stderr,
               "[sim-cuda] stage=%s wall_ms=%.3f\n",
               label,
               std::chrono::duration<double, std::milli>(end - start).count());
}

// Per-level view of the coarsened cell pyramid, passed to the kick kernel by
// value. Each level stores packed (com - short_origin, mass) float4 moments;
// single-precision is ample for monopole approximations evaluated at least one
// cell away, and one vector load per cell keeps the walk bandwidth-bound work
// low.
struct PyramidDeviceView {
  int level_count = 0;
  int nx[kMaxPyramidLevels] = {};
  int ny[kMaxPyramidLevels] = {};
  int nz[kMaxPyramidLevels] = {};
  const float4* moments[kMaxPyramidLevels] = {};
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
  bool short_range_target_baryons_only = false;
  bool force_state_valid = false;
  double last_max_accel_sq = 0.0;
  // Cross-advance KDK fusion: the closing half-kick of an advance is deferred
  // and fused with the next advance's opening half-kick (no drift separates
  // them, so the force state is identical), eliminating one full force
  // evaluation per advance. Any externally visible phase-space read
  // (sim_cuda_copy_particles) synchronizes first, so downloaded state is
  // exact; preview and diagnostics tolerate the half-substep stagger.
  bool half_kick_pending = false;
  double pending_fast_dt_myr = 0.0;
  double pending_slow_dt_myr = 0.0;
  // PM mesh reuse: the long-range field changes slowly, so it is rebuilt
  // every SIM_CUDA_PM_SUBSTEP_INTERVAL substeps (default 2) unless the
  // domain geometry moved.
  int mesh_age = 1000000;
  int pe_age = 1000000;
  double last_potential_energy = 0.0;

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
  double short_domain_origin[3] = {0.0, 0.0, 0.0};
  double short_box_length[3] = {1.0, 1.0, 1.0};
  double short_cell_size[3] = {1.0, 1.0, 1.0};
  double short_cutoff_kpc = 0.0;
  double short_pm_softening_kpc = 0.0;

  // Coarsened pyramid over the fine short-range cells (levels 1..level_count-1;
  // level 0 aliases short_cell_mass/short_cell_com_*).
  int pyramid_level_count = 0;
  int pyramid_nx[kMaxPyramidLevels] = {};
  int pyramid_ny[kMaxPyramidLevels] = {};
  int pyramid_nz[kMaxPyramidLevels] = {};
  std::size_t pyramid_offset[kMaxPyramidLevels] = {};
  std::size_t pyramid_total_cells = 0;
  std::size_t pyramid_capacity = 0;
  double* pyramid_mass = nullptr;
  double* pyramid_com_x = nullptr;
  double* pyramid_com_y = nullptr;
  double* pyramid_com_z = nullptr;

  // Packed single-precision interaction tables rebuilt each force update:
  // per-cell moments for level 0 followed by the pyramid levels, plus the
  // per-octant moments of the fine cells, all relative to short_domain_origin.
  std::size_t moments_f4_capacity = 0;
  float4* cell_moments_f4 = nullptr;
  float4* octant_moments_f4 = nullptr;

  // Short-range sources gathered into sorted order as packed float4
  // (position relative to the source's own cell center, mass) plus a separate
  // softening array. Cell-relative fp32 keeps absolute position error around
  // 1e-7 kpc, far below the smallest softening, at a quarter of the fp64 SoA
  // bandwidth.
  float4* sorted_source_posm = nullptr;
  float* sorted_source_softening = nullptr;

  // Particles that are not short-range sources (SMBHs, massless, filtered),
  // processed after the cell-sorted targets so every particle gets kicked.
  int* non_source_indices = nullptr;
  std::uint32_t non_source_count = 0;

  SimCudaParticle* particles = nullptr;
  // Double buffer + inverse permutation used to compact particles into
  // cell-sorted order each force build, keeping every kernel's access pattern
  // coherent (PM deposit, drift, preview, and the kick's source loads).
  SimCudaParticle* particles_scratch = nullptr;
  int* inverse_index = nullptr;
  // Two-tier time bins: bin 0 kicks every substep, bin 1 every second substep
  // with a doubled dt. Assigned per base step from each particle's velocity
  // and last kick acceleration; permuted alongside the particle array.
  std::uint8_t* time_bins = nullptr;
  // --- Isothermal SPH state (allocated only when gas particles exist) ---
  double gas_sound_speed_kms = 0.0;
  double gas_viscosity_alpha = 1.0;
  double gas_smoothing_eta = 1.3;
  std::uint32_t gas_count = 0;
  // Per-particle smoothing length (meaningful for gas only); permuted with
  // the particle array on every traversal-order compaction.
  float* smoothing_h = nullptr;
  float* smoothing_h_scratch = nullptr;
  // Birth time [Myr] for stars formed in-run (component 5); -1e30 for
  // everything primordial. Permuted alongside the particles.
  float* birth_time_myr = nullptr;
  float* birth_time_scratch = nullptr;
  double star_formation_efficiency = 0.0;
  double star_formation_density = 0.0;
  double feedback_accel = 0.0;
  double feedback_radius_kpc = 0.25;
  int* young_star_indices = nullptr;
  std::uint32_t young_star_capacity = 0;
  std::uint32_t gas_capacity = 0;
  std::uint64_t sph_build_counter = 0;
  int* gas_indices = nullptr;          // canonical index of each gas particle
  int* gas_cell_ids = nullptr;
  int* gas_sorted_canonical = nullptr; // canonical index per sorted gas slot
  int* gas_cell_start = nullptr;
  int* gas_cell_end = nullptr;
  std::size_t gas_cell_capacity = 0;
  float4* gas_posh_sorted = nullptr;   // xyz rel gas-grid origin, w = h
  float4* gas_velm_sorted = nullptr;   // xyz velocity, w = mass
  float* gas_density_sorted = nullptr;
  float4* gas_accel = nullptr;         // canonical-indexed, (km/s)^2/kpc
  float* gas_density_canonical = nullptr; // canonical-indexed, Msun/kpc^3
  int gas_nx = 0;
  int gas_ny = 0;
  int gas_nz = 0;
  double gas_grid_origin[3] = {0.0, 0.0, 0.0};
  double gas_grid_cell_kpc = 0.0;
  // Snapshot of time_bins as of the previous step's kicks; the fused opening
  // kick reads the deferred (pending) dt through these so a particle that
  // switches bins between steps still receives exactly the kick schedule it
  // was integrated under.
  std::uint8_t* previous_time_bins = nullptr;
  std::uint8_t* time_bins_scratch = nullptr;
  float* accel_mag = nullptr;
  float* accel_mag_scratch = nullptr;
  int* galaxy_smbh_indices = nullptr;
  int* preview_visible_particle_indices = nullptr;
  int* short_source_particle_indices = nullptr;
  int* short_sorted_cell_ids = nullptr;
  int* short_sorted_particle_indices = nullptr;
  int* short_cell_start = nullptr;
  int* short_cell_end = nullptr;
  int* short_neighborhood_occupancy = nullptr;
  double* short_cell_mass = nullptr;
  double* short_cell_com_x = nullptr;
  double* short_cell_com_y = nullptr;
  double* short_cell_com_z = nullptr;
  double* short_cell_octant_mass = nullptr;
  double* short_cell_octant_com_x = nullptr;
  double* short_cell_octant_com_y = nullptr;
  double* short_cell_octant_com_z = nullptr;
  double* max_accel_sq = nullptr;
  std::uint32_t short_source_particle_count = 0;
  std::uint32_t preview_visible_particle_count = 0;
  std::vector<int> galaxy_smbh_indices_host;

  cufftReal* density_grid = nullptr;
  cufftComplex* density_k = nullptr;
  cufftComplex* force_k = nullptr;
  cufftReal* force_x = nullptr;
  cufftReal* force_y = nullptr;
  cufftReal* force_z = nullptr;
  cufftReal* potential_grid = nullptr;
  cufftComplex* greens_k = nullptr;
  double greens_box[3] = {0.0, 0.0, 0.0};

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
                               const float* __restrict__ birth_time_myr,
                               const float sim_time_epoch_myr,
                               const float* __restrict__ gas_density_canonical,
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

__device__ inline void atomic_max_positive_double(double* address, const double value) {
  if (!(value > 0.0)) {
    return;
  }
  auto* address_as_ull = reinterpret_cast<unsigned long long*>(address);
  unsigned long long observed = *address_as_ull;
  while (true) {
    const double current = __longlong_as_double(static_cast<long long>(observed));
    if (current >= value) {
      return;
    }
    const unsigned long long desired =
        static_cast<unsigned long long>(__double_as_longlong(value));
    const unsigned long long prior = atomicCAS(address_as_ull, observed, desired);
    if (prior == observed) {
      return;
    }
    observed = prior;
  }
}

void destroy_state(DeviceState* state) {
  if (state == nullptr) {
    return;
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
  cudaFree(state->short_cell_octant_com_z);
  cudaFree(state->short_cell_octant_com_y);
  cudaFree(state->short_cell_octant_com_x);
  cudaFree(state->short_cell_octant_mass);
  cudaFree(state->pyramid_com_z);
  cudaFree(state->pyramid_com_y);
  cudaFree(state->pyramid_com_x);
  cudaFree(state->pyramid_mass);
  cudaFree(state->cell_moments_f4);
  cudaFree(state->octant_moments_f4);
  cudaFree(state->sorted_source_posm);
  cudaFree(state->sorted_source_softening);
  cudaFree(state->non_source_indices);
  cudaFree(state->max_accel_sq);
  cudaFree(state->short_neighborhood_occupancy);
  cudaFree(state->short_cell_end);
  cudaFree(state->short_cell_start);
  cudaFree(state->short_sorted_particle_indices);
  cudaFree(state->short_sorted_cell_ids);
  cudaFree(state->short_source_particle_indices);
  cudaFree(state->preview_visible_particle_indices);
  cudaFreeHost(state->preview_host_particles);
  cudaFree(state->preview_particles);
  cudaFree(state->greens_k);
  cudaFree(state->potential_grid);
  cudaFree(state->force_z);
  cudaFree(state->force_y);
  cudaFree(state->force_x);
  cudaFree(state->force_k);
  cudaFree(state->density_k);
  cudaFree(state->density_grid);
  cudaFree(state->galaxy_smbh_indices);
  cudaFree(state->accel_mag_scratch);
  cudaFree(state->accel_mag);
  cudaFree(state->time_bins_scratch);
  cudaFree(state->time_bins);
  cudaFree(state->previous_time_bins);
  cudaFree(state->smoothing_h);
  cudaFree(state->smoothing_h_scratch);
  cudaFree(state->birth_time_myr);
  cudaFree(state->birth_time_scratch);
  cudaFree(state->gas_indices);
  cudaFree(state->gas_cell_ids);
  cudaFree(state->gas_sorted_canonical);
  cudaFree(state->gas_cell_start);
  cudaFree(state->gas_cell_end);
  cudaFree(state->gas_posh_sorted);
  cudaFree(state->gas_velm_sorted);
  cudaFree(state->gas_density_sorted);
  cudaFree(state->gas_accel);
  cudaFree(state->gas_density_canonical);
  cudaFree(state->young_star_indices);
  cudaFree(state->inverse_index);
  cudaFree(state->particles_scratch);
  cudaFree(state->particles);
  delete state;
}

struct DiagnosticsTotals {
  double kinetic_energy = 0.0;
  double momentum_x = 0.0;
  double momentum_y = 0.0;
  double momentum_z = 0.0;
};

struct DiagnosticsTotalsAccessor {
  __host__ __device__ DiagnosticsTotals operator()(const SimCudaParticle& particle) const {
    const double vx = particle.velocity_kms[0];
    const double vy = particle.velocity_kms[1];
    const double vz = particle.velocity_kms[2];
    const double mass = particle.mass_msun;
    DiagnosticsTotals totals{};
    totals.kinetic_energy = 0.5 * mass * (vx * vx + vy * vy + vz * vz);
    totals.momentum_x = mass * vx;
    totals.momentum_y = mass * vy;
    totals.momentum_z = mass * vz;
    return totals;
  }
};

struct MergeDiagnosticsTotals {
  __host__ __device__ DiagnosticsTotals operator()(const DiagnosticsTotals& lhs,
                                                   const DiagnosticsTotals& rhs) const {
    DiagnosticsTotals totals{};
    totals.kinetic_energy = lhs.kinetic_energy + rhs.kinetic_energy;
    totals.momentum_x = lhs.momentum_x + rhs.momentum_x;
    totals.momentum_y = lhs.momentum_y + rhs.momentum_y;
    totals.momentum_z = lhs.momentum_z + rhs.momentum_z;
    return totals;
  }
};

void fill_basic_diagnostics(const DeviceState& state,
                            const double dt_myr,
                            SimCudaDiagnostics* diagnostics) {
  diagnostics->particle_count = state.particle_count;
  diagnostics->preview_count = 0;
  diagnostics->sim_time_myr = state.sim_time_myr;
  diagnostics->dt_myr = dt_myr;
  thrust::device_ptr<SimCudaParticle> begin(state.particles);
  thrust::device_ptr<SimCudaParticle> end = begin + state.particle_count;
  const DiagnosticsTotals totals = thrust::transform_reduce(
      thrust::cuda::par.on(state.compute_stream),
      begin,
      end,
      DiagnosticsTotalsAccessor{},
      DiagnosticsTotals{},
      MergeDiagnosticsTotals{});
  diagnostics->kinetic_energy = totals.kinetic_energy;
  diagnostics->estimated_potential_energy = 0.0;
  diagnostics->total_momentum[0] = totals.momentum_x;
  diagnostics->total_momentum[1] = totals.momentum_y;
  diagnostics->total_momentum[2] = totals.momentum_z;
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
      state->birth_time_myr,
      static_cast<float>(std::floor(state->sim_time_myr / 6.4) * 6.4),
      state->gas_density_canonical,
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
      // The short-range grid tracks the dense baryonic region (disk, bulge,
      // SMBH) so its cells stay small where close forces matter. Halo
      // particles inside the box still participate; the diffuse outer halo is
      // handled by the PM mesh alone.
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
      // Baryons + SMBHs only: see compute_simulation_domain.
      if (particle.component != 0u) {
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

// Classifies each particle by whether it can take the doubled (slow) substep
// without exceeding the allowed per-kick displacement; the 0.8 margin absorbs
// velocity changes accrued mid-step before the next re-binning.
__global__ void assign_time_bins(const SimCudaParticle* __restrict__ particles,
                                 const float* __restrict__ accel_mag,
                                 const std::uint64_t particle_count,
                                 const double slow_dt_myr,
                                 const double allowed_displacement_kpc,
                                 std::uint8_t* __restrict__ time_bins) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= particle_count) {
    return;
  }
  const SimCudaParticle particle = particles[index];
  const double speed_kms = sqrt(particle.velocity_kms[0] * particle.velocity_kms[0] +
                                particle.velocity_kms[1] * particle.velocity_kms[1] +
                                particle.velocity_kms[2] * particle.velocity_kms[2]);
  const double accel_kpc_per_myr2 =
      static_cast<double>(accel_mag[index]) * kKpcPerKmPerMyr * kKpcPerKmPerMyr;
  const double displacement = speed_kms * slow_dt_myr * kKpcPerKmPerMyr +
                              0.5 * accel_kpc_per_myr2 * slow_dt_myr * slow_dt_myr;
  time_bins[index] = displacement <= 0.8 * allowed_displacement_kpc ? 1 : 0;
}

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
      // Grow with 3% overshoot: a slowly expanding halo otherwise outgrows the
      // box by epsilon every substep, forcing a Green's-function re-FFT and a
      // PM mesh rebuild each time (and defeating the mesh-reuse interval).
      const double next_global_length = required_padded_length > state->box_length[axis]
          ? required_padded_length * 1.03
          : state->box_length[axis];
      const double retained_origin = global_center - 0.5 * next_global_length;
      const double guard = kGlobalDomainGuardFraction * next_global_length;
      const bool leaves_guard_band = min_value < retained_origin + guard ||
                                     max_value > retained_origin + next_global_length - guard;
      const double next_global_center = leaves_guard_band ? center : global_center;
      state->box_length[axis] = next_global_length;
      state->domain_origin[axis] = next_global_center - 0.5 * next_global_length;

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
      state->short_cell_end != nullptr &&
      state->short_neighborhood_occupancy != nullptr) {
    return 0;
  }

  cudaFree(state->short_cell_start);
  cudaFree(state->short_cell_end);
  cudaFree(state->short_neighborhood_occupancy);
  cudaFree(state->short_cell_mass);
  cudaFree(state->short_cell_com_x);
  cudaFree(state->short_cell_com_y);
  cudaFree(state->short_cell_com_z);
  cudaFree(state->short_cell_octant_mass);
  cudaFree(state->short_cell_octant_com_x);
  cudaFree(state->short_cell_octant_com_y);
  cudaFree(state->short_cell_octant_com_z);
  cudaFree(state->octant_moments_f4);
  state->octant_moments_f4 = nullptr;
  state->short_cell_start = nullptr;
  state->short_cell_end = nullptr;
  state->short_neighborhood_occupancy = nullptr;
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

  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->short_neighborhood_occupancy),
                           sizeof(int) * required_cell_count);
  if (cuda_status != cudaSuccess) {
    if (error_buffer != nullptr && error_buffer_len > 0) {
      std::snprintf(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for short-range neighborhood occupancies failed (%zu cells): %s",
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
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->octant_moments_f4),
                           sizeof(float4) * octant_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for packed octant moments failed",
                    cuda_status);
    return 1;
  }

  state->short_cell_capacity = required_cell_count;
  return 0;
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

  // Size the coarsened pyramid: halve the grid per level until the top level
  // is at most 3 cells per axis, so the per-particle V-list walk covers the
  // whole grid without truncation.
  state->pyramid_level_count = 1;
  state->pyramid_nx[0] = short_nx;
  state->pyramid_ny[0] = short_ny;
  state->pyramid_nz[0] = short_nz;
  state->pyramid_offset[0] = 0;
  std::size_t pyramid_cells = 0;
  while (state->pyramid_level_count < kMaxPyramidLevels) {
    const int level = state->pyramid_level_count;
    const int prev_nx = state->pyramid_nx[level - 1];
    const int prev_ny = state->pyramid_ny[level - 1];
    const int prev_nz = state->pyramid_nz[level - 1];
    if (prev_nx <= 3 && prev_ny <= 3 && prev_nz <= 3) {
      break;
    }
    state->pyramid_nx[level] = (prev_nx + 1) / 2;
    state->pyramid_ny[level] = (prev_ny + 1) / 2;
    state->pyramid_nz[level] = (prev_nz + 1) / 2;
    state->pyramid_offset[level] = pyramid_cells;
    pyramid_cells += static_cast<std::size_t>(state->pyramid_nx[level]) *
                     static_cast<std::size_t>(state->pyramid_ny[level]) *
                     static_cast<std::size_t>(state->pyramid_nz[level]);
    ++state->pyramid_level_count;
  }
  state->pyramid_total_cells = pyramid_cells;

  if (pyramid_cells > state->pyramid_capacity) {
    cudaFree(state->pyramid_mass);
    cudaFree(state->pyramid_com_x);
    cudaFree(state->pyramid_com_y);
    cudaFree(state->pyramid_com_z);
    state->pyramid_mass = nullptr;
    state->pyramid_com_x = nullptr;
    state->pyramid_com_y = nullptr;
    state->pyramid_com_z = nullptr;
    state->pyramid_capacity = 0;

    const std::size_t bytes = sizeof(double) * pyramid_cells;
    if (cudaMalloc(reinterpret_cast<void**>(&state->pyramid_mass), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&state->pyramid_com_x), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&state->pyramid_com_y), bytes) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&state->pyramid_com_z), bytes) != cudaSuccess) {
      fill_error(error_buffer, error_buffer_len, "cudaMalloc for cell pyramid failed");
      return 1;
    }
    state->pyramid_capacity = pyramid_cells;
  }

  const std::size_t moments_needed = short_cell_count + pyramid_cells;
  if (moments_needed > state->moments_f4_capacity) {
    cudaFree(state->cell_moments_f4);
    state->cell_moments_f4 = nullptr;
    state->moments_f4_capacity = 0;

    if (cudaMalloc(reinterpret_cast<void**>(&state->cell_moments_f4),
                   sizeof(float4) * moments_needed) != cudaSuccess) {
      fill_error(error_buffer, error_buffer_len, "cudaMalloc for packed moments failed");
      return 1;
    }
    state->moments_f4_capacity = moments_needed;
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
    const cudaError_t accel_status =
        cudaMemcpy(&state->last_max_accel_sq, state->max_accel_sq, sizeof(double), cudaMemcpyDeviceToHost);
    if (accel_status != cudaSuccess) {
      fill_cuda_error(
          error_buffer, error_buffer_len, "max acceleration download failed", accel_status);
      return std::max(1u, state->max_substeps);
    }
    const double min_global_cell = std::max(
        1.0e-4, std::min(state->cell_size[0], std::min(state->cell_size[1], state->cell_size[2])));
    double integration_scale = min_global_cell;
    if (state->short_cell_count > 0) {
      const double min_short_cell =
          std::max(1.0e-4,
                   std::min(state->short_cell_size[0],
                            std::min(state->short_cell_size[1], state->short_cell_size[2])));
      // The near-field correction varies on a much smaller scale than the global PM mesh.
      // Keep substeps small enough that the fastest particles traverse at most about
      // half a short-range cell per kick, otherwise dense regions heat numerically. The
      // floor stops sub-kpc cells from exploding the substep count; below that scale the
      // per-particle softening dominates anyway.
      integration_scale =
          std::min(integration_scale, std::max(0.5 * min_short_cell, 0.25));
    }
    const double allowed_displacement =
        state->cfl_safety_factor * std::max(1.0e-4, integration_scale);
    const double max_accel_kpc_per_myr2 =
        std::sqrt(std::max(0.0, state->last_max_accel_sq)) * kKpcPerKmPerMyr * kKpcPerKmPerMyr;
    // Velocities were read before the deferred closing half-kick was applied,
    // so bound the missing a*dt_pending without consuming the pending kick
    // (which would forfeit the fusion).
    const double pending_speed_kms = state->half_kick_pending
        ? std::sqrt(std::max(0.0, state->last_max_accel_sq)) * kKpcPerKmPerMyr *
              std::max(state->pending_fast_dt_myr, state->pending_slow_dt_myr)
        : 0.0;
    const double predicted_displacement =
        (max_speed_kms + pending_speed_kms) * dt_myr * kKpcPerKmPerMyr +
        0.5 * max_accel_kpc_per_myr2 * dt_myr * dt_myr;
    const double raw_substeps = predicted_displacement / std::max(allowed_displacement, 1.0e-6);
    std::uint32_t substeps = static_cast<std::uint32_t>(
        std::clamp(std::ceil(raw_substeps), 1.0, static_cast<double>(std::max(1u, state->max_substeps))));

    // The two-tier block-step scheme needs an even substep count so the slow
    // bin's kicks land on substep boundaries.
    if (substeps > 1) {
      if ((substeps & 1u) != 0) {
        substeps = substeps + 1 <= state->max_substeps ? substeps + 1
                                                       : std::max(2u, substeps - 1);
      }
      const double slow_dt_myr = 2.0 * dt_myr / static_cast<double>(substeps);
      const int threads_per_block = 256;
      const int blocks =
          static_cast<int>((state->particle_count + threads_per_block - 1) / threads_per_block);
      assign_time_bins<<<blocks, threads_per_block, 0, state->compute_stream>>>(
          state->particles,
          state->accel_mag,
          state->particle_count,
          slow_dt_myr,
          allowed_displacement,
          state->time_bins);
      const cudaError_t bin_status = cudaGetLastError();
      if (bin_status != cudaSuccess) {
        fill_cuda_error(error_buffer, error_buffer_len, "time bin kernel failed", bin_status);
        return std::max(1u, state->max_substeps);
      }
    }
    return std::max(1u, substeps);
  } catch (const std::exception& error) {
    fill_error(error_buffer, error_buffer_len, error.what());
    return std::max(1u, state->max_substeps);
  }
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

// erfc(x) + (2/sqrt(pi)) * x * exp(-x^2): the Gaussian-split short-range
// force factor evaluated directly in fp32. On this latency-bound kernel the
// ALUs sit mostly idle, so ~30 flops beat a dependent global LUT load.
__device__ __forceinline__ float treepm_short_range_force_factor_analytic_f(
    const float r2, const float half_inv_split) {
  const float x = sqrtf(r2) * half_inv_split;
  return erfcf(x) + 1.1283791671f * x * __expf(-x * x);
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
  const double gx = (particle.position_kpc[0] - origin_x) / cell_x;
  const double gy = (particle.position_kpc[1] - origin_y) / cell_y;
  const double gz = (particle.position_kpc[2] - origin_z) / cell_z;
  // Sources outside the (baryon-tight) short-range box are long-range only:
  // park them in the overflow slot past the last real cell so the sort keeps
  // them out of every geometric neighborhood.
  if (gx < 0.0 || gy < 0.0 || gz < 0.0 ||
      gx >= static_cast<double>(nx) ||
      gy >= static_cast<double>(ny) ||
      gz >= static_cast<double>(nz)) {
    sorted_cell_ids[index] = nx * ny * nz;
    sorted_particle_indices[index] = particle_index;
    return;
  }
  const int ix = min(static_cast<int>(floor(gx)), nx - 1);
  const int iy = min(static_cast<int>(floor(gy)), ny - 1);
  const int iz = min(static_cast<int>(floor(gz)), nz - 1);

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
                                              const int cell_count,
                                              int* cell_start,
                                              int* cell_end) {
  const std::uint64_t index =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= particle_count) {
    return;
  }

  const int cell_id = sorted_cell_ids[index];
  if (cell_id < 0 || cell_id >= cell_count) {
    return;
  }
  if (index == 0 || sorted_cell_ids[index - 1] != cell_id) {
    cell_start[cell_id] = static_cast<int>(index);
  }
  if (index + 1 == particle_count || sorted_cell_ids[index + 1] != cell_id) {
    cell_end[cell_id] = static_cast<int>(index + 1);
  }
}

__global__ void compute_short_neighborhood_occupancy(const int* __restrict__ cell_start,
                                                     const int* __restrict__ cell_end,
                                                     const int nx,
                                                     const int ny,
                                                     const int nz,
                                                     int* __restrict__ occupancy) {
  const std::size_t cell_index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t cell_count =
      static_cast<std::size_t>(nx) * static_cast<std::size_t>(ny) * static_cast<std::size_t>(nz);
  if (cell_index >= cell_count) {
    return;
  }

  const int iz = static_cast<int>(cell_index % static_cast<std::size_t>(nz));
  const std::size_t xy = cell_index / static_cast<std::size_t>(nz);
  const int iy = static_cast<int>(xy % static_cast<std::size_t>(ny));
  const int ix = static_cast<int>(xy / static_cast<std::size_t>(ny));
  int total = 0;
  for (int cx = max(0, ix - 1); cx <= min(nx - 1, ix + 1); ++cx) {
    for (int cy = max(0, iy - 1); cy <= min(ny - 1, iy + 1); ++cy) {
      for (int cz = max(0, iz - 1); cz <= min(nz - 1, iz + 1); ++cz) {
        const int neighbor = short_cell_linear_index(cx, cy, cz, ny, nz);
        const int start = cell_start[neighbor];
        if (start >= 0) {
          total += cell_end[neighbor] - start;
        }
      }
    }
  }
  occupancy[cell_index] = total;
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

// Builds one pyramid level from the next-finer level: each coarse cell is the
// mass-weighted merge of its (up to) 8 children. Inputs and outputs are
// normalized (mass, com) pairs.
__global__ void coarsen_cell_moments(const double* __restrict__ fine_mass,
                                     const double* __restrict__ fine_com_x,
                                     const double* __restrict__ fine_com_y,
                                     const double* __restrict__ fine_com_z,
                                     const int fine_nx,
                                     const int fine_ny,
                                     const int fine_nz,
                                     double* __restrict__ coarse_mass,
                                     double* __restrict__ coarse_com_x,
                                     double* __restrict__ coarse_com_y,
                                     double* __restrict__ coarse_com_z,
                                     const int coarse_nx,
                                     const int coarse_ny,
                                     const int coarse_nz) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t coarse_count = static_cast<std::size_t>(coarse_nx) *
                                   static_cast<std::size_t>(coarse_ny) *
                                   static_cast<std::size_t>(coarse_nz);
  if (index >= coarse_count) {
    return;
  }

  const int cz = static_cast<int>(index % static_cast<std::size_t>(coarse_nz));
  const std::size_t xy = index / static_cast<std::size_t>(coarse_nz);
  const int cy = static_cast<int>(xy % static_cast<std::size_t>(coarse_ny));
  const int cx = static_cast<int>(xy / static_cast<std::size_t>(coarse_ny));

  double mass = 0.0;
  double sum_x = 0.0;
  double sum_y = 0.0;
  double sum_z = 0.0;
  const int fx1 = min(2 * cx + 1, fine_nx - 1);
  const int fy1 = min(2 * cy + 1, fine_ny - 1);
  const int fz1 = min(2 * cz + 1, fine_nz - 1);
  for (int fx = 2 * cx; fx <= fx1; ++fx) {
    for (int fy = 2 * cy; fy <= fy1; ++fy) {
      for (int fz = 2 * cz; fz <= fz1; ++fz) {
        const int fine_index = short_cell_linear_index(fx, fy, fz, fine_ny, fine_nz);
        const double child_mass = fine_mass[fine_index];
        if (!(child_mass > 0.0)) {
          continue;
        }
        mass += child_mass;
        sum_x += child_mass * fine_com_x[fine_index];
        sum_y += child_mass * fine_com_y[fine_index];
        sum_z += child_mass * fine_com_z[fine_index];
      }
    }
  }

  coarse_mass[index] = mass;
  if (mass > 0.0) {
    coarse_com_x[index] = sum_x / mass;
    coarse_com_y[index] = sum_y / mass;
    coarse_com_z[index] = sum_z / mass;
  } else {
    coarse_com_x[index] = 0.0;
    coarse_com_y[index] = 0.0;
    coarse_com_z[index] = 0.0;
  }
}

// Packs normalized (mass, com) moments into origin-relative float4 records.
__global__ void pack_cell_moments_f4(const double* __restrict__ mass,
                                     const double* __restrict__ com_x,
                                     const double* __restrict__ com_y,
                                     const double* __restrict__ com_z,
                                     const std::size_t count,
                                     const double origin_x,
                                     const double origin_y,
                                     const double origin_z,
                                     float4* __restrict__ out) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const double cell_mass = mass[index];
  if (cell_mass > 0.0) {
    out[index] = make_float4(static_cast<float>(com_x[index] - origin_x),
                             static_cast<float>(com_y[index] - origin_y),
                             static_cast<float>(com_z[index] - origin_z),
                             static_cast<float>(cell_mass));
  } else {
    out[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}

// Same packing for the octant tables, whose inputs are raw mass-weighted sums.
__global__ void pack_octant_moments_f4(const double* __restrict__ octant_mass,
                                       const double* __restrict__ octant_sum_x,
                                       const double* __restrict__ octant_sum_y,
                                       const double* __restrict__ octant_sum_z,
                                       const std::size_t count,
                                       const double origin_x,
                                       const double origin_y,
                                       const double origin_z,
                                       float4* __restrict__ out) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) {
    return;
  }
  const double mass = octant_mass[index];
  if (mass > 0.0) {
    out[index] = make_float4(static_cast<float>(octant_sum_x[index] / mass - origin_x),
                             static_cast<float>(octant_sum_y[index] / mass - origin_y),
                             static_cast<float>(octant_sum_z[index] / mass - origin_z),
                             static_cast<float>(mass));
  } else {
    out[index] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  }
}

// Gathers the cell-sorted sources into packed (cell-relative position, mass)
// records so direct-sum loops stream one coalesced float4 per source.
__global__ void gather_sorted_sources(const SimCudaParticle* __restrict__ particles,
                                      const int* __restrict__ sorted_particle_indices,
                                      const int* __restrict__ sorted_cell_ids,
                                      const std::uint32_t source_count,
                                      const int nx,
                                      const int ny,
                                      const int nz,
                                      const double origin_x,
                                      const double origin_y,
                                      const double origin_z,
                                      const double cell_x,
                                      const double cell_y,
                                      const double cell_z,
                                      float4* __restrict__ out_posm,
                                      float* __restrict__ out_softening) {
  const std::uint32_t slot = blockIdx.x * blockDim.x + threadIdx.x;
  if (slot >= source_count) {
    return;
  }
  const int cell_id = sorted_cell_ids[slot];
  if (cell_id >= nx * ny * nz) {
    // Out-of-box overflow slot: never scanned by any geometric neighborhood.
    out_posm[slot] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    out_softening[slot] = 0.0f;
    return;
  }
  const int cell_plane = ny * nz;
  const int ix = cell_id / cell_plane;
  const int rem = cell_id - ix * cell_plane;
  const int iy = rem / nz;
  const int iz = rem - iy * nz;
  const double center_x = origin_x + (static_cast<double>(ix) + 0.5) * cell_x;
  const double center_y = origin_y + (static_cast<double>(iy) + 0.5) * cell_y;
  const double center_z = origin_z + (static_cast<double>(iz) + 0.5) * cell_z;

  const int particle_index = sorted_particle_indices[slot];
  const SimCudaParticle particle = particles[particle_index];
  out_posm[slot] = make_float4(static_cast<float>(particle.position_kpc[0] - center_x),
                               static_cast<float>(particle.position_kpc[1] - center_y),
                               static_cast<float>(particle.position_kpc[2] - center_z),
                               static_cast<float>(particle.mass_msun));
  out_softening[slot] = static_cast<float>(particle.softening_kpc);
}

// The traversal order used by the kick: cell-sorted sources first, then the
// small non-source tail. Compacting the particle array into this order keeps
// warp-neighboring particles adjacent in memory.
__device__ __forceinline__ int traversal_particle_index(
    const int* __restrict__ sorted_particle_indices,
    const std::uint32_t source_count,
    const int* __restrict__ non_source_indices,
    const std::uint64_t slot) {
  return slot < source_count
             ? sorted_particle_indices[slot]
             : non_source_indices[slot - source_count];
}

__global__ void gather_particles_traversal_order(const SimCudaParticle* __restrict__ in,
                                                 const std::uint8_t* __restrict__ bins_in,
                                                 const float* __restrict__ accel_in,
                                                 const float* __restrict__ smoothing_in,
                                                 const float* __restrict__ birth_in,
                                                 const int* __restrict__ sorted_particle_indices,
                                                 const std::uint32_t source_count,
                                                 const int* __restrict__ non_source_indices,
                                                 const std::uint64_t particle_count,
                                                 SimCudaParticle* __restrict__ out,
                                                 std::uint8_t* __restrict__ bins_out,
                                                 float* __restrict__ accel_out,
                                                 float* __restrict__ smoothing_out,
                                                 float* __restrict__ birth_out,
                                                 int* __restrict__ inverse_index) {
  const std::uint64_t slot =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (slot >= particle_count) {
    return;
  }
  const int old_index = traversal_particle_index(
      sorted_particle_indices, source_count, non_source_indices, slot);
  out[slot] = in[old_index];
  bins_out[slot] = bins_in[old_index];
  accel_out[slot] = accel_in[old_index];
  smoothing_out[slot] = smoothing_in[old_index];
  birth_out[slot] = birth_in[old_index];
  inverse_index[old_index] = static_cast<int>(slot);
}

__global__ void remap_index_array(int* __restrict__ indices,
                                  const std::uint32_t count,
                                  const int* __restrict__ inverse_index) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  const int old_index = indices[i];
  if (old_index >= 0) {
    indices[i] = inverse_index[old_index];
  }
}

__global__ void fill_iota(int* __restrict__ out,
                          const std::uint32_t count,
                          const int offset) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) {
    return;
  }
  out[i] = offset + static_cast<int>(i);
}

// Real-space free-space (isolated) Green's function on the mesh, using the
// minimum-image displacement. As long as every source-target pair is within
// half a box length per axis (guaranteed by kGlobalDomainPadding > 2), the
// circular convolution with this kernel reproduces the exact open-boundary
// potential — no periodic images, unlike the naive -4*pi*G/k^2 spectrum.
// Values are pre-multiplied by cell_volume so K (x) rho sums mass, not density.
__global__ void fill_freespace_greens(cufftReal* greens,
                                      const int nx,
                                      const int ny,
                                      const int nz,
                                      const double cell_x,
                                      const double cell_y,
                                      const double cell_z,
                                      const double grav_const) {
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

  const double dx = static_cast<double>((ix <= nx / 2) ? ix : ix - nx) * cell_x;
  const double dy = static_cast<double>((iy <= ny / 2) ? iy : iy - ny) * cell_y;
  const double dz = static_cast<double>((iz <= nz / 2) ? iz : iz - nz) * cell_z;
  const double r = sqrt(dx * dx + dy * dy + dz * dz);
  const double cell_volume = cell_x * cell_y * cell_z;
  const double mean_cell = cbrt(cell_volume);
  // The r=0 self-cell value approximates the potential at the center of a
  // uniform cube of the same volume; its exact value barely matters because
  // the k-space split filter suppresses sub-cell scales anyway.
  const double inv_r = (r > 0.25 * mean_cell) ? 1.0 / r : 2.5 / mean_cell;
  greens[index] = static_cast<cufftReal>(-grav_const * cell_volume * inv_r);
}

__global__ void apply_potential_spectrum(const cufftComplex* density_k,
                                         const cufftComplex* greens_k,
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
                                         const double inv_fft_cells) {
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

  const float wx = sinc_pi(0.5f * kx * static_cast<float>(cell_x));
  const float wy = sinc_pi(0.5f * ky * static_cast<float>(cell_y));
  const float wz = sinc_pi(0.5f * kz * static_cast<float>(cell_z));
  const float window = wx * wy * wz;
  const float window_sq = window * window;
  const float deconvolution = 1.0f / fmaxf(window_sq * window_sq, 1.0e-4f);
  const float split = static_cast<float>(split_radius_kpc);
  const float long_range_filter = expf(-(k_squared * split * split));
  // The extra inv_fft_cells normalizes the forward FFT of the Green's function
  // (cuFFT is unnormalized; the density FFT's factor is applied at force
  // sampling time as before).
  const float scale =
      deconvolution * long_range_filter * static_cast<float>(inv_fft_cells);

  const cufftComplex rho = density_k[index];
  const cufftComplex greens = greens_k[index];
  potential_k[index].x = scale * (rho.x * greens.x - rho.y * greens.y);
  potential_k[index].y = scale * (rho.x * greens.y + rho.y * greens.x);
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

__global__ void __launch_bounds__(256, 3) kick_particles_global(
                                      SimCudaParticle* __restrict__ particles,
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
                                      const double cell_x,
                                      const double cell_y,
                                      const double cell_z,
                                      const int* __restrict__ sorted_particle_indices,
                                      const int* __restrict__ non_source_indices,
                                      const std::uint32_t source_count,
                                      const int* __restrict__ short_cell_start,
                                      const int* __restrict__ short_cell_end,
                                      const int* __restrict__ short_neighborhood_occupancy,
                                      const float4* __restrict__ sorted_source_posm,
                                      const float* __restrict__ sorted_source_softening,
                                      const float4* __restrict__ octant_moments,
                                      const PyramidDeviceView pyramid,
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
                                      const double max_softening_kpc,
                                      const int direct_neighborhood_threshold,
                                      const int short_range_target_baryons_only,
                                      const int* __restrict__ galaxy_smbh_indices,
                                      const std::uint32_t galaxy_count,
                                      const double grav_const,
                                      const int enable_smbh_post_newtonian,
                                      double* __restrict__ max_accel_sq,
                                      const double dt_myr,
                                      const double dt_slow_myr,
                                      const double pending_dt_myr,
                                      const double pending_slow_dt_myr,
                                      const std::uint8_t* __restrict__ time_bins,
                                      const std::uint8_t* __restrict__ previous_time_bins,
                                      const float4* __restrict__ gas_hydro_accel,
                                      float* __restrict__ accel_mag_out) {
  const std::uint64_t thread_slot =
      static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (thread_slot >= particle_count) {
    return;
  }

  // Threads walk targets in cell-sorted order so a warp shares its
  // interaction-list cells (and therefore its moment loads); the leftover
  // non-source particles (SMBHs, filtered components) ride at the tail.
  std::uint64_t index = thread_slot;
  int self_slot = -1;
  if (source_count > 0) {
    if (thread_slot < source_count) {
      index = static_cast<std::uint64_t>(sorted_particle_indices[thread_slot]);
      self_slot = static_cast<int>(thread_slot);
    } else {
      index = static_cast<std::uint64_t>(
          non_source_indices[thread_slot - source_count]);
    }
  }

  // Slow-bin particles integrate with a doubled dt on alternate substeps and
  // skip the off substeps entirely (dt 0).
  // This step's dt follows the current bin; any deferred closing half-kick
  // from the previous step follows the bin epoch it was scheduled under.
  const double particle_dt_myr =
      (time_bins[index] != 0 ? dt_slow_myr : dt_myr) +
      (previous_time_bins[index] != 0 ? pending_slow_dt_myr : pending_dt_myr);
  if (particle_dt_myr == 0.0) {
    return;
  }

  SimCudaParticle& particle = particles[index];
  const double particle_pos_x = particle.position_kpc[0];
  const double particle_pos_y = particle.position_kpc[1];
  const double particle_pos_z = particle.position_kpc[2];
  const double particle_vel_x = particle.velocity_kms[0];
  const double particle_vel_y = particle.velocity_kms[1];
  const double particle_vel_z = particle.velocity_kms[2];
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
  // The force grid is already fully normalized: the density FFT and the
  // Green's-function FFT normalizations are both applied in the spectrum pass.

  // Short-range TreePM correction, only inside the (baryon-tight) short box;
  // outside it the PM force is the whole story.
  const double target_gx = (particle_pos_x - short_origin_x) / short_cell_x;
  const double target_gy = (particle_pos_y - short_origin_y) / short_cell_y;
  const double target_gz = (particle_pos_z - short_origin_z) / short_cell_z;
  const bool inside_short_box =
      target_gx >= 0.0 && target_gy >= 0.0 && target_gz >= 0.0 &&
      target_gx < static_cast<double>(short_nx) &&
      target_gy < static_cast<double>(short_ny) &&
      target_gz < static_cast<double>(short_nz);
  const bool wants_short_range = pyramid.level_count > 0 && inside_short_box &&
      !(short_range_target_baryons_only != 0 && particle_component == 0u);
  if (wants_short_range) {
    const int fine_ix = min(static_cast<int>(floor(target_gx)), short_nx - 1);
    const int fine_iy = min(static_cast<int>(floor(target_gy)), short_ny - 1);
    const int fine_iz = min(static_cast<int>(floor(target_gz)), short_nz - 1);

    const int nbr_x0 = max(0, fine_ix - 1);
    const int nbr_x1 = min(short_nx - 1, fine_ix + 1);
    const int nbr_y0 = max(0, fine_iy - 1);
    const int nbr_y1 = min(short_ny - 1, fine_iy + 1);
    const int nbr_z0 = max(0, fine_iz - 1);
    const int nbr_z1 = min(short_nz - 1, fine_iz + 1);

    const int own_cell = short_cell_linear_index(fine_ix, fine_iy, fine_iz, short_ny, short_nz);
    // Region(0): the 3x3x3 fine-cell neighborhood. Its occupancy is cached once
    // per force-structure build instead of serially re-scanned by every target.
    const bool sparse_neighborhood =
        short_neighborhood_occupancy[own_cell] <= direct_neighborhood_threshold;

    const float cutoff_sq_f = static_cast<float>(short_cutoff_sq);
    const float grav_f = static_cast<float>(grav_const);
    const float half_inv_split_f = short_pm_softening_kpc > 0.0
        ? static_cast<float>(0.5 / short_pm_softening_kpc)
        : 0.0f;
    const float particle_softening_f = static_cast<float>(particle_softening);
    float fax = 0.0f;
    float fay = 0.0f;
    float faz = 0.0f;

    // Direct pairs in cell-relative single precision: the target coordinate is
    // re-expressed against each scanned cell's center (exact fp64 subtraction,
    // then cast), so pair deltas carry ~1e-7 kpc error against softenings of
    // 0.02+ kpc while streaming one float4 per source.
    for (int cx = nbr_x0; cx <= nbr_x1; ++cx) {
      for (int cy = nbr_y0; cy <= nbr_y1; ++cy) {
        for (int cz = nbr_z0; cz <= nbr_z1; ++cz) {
          const int cell = short_cell_linear_index(cx, cy, cz, short_ny, short_nz);
          if (!sparse_neighborhood && cell != own_cell) {
            continue;
          }
          const int start = short_cell_start[cell];
          if (start < 0) {
            continue;
          }
          const int end = short_cell_end[cell];
          const float tx = static_cast<float>(
              particle_pos_x - (short_origin_x + (static_cast<double>(cx) + 0.5) * short_cell_x));
          const float ty = static_cast<float>(
              particle_pos_y - (short_origin_y + (static_cast<double>(cy) + 0.5) * short_cell_y));
          const float tz = static_cast<float>(
              particle_pos_z - (short_origin_z + (static_cast<double>(cz) + 0.5) * short_cell_z));
          // Predicated rather than branched: the unconditional math keeps the
          // loop software-pipelined instead of stalling on divergent skips.
          for (int slot = start; slot < end; ++slot) {
            const float4 source = sorted_source_posm[slot];
            const float dx = source.x - tx;
            const float dy = source.y - ty;
            const float dz = source.z - tz;
            const float r2 = dx * dx + dy * dy + dz * dz;
            const bool interacts =
                slot != self_slot && r2 > 1.0e-10f && r2 <= cutoff_sq_f;
            const float short_softening =
                fmaxf(particle_softening_f, sorted_source_softening[slot]);
            const float inv_r = rsqrtf(r2 + short_softening * short_softening);
            const float correction_scale = interacts
                ? grav_f * source.w * inv_r * inv_r * inv_r *
                      treepm_short_range_force_factor_analytic_f(r2, half_inv_split_f)
                : 0.0f;
            fax += correction_scale * dx;
            fay += correction_scale * dy;
            faz += correction_scale * dz;
          }
        }
      }
    }

    // Monopole paths use coordinates relative to the short box origin
    // (~1e-4 kpc float error against evaluation distances of at least a cell).
    const float pxf = static_cast<float>(particle_pos_x - short_origin_x);
    const float pyf = static_cast<float>(particle_pos_y - short_origin_y);
    const float pzf = static_cast<float>(particle_pos_z - short_origin_z);

    if (!sparse_neighborhood) {
      // Just enough softening to keep a near-coincident neighbor-octant COM
      // from producing a huge kick; anything larger systematically suppresses
      // close forces in dense regions.
      const float octant_eps = static_cast<float>(fmax(
          particle_softening, 0.125 * fmax(short_cell_x, fmax(short_cell_y, short_cell_z))));
      const float octant_eps_sq = octant_eps * octant_eps;
      for (int cx = nbr_x0; cx <= nbr_x1; ++cx) {
        for (int cy = nbr_y0; cy <= nbr_y1; ++cy) {
          for (int cz = nbr_z0; cz <= nbr_z1; ++cz) {
            const int cell = short_cell_linear_index(cx, cy, cz, short_ny, short_nz);
            if (cell == own_cell) {
              continue;
            }
            const std::size_t cell_base = static_cast<std::size_t>(cell) * 8u;
            for (int octant = 0; octant < 8; ++octant) {
              const float4 moment = octant_moments[cell_base + static_cast<std::size_t>(octant)];
              const float dx = moment.x - pxf;
              const float dy = moment.y - pyf;
              const float dz = moment.z - pzf;
              const float r2 = dx * dx + dy * dy + dz * dz;
              const bool interacts =
                  moment.w > 0.0f && r2 > 1.0e-8f && r2 <= cutoff_sq_f;
              const float inv_r = rsqrtf(r2 + octant_eps_sq);
              const float scale = interacts
                  ? grav_f * moment.w * inv_r * inv_r * inv_r *
                        treepm_short_range_force_factor_analytic_f(r2, half_inv_split_f)
                  : 0.0f;
              fax += scale * dx;
              fay += scale * dy;
              faz += scale * dz;
            }
          }
        }
      }
    }

    // Interaction lists per pyramid level (FMM-style V-lists): at each level
    // visit the children of the parent cell's neighborhood that are not
    // themselves neighbors of the particle's cell at that level. Together with
    // Region(0) above this tiles the grid exactly once out to the cutoff.
    const float node_eps = static_cast<float>(fmax(particle_softening, max_softening_kpc));
    const float node_eps_sq = node_eps * node_eps;
    for (int level = 0; level + 1 < pyramid.level_count; ++level) {
      const float level_cell_x = static_cast<float>(short_cell_x) * static_cast<float>(1 << level);
      const float level_cell_y = static_cast<float>(short_cell_y) * static_cast<float>(1 << level);
      const float level_cell_z = static_cast<float>(short_cell_z) * static_cast<float>(1 << level);
      const float half_lx = 0.5f * level_cell_x;
      const float half_ly = 0.5f * level_cell_y;
      const float half_lz = 0.5f * level_cell_z;
      const int own_x = fine_ix >> level;
      const int own_y = fine_iy >> level;
      const int own_z = fine_iz >> level;
      const int parent_x = fine_ix >> (level + 1);
      const int parent_y = fine_iy >> (level + 1);
      const int parent_z = fine_iz >> (level + 1);
      const int level_ny = pyramid.ny[level];
      const int level_nz = pyramid.nz[level];
      const float4* __restrict__ level_moments = pyramid.moments[level];

      const int vx0 = max(0, 2 * (parent_x - 1));
      const int vx1 = min(pyramid.nx[level] - 1, 2 * (parent_x + 1) + 1);
      const int vy0 = max(0, 2 * (parent_y - 1));
      const int vy1 = min(level_ny - 1, 2 * (parent_y + 1) + 1);
      const int vz0 = max(0, 2 * (parent_z - 1));
      const int vz1 = min(level_nz - 1, 2 * (parent_z + 1) + 1);

      for (int cx = vx0; cx <= vx1; ++cx) {
        const bool near_x = cx >= own_x - 1 && cx <= own_x + 1;
        const float center_x = (static_cast<float>(cx) + 0.5f) * level_cell_x;
        const float dx_aabb = fmaxf(fabsf(pxf - center_x) - half_lx, 0.0f);
        const float dx_aabb_sq = dx_aabb * dx_aabb;
        if (dx_aabb_sq > cutoff_sq_f) {
          continue;
        }
        for (int cy = vy0; cy <= vy1; ++cy) {
          const bool near_xy = near_x && cy >= own_y - 1 && cy <= own_y + 1;
          const float center_y = (static_cast<float>(cy) + 0.5f) * level_cell_y;
          const float dy_aabb = fmaxf(fabsf(pyf - center_y) - half_ly, 0.0f);
          const float dxy_aabb_sq = dx_aabb_sq + dy_aabb * dy_aabb;
          if (dxy_aabb_sq > cutoff_sq_f) {
            continue;
          }
          for (int cz = vz0; cz <= vz1; ++cz) {
            if (near_xy && cz >= own_z - 1 && cz <= own_z + 1) {
              continue;
            }
            const float center_z = (static_cast<float>(cz) + 0.5f) * level_cell_z;
            const float dz_aabb = fmaxf(fabsf(pzf - center_z) - half_lz, 0.0f);
            if (dxy_aabb_sq + dz_aabb * dz_aabb > cutoff_sq_f) {
              continue;
            }
            const float4 moment =
                level_moments[short_cell_linear_index(cx, cy, cz, level_ny, level_nz)];
            const float dx = moment.x - pxf;
            const float dy = moment.y - pyf;
            const float dz = moment.z - pzf;
            const float r2 = dx * dx + dy * dy + dz * dz;
            const bool interacts = moment.w > 0.0f && r2 > 1.0e-8f;
            const float inv_r = rsqrtf(r2 + node_eps_sq);
            const float scale = interacts
                ? grav_f * moment.w * inv_r * inv_r * inv_r *
                      treepm_short_range_force_factor_analytic_f(r2, half_inv_split_f)
                : 0.0f;
            fax += scale * dx;
            fay += scale * dy;
            faz += scale * dz;
          }
        }
      }

      // Region(level + 1) already covers the cutoff sphere: deeper levels are
      // entirely beyond the correction radius.
      const double coverage = static_cast<double>(1 << (level + 1)) *
                              fmin(short_cell_x, fmin(short_cell_y, short_cell_z));
      if (coverage >= short_cutoff_kpc) {
        break;
      }
    }

    ax += static_cast<double>(fax);
    ay += static_cast<double>(fay);
    az += static_cast<double>(faz);
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

  if (gas_hydro_accel != nullptr && particle_component == 4u) {
    const float4 hydro = gas_hydro_accel[index];
    ax += static_cast<double>(hydro.x);
    ay += static_cast<double>(hydro.y);
    az += static_cast<double>(hydro.z);
  }

  const double accel_sq = ax * ax + ay * ay + az * az;
  if (max_accel_sq != nullptr) {
    atomic_max_positive_double(max_accel_sq, accel_sq);
  }
  accel_mag_out[index] = static_cast<float>(sqrt(accel_sq));
  particle.velocity_kms[0] = particle_vel_x + ax * particle_dt_myr * kKpcPerKmPerMyr;
  particle.velocity_kms[1] = particle_vel_y + ay * particle_dt_myr * kKpcPerKmPerMyr;
  particle.velocity_kms[2] = particle_vel_z + az * particle_dt_myr * kKpcPerKmPerMyr;
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
                               const float* __restrict__ birth_time_myr,
                               const float sim_time_epoch_myr,
                               const float* __restrict__ gas_density_canonical,
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
    // Evenly spread the sample across the whole visible list; an integer
    // stride would leave the tail (an entire galaxy, in creation order)
    // unsampled whenever visible/sampled rounds down.
    const std::uint64_t sample_slot =
        min(static_cast<std::uint64_t>(index - anchor_count) *
                static_cast<std::uint64_t>(visible_particle_count) /
                static_cast<std::uint64_t>(sampled_count),
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
  // Gas particles all share one mass, so their mass slot streams the local
  // SPH density instead (the renderers color and weight gas by it).
  float mass_out = static_cast<float>(particle.mass_msun);
  if (particle.component == 4u && gas_density_canonical != nullptr) {
    const float rho = gas_density_canonical[source_index];
    if (rho > 0.0f) {
      mass_out = rho;
    }
  }
  out_particles[index].mass_msun = mass_out;
  // High byte: stellar age in 6.4 Myr ticks (255 = primordial). The epoch
  // quantization makes every young star's byte change on the same frames,
  // keeping the preview delta encoder effective between those keyframes.
  std::uint32_t age_ticks = 255u;
  const float birth = birth_time_myr[source_index];
  if (birth > -1.0e29f) {
    const float age = fmaxf(0.0f, sim_time_epoch_myr - birth);
    age_ticks = min(255u, static_cast<std::uint32_t>(age / 6.4f));
  }
  out_particles[index].component = particle.component | (age_ticks << 8);
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
  // Host-side previews have no birth-time context: mark everything primordial.
  preview.component = particle.component | (255u << 8);
  return preview;
}

// (Re)builds the FFT of the free-space Green's function whenever the global
// box geometry changes; the grow-only domain ratchet makes this rare.
int ensure_freespace_greens(DeviceState* state,
                            char* error_buffer,
                            const std::size_t error_buffer_len) {
  if (state->greens_box[0] == state->box_length[0] &&
      state->greens_box[1] == state->box_length[1] &&
      state->greens_box[2] == state->box_length[2]) {
    return 0;
  }

  const int threads_per_block = 256;
  const int real_blocks =
      static_cast<int>((state->real_count + threads_per_block - 1) / threads_per_block);
  fill_freespace_greens<<<real_blocks, threads_per_block, 0, state->compute_stream>>>(
      state->potential_grid,
      state->nx,
      state->ny,
      state->nz,
      state->cell_size[0],
      state->cell_size[1],
      state->cell_size[2],
      state->grav_const);
  const cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "Green's function fill kernel failed", cuda_status);
    return 1;
  }

  const cufftResult fft_status =
      cufftExecR2C(state->forward_plan, state->potential_grid, state->greens_k);
  if (fft_status != CUFFT_SUCCESS) {
    fill_cufft_error(error_buffer, error_buffer_len, "Green's function FFT failed", fft_status);
    return 1;
  }

  state->greens_box[0] = state->box_length[0];
  state->greens_box[1] = state->box_length[1];
  state->greens_box[2] = state->box_length[2];
  return 0;
}

int build_force_mesh(DeviceState* state, char* error_buffer, const std::size_t error_buffer_len) {
  if (ensure_freespace_greens(state, error_buffer, error_buffer_len) != 0) {
    return 1;
  }

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
      state->greens_k,
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
      1.0 / static_cast<double>(state->real_count));
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

// Samples the unfiltered mesh potential at each particle and returns
// PE ~= 0.5 * sum(m_i * phi(x_i)). SMBH point masses are not mesh sources, so
// their (negligible) pair energies are undercounted; this is a diagnostic.
struct PotentialEnergyAccessor {
  const SimCudaParticle* particles;
  const cufftReal* potential;
  int nx;
  int ny;
  int nz;
  double origin_x;
  double origin_y;
  double origin_z;
  double cell_x;
  double cell_y;
  double cell_z;

  __device__ double operator()(const std::uint64_t index) const {
    const SimCudaParticle particle = particles[index];
    if (!(particle.mass_msun > 0.0)) {
      return 0.0;
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
    const int i1 = min(i0 + 1, nx - 1);
    const int j1 = min(j0 + 1, ny - 1);
    const int k1 = min(k0 + 1, nz - 1);
    const double tx = gx - floor(gx);
    const double ty = gy - floor(gy);
    const double tz = gz - floor(gz);

    const double phi =
        potential[real_grid_index(i0, j0, k0, ny, nz)] * (1.0 - tx) * (1.0 - ty) * (1.0 - tz) +
        potential[real_grid_index(i0, j0, k1, ny, nz)] * (1.0 - tx) * (1.0 - ty) * tz +
        potential[real_grid_index(i0, j1, k0, ny, nz)] * (1.0 - tx) * ty * (1.0 - tz) +
        potential[real_grid_index(i0, j1, k1, ny, nz)] * (1.0 - tx) * ty * tz +
        potential[real_grid_index(i1, j0, k0, ny, nz)] * tx * (1.0 - ty) * (1.0 - tz) +
        potential[real_grid_index(i1, j0, k1, ny, nz)] * tx * (1.0 - ty) * tz +
        potential[real_grid_index(i1, j1, k0, ny, nz)] * tx * ty * (1.0 - tz) +
        potential[real_grid_index(i1, j1, k1, ny, nz)] * tx * ty * tz;
    return 0.5 * particle.mass_msun * phi;
  }
};

int compute_potential_energy(DeviceState* state,
                             double* out_potential_energy,
                             char* error_buffer,
                             const std::size_t error_buffer_len) {
  *out_potential_energy = 0.0;
  if (!state->force_state_valid) {
    // density_k does not correspond to current positions; skip rather than
    // paying for an extra deposit + FFT.
    return 0;
  }

  const int threads_per_block = 256;
  const int complex_blocks =
      static_cast<int>((state->complex_count + threads_per_block - 1) / threads_per_block);
  // Unfiltered potential: same Green's function, split radius zero.
  apply_potential_spectrum<<<complex_blocks, threads_per_block, 0, state->compute_stream>>>(
      state->density_k,
      state->greens_k,
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
      0.0,
      1.0 / static_cast<double>(state->real_count));
  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "energy spectrum kernel failed", cuda_status);
    return 1;
  }

  const cufftResult inverse_status =
      cufftExecC2R(state->inverse_plan, state->force_k, state->potential_grid);
  if (inverse_status != CUFFT_SUCCESS) {
    fill_cufft_error(error_buffer, error_buffer_len, "energy potential FFT failed", inverse_status);
    return 1;
  }

  try {
    PotentialEnergyAccessor accessor{};
    accessor.particles = state->particles;
    accessor.potential = state->potential_grid;
    accessor.nx = state->nx;
    accessor.ny = state->ny;
    accessor.nz = state->nz;
    accessor.origin_x = state->domain_origin[0];
    accessor.origin_y = state->domain_origin[1];
    accessor.origin_z = state->domain_origin[2];
    accessor.cell_x = state->cell_size[0];
    accessor.cell_y = state->cell_size[1];
    accessor.cell_z = state->cell_size[2];

    const double raw_sum = thrust::transform_reduce(
        thrust::cuda::par.on(state->compute_stream),
        thrust::counting_iterator<std::uint64_t>(0),
        thrust::counting_iterator<std::uint64_t>(state->particle_count),
        accessor,
        0.0,
        thrust::plus<double>());
    // The spectrum pass already carries the full convolution normalization.
    *out_potential_energy = raw_sum;
  } catch (const std::exception& error) {
    fill_error(error_buffer, error_buffer_len, error.what());
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

  if (!particle_reorder_disabled()) {
    // Compact the particle array into traversal order. Everything computed so
    // far (cell ids, moments, octants) is positional and unaffected; only the
    // index arrays need to follow the permutation.
    stage_start = std::chrono::steady_clock::now();
    const int all_particle_blocks =
        static_cast<int>((state->particle_count + threads_per_block - 1) / threads_per_block);
    gather_particles_traversal_order<<<all_particle_blocks, threads_per_block, 0, state->compute_stream>>>(
        state->particles,
        state->time_bins,
        state->accel_mag,
        state->smoothing_h,
        state->birth_time_myr,
        state->short_sorted_particle_indices,
        state->short_source_particle_count,
        state->non_source_indices,
        state->particle_count,
        state->particles_scratch,
        state->time_bins_scratch,
        state->accel_mag_scratch,
        state->smoothing_h_scratch,
        state->birth_time_scratch,
        state->inverse_index);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "particle reorder gather failed", cuda_status);
      return 1;
    }
    std::swap(state->particles, state->particles_scratch);
    std::swap(state->time_bins, state->time_bins_scratch);
    std::swap(state->accel_mag, state->accel_mag_scratch);
    std::swap(state->smoothing_h, state->smoothing_h_scratch);
    std::swap(state->birth_time_myr, state->birth_time_scratch);

    if (state->galaxy_count > 0) {
      const int smbh_blocks =
          static_cast<int>((state->galaxy_count + threads_per_block - 1) / threads_per_block);
      remap_index_array<<<smbh_blocks, threads_per_block, 0, state->compute_stream>>>(
          state->galaxy_smbh_indices, state->galaxy_count, state->inverse_index);
    }
    if (state->preview_visible_particle_count > 0) {
      const int visible_blocks = static_cast<int>(
          (state->preview_visible_particle_count + threads_per_block - 1) / threads_per_block);
      remap_index_array<<<visible_blocks, threads_per_block, 0, state->compute_stream>>>(
          state->preview_visible_particle_indices,
          state->preview_visible_particle_count,
          state->inverse_index);
    }
    fill_iota<<<particle_blocks, threads_per_block, 0, state->compute_stream>>>(
        state->short_sorted_particle_indices, state->short_source_particle_count, 0);
    fill_iota<<<particle_blocks, threads_per_block, 0, state->compute_stream>>>(
        state->short_source_particle_indices, state->short_source_particle_count, 0);
    if (state->non_source_count > 0) {
      const int non_source_blocks =
          static_cast<int>((state->non_source_count + threads_per_block - 1) / threads_per_block);
      fill_iota<<<non_source_blocks, threads_per_block, 0, state->compute_stream>>>(
          state->non_source_indices,
          state->non_source_count,
          static_cast<int>(state->short_source_particle_count));
    }
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "particle reorder remap failed", cuda_status);
      return 1;
    }
    if (finish_stage("short_range.reorder", stage_start) != 0) {
      return 1;
    }
  }

  stage_start = std::chrono::steady_clock::now();
  build_short_range_cell_ranges<<<particle_blocks, threads_per_block, 0, state->compute_stream>>>(
      state->short_sorted_cell_ids,
      state->short_source_particle_count,
      static_cast<int>(state->short_cell_count),
      state->short_cell_start,
      state->short_cell_end);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "short-range range kernel failed", cuda_status);
    return 1;
  }

  const int cell_blocks =
      static_cast<int>((state->short_cell_count + threads_per_block - 1) / threads_per_block);
  compute_short_neighborhood_occupancy<<<cell_blocks,
                                         threads_per_block,
                                         0,
                                         state->compute_stream>>>(
      state->short_cell_start,
      state->short_cell_end,
      state->short_nx,
      state->short_ny,
      state->short_nz,
      state->short_neighborhood_occupancy);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(
        error_buffer, error_buffer_len, "short-range occupancy kernel failed", cuda_status);
    return 1;
  }
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
  for (int level = 1; level < state->pyramid_level_count; ++level) {
    const double* fine_mass;
    const double* fine_com_x;
    const double* fine_com_y;
    const double* fine_com_z;
    if (level == 1) {
      fine_mass = state->short_cell_mass;
      fine_com_x = state->short_cell_com_x;
      fine_com_y = state->short_cell_com_y;
      fine_com_z = state->short_cell_com_z;
    } else {
      const std::size_t prev_offset = state->pyramid_offset[level - 1];
      fine_mass = state->pyramid_mass + prev_offset;
      fine_com_x = state->pyramid_com_x + prev_offset;
      fine_com_y = state->pyramid_com_y + prev_offset;
      fine_com_z = state->pyramid_com_z + prev_offset;
    }
    const std::size_t offset = state->pyramid_offset[level];
    const std::size_t level_cells = static_cast<std::size_t>(state->pyramid_nx[level]) *
                                    static_cast<std::size_t>(state->pyramid_ny[level]) *
                                    static_cast<std::size_t>(state->pyramid_nz[level]);
    const int level_blocks =
        static_cast<int>((level_cells + threads_per_block - 1) / threads_per_block);
    coarsen_cell_moments<<<level_blocks, threads_per_block, 0, state->compute_stream>>>(
        fine_mass,
        fine_com_x,
        fine_com_y,
        fine_com_z,
        state->pyramid_nx[level - 1],
        state->pyramid_ny[level - 1],
        state->pyramid_nz[level - 1],
        state->pyramid_mass + offset,
        state->pyramid_com_x + offset,
        state->pyramid_com_y + offset,
        state->pyramid_com_z + offset,
        state->pyramid_nx[level],
        state->pyramid_ny[level],
        state->pyramid_nz[level]);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "cell pyramid coarsen kernel failed", cuda_status);
      return 1;
    }
  }
  if (finish_stage("short_range.build_pyramid", stage_start) != 0) {
    return 1;
  }

  stage_start = std::chrono::steady_clock::now();
  {
    const int cell_pack_blocks =
        static_cast<int>((state->short_cell_count + threads_per_block - 1) / threads_per_block);
    pack_cell_moments_f4<<<cell_pack_blocks, threads_per_block, 0, state->compute_stream>>>(
        state->short_cell_mass,
        state->short_cell_com_x,
        state->short_cell_com_y,
        state->short_cell_com_z,
        state->short_cell_count,
        state->short_domain_origin[0],
        state->short_domain_origin[1],
        state->short_domain_origin[2],
        state->cell_moments_f4);
    if (state->pyramid_total_cells > 0) {
      const int pyramid_pack_blocks = static_cast<int>(
          (state->pyramid_total_cells + threads_per_block - 1) / threads_per_block);
      pack_cell_moments_f4<<<pyramid_pack_blocks, threads_per_block, 0, state->compute_stream>>>(
          state->pyramid_mass,
          state->pyramid_com_x,
          state->pyramid_com_y,
          state->pyramid_com_z,
          state->pyramid_total_cells,
          state->short_domain_origin[0],
          state->short_domain_origin[1],
          state->short_domain_origin[2],
          state->cell_moments_f4 + state->short_cell_count);
    }
    const std::size_t octant_count = state->short_cell_count * 8u;
    const int octant_pack_blocks =
        static_cast<int>((octant_count + threads_per_block - 1) / threads_per_block);
    pack_octant_moments_f4<<<octant_pack_blocks, threads_per_block, 0, state->compute_stream>>>(
        state->short_cell_octant_mass,
        state->short_cell_octant_com_x,
        state->short_cell_octant_com_y,
        state->short_cell_octant_com_z,
        octant_count,
        state->short_domain_origin[0],
        state->short_domain_origin[1],
        state->short_domain_origin[2],
        state->octant_moments_f4);
    gather_sorted_sources<<<particle_blocks, threads_per_block, 0, state->compute_stream>>>(
        state->particles,
        state->short_sorted_particle_indices,
        state->short_sorted_cell_ids,
        state->short_source_particle_count,
        state->short_nx,
        state->short_ny,
        state->short_nz,
        state->short_domain_origin[0],
        state->short_domain_origin[1],
        state->short_domain_origin[2],
        state->short_cell_size[0],
        state->short_cell_size[1],
        state->short_cell_size[2],
        state->sorted_source_posm,
        state->sorted_source_softening);
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer, error_buffer_len, "moment packing kernels failed", cuda_status);
      return 1;
    }
  }
  if (finish_stage("short_range.pack_moments", stage_start) != 0) {
    return 1;
  }

  return 0;
}


// ---------------------------------------------------------------------------
// Isothermal SPH: cubic-spline kernel, gather density with rate-limited
// adaptive smoothing lengths, and symmetric pressure + Monaghan viscosity
// forces. Gas neighbors come from a dedicated cell grid over the baryon box
// (cells sized so max(h_i, h_j) always fits the 27-cell neighborhood). With
// masses in Msun, distances in kpc, and cs in km/s, accelerations land
// directly in the kick's (km/s)^2/kpc convention.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float sph_kernel_w(const float r, const float h) {
  const float q = r / h;
  if (q >= 1.0f) {
    return 0.0f;
  }
  const float sigma = 2.5464791f / (h * h * h);  // 8 / pi
  if (q <= 0.5f) {
    return sigma * (1.0f - 6.0f * q * q + 6.0f * q * q * q);
  }
  const float t = 1.0f - q;
  return sigma * 2.0f * t * t * t;
}

__device__ __forceinline__ float sph_kernel_dwdr(const float r, const float h) {
  const float q = r / h;
  if (q >= 1.0f) {
    return 0.0f;
  }
  const float sigma_h = 2.5464791f / (h * h * h * h);
  if (q <= 0.5f) {
    return sigma_h * (-12.0f * q + 18.0f * q * q);
  }
  const float t = 1.0f - q;
  return -6.0f * sigma_h * t * t;
}

__global__ void compute_gas_cells(const SimCudaParticle* __restrict__ particles,
                                  const int* __restrict__ gas_indices,
                                  const std::uint32_t gas_count,
                                  const double origin_x,
                                  const double origin_y,
                                  const double origin_z,
                                  const double cell_kpc,
                                  const int nx,
                                  const int ny,
                                  const int nz,
                                  int* __restrict__ cell_ids,
                                  int* __restrict__ sorted_canonical_seed) {
  const std::uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= gas_count) {
    return;
  }
  const int canonical = gas_indices[g];
  const SimCudaParticle& particle = particles[canonical];
  // Escapers clamp to border cells: their densities are negligible there and
  // the clamp keeps every particle inside the range structure.
  const int ix = min(nx - 1, max(0, static_cast<int>(floor((particle.position_kpc[0] - origin_x) / cell_kpc))));
  const int iy = min(ny - 1, max(0, static_cast<int>(floor((particle.position_kpc[1] - origin_y) / cell_kpc))));
  const int iz = min(nz - 1, max(0, static_cast<int>(floor((particle.position_kpc[2] - origin_z) / cell_kpc))));
  cell_ids[g] = (ix * ny + iy) * nz + iz;
  sorted_canonical_seed[g] = canonical;
}

__global__ void build_gas_cell_ranges(const int* __restrict__ sorted_cell_ids,
                                      const std::uint32_t gas_count,
                                      const int cell_count,
                                      int* __restrict__ cell_start,
                                      int* __restrict__ cell_end) {
  const std::uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= gas_count) {
    return;
  }
  const int cell_id = sorted_cell_ids[i];
  if (cell_id < 0 || cell_id >= cell_count) {
    return;
  }
  if (i == 0 || sorted_cell_ids[i - 1] != cell_id) {
    cell_start[cell_id] = static_cast<int>(i);
  }
  if (i + 1 == gas_count || sorted_cell_ids[i + 1] != cell_id) {
    cell_end[cell_id] = static_cast<int>(i + 1);
  }
}

__global__ void gather_gas_sorted(const SimCudaParticle* __restrict__ particles,
                                  const float* __restrict__ smoothing_h,
                                  const int* __restrict__ sorted_canonical,
                                  const std::uint32_t gas_count,
                                  const double origin_x,
                                  const double origin_y,
                                  const double origin_z,
                                  float4* __restrict__ posh_out,
                                  float4* __restrict__ velm_out) {
  const std::uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= gas_count) {
    return;
  }
  const int canonical = sorted_canonical[g];
  const SimCudaParticle& particle = particles[canonical];
  posh_out[g] = make_float4(static_cast<float>(particle.position_kpc[0] - origin_x),
                            static_cast<float>(particle.position_kpc[1] - origin_y),
                            static_cast<float>(particle.position_kpc[2] - origin_z),
                            smoothing_h[canonical]);
  velm_out[g] = make_float4(static_cast<float>(particle.velocity_kms[0]),
                            static_cast<float>(particle.velocity_kms[1]),
                            static_cast<float>(particle.velocity_kms[2]),
                            static_cast<float>(particle.mass_msun));
}

__global__ void gas_density_kernel(const float4* __restrict__ posh,
                                   const float4* __restrict__ velm,
                                   const int* __restrict__ sorted_canonical,
                                   const std::uint32_t gas_count,
                                   const int* __restrict__ cell_start,
                                   const int* __restrict__ cell_end,
                                   const double cell_kpc,
                                   const int nx,
                                   const int ny,
                                   const int nz,
                                   const float smoothing_eta,
                                   float* __restrict__ density_out,
                                   float* __restrict__ smoothing_h_canonical,
                                   float* __restrict__ density_canonical) {
  const std::uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= gas_count) {
    return;
  }
  const float4 self = posh[g];
  const float h_i = self.w;
  const float cell_f = static_cast<float>(cell_kpc);
  const int ix = min(nx - 1, max(0, static_cast<int>(floorf(self.x / cell_f))));
  const int iy = min(ny - 1, max(0, static_cast<int>(floorf(self.y / cell_f))));
  const int iz = min(nz - 1, max(0, static_cast<int>(floorf(self.z / cell_f))));

  float rho = 0.0f;
  int neighbors = 0;
  for (int cx = max(0, ix - 1); cx <= min(nx - 1, ix + 1); ++cx) {
    for (int cy = max(0, iy - 1); cy <= min(ny - 1, iy + 1); ++cy) {
      for (int cz = max(0, iz - 1); cz <= min(nz - 1, iz + 1); ++cz) {
        const int cell = (cx * ny + cy) * nz + cz;
        const int start = cell_start[cell];
        if (start < 0) {
          continue;
        }
        const int end = cell_end[cell];
        for (int j = start; j < end; ++j) {
          const float4 other = posh[j];
          const float dx = other.x - self.x;
          const float dy = other.y - self.y;
          const float dz = other.z - self.z;
          const float r = sqrtf(dx * dx + dy * dy + dz * dz);
          rho += velm[j].w * sph_kernel_w(r, h_i);
          neighbors += (r > 0.0f && r < h_i) ? 1 : 0;
        }
      }
    }
  }
  rho = fmaxf(rho, 1.0e-12f);
  density_out[g] = rho;

  // Self-contribution trap: with no real neighbors inside h, the kernel sum
  // is just the particle's own W(0) term, which reports a huge density and
  // drives the adaptive h onto (and pins it at) the lower clamp -- isolated
  // ambient gas then reads as "collapsing" everywhere. A healthy neighborhood
  // at eta = 1.3 holds ~9 neighbors; below 4 the estimate has no support, so
  // grow h instead of trusting it and pass the self-free density downstream
  // (star formation and the streamed preview).
  const float mass = velm[g].w;
  const bool starved = neighbors < 4;
  float h_new;
  if (starved) {
    h_new = h_i * 1.25f;
  } else {
    // Predictor update for the next build: h -> eta*(m/rho)^(1/3),
    // rate-limited so a noisy estimate cannot slew the neighborhood violently.
    h_new = smoothing_eta * cbrtf(mass / rho);
    h_new = fminf(fmaxf(h_new, 0.8f * h_i), 1.25f * h_i);
  }
  // Clamped so max(h_i, h_j) always fits the 27-cell search (cell >= 2h_max).
  h_new = fminf(fmaxf(h_new, 0.05f), 0.45f);
  smoothing_h_canonical[sorted_canonical[g]] = h_new;
  const float rho_neighbors = fmaxf(rho - mass * sph_kernel_w(0.0f, h_i), 1.0e-12f);
  density_canonical[sorted_canonical[g]] = starved ? rho_neighbors : rho;
}

__global__ void gas_force_kernel(const float4* __restrict__ posh,
                                 const float4* __restrict__ velm,
                                 const float* __restrict__ density,
                                 const int* __restrict__ sorted_canonical,
                                 const std::uint32_t gas_count,
                                 const int* __restrict__ cell_start,
                                 const int* __restrict__ cell_end,
                                 const double cell_kpc,
                                 const int nx,
                                 const int ny,
                                 const int nz,
                                 const float sound_speed_kms,
                                 const float viscosity_alpha,
                                 float4* __restrict__ gas_accel_canonical) {
  const std::uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= gas_count) {
    return;
  }
  const float4 self = posh[g];
  const float4 self_vel = velm[g];
  const float h_i = self.w;
  const float rho_i = density[g];
  const float cs = sound_speed_kms;
  const float cs2_over_rho_i = cs * cs / rho_i;
  const float cell_f = static_cast<float>(cell_kpc);
  const int ix = min(nx - 1, max(0, static_cast<int>(floorf(self.x / cell_f))));
  const int iy = min(ny - 1, max(0, static_cast<int>(floorf(self.y / cell_f))));
  const int iz = min(nz - 1, max(0, static_cast<int>(floorf(self.z / cell_f))));

  float ax = 0.0f;
  float ay = 0.0f;
  float az = 0.0f;
  for (int cx = max(0, ix - 1); cx <= min(nx - 1, ix + 1); ++cx) {
    for (int cy = max(0, iy - 1); cy <= min(ny - 1, iy + 1); ++cy) {
      for (int cz = max(0, iz - 1); cz <= min(nz - 1, iz + 1); ++cz) {
        const int cell = (cx * ny + cy) * nz + cz;
        const int start = cell_start[cell];
        if (start < 0) {
          continue;
        }
        const int end = cell_end[cell];
        for (int j = start; j < end; ++j) {
          const float4 other = posh[j];
          const float dx = self.x - other.x;
          const float dy = self.y - other.y;
          const float dz = self.z - other.z;
          const float r2 = dx * dx + dy * dy + dz * dz;
          const float h_j = other.w;
          const float h_max = fmaxf(h_i, h_j);
          if (r2 <= 1.0e-12f || r2 >= h_max * h_max) {
            continue;
          }
          const float r = sqrtf(r2);
          const float4 other_vel = velm[j];
          const float rho_j = density[j];

          // Symmetrized kernel gradient magnitude (negative toward smaller r).
          const float grad = 0.5f * (sph_kernel_dwdr(r, h_i) + sph_kernel_dwdr(r, h_j));

          // Isothermal pressure: P/rho^2 = cs^2/rho per side.
          float term = cs2_over_rho_i + cs * cs / rho_j;

          // Monaghan viscosity for approaching pairs; the mu clamp bounds the
          // impulse a single violent shock pair can deliver in one substep.
          const float dvx = self_vel.x - other_vel.x;
          const float dvy = self_vel.y - other_vel.y;
          const float dvz = self_vel.z - other_vel.z;
          const float vr = dvx * dx + dvy * dy + dvz * dz;
          if (vr < 0.0f) {
            const float h_bar = 0.5f * (h_i + h_j);
            float mu = h_bar * vr / (r2 + 0.01f * h_bar * h_bar);
            // Clamp the viscous signal to a few sound speeds: at our substep
            // sizes a hypersonic mu delivers a deceleration impulse larger
            // than the approach speed itself, which overshoots and amplifies
            // instead of damping (measured: gas ping-ponged to >2000 km/s).
            mu = fmaxf(mu, -3.0f * cs);
            const float rho_bar = 0.5f * (rho_i + rho_j);
            term += (-viscosity_alpha * cs * mu + 2.0f * viscosity_alpha * mu * mu) / rho_bar;
          }

          const float scale = other_vel.w * term * grad / r;
          ax -= scale * dx;
          ay -= scale * dy;
          az -= scale * dz;
        }
      }
    }
  }
  gas_accel_canonical[sorted_canonical[g]] = make_float4(ax, ay, az, 0.0f);
}

// Schmidt-law star formation: gas above the density threshold converts with
// probability eps * dt / t_ff per conversion epoch. Conversions are batched
// (every 16th SPH build, probability scaled accordingly) so the preview
// delta encoder sees component changes only on occasional keyframes.
__global__ void form_stars_kernel(SimCudaParticle* __restrict__ particles,
                                  float* __restrict__ birth_time_myr,
                                  const float* __restrict__ density_canonical,
                                  const int* __restrict__ sorted_canonical,
                                  const std::uint32_t gas_count,
                                  const float density_threshold,
                                  const float efficiency_dt_scaled,
                                  const float grav_const,
                                  const float sim_time_myr,
                                  const std::uint32_t salt) {
  const std::uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= gas_count) {
    return;
  }
  // The canonical density is the neighbor-supported estimate (self-free when
  // the neighborhood is starved), so isolated gas can never pass the
  // threshold on its own W(0) term.
  const int canonical = sorted_canonical[g];
  const float rho = density_canonical[canonical];
  if (rho < density_threshold) {
    return;
  }
  // Free-fall time in Myr: sqrt(3*pi / (32 G rho)) with G rho in
  // (km/s)^2/kpc^2; sqrt gives kpc/(km/s) = 977.8 Myr.
  const float t_ff_myr =
      sqrtf(0.2945243f / (grav_const * rho)) * 977.79222f;
  const float p_convert = fminf(efficiency_dt_scaled / t_ff_myr, 0.25f);

  // Cheap counter-based RNG (xorshift of slot + salt).
  std::uint32_t r = (g * 2654435761u) ^ (salt * 2246822519u);
  r ^= r << 13; r ^= r >> 17; r ^= r << 5;
  const float u = static_cast<float>(r & 0x00FFFFFFu) / 16777216.0f;
  if (u < p_convert) {
    particles[canonical].component = 5u;
    birth_time_myr[canonical] = sim_time_myr;
  }
}

// Supernova feedback: clusters younger than ~40 Myr push outward on the gas
// around them. The impulse accumulates into the hydro acceleration buffer
// (atomically -- overlapping bubbles add), so it integrates through the same
// kick path as pressure and respects the deferred-kick invariants.
__global__ void apply_sn_feedback(const SimCudaParticle* __restrict__ particles,
                                  const int* __restrict__ young_star_indices,
                                  const std::uint32_t young_count,
                                  const int* __restrict__ gas_cell_start,
                                  const int* __restrict__ gas_cell_end,
                                  const int* __restrict__ gas_sorted_canonical,
                                  const float4* __restrict__ gas_posh,
                                  const double grid_origin_x,
                                  const double grid_origin_y,
                                  const double grid_origin_z,
                                  const double cell_kpc,
                                  const int nx,
                                  const int ny,
                                  const int nz,
                                  const float feedback_accel,
                                  const float feedback_radius,
                                  float4* __restrict__ gas_accel_canonical) {
  const std::uint32_t y = blockIdx.x * blockDim.x + threadIdx.x;
  if (y >= young_count) {
    return;
  }
  const SimCudaParticle& star = particles[young_star_indices[y]];
  const float sx = static_cast<float>(star.position_kpc[0] - grid_origin_x);
  const float sy = static_cast<float>(star.position_kpc[1] - grid_origin_y);
  const float sz = static_cast<float>(star.position_kpc[2] - grid_origin_z);
  const float cell_f = static_cast<float>(cell_kpc);
  const int ix = min(nx - 1, max(0, static_cast<int>(floorf(sx / cell_f))));
  const int iy = min(ny - 1, max(0, static_cast<int>(floorf(sy / cell_f))));
  const int iz = min(nz - 1, max(0, static_cast<int>(floorf(sz / cell_f))));
  const float radius_sq = feedback_radius * feedback_radius;

  for (int cx = max(0, ix - 1); cx <= min(nx - 1, ix + 1); ++cx) {
    for (int cy = max(0, iy - 1); cy <= min(ny - 1, iy + 1); ++cy) {
      for (int cz = max(0, iz - 1); cz <= min(nz - 1, iz + 1); ++cz) {
        const int cell = (cx * ny + cy) * nz + cz;
        const int start = gas_cell_start[cell];
        if (start < 0) {
          continue;
        }
        const int end = gas_cell_end[cell];
        for (int j = start; j < end; ++j) {
          const float4 gp = gas_posh[j];
          const float dx = gp.x - sx;
          const float dy = gp.y - sy;
          const float dz = gp.z - sz;
          const float r2 = dx * dx + dy * dy + dz * dz;
          if (r2 <= 1.0e-8f || r2 >= radius_sq) {
            continue;
          }
          const float r = sqrtf(r2);
          const float taper = 1.0f - r / feedback_radius;
          const float scale = feedback_accel * taper / r;
          float* accel = reinterpret_cast<float*>(&gas_accel_canonical[gas_sorted_canonical[j]]);
          atomicAdd(accel + 0, scale * dx);
          atomicAdd(accel + 1, scale * dy);
          atomicAdd(accel + 2, scale * dz);
        }
      }
    }
  }
}

struct YoungStarPredicate {
  const SimCudaParticle* particles;
  const float* birth_time_myr;
  float now_myr;
  __device__ bool operator()(const int index) const {
    return particles[index].component == 5u &&
           now_myr - birth_time_myr[index] < 40.0f;
  }
};

struct GasComponentPredicate {
  const SimCudaParticle* particles;
  __device__ bool operator()(const int index) const {
    return particles[index].component == 4u;
  }
};

int build_gas_structure(DeviceState* state,
                        char* error_buffer,
                        const std::size_t error_buffer_len) {
  if (state->gas_count == 0) {
    return 0;
  }
  const int threads_per_block = 256;
  const int gas_blocks =
      static_cast<int>((state->gas_count + threads_per_block - 1) / threads_per_block);

  // Canonical gas indices for this build (the compaction permutes canonical
  // order every substep).
  try {
    const auto copied_end = thrust::copy_if(
        thrust::cuda::par.on(state->compute_stream),
        thrust::counting_iterator<int>(0),
        thrust::counting_iterator<int>(static_cast<int>(state->particle_count)),
        thrust::device_pointer_cast(state->gas_indices),
        GasComponentPredicate{state->particles});
    const std::size_t found =
        static_cast<std::size_t>(copied_end - thrust::device_pointer_cast(state->gas_indices));
    if (found > state->gas_capacity) {
      fill_error(error_buffer, error_buffer_len, "gas particle count exceeded its capacity");
      return 1;
    }
    // Star formation converts gas in place, so the population only shrinks.
    state->gas_count = static_cast<std::uint32_t>(found);
    if (state->gas_count == 0) {
      return 0;
    }
  } catch (const std::exception& error) {
    fill_error(error_buffer, error_buffer_len, error.what());
    return 1;
  }

  // Grid over the baryon-tight box; cell >= 2 * h_max (h clamps at 0.45) so
  // the 27-cell neighborhood always covers the interaction radius.
  double max_extent_cells = 0.0;
  double cell_kpc = 0.5;
  int nx = 0;
  int ny = 0;
  int nz = 0;
  for (;;) {
    nx = max(1, static_cast<int>(std::ceil(state->tight_box_length[0] / cell_kpc)) + 2);
    ny = max(1, static_cast<int>(std::ceil(state->tight_box_length[1] / cell_kpc)) + 2);
    nz = max(1, static_cast<int>(std::ceil(state->tight_box_length[2] / cell_kpc)) + 2);
    max_extent_cells = static_cast<double>(nx) * ny * nz;
    if (max_extent_cells <= static_cast<double>(kMaxShortRangeCells)) {
      break;
    }
    cell_kpc *= 1.5;
  }
  state->gas_nx = nx;
  state->gas_ny = ny;
  state->gas_nz = nz;
  state->gas_grid_cell_kpc = cell_kpc;
  state->gas_grid_origin[0] = state->tight_domain_origin[0] - cell_kpc;
  state->gas_grid_origin[1] = state->tight_domain_origin[1] - cell_kpc;
  state->gas_grid_origin[2] = state->tight_domain_origin[2] - cell_kpc;

  const std::size_t cell_count = static_cast<std::size_t>(nx) * ny * nz;
  if (cell_count > state->gas_cell_capacity) {
    cudaFree(state->gas_cell_start);
    cudaFree(state->gas_cell_end);
    state->gas_cell_start = nullptr;
    state->gas_cell_end = nullptr;
    if (cudaMalloc(reinterpret_cast<void**>(&state->gas_cell_start), sizeof(int) * cell_count) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&state->gas_cell_end), sizeof(int) * cell_count) != cudaSuccess) {
      fill_error(error_buffer, error_buffer_len, "cudaMalloc for gas cell ranges failed");
      return 1;
    }
    state->gas_cell_capacity = cell_count;
  }
  cudaMemsetAsync(state->gas_cell_start, 0xFF, sizeof(int) * cell_count, state->compute_stream);
  cudaMemsetAsync(state->gas_cell_end, 0xFF, sizeof(int) * cell_count, state->compute_stream);

  compute_gas_cells<<<gas_blocks, threads_per_block, 0, state->compute_stream>>>(
      state->particles,
      state->gas_indices,
      state->gas_count,
      state->gas_grid_origin[0],
      state->gas_grid_origin[1],
      state->gas_grid_origin[2],
      cell_kpc,
      nx,
      ny,
      nz,
      state->gas_cell_ids,
      state->gas_sorted_canonical);
  cudaError_t cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "gas cell kernel failed", cuda_status);
    return 1;
  }

  try {
    thrust::sort_by_key(thrust::cuda::par.on(state->compute_stream),
                        thrust::device_pointer_cast(state->gas_cell_ids),
                        thrust::device_pointer_cast(state->gas_cell_ids + state->gas_count),
                        thrust::device_pointer_cast(state->gas_sorted_canonical));
  } catch (const std::exception& error) {
    fill_error(error_buffer, error_buffer_len, error.what());
    return 1;
  }

  build_gas_cell_ranges<<<gas_blocks, threads_per_block, 0, state->compute_stream>>>(
      state->gas_cell_ids,
      state->gas_count,
      static_cast<int>(cell_count),
      state->gas_cell_start,
      state->gas_cell_end);
  gather_gas_sorted<<<gas_blocks, threads_per_block, 0, state->compute_stream>>>(
      state->particles,
      state->smoothing_h,
      state->gas_sorted_canonical,
      state->gas_count,
      state->gas_grid_origin[0],
      state->gas_grid_origin[1],
      state->gas_grid_origin[2],
      state->gas_posh_sorted,
      state->gas_velm_sorted);
  gas_density_kernel<<<gas_blocks, threads_per_block, 0, state->compute_stream>>>(
      state->gas_posh_sorted,
      state->gas_velm_sorted,
      state->gas_sorted_canonical,
      state->gas_count,
      state->gas_cell_start,
      state->gas_cell_end,
      cell_kpc,
      nx,
      ny,
      nz,
      static_cast<float>(state->gas_smoothing_eta),
      state->gas_density_sorted,
      state->smoothing_h,
      state->gas_density_canonical);
  gas_force_kernel<<<gas_blocks, threads_per_block, 0, state->compute_stream>>>(
      state->gas_posh_sorted,
      state->gas_velm_sorted,
      state->gas_density_sorted,
      state->gas_sorted_canonical,
      state->gas_count,
      state->gas_cell_start,
      state->gas_cell_end,
      cell_kpc,
      nx,
      ny,
      nz,
      static_cast<float>(state->gas_sound_speed_kms),
      static_cast<float>(state->gas_viscosity_alpha),
      state->gas_accel);
  if (state->feedback_accel > 0.0 && state->young_star_capacity > 0) {
    std::uint32_t young_count = 0;
    try {
      const auto young_end = thrust::copy_if(
          thrust::cuda::par.on(state->compute_stream),
          thrust::counting_iterator<int>(0),
          thrust::counting_iterator<int>(static_cast<int>(state->particle_count)),
          thrust::device_pointer_cast(state->young_star_indices),
          YoungStarPredicate{state->particles,
                             state->birth_time_myr,
                             static_cast<float>(state->sim_time_myr)});
      young_count = static_cast<std::uint32_t>(
          young_end - thrust::device_pointer_cast(state->young_star_indices));
      young_count = min(young_count, state->young_star_capacity);
    } catch (const std::exception& error) {
      fill_error(error_buffer, error_buffer_len, error.what());
      return 1;
    }
    if (young_count > 0) {
      const int young_blocks =
          static_cast<int>((young_count + threads_per_block - 1) / threads_per_block);
      apply_sn_feedback<<<young_blocks, threads_per_block, 0, state->compute_stream>>>(
          state->particles,
          state->young_star_indices,
          young_count,
          state->gas_cell_start,
          state->gas_cell_end,
          state->gas_sorted_canonical,
          state->gas_posh_sorted,
          state->gas_grid_origin[0],
          state->gas_grid_origin[1],
          state->gas_grid_origin[2],
          state->gas_grid_cell_kpc,
          state->gas_nx,
          state->gas_ny,
          state->gas_nz,
          static_cast<float>(state->feedback_accel),
          static_cast<float>(state->feedback_radius_kpc),
          state->gas_accel);
    }
  }

  state->sph_build_counter += 1;
  if (state->star_formation_efficiency > 0.0 &&
      state->star_formation_density > 0.0 &&
      state->sph_build_counter % 16 == 0) {
    // dt per conversion epoch: 16 builds at roughly the current substep
    // cadence; base_timestep is a good stand-in since builds track substeps.
    const float epoch_dt_myr =
        static_cast<float>(16.0 * state->base_timestep_myr * 0.5);
    form_stars_kernel<<<gas_blocks, threads_per_block, 0, state->compute_stream>>>(
        state->particles,
        state->birth_time_myr,
        state->gas_density_canonical,
        state->gas_sorted_canonical,
        state->gas_count,
        static_cast<float>(state->star_formation_density),
        static_cast<float>(state->star_formation_efficiency) * epoch_dt_myr,
        static_cast<float>(state->grav_const),
        static_cast<float>(state->sim_time_myr),
        static_cast<std::uint32_t>(state->sph_build_counter));
  }
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "SPH kernel launch failed", cuda_status);
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
  const double previous_domain_origin[3] = {
      state->domain_origin[0], state->domain_origin[1], state->domain_origin[2]};
  const double previous_cell_size[3] = {
      state->cell_size[0], state->cell_size[1], state->cell_size[2]};
  if (update_simulation_domain_from_device(state, error_buffer, error_buffer_len) != 0) {
    return 1;
  }
  bool domain_moved = state->cell_size[0] != previous_cell_size[0] ||
                      state->cell_size[1] != previous_cell_size[1] ||
                      state->cell_size[2] != previous_cell_size[2];
  for (int axis = 0; axis < 3 && !domain_moved; ++axis) {
    if (std::abs(state->domain_origin[axis] - previous_domain_origin[axis]) >
        0.5 * state->cell_size[axis]) {
      domain_moved = true;
    }
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

  // The long-range PM field evolves on multi-substep timescales, so reuse it
  // between rebuilds. Any domain-geometry change forces a rebuild because the
  // kick samples the grid through the (updated) origin and cell size.
  state->mesh_age += 1;
  const bool rebuild_mesh = domain_moved || state->mesh_age >= pm_substep_interval();
  if (!rebuild_mesh) {
    // Keep sampling geometry consistent with the retained grid contents; the
    // next rebuild recomputes the domain from the device afresh.
    for (int axis = 0; axis < 3; ++axis) {
      state->domain_origin[axis] = previous_domain_origin[axis];
    }
  }
  if (rebuild_mesh) {
    stage_start = std::chrono::steady_clock::now();
    if (build_force_mesh(state, error_buffer, error_buffer_len) != 0) {
      return 1;
    }
    state->mesh_age = 0;
    if (finish_stage("build_force_mesh", stage_start) != 0) {
      return 1;
    }
  }

  stage_start = std::chrono::steady_clock::now();
  if (build_short_range_structure(state, error_buffer, error_buffer_len) != 0) {
    return 1;
  }
  if (finish_stage("build_short_range_structure", stage_start) != 0) {
    return 1;
  }

  if (state->gas_count > 0) {
    stage_start = std::chrono::steady_clock::now();
    if (build_gas_structure(state, error_buffer, error_buffer_len) != 0) {
      return 1;
    }
    if (finish_stage("sph_pass", stage_start) != 0) {
      return 1;
    }
  }

  state->force_state_valid = true;
  if (profile_stages) {
    flush_profile_stage("build_force_state_total", total_start, std::chrono::steady_clock::now());
  }
  return 0;
}

int apply_force_kick_from_state(DeviceState* state,
                                const int threads_per_block,
                                const double dt_fast_myr,
                                const double dt_slow_myr,
                                const double pending_fast_dt_myr,
                                const double pending_slow_dt_myr,
                                char* error_buffer,
                                const std::size_t error_buffer_len) {
  cudaError_t cuda_status =
      cudaMemsetAsync(state->max_accel_sq, 0, sizeof(double), state->compute_stream);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "max acceleration clear failed", cuda_status);
    return 1;
  }
  PyramidDeviceView pyramid;
  std::uint32_t source_count = 0;
  if (state->short_cell_count > 0 && state->short_source_particle_count > 0) {
    source_count = state->short_source_particle_count;
    pyramid.level_count = state->pyramid_level_count;
    for (int level = 0; level < state->pyramid_level_count; ++level) {
      pyramid.nx[level] = state->pyramid_nx[level];
      pyramid.ny[level] = state->pyramid_ny[level];
      pyramid.nz[level] = state->pyramid_nz[level];
      pyramid.moments[level] = level == 0
          ? state->cell_moments_f4
          : state->cell_moments_f4 + state->short_cell_count + state->pyramid_offset[level];
    }
  }
  const int direct_neighborhood_threshold =
      state->particle_count <= 300'000u ? kShortRangeDirectNeighborhoodThresholdLowN
                                        : kShortRangeDirectNeighborhoodThresholdDefault;

  const int kick_blocks = static_cast<int>(
      (state->particle_count + threads_per_block - 1) / threads_per_block);
  kick_particles_global<<<kick_blocks, threads_per_block, 0, state->compute_stream>>>(
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
    state->cell_size[0],
    state->cell_size[1],
    state->cell_size[2],
    state->short_sorted_particle_indices,
    state->non_source_indices,
    source_count,
    state->short_cell_start,
    state->short_cell_end,
    state->short_neighborhood_occupancy,
    state->sorted_source_posm,
    state->sorted_source_softening,
    state->octant_moments_f4,
    pyramid,
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
    state->max_softening_kpc,
    direct_neighborhood_threshold,
    state->short_range_target_baryons_only ? 1 : 0,
    state->galaxy_smbh_indices,
    state->galaxy_count,
    state->grav_const,
    state->enable_smbh_post_newtonian ? 1 : 0,
    state->max_accel_sq,
    dt_fast_myr,
    dt_slow_myr,
    pending_fast_dt_myr,
    pending_slow_dt_myr,
    state->time_bins,
    state->previous_time_bins,
    state->gas_accel,
    state->accel_mag);
  cuda_status = cudaGetLastError();
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
    // The slow bin's leapfrog runs at twice the substep: its opening half-kick
    // is a full substep_dt, interior kicks are 2*substep_dt on even substep
    // boundaries (dt 0 in between skips the work), and the closing half-kick
    // is substep_dt. With a single substep the bins collapse to one tier.
    const bool use_bins = substeps >= 2;

    // Kick-drift-kick with merged interior kicks: the closing half-kick of one
    // substep and the opening half-kick of the next share the same force
    // state, so they collapse into a single full kick. Positions and
    // velocities are re-synchronized by the trailing half-kick, keeping every
    // FFI call boundary a valid phase-space snapshot.
    auto stage_start = std::chrono::steady_clock::now();
    if (!state->force_state_valid &&
        build_force_state(state, error_buffer, error_buffer_len) != 0) {
      return 1;
    }
    if (finish_profile_stage("run_steps.build_force_state_open", stage_start) != 0) {
      return 1;
    }
    // Fuse the previous advance's deferred closing half-kick into this
    // opening kick: no drift separates them, so the force state is identical
    // and the two launches collapse into one (K(a) then K(b) with the same
    // acceleration equals K(a+b) up to rounding).
    double pending_fast_dt_myr = 0.0;
    double pending_slow_dt_myr = 0.0;
    if (state->half_kick_pending) {
      pending_fast_dt_myr = state->pending_fast_dt_myr;
      pending_slow_dt_myr = state->pending_slow_dt_myr;
      state->half_kick_pending = false;
    }
    stage_start = std::chrono::steady_clock::now();
    if (apply_force_kick_from_state(state,
                                    threads_per_block,
                                    half_substep_dt_myr,
                                    use_bins ? substep_dt_myr : half_substep_dt_myr,
                                    pending_fast_dt_myr,
                                    pending_slow_dt_myr,
                                    error_buffer,
                                    error_buffer_len) != 0) {
      return 1;
    }
    if (finish_profile_stage("run_steps.kick_open", stage_start) != 0) {
      return 1;
    }

    for (std::uint32_t substep = 0; substep < substeps; ++substep) {
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
      if (finish_profile_stage("run_steps.build_force_state", stage_start) != 0) {
        return 1;
      }

      const bool last_substep = substep + 1 == substeps;
      const double kick_dt_myr = last_substep ? half_substep_dt_myr : substep_dt_myr;
      double slow_kick_dt_myr = kick_dt_myr;
      if (use_bins) {
        const bool slow_boundary = (substep + 1) % 2 == 0;
        slow_kick_dt_myr = !slow_boundary ? 0.0
            : last_substep ? substep_dt_myr
                           : 2.0 * substep_dt_myr;
      }
      if (last_substep) {
        // Defer the closing half-kick: it fuses with the next advance's
        // opening kick against this exact force state (which stays valid —
        // nothing drifts between advances). External phase-space reads go
        // through synchronize_pending_kick, so downloads stay exact.
        // Snapshot the bin epoch HERE: the pending dts were scheduled under
        // the current bins, and both consumption paths (the next advance's
        // fused opening kick, after assign_time_bins reassigns, and an
        // immediate synchronize-on-download) must select through this epoch.
        cudaMemcpyAsync(state->previous_time_bins,
                        state->time_bins,
                        state->particle_count,
                        cudaMemcpyDeviceToDevice,
                        state->compute_stream);
        state->pending_fast_dt_myr = kick_dt_myr;
        state->pending_slow_dt_myr = slow_kick_dt_myr;
        state->half_kick_pending = true;
        continue;
      }
      stage_start = std::chrono::steady_clock::now();
      if (apply_force_kick_from_state(state,
                                      threads_per_block,
                                      kick_dt_myr,
                                      slow_kick_dt_myr,
                                      0.0,
                                      0.0,
                                      error_buffer,
                                      error_buffer_len) != 0) {
        return 1;
      }
      if (finish_profile_stage("run_steps.kick", stage_start) != 0) {
        return 1;
      }
    }
  }

  state->sim_time_myr += advanced_myr;
  fill_basic_diagnostics(*state, advanced_myr, diagnostics);
  // The potential energy estimate only feeds the diagnostics panel; refresh
  // it every few advances instead of paying an extra FFT per step.
  state->pe_age = std::min(
      4, state->pe_age + static_cast<int>(std::min<std::uint32_t>(step_count, 4u)));
  if (state->pe_age >= 4 || state->last_potential_energy == 0.0) {
    double potential_energy = 0.0;
    if (compute_potential_energy(state, &potential_energy, error_buffer, error_buffer_len) != 0) {
      return 1;
    }
    if (potential_energy != 0.0) {
      state->last_potential_energy = potential_energy;
    }
    state->pe_age = 0;
  }
  diagnostics->estimated_potential_energy = state->last_potential_energy;
  return 0;
}

// Applies any deferred closing half-kick so external readers observe exact
// (non-staggered) phase space. The force state is still valid whenever a kick
// is pending — nothing drifts between advances — so this is exactly the
// launch run_steps skipped, just performed early.
int synchronize_pending_kick(DeviceState* state,
                             char* error_buffer,
                             const std::size_t error_buffer_len) {
  if (!state->half_kick_pending) {
    return 0;
  }
  if (!state->force_state_valid &&
      build_force_state(state, error_buffer, error_buffer_len) != 0) {
    return 1;
  }
  if (apply_force_kick_from_state(state,
                                  256,
                                  0.0,
                                  0.0,
                                  state->pending_fast_dt_myr,
                                  state->pending_slow_dt_myr,
                                  error_buffer,
                                  error_buffer_len) != 0) {
    return 1;
  }
  state->half_kick_pending = false;
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

  cuda_status =
      cudaMalloc(reinterpret_cast<void**>(&state->particles_scratch), particle_bytes);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for particle scratch failed", cuda_status);
    destroy_state(state);
    return 1;
  }
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->inverse_index),
                           sizeof(int) * params->particle_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for inverse index failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  if (cudaMalloc(reinterpret_cast<void**>(&state->time_bins), params->particle_count) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&state->time_bins_scratch), params->particle_count) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&state->previous_time_bins), params->particle_count) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&state->accel_mag),
                 sizeof(float) * params->particle_count) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&state->accel_mag_scratch),
                 sizeof(float) * params->particle_count) != cudaSuccess) {
    fill_error(error_buffer, error_buffer_len, "cudaMalloc for time bins failed");
    destroy_state(state);
    return 1;
  }
  state->star_formation_efficiency = params->star_formation_efficiency;
  state->star_formation_density = params->star_formation_density_msun_kpc3;
  state->feedback_accel = params->feedback_accel;
  state->feedback_radius_kpc = params->feedback_radius_kpc > 0.0 ? params->feedback_radius_kpc : 0.25;
  state->gas_sound_speed_kms = params->gas_sound_speed_kms;
  state->gas_viscosity_alpha = params->gas_viscosity_alpha > 0.0 ? params->gas_viscosity_alpha : 1.0;
  state->gas_smoothing_eta = params->gas_smoothing_eta > 0.0 ? params->gas_smoothing_eta : 1.3;
  std::uint32_t gas_count = 0;
  for (std::uint64_t i = 0; i < params->particle_count; ++i) {
    if (particles[i].component == 4u) {
      gas_count += 1;
    }
  }
  state->gas_count = gas_count;
  state->gas_capacity = gas_count;
  state->young_star_capacity = gas_count;
  if (cudaMalloc(reinterpret_cast<void**>(&state->smoothing_h),
                 sizeof(float) * params->particle_count) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&state->smoothing_h_scratch),
                 sizeof(float) * params->particle_count) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&state->birth_time_myr),
                 sizeof(float) * params->particle_count) != cudaSuccess ||
      cudaMalloc(reinterpret_cast<void**>(&state->birth_time_scratch),
                 sizeof(float) * params->particle_count) != cudaSuccess) {
    fill_error(error_buffer, error_buffer_len, "cudaMalloc for smoothing lengths failed");
    destroy_state(state);
    return 1;
  }
  {
    std::vector<float> births_host(params->particle_count, -1.0e30f);
    if (cudaMemcpy(state->birth_time_myr,
                   births_host.data(),
                   sizeof(float) * params->particle_count,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      fill_error(error_buffer, error_buffer_len, "birth time upload failed");
      destroy_state(state);
      return 1;
    }
  }
  {
    // Seed h near the ambient-disk equilibrium (~0.2 kpc); the density pass
    // refines it toward eta*(m/rho)^(1/3) within a few builds (rate-limited
    // per step). Seeding low is the dangerous direction: the self-term
    // inflates the density estimate and drags h toward the floor.
    std::vector<float> smoothing_host(params->particle_count, 0.0f);
    for (std::uint64_t i = 0; i < params->particle_count; ++i) {
      if (particles[i].component == 4u) {
        smoothing_host[i] = 0.2f;
      }
    }
    if (cudaMemcpy(state->smoothing_h,
                   smoothing_host.data(),
                   sizeof(float) * params->particle_count,
                   cudaMemcpyHostToDevice) != cudaSuccess) {
      fill_error(error_buffer, error_buffer_len, "smoothing length upload failed");
      destroy_state(state);
      return 1;
    }
  }
  if (gas_count > 0) {
    const std::size_t n = gas_count;
    if (cudaMalloc(reinterpret_cast<void**>(&state->gas_indices), sizeof(int) * n) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&state->gas_cell_ids), sizeof(int) * n) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&state->gas_sorted_canonical), sizeof(int) * n) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&state->gas_posh_sorted), sizeof(float4) * n) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&state->gas_velm_sorted), sizeof(float4) * n) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&state->gas_density_sorted), sizeof(float) * n) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&state->gas_accel),
                   sizeof(float4) * params->particle_count) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&state->young_star_indices),
                   sizeof(int) * n) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&state->gas_density_canonical),
                   sizeof(float) * params->particle_count) != cudaSuccess) {
      fill_error(error_buffer, error_buffer_len, "cudaMalloc for SPH buffers failed");
      destroy_state(state);
      return 1;
    }
    if (cudaMemset(state->gas_accel, 0, sizeof(float4) * params->particle_count) != cudaSuccess ||
        cudaMemset(state->gas_density_canonical, 0, sizeof(float) * params->particle_count) != cudaSuccess) {
      fill_error(error_buffer, error_buffer_len, "cudaMemset for SPH accel failed");
      destroy_state(state);
      return 1;
    }
  }
  if (cudaMemset(state->time_bins, 0, params->particle_count) != cudaSuccess ||
      cudaMemset(state->previous_time_bins, 0, params->particle_count) != cudaSuccess ||
      cudaMemset(state->accel_mag, 0, sizeof(float) * params->particle_count) != cudaSuccess) {
    fill_error(error_buffer, error_buffer_len, "cudaMemset for time bins failed");
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
  std::vector<int> non_source_indices_host;
  std::vector<int> preview_visible_particle_indices_host;
  short_source_particle_indices_host.reserve(params->particle_count / 3);
  preview_visible_particle_indices_host.reserve(params->particle_count / 8);
  const bool baryons_only = short_range_baryons_only();
  state->short_range_target_baryons_only = short_range_target_baryons_only();
  for (std::uint64_t i = 0; i < params->particle_count; ++i) {
    if (particles[i].component != 0u && particles[i].component != 3u && particles[i].mass_msun > 0.0) {
      preview_visible_particle_indices_host.push_back(static_cast<int>(i));
    }
    if (particles[i].component == 3u || !(particles[i].mass_msun > 0.0) ||
        (baryons_only && particles[i].component == 0u)) {
      non_source_indices_host.push_back(static_cast<int>(i));
      continue;
    }
    short_source_particle_indices_host.push_back(static_cast<int>(i));
  }
  state->short_source_particle_count =
      static_cast<std::uint32_t>(short_source_particle_indices_host.size());
  state->non_source_count = static_cast<std::uint32_t>(non_source_indices_host.size());
  state->preview_visible_particle_count =
      static_cast<std::uint32_t>(preview_visible_particle_indices_host.size());
  cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->max_accel_sq), sizeof(double));
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMalloc for max acceleration failed",
                    cuda_status);
    destroy_state(state);
    return 1;
  }
  cuda_status = cudaMemset(state->max_accel_sq, 0, sizeof(double));
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer,
                    error_buffer_len,
                    "cudaMemset for max acceleration failed",
                    cuda_status);
    destroy_state(state);
    return 1;
  }
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

    const std::size_t source_count = state->short_source_particle_count;
    if (cudaMalloc(reinterpret_cast<void**>(&state->sorted_source_posm),
                   sizeof(float4) * source_count) != cudaSuccess ||
        cudaMalloc(reinterpret_cast<void**>(&state->sorted_source_softening),
                   sizeof(float) * source_count) != cudaSuccess) {
      fill_error(error_buffer, error_buffer_len, "cudaMalloc for sorted source SoA failed");
      destroy_state(state);
      return 1;
    }
  }
  if (state->non_source_count > 0) {
    cuda_status = cudaMalloc(reinterpret_cast<void**>(&state->non_source_indices),
                             sizeof(int) * state->non_source_count);
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer,
                      error_buffer_len,
                      "cudaMalloc for non-source indices failed",
                      cuda_status);
      destroy_state(state);
      return 1;
    }
    cuda_status = cudaMemcpy(state->non_source_indices,
                             non_source_indices_host.data(),
                             sizeof(int) * state->non_source_count,
                             cudaMemcpyHostToDevice);
    if (cuda_status != cudaSuccess) {
      fill_cuda_error(error_buffer,
                      error_buffer_len,
                      "cudaMemcpy for non-source indices failed",
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

  cuda_status = cudaMalloc(
      reinterpret_cast<void**>(&state->potential_grid), sizeof(cufftReal) * state->real_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for potential grid failed", cuda_status);
    destroy_state(state);
    return 1;
  }

  cuda_status = cudaMalloc(
      reinterpret_cast<void**>(&state->greens_k), sizeof(cufftComplex) * state->complex_count);
  if (cuda_status != cudaSuccess) {
    fill_cuda_error(error_buffer, error_buffer_len, "cudaMalloc for Green's function spectrum failed", cuda_status);
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
  if (synchronize_pending_kick(state, error_buffer, error_buffer_len) != 0) {
    return 1;
  }
  // The compute stream is non-blocking: the default-stream memcpy below does
  // not order against it, so drain it (including the sync kick just queued)
  // before reading the particle buffer.
  const cudaError_t stream_sync_status = cudaStreamSynchronize(state->compute_stream);
  if (stream_sync_status != cudaSuccess) {
    fill_cuda_error(
        error_buffer, error_buffer_len, "compute stream sync failed", stream_sync_status);
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
