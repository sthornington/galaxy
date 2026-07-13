cmake_minimum_required(VERSION 3.24)

# Allow builders (or CI without a GPU) to pin the target architecture; default
# to detecting the local GPU, with GB10 (sm_121) as the documented target.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES AND NOT DEFINED ENV{CMAKE_CUDA_ARCHITECTURES})
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

project(sim_cuda LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

add_library(sim_cuda STATIC src/sim_cuda.cu)
target_include_directories(sim_cuda PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(sim_cuda PUBLIC CUDA::cudart CUDA::cufft)

set_target_properties(
  sim_cuda
  PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)

install(TARGETS sim_cuda
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
