cmake_minimum_required(VERSION 3.24)
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
    CUDA_ARCHITECTURES native
)

install(TARGETS sim_cuda
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
