cmake_minimum_required(VERSION 3.26)
project(gtapilot_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_compile_definitions(NOMINMAX WIN32_LEAN_AND_MEAN UNICODE _UNICODE)

# Ensure vcpkg copies runtime DLLs next to built executables on Windows
set(VCPKG_APPLOCAL_DEPS ON)

# --- Put all artifacts in repo-root/bin (no config subdir) ---
# CMAKE_SOURCE_DIR here is the path to gtapilot/native; repo root is two levels up.
set(REPO_ROOT "${CMAKE_SOURCE_DIR}/../..")
set(BIN_DIR "${REPO_ROOT}/bin")
file(MAKE_DIRECTORY "${BIN_DIR}")

# Multi-config generators (Visual Studio): set per-config to avoid /Release suffix
foreach(cfg IN ITEMS Debug Release RelWithDebInfo MinSizeRel)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${cfg} "${BIN_DIR}")
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${cfg} "${BIN_DIR}")
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${cfg} "${BIN_DIR}")
endforeach()

# Deps (provided via vcpkg manifest)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
find_package(ZeroMQ CONFIG REQUIRED)

# System libs required by the DXGI/D3D11 capture and ZeroMQ
set(SYS_LIBS d3d11 dxgi dxguid ws2_32 winmm)

# --- DisplayCaptureDX11 ---
add_executable(DisplayCaptureDX11
  display_capture/DisplayCaptureDX11.cpp
)
target_link_libraries(DisplayCaptureDX11 PRIVATE
  nlohmann_json::nlohmann_json
  cppzmq
  libzmq
  ${SYS_LIBS}
)

# TODO: Uncomment this once GPU service is implemented
# # --- GpuService (consumer skeleton) ---
# add_executable(GpuService
#   gpu_service/GpuService.cpp
# )
# target_link_libraries(GpuService PRIVATE
#   nlohmann_json::nlohmann_json
#   cppzmq
#   CUDA::cuda_driver
#   CUDA::cudart
#   ${SYS_LIBS}
# )
