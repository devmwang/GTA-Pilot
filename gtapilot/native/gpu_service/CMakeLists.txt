set(TARGET_NAME GpuService)
set(SRC_FILE GpuService.cpp)

set_source_files_properties(${SRC_FILE} PROPERTIES LANGUAGE CUDA)

if(NOT DEFINED CUDAToolkit_ROOT AND DEFINED ENV{CUDA_PATH})
    set(CUDAToolkit_ROOT $ENV{CUDA_PATH})
endif()
find_package(CUDAToolkit REQUIRED)

pybind11_add_module(${TARGET_NAME} MODULE ${SRC_FILE})

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
)

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${JSON_INCLUDE_DIR}
)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        CUDA::cuda_driver
        CUDA::cudart
        d3d11
        dxgi
        dxguid
)
