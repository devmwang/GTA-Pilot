cmake_minimum_required(VERSION 3.26)

set(CMAKE_POLICY_VERSION 3.25)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

project(gtapilot_native LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75;86;89;90)
endif()

set(GTAPILOT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GTAPILOT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(JSON_INCLUDE_DIR "${GTAPILOT_SOURCE_DIR}/gtapilot/external/json/include" CACHE PATH "Location of nlohmann json headers" FORCE)

set(Python3_FIND_VIRTUALENV FIRST)
set(Python3_FIND_STRATEGY LOCATION)
set(Python_FIND_VIRTUALENV FIRST)
set(Python_FIND_STRATEGY LOCATION)
if(EXISTS "${GTAPILOT_SOURCE_DIR}/.venv/Scripts/python.exe")
    set(_gtapilot_python "${GTAPILOT_SOURCE_DIR}/.venv/Scripts/python.exe")
elseif(EXISTS "${GTAPILOT_SOURCE_DIR}/.venv/bin/python")
    set(_gtapilot_python "${GTAPILOT_SOURCE_DIR}/.venv/bin/python")
endif()
if(DEFINED _gtapilot_python)
    set(Python3_EXECUTABLE "${_gtapilot_python}" CACHE FILEPATH "Python interpreter" FORCE)
    set(Python_EXECUTABLE "${_gtapilot_python}" CACHE FILEPATH "Python interpreter" FORCE)
    set(Python3_ROOT_DIR "${GTAPILOT_SOURCE_DIR}/.venv" CACHE PATH "Python root" FORCE)
    set(Python_ROOT_DIR "${GTAPILOT_SOURCE_DIR}/.venv" CACHE PATH "Python root" FORCE)
endif()

set(PYBIND11_TEST OFF CACHE BOOL "Disable pybind11 tests" FORCE)
set(PYBIND11_INSTALL OFF CACHE BOOL "Disable pybind11 install" FORCE)
add_subdirectory("${GTAPILOT_SOURCE_DIR}/gtapilot/external/pybind11" "${CMAKE_BINARY_DIR}/pybind11")

include(FetchContent)

set(BUILD_TESTS OFF CACHE BOOL "Disable libzmq tests" FORCE)
set(ENABLE_DRAFTS OFF CACHE BOOL "Disable libzmq draft APIs" FORCE)
set(WITH_PERF_TOOL OFF CACHE BOOL "Disable libzmq perf tooling" FORCE)
set(WITH_DOC OFF CACHE BOOL "Disable libzmq doc build" FORCE)
set(WITH_DOCS OFF CACHE BOOL "Disable libzmq docs" FORCE)
set(WITH_TLS OFF CACHE BOOL "Disable libzmq TLS" FORCE)
set(WITH_NSS OFF CACHE BOOL "Disable libzmq NSS" FORCE)
set(WITH_LIBSODIUM OFF CACHE BOOL "Disable libsodium" FORCE)
set(BUILD_SHARED OFF CACHE BOOL "Do not build libzmq shared library" FORCE)
set(BUILD_STATIC ON CACHE BOOL "Build libzmq static library" FORCE)

FetchContent_Declare(
    libzmq
    GIT_REPOSITORY https://github.com/zeromq/libzmq.git
    GIT_TAG v4.3.5
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(libzmq)

set(CPPZMQ_BUILD_TESTS OFF CACHE BOOL "Disable cppzmq tests" FORCE)
FetchContent_Declare(
    cppzmq
    GIT_REPOSITORY https://github.com/zeromq/cppzmq.git
    GIT_TAG v4.10.0
    GIT_SHALLOW TRUE
)
FetchContent_MakeAvailable(cppzmq)

if(NOT TARGET gtapilot_json_headers)
    add_library(gtapilot_json_headers INTERFACE)
    target_include_directories(gtapilot_json_headers INTERFACE "${JSON_INCLUDE_DIR}")
endif()
if(NOT TARGET gtapilot::json)
    add_library(gtapilot::json ALIAS gtapilot_json_headers)
endif()

add_subdirectory(gtapilot/native)
