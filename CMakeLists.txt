cmake_minimum_required(VERSION 3.26)

set(CMAKE_POLICY_VERSION 3.25)
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

project(gtapilot_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(GTAPILOT_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(GTAPILOT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

set(JSON_INCLUDE_DIR "${GTAPILOT_SOURCE_DIR}/gtapilot/external/json/include" CACHE PATH "Location of nlohmann json headers" FORCE)



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
