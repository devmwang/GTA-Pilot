set(TARGET_NAME DisplayCaptureDX11)

add_executable(${TARGET_NAME}
    DisplayCaptureDX11.cpp
)

if(MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE /permissive- /W3)
endif()

target_compile_definitions(${TARGET_NAME}
    PRIVATE
        NOMINMAX
        WIN32_LEAN_AND_MEAN
        _CRT_SECURE_NO_WARNINGS
        ZMQ_STATIC
)

target_include_directories(${TARGET_NAME}
    PRIVATE
        ${JSON_INCLUDE_DIR}
)

# libzmq-static already links the Win32 socket libs, but list explicit deps for clarity
set(_extra_win_sock_libs ws2_32 iphlpapi rpcrt4)

target_link_libraries(${TARGET_NAME}
    PRIVATE
        cppzmq-static
        libzmq-static
        d3d11
        dxgi
        dxguid
        winmm
        ${_extra_win_sock_libs}
)
