// RAW-only capture: DXGI Desktop Duplication -> D3D11 shared texture ring
// (keyed mutex) + ZeroMQ pubs. Publishes:
//   - RAW RGB (resized) on tcp://127.0.0.1:55550 topic "frames"
//   - GPU handles on tcp://127.0.0.1:55551 topic "frames_gpu"

#include <d3d11.h>
#include <dxgi1_6.h>
#include <mmsystem.h>
#include <windows.h>
#include <wrl/client.h>

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#pragma comment(lib, "winmm.lib")
static constexpr const char *PUBLISH_ADDRESS = "tcp://127.0.0.1:55550";
static constexpr const char *FRAMES_CPU_TOPIC = "frames";
// static constexpr const char *FRAMES_GPU_TOPIC = "frames_gpu";

using Microsoft::WRL::ComPtr;
using json = nlohmann::json;

static constexpr int CAP_FPS = 20;
static constexpr int RING_SIZE = 4;
static constexpr DXGI_FORMAT CAP_FMT = DXGI_FORMAT_B8G8R8A8_UNORM;

static void hrx(HRESULT hr, const char *where) {
    if (FAILED(hr)) {
        std::cerr << where << " failed: 0x" << std::hex << hr << std::dec
                  << "\n";
        throw std::runtime_error(where);
    }
}

struct DupCtx {
    ComPtr<ID3D11Device> dev;
    ComPtr<ID3D11DeviceContext> ctx;
    ComPtr<IDXGIOutputDuplication> dup;
    UINT W = 0, H = 0;

    void init_with_output(ComPtr<IDXGIOutput> output) {
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT |
                     D3D11_CREATE_DEVICE_SINGLETHREADED;
#ifdef _DEBUG
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        // Create device on the adapter that owns the chosen output
        ComPtr<IDXGIAdapter> adapter;
        hrx(output->GetParent(
                __uuidof(IDXGIAdapter),
                reinterpret_cast<void **>(adapter.ReleaseAndGetAddressOf())),
            "GetParent IDXGIAdapter");
        D3D_FEATURE_LEVEL fl;
        hrx(D3D11CreateDevice(adapter.Get(), D3D_DRIVER_TYPE_UNKNOWN, nullptr,
                              flags, nullptr, 0, D3D11_SDK_VERSION, &dev, &fl,
                              &ctx),
            "D3D11CreateDevice");

        ComPtr<IDXGIOutput1> out1;
        hrx(output.As(&out1), "QI IDXGIOutput1");
        hrx(out1->DuplicateOutput(dev.Get(), &dup), "DuplicateOutput");
        DXGI_OUTDUPL_DESC desc{};
        dup->GetDesc(&desc);
        W = desc.ModeDesc.Width;
        H = desc.ModeDesc.Height;
        std::cout << "[DisplayCaptureDX11] DesktopDup " << W << "x" << H
                  << "\n";
    }

    ComPtr<ID3D11Texture2D> acquire() {
        ComPtr<IDXGIResource> res;
        DXGI_OUTDUPL_FRAME_INFO finfo{};
        HRESULT hr = dup->AcquireNextFrame(16, &finfo, &res);
        if (hr == DXGI_ERROR_WAIT_TIMEOUT)
            return nullptr;
        if (hr == DXGI_ERROR_ACCESS_LOST)
            throw std::runtime_error("DXGI_ERROR_ACCESS_LOST");
        if (hr == DXGI_ERROR_DEVICE_REMOVED)
            throw std::runtime_error("DXGI_ERROR_DEVICE_REMOVED");
        hrx(hr, "AcquireNextFrame");
        ComPtr<ID3D11Texture2D> tex;
        hrx(res.As(&tex), "As Texture2D");
        return tex;
    }
};

struct Slot {
    ComPtr<ID3D11Texture2D> tex;
    ComPtr<IDXGIKeyedMutex> mtx;
    HANDLE shared = nullptr;
    UINT W = 0, H = 0;
};

static HANDLE createShared(ID3D11Texture2D *t) {
    ComPtr<IDXGIResource1> r1;
    hrx(t->QueryInterface(IID_PPV_ARGS(&r1)), "QI IDXGIResource1");
    HANDLE h{};
    hrx(r1->CreateSharedHandle(
            nullptr, DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
            nullptr, &h),
        "CreateSharedHandle");
    return h;
}

static ComPtr<ID3D11Texture2D> makeSharedTex(ID3D11Device *dev, UINT W,
                                             UINT H) {
    D3D11_TEXTURE2D_DESC d{};
    d.Width = W;
    d.Height = H;
    d.MipLevels = 1;
    d.ArraySize = 1;
    d.Format = CAP_FMT;
    d.SampleDesc.Count = 1;
    d.Usage = D3D11_USAGE_DEFAULT;
    d.BindFlags = D3D11_BIND_SHADER_RESOURCE | D3D11_BIND_RENDER_TARGET;
    d.MiscFlags = D3D11_RESOURCE_MISC_SHARED_NTHANDLE |
                  D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX;
    ComPtr<ID3D11Texture2D> t;
    hrx(dev->CreateTexture2D(&d, nullptr, &t), "CreateTexture2D");
    return t;
}

// CPU: convert BGRA8 to RGB8 with nearest resize to (outW,outH).
static void convertBGRA_to_RGB_resized(const uint8_t *bgra, int srcW, int srcH,
                                       int srcPitch, int outW, int outH,
                                       std::vector<uint8_t> &outRGB) {
    outRGB.resize(outW * outH * 3);
    for (int y = 0; y < outH; ++y) {
        int sy = y * srcH / outH;
        const uint8_t *srow = bgra + sy * srcPitch;
        uint8_t *drow = &outRGB[y * outW * 3];
        for (int x = 0; x < outW; ++x) {
            int sx = x * srcW / outW;
            const uint8_t *p = srow + sx * 4;
            drow[x * 3 + 0] = p[2]; // R  (BGRA -> RGB)
            drow[x * 3 + 1] = p[1]; // G
            drow[x * 3 + 2] = p[0]; // B
        }
    }
}

// RAII: request high‑resolution Windows scheduler timer for more accurate
// sleeps. Without this, default timer granularity (~15.6 ms) makes 50 ms
// periods jittery and tends to quantize around ~16 FPS.
struct ScopedTimerResolution {
    explicit ScopedTimerResolution(UINT ms) : _ms(ms), _active(false) {
        if (timeBeginPeriod(_ms) == TIMERR_NOERROR) {
            _active = true;
        }
    }
    ~ScopedTimerResolution() {
        if (_active) {
            timeEndPeriod(_ms);
        }
    }

  private:
    UINT _ms;
    bool _active;
};

struct ScopedReleaseFrame {
    IDXGIOutputDuplication *dup;
    bool armed;
    ScopedReleaseFrame(IDXGIOutputDuplication *d, bool a) : dup(d), armed(a) {}
    ~ScopedReleaseFrame() {
        if (armed && dup) {
            dup->ReleaseFrame();
        }
    }
};

// static inline uint64_t qpc_now() {
//     LARGE_INTEGER li{};
//     QueryPerformanceCounter(&li);
//     return static_cast<uint64_t>(li.QuadPart);
// }

// static inline uint64_t qpc_hz() {
//     static uint64_t hz = []() {
//         LARGE_INTEGER li{};
//         QueryPerformanceFrequency(&li);
//         return static_cast<uint64_t>(li.QuadPart);
//     }();
//     return hz;
// }

static ComPtr<IDXGIOutput> select_output_by_index(int index, int &out_count) {
    out_count = 0;
    ComPtr<IDXGIFactory1> factory;
    hrx(CreateDXGIFactory1(
            __uuidof(IDXGIFactory1),
            reinterpret_cast<void **>(factory.ReleaseAndGetAddressOf())),
        "CreateDXGIFactory1");
    for (UINT a = 0;; ++a) {
        ComPtr<IDXGIAdapter> adp;
        if (factory->EnumAdapters(a, &adp) == DXGI_ERROR_NOT_FOUND)
            break;
        for (UINT o = 0;; ++o) {
            ComPtr<IDXGIOutput> out;
            if (adp->EnumOutputs(o, &out) == DXGI_ERROR_NOT_FOUND)
                break;
            if (index < 0 || out_count == index) {
                return out;
            }
            ++out_count;
        }
    }
    return nullptr;
}

int main(int argc, char **argv) {
    try {
        // Improve sleep precision to ~1 ms to reduce frame pacing jitter.
        ScopedTimerResolution _timerRes(1);
        // Slightly elevate thread priority to reduce scheduling latency.
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);

        // Parse CLI for --display-id
        int requestedDisplay = -1; // -1 means default (first)
        for (int i = 1; i < argc; ++i) {
            const char *arg = argv[i];
            if (strcmp(arg, "--display-id") == 0 && i + 1 < argc) {
                requestedDisplay = std::atoi(argv[++i]);
            } else if (strncmp(arg, "--display-id=", 13) == 0) {
                requestedDisplay = std::atoi(arg + 13);
            }
        }

        // RAW publish size (defaults to 1920x1080); override with
        // GTAPILOT_RAW_SIZE=WxH
        int rawW = 1920, rawH = 1080;
        if (const char *s = getenv("GTAPILOT_RAW_SIZE")) {
            int w = 0, h = 0;
            if (sscanf_s(s, "%dx%d", &w, &h) == 2 && w > 0 && h > 0) {
                rawW = w;
                rawH = h;
            }
        }
        std::cout << "[DisplayCaptureDX11] RAW size " << rawW << "x" << rawH
                  << "\n";

        // Select output
        int totalOutputs = 0;
        ComPtr<IDXGIOutput> chosenOutput = select_output_by_index(
            requestedDisplay < 0 ? 0 : requestedDisplay, totalOutputs);
        if (!chosenOutput) {
            std::cerr << "[DisplayCaptureDX11] No display outputs found or "
                         "index out of range.\n";
            return 1;
        }

        // Describe selected output
        DXGI_OUTPUT_DESC odesc{};
        chosenOutput->GetDesc(&odesc);
        std::wstring devName(odesc.DeviceName);
        std::wcout << L"[DisplayCaptureDX11] Selected display #"
                   << (requestedDisplay < 0 ? 0 : requestedDisplay) << L" ("
                   << devName << L")" << std::endl;

        // ZMQ publishers (created once; sockets survive D3D re-inits)
        zmq::context_t zctx(1);

        zmq::socket_t pubRAW(zctx, zmq::socket_type::pub);
        pubRAW.set(zmq::sockopt::sndhwm, 1);
        pubRAW.set(zmq::sockopt::linger, 0);
        pubRAW.bind(PUBLISH_ADDRESS); // frames_raw

        // Disable GPU-only path since we're not using it currently
        // zmq::socket_t pubGPU(zctx, zmq::socket_type::pub);
        // pubGPU.set(zmq::sockopt::sndhwm, 1);
        // pubGPU.set(zmq::sockopt::linger, 0);
        // pubGPU.bind("tcp://127.0.0.1:55551"); // frames_gpu

        // Use nanoseconds to avoid integer rounding; target period = 50 ms @ 20

        // FPS
        const auto period = std::chrono::nanoseconds(1'000'000'000 / CAP_FPS);
        uint64_t fid = 1;

        std::vector<uint8_t> rgb;

        // Track last published so we can republish when no new frame arrives
        // (DXGI_ERROR_WAIT_TIMEOUT)
        bool haveLastCPU = false;

        // Disable GPU-only path since we're not using it currently
        // bool haveLastGPU = false;
        // int lastGPUSlot = -1;

        // Re-init loop: rebuild D3D resources upon duplication/device loss
        for (;;) {
            try {
                DupCtx cap;
                cap.init_with_output(chosenOutput);
                auto dev = cap.dev;
                auto ctx = cap.ctx;
                UINT W = cap.W, H = cap.H;

                // Shared ring for GPU path
                std::vector<Slot> ring(RING_SIZE);
                for (int i = 0; i < RING_SIZE; ++i) {
                    ring[i].tex = makeSharedTex(dev.Get(), W, H);
                    ring[i].tex.As(&ring[i].mtx);
                    ring[i].shared = createShared(ring[i].tex.Get());
                    ring[i].W = W;
                    ring[i].H = H;
                }

                // Double-buffered staging textures to reduce Map stalls
                D3D11_TEXTURE2D_DESC sd{};
                sd.Width = W;
                sd.Height = H;
                sd.MipLevels = 1;
                sd.ArraySize = 1;
                sd.Format = CAP_FMT;
                sd.SampleDesc.Count = 1;
                sd.Usage = D3D11_USAGE_STAGING;
                sd.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
                std::array<ComPtr<ID3D11Texture2D>, 2> stagingRing;
                hrx(dev->CreateTexture2D(&sd, nullptr, &stagingRing[0]),
                    "Create staging[0]");
                hrx(dev->CreateTexture2D(&sd, nullptr, &stagingRing[1]),
                    "Create staging[1]");
                int copyIndex = 0;
                int readIndex = 1;
                bool stagingPrimed = false;

                // Reset last-frame state on reinit (avoid stale publishes)
                haveLastCPU = false;
                // haveLastGPU = false;
                // lastGPUSlot = -1;

                // Fixed‑deadline scheduler
                auto next_deadline = std::chrono::steady_clock::now() + period;

                while (true) {
                    ComPtr<ID3D11Texture2D> src = cap.acquire();
                    bool acquired = (src != nullptr);
                    ScopedReleaseFrame _rel(cap.dup.Get(), acquired);
                    bool published_any = false;

                    if (acquired) {
                        // Disable GPU-only path since we're not using it
                        // currently

                        // // Write to GPU ring (non-blocking)
                        // int slot = int(fid % RING_SIZE);
                        // Slot &s = ring[slot];
                        // HRESULT kmhr =
                        //     s.mtx->AcquireSync(/*Key*/ 0, /*msTimeout*/ 0);

                        // if (kmhr == WAIT_TIMEOUT) {
                        //     // Skip GPU ring update this frame.
                        // } else {
                        //     hrx(kmhr, "Acquire writer mutex");
                        //     ctx->CopyResource(s.tex.Get(), src.Get());
                        //     hrx(s.mtx->ReleaseSync(/*Key*/ 0),
                        //         "Release writer mutex");

                        //     haveLastGPU = true;
                        //     lastGPUSlot = slot;

                        //     json meta = {{"slot", slot},
                        //                  {"handle", (uint64_t)s.shared},
                        //                  {"w", s.W},
                        //                  {"h", s.H},
                        //                  {"format", "BGRA8"},
                        //                  {"frame_id", fid},
                        //                  {"qpc", qpc_now()},
                        //                  {"qpc_hz", qpc_hz()}};
                        //     std::string m = meta.dump();
                        //     zmq::message_t t(FRAMES_GPU_TOPIC,
                        //                      strlen(FRAMES_GPU_TOPIC)),
                        //         j(m.data(), m.size());
                        //     pubGPU.send(t, zmq::send_flags::sndmore);
                        //     pubGPU.send(j, zmq::send_flags::none);
                        //     published_any = true; // GPU published
                        // }

                        // RAW CPU path using staging ring
                        ctx->CopyResource(stagingRing[copyIndex].Get(),
                                          src.Get());
                        if (stagingPrimed) {
                            D3D11_MAPPED_SUBRESOURCE map{};
                            hrx(ctx->Map(stagingRing[readIndex].Get(), 0,
                                         D3D11_MAP_READ, 0, &map),
                                "Map staging");
                            convertBGRA_to_RGB_resized(
                                reinterpret_cast<const uint8_t *>(map.pData),
                                static_cast<int>(W), static_cast<int>(H),
                                static_cast<int>(map.RowPitch), rawW, rawH,
                                rgb);
                            ctx->Unmap(stagingRing[readIndex].Get(), 0);
                            haveLastCPU = true;

                            json metadata = {
                                {"encoding", "raw"}, {"w", rawW},
                                {"h", rawH},         {"channels", 3},
                                {"dtype", "uint8"},  {"frame_id", fid}};
                            std::string metadata_bytes = metadata.dump();
                            zmq::message_t topic_payload(
                                FRAMES_CPU_TOPIC, strlen(FRAMES_CPU_TOPIC));
                            zmq::message_t metadata_payload(
                                metadata_bytes.data(), metadata_bytes.size());
                            zmq::message_t frame_payload(rgb.data(),
                                                         rgb.size());
                            pubRAW.send(topic_payload,
                                        zmq::send_flags::sndmore);
                            pubRAW.send(metadata_payload,
                                        zmq::send_flags::sndmore);
                            pubRAW.send(frame_payload, zmq::send_flags::none);
                            published_any = true; // RAW published
                        }
                        // Rotate staging indices
                        std::swap(copyIndex, readIndex);
                        stagingPrimed = true;
                    } else {
                        // Republish last known GPU/CPU frames when no new frame
                        // arrived

                        // Disable GPU-only path since we're not using it
                        // currently if (haveLastGPU && lastGPUSlot >= 0) {
                        //     const Slot &s = ring[lastGPUSlot];
                        //     json meta = {{"slot", lastGPUSlot},
                        //                  {"handle", (uint64_t)s.shared},
                        //                  {"w", s.W},
                        //                  {"h", s.H},
                        //                  {"format", "BGRA8"},
                        //                  {"frame_id", fid},
                        //                  {"qpc", qpc_now()},
                        //                  {"qpc_hz", qpc_hz()}};
                        //     std::string m = meta.dump();
                        //     zmq::message_t t(FRAMES_GPU_TOPIC,
                        //                      strlen(FRAMES_GPU_TOPIC)),
                        //         j(m.data(), m.size());
                        //     pubGPU.send(t, zmq::send_flags::sndmore);
                        //     pubGPU.send(j, zmq::send_flags::none);
                        //     published_any = true;
                        // }

                        if (haveLastCPU) {
                            json metadata = {
                                {"encoding", "raw"}, {"w", rawW},
                                {"h", rawH},         {"channels", 3},
                                {"dtype", "uint8"},  {"frame_id", fid}};
                            std::string metadata_bytes = metadata.dump();
                            zmq::message_t topic_payload(
                                FRAMES_CPU_TOPIC, strlen(FRAMES_CPU_TOPIC));
                            zmq::message_t metadata_payload(
                                metadata_bytes.data(), metadata_bytes.size());
                            zmq::message_t frame_payload(rgb.data(),
                                                         rgb.size());
                            pubRAW.send(topic_payload,
                                        zmq::send_flags::sndmore);
                            pubRAW.send(metadata_payload,
                                        zmq::send_flags::sndmore);
                            pubRAW.send(frame_payload, zmq::send_flags::none);
                            published_any = true;
                        }
                    }

                    if (published_any) {
                        ++fid;
                    }

                    // Sleep until the fixed next deadline to maintain
                    // consistent cadence
                    auto now = std::chrono::steady_clock::now();
                    if (now < next_deadline) {
                        std::this_thread::sleep_until(next_deadline);
                    }
                    next_deadline += period;
                    auto lag = std::chrono::steady_clock::now() - next_deadline;
                    if (lag > period) {
                        next_deadline =
                            std::chrono::steady_clock::now() + period;
                    }
                }
            } catch (const std::runtime_error &e) {
                // Duplication or device loss: try to reinitialize once more
                if (std::string(e.what()) ==
                        std::string("DXGI_ERROR_ACCESS_LOST") ||
                    std::string(e.what()) ==
                        std::string("DXGI_ERROR_DEVICE_REMOVED")) {
                    std::cerr << "[DisplayCaptureDX11] Duplication/device "
                                 "lost, reinitializing...\n";
                    continue;
                }

                throw;
            }
        }
    } catch (const std::exception &e) {
        std::cerr << "[DisplayCaptureDX11] Fatal: " << e.what() << "\n";
        return 1;
    }
}