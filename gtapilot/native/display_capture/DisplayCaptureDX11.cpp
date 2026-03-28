// Atlas capture source: DXGI Desktop Duplication -> D3D11 staging ring ->
// CPU conversion -> fixed-deadline 60 Hz publisher. Publishes raw RGB frames on
// tcp://127.0.0.1:55550 topic "frames" using the shared generic channel
// envelope contract.

#include <d3d11.h>
#include <dxgi1_6.h>
#include <mmsystem.h>
#include <windows.h>
#include <wrl/client.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <zmq.hpp>

#pragma comment(lib, "winmm.lib")

static constexpr const char *PUBLISH_ADDRESS = "tcp://127.0.0.1:55550";
static constexpr const char *FRAMES_CPU_TOPIC = "frames";
static constexpr const char *VISION_CHANNEL = "vision.frames";
static constexpr const char *FRAME_SOURCE = "display_capture_dx11";
static constexpr int CHANNEL_ENVELOPE_VERSION = 1;

using Microsoft::WRL::ComPtr;
using json = nlohmann::json;

static constexpr int CAP_FPS = 60;
static constexpr int CAP_FRAME_TIMEOUT_MS = 17;
static constexpr int STAGING_RING_SIZE = 4;
static constexpr DXGI_FORMAT CAP_FMT = DXGI_FORMAT_B8G8R8A8_UNORM;
static constexpr uint64_t MISSED_DEADLINE_THRESHOLD_NS = 2'000'000ULL;

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
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
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
        HRESULT hr = dup->AcquireNextFrame(CAP_FRAME_TIMEOUT_MS, &finfo, &res);
        if (hr == DXGI_ERROR_WAIT_TIMEOUT) {
            return nullptr;
        }
        if (hr == DXGI_ERROR_ACCESS_LOST) {
            throw std::runtime_error("DXGI_ERROR_ACCESS_LOST");
        }
        if (hr == DXGI_ERROR_DEVICE_REMOVED) {
            throw std::runtime_error("DXGI_ERROR_DEVICE_REMOVED");
        }
        hrx(hr, "AcquireNextFrame");
        ComPtr<ID3D11Texture2D> tex;
        hrx(res.As(&tex), "As Texture2D");
        return tex;
    }
};

static void convertBGRA_to_RGB_resized(const uint8_t *bgra, int srcW, int srcH,
                                       int srcPitch, int outW, int outH,
                                       std::vector<uint8_t> &outRGB) {
    outRGB.resize(static_cast<size_t>(outW) * static_cast<size_t>(outH) * 3U);
    for (int y = 0; y < outH; ++y) {
        int sy = y * srcH / outH;
        const uint8_t *srow = bgra + sy * srcPitch;
        uint8_t *drow = &outRGB[static_cast<size_t>(y) * outW * 3U];
        for (int x = 0; x < outW; ++x) {
            int sx = x * srcW / outW;
            const uint8_t *p = srow + sx * 4;
            drow[x * 3 + 0] = p[2];
            drow[x * 3 + 1] = p[1];
            drow[x * 3 + 2] = p[0];
        }
    }
}

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

static inline uint64_t unix_time_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

static ComPtr<IDXGIOutput> select_output_by_index(int index, int &out_count) {
    out_count = 0;
    ComPtr<IDXGIFactory1> factory;
    hrx(CreateDXGIFactory1(
            __uuidof(IDXGIFactory1),
            reinterpret_cast<void **>(factory.ReleaseAndGetAddressOf())),
        "CreateDXGIFactory1");
    for (UINT a = 0;; ++a) {
        ComPtr<IDXGIAdapter> adp;
        if (factory->EnumAdapters(a, &adp) == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        for (UINT o = 0;; ++o) {
            ComPtr<IDXGIOutput> out;
            if (adp->EnumOutputs(o, &out) == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            if (index < 0 || out_count == index) {
                return out;
            }
            ++out_count;
        }
    }
    return nullptr;
}

struct StagingSlot {
    ComPtr<ID3D11Texture2D> staging;
};

struct RawFrame {
    std::vector<uint8_t> bgra;
    uint64_t captureTimestampNs = 0;
    uint64_t captureFrameId = 0;
    int width = 0;
    int height = 0;
};

struct ConvertedFrame {
    std::vector<uint8_t> rgb;
    uint64_t captureTimestampNs = 0;
    uint64_t captureFrameId = 0;
};

struct PipelineTelemetry {
    std::atomic<uint64_t> freshFramesAcquired{0};
    std::atomic<uint64_t> convertedFramesCompleted{0};
    std::atomic<uint64_t> repeatedOutputsPublished{0};
    std::atomic<uint64_t> freshFramesSkippedByPublisher{0};
    std::atomic<uint64_t> acquisitionDropCount{0};
    std::atomic<uint64_t> missedPublishDeadlines{0};
    std::atomic<uint64_t> overloadWindows{0};
    std::atomic<uint64_t> overloadStreak{0};
    std::atomic<int> conversionBacklogDepth{0};
    std::atomic<int> maxConversionBacklogDepth{0};
};

struct SharedPipelineState {
    std::mutex mutex;
    std::condition_variable readyAvailable;
    std::deque<RawFrame> rawFrames;
    std::shared_ptr<ConvertedFrame> latestCompletedFrame;
    std::atomic<bool> running{true};
};

static void update_max_atomic(std::atomic<int> &target, int value) {
    int current = target.load();
    while (value > current &&
           !target.compare_exchange_weak(current, value)) {
    }
}

static void copy_texture_to_bgra(ID3D11DeviceContext *ctx,
                                 ID3D11Texture2D *staging, UINT width,
                                 UINT height,
                                 std::vector<uint8_t> &bgraOut) {
    D3D11_MAPPED_SUBRESOURCE map{};
    bgraOut.resize(static_cast<size_t>(width) * static_cast<size_t>(height) *
                   4U);
    hrx(ctx->Map(staging, 0, D3D11_MAP_READ, 0, &map), "Map staging");
    const auto *src = reinterpret_cast<const uint8_t *>(map.pData);
    for (UINT y = 0; y < height; ++y) {
        std::memcpy(&bgraOut[static_cast<size_t>(y) * width * 4U],
                    src + static_cast<size_t>(y) * map.RowPitch,
                    static_cast<size_t>(width) * 4U);
    }
    ctx->Unmap(staging, 0);
}

static json pipeline_stats_json(const PipelineTelemetry &telemetry,
                                bool overloadActive) {
    return json{
        {"fresh_frames_acquired",
         telemetry.freshFramesAcquired.load()},
        {"fresh_frames_converted",
         telemetry.convertedFramesCompleted.load()},
        {"repeated_outputs_published",
         telemetry.repeatedOutputsPublished.load()},
        {"fresh_frames_skipped_by_publisher",
         telemetry.freshFramesSkippedByPublisher.load()},
        {"acquisition_drop_count",
         telemetry.acquisitionDropCount.load()},
        {"missed_publish_deadlines",
         telemetry.missedPublishDeadlines.load()},
        {"conversion_backlog_depth",
         telemetry.conversionBacklogDepth.load()},
        {"max_conversion_backlog_depth",
         telemetry.maxConversionBacklogDepth.load()},
        {"overload_windows", telemetry.overloadWindows.load()},
        {"overload_streak", telemetry.overloadStreak.load()},
        {"overload_active", overloadActive}};
}

static void publish_frame(zmq::socket_t &pubRAW,
                          const std::shared_ptr<ConvertedFrame> &frame,
                          uint64_t publishFrameId, bool isRepeat, int rawW,
                          int rawH, const PipelineTelemetry &telemetry,
                          bool overloadActive) {
    const uint64_t publishTimestampNs = unix_time_ns();
    json envelope = {
        {"v", CHANNEL_ENVELOPE_VERSION},
        {"channel", VISION_CHANNEL},
        {"encoding", "raw_rgb_v1"},
        {"sequence_id", publishFrameId},
        {"message_timestamp_ns", frame->captureTimestampNs},
        {"publish_timestamp_ns", publishTimestampNs},
        {"source", FRAME_SOURCE},
        {"metadata",
         {{"w", rawW},
          {"h", rawH},
          {"channels", 3},
          {"dtype", "uint8"},
          {"frame_id", publishFrameId},
          {"capture_frame_id", frame->captureFrameId},
          {"nominal_fps", 60.0},
          {"capture_timestamp_ns", frame->captureTimestampNs},
          {"is_repeat", isRepeat},
          {"pipeline_stats", pipeline_stats_json(telemetry, overloadActive)}}}};
    std::string envelopeBytes = envelope.dump();
    zmq::message_t topicPayload(FRAMES_CPU_TOPIC, strlen(FRAMES_CPU_TOPIC));
    zmq::message_t envelopePayload(envelopeBytes.data(),
                                   envelopeBytes.size());
    zmq::message_t framePayload(frame->rgb.data(), frame->rgb.size());
    pubRAW.send(topicPayload, zmq::send_flags::sndmore);
    pubRAW.send(envelopePayload, zmq::send_flags::sndmore);
    pubRAW.send(framePayload, zmq::send_flags::none);
}

int main(int argc, char **argv) {
    try {
        ScopedTimerResolution timerResolution(1);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);

        int requestedDisplay = -1;
        for (int i = 1; i < argc; ++i) {
            const char *arg = argv[i];
            if (strcmp(arg, "--display-id") == 0 && i + 1 < argc) {
                requestedDisplay = std::atoi(argv[++i]);
            } else if (strncmp(arg, "--display-id=", 13) == 0) {
                requestedDisplay = std::atoi(arg + 13);
            }
        }

        int rawW = 1920;
        int rawH = 1080;
        if (const char *s = getenv("GTAPILOT_RAW_SIZE")) {
            int w = 0;
            int h = 0;
            if (sscanf_s(s, "%dx%d", &w, &h) == 2 && w > 0 && h > 0) {
                rawW = w;
                rawH = h;
            }
        }
        std::cout << "[DisplayCaptureDX11] RAW size " << rawW << "x" << rawH
                  << "\n";

        int totalOutputs = 0;
        ComPtr<IDXGIOutput> chosenOutput = select_output_by_index(
            requestedDisplay < 0 ? 0 : requestedDisplay, totalOutputs);
        if (!chosenOutput) {
            std::cerr << "[DisplayCaptureDX11] No display outputs found or "
                         "index out of range.\n";
            return 1;
        }

        DXGI_OUTPUT_DESC outputDesc{};
        chosenOutput->GetDesc(&outputDesc);
        std::wstring deviceName(outputDesc.DeviceName);
        std::wcout << L"[DisplayCaptureDX11] Selected display #"
                   << (requestedDisplay < 0 ? 0 : requestedDisplay) << L" ("
                   << deviceName << L")" << std::endl;

        zmq::context_t zctx(1);
        zmq::socket_t pubRAW(zctx, zmq::socket_type::pub);
        pubRAW.set(zmq::sockopt::sndhwm, 8);
        pubRAW.set(zmq::sockopt::linger, 0);
        pubRAW.bind(PUBLISH_ADDRESS);

        const auto period =
            std::chrono::nanoseconds(1'000'000'000ULL / CAP_FPS);
        uint64_t publishFrameId = 1;

        for (;;) {
            try {
                SharedPipelineState state;
                PipelineTelemetry telemetry;
                std::exception_ptr workerError;
                std::mutex workerErrorMutex;
                auto setWorkerError = [&](std::exception_ptr error) {
                    std::lock_guard<std::mutex> errorLock(workerErrorMutex);
                    if (!workerError) {
                        workerError = error;
                    }
                    state.running.store(false);
                    state.readyAvailable.notify_all();
                };

                std::thread acquisitionThread([&]() {
                    try {
                        int threadOutputCount = 0;
                        ComPtr<IDXGIOutput> threadOutput =
                            select_output_by_index(
                                requestedDisplay < 0 ? 0 : requestedDisplay,
                                threadOutputCount);
                        if (!threadOutput) {
                            throw std::runtime_error(
                                "Failed to reacquire selected display output");
                        }

                        DupCtx cap;
                        cap.init_with_output(threadOutput);
                        auto ctx = cap.ctx;
                        const UINT width = cap.W;
                        const UINT height = cap.H;

                        D3D11_TEXTURE2D_DESC stagingDesc{};
                        stagingDesc.Width = width;
                        stagingDesc.Height = height;
                        stagingDesc.MipLevels = 1;
                        stagingDesc.ArraySize = 1;
                        stagingDesc.Format = CAP_FMT;
                        stagingDesc.SampleDesc.Count = 1;
                        stagingDesc.Usage = D3D11_USAGE_STAGING;
                        stagingDesc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;

                        std::vector<StagingSlot> stagingSlots(
                            STAGING_RING_SIZE);
                        for (int slotIndex = 0; slotIndex < STAGING_RING_SIZE;
                             ++slotIndex) {
                            hrx(cap.dev->CreateTexture2D(
                                    &stagingDesc, nullptr,
                                    &stagingSlots[slotIndex].staging),
                                "Create staging texture");
                        }

                        uint64_t captureFrameId = 0;
                        int stagingSlotIndex = 0;
                        std::vector<uint8_t> bgra;
                        while (state.running.load()) {
                            ComPtr<ID3D11Texture2D> src = cap.acquire();
                            const bool acquired = (src != nullptr);
                            ScopedReleaseFrame releaseFrame(cap.dup.Get(),
                                                            acquired);
                            if (!acquired) {
                                continue;
                            }

                            const uint64_t captureTimestampNs = unix_time_ns();
                            ctx->CopyResource(
                                stagingSlots[stagingSlotIndex].staging.Get(),
                                src.Get());
                            copy_texture_to_bgra(
                                ctx.Get(),
                                stagingSlots[stagingSlotIndex].staging.Get(),
                                width, height, bgra);
                            RawFrame rawFrame;
                            rawFrame.bgra = std::move(bgra);
                            rawFrame.captureTimestampNs = captureTimestampNs;
                            rawFrame.captureFrameId = ++captureFrameId;
                            rawFrame.width = static_cast<int>(width);
                            rawFrame.height = static_cast<int>(height);
                            telemetry.freshFramesAcquired.store(captureFrameId);

                            {
                                std::lock_guard<std::mutex> lock(state.mutex);
                                if (state.rawFrames.size() >=
                                    STAGING_RING_SIZE) {
                                    state.rawFrames.pop_front();
                                    telemetry.acquisitionDropCount.fetch_add(1);
                                }
                                state.rawFrames.push_back(std::move(rawFrame));
                                const int backlogDepth =
                                    static_cast<int>(state.rawFrames.size());
                                telemetry.conversionBacklogDepth.store(
                                    backlogDepth);
                                update_max_atomic(
                                    telemetry.maxConversionBacklogDepth,
                                    backlogDepth);
                            }
                            state.readyAvailable.notify_one();
                            stagingSlotIndex =
                                (stagingSlotIndex + 1) % STAGING_RING_SIZE;
                        }
                    } catch (...) {
                        setWorkerError(std::current_exception());
                    }
                });

                std::thread conversionThread([&]() {
                    try {
                        std::vector<uint8_t> bgra;
                        while (state.running.load()) {
                            RawFrame rawFrame;
                            {
                                std::unique_lock<std::mutex> lock(state.mutex);
                                state.readyAvailable.wait(lock, [&]() {
                                    return !state.running.load() ||
                                           !state.rawFrames.empty();
                                });
                                if (!state.running.load() &&
                                    state.rawFrames.empty()) {
                                    break;
                                }
                                rawFrame = std::move(state.rawFrames.front());
                                state.rawFrames.pop_front();
                                telemetry.conversionBacklogDepth.store(
                                    static_cast<int>(
                                        state.rawFrames.size()));
                            }

                            auto convertedFrame =
                                std::make_shared<ConvertedFrame>();
                            convertedFrame->captureTimestampNs =
                                rawFrame.captureTimestampNs;
                            convertedFrame->captureFrameId =
                                rawFrame.captureFrameId;
                            convertBGRA_to_RGB_resized(
                                rawFrame.bgra.data(), rawFrame.width,
                                rawFrame.height, rawFrame.width * 4, rawW, rawH,
                                convertedFrame->rgb);
                            telemetry.convertedFramesCompleted.fetch_add(1);

                            {
                                std::lock_guard<std::mutex> lock(state.mutex);
                                state.latestCompletedFrame = convertedFrame;
                            }
                        }
                    } catch (...) {
                        setWorkerError(std::current_exception());
                    }
                });

                std::shared_ptr<ConvertedFrame> lastPublishedFrame;
                auto nextDeadline = std::chrono::steady_clock::now();
                std::exception_ptr loopError;

                try {
                    while (state.running.load()) {
                        std::this_thread::sleep_until(nextDeadline);
                        const auto publishStart =
                            std::chrono::steady_clock::now();
                        const auto deadlineLagNs =
                            std::chrono::duration_cast<
                                std::chrono::nanoseconds>(publishStart -
                                                          nextDeadline)
                                .count();
                        const bool missedDeadline =
                            deadlineLagNs >
                            static_cast<int64_t>(
                                MISSED_DEADLINE_THRESHOLD_NS);
                        if (missedDeadline) {
                            telemetry.missedPublishDeadlines.fetch_add(1);
                        }

                        std::shared_ptr<ConvertedFrame> newestFrame;
                        {
                            std::lock_guard<std::mutex> lock(state.mutex);
                            newestFrame = state.latestCompletedFrame;
                        }

                        const bool overloaded =
                            missedDeadline ||
                            telemetry.conversionBacklogDepth.load() > 1;
                        uint64_t overloadStreak = 0;
                        if (overloaded) {
                            overloadStreak =
                                telemetry.overloadStreak.fetch_add(1) + 1;
                            if (overloadStreak == 1) {
                                telemetry.overloadWindows.fetch_add(1);
                            }
                        } else {
                            telemetry.overloadStreak.store(0);
                        }
                        const bool overloadActive =
                            overloaded && overloadStreak > 0;

                        std::shared_ptr<ConvertedFrame> frameToPublish;
                        bool isRepeat = false;
                        if (newestFrame &&
                            (!lastPublishedFrame ||
                             newestFrame->captureFrameId !=
                                 lastPublishedFrame->captureFrameId)) {
                            if (lastPublishedFrame &&
                                newestFrame->captureFrameId >
                                    lastPublishedFrame->captureFrameId + 1) {
                                telemetry.freshFramesSkippedByPublisher
                                    .fetch_add(
                                        newestFrame->captureFrameId -
                                        lastPublishedFrame->captureFrameId -
                                        1);
                            }
                            frameToPublish = newestFrame;
                            lastPublishedFrame = newestFrame;
                        } else if (lastPublishedFrame) {
                            frameToPublish = lastPublishedFrame;
                            isRepeat = true;
                            telemetry.repeatedOutputsPublished.fetch_add(1);
                        }

                        if (frameToPublish) {
                            publish_frame(pubRAW, frameToPublish,
                                          publishFrameId, isRepeat, rawW,
                                          rawH, telemetry, overloadActive);
                            ++publishFrameId;
                        }

                        nextDeadline += period;
                        const auto now = std::chrono::steady_clock::now();
                        if (now - nextDeadline > period) {
                            nextDeadline = now + period;
                        }

                        std::lock_guard<std::mutex> errorLock(
                            workerErrorMutex);
                        if (workerError) {
                            loopError = workerError;
                            break;
                        }
                    }
                } catch (...) {
                    loopError = std::current_exception();
                }

                state.running.store(false);
                state.readyAvailable.notify_all();
                if (acquisitionThread.joinable()) {
                    acquisitionThread.join();
                }
                if (conversionThread.joinable()) {
                    conversionThread.join();
                }
                if (loopError) {
                    std::rethrow_exception(loopError);
                }
            } catch (const std::runtime_error &error) {
                const std::string what = error.what();
                if (what == "DXGI_ERROR_ACCESS_LOST" ||
                    what == "DXGI_ERROR_DEVICE_REMOVED") {
                    std::cerr << "[DisplayCaptureDX11] Duplication/device "
                                 "lost, reinitializing...\n";
                    continue;
                }
                throw;
            }
        }
    } catch (const std::exception &error) {
        std::cerr << "[DisplayCaptureDX11] Fatal: " << error.what() << "\n";
        return 1;
    }
}
