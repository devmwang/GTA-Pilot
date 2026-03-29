// Atlas capture source: Windows Graphics Capture of the GTA window ->
// CPU conversion -> fixed-deadline 60 Hz publisher. Publishes raw RGB frames
// on tcp://127.0.0.1:55550 topic "frames" using the shared generic channel
// envelope contract.

#include <d3d11.h>
#include <dwmapi.h>
#include <mmsystem.h>
#include <windows.h>
#include <windows.graphics.capture.interop.h>
#include <windows.graphics.directx.direct3d11.interop.h>
#include <wrl/client.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cwctype>
#include <deque>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <nlohmann/json.hpp>
#include <winrt/Windows.Foundation.Metadata.h>
#include <winrt/Windows.Graphics.Capture.h>
#include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
#include <winrt/base.h>
#include <zmq.hpp>

#pragma comment(lib, "dwmapi.lib")
#pragma comment(lib, "winmm.lib")

static constexpr const char *PUBLISH_ADDRESS = "tcp://127.0.0.1:55550";
static constexpr const char *FRAMES_CPU_TOPIC = "frames";
static constexpr const char *VISION_CHANNEL = "vision.frames";
static constexpr const char *FRAME_SOURCE = "display_capture_dx11";
static constexpr const char *CAPTURE_MODE = "window_graphics_capture";
static constexpr const wchar_t *TARGET_WINDOW_TITLE = L"grand theft auto v";
static constexpr int CHANNEL_ENVELOPE_VERSION = 1;

using Microsoft::WRL::ComPtr;
using json = nlohmann::json;
namespace WGC = winrt::Windows::Graphics::Capture;
namespace WGD = winrt::Windows::Graphics::DirectX::Direct3D11;
namespace WFM = winrt::Windows::Foundation::Metadata;

static constexpr int CAP_FPS = 60;
static constexpr int CAP_FRAME_TIMEOUT_MS = 17;
static constexpr int FRAME_POOL_BUFFER_COUNT = 2;
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

static std::string utf8_from_wide(const std::wstring &value) {
    if (value.empty()) {
        return "";
    }
    const int required = WideCharToMultiByte(
        CP_UTF8, 0, value.c_str(), static_cast<int>(value.size()), nullptr, 0,
        nullptr, nullptr);
    if (required <= 0) {
        return "";
    }
    std::string utf8(static_cast<size_t>(required), '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.c_str(),
                        static_cast<int>(value.size()), utf8.data(), required,
                        nullptr, nullptr);
    return utf8;
}

static std::string utf8_from_hwnd(HWND hwnd) {
    char buffer[32]{};
    sprintf_s(buffer, "0x%p", hwnd);
    return std::string(buffer);
}

static std::wstring lower_wide(std::wstring value) {
    std::transform(value.begin(), value.end(), value.begin(),
                   [](wchar_t c) { return std::towlower(c); });
    return value;
}

static bool contains_case_insensitive(const std::wstring &haystack,
                                      const std::wstring &needle) {
    if (needle.empty()) {
        return true;
    }
    return lower_wide(haystack).find(needle) != std::wstring::npos;
}

static inline uint64_t unix_time_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
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

struct WindowCandidate {
    HWND hwnd = nullptr;
    std::wstring title;
    std::wstring executable;
};

struct WindowMatch {
    WindowCandidate candidate;
    std::vector<WindowCandidate> candidates;
};

static std::wstring read_window_text(HWND hwnd) {
    const int length = GetWindowTextLengthW(hwnd);
    if (length <= 0) {
        return L"";
    }
    std::wstring buffer(static_cast<size_t>(length) + 1, L'\0');
    const int copied = GetWindowTextW(hwnd, buffer.data(), length + 1);
    if (copied <= 0) {
        return L"";
    }
    buffer.resize(static_cast<size_t>(copied));
    return buffer;
}

static std::wstring read_window_executable(HWND hwnd) {
    DWORD processId = 0;
    GetWindowThreadProcessId(hwnd, &processId);
    if (processId == 0) {
        return L"";
    }

    HANDLE process = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, FALSE,
                                 processId);
    if (process == nullptr) {
        return L"";
    }

    std::wstring path(32768, L'\0');
    DWORD size = static_cast<DWORD>(path.size());
    std::wstring executable;
    if (QueryFullProcessImageNameW(process, 0, path.data(), &size)) {
        path.resize(static_cast<size_t>(size));
        const size_t slashPos = path.find_last_of(L"\\/");
        executable =
            slashPos == std::wstring::npos ? path : path.substr(slashPos + 1);
    }
    CloseHandle(process);
    return executable;
}

static bool is_top_level_capture_window(HWND hwnd) {
    if (!IsWindow(hwnd) || !IsWindowVisible(hwnd) || IsIconic(hwnd)) {
        return false;
    }
    if (GetAncestor(hwnd, GA_ROOT) != hwnd) {
        return false;
    }
    if (GetWindow(hwnd, GW_OWNER) != nullptr) {
        return false;
    }

    const LONG_PTR exStyle = GetWindowLongPtrW(hwnd, GWL_EXSTYLE);
    if ((exStyle & WS_EX_TOOLWINDOW) != 0) {
        return false;
    }

    DWORD cloaked = 0;
    if (SUCCEEDED(DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, &cloaked,
                                        sizeof(cloaked))) &&
        cloaked != 0) {
        return false;
    }

    return GetWindowTextLengthW(hwnd) > 0;
}

struct WindowSearchState {
    std::vector<WindowCandidate> candidates;
};

static BOOL CALLBACK enum_windows_for_gta(HWND hwnd, LPARAM lParam) {
    auto &state = *reinterpret_cast<WindowSearchState *>(lParam);
    if (!is_top_level_capture_window(hwnd)) {
        return TRUE;
    }

    const std::wstring title = read_window_text(hwnd);
    if (!contains_case_insensitive(title, TARGET_WINDOW_TITLE)) {
        return TRUE;
    }

    WindowCandidate candidate;
    candidate.hwnd = hwnd;
    candidate.title = title;
    candidate.executable = read_window_executable(hwnd);
    state.candidates.push_back(std::move(candidate));
    return TRUE;
}

static std::string describe_candidate(const WindowCandidate &candidate) {
    return "title=\"" + utf8_from_wide(candidate.title) + "\" exe=\"" +
           utf8_from_wide(candidate.executable) + "\" hwnd=" +
           utf8_from_hwnd(candidate.hwnd);
}

static WindowMatch find_gta_window_or_throw() {
    WindowSearchState state;
    EnumWindows(enum_windows_for_gta, reinterpret_cast<LPARAM>(&state));

    if (state.candidates.empty()) {
        throw std::runtime_error(
            "Grand Theft Auto V window not found. Start GTA in borderless or "
            "windowed mode before launching the runtime.");
    }

    WindowMatch match;
    match.candidates = state.candidates;
    if (state.candidates.size() == 1) {
        match.candidate = state.candidates.front();
        return match;
    }

    const HWND foreground = GetAncestor(GetForegroundWindow(), GA_ROOT);
    WindowCandidate *foregroundCandidate = nullptr;
    int foregroundMatchCount = 0;
    for (auto &candidate : state.candidates) {
        if (candidate.hwnd == foreground) {
            foregroundCandidate = &candidate;
            ++foregroundMatchCount;
        }
    }
    if (foregroundMatchCount == 1 && foregroundCandidate != nullptr) {
        match.candidate = *foregroundCandidate;
        return match;
    }

    std::string message =
        "Ambiguous Grand Theft Auto V window match. Matching windows:";
    for (const auto &candidate : state.candidates) {
        message += "\n  - " + describe_candidate(candidate);
    }
    throw std::runtime_error(message);
}

static bool get_client_box(HWND window, UINT frameWidth, UINT frameHeight,
                           RECT &clientBox) {
    RECT clientRect{};
    RECT windowRect{};
    POINT upperLeft{};

    const bool available =
        !IsIconic(window) && GetClientRect(window, &clientRect) &&
        !IsIconic(window) && clientRect.right > 0 && clientRect.bottom > 0 &&
        DwmGetWindowAttribute(window, DWMWA_EXTENDED_FRAME_BOUNDS, &windowRect,
                              sizeof(windowRect)) == S_OK &&
        ClientToScreen(window, &upperLeft);
    if (!available) {
        return false;
    }

    clientBox.left =
        std::max(0L, static_cast<LONG>(upperLeft.x - windowRect.left));
    clientBox.top =
        std::max(0L, static_cast<LONG>(upperLeft.y - windowRect.top));
    clientBox.right = std::min<LONG>(
        static_cast<LONG>(frameWidth),
        clientBox.left + std::max(1L, clientRect.right - clientRect.left));
    clientBox.bottom = std::min<LONG>(
        static_cast<LONG>(frameHeight),
        clientBox.top + std::max(1L, clientRect.bottom - clientRect.top));
    return clientBox.right > clientBox.left && clientBox.bottom > clientBox.top;
}

template <typename T>
static winrt::com_ptr<T> get_dxgi_interface_from_object(
    winrt::Windows::Foundation::IInspectable const &object) {
    auto access = object.as<
        ::Windows::Graphics::DirectX::Direct3D11::
            IDirect3DDxgiInterfaceAccess>();
    winrt::com_ptr<T> result;
    hrx(access->GetInterface(winrt::guid_of<T>(), result.put_void()),
        "GetInterface");
    return result;
}

static bool is_cursor_toggle_supported() {
    return WFM::ApiInformation::IsPropertyPresent(
        L"Windows.Graphics.Capture.GraphicsCaptureSession",
        L"IsCursorCaptureEnabled");
}

static bool is_border_toggle_supported() {
    return WFM::ApiInformation::IsPropertyPresent(
        L"Windows.Graphics.Capture.GraphicsCaptureSession",
        L"IsBorderRequired");
}

static WGD::IDirect3DDevice create_winrt_device_from_d3d11(
    ID3D11Device *device) {
    ComPtr<IDXGIDevice> dxgiDevice;
    hrx(device->QueryInterface(dxgiDevice.GetAddressOf()),
        "QueryInterface IDXGIDevice");

    winrt::com_ptr<::IInspectable> inspectable;
    hrx(CreateDirect3D11DeviceFromDXGIDevice(dxgiDevice.Get(),
                                             inspectable.put()),
        "CreateDirect3D11DeviceFromDXGIDevice");
    return inspectable.as<WGD::IDirect3DDevice>();
}

struct ReadbackSlot {
    ComPtr<ID3D11Texture2D> staging;
    UINT width = 0;
    UINT height = 0;
    std::vector<uint8_t> bgra;
    uint64_t captureTimestampNs = 0;
    uint64_t captureFrameId = 0;
    uint64_t frameArrivalWaitNs = 0;
    uint64_t gpuReadbackNs = 0;
};

struct ConvertedSlot {
    std::vector<uint8_t> rgb;
    uint64_t captureTimestampNs = 0;
    uint64_t captureFrameId = 0;
    uint64_t frameArrivalWaitNs = 0;
    uint64_t gpuReadbackNs = 0;
    uint64_t cpuConvertNs = 0;
};

static void ensure_readback_slot(ID3D11Device *device, UINT width, UINT height,
                                 ReadbackSlot &slot) {
    if (!slot.staging || slot.width != width || slot.height != height) {
        D3D11_TEXTURE2D_DESC desc{};
        desc.Width = width;
        desc.Height = height;
        desc.MipLevels = 1;
        desc.ArraySize = 1;
        desc.Format = CAP_FMT;
        desc.SampleDesc.Count = 1;
        desc.Usage = D3D11_USAGE_STAGING;
        desc.CPUAccessFlags = D3D11_CPU_ACCESS_READ;
        desc.BindFlags = 0;
        desc.MiscFlags = 0;

        slot.staging.Reset();
        hrx(device->CreateTexture2D(&desc, nullptr,
                                    slot.staging.GetAddressOf()),
            "Create cropped staging texture");
        slot.width = width;
        slot.height = height;
    }
    slot.bgra.resize(static_cast<size_t>(width) * static_cast<size_t>(height) *
                     4U);
}

struct PipelineTelemetry {
    std::atomic<uint64_t> freshFramesAcquired{0};
    std::atomic<uint64_t> convertedFramesCompleted{0};
    std::atomic<uint64_t> repeatedOutputsPublished{0};
    std::atomic<uint64_t> freshFramesSkippedByPublisher{0};
    std::atomic<uint64_t> acquisitionDropCount{0};
    std::atomic<uint64_t> droppedStagingSlots{0};
    std::atomic<uint64_t> droppedConvertedSlots{0};
    std::atomic<uint64_t> missedPublishDeadlines{0};
    std::atomic<uint64_t> overloadWindows{0};
    std::atomic<uint64_t> overloadStreak{0};
    std::atomic<uint64_t> sampleCaptureFrameId{0};
    std::atomic<uint64_t> lastFrameArrivalWaitNs{0};
    std::atomic<uint64_t> maxFrameArrivalWaitNs{0};
    std::atomic<uint64_t> lastGpuReadbackNs{0};
    std::atomic<uint64_t> maxGpuReadbackNs{0};
    std::atomic<uint64_t> lastCpuConvertNs{0};
    std::atomic<uint64_t> maxCpuConvertNs{0};
    std::atomic<uint64_t> lastPublishDeadlineLagNs{0};
    std::atomic<uint64_t> maxPublishDeadlineLagNs{0};
    std::atomic<int> conversionBacklogDepth{0};
    std::atomic<int> maxConversionBacklogDepth{0};
};

struct SharedPipelineState {
    std::mutex mutex;
    std::condition_variable rawReadyAvailable;
    std::array<ReadbackSlot, STAGING_RING_SIZE> rawSlots;
    std::array<ConvertedSlot, STAGING_RING_SIZE> convertedSlots;
    std::deque<int> freeRawSlots;
    std::deque<int> readyRawSlots;
    std::deque<int> freeConvertedSlots;
    std::deque<int> readyConvertedSlots;
    int publishedConvertedSlot = -1;
    std::atomic<bool> running{true};
};

static void initialize_slot_queues(SharedPipelineState &state) {
    for (int i = 0; i < STAGING_RING_SIZE; ++i) {
        state.freeRawSlots.push_back(i);
        state.freeConvertedSlots.push_back(i);
    }
}

static void update_max_atomic(std::atomic<int> &target, int value) {
    int current = target.load();
    while (value > current &&
           !target.compare_exchange_weak(current, value)) {
    }
}

static void update_max_atomic_u64(std::atomic<uint64_t> &target,
                                  uint64_t value) {
    uint64_t current = target.load();
    while (value > current &&
           !target.compare_exchange_weak(current, value)) {
    }
}

static void convertBGRA_to_RGB_resized(const uint8_t *bgra, int srcW, int srcH,
                                       int srcPitch, int outW, int outH,
                                       std::vector<uint8_t> &outRGB) {
    const size_t requiredSize =
        static_cast<size_t>(outW) * static_cast<size_t>(outH) * 3U;
    if (outRGB.size() != requiredSize) {
        outRGB.resize(requiredSize);
    }
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
        {"dropped_staging_slots",
         telemetry.droppedStagingSlots.load()},
        {"dropped_converted_slots",
         telemetry.droppedConvertedSlots.load()},
        {"missed_publish_deadlines",
         telemetry.missedPublishDeadlines.load()},
        {"sample_capture_frame_id",
         telemetry.sampleCaptureFrameId.load()},
        {"frame_arrival_wait_ns",
         telemetry.lastFrameArrivalWaitNs.load()},
        {"max_frame_arrival_wait_ns",
         telemetry.maxFrameArrivalWaitNs.load()},
        {"gpu_readback_ns",
         telemetry.lastGpuReadbackNs.load()},
        {"max_gpu_readback_ns",
         telemetry.maxGpuReadbackNs.load()},
        {"cpu_convert_ns",
         telemetry.lastCpuConvertNs.load()},
        {"max_cpu_convert_ns",
         telemetry.maxCpuConvertNs.load()},
        {"publish_deadline_lag_ns",
         telemetry.lastPublishDeadlineLagNs.load()},
        {"max_publish_deadline_lag_ns",
         telemetry.maxPublishDeadlineLagNs.load()},
        {"conversion_backlog_depth",
         telemetry.conversionBacklogDepth.load()},
        {"max_conversion_backlog_depth",
         telemetry.maxConversionBacklogDepth.load()},
        {"overload_windows", telemetry.overloadWindows.load()},
        {"overload_streak", telemetry.overloadStreak.load()},
        {"overload_active", overloadActive}};
}

static void publish_frame(zmq::socket_t &pubRAW,
                          const ConvertedSlot &frame,
                          uint64_t publishFrameId, bool isRepeat, int rawW,
                          int rawH, const PipelineTelemetry &telemetry,
                          bool overloadActive,
                          const WindowCandidate &targetWindow) {
    const uint64_t publishTimestampNs = unix_time_ns();
    json envelope = {
        {"v", CHANNEL_ENVELOPE_VERSION},
        {"channel", VISION_CHANNEL},
        {"encoding", "raw_rgb_v1"},
        {"sequence_id", publishFrameId},
        {"message_timestamp_ns", frame.captureTimestampNs},
        {"publish_timestamp_ns", publishTimestampNs},
        {"source", FRAME_SOURCE},
        {"metadata",
         {{"w", rawW},
          {"h", rawH},
          {"channels", 3},
          {"dtype", "uint8"},
          {"frame_id", publishFrameId},
          {"capture_frame_id", frame.captureFrameId},
          {"nominal_fps", 60.0},
          {"capture_timestamp_ns", frame.captureTimestampNs},
          {"is_repeat", isRepeat},
          {"capture_mode", CAPTURE_MODE},
          {"target_window_title", utf8_from_wide(targetWindow.title)},
          {"target_window_hwnd",
           static_cast<uint64_t>(
               reinterpret_cast<uintptr_t>(targetWindow.hwnd))},
          {"pipeline_stats", pipeline_stats_json(telemetry, overloadActive)}}}};
    std::string envelopeBytes = envelope.dump();
    zmq::message_t topicPayload(FRAMES_CPU_TOPIC, strlen(FRAMES_CPU_TOPIC));
    zmq::message_t envelopePayload(envelopeBytes.data(),
                                   envelopeBytes.size());
    zmq::message_t framePayload(frame.rgb.data(), frame.rgb.size());
    pubRAW.send(topicPayload, zmq::send_flags::sndmore);
    pubRAW.send(envelopePayload, zmq::send_flags::sndmore);
    pubRAW.send(framePayload, zmq::send_flags::none);
}

class WindowCaptureSession {
  public:
    explicit WindowCaptureSession(WindowCandidate targetWindow)
        : _targetWindow(std::move(targetWindow)) {}

    ~WindowCaptureSession() { stop(); }

    void initialize() {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);

        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        D3D_FEATURE_LEVEL fl{};
        hrx(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr,
                              flags, nullptr, 0, D3D11_SDK_VERSION,
                              _device.GetAddressOf(), &fl,
                              _context.GetAddressOf()),
            "D3D11CreateDevice");
        _winrtDevice = create_winrt_device_from_d3d11(_device.Get());

        _frameArrivedEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
        if (_frameArrivedEvent == nullptr) {
            hrx(HRESULT_FROM_WIN32(GetLastError()), "CreateEventW");
        }

        auto activationFactory =
            winrt::get_activation_factory<WGC::GraphicsCaptureItem>();
        auto interopFactory =
            activationFactory.as<IGraphicsCaptureItemInterop>();

        hrx(interopFactory->CreateForWindow(
                _targetWindow.hwnd,
                winrt::guid_of<
                    ABI::Windows::Graphics::Capture::IGraphicsCaptureItem>(),
                reinterpret_cast<void **>(winrt::put_abi(_item))),
            "CreateForWindow");

        _lastSize = _item.Size();
        _framePool = WGC::Direct3D11CaptureFramePool::CreateFreeThreaded(
            _winrtDevice,
            static_cast<winrt::Windows::Graphics::DirectX::DirectXPixelFormat>(
                CAP_FMT),
            FRAME_POOL_BUFFER_COUNT, _lastSize);
        _session = _framePool.CreateCaptureSession(_item);

        if (is_border_toggle_supported()) {
            WGC::GraphicsCaptureAccess::RequestAccessAsync(
                WGC::GraphicsCaptureAccessKind::Borderless)
                .get();
            _session.IsBorderRequired(false);
        }
        if (is_cursor_toggle_supported()) {
            _session.IsCursorCaptureEnabled(false);
        }

        _closedRevoker = _item.Closed(
            winrt::auto_revoke,
            [this](WGC::GraphicsCaptureItem const &,
                   winrt::Windows::Foundation::IInspectable const &) {
                _active.store(false);
                if (_frameArrivedEvent != nullptr) {
                    SetEvent(_frameArrivedEvent);
                }
            });
        _frameArrivedRevoker = _framePool.FrameArrived(
            winrt::auto_revoke,
            [this](WGC::Direct3D11CaptureFramePool const &,
                   winrt::Windows::Foundation::IInspectable const &) {
                if (_frameArrivedEvent != nullptr) {
                    SetEvent(_frameArrivedEvent);
                }
            });

        _session.StartCapture();
        _active.store(true);

        std::cout << "[DisplayCaptureDX11] Target window "
                  << describe_candidate(_targetWindow) << "\n";
    }

    WGC::Direct3D11CaptureFrame wait_for_latest_frame(int timeoutMs) {
        while (true) {
            if (!_active.load()) {
                throw std::runtime_error(
                    "Grand Theft Auto V window capture session became inactive.");
            }

            const DWORD waitResult =
                WaitForSingleObject(_frameArrivedEvent, timeoutMs);
            if (waitResult == WAIT_TIMEOUT) {
                return nullptr;
            }
            if (waitResult != WAIT_OBJECT_0) {
                hrx(HRESULT_FROM_WIN32(GetLastError()), "WaitForSingleObject");
            }

            ResetEvent(_frameArrivedEvent);
            WGC::Direct3D11CaptureFrame latestFrame{nullptr};
            while (true) {
                WGC::Direct3D11CaptureFrame nextFrame{nullptr};
                try {
                    nextFrame = _framePool.TryGetNextFrame();
                } catch (...) {
                    break;
                }
                if (!nextFrame) {
                    break;
                }
                latestFrame = nextFrame;
            }

            if (latestFrame) {
                return latestFrame;
            }
            if (!_active.load()) {
                throw std::runtime_error(
                    "Grand Theft Auto V window capture session became inactive.");
            }
        }
    }

    void copy_client_frame_to_bgra(WGC::Direct3D11CaptureFrame const &frame,
                                   ReadbackSlot &slot) {
        auto frameSurface =
            get_dxgi_interface_from_object<ID3D11Texture2D>(frame.Surface());

        D3D11_TEXTURE2D_DESC desc{};
        frameSurface->GetDesc(&desc);
        if (desc.Format != CAP_FMT) {
            throw std::runtime_error(
                "Unexpected GTA window capture pixel format.");
        }

        RECT clientBox{};
        if (!get_client_box(_targetWindow.hwnd, desc.Width, desc.Height,
                            clientBox)) {
            throw std::runtime_error(
                "Failed to resolve the Grand Theft Auto V client area.");
        }

        const UINT cropWidth =
            static_cast<UINT>(clientBox.right - clientBox.left);
        const UINT cropHeight =
            static_cast<UINT>(clientBox.bottom - clientBox.top);
        if (cropWidth == 0 || cropHeight == 0) {
            throw std::runtime_error(
                "Grand Theft Auto V client area resolved to an empty region.");
        }

        ensure_readback_slot(_device.Get(), cropWidth, cropHeight, slot);

        D3D11_BOX srcBox{};
        srcBox.left = static_cast<UINT>(clientBox.left);
        srcBox.top = static_cast<UINT>(clientBox.top);
        srcBox.right = static_cast<UINT>(clientBox.right);
        srcBox.bottom = static_cast<UINT>(clientBox.bottom);
        srcBox.front = 0;
        srcBox.back = 1;

        const auto readbackStart = std::chrono::steady_clock::now();
        _context->CopySubresourceRegion(slot.staging.Get(), 0, 0, 0, 0,
                                        frameSurface.get(), 0, &srcBox);

        D3D11_MAPPED_SUBRESOURCE mapped{};
        hrx(_context->Map(slot.staging.Get(), 0, D3D11_MAP_READ, 0, &mapped),
            "Map staging");
        const auto *src = reinterpret_cast<const uint8_t *>(mapped.pData);
        for (UINT y = 0; y < cropHeight; ++y) {
            const auto *srcRow =
                src + static_cast<size_t>(y) * mapped.RowPitch;
            std::memcpy(
                &slot.bgra[static_cast<size_t>(y) *
                           static_cast<size_t>(cropWidth) * 4U],
                srcRow, static_cast<size_t>(cropWidth) * 4U);
        }
        _context->Unmap(slot.staging.Get(), 0);
        slot.gpuReadbackNs =
            static_cast<uint64_t>(std::chrono::duration_cast<
                                      std::chrono::nanoseconds>(
                                      std::chrono::steady_clock::now() -
                                      readbackStart)
                                      .count());

        const auto frameSize = frame.ContentSize();
        if (frameSize.Width != _lastSize.Width ||
            frameSize.Height != _lastSize.Height) {
            _framePool.Recreate(
                _winrtDevice,
                static_cast<winrt::Windows::Graphics::DirectX::DirectXPixelFormat>(
                    CAP_FMT),
                FRAME_POOL_BUFFER_COUNT, frameSize);
            _lastSize = frameSize;
        }
        slot.width = cropWidth;
        slot.height = cropHeight;
    }

    bool target_still_valid() const {
        return IsWindow(_targetWindow.hwnd) &&
               !IsIconic(_targetWindow.hwnd) &&
               IsWindowVisible(_targetWindow.hwnd);
    }

    void stop() {
        _active.store(false);
        if (_frameArrivedRevoker) {
            _frameArrivedRevoker.revoke();
        }
        if (_closedRevoker) {
            _closedRevoker.revoke();
        }
        if (_session) {
            _session.Close();
            _session = nullptr;
        }
        if (_framePool) {
            _framePool.Close();
            _framePool = nullptr;
        }
        _item = nullptr;
        _winrtDevice = nullptr;
        _context.Reset();
        _device.Reset();
        if (_frameArrivedEvent != nullptr) {
            CloseHandle(_frameArrivedEvent);
            _frameArrivedEvent = nullptr;
        }
    }

  private:
    WindowCandidate _targetWindow;
    ComPtr<ID3D11Device> _device;
    ComPtr<ID3D11DeviceContext> _context;
    WGD::IDirect3DDevice _winrtDevice{nullptr};
    WGC::GraphicsCaptureItem _item{nullptr};
    WGC::Direct3D11CaptureFramePool _framePool{nullptr};
    WGC::GraphicsCaptureSession _session{nullptr};
    WGC::GraphicsCaptureItem::Closed_revoker _closedRevoker;
    WGC::Direct3D11CaptureFramePool::FrameArrived_revoker _frameArrivedRevoker;
    winrt::Windows::Graphics::SizeInt32 _lastSize{};
    HANDLE _frameArrivedEvent = nullptr;
    std::atomic<bool> _active{false};
};

int main(int argc, char **argv) {
    try {
        ScopedTimerResolution timerResolution(1);

        for (int i = 1; i < argc; ++i) {
            std::string arg = argv[i];
            if (arg == "--display-id" || arg.rfind("--display-id=", 0) == 0) {
                throw std::runtime_error(
                    "--display-id was removed. Live capture now always targets "
                    "the Grand Theft Auto V window.");
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

        WindowMatch targetMatch = find_gta_window_or_throw();

        zmq::context_t zctx(1);
        zmq::socket_t pubRAW(zctx, zmq::socket_type::pub);
        pubRAW.set(zmq::sockopt::sndhwm, 8);
        pubRAW.set(zmq::sockopt::linger, 0);
        pubRAW.bind(PUBLISH_ADDRESS);

        const auto period =
            std::chrono::nanoseconds(1'000'000'000ULL / CAP_FPS);
        uint64_t publishFrameId = 1;

        SharedPipelineState state;
        initialize_slot_queues(state);
        for (auto &slot : state.rawSlots) {
            slot.bgra.reserve(static_cast<size_t>(rawW) *
                              static_cast<size_t>(rawH) * 4U);
        }
        for (auto &slot : state.convertedSlots) {
            slot.rgb.resize(static_cast<size_t>(rawW) * static_cast<size_t>(rawH) *
                            3U);
        }
        PipelineTelemetry telemetry;
        std::exception_ptr workerError;
        std::mutex workerErrorMutex;
        auto setWorkerError = [&](std::exception_ptr error) {
            std::lock_guard<std::mutex> errorLock(workerErrorMutex);
            if (!workerError) {
                workerError = error;
            }
            state.running.store(false);
            state.rawReadyAvailable.notify_all();
        };

        std::thread acquisitionThread([&]() {
            try {
                WindowCaptureSession capture(targetMatch.candidate);
                capture.initialize();

                uint64_t captureFrameId = 0;
                while (state.running.load()) {
                    if (!capture.target_still_valid()) {
                        throw std::runtime_error(
                            "Grand Theft Auto V window was lost or minimized.");
                    }

                    const auto waitStart = std::chrono::steady_clock::now();
                    auto frame = capture.wait_for_latest_frame(CAP_FRAME_TIMEOUT_MS);
                    const uint64_t frameArrivalWaitNs = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - waitStart)
                            .count());
                    if (!frame) {
                        continue;
                    }

                    int rawSlotIndex = -1;
                    {
                        std::lock_guard<std::mutex> lock(state.mutex);
                        if (!state.freeRawSlots.empty()) {
                            rawSlotIndex = state.freeRawSlots.front();
                            state.freeRawSlots.pop_front();
                        } else if (!state.readyRawSlots.empty()) {
                            rawSlotIndex = state.readyRawSlots.front();
                            state.readyRawSlots.pop_front();
                            telemetry.acquisitionDropCount.fetch_add(1);
                            telemetry.droppedStagingSlots.fetch_add(1);
                        } else {
                            telemetry.acquisitionDropCount.fetch_add(1);
                            telemetry.droppedStagingSlots.fetch_add(1);
                            continue;
                        }
                    }

                    auto &rawSlot = state.rawSlots[rawSlotIndex];
                    capture.copy_client_frame_to_bgra(frame, rawSlot);
                    rawSlot.captureTimestampNs = unix_time_ns();
                    rawSlot.captureFrameId = ++captureFrameId;
                    rawSlot.frameArrivalWaitNs = frameArrivalWaitNs;
                    telemetry.freshFramesAcquired.store(captureFrameId);
                    telemetry.sampleCaptureFrameId.store(captureFrameId);
                    telemetry.lastFrameArrivalWaitNs.store(frameArrivalWaitNs);
                    update_max_atomic_u64(telemetry.maxFrameArrivalWaitNs,
                                          frameArrivalWaitNs);
                    telemetry.lastGpuReadbackNs.store(rawSlot.gpuReadbackNs);
                    update_max_atomic_u64(telemetry.maxGpuReadbackNs,
                                          rawSlot.gpuReadbackNs);

                    {
                        std::lock_guard<std::mutex> lock(state.mutex);
                        state.readyRawSlots.push_back(rawSlotIndex);
                        const int backlogDepth =
                            static_cast<int>(state.readyRawSlots.size());
                        telemetry.conversionBacklogDepth.store(backlogDepth);
                        update_max_atomic(
                            telemetry.maxConversionBacklogDepth,
                            backlogDepth);
                    }
                    state.rawReadyAvailable.notify_one();
                }
            } catch (...) {
                setWorkerError(std::current_exception());
            }
        });

        std::thread conversionThread([&]() {
            try {
                while (state.running.load()) {
                    int rawSlotIndex = -1;
                    {
                        std::unique_lock<std::mutex> lock(state.mutex);
                        state.rawReadyAvailable.wait(lock, [&]() {
                            return !state.running.load() ||
                                   !state.readyRawSlots.empty();
                        });
                        if (!state.running.load() && state.readyRawSlots.empty()) {
                            break;
                        }
                        rawSlotIndex = state.readyRawSlots.front();
                        state.readyRawSlots.pop_front();
                        telemetry.conversionBacklogDepth.store(
                            static_cast<int>(state.readyRawSlots.size()));
                    }

                    int convertedSlotIndex = -1;
                    {
                        std::lock_guard<std::mutex> lock(state.mutex);
                        if (!state.freeConvertedSlots.empty()) {
                            convertedSlotIndex = state.freeConvertedSlots.front();
                            state.freeConvertedSlots.pop_front();
                        } else if (!state.readyConvertedSlots.empty()) {
                            convertedSlotIndex = state.readyConvertedSlots.front();
                            state.readyConvertedSlots.pop_front();
                            telemetry.droppedConvertedSlots.fetch_add(1);
                        }
                    }

                    auto &rawSlot = state.rawSlots[rawSlotIndex];
                    if (convertedSlotIndex < 0) {
                        telemetry.droppedConvertedSlots.fetch_add(1);
                        std::lock_guard<std::mutex> lock(state.mutex);
                        state.freeRawSlots.push_back(rawSlotIndex);
                        continue;
                    }

                    auto &convertedSlot = state.convertedSlots[convertedSlotIndex];
                    const auto convertStart = std::chrono::steady_clock::now();
                    convertBGRA_to_RGB_resized(
                        rawSlot.bgra.data(), static_cast<int>(rawSlot.width),
                        static_cast<int>(rawSlot.height),
                        static_cast<int>(rawSlot.width) * 4, rawW, rawH,
                        convertedSlot.rgb);
                    const uint64_t cpuConvertNs = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - convertStart)
                            .count());
                    convertedSlot.captureTimestampNs = rawSlot.captureTimestampNs;
                    convertedSlot.captureFrameId = rawSlot.captureFrameId;
                    convertedSlot.frameArrivalWaitNs = rawSlot.frameArrivalWaitNs;
                    convertedSlot.gpuReadbackNs = rawSlot.gpuReadbackNs;
                    convertedSlot.cpuConvertNs = cpuConvertNs;
                    telemetry.convertedFramesCompleted.fetch_add(1);
                    telemetry.sampleCaptureFrameId.store(rawSlot.captureFrameId);
                    telemetry.lastCpuConvertNs.store(cpuConvertNs);
                    update_max_atomic_u64(telemetry.maxCpuConvertNs,
                                          cpuConvertNs);

                    {
                        std::lock_guard<std::mutex> lock(state.mutex);
                        state.freeRawSlots.push_back(rawSlotIndex);
                        state.readyConvertedSlots.push_back(convertedSlotIndex);
                    }
                }
            } catch (...) {
                setWorkerError(std::current_exception());
            }
        });

        uint64_t lastPublishedCaptureFrameId = 0;
        auto nextDeadline = std::chrono::steady_clock::now();
        std::exception_ptr loopError;

        try {
            while (state.running.load()) {
                std::this_thread::sleep_until(nextDeadline);
                const auto publishStart = std::chrono::steady_clock::now();
                uint64_t deadlineLagNs = 0;
                if (publishStart > nextDeadline) {
                    deadlineLagNs = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            publishStart - nextDeadline)
                            .count());
                }
                telemetry.lastPublishDeadlineLagNs.store(deadlineLagNs);
                update_max_atomic_u64(telemetry.maxPublishDeadlineLagNs,
                                      deadlineLagNs);
                const bool missedDeadline =
                    deadlineLagNs >
                    MISSED_DEADLINE_THRESHOLD_NS;
                if (missedDeadline) {
                    telemetry.missedPublishDeadlines.fetch_add(1);
                }

                {
                    std::lock_guard<std::mutex> errorLock(workerErrorMutex);
                    if (workerError) {
                        loopError = workerError;
                        break;
                    }
                }

                int frameSlotIndex = -1;
                bool isRepeat = false;
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    if (!state.readyConvertedSlots.empty()) {
                        while (state.readyConvertedSlots.size() > 1) {
                            const int staleSlotIndex =
                                state.readyConvertedSlots.front();
                            state.readyConvertedSlots.pop_front();
                            state.freeConvertedSlots.push_back(staleSlotIndex);
                        }
                        frameSlotIndex = state.readyConvertedSlots.front();
                        state.readyConvertedSlots.pop_front();

                        const int previousPublishedSlot =
                            state.publishedConvertedSlot;
                        state.publishedConvertedSlot = frameSlotIndex;
                        if (previousPublishedSlot != -1 &&
                            previousPublishedSlot != frameSlotIndex) {
                            state.freeConvertedSlots.push_back(
                                previousPublishedSlot);
                        }
                    } else if (state.publishedConvertedSlot != -1) {
                        frameSlotIndex = state.publishedConvertedSlot;
                        isRepeat = true;
                    }
                }

                const bool overloaded =
                    missedDeadline || telemetry.conversionBacklogDepth.load() > 1;
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

                if (frameSlotIndex >= 0) {
                    const auto &frameToPublish =
                        state.convertedSlots[frameSlotIndex];
                    if (!isRepeat) {
                        if (lastPublishedCaptureFrameId != 0 &&
                            frameToPublish.captureFrameId >
                                lastPublishedCaptureFrameId + 1) {
                            telemetry.freshFramesSkippedByPublisher.fetch_add(
                                frameToPublish.captureFrameId -
                                lastPublishedCaptureFrameId - 1);
                        }
                        lastPublishedCaptureFrameId =
                            frameToPublish.captureFrameId;
                    } else {
                        telemetry.repeatedOutputsPublished.fetch_add(1);
                    }

                    publish_frame(pubRAW, frameToPublish, publishFrameId,
                                  isRepeat, rawW, rawH, telemetry,
                                  overloadActive, targetMatch.candidate);
                    ++publishFrameId;
                }

                nextDeadline += period;
                const auto now = std::chrono::steady_clock::now();
                if (now - nextDeadline > period) {
                    nextDeadline = now + period;
                }
            }
        } catch (...) {
            loopError = std::current_exception();
        }

        state.running.store(false);
        state.rawReadyAvailable.notify_all();
        if (acquisitionThread.joinable()) {
            acquisitionThread.join();
        }
        if (conversionThread.joinable()) {
            conversionThread.join();
        }
        if (loopError) {
            std::rethrow_exception(loopError);
        }
    } catch (const std::exception &error) {
        std::cerr << "[DisplayCaptureDX11] Fatal: " << error.what() << "\n";
        return 1;
    }
}
