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

struct StagingBuffer {
    ComPtr<ID3D11Texture2D> staging;
    UINT width = 0;
    UINT height = 0;
};

static void ensure_staging_texture(ID3D11Device *device, UINT width,
                                   UINT height, StagingBuffer &buffer) {
    if (buffer.staging && buffer.width == width && buffer.height == height) {
        return;
    }

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

    buffer.staging.Reset();
    hrx(device->CreateTexture2D(&desc, nullptr, buffer.staging.GetAddressOf()),
        "Create staging texture");
    buffer.width = width;
    buffer.height = height;
}

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
                          bool overloadActive,
                          const WindowCandidate &targetWindow) {
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
    zmq::message_t framePayload(frame->rgb.data(), frame->rgb.size());
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

    void copy_client_frame_to_bgra(
        WGC::Direct3D11CaptureFrame const &frame, std::vector<uint8_t> &bgraOut,
        int &outWidth, int &outHeight) {
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

        ensure_staging_texture(_device.Get(), desc.Width, desc.Height,
                               _stagingBuffer);
        _context->CopyResource(_stagingBuffer.staging.Get(), frameSurface.get());

        D3D11_MAPPED_SUBRESOURCE mapped{};
        hrx(_context->Map(_stagingBuffer.staging.Get(), 0, D3D11_MAP_READ, 0,
                          &mapped),
            "Map staging");

        const int cropWidth = clientBox.right - clientBox.left;
        const int cropHeight = clientBox.bottom - clientBox.top;
        if (cropWidth <= 0 || cropHeight <= 0) {
            _context->Unmap(_stagingBuffer.staging.Get(), 0);
            throw std::runtime_error(
                "Grand Theft Auto V client area resolved to an empty region.");
        }

        bgraOut.resize(static_cast<size_t>(cropWidth) *
                       static_cast<size_t>(cropHeight) * 4U);
        const auto *src = reinterpret_cast<const uint8_t *>(mapped.pData);
        for (int y = 0; y < cropHeight; ++y) {
            const auto *srcRow =
                src + static_cast<size_t>(clientBox.top + y) * mapped.RowPitch +
                static_cast<size_t>(clientBox.left) * 4U;
            std::memcpy(
                &bgraOut[static_cast<size_t>(y) * static_cast<size_t>(cropWidth) *
                         4U],
                srcRow, static_cast<size_t>(cropWidth) * 4U);
        }
        _context->Unmap(_stagingBuffer.staging.Get(), 0);

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

        outWidth = cropWidth;
        outHeight = cropHeight;
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
        _stagingBuffer.staging.Reset();
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
    StagingBuffer _stagingBuffer;
    HANDLE _frameArrivedEvent = nullptr;
    std::atomic<bool> _active{false};
};

int main(int argc, char **argv) {
    try {
        ScopedTimerResolution timerResolution(1);
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL);

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
                WindowCaptureSession capture(targetMatch.candidate);
                capture.initialize();

                uint64_t captureFrameId = 0;
                std::vector<uint8_t> bgra;
                while (state.running.load()) {
                    if (!capture.target_still_valid()) {
                        throw std::runtime_error(
                            "Grand Theft Auto V window was lost or minimized.");
                    }

                    auto frame = capture.wait_for_latest_frame(CAP_FRAME_TIMEOUT_MS);
                    if (!frame) {
                        continue;
                    }

                    RawFrame rawFrame;
                    capture.copy_client_frame_to_bgra(frame, bgra, rawFrame.width,
                                                      rawFrame.height);
                    rawFrame.bgra = std::move(bgra);
                    rawFrame.captureTimestampNs = unix_time_ns();
                    rawFrame.captureFrameId = ++captureFrameId;
                    telemetry.freshFramesAcquired.store(captureFrameId);

                    {
                        std::lock_guard<std::mutex> lock(state.mutex);
                        if (state.rawFrames.size() >= STAGING_RING_SIZE) {
                            state.rawFrames.pop_front();
                            telemetry.acquisitionDropCount.fetch_add(1);
                        }
                        state.rawFrames.push_back(std::move(rawFrame));
                        const int backlogDepth =
                            static_cast<int>(state.rawFrames.size());
                        telemetry.conversionBacklogDepth.store(backlogDepth);
                        update_max_atomic(
                            telemetry.maxConversionBacklogDepth,
                            backlogDepth);
                    }
                    state.readyAvailable.notify_one();
                }
            } catch (...) {
                setWorkerError(std::current_exception());
            }
        });

        std::thread conversionThread([&]() {
            try {
                while (state.running.load()) {
                    RawFrame rawFrame;
                    {
                        std::unique_lock<std::mutex> lock(state.mutex);
                        state.readyAvailable.wait(lock, [&]() {
                            return !state.running.load() ||
                                   !state.rawFrames.empty();
                        });
                        if (!state.running.load() && state.rawFrames.empty()) {
                            break;
                        }
                        rawFrame = std::move(state.rawFrames.front());
                        state.rawFrames.pop_front();
                        telemetry.conversionBacklogDepth.store(
                            static_cast<int>(state.rawFrames.size()));
                    }

                    auto convertedFrame = std::make_shared<ConvertedFrame>();
                    convertedFrame->captureTimestampNs =
                        rawFrame.captureTimestampNs;
                    convertedFrame->captureFrameId = rawFrame.captureFrameId;
                    convertBGRA_to_RGB_resized(
                        rawFrame.bgra.data(), rawFrame.width, rawFrame.height,
                        rawFrame.width * 4, rawW, rawH, convertedFrame->rgb);
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
                const auto publishStart = std::chrono::steady_clock::now();
                const auto deadlineLagNs =
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        publishStart - nextDeadline)
                        .count();
                const bool missedDeadline =
                    deadlineLagNs >
                    static_cast<int64_t>(MISSED_DEADLINE_THRESHOLD_NS);
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

                std::shared_ptr<ConvertedFrame> newestFrame;
                {
                    std::lock_guard<std::mutex> lock(state.mutex);
                    newestFrame = state.latestCompletedFrame;
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

                std::shared_ptr<ConvertedFrame> frameToPublish;
                bool isRepeat = false;
                if (newestFrame &&
                    (!lastPublishedFrame ||
                     newestFrame->captureFrameId !=
                         lastPublishedFrame->captureFrameId)) {
                    if (lastPublishedFrame &&
                        newestFrame->captureFrameId >
                            lastPublishedFrame->captureFrameId + 1) {
                        telemetry.freshFramesSkippedByPublisher.fetch_add(
                            newestFrame->captureFrameId -
                            lastPublishedFrame->captureFrameId - 1);
                    }
                    frameToPublish = newestFrame;
                    lastPublishedFrame = newestFrame;
                } else if (lastPublishedFrame) {
                    frameToPublish = lastPublishedFrame;
                    isRepeat = true;
                    telemetry.repeatedOutputsPublished.fetch_add(1);
                }

                if (frameToPublish) {
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
    } catch (const std::exception &error) {
        std::cerr << "[DisplayCaptureDX11] Fatal: " << error.what() << "\n";
        return 1;
    }
}
