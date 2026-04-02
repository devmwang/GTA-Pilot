// Atlas capture source: Windows Graphics Capture of the GTA window ->
// CPU conversion -> fresh-frame publisher. Publishes each fresh full-resolution
// frame once, plus a decimated preview stream, through shared-memory-backed
// channel descriptors.

#include <d3d11.h>
#include <dxgi1_6.h>
#include <dwmapi.h>
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

static constexpr const char *PUBLISH_ADDRESS = "tcp://127.0.0.1:55550";
static constexpr const char *PREVIEW_PUBLISH_ADDRESS = "tcp://127.0.0.1:55551";
static constexpr const char *FRAMES_CPU_TOPIC = "frames";
static constexpr const char *FRAMES_PREVIEW_TOPIC = "preview";
static constexpr const char *VISION_CHANNEL = "vision.frames";
static constexpr const char *PREVIEW_CHANNEL = "vision.preview";
static constexpr const char *FRAME_SOURCE = "display_capture_dx11";
static constexpr const char *PREVIEW_SOURCE = "display_capture_dx11_preview";
static constexpr const char *CAPTURE_MODE = "window_graphics_capture";
static constexpr const char *FRAME_ENCODING = "shm_rgb_v1";
static constexpr const wchar_t *TARGET_WINDOW_TITLE = L"grand theft auto v";
static constexpr int CHANNEL_ENVELOPE_VERSION = 1;

using Microsoft::WRL::ComPtr;
using json = nlohmann::json;
namespace WGC = winrt::Windows::Graphics::Capture;
namespace WGD = winrt::Windows::Graphics::DirectX::Direct3D11;
namespace WFM = winrt::Windows::Foundation::Metadata;

static constexpr int CAP_FRAME_TIMEOUT_MS = 17;
static constexpr int FRAME_POOL_BUFFER_COUNT = 4;
static constexpr int STAGING_RING_SIZE = 8;
static constexpr int SHARED_FRAME_SLOT_COUNT = 8;
static constexpr int SHARED_PREVIEW_SLOT_COUNT = 4;
static constexpr int PREVIEW_W = 1280;
static constexpr int PREVIEW_H = 720;
static constexpr DXGI_FORMAT CAP_FMT = DXGI_FORMAT_B8G8R8A8_UNORM;
static constexpr double FRAME_NOMINAL_FPS = 60.0;
static constexpr double DEFAULT_PREVIEW_MAX_FPS = 30.0;
static constexpr uint64_t DEFAULT_PIPELINE_TELEMETRY_INTERVAL = 30;

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

static std::string sanitize_channel_name(std::string value) {
    std::replace(value.begin(), value.end(), '.', '_');
    return value;
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

static double parse_positive_env_double(const char *name, double defaultValue,
                                        bool allowZero = false) {
    const char *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return defaultValue;
    }
    char *end = nullptr;
    const double value = std::strtod(raw, &end);
    if (end == raw || *end != '\0') {
        return defaultValue;
    }
    if (allowZero && value == 0.0) {
        return 0.0;
    }
    return value > 0.0 ? value : defaultValue;
}

static uint64_t parse_positive_env_u64(const char *name, uint64_t defaultValue) {
    const char *raw = std::getenv(name);
    if (raw == nullptr || *raw == '\0') {
        return defaultValue;
    }
    char *end = nullptr;
    const unsigned long long value = std::strtoull(raw, &end, 10);
    if (end == raw || *end != '\0' || value == 0ULL) {
        return defaultValue;
    }
    return static_cast<uint64_t>(value);
}

static inline uint64_t unix_time_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

static inline uint64_t steady_time_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

struct WindowCandidate {
    HWND hwnd = nullptr;
    std::wstring title;
    std::wstring executable;
};

struct AdapterSelection {
    ComPtr<IDXGIAdapter1> adapter;
    DXGI_ADAPTER_DESC1 desc{};
    bool matchedWindowMonitor = false;
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

static WindowCandidate find_gta_window_or_throw() {
    WindowSearchState state;
    EnumWindows(enum_windows_for_gta, reinterpret_cast<LPARAM>(&state));

    if (state.candidates.empty()) {
        throw std::runtime_error(
            "Grand Theft Auto V window not found. Start GTA in borderless or "
            "windowed mode before launching the runtime.");
    }

    if (state.candidates.size() == 1) {
        return state.candidates.front();
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
        return *foregroundCandidate;
    }

    std::string message =
        "Ambiguous Grand Theft Auto V window match. Matching windows:";
    for (const auto &candidate : state.candidates) {
        message += "\n  - " + describe_candidate(candidate);
    }
    throw std::runtime_error(message);
}

static std::string adapter_description_utf8(const DXGI_ADAPTER_DESC1 &desc) {
    return utf8_from_wide(desc.Description);
}

static std::string adapter_luid_string(const LUID &luid) {
    char buffer[64]{};
    sprintf_s(buffer, "%08lx:%08lx", static_cast<unsigned long>(luid.HighPart),
              static_cast<unsigned long>(luid.LowPart));
    return std::string(buffer);
}

static AdapterSelection select_adapter_for_window(HWND hwnd) {
    const HMONITOR targetMonitor =
        MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);
    if (targetMonitor == nullptr) {
        throw std::runtime_error(
            "Failed to resolve a monitor for the Grand Theft Auto V window.");
    }

    ComPtr<IDXGIFactory1> factory;
    hrx(CreateDXGIFactory1(IID_PPV_ARGS(factory.GetAddressOf())),
        "CreateDXGIFactory1");

    AdapterSelection firstHardware{};
    bool haveFirstHardware = false;

    for (UINT adapterIndex = 0;; ++adapterIndex) {
        ComPtr<IDXGIAdapter1> adapter;
        const HRESULT adapterHr =
            factory->EnumAdapters1(adapterIndex, adapter.GetAddressOf());
        if (adapterHr == DXGI_ERROR_NOT_FOUND) {
            break;
        }
        hrx(adapterHr, "EnumAdapters1");

        DXGI_ADAPTER_DESC1 adapterDesc{};
        hrx(adapter->GetDesc1(&adapterDesc), "IDXGIAdapter1::GetDesc1");
        if ((adapterDesc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0) {
            continue;
        }

        if (!haveFirstHardware) {
            firstHardware.adapter = adapter;
            firstHardware.desc = adapterDesc;
            haveFirstHardware = true;
        }

        for (UINT outputIndex = 0;; ++outputIndex) {
            ComPtr<IDXGIOutput> output;
            const HRESULT outputHr =
                adapter->EnumOutputs(outputIndex, output.GetAddressOf());
            if (outputHr == DXGI_ERROR_NOT_FOUND) {
                break;
            }
            hrx(outputHr, "IDXGIAdapter1::EnumOutputs");

            DXGI_OUTPUT_DESC outputDesc{};
            hrx(output->GetDesc(&outputDesc), "IDXGIOutput::GetDesc");
            if (outputDesc.Monitor == targetMonitor) {
                return AdapterSelection{
                    adapter,
                    adapterDesc,
                    true,
                };
            }
        }
    }

    if (haveFirstHardware) {
        std::cerr << "[DisplayCaptureDX11] Warning: failed to match the GTA "
                     "window monitor to a DXGI output, falling back to the "
                     "first hardware adapter.\n";
        return firstHardware;
    }

    throw std::runtime_error("No hardware DXGI adapter was available.");
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
    ComPtr<ID3D11Query> readyQuery;
    UINT width = 0;
    UINT height = 0;
    std::vector<uint8_t> bgra;
    uint64_t captureTimestampNs = 0;
    uint64_t captureFrameId = 0;
    uint64_t copyIssuedSteadyNs = 0;
    uint64_t gpuReadbackNs = 0;
};

struct ConvertedSlot {
    std::vector<uint8_t> rgb;
    std::vector<uint8_t> previewRgb;
    uint64_t captureTimestampNs = 0;
    uint64_t captureFrameId = 0;
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
    if (!slot.readyQuery) {
        D3D11_QUERY_DESC queryDesc{};
        queryDesc.Query = D3D11_QUERY_EVENT;
        queryDesc.MiscFlags = 0;
        hrx(device->CreateQuery(&queryDesc, slot.readyQuery.GetAddressOf()),
            "Create readback ready query");
    }
    slot.bgra.resize(static_cast<size_t>(width) * static_cast<size_t>(height) *
                     4U);
}

struct PipelineTelemetry {
    std::atomic<uint64_t> freshFramesAcquired{0};
    std::atomic<uint64_t> convertedFramesCompleted{0};
    std::atomic<uint64_t> publishedFreshFrames{0};
    std::atomic<uint64_t> previewFramesPublished{0};
    std::atomic<uint64_t> acquisitionDropCount{0};
    std::atomic<uint64_t> droppedStagingSlots{0};
    std::atomic<uint64_t> sampleCaptureFrameId{0};
    std::atomic<uint64_t> lastFrameArrivalWaitNs{0};
    std::atomic<uint64_t> maxFrameArrivalWaitNs{0};
    std::atomic<uint64_t> lastGpuReadbackNs{0};
    std::atomic<uint64_t> maxGpuReadbackNs{0};
    std::atomic<uint64_t> lastCpuConvertNs{0};
    std::atomic<uint64_t> maxCpuConvertNs{0};
    std::atomic<int> pendingReadbackDepth{0};
    std::atomic<int> maxPendingReadbackDepth{0};
    std::atomic<int> conversionBacklogDepth{0};
    std::atomic<int> maxConversionBacklogDepth{0};
};

struct SharedPipelineState {
    std::mutex mutex;
    std::condition_variable rawReadyAvailable;
    std::array<ReadbackSlot, STAGING_RING_SIZE> rawSlots;
    std::deque<int> freeRawSlots;
    std::deque<int> pendingReadbackSlots;
    std::deque<int> readyRawSlots;
    std::atomic<bool> running{true};
};

static void initialize_slot_queues(SharedPipelineState &state) {
    for (int i = 0; i < STAGING_RING_SIZE; ++i) {
        state.freeRawSlots.push_back(i);
    }
}

template <typename T>
static void update_max_atomic(std::atomic<T> &target, T value) {
    T current = target.load();
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
    if (srcW == outW && srcH == outH && srcPitch == srcW * 4) {
        for (int i = 0; i < srcW * srcH; ++i) {
            const uint8_t *p = bgra + static_cast<size_t>(i) * 4U;
            uint8_t *d = outRGB.data() + static_cast<size_t>(i) * 3U;
            d[0] = p[2];
            d[1] = p[1];
            d[2] = p[0];
        }
        return;
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
        {"published_fresh_frames",
         telemetry.publishedFreshFrames.load()},
        {"preview_frames_published",
         telemetry.previewFramesPublished.load()},
        {"acquisition_drop_count",
         telemetry.acquisitionDropCount.load()},
        {"dropped_staging_slots",
         telemetry.droppedStagingSlots.load()},
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
        {"pending_readback_depth",
         telemetry.pendingReadbackDepth.load()},
        {"max_pending_readback_depth",
         telemetry.maxPendingReadbackDepth.load()},
        {"conversion_backlog_depth",
         telemetry.conversionBacklogDepth.load()},
        {"max_conversion_backlog_depth",
         telemetry.maxConversionBacklogDepth.load()},
        {"overload_active", overloadActive}};
}

struct SharedFrameWriteDescriptor {
    std::string shmName;
    size_t slotBytes = 0;
    int slotIndex = 0;
    uint64_t slotGeneration = 0;
    size_t frameBytes = 0;
};

struct CaptureTargetMetadata {
    std::string title;
    uint64_t hwnd = 0;
};

class SharedFrameRingWriter {
  public:
    SharedFrameRingWriter(std::string channelName, int slotCount)
        : _channelName(std::move(channelName)), _slotCount(slotCount) {}

    ~SharedFrameRingWriter() { close(); }

    SharedFrameWriteDescriptor write(const uint8_t *data, size_t frameBytes) {
        if (frameBytes == 0) {
            throw std::runtime_error("Shared frame payload must be non-empty.");
        }
        ensure_capacity(frameBytes);
        if (_view == nullptr) {
            throw std::runtime_error("Shared frame mapping is not available.");
        }
        const int slotIndex = _nextSlotIndex;
        _nextSlotIndex = (_nextSlotIndex + 1) % _slotCount;
        _slotGenerations[slotIndex] += 1;
        const uint64_t slotGeneration = _slotGenerations[slotIndex];
        const size_t offset =
            static_cast<size_t>(slotIndex) * static_cast<size_t>(_slotBytes);
        std::memcpy(_view + offset, data, frameBytes);
        return SharedFrameWriteDescriptor{
            _mappingName,
            _slotBytes,
            slotIndex,
            slotGeneration,
            frameBytes,
        };
    }

    void close() {
        if (_view != nullptr) {
            UnmapViewOfFile(_view);
            _view = nullptr;
        }
        if (_mapping != nullptr) {
            CloseHandle(_mapping);
            _mapping = nullptr;
        }
        _slotBytes = 0;
        _slotGenerations.clear();
        _nextSlotIndex = 0;
        _mappingName.clear();
    }

  private:
    void ensure_capacity(size_t frameBytes) {
        if (_view != nullptr && frameBytes <= _slotBytes) {
            return;
        }
        close();
        _slotBytes = frameBytes;
        _slotGenerations.assign(static_cast<size_t>(_slotCount), 0);
        _mappingName = "gtapilot_" + sanitize_channel_name(_channelName) + "_" +
                       std::to_string(GetCurrentProcessId()) + "_" +
                       std::to_string(unix_time_ns()) + "_" +
                       std::to_string(_slotBytes);
        const uint64_t totalBytes = static_cast<uint64_t>(_slotCount) *
                                    static_cast<uint64_t>(_slotBytes);
        const DWORD totalBytesLow = static_cast<DWORD>(totalBytes & 0xffffffffULL);
        const DWORD totalBytesHigh = static_cast<DWORD>(totalBytes >> 32ULL);
        _mapping = CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                                      totalBytesHigh, totalBytesLow,
                                      _mappingName.c_str());
        if (_mapping == nullptr) {
            hrx(HRESULT_FROM_WIN32(GetLastError()), "CreateFileMappingA");
        }
        _view = static_cast<uint8_t *>(
            MapViewOfFile(_mapping, FILE_MAP_ALL_ACCESS, 0, 0, totalBytes));
        if (_view == nullptr) {
            hrx(HRESULT_FROM_WIN32(GetLastError()), "MapViewOfFile");
        }
    }

    std::string _channelName;
    int _slotCount = 0;
    size_t _slotBytes = 0;
    int _nextSlotIndex = 0;
    HANDLE _mapping = nullptr;
    uint8_t *_view = nullptr;
    std::string _mappingName;
    std::vector<uint64_t> _slotGenerations;
};

static json build_frame_metadata(
    int width, int height, uint64_t frameId, uint64_t captureFrameId,
    uint64_t captureTimestampNs, double nominalFps,
    const CaptureTargetMetadata &targetWindow,
    const SharedFrameWriteDescriptor &descriptor, const PipelineTelemetry *telemetry,
    bool overloadActive, const char *captureSource) {
    json metadata = {
        {"w", width},
        {"h", height},
        {"channels", 3},
        {"dtype", "uint8"},
        {"frame_id", frameId},
        {"capture_frame_id", captureFrameId},
        {"nominal_fps", nominalFps},
        {"capture_timestamp_ns", captureTimestampNs},
        {"is_repeat", false},
        {"capture_mode", CAPTURE_MODE},
        {"target_window_title", targetWindow.title},
        {"target_window_hwnd", targetWindow.hwnd},
        {"shm_name", descriptor.shmName},
        {"slot_bytes", descriptor.slotBytes},
        {"slot_index", descriptor.slotIndex},
        {"slot_generation", descriptor.slotGeneration},
        {"frame_bytes", descriptor.frameBytes},
        {"capture_source", captureSource},
    };
    if (telemetry != nullptr) {
        metadata["pipeline_stats"] = pipeline_stats_json(*telemetry, overloadActive);
    }
    return metadata;
}

static void publish_shared_frame(zmq::socket_t &socket,
                                 SharedFrameRingWriter &writer,
                                 const char *topic,
                                 const char *channel,
                                 const char *sourceName,
                                 const uint8_t *frameData,
                                 size_t frameBytes,
                                 int width,
                                 int height,
                                 uint64_t sequenceId,
                                 uint64_t frameId,
                                 uint64_t captureFrameId,
                                 uint64_t captureTimestampNs,
                                 double nominalFps,
                                 const CaptureTargetMetadata &targetWindow,
                                 const PipelineTelemetry *telemetry,
                                 bool overloadActive,
                                 json extraMetadata = json::object()) {
    const SharedFrameWriteDescriptor descriptor = writer.write(frameData, frameBytes);
    json metadata = build_frame_metadata(width, height, frameId,
                                         captureFrameId, captureTimestampNs,
                                         nominalFps, targetWindow, descriptor,
                                         telemetry, overloadActive, sourceName);
    for (auto it = extraMetadata.begin(); it != extraMetadata.end(); ++it) {
        metadata[it.key()] = it.value();
    }
    const uint64_t publishTimestampNs = unix_time_ns();
    json envelope = {{"v", CHANNEL_ENVELOPE_VERSION},
                     {"channel", channel},
                     {"encoding", FRAME_ENCODING},
                     {"sequence_id", sequenceId},
                     {"message_timestamp_ns", captureTimestampNs},
                     {"publish_timestamp_ns", publishTimestampNs},
                     {"source", sourceName},
                     {"metadata", metadata}};
    std::string envelopeBytes = envelope.dump();
    zmq::message_t topicPayload(topic, strlen(topic));
    zmq::message_t envelopePayload(envelopeBytes.data(), envelopeBytes.size());
    zmq::message_t framePayload(0);
    socket.send(topicPayload, zmq::send_flags::sndmore);
    socket.send(envelopePayload, zmq::send_flags::sndmore);
    socket.send(framePayload, zmq::send_flags::none);
}

static void publish_frame(zmq::socket_t &pubRAW,
                          SharedFrameRingWriter &frameWriter,
                          const ConvertedSlot &frame,
                          uint64_t publishFrameId, int rawW,
                          int rawH, const PipelineTelemetry &telemetry,
                          bool overloadActive,
                          const CaptureTargetMetadata &targetWindow,
                          bool includeTelemetry) {
    publish_shared_frame(pubRAW, frameWriter, FRAMES_CPU_TOPIC, VISION_CHANNEL,
                         FRAME_SOURCE, frame.rgb.data(), frame.rgb.size(), rawW,
                         rawH, publishFrameId, publishFrameId,
                         frame.captureFrameId, frame.captureTimestampNs,
                         FRAME_NOMINAL_FPS, targetWindow,
                         includeTelemetry ? &telemetry : nullptr, overloadActive);
}

static void publish_preview_frame(zmq::socket_t &pubPreview,
                                  SharedFrameRingWriter &previewWriter,
                                  const ConvertedSlot &frame,
                                  uint64_t previewSequenceId,
                                  uint64_t sourceFrameId,
                                  const CaptureTargetMetadata &targetWindow,
                                  double previewNominalFps) {
    publish_shared_frame(
        pubPreview, previewWriter, FRAMES_PREVIEW_TOPIC, PREVIEW_CHANNEL,
        PREVIEW_SOURCE, frame.previewRgb.data(), frame.previewRgb.size(), PREVIEW_W,
        PREVIEW_H, previewSequenceId, sourceFrameId, frame.captureFrameId,
        frame.captureTimestampNs, previewNominalFps, targetWindow, nullptr,
        false, json{{"preview_source", VISION_CHANNEL},
                    {"source_nominal_fps", FRAME_NOMINAL_FPS}});
}

class WindowCaptureSession {
  public:
    explicit WindowCaptureSession(WindowCandidate targetWindow)
        : _targetWindow(std::move(targetWindow)) {}

    ~WindowCaptureSession() { stop(); }

    void initialize() {
        winrt::init_apartment(winrt::apartment_type::multi_threaded);

        const AdapterSelection adapterSelection =
            select_adapter_for_window(_targetWindow.hwnd);

        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT;
#ifdef _DEBUG
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        D3D_FEATURE_LEVEL fl{};
        hrx(D3D11CreateDevice(adapterSelection.adapter.Get(),
                              D3D_DRIVER_TYPE_UNKNOWN, nullptr,
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
        std::cout << "[DisplayCaptureDX11] Capture adapter "
                  << adapter_description_utf8(adapterSelection.desc)
                  << " vendor=0x" << std::hex << adapterSelection.desc.VendorId
                  << " device=0x" << adapterSelection.desc.DeviceId << std::dec
                  << " luid="
                  << adapter_luid_string(adapterSelection.desc.AdapterLuid)
                  << " matched_monitor="
                  << (adapterSelection.matchedWindowMonitor ? "1" : "0") << "\n";
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

    void issue_copy_to_staging(WGC::Direct3D11CaptureFrame const &frame,
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
        slot.width = cropWidth;
        slot.height = cropHeight;

        D3D11_BOX srcBox{};
        srcBox.left = static_cast<UINT>(clientBox.left);
        srcBox.top = static_cast<UINT>(clientBox.top);
        srcBox.right = static_cast<UINT>(clientBox.right);
        srcBox.bottom = static_cast<UINT>(clientBox.bottom);
        srcBox.front = 0;
        srcBox.back = 1;

        _context->CopySubresourceRegion(slot.staging.Get(), 0, 0, 0, 0,
                                        frameSurface.get(), 0, &srcBox);
        _context->End(slot.readyQuery.Get());

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
    }

    bool try_complete_readback(ReadbackSlot &slot) {
        BOOL ready = FALSE;
        const HRESULT queryHr = _context->GetData(
            slot.readyQuery.Get(), &ready, sizeof(ready),
            D3D11_ASYNC_GETDATA_DONOTFLUSH);
        if (queryHr == S_FALSE || !ready) {
            return false;
        }
        hrx(queryHr, "GetData readback ready query");

        D3D11_MAPPED_SUBRESOURCE mapped{};
        hrx(_context->Map(slot.staging.Get(), 0, D3D11_MAP_READ, 0, &mapped),
            "Map staging");
        const auto *src = reinterpret_cast<const uint8_t *>(mapped.pData);
        const size_t rowBytes =
            static_cast<size_t>(slot.width) * 4U;
        if (mapped.RowPitch == rowBytes) {
            std::memcpy(slot.bgra.data(), src,
                        rowBytes * static_cast<size_t>(slot.height));
        } else {
            for (UINT y = 0; y < slot.height; ++y) {
                const auto *srcRow =
                    src + static_cast<size_t>(y) * mapped.RowPitch;
                std::memcpy(
                    slot.bgra.data() +
                        static_cast<size_t>(y) * rowBytes,
                    srcRow, rowBytes);
            }
        }
        _context->Unmap(slot.staging.Get(), 0);
        slot.gpuReadbackNs =
            steady_time_ns() - slot.copyIssuedSteadyNs;
        return true;
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
        const double previewMaxFps = std::min(
            FRAME_NOMINAL_FPS,
            parse_positive_env_double("GTAPILOT_PREVIEW_MAX_FPS",
                                      DEFAULT_PREVIEW_MAX_FPS, true));
        const bool previewEnabled = previewMaxFps > 0.0;
        const uint64_t previewMinIntervalNs =
            previewEnabled
                ? static_cast<uint64_t>(1'000'000'000.0 / previewMaxFps)
                : 0ULL;
        const uint64_t telemetrySampleInterval = parse_positive_env_u64(
            "GTAPILOT_CAPTURE_TELEMETRY_INTERVAL",
            DEFAULT_PIPELINE_TELEMETRY_INTERVAL);
        std::cout << "[DisplayCaptureDX11] RAW size " << rawW << "x" << rawH
                  << " preview_max_fps=" << previewMaxFps
                  << " telemetry_interval=" << telemetrySampleInterval
                  << "\n";

        WindowCandidate targetWindow = find_gta_window_or_throw();
        const CaptureTargetMetadata targetWindowMetadata{
            utf8_from_wide(targetWindow.title),
            static_cast<uint64_t>(reinterpret_cast<uintptr_t>(targetWindow.hwnd)),
        };

        zmq::context_t zctx(1);
        zmq::socket_t pubRAW(zctx, zmq::socket_type::pub);
        pubRAW.set(zmq::sockopt::sndhwm, 8);
        pubRAW.set(zmq::sockopt::linger, 0);
        pubRAW.bind(PUBLISH_ADDRESS);
        SharedFrameRingWriter frameWriter(VISION_CHANNEL, SHARED_FRAME_SLOT_COUNT);
        std::unique_ptr<zmq::socket_t> pubPreview;
        std::unique_ptr<SharedFrameRingWriter> previewWriter;
        if (previewEnabled) {
            pubPreview = std::make_unique<zmq::socket_t>(zctx, zmq::socket_type::pub);
            pubPreview->set(zmq::sockopt::sndhwm, 1);
            pubPreview->set(zmq::sockopt::linger, 0);
            pubPreview->bind(PREVIEW_PUBLISH_ADDRESS);
            previewWriter = std::make_unique<SharedFrameRingWriter>(
                PREVIEW_CHANNEL, SHARED_PREVIEW_SLOT_COUNT);
        }

        SharedPipelineState state;
        initialize_slot_queues(state);
        for (auto &slot : state.rawSlots) {
            slot.bgra.reserve(static_cast<size_t>(rawW) *
                              static_cast<size_t>(rawH) * 4U);
        }
        PipelineTelemetry telemetry;
        auto updateQueueTelemetryLocked = [&]() {
            const int pendingDepth =
                static_cast<int>(state.pendingReadbackSlots.size());
            telemetry.pendingReadbackDepth.store(pendingDepth);
            update_max_atomic(telemetry.maxPendingReadbackDepth, pendingDepth);
            const int readyDepth =
                static_cast<int>(state.readyRawSlots.size());
            telemetry.conversionBacklogDepth.store(readyDepth);
            update_max_atomic(telemetry.maxConversionBacklogDepth, readyDepth);
        };
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
                WindowCaptureSession capture(targetWindow);
                capture.initialize();
                auto retireCompletedReadbacks = [&]() {
                    bool retiredAny = false;
                    while (state.running.load()) {
                        int pendingSlotIndex = -1;
                        {
                            std::lock_guard<std::mutex> lock(state.mutex);
                            if (state.pendingReadbackSlots.empty()) {
                                break;
                            }
                            pendingSlotIndex = state.pendingReadbackSlots.front();
                        }
                        auto &pendingSlot = state.rawSlots[pendingSlotIndex];
                        if (!capture.try_complete_readback(pendingSlot)) {
                            break;
                        }
                        telemetry.lastGpuReadbackNs.store(pendingSlot.gpuReadbackNs);
                        update_max_atomic(telemetry.maxGpuReadbackNs,
                                          pendingSlot.gpuReadbackNs);
                        {
                            std::lock_guard<std::mutex> lock(state.mutex);
                            state.pendingReadbackSlots.pop_front();
                            state.readyRawSlots.push_back(pendingSlotIndex);
                            updateQueueTelemetryLocked();
                        }
                        retiredAny = true;
                    }
                    if (retiredAny) {
                        state.rawReadyAvailable.notify_one();
                    }
                };

                uint64_t captureFrameId = 0;
                while (state.running.load()) {
                    if (!capture.target_still_valid()) {
                        throw std::runtime_error(
                            "Grand Theft Auto V window was lost or minimized.");
                    }

                    retireCompletedReadbacks();

                    const auto waitStart = std::chrono::steady_clock::now();
                    auto frame = capture.wait_for_latest_frame(CAP_FRAME_TIMEOUT_MS);
                    const uint64_t frameArrivalWaitNs = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - waitStart)
                            .count());
                    if (!frame) {
                        retireCompletedReadbacks();
                        continue;
                    }

                    int rawSlotIndex = -1;
                    {
                        std::lock_guard<std::mutex> lock(state.mutex);
                        if (!state.freeRawSlots.empty()) {
                            rawSlotIndex = state.freeRawSlots.front();
                            state.freeRawSlots.pop_front();
                        } else {
                            telemetry.acquisitionDropCount.fetch_add(1);
                            telemetry.droppedStagingSlots.fetch_add(1);
                        }
                    }
                    if (rawSlotIndex < 0) {
                        continue;
                    }

                    auto &rawSlot = state.rawSlots[rawSlotIndex];
                    rawSlot.captureFrameId = ++captureFrameId;
                    rawSlot.captureTimestampNs = unix_time_ns();
                    rawSlot.copyIssuedSteadyNs = steady_time_ns();
                    capture.issue_copy_to_staging(frame, rawSlot);
                    telemetry.freshFramesAcquired.store(captureFrameId);
                    telemetry.sampleCaptureFrameId.store(captureFrameId);
                    telemetry.lastFrameArrivalWaitNs.store(frameArrivalWaitNs);
                    update_max_atomic(telemetry.maxFrameArrivalWaitNs,
                                      frameArrivalWaitNs);

                    {
                        std::lock_guard<std::mutex> lock(state.mutex);
                        state.pendingReadbackSlots.push_back(rawSlotIndex);
                        updateQueueTelemetryLocked();
                    }
                    retireCompletedReadbacks();
                }
            } catch (...) {
                setWorkerError(std::current_exception());
            }
        });

        std::thread conversionThread([&]() {
            try {
                ConvertedSlot convertedFrame;
                convertedFrame.rgb.resize(static_cast<size_t>(rawW) *
                                          static_cast<size_t>(rawH) * 3U);
                if (previewEnabled) {
                    convertedFrame.previewRgb.resize(static_cast<size_t>(PREVIEW_W) *
                                                     static_cast<size_t>(PREVIEW_H) *
                                                     3U);
                }
                uint64_t publishFrameId = 1;
                uint64_t previewSequenceId = 1;
                uint64_t lastPreviewCaptureTimestampNs = 0;
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
                        updateQueueTelemetryLocked();
                    }
                    auto &rawSlot = state.rawSlots[rawSlotIndex];
                    const auto convertStart = std::chrono::steady_clock::now();
                    convertBGRA_to_RGB_resized(
                        rawSlot.bgra.data(), static_cast<int>(rawSlot.width),
                        static_cast<int>(rawSlot.height),
                        static_cast<int>(rawSlot.width) * 4, rawW, rawH,
                        convertedFrame.rgb);
                    if (previewEnabled) {
                        convertBGRA_to_RGB_resized(
                            rawSlot.bgra.data(), static_cast<int>(rawSlot.width),
                            static_cast<int>(rawSlot.height),
                            static_cast<int>(rawSlot.width) * 4, PREVIEW_W,
                            PREVIEW_H, convertedFrame.previewRgb);
                    }
                    const uint64_t cpuConvertNs = static_cast<uint64_t>(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now() - convertStart)
                            .count());
                    convertedFrame.captureTimestampNs = rawSlot.captureTimestampNs;
                    convertedFrame.captureFrameId = rawSlot.captureFrameId;
                    telemetry.convertedFramesCompleted.fetch_add(1);
                    telemetry.sampleCaptureFrameId.store(rawSlot.captureFrameId);
                    telemetry.lastCpuConvertNs.store(cpuConvertNs);
                    update_max_atomic(telemetry.maxCpuConvertNs, cpuConvertNs);

                    const bool overloadActive =
                        telemetry.conversionBacklogDepth.load() > 1;
                    const bool includeTelemetry =
                        telemetrySampleInterval <= 1 ||
                        (convertedFrame.captureFrameId % telemetrySampleInterval) == 0;

                    publish_frame(pubRAW, frameWriter, convertedFrame,
                                  publishFrameId, rawW, rawH, telemetry,
                                  overloadActive, targetWindowMetadata,
                                  includeTelemetry);
                    telemetry.publishedFreshFrames.fetch_add(1);

                    if (
                        previewEnabled && pubPreview && previewWriter &&
                        (lastPreviewCaptureTimestampNs == 0 ||
                         convertedFrame.captureTimestampNs >=
                             lastPreviewCaptureTimestampNs + previewMinIntervalNs)
                    ) {
                        publish_preview_frame(*pubPreview, *previewWriter,
                                              convertedFrame, previewSequenceId,
                                              publishFrameId, targetWindowMetadata,
                                              previewMaxFps);
                        telemetry.previewFramesPublished.fetch_add(1);
                        lastPreviewCaptureTimestampNs =
                            convertedFrame.captureTimestampNs;
                        ++previewSequenceId;
                    }
                    ++publishFrameId;

                    {
                        std::lock_guard<std::mutex> lock(state.mutex);
                        state.freeRawSlots.push_back(rawSlotIndex);
                    }
                }
            } catch (...) {
                setWorkerError(std::current_exception());
            }
        });

        if (acquisitionThread.joinable()) {
            acquisitionThread.join();
        }
        state.running.store(false);
        state.rawReadyAvailable.notify_all();
        if (conversionThread.joinable()) {
            conversionThread.join();
        }

        {
            std::lock_guard<std::mutex> errorLock(workerErrorMutex);
            if (workerError) {
                std::rethrow_exception(workerError);
            }
        }
    } catch (const std::exception &error) {
        std::cerr << "[DisplayCaptureDX11] Fatal: " << error.what() << "\n";
        return 1;
    }
}
