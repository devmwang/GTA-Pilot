// Build as CUDA-enabled Python extension (see CMake below).
// API (pybind11):
//   open_shared(handle:int, w:int, h:int) -> slot_id:int
//   begin_frame(slot_id:int, frame_id:uint64) -> token:int
//   preprocess_into(token:int, out_ptr:uint64, outW:int, outH:int,
//                   meanR:float,meanG:float,meanB:float,
//                   stdR:float,stdG:float,stdB:float, stream_ptr:uint64) ->
//                   None
//   end_frame(token:int) -> None
//   close_slot(slot_id:int) -> None

#include <d3d11.h>
#include <dxgi1_6.h>
#include <windows.h>
#include <wrl/client.h>

#include <cuda.h>
#include <cuda_d3d11_interop.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <pybind11/pybind11.h>

using Microsoft::WRL::ComPtr;
namespace py = pybind11;

#define HRX(hr, msg)                                                           \
    do {                                                                       \
        if (FAILED(hr)) {                                                      \
            std::ostringstream _oss;                                           \
            _oss << msg << " hr=0x" << std::hex << (hr);                       \
            throw std::runtime_error(_oss.str());                              \
        }                                                                      \
    } while (0)
#define CUX(c)                                                                 \
    do {                                                                       \
        auto _e = (c);                                                         \
        if (_e != cudaSuccess) {                                               \
            std::ostringstream _oss;                                           \
            _oss << #c << " -> " << cudaGetErrorString(_e);                    \
            throw std::runtime_error(_oss.str());                              \
        }                                                                      \
    } while (0)

// --- Single D3D11 device shared by all slots ---
static ComPtr<ID3D11Device> gDev;
static ComPtr<ID3D11DeviceContext> gCtx;
static std::once_flag gInitOnce;

static void init_once() {
    std::call_once(gInitOnce, [] {
        UINT flags = D3D11_CREATE_DEVICE_BGRA_SUPPORT |
                     D3D11_CREATE_DEVICE_SINGLETHREADED;
#ifdef _DEBUG
        flags |= D3D11_CREATE_DEVICE_DEBUG;
#endif
        D3D_FEATURE_LEVEL fl{};
        HRX(D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, flags,
                              nullptr, 0, D3D11_SDK_VERSION, &gDev, &fl, &gCtx),
            "D3D11CreateDevice");
        CUX(cudaSetDevice(0));
        CUX(cudaFree(0)); // create context
    });
}

struct Slot {
    ComPtr<ID3D11Texture2D> tex;
    ComPtr<IDXGIKeyedMutex> mtx;
    cudaGraphicsResource *cudaRes = nullptr;
    int w = 0, h = 0;
    bool inited = false;
};

struct FrameToken {
    int slot_id{};
    uint64_t frame_id{};
    cudaArray_t array{};
    cudaTextureObject_t tex{};
    bool active{false};
    int inW{0};
    int inH{0};
};

// Kernel: BGRA8 -> RGB FP16 (NCHW) with letterbox + normalization
__global__ void k_bgra_to_rgb_fp16_letterbox(
    cudaTextureObject_t src, int inW, int inH, half *__restrict__ out, int outW,
    int outH, float scale, float padX, float padY, float meanR, float meanG,
    float meanB, float stdR, float stdG, float stdB) {
    int ox = blockIdx.x * blockDim.x + threadIdx.x;
    int oy = blockIdx.y * blockDim.y + threadIdx.y;
    if (ox >= outW || oy >= outH)
        return;

    float sx = (ox - padX) / scale;
    float sy = (oy - padY) / scale;

    float r = 0.f, g = 0.f, b = 0.f;
    if (sx >= 0.f && sx <= inW - 1 && sy >= 0.f && sy <= inH - 1) {
        float u = (sx + 0.5f) / (float)inW;
        float v = (sy + 0.5f) / (float)inH;
        float4 bgra = tex2D<float4>(src, u, v); // [0,1]
        r = (bgra.z - meanR) / stdR;
        g = (bgra.y - meanG) / stdG;
        b = (bgra.x - meanB) / stdB;
    }
    size_t plane = (size_t)outW * (size_t)outH;
    out[0 * plane + (size_t)oy * outW + ox] = __float2half(r);
    out[1 * plane + (size_t)oy * outW + ox] = __float2half(g);
    out[2 * plane + (size_t)oy * outW + ox] = __float2half(b);
}

static cudaTextureObject_t make_bgra_tex(cudaArray_t arr) {
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = arr;
    cudaTextureDesc td{};
    td.addressMode[0] = td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear;
    td.readMode = cudaReadModeNormalizedFloat;
    td.normalizedCoords = 1;
    cudaTextureObject_t tex = 0;
    CUX(cudaCreateTextureObject(&tex, &res, &td, nullptr));
    return tex;
}

static std::mutex gMu;
static std::unordered_map<int, Slot> gSlots;
static std::unordered_map<int, FrameToken> gTokens;
static std::atomic<int> gNextSlot{1};
static std::atomic<int> gNextTok{1};

static int open_shared(uint64_t handle, int w, int h) {
    init_once();
    std::lock_guard<std::mutex> lk(gMu);
    int id = gNextSlot++;
    Slot s;
    HANDLE os = (HANDLE)handle;
    HRX(gDev->OpenSharedResource(os, __uuidof(ID3D11Texture2D),
                                 (void **)&s.tex),
        "OpenSharedResource");
    HRX(s.tex.As(&s.mtx), "QI IDXGIKeyedMutex");
    // We only read from the D3D11 texture on the CUDA side; register as
    // read-only
    CUX(cudaGraphicsD3D11RegisterResource(&s.cudaRes, s.tex.Get(),
                                          cudaGraphicsRegisterFlagsReadOnly));
    s.w = w;
    s.h = h;
    s.inited = true;
    gSlots.emplace(id, std::move(s));
    return id;
}

static int begin_frame(int slot_id, uint64_t frame_id) {
    init_once();
    std::lock_guard<std::mutex> lk(gMu);
    auto it = gSlots.find(slot_id);
    if (it == gSlots.end())
        throw std::runtime_error("invalid slot_id");
    Slot &s = it->second;

    // Acquire keyed mutex for this frame and map CUDA resource
    // Writer uses key 0; reader must also use key 0.
    HRX(s.mtx->AcquireSync(0, INFINITE), "AcquireSync(reader)");
    CUX(cudaGraphicsMapResources(1, &s.cudaRes, 0));
    cudaArray_t arr{};
    CUX(cudaGraphicsSubResourceGetMappedArray(&arr, s.cudaRes, 0, 0));

    FrameToken ft;
    ft.slot_id = slot_id;
    ft.frame_id = frame_id;
    ft.array = arr;
    ft.tex = make_bgra_tex(arr);
    ft.active = true;
    ft.inW = s.w;
    ft.inH = s.h;

    int tok = gNextTok++;
    gTokens.emplace(tok, ft);
    return tok;
}

static void preprocess_into(int token, uint64_t out_ptr, int outW, int outH,
                            float meanR, float meanG, float meanB, float stdR,
                            float stdG, float stdB, uint64_t stream_ptr) {
    FrameToken ft;
    {
        std::lock_guard<std::mutex> lk(gMu);
        auto it = gTokens.find(token);
        if (it == gTokens.end() || !it->second.active)
            throw std::runtime_error("invalid token");
        ft = it->second; // local copy of handles
    }
    if (out_ptr == 0) {
        throw std::runtime_error("preprocess_into: out_ptr is null");
    }
    if (outW <= 0 || outH <= 0) {
        throw std::runtime_error("preprocess_into: invalid output dimensions");
    }
    half *out = reinterpret_cast<half *>(out_ptr);
    auto st = reinterpret_cast<cudaStream_t>(stream_ptr); // stream 0 allowed

    // Compute letterbox parameters
    int inW = ft.inW;
    int inH = ft.inH;
    float scale = std::min((float)outW / (float)inW, (float)outH / (float)inH);
    float newW = inW * scale;
    float newH = inH * scale;
    float padX = (outW - newW) * 0.5f;
    float padY = (outH - newH) * 0.5f;

    dim3 blk(16, 16);
    dim3 grd((outW + blk.x - 1) / blk.x, (outH + blk.y - 1) / blk.y);
    k_bgra_to_rgb_fp16_letterbox<<<grd, blk, 0, st>>>(
        ft.tex, inW, inH, out, outW, outH, scale, padX, padY, meanR, meanG,
        meanB, stdR, stdG, stdB);
    CUX(cudaGetLastError());
}

static void end_frame(int token) {
    std::lock_guard<std::mutex> lk(gMu);
    auto tit = gTokens.find(token);
    if (tit == gTokens.end() || !tit->second.active)
        return;
    auto ft = tit->second;
    auto sit = gSlots.find(ft.slot_id);
    if (sit == gSlots.end()) {
        gTokens.erase(tit);
        return;
    }
    Slot &s = sit->second;

    CUX(cudaDestroyTextureObject(ft.tex));
    CUX(cudaGraphicsUnmapResources(1, &s.cudaRes, 0));
    // Writer uses key 0; reader must also use key 0.
    HRX(s.mtx->ReleaseSync(0), "ReleaseSync(reader)");

    tit->second.active = false;
    gTokens.erase(tit);
}

static void close_slot(int slot_id) {
    std::lock_guard<std::mutex> lk(gMu);
    // Ensure no active tokens for this slot
    for (auto it = gTokens.begin(); it != gTokens.end();) {
        if (it->second.slot_id == slot_id) {
            // Best-effort cleanup (should not happen if orchestrator ends
            // frames)
            try {
                CUX(cudaDestroyTextureObject(it->second.tex));
                auto sit = gSlots.find(slot_id);
                if (sit != gSlots.end()) {
                    CUX(cudaGraphicsUnmapResources(1, &sit->second.cudaRes, 0));
                    HRX(sit->second.mtx->ReleaseSync(it->second.frame_id),
                        "ReleaseSync(close)");
                }
            } catch (...) {
            }
            it = gTokens.erase(it);
        } else
            ++it;
    }

    auto s = gSlots.find(slot_id);
    if (s == gSlots.end())
        return;
    if (s->second.cudaRes) {
        cudaGraphicsUnregisterResource(s->second.cudaRes);
        s->second.cudaRes = nullptr;
    }
    s->second.mtx.Reset();
    s->second.tex.Reset();
    gSlots.erase(s);
}

PYBIND11_MODULE(GpuService, m) {
    m.doc() =
        "Frame-scoped D3D11->CUDA bridge with letterbox preprocess (CUDA 12.x)";
    m.def("open_shared", &open_shared, "Open D3D11 shared texture");
    m.def("begin_frame", &begin_frame,
          "Begin frame (acquire+map) and return token");
    m.def("preprocess_into", &preprocess_into, py::arg("token"),
          py::arg("out_ptr"), py::arg("outW"), py::arg("outH"),
          py::arg("meanR") = 0.f, py::arg("meanG") = 0.f,
          py::arg("meanB") = 0.f, py::arg("stdR") = 1.f, py::arg("stdG") = 1.f,
          py::arg("stdB") = 1.f, py::arg("stream_ptr"));
    m.def("end_frame", &end_frame, "End frame (unmap+release)");
    m.def("close_slot", &close_slot, "Close slot");
}
