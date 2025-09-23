# gtapilot/orchestrator/gpu_orchestrator.py
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple

import torch
from gtapilot.ipc.vision_ipc import VisionIpcGpuSubscriber

# Import model modules (each exposes build_runner())
# from . import lane_det as lane_model
# from . import yolop_v2 as yolop_model
# from . import depth_pro as depth_model

repo_root = Path(__file__).parent.parent.parent


def _import_gpu_service() -> Any:
    """Import the native pybind11 module after ensuring bin/ is on sys.path."""
    bin_dir = repo_root / "bin"
    if str(bin_dir) not in sys.path:
        sys.path.insert(0, str(bin_dir))
    import importlib

    return importlib.import_module("GpuService")


def _kick_preprocess(gpu_mod: Any, token: int, runner) -> torch.cuda.Event:
    """
    Launch the fused BGRA->RGB FP16 letterbox preprocess into runner.static_input
    on runner.stream. Returns a CUDA event recorded on that stream to signal
    when preprocess has finished (so we can safely end the frame).
    """
    ev = torch.cuda.Event(enable_timing=False, blocking=False)
    gpu_mod.preprocess_into(
        token,
        int(runner.static_input.data_ptr()),  # device pointer to [1,3,H,W] FP16 NCHW
        runner.inW,
        runner.inH,
        0.0,
        0.0,
        0.0,  # mean (R,G,B)
        1.0,
        1.0,
        1.0,  # std  (R,G,B)
        int(runner.stream.cuda_stream),
    )
    # Record after the kernel so we can wait before end_frame
    with torch.cuda.stream(runner.stream):
        ev.record()
    return ev


def main():
    # Build the three model runners.
    # Each runner provides:
    #   - name: str
    #   - inW, inH: int
    #   - every: int         (run-every-N-frames)
    #   - stream: torch.cuda.Stream
    #   - static_input: torch.Tensor (FP16 [1,3,H,W])
    #   - infer(): None      (launch inference on its stream)
    #   - (optional) capture(): None  (done internally in build_runner)
    runners = [
        # TODO: Uncomment once models are implemented
        # lane_model.build_runner(),
        # yolop_model.build_runner(),
        # depth_model.build_runner(),
    ]

    # Import native module lazily to ensure path is set up
    gpu = _import_gpu_service()

    # GPU-frame subscription
    sub = VisionIpcGpuSubscriber()
    slot_to_bridge: Dict[int, int] = {}

    print("[Orchestrator] Ready; listening on frames_gpu")
    while True:
        meta = sub.receive_frame_metadata()  # slot, handle, w, h, frame_id

        if meta is None:
            continue

        # Lazily open the shared D3D11 texture (one-time per slot id)
        if meta.slot not in slot_to_bridge:
            slot_to_bridge[meta.slot] = gpu.open_shared(meta.handle, meta.w, meta.h)
        slot_id = slot_to_bridge[meta.slot]

        # Frame scope: acquire+map once
        token = gpu.begin_frame(slot_id, int(meta.frame_id))

        scheduled: List[Tuple[object, torch.cuda.Event]] = []
        try:
            # Schedule preprocess for models that should run this frame
            for r in runners:
                if meta.frame_id % r.every == 0:
                    ev = _kick_preprocess(gpu, token, r)
                    scheduled.append((r, ev))

            # If nothing scheduled this frame, don't hold the keyed mutex
            if not scheduled:
                continue  # finally will end_frame()

            # Wait for preprocess to finish before unmapping/releasing
            for _, ev in scheduled:
                ev.synchronize()
        finally:
            # Always release mapped resources even on error
            gpu.end_frame(token)

        # Launch inference (overlaps across model streams)
        for r, _ in scheduled:
            r.infer()  # pyright: ignore[reportAttributeAccessIssue]

        # TODO: Publish outputs to messaging system


if __name__ == "__main__":
    main()
