# GTA Pilot — Project-Specific Assistant Instructions

These instructions tailor Copilot / Chat assistance to the **current codebase** in this repo. Remove or adapt only with a clear reason in a PR.

## Current Scope (Reality Check)

Implemented today:

1. Multi-process coordinator (`gtapilot.coordinator.coordinator`) launching:
    - Display capture (live) `gtapilot.display_capture.display_capture`
    - OR video file override `gtapilot.display_capture.display_override` (via `--video-override` flag to `gtapilot.main`)
    - Visualization consumer `gtapilot.visualization.visualization`
    - Optional blackbox recorder `gtapilot.blackbox.blackbox` (disabled unless `BLACKBOX_ENABLED = True` in `gtapilot.config`)
2. High-throughput frame transport over ZeroMQ PUB/SUB (`gtapilot.ipc.vision_ipc`).
3. Frame pipeline: capture/override -> publish (RGB uint8, NHWC) -> subscribe -> display / record.

Not yet implemented (treat as roadmap, do NOT reference as existing code): depth estimation, object detection / segmentation workers, a separate control messaging IPC, CUDA model loading, Torch tensors, heartbeat messages.

## Python & Environment

Python version (from `pyproject.toml`): **3.14**.
Package name: `gtapilot`.
Package dependencies: `bettercam`, `opencv-python`, `setproctitle`, `pyzmq` (listed as `zmq` in `pyproject` but actually used via `pyzmq`).
Use `uv` only (no raw `pip`). Virtual environment located at `.venv/`.

### Setup

```bash
uv venv .venv
uv pip install -e . || uv pip install -r requirements.txt
```

If `pyzmq` complains about missing wheels on Windows, ensure build tools are present or fall back to the pinned version in `requirements.txt`.

## Running the System

Primary entrypoint (with argument parsing):

```bash
uv run ./gtapilot/main.py                 # live display capture (default display index = 1 inside module call)
uv run ./gtapilot/main.py --video-override path/to/video.mp4
```

Graceful shutdown options:

-   Press ESC in console (Windows-only listener thread) or window ESC / q (visualization window) or Ctrl+C.

### Blackbox Recording

To enable recording tar archive + JSON metadata (see `blackbox-recordings/`): set `BLACKBOX_ENABLED = True` in `gtapilot/config.py` before run. (Future improvement: gate via env var; mention in PR if you add it.)

Artifacts:

-   Frames stored inside `capture_<timestamp>_frames.tar` as BMP (lossless, simple) with parallel `<timestamp>_metadata.json` describing frame indices, resolution, and timestamps.

## Process Architecture

Process spawning logic lives in `gtapilot.coordinator.coordinator.build_processes` returning a list of `PythonProcess` objects (wrapper in `gtapilot.coordinator.process`). Each target module must expose a top-level `main(**kwargs)` function. A shared `multiprocessing.Event` (`shutdown_event`) is injected automatically if the worker signature allows it.

Process naming: The OS process title is set via `setproctitle` inside the generic `launcher` before invoking the worker `main`.

## Vision IPC (ZeroMQ PUB/SUB)

Located in `gtapilot.ipc.vision_ipc`.

-   Publisher (`VisionIPCPublisher`): binds (default) `tcp://127.0.0.1:55550` and sends multipart: `[topic, metadata_pickle, raw_bytes]`.
-   Subscriber (`VisionIPCSubscriber`): background thread receives frames and pushes them into a bounded deque buffer. Client calls `receive_frame(blocking=True)` to pop oldest or `get_latest_frames(count)` for recency.
-   Topic constant: `b"frames"`.
-   Frames: NumPy `uint8`, shape `(H, W, 3)`, RGB.

Assistant guidance when modifying:

1. Maintain backward-compatible defaults (`DEFAULT_VIPC_ZMQ_*`).
2. Keep serialization (pickle metadata + raw bytes) in sync for both ends; if adding compression or new metadata fields, version the metadata dict (e.g., add `"v": 2`).
3. Avoid per-frame reallocation of large buffers if performance work begins (future optimization placeholder).

## Worker / Module Conventions

For any new worker-style module to be launched by the coordinator:

```python
def main(shutdown_event=None, **kwargs):
    # Initialize resources (cameras, subscribers, files)
    while True:
        if shutdown_event is not None and shutdown_event.is_set():
            break
        # ...work loop...
```

Avoid global side effects at import time (keep heavy initialization inside `main`).

## Logging & Diagnostics

Current code prints directly (`print`). When adding features, prefer the standard `logging` module with module-level loggers named after the process (e.g., `logging.getLogger("DisplayCapture")`). Keep prints in legacy modules until refactor PR introduces unified logging.

## Adding Future ML (Roadmap Notes)

If/when Torch-based workers (depth, detection) are introduced:

1. Import Torch only inside the worker module (not coordinator) to prevent CUDA init in parent.
2. Keep frame format conversion singular: subscriber receives RGB uint8 NHWC -> convert once to Torch tensor (float32 or float16) NCHW contiguous.
3. Consider separating a control IPC (heartbeat, metrics) from vision IPC; until then, do not overload frame channel with non-frame data.
4. Document any new message schema alongside code (e.g., `ipc/messages_depth.py`).

These ML notes are intentionally aspirational; do not claim they already exist.

## Code Style & Tooling

-   Black formatting and Ruff linting recommended (not yet configured in repo). If you add them, include `pyproject.toml` sections and a short doc update.
-   Type hints: Add progressively; start with public function signatures.
-   Tests: Introduce `pytest` with small synthetic frames (e.g., 8x8 RGB) to exercise publisher/subscriber and blackbox writer.

Suggested initial test cases (add under `tests/` when created):

1. Publisher/subscriber round-trip shape & dtype.
2. Blackbox recording produces valid tar + JSON with matching frame count.

## Performance Considerations (current state)

-   Frame loop uses a tiny `time.sleep(0.001)` to reduce CPU churn; tune if high CPU usage is observed.
-   Visualization rescales every frame to 1920x1080; if input already matches, skip resize for speed.
-   If backlog occurs (subscriber buffer fills), consider enabling `conflate=True` to keep only the newest frame.

## Safe Refactors / PR Guidelines

When modifying existing functionality:

1. Keep public function names (`main`) unchanged unless you update coordinator references.
2. For IPC changes, maintain backward compatibility or bump a protocol version field.
3. Add a brief "Run Instructions" section in the PR description with `uv` commands and any flags used for testing.
4. Ensure graceful shutdown paths always close sockets (`VisionIPCPublisher.close()` / `VisionIPCSubscriber.close()`) and release cameras or video handles.

## FAQ / Quick Answers

Q: How do I run using a video file instead of the live display?  
A: `uv run ./gtapilot/main.py --video-override path/to/file.mp4`.

Q: I see no blackbox output.  
A: Set `BLACKBOX_ENABLED = True` in `gtapilot/config.py` before launching.

Q: Where do I change the display being captured?  
A: In `build_processes` the display capture is constructed with `{"display": 1}`; adjust there or parameterize.

Q: Frames look slow.  
A: Experiment with `VisionIPCSubscriber(conflate=True)` in visualization to drop backlog and keep latest frame.

## Glossary (Trimmed to Current Needs)

-   **Coordinator**: Parent process that spawns and monitors child processes.
-   **Vision IPC**: ZeroMQ PUB/SUB channel transporting raw RGB frames.
-   **Publisher**: Single producer of frames (capture or video override).
-   **Subscriber**: Any consumer (visualization, blackbox, future ML workers).
-   **Conflate**: ZeroMQ socket option keeping only the most recent message.

## When Unsure

Inspect these modules first: `gtapilot/coordinator/`, `gtapilot/display_capture/`, `gtapilot/ipc/`, `gtapilot/visualization/`, `gtapilot/blackbox/`. Align new contributions with existing patterns and keep changes narrowly scoped.
