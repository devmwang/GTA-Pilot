# AGENTS.md — GTA Pilot Coding Agent Playbook (Current Codebase Aware)

Audience: autonomous / semi-autonomous coding agent operating on this repository.
Goal (today): Maintain and extend a multi-process frame pipeline (display capture OR video file → ZeroMQ vision IPC → visualization (+ optional blackbox recorder)) while preserving stability and correctness. Future ML (depth/detection) is a roadmap item, not yet implemented.

## 1. Repository Facts (Ground Truth Now)

-   Language: Python 3.14 (`pyproject.toml`).
-   Package: `gtapilot` (import path matches directory name; no underscore variant).
-   Existing external deps: `bettercam`, `opencv-python`, `pyzmq` (declared as `zmq` in pyproject), `setproctitle`, `numpy`.
-   No PyTorch / CUDA code yet. Any ML additions must add dependency explicitly & minimally.
-   Virtual env: use `uv`; location `.venv/`.

## 2. Executables / Entry Points

Main user entrypoint (argument parsing): `gtapilot/main.py`.

Run commands:

```bash
uv venv .venv
uv pip install -e . || uv pip install -r requirements.txt

# Live display capture (display index hard-coded as 1 in coordinator build)
uv run ./gtapilot/main.py

# Video override (plays file frames instead of live capture)
uv run ./gtapilot/main.py --video-override path/to/video.mp4
```

Graceful shutdown signals: ESC key (console listener on Windows via `msvcrt`), ESC / 'q' in OpenCV visualization window, or Ctrl+C (SIGINT). All propagate by setting a shared `multiprocessing.Event` (`shutdown_event`).

## 3. Processes & Responsibilities

Coordinator (`gtapilot.coordinator.coordinator.main`):

-   Builds list of `PythonProcess` objects (see `gtapilot.coordinator.process`), starts them, monitors shutdown event, joins, force-terminates stragglers.

Workers launched (current set):

1. `DisplayCapture` (`gtapilot.display_capture.display_capture.main`) — live screen capture via `bettercam`; publishes frames.
2. `DisplayOverride` (mutually exclusive) (`gtapilot.display_capture.display_override.main`) — reads frames from video file path provided by `--video-override`.
3. `Visualization` (`gtapilot.visualization.visualization.main`) — subscribes to frames, overlays FPS, displays via OpenCV window.
4. `Blackbox` (`gtapilot.blackbox.blackbox.main`) — optional; only spawned if `BLACKBOX_ENABLED = True` in `gtapilot.config`.

Future (NOT implemented; treat as roadmap): depth worker, detection worker, separate messaging IPC, protocol-structured control messages.

## 4. Vision IPC (Only IPC Layer Present)

Module: `gtapilot.ipc.vision_ipc`.
Pattern: ZeroMQ PUB/SUB; publisher binds; subscribers connect. Topic: `b"frames"`.
Frame format: NumPy `uint8` RGB, shape `(H, W, 3)` (NHWC). Serialization: pickle metadata dict (`dtype`, `shape`) + raw contiguous bytes (`frame.tobytes()`). No compression. Not zero-copy yet; full copy per publish.

Subscriber buffering: bounded deque (default `buffer_size=10`) + background receive thread. Consumer pops frames with `receive_frame(blocking=True)` or `get_latest_frames(count)`. Optional conflation (`conflate=True`) keeps only most recent frame.

Agent rules for modifying Vision IPC:

1. Maintain current wire contract unless versioning metadata (add key `"v"` if expanded).
2. If adding compression or alt transport, keep old path configurable and default stable.
3. Ensure clean shutdown: stop receive thread, close socket, term context.

## 5. Blackbox Recorder

Writes BMP frames into a TAR archive plus JSON metadata file inside `blackbox-recordings/` using a timestamped prefix `capture_<YYYYmmdd_HHMMSS>`. Activation: toggle `BLACKBOX_ENABLED` in `gtapilot/config.py`. Avoid changing output format silently—if format evolves, include a `schema_version` field in JSON.

## 6. Safe Extension Patterns

Adding a new worker today (e.g., experimental analytics):

1. Create `gtapilot/<new_module>.py` with `def main(shutdown_event=None, **kwargs):`.
2. Minimal loop pattern:
    ```python
    def main(shutdown_event=None):
      # init resources
      while True:
        if shutdown_event and shutdown_event.is_set():
          break
        # work
    ```
3. Register in `build_processes` (preserve ordering; cheap producers first, consumers later is fine here).
4. Keep imports light; delay heavy imports until inside `main` if they may become optional.

Introducing ML (future): add `torch` only to `pyproject` when first ML worker lands; ensure each ML worker lazily loads model after process spawn inside its `main` to avoid parent CUDA init.

## 7. Testing Strategy (To Be Implemented)

Create `tests/` directory. Suggested initial tests:

1. Vision IPC round-trip: publish synthetic `(8, 8, 3)` frame; subscriber receives correct shape & dtype within timeout.
2. Blackbox recording: simulate N frames, force shutdown, assert TAR + JSON exist; verify metadata length matches stored entries.
3. Graceful shutdown: start a minimal publisher & subscriber in processes with a short runtime and ensure processes exit on event set.

Fixtures: Provide helper to spin up a `VisionIPCPublisher` bound to ephemeral port (parameterize port), and an isolated subscriber. Use `pytest` markers to skip tests if Windows-specific constraints arise (e.g., screen capture unavailable in CI).

## 8. Performance Considerations (Current State)

-   Target capture FPS: 20 (`TARGET_FPS = 20`).
-   Avoid unnecessary `time.sleep()` in visualization; currently acceptable. If latency spikes, enable subscriber conflation.
-   Resize always to 1920x1080 in visualization; add a conditional skip when source already matches to save cycles.
-   If backlog / dropped frames appear: increase `buffer_size` or enable `conflate=True`.

Future ML guidance (when added): single NHWC→NCHW conversion, `.contiguous()`, reuse device tensors, potential pinned host buffers for faster H2D copies.

## 9. Refactor & Change Control Guidelines

1. Do not rename existing `main` functions without adjusting coordinator references.
2. Preserve argument names used in `build_processes` (e.g., `video_path`, `display`).
3. Always ensure new long-running loops check `shutdown_event` each iteration.
4. On IPC modifications, document change in both this file and `copilot-instructions.md`.
5. Add minimal inline type hints for new public functions.

## 10. Logging & Diagnostics Roadmap

Current state: `print()` statements. Roadmap PR (small, self-contained): introduce `logging` with process-aware format (e.g., `%(asctime)s %(processName)s %(levelname)s %(message)s`). Avoid coupling logging config into worker import side effects—configure once in coordinator before spawning (stdout separation). For Windows, ensure flush on critical messages before shutdown.

## 11. Commit Message Template

```
area: concise change summary

Why: short rationale
How: key implementation notes
Tests: new/updated tests & coverage focus
Impact: perf, memory, compatibility (mention if any IPC contract changes)
```

## 12. PR Expectations

-   If adding tests: `uv run pytest -q` (once tests directory exists) passes locally.
-   If adding deps: update `pyproject.toml` and justify necessity (prefer optional extras if large).
-   Provide a short manual run snippet demonstrating feature (esp. new worker).
-   Confirm blackbox still functions (or explain intentional changes).

## 13. Common Pitfalls (Current Tech Stack)

-   Forgetting to close / term ZMQ sockets -> hanging process on Windows.
-   Letting subscriber buffer starve the consumer (always pop frames promptly if real-time display is desired).
-   Hard-coding display index incorrectly (currently `1`); if user has single monitor, may need `0`—make configurable in future PR.
-   Assuming Torch exists (it does not yet). Do not import `torch` until added as dependency.

## 14. Roadmap (Explicit – Not Yet Implemented)

| Feature            | Outline                                                           | Notes                                                                 |
| ------------------ | ----------------------------------------------------------------- | --------------------------------------------------------------------- |
| Depth worker       | Torch model consuming frames, producing per-pixel depth           | Add messaging or reuse vision IPC with variant topic (`frames_depth`) |
| Detection worker   | YOLO/segmentation inference; overlay engine                       | Keep result data lightweight (boxes, scores)                          |
| Messaging IPC      | Lightweight control channel (heartbeats, shutdown reasons, stats) | Could be separate ZMQ PUB/SUB or `multiprocessing.Queue`              |
| Structured logging | Logging config + optional JSON                                    | Enables easier telemetry collection                                   |
| Config system      | Dataclass or `pydantic` central config                            | Reduces hard-coded values                                             |

Do not reference roadmap features as if they already exist in user-facing docs or code.

## 15. When Unsure

1. Re-read `copilot-instructions.md` (kept in sync with this file).
2. Inspect modules in order of data flow: `display_capture/` → `ipc/vision_ipc.py` → `visualization/` → `blackbox/`.
3. Prefer additive, reversible changes with tests over speculative rewrites.
