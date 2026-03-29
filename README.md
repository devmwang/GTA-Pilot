# GTA Pilot

This is a project to experiment with different ML and software techniques to create an ADAS/AV system for use in Grand Theft Auto V.

## Inspirations

GTA Pilot's future experiments will be heavily inspired by Tesla's FSD, Mobileye's Supervision/REM, and Comma.ai's OpenPilot.

## Potential Ideas to Explore

-   "Vision"-based perception (FSD, Supervision/REM, OpenPilot)
-   Multi-task "HydraNet" ML architecture (FSD)
-   "Data Engine" for data collection (FSD)
-   Map-based localization and integration with perception/planning (Supervision/REM)
-   "Crowd-sourced" HD map creation (REM)
-   "RSS" driving policy model (Supervision)
-   Asynchronous system architecture (OpenPilot)
-   Fast IPC via ZeroMQ (OpenPilot)

## Runtime & Shutdown System

The current runtime is a small multi-process data pipeline supervised by a
central coordinator. It is intentionally focused on Atlas inference-time data
sources only:

- front RGB frames
- executed actions
- timestamps and source metadata

Processes currently in the live graph:

- settings runtime service
- native GTA window capture or Python video override
- generalized manual input capture
- visualization
- optional blackbox recorder

Supervisor model:

-   Coordinator starts all subprocesses.
-   Any of these conditions triggers a full system shutdown:
    -   ESC key pressed in the coordinator console (Windows).
    -   Ctrl+C / SIGINT / SIGTERM received by the coordinator.
    -   Any child process exits (normal return or crash).
-   Once triggered, the coordinator force-terminates remaining processes (no cooperative polling).

## Current State

Right now this repository is primarily a data-collection runtime for Atlas.
The live system does **not** yet run a driving model, privileged sensor stack,
or ScriptHook-based label extraction pipeline.

What it does today:

- capture front RGB frames at a 60 Hz runtime cadence
- capture current keyboard and Xbox controller inputs as Atlas-format action packets at 60 Hz
- expose mutable runtime settings through a central settings service
- visualize the live stream with action overlays
- optionally record frame/action sessions to disk for later training

What it does **not** do yet:

- run Atlas inference in the live driving loop
- collect privileged GTA state such as pose, actors, lanes, occupancy, or map labels
- infer reverse gear or vehicle state from the game

## Requirements

Current practical assumptions:

- Windows for the native GTA window capture path
- Python `3.13`
- `uv` for environment management
- `ffmpeg` on `PATH` if `BLACKBOX_ENABLED = True`
- GTA V running in borderless or windowed mode with a title containing
  `Grand Theft Auto V`

Install the project:

```bash
uv venv .venv
uv pip install -e .
```

If you only want to exercise the pipeline with a prerecorded clip instead of
live GTA window capture, `--video-override` is the easiest path.

## Build The Native GTA Window Capture Module

The live capture path depends on the native GTA window capture binary:

```text
bin/DisplayCaptureDX11.exe
```

If you need to rebuild it, use the repo build script:

```powershell
.\build-native.ps1
```

That script is the supported path for native builds right now. It:

- initializes the Visual Studio C++ build environment
- configures the CMake builds
- builds the native `DisplayCaptureDX11` target
- updates `compile_commands.json`
- copies the built executable into `bin/DisplayCaptureDX11.exe`

Default behavior:

- configuration: `Release`
- builds both the `ninja-multi` and `vs2022` presets
- publishes the final executable into the runtime `bin/` directory

Useful options:

```powershell
# Debug build
.\build-native.ps1 -Configuration Debug

# Skip the Visual Studio preset and only build the Ninja preset
.\build-native.ps1 -SkipVS
```

Requirements for the script:

- Windows
- CMake on `PATH`
- Ninja on `PATH`
- Visual Studio 2022 Build Tools or Visual Studio with C++ tooling

The root CMake project will fetch `libzmq` and `cppzmq` automatically, and it
uses the vendored `nlohmann/json` headers from `gtapilot/external/json`.

If you need to inspect the raw build outputs, the script writes into:

```text
build/ninja-multi/
build/vs2022/
```

If you only use the video override path, you do not need to build the native
capture module.

## How To Use It

### 1. Live GTA window capture

Run the coordinator:

```bash
uv run ./gtapilot/main.py
```

By default the coordinator starts:

- `SettingsRuntime`
- `DisplayCaptureDX11`
- `ActionCapture`
- `Visualization`
- `Blackbox` only if enabled in config

The live capture path uses the native DX11 executable in
`bin/DisplayCaptureDX11.exe` by default.

The live capture process targets the GTA V window by title using Windows
Graphics Capture and publishes a fixed 60 Hz `vision.frames` stream. If the
runtime cannot find the GTA window at startup, or if the window later becomes
invalid or minimized, the capture worker exits fatally and the coordinator
shuts down the full system.

### 2. Video override

To test the pipeline without GTA running, feed it a local video file:

```bash
uv run ./gtapilot/main.py --video-override path/to/video.mp4
```

This replaces live GTA window capture with OpenCV video decode while keeping the
rest of the runtime the same.

The override path also publishes a fixed 60 Hz `vision.frames` stream. Lower
FPS source files duplicate frames with `is_repeat = true`, and higher FPS
source files are decimated to the 60 Hz output cadence.

### 3. Stop the system

Any of these will stop the full runtime:

- press `Esc` in the coordinator console window on Windows
- press `Ctrl+C`
- close the visualization window with `q` or `Esc`
- let any child process exit or crash

The coordinator then terminates the remaining processes.

## Inputs Captured Today

### Vision stream

The `vision.frames` channel carries raw RGB frames plus metadata. Current frame
metadata includes:

- `frame_id`
- `capture_frame_id`
- `capture_timestamp_ns`
- `nominal_fps`
- `publish_timestamp_ns`
- `source`
- `is_repeat`
- `capture_mode`
- `target_window_title`
- `target_window_hwnd`
- `pipeline_stats` for native live-capture overload telemetry
- `w`
- `h`
- `channels`
- `dtype`

Current capture producers now publish `nominal_fps = 60.0`.
`frame_id` advances for every published 60 Hz output. `capture_frame_id`
advances only when a fresh source frame is captured; repeated 60 Hz outputs keep
the same `capture_frame_id`.

### Action stream

The `input.actions` channel carries Atlas-format action packets with vector
order:

`[steer, throttle, brake, handbrake, reverse, pilot_active]`

Current action capture supports keyboard plus one XInput Xbox controller and is
intended as an inference-time data source only.

The action capture loop polls inputs and publishes packets at 60 Hz.

The current keyboard mapping is:

- `A` / left arrow: steer left
- `D` / right arrow: steer right
- `W` / up arrow: throttle
- `S` / down arrow: brake
- `Space`: handbrake

The current controller mapping is:

- left stick X: steer
- right trigger: throttle
- left trigger: brake
- `RB`: handbrake

Each action packet contains:

- the top-level normalized action vector used by Atlas/blackbox consumers
- `active_device` indicating `none`, `keyboard`, or `xinput_controller`
- per-device normalized actions
- per-device raw input state, including controller analog values

Current action semantics:

- `pilot_active = 0.0` means manual control / human intervention
- `pilot_active = 1.0` is reserved for future policy control
- `reverse` is currently always `0.0` because vehicle-state integration is not implemented yet

## Recording Data With Blackbox

Blackbox is disabled by default. To enable it, edit
`gtapilot/config.py` and set:

```python
BLACKBOX_ENABLED = True
```

Useful related defaults:

```python
BLACKBOX_RECORD_ON_START = False
BLACKBOX_PREROLL_SECONDS = 0.0
BLACKBOX_RECORD_HOTKEY = "F8"
```

Then run the coordinator normally:

```bash
uv run ./gtapilot/main.py
```

Once the runtime is up:

- get into the desired in-car camera/view
- press `F8` to start recording
- press `F8` again to stop and finalize that clip

Blackbox no longer records full sessions by default. It starts idle and only
creates output files when recording is toggled on.

When `blackbox.recording_enabled = false` and `BLACKBOX_PREROLL_SECONDS = 0.0`,
blackbox now enters an inactive ingest mode and does not decode or copy live
vision frames. If preroll is enabled, idle blackbox still buffers bounded
pre-roll data in memory.

The hotkey does not go through a one-off control stream anymore. The input
process flips the runtime setting `blackbox.recording_enabled`, and the blackbox
and visualization processes read the latest value from the shared settings
service.

When multiple monitors are available, the visualization window is moved to a
non-primary monitor by default when possible.

Recordings are written to:

```text
blackbox-recordings/
```

Each session currently produces:

- `capture_<timestamp>_video.mkv`
- `capture_<timestamp>_metadata.json`

Each start/stop cycle produces a separate recording pair. If you toggle
recording on twice in one runtime, you will get two clips.

Blackbox now requires `ffmpeg` on `PATH` when recording is enabled. Frames are
streamed into ffmpeg as raw `bgr24` and encoded as H.264 in an MKV container.
Current runtime producers publish 60 Hz vision streams, so new live and video
override blackbox clips are authored with `video_nominal_fps = 60.0`.
The JSON manifest is still the authoritative source for frame timing and action
alignment. The current manifest schema version is `7`. Recording now uses
append-only temporary frame/action journals during capture and synthesizes the
final `capture_<timestamp>_metadata.json` once when the session stops.

Schema `7` manifests contain:

- session-level metadata
- session video settings
- session integrity status plus structured drop/overflow events
- performance stats for native capture timing, blackbox ingest mode, idle
  vision-decode counters, and writer lag
- transport stats for `vision.frames` and `input.actions`
- writer queue / writer-lag stats
- frame envelope and frame metadata
- per-frame video file name and video frame index
- per-frame `capture_frame_id`
- per-frame `subscriber_received_timestamp_ns`
- per-frame `writer_committed_timestamp_ns`
- per-frame `subscriber_queue_latency_ns`
- frame-aligned action payloads
- frame-aligned action vectors
- the raw action stream seen during capture

Any detected transport gap, subscriber overflow, writer overflow, or native
capture overload marks the session integrity status as `degraded` instead of
silently hiding the issue.

To audit a saved clip:

```bash
uv run python -m gtapilot.blackbox.audit --metadata-path blackbox-recordings/capture_<timestamp>_metadata.json
```

This is the current usable dataset path for Atlas Stage 1A and related
inference-time training work.

## Atlas Temporal Training Contract

Atlas training is now built around a three-tier temporal/context interface
rather than the older flat short-window design.

Student baseline:

- `24 Hz` fast loop
- `32` recent full-token frames
- `64` older compressed-history frames
- `120` sparse summary steps at `6 Hz`
- `128` action-history steps
- `64` dynamic slots
- `16` speculative slots

Teacher target:

- `36 Hz` fast loop
- `48` recent full-token frames
- `96` older compressed-history frames
- `180` sparse summary steps at `6 Hz`
- `192` action-history steps
- `96` dynamic slots
- `24` speculative slots

The canonical Atlas training call is now:

```python
forward_train(
    rgb_recent,
    dt_recent,
    rgb_older,
    dt_older,
    rgb_mid,
    dt_mid,
    actions_hist,
    dt_hist,
    ...,
)
```

Where:

- `rgb_recent` is the dense recent bank
- `rgb_older` is the older dense-rate history that gets compressed into the
  older bank
- `rgb_mid` is the sparse long-horizon summary input
- `actions_hist` and `dt_hist` carry the long action/ego prior

Blackbox `*_video.mkv + *_metadata.json` recordings are the canonical student
training source. Runtime blackbox collection stays at `60 Hz` for both video
and action packets, and the Stage 1 clip loader resamples those recordings onto
the model-time grids:

- student recent / older / action history at `24 Hz`
- teacher recent / older / action history at `36 Hz`
- mid-summary sampling at `6 Hz`

Frame selection uses the latest source frame at or before each desired model
timestamp, and action history is built from the raw blackbox action stream when
it exists. Older recordings without a raw `actions` stream fall back to the
frame-aligned `action_vector` path while still preserving exact `dt`.

Privileged Stage 1B / Stage 1C targets are a separate sibling package:

- `capture_<timestamp>_privileged/`
- built with `python -m gtapilot.atlas.data.build_stage1b_privileged_dataset ...`
- indexed through `AtlasTemporalClipIndex.privileged_dir`

The privileged manifest now carries source-metadata alignment fields
(`source_metadata_file`, source hash, frame ids, and capture timestamps), and
the privileged dataset loader fails fast if those arrays do not match the
source blackbox metadata exactly.

Stage 1B track supervision is now sparse-lag only. The geometry head predicts
the configured `track_lag_indices` set instead of a dense lag volume, and the
privileged dataset keeps those sparse lag targets sparse end to end.

## IPC

The runtime uses two IPC layers:

### Stream channels

The generic ZeroMQ PUB/SUB channel framework carries the live data streams:

- `vision.frames`: raw RGB frames plus metadata
- `input.actions`: action packets in Atlas order

Every stream channel uses the same multipart wire format:

- topic
- envelope JSON
- payload bytes

Current action vector order:

`[steer, throttle, brake, handbrake, reverse, pilot_active]`

`pilot_active=0` currently means manual control / human intervention.

### Runtime settings

Mutable runtime settings live on a separate stateful IPC plane:

- `settings.updates` on port `55553` broadcasts accepted setting changes
- `settings.rpc` on port `55554` serves snapshots and validated writes

This is how late subscribers can always read the latest value immediately.

Current settings exposed at runtime:

- `blackbox.enabled`
- `blackbox.recording_enabled`
- `blackbox.preroll_seconds`
- `blackbox.record_hotkey`

## Blackbox

The blackbox recorder is a Python subprocess that records frame and action data
for Atlas. It is enabled by setting `BLACKBOX_ENABLED = True` in
`gtapilot/config.py`.

Current blackbox outputs:

- an MKV video clip encoded via ffmpeg
- a JSON manifest containing:
  - session video settings
  - frame envelope metadata
  - per-frame metadata
  - per-frame video file name and frame index
  - frame-aligned action payloads and action envelope data
  - frame-aligned action vectors
  - the raw action stream seen during capture

This gives us a usable interim dataset for Atlas training and replay using only
the sources available at inference time. Privileged sensor capture is a later
project.

## Notes And Limitations

- The runtime is currently a capture/recording system, not an end-to-end autonomy stack.
- The native GTA window capture path is the main live capture path; Python video override is mainly for testing.
- The action stream reflects human input intent, not authoritative in-game vehicle state.
- Blackbox pre-roll is in-memory only; if the process dies before recording is toggled on, that buffered data is lost.
- With zero preroll, idle blackbox should be effectively non-participating for vision-frame processing.
- Privileged labels and GTA-native state extraction will come later through a separate ScriptHook-based pipeline.
