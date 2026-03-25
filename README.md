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

- native DX11 display capture or Python video override
- keyboard action capture
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

- capture front RGB frames
- capture current manual keyboard inputs as Atlas-format action packets
- visualize the live stream with action overlays
- optionally record frame/action sessions to disk for later training

What it does **not** do yet:

- run Atlas inference in the live driving loop
- collect privileged GTA state such as pose, actors, lanes, occupancy, or map labels
- infer reverse gear or vehicle state from the game

## Requirements

Current practical assumptions:

- Windows for the native DX11 desktop capture path
- Python `3.13`
- `uv` for environment management

Install the project:

```bash
uv venv .venv
uv pip install -e .
```

If you only want to exercise the pipeline with a prerecorded clip instead of
live desktop capture, `--video-override` is the easiest path.

## How To Use It

### 1. Live desktop capture

Run the coordinator:

```bash
uv run ./gtapilot/main.py
```

By default the coordinator starts:

- `DisplayCaptureDX11`
- `ActionCapture`
- `Visualization`
- `Blackbox` only if enabled in config

If you need a different monitor:

```bash
uv run ./gtapilot/main.py --display-id 0
```

The live capture path uses the native DX11 executable in
`bin/DisplayCaptureDX11.exe` by default.

### 2. Video override

To test the pipeline without GTA running, feed it a local video file:

```bash
uv run ./gtapilot/main.py --video-override path/to/video.mp4
```

This replaces live desktop capture with OpenCV video decode while keeping the
rest of the runtime the same.

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
- `capture_timestamp_ns`
- `publish_timestamp_ns`
- `source`
- `is_repeat`
- `w`
- `h`
- `channels`
- `dtype`

### Action stream

The `input.actions` channel carries Atlas-format action packets with vector
order:

`[steer, throttle, brake, handbrake, reverse, pilot_active]`

Current action capture is keyboard-based and intended as an inference-time data
source only.

The current keyboard mapping is:

- `A` / left arrow: steer left
- `D` / right arrow: steer right
- `W` / up arrow: throttle
- `S` / down arrow: brake
- `Space`: handbrake

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

Then run the coordinator normally:

```bash
uv run ./gtapilot/main.py
```

Recordings are written to:

```text
blackbox-recordings/
```

Each session currently produces:

- `capture_<timestamp>_frames.tar`
- `capture_<timestamp>_metadata.json`

The tar archive contains BMP frames. The JSON manifest contains:

- session-level metadata
- frame envelope and frame metadata
- frame-aligned action payloads
- frame-aligned action vectors
- the raw action stream seen during capture

This is the current usable dataset path for Atlas Stage 1A and related
inference-time training work.

## IPC

The runtime now uses one generic ZeroMQ PUB/SUB channel framework with typed
channel specs:

- `vision.frames`: raw RGB frames plus metadata
- `input.actions`: action packets in Atlas order

Every stream uses the same multipart wire format:

- topic
- envelope JSON
- payload bytes

Current action vector order:

`[steer, throttle, brake, handbrake, reverse, pilot_active]`

`pilot_active=0` currently means manual control / human intervention.

## Blackbox

The blackbox recorder is a Python subprocess that records frame and action data
for Atlas. It is enabled by setting `BLACKBOX_ENABLED = True` in
`gtapilot/config.py`.

Current blackbox outputs:

- a tar archive of BMP frames
- a JSON manifest containing:
  - frame envelope metadata
  - per-frame metadata
  - frame-aligned action payloads and action envelope data
  - frame-aligned action vectors
  - the raw action stream seen during capture

This gives us a usable interim dataset for Atlas training and replay using only
the sources available at inference time. Privileged sensor capture is a later
project.

## Notes And Limitations

- The runtime is currently a capture/recording system, not an end-to-end autonomy stack.
- The DX11 desktop capture path is the main live capture path; Python video override is mainly for testing.
- The action stream reflects keyboard intent, not authoritative in-game vehicle state.
- Privileged labels and GTA-native state extraction will come later through a separate ScriptHook-based pipeline.
