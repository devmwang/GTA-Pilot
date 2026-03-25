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
