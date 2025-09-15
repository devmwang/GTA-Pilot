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

The system is composed of multiple Python subprocesses (video override, visualization, optional blackbox recorder, etc.) and/or native subprocesses (C++ Windows display capture) supervised by a central coordinator.

Supervisor model:

-   Coordinator starts all subprocesses.
-   Any of these conditions triggers a full system shutdown:
    -   ESC key pressed in the coordinator console (Windows).
    -   Ctrl+C / SIGINT / SIGTERM received by the coordinator.
    -   Any child process exits (normal return or crash).
-   Once triggered, the coordinator force-terminates remaining processes (no cooperative polling).

## Blackbox

The blackbox recorder is a Python subprocess that records frames and metadata to a tar archive. It's enabled by setting `BLACKBOX_ENABLED = True` in `gtapilot/config.py`. The blackbox may be useful in the future for debugging, analysis, or training (maybe even training e2e, though very difficult with small quantity of data and compute).
