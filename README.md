# GTA Pilot

This is a project to experiment with different ML and software techniques to create an ADAS/AV system for use in Grand Theft Auto V.

## Inspirations

GTA Pilot is heavily inspired by Tesla's FSD, Mobileye's Supervision/REM, and Comma.ai's OpenPilot.

## Ideas

-   "Vision"-based perception (FSD, Supervision/REM, OpenPilot)
-   Multi-task "HydraNet" ML architecture (FSD)
-   "Data Engine" for data collection (FSD)
-   Map-based localization and integration with perception/planning (Supervision/REM)
-   "Crowd-sourced" HD map creation (REM)
-   "RSS" driving policy model (Supervision)
-   Asynchronous system architecture (OpenPilot)
-   Fast IPC (OpenPilot)

## Runtime & Shutdown Semantics

The system is composed of multiple Python subprocesses (display capture or video override, visualization, optional blackbox recorder, etc.) supervised by a central coordinator.

Supervisor model:

-   Coordinator starts all subprocesses.
-   Any of these conditions triggers a full system shutdown:
    -   ESC key pressed in the coordinator console (Windows).
    -   Ctrl+C / SIGINT / SIGTERM received by the coordinator.
    -   Any child process exits (normal return or crash).
-   Once triggered, the coordinator force-terminates remaining processes (no cooperative polling).

Implications:

-   Child `main()` functions no longer receive or poll a shared `shutdown_event`.
-   Abrupt termination may skip some cleanup/finalization logic inside children. (Future improvements may add a soft-stop channel.)
-   The optional blackbox recorder flushes frame data incrementally, but very recent metadata (last <1s) might be lost if termination is abrupt.

Logs from the coordinator use a `[Coordinator]` prefix and include the root cause and each process exit code for post-run inspection.
