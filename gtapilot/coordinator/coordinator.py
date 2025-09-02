import msvcrt
import signal
import sys
import threading
import time
from typing import List, Optional

from gtapilot.config import BLACKBOX_ENABLED
from gtapilot.coordinator.process import BaseProcess, PythonProcess


def build_processes(video_override: Optional[str] = None) -> List[BaseProcess]:
    procs: List[BaseProcess] = []
    if video_override:
        procs.append(
            PythonProcess(
                "DisplayOverride",
                "gtapilot.display_capture.display_override",
                {"video_path": video_override},
            )
        )
    else:
        procs.append(
            PythonProcess(
                "DisplayCapture",
                "gtapilot.display_capture.display_capture",
                {"display": 1},
            )
        )

    # Visualization always included for now
    procs.append(PythonProcess("Visualization", "gtapilot.visualization.visualization"))

    if BLACKBOX_ENABLED:
        procs.append(PythonProcess("Blackbox", "gtapilot.blackbox.blackbox"))
    return procs


def main(video_override: Optional[str] = None):
    """Coordinator entrypoint (supervisor model).

    Starts all subsystem processes and monitors them. Any of these conditions
    trigger a full system shutdown cascade:
      * ESC key pressed in console (Windows)
      * SIGINT/SIGTERM received
      * Any child process exits for any reason (exitcode 0 or non‑zero)
    Remaining processes are forcefully terminated (terminate + join timeout).
    """

    processes = build_processes(video_override=video_override)

    print("[Coordinator] Launching processes: " + ", ".join(p.name for p in processes))

    shutting_down = False
    shutdown_origin = None  # dict with keys: type, process, exitcode, timestamp, message
    cascade_executed = False
    poll_interval = 0.05
    shutdown_start_ts: float | None = None

    # --- Helper functions -------------------------------------------------
    def log(msg: str):
        print(f"[Coordinator] {msg}")

    def mark_shutdown(origin_type: str, process_name: str | None = None, exitcode=None, message: str | None = None):
        nonlocal shutting_down, shutdown_origin, shutdown_start_ts
        if shutting_down:
            return
        shutting_down = True
        shutdown_start_ts = time.time()
        shutdown_origin = {
            "type": origin_type,
            "process": process_name,
            "exitcode": exitcode,
            "timestamp": shutdown_start_ts,
            "message": message,
        }
        detail = f"type={origin_type}"
        if process_name is not None:
            detail += f" process={process_name}"
        if exitcode is not None:
            detail += f" exitcode={exitcode}"
        if message:
            detail += f" message='{message}'"
        log(f"Shutdown initiated: {detail}")

    def esc_listener():
        if sys.platform != "win32" or msvcrt is None:
            log("ESC hotkey unsupported on this platform; use Ctrl+C to exit.")
            return
        while True:
            if shutting_down:
                break
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b"\x1b",):
                    mark_shutdown("esc", message="ESC key pressed")
                    break
            time.sleep(0.05)

    def handle_signal(signum, frame):  # noqa: ARG001
        mark_shutdown("signal", message=f"Signal {signum} received")

    # Register signals
    try:
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    except Exception:  # pragma: no cover - platform differences
        pass

    threading.Thread(target=esc_listener, daemon=True).start()

    # Start child processes
    for p in processes:
        log(f"Starting {p.name}")
        try:
            p.start()
        except Exception as e:  # startup failure
            log(f"Failed to start {p.name}: {e}")
            mark_shutdown("startup_failure", process_name=p.name, message=str(e))
            break

    # Monitoring loop
    try:
        while True:
            # Detect child exits if not already shutting down
            if not shutting_down:
                for p in processes:
                    if p.exitcode is not None:
                        mark_shutdown("child_exit", process_name=p.name, exitcode=p.exitcode)
                        break

            # Execute cascade once when shutdown begins
            if shutting_down and not cascade_executed:
                cascade_executed = True
                # Terminate every still‑alive process (those already exited are skipped)
                survivors = [p for p in processes if p.alive()]
                if survivors:
                    log("Terminating remaining processes: " + ", ".join(p.name for p in survivors))
                for p in survivors:
                    p.stop()
                # Second pass: log any still alive
                stubborn = [p for p in processes if p.alive()]
                if stubborn:
                    log("Processes still alive after initial terminate: " + ", ".join(p.name for p in stubborn))

            # Exit condition: all processes have exitcode assigned AND (if shutting_down) cascade executed
            if shutting_down and all(p.exitcode is not None for p in processes):
                break

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        mark_shutdown("signal", message="KeyboardInterrupt")
    except Exception as e:  # Unexpected coordinator error
        log(f"Coordinator internal error: {e}")
        mark_shutdown("coordinator_error", message=str(e))

    duration = 0.0
    if shutdown_start_ts is not None:
        duration = time.time() - shutdown_start_ts

    if shutdown_origin is None:
        log("Exiting without explicit shutdown origin (no processes started?)")
    else:
        log(
            "Shutdown complete. Origin="
            f"{shutdown_origin['type']} process={shutdown_origin['process']} exitcode={shutdown_origin['exitcode']} "
            f"duration={duration:.2f}s"
        )

    # Summary of exit codes
    log("Process exit codes: " + ", ".join(f"{p.name}={p.exitcode}" for p in processes))
