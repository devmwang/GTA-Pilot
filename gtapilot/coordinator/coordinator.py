import msvcrt
import signal
import sys
import threading
from multiprocessing import Event
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

    # Visualizer always included for now
    procs.append(PythonProcess("Visualizer", "gtapilot.visualizer.visualizer"))

    if BLACKBOX_ENABLED:
        procs.append(PythonProcess("Blackbox", "gtapilot.blackbox.blackbox"))
    return procs


def _esc_listener(shutdown_event):
    """Background thread to watch for ESC key press (Windows) and set shutdown_event."""
    if sys.platform == "win32" and msvcrt is not None:
        import time

        while not shutdown_event.is_set():
            if msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b"\x1b",):  # ESC byte
                    print("ESC detected. Initiating shutdown...")
                    shutdown_event.set()
                    break
            time.sleep(0.05)
    else:
        print("ESC hotkey unsupported on this platform; use Ctrl+C to exit.")


def main(video_override: Optional[str] = None):
    shutdown_event = Event()

    def _signal_handler(signum, frame):
        if not shutdown_event.is_set():
            print(f"Signal {signum} received. Initiating shutdown...")
            shutdown_event.set()

    # Register common termination signals
    try:
        signal.signal(signal.SIGINT, _signal_handler)
        signal.signal(signal.SIGTERM, _signal_handler)
    except Exception:
        pass

    # Start ESC listener thread
    esc_thread = threading.Thread(
        target=_esc_listener, args=(shutdown_event,), daemon=True
    )
    esc_thread.start()

    processes = build_processes(video_override=video_override)
    system_processes = {process.name: process for process in processes}

    if video_override:
        print(
            f"Starting system processes using video override '{video_override}'. Press ESC for graceful shutdown (or Ctrl+C)."
        )
    else:
        print(
            "Starting system processes with live display capture. Press ESC for graceful shutdown (or Ctrl+C)."
        )

    startAllProcesses(processes, shutdown_event=shutdown_event)
    waitForProcesses(processes, shutdown_event=shutdown_event)

    print("Coordinator shutdown complete.")


def startAllProcesses(processes: List[BaseProcess], shutdown_event=None):
    for process in processes:
        process.start(shutdown_event=shutdown_event)


def waitForProcesses(
    processes: List[BaseProcess], shutdown_event=None, poll_interval=0.2
):
    import time

    try:
        while shutdown_event is not None and not shutdown_event.is_set():
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        if shutdown_event is not None:
            print("KeyboardInterrupt received: initiating shutdown...")
            shutdown_event.set()

    # Begin graceful join
    for process in processes:
        if process.process is not None:
            process.process.join(timeout=5)

    # Force terminate any stragglers
    for process in processes:
        if process.process is not None and process.process.is_alive():
            print(f"Force terminating process {process.name}")
            process.stop()
