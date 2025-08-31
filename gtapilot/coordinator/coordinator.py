from gtapilot.config import BLACKBOX_ENABLED

from typing import List
from gtapilot.coordinator.process import (
    startAllProcesses,
    waitForProcesses,
    BaseProcess,
    PythonProcess,
)

processes: List[BaseProcess] = [
    PythonProcess(
        "DisplayCapture",
        "gtapilot.display_capture.display_capture",
        {"display": 1},
    ),
    PythonProcess("Visualizer", "gtapilot.visualizer.visualizer"),
]

if BLACKBOX_ENABLED:
    processes.append(PythonProcess("Blackbox", "gtapilot.blackbox.blackbox"))

system_processes = {process.name: process for process in processes}


def main():
    # Start all processes
    startAllProcesses(processes)

    # Wait for processes to finish
    waitForProcesses(processes)
