from gtapilot.coordinator.process import (
    startAllProcesses,
    waitForProcesses,
    PythonProcess,
)

# from gtapilot.ipc.vision_ipc import VisionIPC

processes = [
    PythonProcess(
        "DisplayCapture",
        "gtapilot.display_capture.display_capture",
        {"display": 1},
    ),
    PythonProcess("Visualizer", "gtapilot.visualizer.visualizer"),
]

system_processes = {process.name: process for process in processes}


def main():
    # Start all processes
    startAllProcesses(processes)

    # Wait for processes to finish
    waitForProcesses(processes)
