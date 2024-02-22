from gtapilot.coordinator.process import (
    startAllProcesses,
    waitForProcesses,
    PythonProcess,
)

processes = [PythonProcess("DisplayCapture", "gtap.display_capture.display_capture")]

system_processes = {process.name: process for process in processes}


def main():
    startAllProcesses(processes)
    waitForProcesses(processes)
