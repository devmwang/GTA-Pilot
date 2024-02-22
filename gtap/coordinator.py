from gtap.process.process import PythonProcess

processes = [PythonProcess("DisplayCapture", "gtap.display_capture.display_capture")]

system_processes = {process.name: process for process in processes}
