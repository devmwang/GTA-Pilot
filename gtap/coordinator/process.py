from abc import ABC, abstractmethod
from multiprocessing import Process
from importlib import import_module
from setproctitle import setproctitle
from typing import List


def launcher(module, name):
    module = import_module(module)

    setproctitle(name)

    module.main()


class BaseProcess(ABC):
    name = ""
    process = None

    @abstractmethod
    def start(self):
        pass

    def restart(self):
        self.stop()
        self.start()

    def stop(self):
        if self.process == None:
            return None

        # TODO: Initiate process shutdown and confirm


class PythonProcess(BaseProcess):
    def __init__(self, name, module):
        self.name = name
        self.module = module

    def start(self):
        if self.process != None:
            return None

        self.process = Process(
            name=self.name, target=launcher, args=(self.module, self.name)
        )
        self.process.start()


def startAllProcesses(processes: List[BaseProcess]):
    for process in processes:
        process.start()


def waitForProcesses(processes: List[BaseProcess]):
    for process in processes:
        if process.process != None:
            process.process.join()
