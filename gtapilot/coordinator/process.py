from abc import ABC, abstractmethod
from multiprocessing import Process
from importlib import import_module
from setproctitle import setproctitle
from typing import List


def launcher(module, name, args):
    module = import_module(module)

    setproctitle(name)

    # Resolve main function from imported module and ensure it's callable.
    main_func = getattr(module, "main", None)
    if not callable(main_func):
        raise AttributeError(
            f"Module '{getattr(module, '__name__', str(module))}' has no callable 'main' attribute"
        )
    main_func(**args)


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
        if self.process is None:
            return None

        # TODO: Initiate process shutdown and confirm


class PythonProcess(BaseProcess):
    def __init__(self, name, module, args=None):
        self.name = name
        self.module = module
        self.args = args if args is not None else {}

    def start(self):
        if self.process is not None:
            return None

        self.process = Process(
            name=self.name, target=launcher, args=(self.module, self.name, self.args)
        )
        self.process.start()


def startAllProcesses(processes: List[BaseProcess]):
    for process in processes:
        process.start()


def waitForProcesses(processes: List[BaseProcess]):
    for process in processes:
        if process.process is not None:
            process.process.join()
