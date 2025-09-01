from abc import ABC, abstractmethod
from importlib import import_module
from multiprocessing import Process
from typing import List

from setproctitle import setproctitle


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
    def start(self, shutdown_event=None):
        pass

    def restart(self, shutdown_event=None):
        self.stop()
        self.start(shutdown_event=shutdown_event)

    def stop(self):
        if self.process is None:
            return None

        # Actual graceful stop is coordinated via a shared shutdown_event.
        # This method here is only for a forced terminate fallback.
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=2)


class PythonProcess(BaseProcess):
    def __init__(self, name, module, args=None):
        self.name = name
        self.module = module
        self.args = args if args is not None else {}

    def start(self, shutdown_event=None):
        if self.process is not None:
            return None

        launch_args = dict(self.args)
        if shutdown_event is not None:
            # Inject only if the user code accepts it; safe to always include.
            launch_args["shutdown_event"] = shutdown_event

        self.process = Process(
            name=self.name, target=launcher, args=(self.module, self.name, launch_args)
        )
        self.process.start()
