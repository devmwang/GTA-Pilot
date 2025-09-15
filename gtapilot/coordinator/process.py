from abc import ABC, abstractmethod
from importlib import import_module
from multiprocessing import Process
import os
import subprocess
import sys
import traceback

from setproctitle import setproctitle


def launcher(module, name, args):
    """Module process entry wrapper.

    Imports the module, resolves its main(**args) and executes it. Any exception is
    printed with traceback and the interpreter exits immediately with a non‑zero
    status to ensure the coordinator sees the crash quickly. A normal return exits
    with code 0 (bypassing atexit handlers for faster, deterministic shutdown).
    """
    try:
        module_obj = import_module(module)
        setproctitle(name)
        main_func = getattr(module_obj, "main", None)
        if not callable(main_func):
            raise AttributeError(
                f"Module '{getattr(module_obj, '__name__', str(module_obj))}' has no callable 'main' attribute"
            )
        main_func(**args)
    except (
        Exception
    ):  # noqa: BLE001 broad by design to convert any failure to exitcode!=0
        try:
            traceback.print_exc()
            sys.stderr.flush()
            sys.stdout.flush()
        finally:
            os._exit(1)
    os._exit(0)


class BaseProcess(ABC):
    name = ""
    process: Process | None = None

    @abstractmethod
    def start(self):  # pragma: no cover - interface definition
        pass

    def restart(self):
        self.stop()
        self.start()

    def alive(self) -> bool:
        return bool(self.process and self.process.is_alive())

    @property
    def exitcode(self):
        return None if self.process is None else self.process.exitcode

    def stop(self, grace_timeout: float = 1.5):
        """Forcefully stop the underlying process.

        We don't attempt a soft cooperative signal here (future enhancement) –
        coordinator has already decided to terminate.
        """
        if self.process is None:
            return
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=grace_timeout)
            if self.process.is_alive():  # Rare on Windows; log from caller.
                # Leave second terminate attempt to caller if desired.
                pass


class PythonProcess(BaseProcess):
    def __init__(self, name: str, module: str, args: dict | None = None):
        self.name = name
        self.module = module
        self.args = args if args is not None else {}

    def start(self):
        if self.process is not None:
            return
        launch_args = dict(self.args)
        self.process = Process(
            name=self.name,
            target=launcher,
            args=(self.module, self.name, launch_args),
        )
        self.process.start()


class ExecutableProcess(BaseProcess):
    def __init__(self, name: str, exe_path: str, args: list[str] | None = None):
        self.name = name
        self.exe_path = exe_path
        self.args = args or []
        self._popen: subprocess.Popen | None = None

    def start(self):
        if self._popen is not None:
            return

        self._popen = subprocess.Popen([self.exe_path, *self.args])

    def alive(self) -> bool:
        return bool(self._popen and self._popen.poll() is None)

    @property
    def exitcode(self):
        return None if self._popen is None else self._popen.poll()

    def stop(self, grace_timeout: float = 1.5):
        if not self._popen:
            return

        try:
            self._popen.terminate()
            self._popen.wait(timeout=grace_timeout)

        except Exception:
            pass
