from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class JsonlLogger:
    def __init__(self, output_dir: str | Path, file_name: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / file_name

    def log(self, payload: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, default=float) + "\n")


class ScalarLogger:
    def __init__(
        self,
        output_dir: str | Path,
        *,
        jsonl_file: str = "metrics.jsonl",
        tensorboard_dir: str | None = None,
    ):
        self.jsonl = JsonlLogger(output_dir, jsonl_file)
        self.writer = None
        if tensorboard_dir:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self.writer = SummaryWriter(log_dir=str(Path(output_dir) / tensorboard_dir))
            except Exception:
                self.writer = None

    def log_scalars(self, step: int, scalars: dict[str, float], prefix: str = "") -> None:
        flat = {
            f"{prefix}{key}" if prefix else key: float(value)
            for key, value in scalars.items()
        }
        payload = {"step": int(step), **flat}
        self.jsonl.log(payload)
        print(" ".join([f"step={step}"] + [f"{key}={value:.4f}" for key, value in flat.items()]))
        if self.writer is not None:
            for key, value in flat.items():
                self.writer.add_scalar(key, value, step)

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
