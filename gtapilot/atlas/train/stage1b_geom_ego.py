from __future__ import annotations

from .common import run_stage_smoke


def main() -> None:
    result = run_stage_smoke("stage1b")
    print(f"stage1b total_loss={result['losses']['total'].item():.4f}")


if __name__ == "__main__":
    main()
