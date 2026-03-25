from __future__ import annotations

from .common import run_stage_smoke


def main() -> None:
    result = run_stage_smoke("stage3")
    print(f"stage3 total_loss={result['losses']['total'].item():.4f}")


if __name__ == "__main__":
    main()
