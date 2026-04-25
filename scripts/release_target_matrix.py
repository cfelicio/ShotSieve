from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

_release_targets = importlib.import_module("shotsieve.release_targets")
tier1_release_matrix = _release_targets.tier1_release_matrix


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Emit ShotSieve release target matrix as JSON")
    parser.add_argument(
        "--kind",
        choices=("runtime",),
        default="runtime",
        help="Which release target matrix to emit",
    )
    return parser


def main() -> None:
    parser = build_parser()
    parser.parse_args()

    matrix = tier1_release_matrix()

    print(json.dumps(matrix, indent=2))


if __name__ == "__main__":
    main()