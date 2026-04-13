"""
Command-line entry point.

Usage:
    python -m src.cli --config configs/base.yaml
    python -m src.cli --config configs/base.yaml --model Qwen/Qwen1.5-1.8B-Chat
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from .pipeline import run_pipeline


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Refusal-direction pipeline")
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml",
        help="YAML config file (see configs/base.yaml for the schema).",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override the model name from the config.",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Override the device (cuda / cpu).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="outputs",
        help="Where to write the selected direction and the metric summary.",
    )
    args = parser.parse_args()

    cfg = _load_config(args.config)
    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.device is not None:
        cfg["model"]["device"] = args.device

    result, direction = run_pipeline(cfg)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = cfg["model"]["name"].replace("/", "__")
    torch.save(direction, out_dir / f"{safe_name}_refusal_direction.pt")
    with open(out_dir / f"{safe_name}_metrics.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2)
    print("[pipeline] metrics:")
    print(json.dumps(result.to_dict(), indent=2))


if __name__ == "__main__":
    main()
