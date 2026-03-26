#!/usr/bin/env python3
"""
Inference runner – compute daily ignition probability map.

Usage:
    python run_inference.py --date 2023-06-15
"""

import argparse, os, sys, yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.inference import run_inference
from src.visualize import plot_probability_map
from src.grid import grid_info


def main():
    parser = argparse.ArgumentParser(description="Daily inference")
    parser.add_argument("--date", required=True, help="Date YYYY-MM-DD")
    args = parser.parse_args()

    with open(os.path.join(ROOT, "config.yaml")) as f:
        cfg = yaml.safe_load(f)

    gi = grid_info(cfg)

    print("=" * 60)
    print(f"Inference for {args.date}")
    print("=" * 60)

    prob_da = run_inference(
        date_str=args.date,
        cfg=cfg,
        model_dir=os.path.join(ROOT, cfg["paths"]["models"]),
        norm_params_path=os.path.join(ROOT, cfg["paths"]["processed_data"],
                                      "norm_params.json"),
        output_dir=os.path.join(ROOT, cfg["paths"]["outputs"]),
    )

    # Plot
    plot_probability_map(
        prob_da.values, gi["lats"], gi["lons"],
        date_str=args.date,
        save_path=os.path.join(ROOT, cfg["paths"]["outputs"],
                               f"map_{args.date}.png"),
    )

    print("\n✅ Inference complete!")


if __name__ == "__main__":
    main()
