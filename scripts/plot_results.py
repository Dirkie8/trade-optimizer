#!/usr/bin/env python3
"""
Plot evaluation results: equity curve and key metrics.

Examples:
    # Single file
    python scripts/plot_results.py --input results/Strategy/evaluations/full_dataset_backtest.json

    # Batch over all strategies, saving into each evaluation folder
    python scripts/plot_results.py --batch

    # Batch into a consolidated images folder under results/images
    python scripts/plot_results.py --batch --output-root results/images
"""
import argparse
import json
import os
import sys
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

# Ensure project root is on sys.path to allow relative imports if needed later
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="Plot evaluation results")
    parser.add_argument("--input", required=False, help="Path to a single evaluation JSON to plot")
    parser.add_argument("--batch", action="store_true", help="If set, plot all evaluations found under results/*/evaluations/")
    parser.add_argument("--output-root", default=None, help="If set in batch mode, save images under this root (e.g., results/images/<strategy>/)")
    parser.add_argument("--output", default=None, help="Output path for plot image (auto-generated if not specified)")
    parser.add_argument("--show", action="store_true", default=False, help="Display plot interactively")
    args = parser.parse_args()

    def render_one(input_path: str, output_override: str | None = None):
        with open(input_path, "r") as f:
            res = json.load(f)
        return plot_payload(res, input_path, output_override)

    def plot_payload(results, input_path: str, output_override: str | None = None):
        eq = pd.DataFrame(results["equity_curve"])  # time, equity
        eq["time"] = pd.to_datetime(eq["time"], utc=True)
        eq = eq.set_index("time")

        metrics = results.get("metrics", {})

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot(eq.index, eq["equity"], label="Equity")
        ax.set_title(f"Equity Curve — {results.get('strategy')} ({results.get('symbol')} {results.get('timeframe')})")
        ax.set_xlabel("Time")
        ax.set_ylabel("Account Balance")
        ax.grid(True, alpha=0.3)

        text = (
            f"Return: {metrics.get('total_return_pct', 0):.2f}%\n"
            f"Sharpe: {metrics.get('sharpe', 0):.2f}\n"
            f"Max DD: {metrics.get('max_drawdown_pct', 0):.2f}%\n"
            f"Trades: {metrics.get('trades', 0)} | Win%: {metrics.get('win_rate_pct', 0):.1f}%\n"
        )
        ax.legend(loc="upper left")
        ax.text(0.995, 0.02, text, transform=ax.transAxes, fontsize=10, va="bottom", ha="right",
                bbox=dict(facecolor="white", alpha=0.7, boxstyle="round"))

        plt.tight_layout()

        # Determine output path
        if output_override:
            output_path = output_override
        elif args.output:
            output_path = args.output
        else:
            # Auto-generate output path in the same directory as input file
            input_dir = os.path.dirname(input_path)
            input_basename = os.path.splitext(os.path.basename(input_path))[0]
            output_path = os.path.join(input_dir, f"{input_basename}_plot.png")

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_path}")
        if args.show:
            plt.show()
        else:
            plt.close()
        return output_path

    if args.batch:
        # Discover all evaluation JSONs
        eval_files = []
        for root, dirs, files in os.walk("results"):
            if os.path.basename(root) == "evaluations":
                for fn in files:
                    if fn in ("full_dataset_backtest.json", "eval_results.json"):
                        eval_files.append(os.path.join(root, fn))
        eval_files.sort()
        if not eval_files:
            print("No evaluation JSONs found under results/*/evaluations/")
            sys.exit(0)

        for ip in eval_files:
            if args.output_root:
                # Save under output-root with a flat structure and distinguishing filenames
                # e.g., results/images/<strategy>_<basename>_plot.png
                parts = ip.split(os.sep)
                try:
                    strat_idx = parts.index("results") + 1
                    strategy = parts[strat_idx]
                except Exception:
                    strategy = "Strategy"
                base = os.path.splitext(os.path.basename(ip))[0]
                out_dir = args.output_root
                out_path = os.path.join(out_dir, f"{strategy}_{base}_plot.png")
                render_one(ip, out_path)
            else:
                render_one(ip, None)
        print("All plots generated.")
        return

    # Single-file mode
    if not args.input:
        print("--input is required unless --batch is provided")
        sys.exit(1)
    render_one(args.input, None)



if __name__ == "__main__":
    main()
