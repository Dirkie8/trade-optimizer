#!/usr/bin/env python3
"""
Hybrid Optimization Strategy: Random Search followed by Gradient-based Refinement

This demonstrates the powerful combination of:
1. Random search to explore the parameter space broadly
2. TPE (Bayesian/gradient-based) to refine around promising regions

Example usage:
    python scripts/hybrid_optimize.py \
        --strategy_config functions/configs/rsi_strategy.yaml \
        --main_config configs/main_config.yaml
"""
import subprocess
import sys
import json
import os
import pandas as pd
from pathlib import Path
import yaml

# ANSI colors (disable with NO_COLOR env var)
if os.environ.get("NO_COLOR"):
    RESET = BOLD = CYAN = GREEN = YELLOW = MAGENTA = ""
else:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    CYAN = "\033[36m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    MAGENTA = "\033[35m"

def run_optimization(strategy_config, main_config, method, n_samples, walk_forward=True, n_jobs=1, n_folds=3, output_path=None, warmstart_csv=None):
    """Run optimization and return results path"""
    # Use the same Python interpreter that's running this script
    python_exec = sys.executable or "python"
    cmd = [
        python_exec, "scripts/optimize_strategy.py",
        "--strategy_config", strategy_config,
        "--main_config", main_config,
        "--method", method,
        "--sort-by", "consistency_score",
        "--n_jobs", str(n_jobs),
    ]
    
    if method in ["tpe", "optuna"]:
        cmd.extend(["--n_trials", str(n_samples)])
    else:
        cmd.extend(["--n_random", str(n_samples)])
    
    if walk_forward:
        cmd.extend(["--walk-forward", "--n-folds", str(n_folds)])

    if output_path:
        cmd.extend(["--output", output_path])
    if warmstart_csv and method in ["tpe", "optuna"]:
        cmd.extend(["--warmstart_csv", warmstart_csv])
    
    print(f"\n{'='*80}")
    print(f"Running {method.upper()} optimization with {n_samples} samples...")
    print(f"{'='*80}\n")
    
    result = subprocess.run(cmd, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: {method} optimization failed!")
        return None
    
    return True


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Hybrid Random + TPE optimization")
    parser.add_argument("--strategy_config", required=True)
    parser.add_argument("--main_config", required=True)
    parser.add_argument("--random_samples", type=int, default=30,
                       help="Number of random samples in phase 1 (default: 30)")
    parser.add_argument("--tpe_trials", type=int, default=50,
                       help="Number of TPE trials in phase 2 (default: 50)")
    parser.add_argument("--walk-forward", action="store_true",
                       help="Use walk-forward validation (recommended)")
    parser.add_argument("--n_jobs", type=int, default=1,
                       help="Number of parallel workers (default: 1, use 4 for speed)")
    parser.add_argument("--n-folds", type=int, default=3, help="Folds for walk-forward when enabled")
    args = parser.parse_args()
    
    banner = (
        f"{BOLD}{CYAN}\n"
        "╔══════════════════════════════════════════════════════════════════════════════╗\n"
        "║                        HYBRID OPTIMIZATION STRATEGY                          ║\n"
        "╚══════════════════════════════════════════════════════════════════════════════╝\n\n"
        "Phase 1: Random Search (Exploration)\n"
        f"    - Samples: {args.random_samples}\n"
        "    - Goal: Broadly explore parameter space\n"
        "    - Finds promising regions\n\n"
        "Phase 2: TPE/Bayesian (Exploitation)\n"
        f"    - Trials: {args.tpe_trials}\n"
        "    - Goal: Refine around best regions found in Phase 1\n"
        "    - Uses gradient information to converge faster\n\n"
        f"Walk-Forward: {'Enabled ✓' if args.walk_forward else 'Disabled'}\n"
        f"    {RESET}"
    )
    print(banner, flush=True)
    
    # Load strategy config to determine results dir
    with open(args.strategy_config, "r") as f:
        strat_conf = yaml.safe_load(f)
    strategy_class_name = strat_conf['strategy']['class']
    results_dir = strat_conf['strategy'].get('results_dir', strategy_class_name)
    base_opt_dir = Path("results") / results_dir / "optimizations"
    base_opt_dir.mkdir(parents=True, exist_ok=True)

    # Clear, early banner so batch logs show which strategy just started
    print("\n" + "#" * 88, flush=True)
    print(f"{BOLD}{GREEN}### STARTING OPTIMIZATION: {strategy_class_name}{RESET}  |  config={args.strategy_config}", flush=True)
    wf_txt = f"walk-forward {args.n_folds} folds" if args.walk_forward else "no walk-forward"
    print(f"{YELLOW}### SETTINGS: random_samples={args.random_samples}, tpe_trials={args.tpe_trials}, n_jobs={args.n_jobs}, {wf_txt}{RESET}", flush=True)
    print("#" * 88 + "\n", flush=True)

    random_csv = str(base_opt_dir / "optimization_random.csv")
    tpe_csv = str(base_opt_dir / "optimization_tpe.csv")
    final_csv = str(base_opt_dir / "optimization_results.csv")
    best_json = str(base_opt_dir / "optimization_results_best.json")

    # Phase 1: Random Search
    print(f"\n{MAGENTA}🎲 PHASE 1: RANDOM SEARCH (EXPLORATION){RESET}", flush=True)
    print("─" * 80, flush=True)
    success = run_optimization(
        args.strategy_config,
        args.main_config,
        method="random",
        n_samples=args.random_samples,
        walk_forward=args.walk_forward,
        n_jobs=args.n_jobs,
        n_folds=args.n_folds,
        output_path=random_csv,
    )
    
    if not success:
        print("Phase 1 failed. Aborting.")
        sys.exit(1)
    
    # Phase 2: TPE (Gradient-based)
    print(f"\n{GREEN}🎯 PHASE 2: TPE OPTIMIZATION (EXPLOITATION){RESET}", flush=True)
    print("─" * 80, flush=True)
    print("TPE will learn from Phase 1 results and focus on promising regions...", flush=True)
    success = run_optimization(
        args.strategy_config,
        args.main_config,
        method="tpe",
        n_samples=args.tpe_trials,
        walk_forward=args.walk_forward,
        n_jobs=args.n_jobs,
        n_folds=args.n_folds,
        output_path=tpe_csv,
        warmstart_csv=random_csv,
    )
    
    if not success:
        print("Phase 2 failed. Proceeding with available results (random phase).")
    
    # Merge phase outputs into final CSV and best JSON
    frames = []
    for p in [random_csv, tpe_csv]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df["phase"] = Path(p).stem.replace("optimization_", "")
                frames.append(df)
            except Exception as e:
                print(f"Warning: failed to read {p}: {e}", flush=True)

    if frames:
        merged = pd.concat(frames, ignore_index=True)
        # prefer consistency_score then total_return_pct
        # Prefer consistency, then more trades, then total return
        sort_cols = []
        if "consistency_score" in merged.columns:
            sort_cols.append("consistency_score")
        if "trades_count" in merged.columns:
            sort_cols.append("trades_count")
        if "total_return_pct" in merged.columns:
            sort_cols.append("total_return_pct")
        if sort_cols:
            merged = merged.sort_values(sort_cols, ascending=[False]*len(sort_cols))
        merged.to_csv(final_csv, index=False)
        print(f"Merged results written to {final_csv}", flush=True)

        if not merged.empty:
            best_row = merged.iloc[0]
            params = {k.replace("param_", ""): best_row[k] for k in merged.columns if k.startswith("param_")}
            # coerce numpy types
            clean = {}
            for k, v in params.items():
                if hasattr(v, 'item'):
                    try:
                        clean[k] = v.item()
                        continue
                    except Exception:
                        pass
                clean[k] = v

            # Build a richer payload to expose phase and best metrics
            metric_fields = [
                "consistency_score",
                "total_return_pct",
                "sharpe_ratio",
                "sortino_ratio",
                "calmar_ratio",
                "avg_drawdown_pct",
                "max_drawdown_pct",
                "trades_count",
            ]
            metrics_payload = {}
            for m in metric_fields:
                if m in merged.columns:
                    val = best_row.get(m)
                    try:
                        if hasattr(val, 'item'):
                            val = val.item()
                    except Exception:
                        pass
                    metrics_payload[m] = val

            payload = {
                "source_phase": str(best_row.get("phase", "unknown")),
                "params": clean,
            }
            if metrics_payload:
                payload["metrics"] = metrics_payload

            with open(best_json, "w") as f:
                json.dump(payload, f, indent=2)
            print(f"Best parameters saved to {best_json}", flush=True)

    print("\n" + "="*80, flush=True)
    print((f"{YELLOW}✓ HYBRID OPTIMIZATION COMPLETE (with warnings){RESET}" if not success else f"{GREEN}✓ HYBRID OPTIMIZATION COMPLETE!{RESET}"), flush=True)
    print("="*80, flush=True)
    print("\nRun 'python scripts/evaluate_strategy.py --strategy_config <cfg> --main_config configs/main_config.yaml' to evaluate on full data.", flush=True)


if __name__ == "__main__":
    main()
