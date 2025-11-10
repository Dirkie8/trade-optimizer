#!/usr/bin/env python3
"""
Compare Bayesian Optimization Results Across All Strategies

This script analyzes train/validation metrics from Bayesian optimization runs
and provides recommendations on the best performing strategies.

Usage:
    python scripts/compare_bayesian_results.py [--output report.md] [--sort-by val_sharpe|val_reward] [--subdir optimizations|evaluations|auto]
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_bayesian_results(results_dir: Path, subdir: str = "optimizations") -> Dict[str, dict]:
    """
    Load all Bayesian optimization validation results.
    
    Returns:
        Dict mapping strategy name to {params, train_metrics, validation_metrics}
    """
    strategies = {}
    
    for strategy_folder in results_dir.iterdir():
        if not strategy_folder.is_dir():
            continue
            
        strategy_name = strategy_folder.name
        try:
            if subdir in ("optimizations", "auto"):
                opt_dir = strategy_folder / "optimizations"
                validation_file_wf = opt_dir / "bayesian_wf_optimization_results_validation.json"
                validation_file_std = opt_dir / "bayesian_optimization_results_validation.json"
                validation_file = validation_file_wf if validation_file_wf.exists() else validation_file_std
                if validation_file.exists():
                    with open(validation_file, 'r') as f:
                        data = json.load(f)
                    strategies[strategy_name] = {
                        'params': data.get('params', {}),
                        'train': {
                            'reward': data.get('train_metrics', {}).get('reward_metric', 0),
                            'return': data.get('train_metrics', {}).get('total_return_pct', 0),
                            'sharpe': data.get('train_metrics', {}).get('sharpe', 0),
                            'sortino': data.get('train_metrics', {}).get('sortino', 0),
                            'calmar': data.get('train_metrics', {}).get('calmar', 0),
                            'drawdown': data.get('train_metrics', {}).get('max_drawdown_pct', 0),
                            'trades': data.get('train_metrics', {}).get('trades', 0),
                            'consistency': data.get('train_metrics', {}).get('consistency_score', 0),
                        },
                        'validation': {
                            'reward': data.get('validation_metrics', {}).get('reward_metric', 0),
                            'reward_raw': data.get('validation_metrics', {}).get('reward_raw'),
                            'reward_reason': data.get('validation_metrics', {}).get('reward_reason'),
                            'reward_floor_applied': data.get('validation_metrics', {}).get('reward_floor_applied', False),
                            'return': data.get('validation_metrics', {}).get('total_return_pct', 0),
                            'sharpe': data.get('validation_metrics', {}).get('sharpe', 0),
                            'sortino': data.get('validation_metrics', {}).get('sortino', 0),
                            'calmar': data.get('validation_metrics', {}).get('calmar', 0),
                            'drawdown': data.get('validation_metrics', {}).get('max_drawdown_pct', 0),
                            'trades': data.get('validation_metrics', {}).get('trades', 0),
                        },
                    }
                    continue  # done with this strategy

            if subdir in ("evaluations", "auto"):
                eval_dir = strategy_folder / "evaluations"
                eval_file = None
                # Prefer full dataset backtest if present
                cand1 = eval_dir / "full_dataset_backtest.json"
                cand2 = eval_dir / "eval_results.json"
                if cand1.exists():
                    eval_file = cand1
                elif cand2.exists():
                    eval_file = cand2
                if eval_file and eval_file.exists():
                    with open(eval_file, 'r') as f:
                        data = json.load(f)
                    metrics = data.get('metrics', data)
                    strategies[strategy_name] = {
                        'params': data.get('params', {}),
                        'train': {
                            'reward': 0,
                            'return': 0,
                            'sharpe': 0,
                            'sortino': 0,
                            'calmar': 0,
                            'drawdown': 0,
                            'trades': 0,
                            'consistency': 0,
                        },
                        'validation': {
                            'reward': 0,
                            'return': metrics.get('total_return_pct', 0),
                            'sharpe': metrics.get('sharpe', 0),
                            'sortino': metrics.get('sortino', 0),
                            'calmar': metrics.get('calmar', 0),
                            'drawdown': metrics.get('max_drawdown_pct', 0),
                            'trades': metrics.get('trades', 0),
                        },
                    }
                    continue
        except Exception as e:
            print(f"Warning: Could not load {strategy_name}: {e}", file=sys.stderr)
            continue
    
    return strategies


def calculate_overfitting_score(train_metric: float, val_metric: float) -> float:
    """
    Calculate overfitting score (lower is better).
    Returns the percentage degradation from train to validation.
    Negative values mean validation performed better than train.
    """
    if train_metric == 0:
        return 0.0
    return ((train_metric - val_metric) / abs(train_metric)) * 100


def calculate_robustness_score(strategies_data: dict) -> Dict[str, float]:
    """
    Calculate robustness score for each strategy.
    
    Robustness = validation_sharpe * (1 - overfitting_penalty)
    Where overfitting_penalty is based on the degradation from train to validation.
    """
    scores = {}
    
    for name, data in strategies_data.items():
        train_sharpe = data['train']['sharpe']
        val_sharpe = data['validation']['sharpe']
        
        # Calculate degradation (positive = worse performance on validation)
        if train_sharpe != 0:
            degradation = (train_sharpe - val_sharpe) / abs(train_sharpe)
            degradation = max(0, degradation)  # Only penalize degradation, not improvement
        else:
            degradation = 0
        
        # Robustness score: validation performance with overfitting penalty
        # If validation > train, no penalty (degradation = 0)
        robustness = val_sharpe * (1 - degradation * 0.5)  # 50% weight on overfitting
        scores[name] = robustness
    
    return scores


def generate_report(strategies: Dict[str, dict], sort_by: str = "val_sharpe") -> str:
    """Generate a comprehensive analysis report."""
    
    if not strategies:
        return "No Bayesian optimization results found. Run optimizations first.\n"
    
    report_lines = []
    report_lines.append("=" * 100)
    report_lines.append("BAYESIAN OPTIMIZATION RESULTS ANALYSIS")
    report_lines.append("=" * 100)
    report_lines.append(f"\nTotal Strategies Analyzed: {len(strategies)}\n")
    
    # Calculate robustness scores
    robustness_scores = calculate_robustness_score(strategies)
    
    # Sorting options
    if sort_by == "val_reward":
        primary_sorted = sorted(
            strategies.items(),
            key=lambda x: x[1]['validation']['reward'],
            reverse=True
        )
        primary_label = "TOP 10 STRATEGIES BY VALIDATION REWARD"
    else:
        primary_sorted = sorted(
            strategies.items(),
            key=lambda x: x[1]['validation']['sharpe'],
            reverse=True
        )
        primary_label = "TOP 10 STRATEGIES BY VALIDATION SHARPE RATIO"
    
    # Sort by robustness score
    sorted_by_robustness = sorted(
        robustness_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # ===== SUMMARY TABLE =====
    report_lines.append("\n" + "=" * 100)
    report_lines.append("SUMMARY: TRAIN vs VALIDATION PERFORMANCE")
    report_lines.append("=" * 100)
    report_lines.append(f"\n{'Strategy':<30} {'Train':^30} {'Validation':^30} {'Overfitting':<10}")
    report_lines.append(f"{'':<30} {'Return%':>8} {'Sharpe':>8} {'DD%':>8}   {'Return%':>8} {'Sharpe':>8} {'DD%':>8}   {'Score':<10}")
    report_lines.append("-" * 100)
    
    for name, data in primary_sorted:
        train = data['train']
        val = data['validation']
        
        # Calculate overfitting on Sharpe
        overfitting = calculate_overfitting_score(train['sharpe'], val['sharpe'])
        
        report_lines.append(
            f"{name:<30} "
            f"{train['return']:>8.1f} {train['sharpe']:>8.3f} {train['drawdown']:>8.1f}   "
            f"{val['return']:>8.1f} {val['sharpe']:>8.3f} {val['drawdown']:>8.1f}   "
            f"{overfitting:>8.1f}%"
        )
    
    # ===== TOP PERFORMERS BY PRIMARY SORT =====
    report_lines.append("\n\n" + "=" * 100)
    report_lines.append(primary_label)
    report_lines.append("=" * 100)
    if sort_by == "val_reward":
        report_lines.append(f"\n{'Rank':<6} {'Strategy':<30} {'Val Reward':>12} {'Val Sharpe':>12} {'Val Return%':>12} {'Val DD%':>12} {'Trades':>10}")
    else:
        report_lines.append(f"\n{'Rank':<6} {'Strategy':<30} {'Val Sharpe':>12} {'Val Return%':>12} {'Val DD%':>12} {'Trades':>10}")
    report_lines.append("-" * 100)
    
    for i, (name, data) in enumerate(primary_sorted[:10], 1):
        val = data['validation']
        if sort_by == "val_reward":
            report_lines.append(
                f"{i:<6} {name:<30} {val['reward']:>12.6f} {val['sharpe']:>12.4f} {val['return']:>12.1f} "
                f"{val['drawdown']:>12.1f} {val['trades']:>10}"
            )
        else:
            report_lines.append(
                f"{i:<6} {name:<30} {val['sharpe']:>12.4f} {val['return']:>12.1f} "
                f"{val['drawdown']:>12.1f} {val['trades']:>10}"
            )
    
    # ===== TOP PERFORMERS BY ROBUSTNESS =====
    report_lines.append("\n\n" + "=" * 100)
    report_lines.append("TOP 10 STRATEGIES BY ROBUSTNESS (LOW OVERFITTING)")
    report_lines.append("=" * 100)
    report_lines.append(f"\n{'Rank':<6} {'Strategy':<30} {'Robustness':>12} {'Val Sharpe':>12} {'Overfit%':>12}")
    report_lines.append("-" * 100)
    
    for i, (name, robustness) in enumerate(sorted_by_robustness[:10], 1):
        data = strategies[name]
        val_sharpe = data['validation']['sharpe']
        train_sharpe = data['train']['sharpe']
        overfitting = calculate_overfitting_score(train_sharpe, val_sharpe)
        
        report_lines.append(
            f"{i:<6} {name:<30} {robustness:>12.4f} {val_sharpe:>12.4f} {overfitting:>12.1f}%"
        )
    
    # ===== OVERFITTING ANALYSIS =====
    report_lines.append("\n\n" + "=" * 100)
    report_lines.append("OVERFITTING ANALYSIS")
    report_lines.append("=" * 100)
    
    # Calculate overfitting metrics
    overfitting_sharpe = {
        name: calculate_overfitting_score(data['train']['sharpe'], data['validation']['sharpe'])
        for name, data in strategies.items()
    }
    
    # Categorize strategies
    severe_overfit = {k: v for k, v in overfitting_sharpe.items() if v > 50}
    moderate_overfit = {k: v for k, v in overfitting_sharpe.items() if 20 < v <= 50}
    mild_overfit = {k: v for k, v in overfitting_sharpe.items() if 0 < v <= 20}
    improved = {k: v for k, v in overfitting_sharpe.items() if v < 0}
    
    report_lines.append(f"\nSevere Overfitting (>50% degradation): {len(severe_overfit)} strategies")
    for name in sorted(severe_overfit.keys(), key=lambda x: severe_overfit[x], reverse=True)[:5]:
        report_lines.append(f"  - {name}: {severe_overfit[name]:.1f}%")
    
    report_lines.append(f"\nModerate Overfitting (20-50% degradation): {len(moderate_overfit)} strategies")
    report_lines.append(f"Mild Overfitting (0-20% degradation): {len(mild_overfit)} strategies")
    report_lines.append(f"Improved on Validation: {len(improved)} strategies")
    
    if improved:
        report_lines.append("\nStrategies that IMPROVED on validation:")
        for name in sorted(improved.keys(), key=lambda x: improved[x]):
            report_lines.append(f"  - {name}: {improved[name]:.1f}% (validation better!)")
    
    # ===== DETAILED BEST STRATEGY =====
    best_name, best_data = primary_sorted[0]
    report_lines.append("\n\n" + "=" * 100)
    report_lines.append(f"RECOMMENDED STRATEGY: {best_name}")
    report_lines.append("=" * 100)
    
    report_lines.append("\nOptimized Parameters:")
    for param, value in best_data['params'].items():
        report_lines.append(f"  {param}: {value}")
    
    report_lines.append("\nTraining Performance:")
    train = best_data['train']
    report_lines.append(f"  Return: {train['return']:.2f}%")
    report_lines.append(f"  Sharpe Ratio: {train['sharpe']:.4f}")
    report_lines.append(f"  Sortino Ratio: {train['sortino']:.4f}")
    report_lines.append(f"  Calmar Ratio: {train['calmar']:.4f}")
    report_lines.append(f"  Max Drawdown: {train['drawdown']:.2f}%")
    report_lines.append(f"  Trades: {train['trades']}")
    # Training reward (may be floored logic not applied here)
    report_lines.append(f"  Reward Metric: {train['reward']:.6f}")
    
    report_lines.append("\nValidation Performance:")
    val = best_data['validation']
    report_lines.append(f"  Return: {val['return']:.2f}%")
    report_lines.append(f"  Sharpe Ratio: {val['sharpe']:.4f}")
    report_lines.append(f"  Sortino Ratio: {val['sortino']:.4f}")
    report_lines.append(f"  Calmar Ratio: {val['calmar']:.4f}")
    report_lines.append(f"  Max Drawdown: {val['drawdown']:.2f}%")
    report_lines.append(f"  Trades: {val['trades']}")
    if 'reward_floor_applied' in val and val.get('reward_floor_applied'):
        rr = val.get('reward_reason', 'non_finite')
        raw = val.get('reward_raw')
        report_lines.append(f"  Reward Metric: {val['reward']:.6f} (floored; raw={raw} reason={rr})")
    else:
        report_lines.append(f"  Reward Metric: {val['reward']:.6f}")
    
    overfitting = calculate_overfitting_score(train['sharpe'], val['sharpe'])
    report_lines.append(f"\nOverfitting Score: {overfitting:.1f}%")
    if overfitting < 0:
        report_lines.append("  ✓ Strategy IMPROVED on unseen data (excellent generalization!)")
    elif overfitting < 20:
        report_lines.append("  ✓ Minimal overfitting (good generalization)")
    elif overfitting < 50:
        report_lines.append("  ⚠ Moderate overfitting (acceptable but monitor closely)")
    else:
        report_lines.append("  ✗ Severe overfitting (consider retraining or regularization)")
    
    # ===== RECOMMENDATIONS =====
    report_lines.append("\n\n" + "=" * 100)
    report_lines.append("RECOMMENDATIONS & NEXT STEPS")
    report_lines.append("=" * 100)
    
    # Get top 3 by robustness
    top_robust = sorted_by_robustness[:3]
    
    report_lines.append("\n1. IMMEDIATE DEPLOYMENT:")
    report_lines.append(f"   Primary: {best_name}")
    if sort_by == 'val_reward':
        report_lines.append(f"   - Highest validation Reward: {best_data['validation']['reward']:.6f}")
    else:
        report_lines.append(f"   - Highest validation Sharpe: {best_data['validation']['sharpe']:.4f}")
    report_lines.append(f"   - Run full backtest: ./run_full_backtest.sh {best_name}")
    
    report_lines.append("\n2. ALTERNATIVE CANDIDATES:")
    for i, (name, _) in enumerate(top_robust[:3], 1):
        data = strategies[name]
        report_lines.append(f"   {i}. {name}")
        report_lines.append(f"      - Validation Sharpe: {data['validation']['sharpe']:.4f}")
        report_lines.append(f"      - Robustness Score: {robustness_scores[name]:.4f}")
    
    report_lines.append("\n3. STRATEGIES REQUIRING ATTENTION:")
    if severe_overfit:
        report_lines.append("   The following strategies show severe overfitting:")
        for name in list(severe_overfit.keys())[:5]:
            report_lines.append(f"   - {name}: {severe_overfit[name]:.1f}% degradation")
        report_lines.append("   Consider: Re-optimization with more regularization or larger validation set")
    else:
        report_lines.append("   ✓ No strategies with severe overfitting detected")
    
    report_lines.append("\n4. NEXT STEPS:")
    report_lines.append("   a) Run full dataset backtests on top 3 strategies:")
    report_lines.append(f"      ./run_full_backtest.sh {best_name}")
    for name, _ in sorted_by_robustness[1:3]:
        report_lines.append(f"      ./run_full_backtest.sh {name}")
    
    report_lines.append("\n   b) Compare full backtest results:")
    report_lines.append("      python scripts/compare_strategies.py")
    
    report_lines.append("\n   c) Consider ensemble strategies:")
    report_lines.append("      Combine top performers with different characteristics")
    report_lines.append("      (e.g., trend-following + mean-reversion)")
    
    report_lines.append("\n   d) Monitor live performance:")
    report_lines.append("      Start with paper trading to validate real-world performance")
    
    report_lines.append("\n" + "=" * 100)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 100 + "\n")
    
    return "\n".join(report_lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare Bayesian optimization results across all strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        help='Output file for the report (default: print to stdout)'
    )
    parser.add_argument(
        '--results-dir',
        type=Path,
        default=Path('results'),
        help='Directory containing strategy results (default: results/)'
    )
    parser.add_argument(
        '--subdir',
        choices=['optimizations', 'evaluations', 'auto'],
        default='optimizations',
        help='Which subdirectory to read per-strategy results from (default: optimizations). Use auto to try optimizations then evaluations.'
    )
    parser.add_argument(
        '--sort-by',
        choices=['val_sharpe', 'val_reward'],
        default='val_sharpe',
        help='Primary sort for the report (default: val_sharpe)'
    )
    
    args = parser.parse_args()
    
    # Load all Bayesian results
    print("Loading Bayesian optimization results...", file=sys.stderr)
    strategies = load_bayesian_results(args.results_dir, subdir=args.subdir)
    
    if not strategies:
        print("\nNo strategy results found!", file=sys.stderr)
        print("Expected to find one of:", file=sys.stderr)
        print("  results/<strategy>/optimizations/bayesian_wf_optimization_results_validation.json", file=sys.stderr)
        print("  results/<strategy>/optimizations/bayesian_optimization_results_validation.json", file=sys.stderr)
        print("  results/<strategy>/evaluations/full_dataset_backtest.json", file=sys.stderr)
        print("  results/<strategy>/evaluations/eval_results.json", file=sys.stderr)
        print("Run optimizations first with:", file=sys.stderr)
        print("  ./optimize_bayesian_wf_batch.sh", file=sys.stderr)
        sys.exit(1)
    
    print(f"Found {len(strategies)} strategies with Bayesian results", file=sys.stderr)
    
    # Generate report
    report = generate_report(strategies, sort_by=args.sort_by)
    
    # Output report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}", file=sys.stderr)
    else:
        print(report)


if __name__ == "__main__":
    main()
