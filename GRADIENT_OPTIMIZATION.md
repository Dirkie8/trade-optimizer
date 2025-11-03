# Gradient-Based Optimization with Optuna

## Overview

We've added **TPE (Tree-structured Parzen Estimator)** optimization using Optuna - a gradient-based/Bayesian approach that's much smarter than random search!

## 🎯 Three Optimization Methods

### 1. **Grid Search** (Exhaustive)
```bash
python scripts/optimize_strategy.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --method grid
```
- **How it works**: Tests every possible parameter combination
- **Pros**: Guaranteed to find global optimum
- **Cons**: VERY slow (800+ combinations for RSI)
- **When to use**: Small parameter spaces only

### 2. **Random Search** (Exploration)
```bash
python scripts/optimize_strategy.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --method random \
  --n_random 50
```
- **How it works**: Randomly samples N parameter combinations
- **Pros**: Fast, good for exploration
- **Cons**: Might miss optimal regions, no learning
- **When to use**: Initial exploration, large parameter spaces

### 3. **TPE/Bayesian** (Gradient-Based) ✨ NEW!
```bash
python scripts/optimize_strategy.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --method tpe \
  --n_trials 50 \
  --sort-by consistency_score
```
- **How it works**: Learns from previous trials, focuses on promising regions
- **Pros**: Smarter than random, converges faster, uses gradient information
- **Cons**: Can get stuck in local optima
- **When to use**: Refinement, when you want efficient optimization

## 🔥 Hybrid Strategy (RECOMMENDED!)

Combine random + TPE for best results:

### Option A: Sequential Script
```bash
# Phase 1: Explore with random search
python scripts/optimize_strategy.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --method random \
  --n_random 30 \
  --walk-forward \
  --n-folds 3

# Phase 2: Refine with TPE
python scripts/optimize_strategy.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --method tpe \
  --n_trials 50 \
  --walk-forward \
  --n-folds 3
```

### Option B: Automated Hybrid Script
```bash
python scripts/hybrid_optimize.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --random_samples 30 \
  --tpe_trials 50 \
  --walk-forward
```

## 📊 How TPE Works (Simple Explanation)

### Random Search (Dumb):
```
Trial 1: {rsi_period: 9, oversold: 20}  → Score: 12.5
Trial 2: {rsi_period: 28, oversold: 35} → Score: 15.2
Trial 3: {rsi_period: 14, oversold: 25} → Score: 17.8  ← Best so far
Trial 4: {rsi_period: 21, oversold: 30} → Score: 14.1
...completely random choices, no learning!
```

### TPE/Bayesian (Smart):
```
Trial 1: {rsi_period: 9, oversold: 20}  → Score: 12.5
Trial 2: {rsi_period: 28, oversold: 35} → Score: 15.2
Trial 3: {rsi_period: 14, oversold: 25} → Score: 17.8  ← Best so far!

🧠 TPE thinks: "Hmm, rsi_period=14 and oversold=25 worked well.
              Let me try nearby values..."

Trial 4: {rsi_period: 12, oversold: 23} → Score: 18.1  ← Even better!
Trial 5: {rsi_period: 16, oversold: 27} → Score: 17.9  ← Good!

🧠 TPE thinks: "The region around rsi_period=12-16, oversold=23-27 is promising.
              Let me focus my search there..."

Trial 6: {rsi_period: 13, oversold: 24} → Score: 18.5  ← Best yet!
```

TPE uses **gradient information** to understand which direction improves the score!

## 🎓 Technical Details

### What is TPE?
- **Tree-structured Parzen Estimator**
- A Bayesian optimization algorithm
- Builds probability models of good vs bad parameter regions
- Uses past trials to guide future parameter selection

### Key Advantages:
1. **Converges faster** than random search (needs fewer trials)
2. **Learns from history** - doesn't waste time on known bad regions
3. **Balances exploration vs exploitation** automatically
4. **Handles mixed parameter types** (categorical + continuous)

### Mathematical Intuition:
```python
# TPE maintains two probability distributions:
P(x | y > threshold)  # Distribution of good parameters
P(x | y ≤ threshold)  # Distribution of bad parameters

# Next trial samples from regions where:
ratio = P(x | y > threshold) / P(x | y ≤ threshold)
# is maximized (high chance of being good, low chance of being bad)
```

## 📈 Performance Comparison

Based on typical hyperparameter optimization benchmarks:

| Method | Trials to 95% Optimal | Relative Speed |
|--------|----------------------|----------------|
| Grid Search | 800 (all combinations) | 1x (baseline) |
| Random Search | ~100-150 trials | ~5-8x faster |
| TPE (Bayesian) | ~30-50 trials | **~15-25x faster** |

**Example**: Finding good RSI parameters
- Grid: Test all 800 combinations → 2 hours
- Random (100 samples): → 15 minutes
- TPE (50 trials): → **5-7 minutes** ✨

## 🚀 Quick Start Examples

### Example 1: Quick TPE Optimization
```bash
# Fast gradient-based optimization (5-10 minutes)
python scripts/optimize_strategy.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --method tpe \
  --n_trials 30 \
  --sort-by consistency_score
```

### Example 2: Robust TPE with Walk-Forward
```bash
# More reliable but slower (15-20 minutes)
python scripts/optimize_strategy.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --method tpe \
  --n_trials 50 \
  --walk-forward \
  --n-folds 3 \
  --sort-by consistency_score
```

### Example 3: Best Practice Hybrid
```bash
# Exploration + Exploitation (20-25 minutes total)
python scripts/hybrid_optimize.py \
  --strategy_config functions/configs/rsi_strategy.yaml \
  --main_config configs/main_config.yaml \
  --random_samples 20 \
  --tpe_trials 40 \
  --walk-forward
```

## 🎯 When to Use What?

### Use **Random Search** when:
- ✓ First time exploring a new strategy
- ✓ Very large parameter space
- ✓ Quick and dirty optimization
- ✓ Want diverse parameter sets

### Use **TPE/Bayesian** when:
- ✓ Refining parameters after initial exploration
- ✓ Want efficient optimization
- ✓ Limited compute time
- ✓ Parameters have smooth relationships to performance

### Use **Hybrid** when:
- ✓ Want best of both worlds
- ✓ Production use cases
- ✓ Have time for proper optimization
- ✓ Maximum reliability needed

## 🔬 Advanced: Understanding the Algorithm

### TPE Algorithm Steps:

1. **Initialize**: Run some random trials (warmup phase)
2. **Split History**: Divide trials into "good" (top 20%) and "bad" (bottom 80%)
3. **Model Building**:
   - Build probability model of parameters in "good" trials: `l(x)`
   - Build probability model of parameters in "bad" trials: `g(x)`
4. **Acquisition Function**:
   - Calculate `l(x) / g(x)` for each possible parameter set
   - Choose parameter set that maximizes this ratio
5. **Evaluate**: Run trial with chosen parameters
6. **Update**: Add result to history and repeat from step 2

### Visualization:
```
Score
  ^
  |              TPE focuses here ↓
18|           ●     ●     ●
  |        ●     ●     ●     ●
16|     ●     ●     ●     ●     ●
  |  ●     ●     ●     ●     ●     ●
14|●     ●     ●     ●     ●     ●     ●
  |  ●     ●     ●     ●     ●     ●
12|     ●     ●     ●     ●     ●
  |  ●     ●
10|●     ●
  +---------------------------------> Trials
   5    10    15    20    25    30

Random search: ● scattered everywhere
TPE: ● converge toward high-score region
```

## 📚 Additional Resources

- **Optuna Documentation**: https://optuna.org/
- **TPE Paper**: Bergstra et al. (2011) - "Algorithms for Hyper-Parameter Optimization"
- **Comparison Study**: https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html

## 🐛 Troubleshooting

### "Optuna not installed"
```bash
pip install optuna
# or
python -m pip install optuna
```

### TPE seems stuck
- Try increasing `--n_trials` (TPE needs more samples than you think)
- Check if parameters have smooth impact on score
- Consider adding more random warmup trials

### Poor results with TPE
- Try hybrid approach (random first, then TPE)
- Ensure parameter ranges aren't too narrow
- Check if walk-forward is enabled for reliable scores

## 🎊 Summary

**Key Takeaway**: TPE is like having a smart assistant that learns which parameter regions work well and focuses the search there, instead of blindly trying random combinations!

| Feature | Random | TPE |
|---------|--------|-----|
| Learning | ❌ No | ✅ Yes |
| Efficiency | Medium | ✅ High |
| Exploration | ✅ Excellent | Medium |
| Exploitation | Poor | ✅ Excellent |
| Best For | Initial search | Refinement |
