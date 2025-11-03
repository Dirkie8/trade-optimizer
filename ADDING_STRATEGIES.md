# Adding New Strategies - Quick Guide

## Option 2 Implementation - No __init__.py Maintenance Required!

### How It Works

The optimizer now uses **dynamic imports** based on your YAML config. The `strategy.class` and `strategy.module` fields in your config tell the optimizer exactly which class to load.

### Adding a New Strategy (3 Simple Steps)

#### Step 1: Create Your Strategy File in `functions/strategies/`

Create a new file `functions/strategies/my_new_strategy.py`:

```python
from functions.base_strategy import BaseStrategy, Action
from typing import Tuple, Optional
import pandas as pd

class MyNewStrategy(BaseStrategy):
    """Your strategy description."""
    
    def generate_signals(self) -> Tuple[Action, Optional[float], Optional[float]]:
        # Get parameters from config
        param1 = self.config["param1"]
        param2 = self.config["param2"]
        tp_pips = float(self.config["take_profit_pips"])
        sl_pips = float(self.config["stop_loss_pips"])
        
        # Your strategy logic here
        # ...
        
        if buy_condition:
            return "BUY", sl_pips, tp_pips
        elif sell_condition:
            return "SELL", sl_pips, tp_pips
        else:
            return "HOLD", None, None
```

#### Step 2: Create Config File `functions/configs/my_new_strategy.yaml`

```yaml
strategy:
  class: "MyNewStrategy"                         # Must match class name exactly
  module: "functions.strategies.my_new_strategy" # Module path (without .py)
  results_dir: "MyNewStrategy"                   # Optional; defaults to class name

name: "My New Strategy Display Name"

parameters:
  param1: [10, 20, 30]
  param2: [0.5, 1.0, 1.5]
  take_profit_pips: [20, 40]
  stop_loss_pips: [15, 30]
```

#### Step 3: Run Optimization

```bash
python scripts/optimize_strategy.py \
    --strategy_config functions/configs/my_new_strategy.yaml \
    --main_config configs/main_config.yaml \
    --method grid \
    --n_jobs 4
```

**That's it!** No need to touch `__init__.py` or any other files.

---

## Directory Structure

Strategies are organized in the `functions/strategies/` folder:

```
functions/
  strategies/              # All strategy implementations
    __init__.py
    moving_average_cross.py
    my_rsi_strategy.py
    my_trend_strategy.py
  configs/                 # Strategy configurations
    example_strategy.yaml
    my_rsi_strategy.yaml
  _strategy_template.py           # Detailed template (in main functions/ folder)
  _strategy_template_simplified.py # Quick template (in main functions/ folder)
        pass
```

### Step 2: Update Config to Point to New Module

`functions/configs/trend_follower.yaml`:
```yaml
strategy:
  class: "TrendFollower"
  module: "functions.trend_strategies"  # ← Point to your new file
  results_dir: "TrendFollower"  # Optional; keeps results organized by strategy

name: "Trend Following Strategy"
parameters:
  # ...
```

### Step 3: Run It

```bash
python scripts/optimize_strategy.py \
    --strategy_config functions/configs/trend_follower.yaml \
    --main_config configs/main_config.yaml \
    --method random \
    --n_random 50 \
    --n_jobs 4
```

---

## Results Organization

Each strategy's results are now saved in a dedicated subdirectory under `results/`:

```
results/
├── MovingAverageCross/        # ← Strategy-specific folder
│   ├── optimizations/
│   │   ├── optimization_results.csv
│   │   └── optimization_results_best.json
│   └── evaluations/
│       ├── evaluation_results.csv
│       ├── evaluation_results_best.json
│       └── eval_results.json
├── RSIStrategy/               # ← Another strategy's results
│   ├── optimizations/
│   └── evaluations/
└── TrendFollower/             # ← And another...
    ├── optimizations/
    └── evaluations/
```

### Configuring Results Directory

Set the `results_dir` in your strategy config:

```yaml
strategy:
  class: "MyStrategy"
  results_dir: "MyStrategy_v2"  # Custom directory name
```

**Default behavior**: If `results_dir` is omitted, it defaults to the class name.

### Benefits

- **Clean separation**: Each strategy's results are isolated
- **Easy comparison**: Compare different strategies side-by-side
- **Version control**: Test strategy variations with different `results_dir` names
- **No conflicts**: Multiple strategies can run without overwriting each other's results

---

## What Changed?

### Before (Old Method)
- Had to manually add imports to `functions/__init__.py` for every new strategy
- Optimizer hardcoded strategy lookup in a `STRATEGIES` dict
- All results saved to same `results/` directory (easy to mix up strategies)

### After (Option 2 - Current)
- ✅ No `__init__.py` maintenance
- ✅ Strategy class dynamically loaded from YAML config
- ✅ Can organize strategies however you want (single file, multiple files, subdirectories)
- ✅ YAML config explicitly shows module path
- ✅ Each strategy has its own results directory

---

## Files Modified

1. **`scripts/optimize_strategy.py`**: Now uses `importlib` to load strategy class dynamically
2. **`functions/strategies.py`**: Created as central location for all strategies
3. **`functions/configs/example_strategy.yaml`**: Updated to include `module` field
4. **`functions/configs/_template_new_strategy.yaml`**: Template for new strategies

---

## Template Files Provided

- **`functions/configs/_template_new_strategy.yaml`**: Copy and customize for new strategies
- **`functions/strategies.py`**: Add new strategy classes here (commented example included)

---

## Tips

1. **Keep it simple**: Start by adding all strategies to `functions/strategies.py`
2. **Scale later**: If you have many strategies, organize into multiple files
3. **Use descriptive names**: Class names should be clear (e.g., `RSIMeanReversion`, `BollingerBreakout`)
4. **Test incrementally**: Run a small optimization (`--n_random 4`) to verify your new strategy works before full grid search

---

## Example: Complete Workflow

```bash
# 1. Edit functions/strategies.py - add your new class
# 2. Copy template
cp functions/configs/_template_new_strategy.yaml functions/configs/rsi_strategy.yaml

# 3. Edit rsi_strategy.yaml - set class name and parameters
# 4. Run optimization
python scripts/optimize_strategy.py \
    --strategy_config functions/configs/rsi_strategy.yaml \
    --main_config configs/main_config.yaml \
    --method random \
    --n_random 20 \
    --n_jobs 4

# 5. Results automatically saved to results/
```

Enjoy zero-maintenance strategy development! 🚀
