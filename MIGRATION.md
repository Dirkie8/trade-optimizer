# Strategy File Organization - MIGRATION NOTE

## What Changed?

Strategies have been reorganized for better structure:

### OLD Structure:
```
functions/
  strategies.py              # All strategies in one file
  strategy_example.py        # Example strategy
```

### NEW Structure:
```
functions/
  strategies/                # Each strategy in its own file
    __init__.py
    moving_average_cross.py
    your_strategy.py
  _strategy_template.py           # Templates stay at top level
  _strategy_template_simplified.py
```

## Migration Guide

**Old files (deprecated):**
- `functions/strategies.py` - No longer used (strategies moved to strategies/ folder)
- `functions/strategy_example.py` - Replaced by `strategies/moving_average_cross.py`

**To migrate existing strategies:**
1. Move your strategy class from `strategies.py` to `strategies/your_strategy.py`
2. Update your config's `module` field to: `"functions.strategies.your_strategy"`
3. Keep the same `class` name

**Example config update:**
```yaml
# OLD
strategy:
  class: "MyStrategy"
  module: "functions.strategies"

# NEW
strategy:
  class: "MyStrategy"
  module: "functions.strategies.my_strategy"
```

## Benefits

- ✅ Each strategy in its own file (easier to manage)
- ✅ Configs in `configs/` folder (parallel structure)
- ✅ Results in `results/{strategy_name}/` (organized by strategy)
- ✅ Templates at top level (easy to find and copy)
- ✅ No more giant single file with all strategies

## Quick Start

See the simplified templates for quick strategy creation:
- `functions/_strategy_template_simplified.py`
- `functions/configs/_template_new_strategy_simplified.yaml`
