# Packaging `quantstream`

This guide explains how the `quantstream` Python package is built, checked,
published, and installed — the same workflow used by real open-source
libraries.

## What the artifact is

`python -m build` produces two artifacts in `dist/`:

- **sdist** (`quantstream-0.1.0.tar.gz`) — the source distribution: all
  Python sources plus `pyproject.toml` and `README.md`. Anyone can rebuild
  from it.
- **wheel** (`quantstream-0.1.0-py3-none-any.whl`) — the built distribution:
  a zip of the `quantstream/` package ready to drop onto `sys.path`. This is
  what `pip install` downloads and installs in seconds, no build step needed.

Both are versioned (`0.1.0`) so users can pin exact releases.

## Prerequisites

```bash
pip install build twine
# or: pip install -e ".[dev]"
```

## How to build

```bash
python -m build
# → dist/quantstream-0.1.0.tar.gz
# → dist/quantstream-0.1.0-py3-none-any.whl
```

## How to check

```bash
twine check dist/*
```

This validates the README renders correctly on PyPI and the metadata is
well-formed. Fix any warnings before publishing.

## How to publish

Publishing uses an API token — never a password, and never commit the token
anywhere. Create one at <https://pypi.org/manage/account/token/> (scoped to
the `quantstream` project).

**1. TestPyPI first** (always — this is the dress rehearsal):

```bash
twine upload --repository testpypi dist/*
```

Install from TestPyPI in a fresh venv to verify end to end:

```bash
pip install --index-url https://test.pypi.org/simple/ quantstream
quantstream-backtest --help
```

**2. Then the real PyPI:**

```bash
twine upload dist/*
```

## How end users install and use it

```bash
pip install quantstream
```

```python
import pandas as pd
from quantstream import momentum_signals, backtest, Settings

df = pd.read_csv("ticks.csv")          # needs a 'price' column
signals = momentum_signals(df, fast=10, slow=30)
result = backtest(signals, commission_bps=2.0, slippage_bps=1.0)
print(result)
# {'total_return': ..., 'sharpe': ..., 'max_drawdown': ...,
#  'n_trades': ..., 'final_equity': ...}

# Or from the command line:
# quantstream-produce --rate 2 --duration 60
# quantstream-backtest --csv data/sample_ticks.csv
```

## Versioning policy

This project follows [semantic versioning](https://semver.org/):

- **MAJOR** — breaking API changes (renamed functions, changed schemas)
- **MINOR** — new features, backwards compatible (new signals, new models)
- **PATCH** — bug fixes, docs, no API change

To cut a new release:

1. Bump `version` in `pyproject.toml` and `__version__` in
   `quantstream/__init__.py` (keep them in sync).
2. Update the changelog section in `README.md` if one exists.
3. `python -m build && twine check dist/*`
4. Upload to TestPyPI, smoke-test, then upload to PyPI.
5. Tag the release in git: `git tag v0.2.0 && git push --tags`.
