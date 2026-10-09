"""Plots (needs matplotlib)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .metrics import rolling_sharpe


def plot_results(results: dict, path: Path, rf: pd.Series | None = None, window: int = 63) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    for name, res in results.items():
        a1.plot(res.nav.index, res.nav.values - 1.0, label=name)
        a2.plot(res.returns.index, rolling_sharpe(res.returns, rf, window), label=name)
    a1.set_title("Cumulative return (after trading costs)")
    a1.axhline(0, color="grey", lw=0.5)
    a1.legend(fontsize=8)
    a2.set_title(f"Rolling Sharpe ratio ({window}-day window)")
    a2.axhline(0, color="grey", lw=0.5)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
