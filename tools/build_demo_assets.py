"""Draw saved-result and synthetic-attribution figures without competition assets."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "ship-xai-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "runtime" / "src"))
from explainer import custom_explain_func
from smoke import LiteralLinearLogits, literal_inputs, run_toy
from check_saved_results import check_readme, comparisons


def main() -> None:
    output = ROOT / "docs" / "assets"
    output.mkdir(parents=True, exist_ok=True)
    results = comparisons()
    check_readme(results)
    run_toy(torch.device("cpu"))
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(8.4, 4.8), layout="constrained")
    x = np.arange(len(results))
    for offset, key, label, color in ((-0.18, "control", "8 x 8 control", "#94a3b8"),
                                      (0.18, "candidate", "4 x 4 soft15", "#2563eb")):
        bars = ax.bar(x + offset, [r[key] for r in results], 0.32, label=label, color=color)
        ax.bar_label(bars, fmt="%.2f", padding=4)
    ax.set(xticks=x, xticklabels=[f"Seed {r['seed']}" for r in results],
           ylabel="Mean formula score", ylim=(0, 100),
           title="Paired comparison on the same nine images")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(axis="y", alpha=0.15)
    ax.set_axisbelow(True)
    for i, result in enumerate(results):
        ax.text(i, 93, f"+{result['difference']:.2f} points | 9/9 improved", ha="center", fontsize=10)
    fig.savefig(output / "paired-results.png", dpi=160)
    plt.close(fig)

    inputs, targets = literal_inputs()
    model = LiteralLinearLogits()
    traces = []
    final = custom_explain_func(model, inputs, targets, "cpu", trace_hook=traces.append)
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.8), layout="constrained")
    for row in range(2):
        arrays = (inputs[row, 2], traces[row]["raw"][0, 2], final[row, 2])
        for col, array in enumerate(arrays):
            ax = axes[row, col]
            bound = max(float(np.abs(array).max()), 1e-6)
            img = ax.imshow(array, cmap="RdBu_r", vmin=-bound, vmax=bound, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(("Synthetic input", "Signed raw attribution", "After soft15")[col])
            if col == 0:
                ax.set_ylabel(f"Target {int(targets[row])}\nArgmax 4", fontsize=11)
            fig.colorbar(img, ax=ax, shrink=0.75)
    fig.suptitle("Linear-model check: channel 2 of a 3 x 8 x 8 input", fontsize=13)
    fig.savefig(output / "synthetic-attribution.png", dpi=160)
    plt.close(fig)
    print("Saved docs/assets/paired-results.png and docs/assets/synthetic-attribution.png")


if __name__ == "__main__":
    main()
