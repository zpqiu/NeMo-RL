#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Render the report figures in the low-precision-rl-tech-report house style.

Color convention follows that report's figures (validated for CVD):
orange = BF16 reference, blue = FP8 rollout (recommended config, router BF16),
green = ablation arm (router quantized to FP8).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
FIG_DIR = Path(__file__).parent / "figures"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e8e7e3"
CHAINS = [  # tech-report convention: orange BF16, blue FP8(main), green ablation
    ("bf16", "BF16", "#ff7f0e"),
    ("fp8_v4", "FP8 (router BF16)", "#1f77b4"),
    ("fp8_v3", "FP8 (router FP8)", "#2ca02c"),
]
DYNAMICS_PANELS = [
    # (column, y-label, panel title, rolling-window)
    ("validation/accuracy", "accuracy", "Validation accuracy (AIME 2025)", 1),
    ("train/reward", "reward", "Training reward (rolling median, w=9)", 9),
    ("train/mean_gen_tokens_per_sample", "tokens / sample",
     "Response length (rolling median, w=9)", 9),
    ("train/gen_kl_error", "KL", "Mismatch KL (rolling median, w=9)", 9),
]
ABLATION_PANELS = [
    ("train/approx_entropy", "approx. entropy",
     "Policy entropy (rolling median, w=9)", 9),
    ("train/math_code_harbor_agent/num_tool_calls/mean", "tool calls / rollout",
     "Tool use (rolling median, w=9)", 9),
]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8)


def plot_panel(ax, frames, column, ylabel, title, window):
    style_axes(ax)
    for alias, label, color in CHAINS:
        df = frames[alias][["step", column]].dropna()
        if df.empty:
            continue
        if window > 1:
            ax.plot(df["step"], df[column], color=color, linewidth=0.7, alpha=0.22)
            smooth = df[column].rolling(window, center=True, min_periods=3).median()
            ax.plot(df["step"], smooth, color=color, linewidth=1.8, label=label)
        else:
            ax.plot(df["step"], df[column], color=color, linewidth=1.8,
                    marker="o", markersize=3.2, label=label)
    ax.set_xlabel("training step", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=9)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=10, loc="left")


def render(frames, panels, ncols, out_name, figsize):
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (column, ylabel, title, window) in zip(flat, panels):
        plot_panel(ax, frames, column, ylabel, title, window)
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(CHAINS),
               frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = FIG_DIR / out_name
    fig.savefig(out, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    frames = {alias: pd.read_csv(DATA_DIR / f"{alias}.csv") for alias, _, _ in CHAINS}

    render(frames, DYNAMICS_PANELS, 2, "training_dynamics.png", (9.6, 6.4))
    render(frames, ABLATION_PANELS, 2, "ablation_router.png", (9.6, 3.6))

    print("\n| configuration | best val acc | last val acc (step) | median step "
          "time (s) | median exposed gen (s) | median gen tok/s |")
    print("|---|---|---|---|---|---|")
    for alias, label, _ in CHAINS:
        df = frames[alias]
        acc = df[["step", "validation/accuracy"]].dropna()
        best = acc["validation/accuracy"].max()
        last = acc.iloc[-1]
        med = lambda c: df[c].dropna().median()
        print(f"| {label} | {best:.4f} | {last['validation/accuracy']:.4f} "
              f"(s{int(last['step'])}) | {med('timing/train/total_step_time'):.0f} "
              f"| {med('timing/train/exposed_generation'):.1f} "
              f"| {med('performance/generation_tokens_per_sec'):.0f} |")


if __name__ == "__main__":
    main()
