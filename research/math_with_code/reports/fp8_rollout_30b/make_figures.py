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
"""Render the BF16-vs-FP8-rollout report figures from the pulled CSVs."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path(__file__).parent / "data"
FIG_DIR = Path(__file__).parent / "figures"

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e8e7e3"
CHAINS = [  # fixed categorical order: slot1 blue, slot2 green, slot3 magenta
    ("bf16", "BF16", "#2a78d6"),
    ("fp8_v3", "FP8 v3", "#008300"),
    ("fp8_v4", "FP8 v4 (router bf16)", "#e87ba4"),
]
FIGURES = [
    # (column, filename, y-label, title, rolling-window)
    ("validation/accuracy", "val_accuracy", "AIME 2025 accuracy (30 tasks x 16)",
     "Validation accuracy", 1),
    ("train/approx_entropy", "entropy", "approx. policy entropy",
     "Policy entropy (rolling median, w=9)", 9),
    ("train/gen_kl_error", "gen_kl_error", "KL(train logprobs ‖ gen logprobs)",
     "Generation logprob bias (rolling median, w=9)", 9),
    ("train/math_code_harbor_agent/num_tool_calls/mean", "tool_calls",
     "tool calls per rollout (mean)",
     "Tool use per rollout (rolling median, w=9)", 9),
]


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9)


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    frames = {alias: pd.read_csv(DATA_DIR / f"{alias}.csv") for alias, _, _ in CHAINS}

    for column, stem, ylabel, title, window in FIGURES:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
        fig.patch.set_facecolor(SURFACE)
        style_axes(ax)
        for alias, label, color in CHAINS:
            df = frames[alias][["step", column]].dropna()
            if df.empty:
                continue
            if window > 1:
                ax.plot(df["step"], df[column], color=color, linewidth=0.8, alpha=0.25)
                smooth = df[column].rolling(window, center=True, min_periods=3).median()
                ax.plot(df["step"], smooth, color=color, linewidth=2, label=label)
                y_end = smooth.dropna().iloc[-1]
            else:
                ax.plot(df["step"], df[column], color=color, linewidth=2,
                        marker="o", markersize=4, label=label)
                y_end = df[column].iloc[-1]
            ax.annotate(label, (df["step"].iloc[-1], y_end),
                        xytext=(6, 0), textcoords="offset points",
                        color=color, fontsize=9, va="center")
        ax.set_xlabel("training step", color=TEXT_SECONDARY, fontsize=10)
        ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=12, loc="left")
        ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY)
        ax.margins(x=0.12)
        fig.tight_layout()
        out = FIG_DIR / f"{stem}.png"
        fig.savefig(out, facecolor=SURFACE)
        plt.close(fig)
        print(f"wrote {out}")

    # Throughput / accuracy summary for the report tables.
    print("\n| chain | best val acc | last val acc (step) | median step time (s) "
          "| median exposed gen (s) | median gen tok/s |")
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
