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
"""Render the eagle3 report figures in the low-precision-rl-tech-report style.

Style and metric definitions follow `../fp8_rollout_30b/make_figures.py` so the
two rollout-acceleration studies can be read side by side: orange = the BF16
reference (no speculation), blue = the recommended config (online-trained
drafter), green = the ablation arm (frozen drafter). Noisy per-step series are
smoothed with a centered rolling median (w=9).

Two deliberate additions. The frozen arm is drawn dashed: blue and green
separate at deutan ΔE 23.5 but only 6.8 under tritanopia, and here they are the
*primary* comparison rather than an ablation pair, so they carry a second,
non-color encoding. And the BF16 replicate shares BF16's hue with a dotted
stroke — it is the same entity sampled twice, and colour follows the entity.
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
# (alias, label, color, linestyle)
BF16 = ("bf16", "BF16 (no speculation)", "#ff7f0e", "-")
# Same entity as BF16, a second sample of it — so the same hue, distinguished by
# stroke rather than by a fourth colour.
BF16_REP = ("bf16_replicate", "BF16 replicate (same seed)", "#ff7f0e", ":")
ONLINE = ("online", "EAGLE-3, drafter trained", "#1f77b4", "-")
FROZEN = ("frozen", "EAGLE-3, drafter frozen", "#2ca02c", "--")

ACC = "train/vllm/spec_acceptance_length"
GEN_TOKENS = "train/math_code_harbor_agent/generated_tokens/mean"
GEN_SEC = "train/math_code_harbor_agent/model_generation_sec/mean"
DELTA = "acceptance_delta"

# The no-speculation arms log acceptance_length as a constant 1.0 (no drafts),
# which is true but flattens the panel's y-range, so the speculation panels are
# restricted to the arms that actually speculate.
SPEC_ONLY = {"online", "frozen"}
RESULTS_PANELS = [
    # (column, y-label, panel title, rolling-window, alias filter or None)
    (ACC, "tokens / draft", "Acceptance length", 9, SPEC_ONLY),
    (DELTA, "Δ tokens / draft", "Acceptance advantage (trained − frozen)", 9, SPEC_ONLY),
    ("validation/accuracy", "accuracy", "Validation accuracy (AIME 2025)", 1, None),
    ("train/reward", "reward", "Training reward", 9, None),
    ("train/approx_entropy", "approx. entropy", "Policy entropy", 9, None),
    ("train/turns_per_sample/mean", "turns / sample", "Tool-use depth", 9, None),
]
PERF_PANELS = [
    ("decode_ms_per_token", "ms / token", "Decode time per output token", 9, None),
    ("time_per_output_token_ms", "ms / token", "HTTP wall per output token", 9, None),
    ("gen_kl_error_x1000", "KL x1000", "Mismatch KL (rollout vs training)", 9, None),
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


def plot_panel(ax, frames, chains, column, ylabel, title, window, only=None):
    style_axes(ax)
    for alias, label, color, dash in chains:
        if only is not None and alias not in only:
            continue
        df = frames[alias]
        if column not in df.columns:
            continue
        series = df[["step", column]].dropna()
        if series.empty:
            continue
        if window > 1:
            ax.plot(series["step"], series[column], color=color, linewidth=0.7,
                    alpha=0.22, linestyle=dash)
            smooth = series[column].rolling(window, center=True, min_periods=3).median()
            ax.plot(series["step"], smooth, color=color, linewidth=1.8,
                    linestyle=dash, label=label)
        else:
            ax.plot(series["step"], series[column], color=color, linewidth=1.8,
                    linestyle=dash, marker="o", markersize=3.2, label=label)
    ax.set_xlabel("training step", color=TEXT_SECONDARY, fontsize=10)
    ax.set_ylabel(ylabel, color=TEXT_SECONDARY, fontsize=10)
    ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, loc="left")


def render(frames, chains, panels, ncols, out_name, figsize):
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=150)
    fig.patch.set_facecolor(SURFACE)
    flat = axes.flatten() if hasattr(axes, "flatten") else [axes]
    for ax, (column, ylabel, title, window, only) in zip(flat, panels):
        plot_panel(ax, frames, chains, column, ylabel, title, window, only)
    for ax in flat[len(panels):]:
        ax.set_visible(False)
    handles, labels = [], []
    for ax in flat[:len(panels)]:
        for handle, label in zip(*ax.get_legend_handles_labels()):
            if label not in labels:
                handles.append(handle)
                labels.append(label)
    fig.legend(handles, labels, loc="lower center", ncol=len(labels),
               frameon=False, fontsize=10, labelcolor=TEXT_SECONDARY)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    out = FIG_DIR / out_name
    fig.savefig(out, facecolor=SURFACE)
    preview = Path("/lustre/fsw/general_sa/alexq/tools/fig_preview")
    preview.mkdir(parents=True, exist_ok=True)
    fig.savefig(preview / (out.stem + ".png"), facecolor=SURFACE, dpi=170)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    frames = {}
    for alias, _, _, _ in (BF16, BF16_REP, ONLINE, FROZEN):
        df = pd.read_csv(DATA_DIR / f"{alias}.csv")
        # Time per output token (ms/token), the tech report's rollout-perf
        # metric: mean model-call wall seconds / mean generated tokens per
        # call (batch sum/sum identity). Wall clock is the HTTP call — vLLM
        # queue + prefill + decode — with tool execution excluded, so it
        # measures per-request serving speed at matched load.
        df["time_per_output_token_ms"] = 1000.0 * df[GEN_SEC] / df[GEN_TOKENS]
        # Decode-only cost per generated token: vLLM's per-request decode time
        # over tokens per request. The HTTP wall above accumulates queue, per-turn
        # round trips and prefill across every turn of a rollout, so it rises with
        # tool-use depth even when decode speed is flat — the frozen arm's jump to
        # ~14 ms/token after its step-195 transition is entirely that, its decode
        # cost is unchanged. The per-request metrics landed partway through the
        # original BF16 chain, so it only carries this series from step 227; the
        # replicate has it from step 1 and is the usable no-speculation baseline.
        if "train/vllm/request_decode_time_mean_s" in df.columns:
            tokens_per_call = (df[GEN_TOKENS]
                               / df["train/math_code_harbor_agent/num_model_calls/mean"])
            df["decode_ms_per_token"] = (
                1000.0 * df["train/vllm/request_decode_time_mean_s"] / tokens_per_call)
        # Mismatch KL is ~1.5e-3 in every arm; plot it scaled so the panel is
        # readable rather than a flat line at the axis floor.
        df["gen_kl_error_x1000"] = 1000.0 * df["train/gen_kl_error"]
        frames[alias] = df

    # Paired advantage, defined only where both arms logged the same step.
    paired = frames["online"][["step", ACC]].merge(
        frames["frozen"][["step", ACC]], on="step", suffixes=("_on", "_fr")
    ).dropna()
    paired[DELTA] = paired[f"{ACC}_on"] - paired[f"{ACC}_fr"]
    frames["online"] = frames["online"].merge(paired[["step", DELTA]], on="step",
                                              how="left")

    render(frames, (ONLINE, FROZEN, BF16, BF16_REP), RESULTS_PANELS, 3,
           "results_dynamics.svg", (12.8, 6.6))
    render(frames, (ONLINE, FROZEN, BF16, BF16_REP), PERF_PANELS, 3,
           "rollout_perf.svg", (14.0, 4.1))

    # Windowed medians for the report text (matched-load comparisons).
    print("\n| window | metric | trained | frozen | BF16 replicate |")
    print("|---|---|---|---|---|")
    for name, lo, hi in [("steps 10-60", 10, 60), ("steps 150-190", 150, 190),
                         ("steps 200-221", 200, 221)]:
        for col, label in (("decode_ms_per_token", "decode ms/token"),
                           ("time_per_output_token_ms", "HTTP ms/token")):
            cells = []
            for alias, _, _, _ in (ONLINE, FROZEN, BF16_REP):
                win = frames[alias]
                win = win[(win["step"] >= lo) & (win["step"] <= hi)]
                val = win[col].median() if col in win.columns else float("nan")
                cells.append(f"{val:.2f}" if val == val else "-")
            print(f"| {name} | {label} | {cells[0]} | {cells[1]} | {cells[2]} |")

    last = paired.tail(10)
    print(f"\nfinal 10 matched steps: trained {last[f'{ACC}_on'].mean():.3f} vs "
          f"frozen {last[f'{ACC}_fr'].mean():.3f} "
          f"(+{100 * last[DELTA].mean() / last[f'{ACC}_fr'].mean():.1f}%)")


if __name__ == "__main__":
    main()
