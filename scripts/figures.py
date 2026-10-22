"""Figures for the PEFT comparison from results/analysis.json + runs.json. Run: python scripts/figures.py"""
import json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

D = os.path.join(os.path.dirname(__file__), "..")
A = json.load(open(f"{D}/results/analysis.json"))
runs = [r for r in json.load(open(f"{D}/results/runs.json")) if r.get("status", "ok") == "ok"]
rows = A["rows"]; arms = [r["arm"] for r in rows]; labels = [r["method"] for r in rows]
COL = {"lora": "#2a78d6", "dora": "#eb6834", "qlora": "#1baf7a", "full": "#eda100"}  # validated categorical slots 1-4
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "axes.titleweight": "bold", "axes.edgecolor": INK2,
                     "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                     "axes.spines.right": False, "figure.facecolor": "white"})
SUB = f"Qwen3-4B on AG News, 4,000 train / 2,000 test, {len(A['seeds'])} seed{'s' if len(A['seeds']) > 1 else ''}, one NVIDIA L4"

def bars(ax, vals, fmt, err=None):
    x = np.arange(len(arms))
    ax.bar(x, vals, color=[COL[a] for a in arms], width=0.6, edgecolor="white", linewidth=2,
           yerr=err, capsize=6 if err is not None else 0, error_kw=dict(ecolor=INK2, lw=1.2))
    top = vals if err is None else vals + err[1]
    for i, v in enumerate(vals): ax.text(i, top[i], fmt(v), ha="center", va="bottom", color=INK, fontsize=11)
    ax.set_xticks(x, labels); ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)

# 1. accuracy with 95% CI
fig, ax = plt.subplots(figsize=(9, 5.5))
acc = np.array([r["acc_mean"] for r in rows])
err = np.array([[r["acc_mean"] - r["acc_ci_lo"] for r in rows], [r["acc_ci_hi"] - r["acc_mean"] for r in rows]])
bars(ax, acc, lambda v: f"{v:.2f}%\n", err)
lo = min(r["acc_ci_lo"] for r in rows)
ax.set_ylim(np.floor(lo - 1.5), min(100, np.ceil(max(r["acc_ci_hi"] for r in rows) + 1)))
ax.set_ylabel("Test accuracy (%)"); ax.set_title("All four methods land within 0.4 points of each other", loc="left")
fig.text(0.01, 0.01, f"{SUB}. Error bars: 95% bootstrap CI (seeds × test items).\nZero-shot: {A["zero_shot_acc"]:.1f}%. Y-axis truncated.",
         fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(f"{D}/figures/accuracy.png", dpi=150); plt.close(fig)

# 2. cost: peak VRAM and training minutes (two panels, never dual axis)
fig, axs = plt.subplots(1, 2, figsize=(12, 5.2))
bars(axs[0], np.array([r["peak_alloc_gb"] for r in rows]), lambda v: f"{v:.1f} GB")
axs[0].axhline(24, color=INK2, ls="--", lw=1); axs[0].text(len(arms) - 0.5, 24, "L4 = 24 GB", ha="right", va="bottom", color=INK2, fontsize=10)
axs[0].set_ylim(0, 27); axs[0].set_ylabel("Peak GPU memory during training (GB)"); axs[0].set_title("Peak VRAM", loc="left")
bars(axs[1], np.array([r["train_min"] for r in rows]), lambda v: f"{v:.1f} min")
axs[1].set_ylabel("Training wall-clock (minutes, 250 steps)"); axs[1].set_title("Training time", loc="left")
for a in axs: a.tick_params(axis="x", labelsize=10)
fig.suptitle("What each method costs on one L4", x=0.01, ha="left", fontweight="bold", fontsize=15)
fig.text(0.01, 0.01, f"{SUB}. Mean over seeds; torch max_memory_allocated.", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 0.95)); fig.savefig(f"{D}/figures/cost.png", dpi=150); plt.close(fig)

# 3. headline: accuracy gain per GPU-minute
fig, ax = plt.subplots(figsize=(9, 5.5))
eff = np.array([r["gain_pts_per_gpu_min"] for r in rows])
e_err = np.array([[r["gain_pts_per_gpu_min"] - r["gain_per_min_ci"][0] for r in rows], [r["gain_per_min_ci"][1] - r["gain_pts_per_gpu_min"] for r in rows]])
bars(ax, eff, lambda v: f"{v:.1f}\n", e_err)
ax.set_ylim(0, (eff + e_err[1]).max() * 1.18)
ax.set_ylabel("Accuracy gain over zero-shot\n(points per training GPU-minute)")
best = rows[int(eff.argmax())]["method"]
ax.set_title("QLoRA and LoRA tie for the most accuracy per GPU-minute", loc="left")
fig.text(0.01, 0.01, f"{SUB}. Gain = accuracy − zero-shot ({A['zero_shot_acc']:.1f}%). Error bars: 95% bootstrap CI.", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{D}/figures/accuracy_per_gpu_minute.png", dpi=150); plt.close(fig)

# 4. training loss curves (seed-mean, smoothed)
fig, ax = plt.subplots(figsize=(9, 5.5))
for a, lab in zip(arms, labels):
    curves = [json.load(open(f"{D}/results/loss_{a}_s{s}.json")) for s in A["seeds"] if os.path.exists(f"{D}/results/loss_{a}_s{s}.json")]
    m = np.mean(curves, 0); k = 10; sm = np.convolve(m, np.ones(k) / k, mode="valid")
    ax.plot(np.arange(k, len(m) + 1), sm, color=COL[a], lw=2, label=lab)
ax.set_xlim(0, 255); ax.set_yscale("log"); ax.set_xlabel("Training step (batch 16)"); ax.set_ylabel("Train loss on answer tokens (log scale)")
ax.yaxis.grid(True, color=GRID); ax.legend(frameon=False, loc="upper right"); ax.set_title("Training loss: all methods converge in one epoch", loc="left")
fig.text(0.01, 0.01, f"{SUB}. Mean over seeds, 10-step moving average.", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{D}/figures/train_loss.png", dpi=150); plt.close(fig)

# table.png
cols = ["Method", "Accuracy (95% CI)", "Eval loss", "Trainable params", "Peak VRAM", "Train time", "Train CU", "Gain pts / GPU-min"]
cells = [[r["method"], f"{r['acc_mean']:.2f}% [{r['acc_ci_lo']:.2f}, {r['acc_ci_hi']:.2f}]", f"{r['eval_loss']:.3f}",
          f"{r['trainable_params']/1e6:,.1f}M ({r['trainable_pct']:.2f}%)", f"{r['peak_alloc_gb']:.1f} GB", f"{r['train_min']:.1f} min",
          f"{r['train_cu']:.3f}" if r["train_cu"] else "n/a", f"{r['gain_pts_per_gpu_min']:.1f}"] for r in rows]
cells.append(["Zero-shot", f"{A['zero_shot_acc']:.2f}%", "", "0", "", "", "", ""])
fig, ax = plt.subplots(figsize=(15, 0.36 * (len(cells) + 1) + 0.4)); ax.axis("off")
t = ax.table(cellText=cells, colLabels=cols, loc="upper center", cellLoc="center"); t.auto_set_font_size(False); t.set_fontsize(11); t.scale(1, 1.8); t.auto_set_column_width(list(range(len(cols))))
for (i, j), c in t.get_celld().items():
    c.set_edgecolor(GRID)
    if i == 0: c.set_text_props(weight="bold", color=INK); c.set_facecolor("#f1f0ec")
ax.set_title(f"PEFT head-to-head on one L4 — Qwen3-4B, AG News ({len(A['seeds'])} seed{'s' if len(A['seeds']) > 1 else ''})", fontweight="bold", loc="left")
fig.savefig(f"{D}/figures/table.png", dpi=150, bbox_inches="tight", pad_inches=0.15); plt.close(fig)
print("figures written")
