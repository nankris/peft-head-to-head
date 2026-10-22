# LoRA vs DoRA vs QLoRA vs full fine-tune on one L4

**Verdict: INCONCLUSIVE.** This follows the preregistered rule in `README.md`. H2 is **supported**: QLoRA matches LoRA at
0.44× the VRAM. H1 is **inconclusive**: DoRA did *not* beat LoRA (the point estimate is −0.10 points), but the CI still crosses 0.

**Headline number: 2.5 accuracy points per training GPU-minute.** QLoRA 4-bit r16 scored 2.49 [2.24, 2.74] and plain LoRA
scored 2.48 [2.23, 2.73]. That is a statistical tie (paired difference +0.02 [−0.05, +0.08]). QLoRA got there at **4.3 GB peak
VRAM instead of 9.9 GB**. So "QLoRA gave the best accuracy per GPU-minute" is true only as a tie with LoRA. Its real win is memory.

**Takeaway:** On a 4B model and an easy classification task, all four methods ended within 0.4 points of each other, so the
choice comes down to cost. QLoRA was as fast and as accurate as LoRA on less than half the memory. DoRA cost 2.6× the training
time for no gain.

## Setup (short; full protocol in README.md)
Qwen/Qwen3-4B, AG News (4,000 train / 2,000 test, same subset as my [LoRA rank sweep](https://github.com/nankris/lora-rank-sweep)), 250 steps at batch 16, 3 seeds (1234–1236) per arm, one
Colab NVIDIA L4 (22 GiB usable). Accuracy is the argmax over the summed log-probs of the 4 label strings. Zero-shot baseline:
**73.30%**. Full fine-tune fit on the L4 (bf16 weights + 8-bit AdamW, optimizer step in backward), so **no fallback to 1.7B was needed**.

## Main results (mean of 3 seeds)

| Method | Accuracy (95% CI) | Seed SD | Eval loss | Trainable params | Peak VRAM (alloc / reserved) | Train time | Eval time | Train CU | Gain pts / GPU-min (95% CI) |
|---|---|---|---|---|---|---|---|---|---|
| LoRA r16 | 91.48% [90.27, 92.70] | 0.19 | 0.218 | 33.0M (0.81%) | 9.9 / 10.5 GB | 7.3 min | 5.4 min | 0.19 | 2.48 [2.23, 2.73] |
| DoRA r16 | 91.38% [90.20, 92.57] | 0.13 | 0.214 | 34.1M (0.84%) | 10.9 / 11.5 GB | 18.9 min | 9.8 min | 0.49 | 0.95 [0.86, 1.05] |
| QLoRA 4-bit r16 | 91.58% [90.35, 92.75] | 0.31 | 0.213 | 33.0M (0.81%) | **4.3** / 4.9 GB | 7.3 min | 5.1 min | 0.19 | **2.49** [2.24, 2.74] |
| Full fine-tune | **91.75%** [90.58, 92.90] | 0.15 | **0.210** | 4,022M (100%) | 20.3 / 22.4 GB | 9.0 min | 3.4 min | 0.23 | 2.05 [1.84, 2.25] |
| Zero-shot | 73.30% | – | 2.650 | 0 | – | – | 3.5 min | – | – |

The table is also rendered as `figures/table.png`. Train CU = mean training minutes × 1.54 CU/h (measured L4 rate).
"Gain pts / GPU-min" = (accuracy − 73.30) ÷ training minutes.

## Hypothesis tests (paired two-level bootstrap over seeds × test items, 10,000 draws)

| Comparison | Δ accuracy (points) | 95% CI | Rule | Result |
|---|---|---|---|---|
| H1: DoRA − LoRA | −0.10 | [−0.47, +0.30] | supported if CI ⊂ (0, +1) | **inconclusive** (CI crosses 0; point estimate is negative) |
| H2: QLoRA − LoRA | +0.10 | [−0.35, +0.55] | CI ⊂ [−1, +1] and VRAM ratio ≤ 0.60 | **supported** (VRAM ratio 0.435) |
| Full − LoRA (extra) | +0.27 | [−0.22, +0.77] | – | not significant |
| DoRA − QLoRA (extra) | −0.20 | [−0.70, +0.33] | – | not significant |
| Efficiency: QLoRA − LoRA (pts/GPU-min) | +0.015 | [−0.049, +0.077] | – | tie |

**Overall: INCONCLUSIVE** (H1 neither supported nor contradicted, H2 supported).

## Per-seed results

| Run | Accuracy | Eval loss | Train s | Eval s | Peak alloc GB |
|---|---|---|---|---|---|
| full s1234 / s1235 / s1236 | 91.90 / 91.75 / 91.60 | 0.205 / 0.214 / 0.211 | 548 / 535 / 539 | 207 / 203 / 205 | 20.33 / 20.33 / 20.21 |
| lora s1234 / s1235 / s1236 | 91.70 / 91.40 / 91.35 | 0.218 / 0.224 / 0.213 | 443 / 441 / 438 | 325 / 322 / 321 | 9.95 / 9.95 / 9.89 |
| dora s1234 / s1235 / s1236 | 91.50 / 91.25 / 91.40 | 0.213 / 0.226 / 0.204 | 1134 / 1140 / 1136 | 588 / 588 / 586 | 10.93 / 10.93 / 10.87 |
| qlora s1234 / s1235 / s1236 | 91.85 / 91.65 / 91.25 | 0.213 / 0.211 / 0.216 | 439 / 443 / 439 | 304 / 304 / 303 | 4.33 / 4.33 / 4.30 |

## What we learned
1. **Accuracy is a wash.** The best and worst arms are 0.37 points apart. Every paired CI includes 0. Full fine-tuning is
   nominally best (91.75%, lowest eval loss) but not significantly so.
2. **DoRA did not pay off here.** It was nominally 0.10 points *below* LoRA, and it took 2.6× longer to train (18.9 vs 7.3 min)
   and 1.8× longer to evaluate in PEFT's unmerged form. Our rule calls H1 inconclusive, not contradicted, because the CI's upper
   bound is +0.30. The data does rule out a DoRA gain of more than about 0.3 points.
3. **QLoRA was free memory.** It used 4.3 GB peak versus 9.9 GB for LoRA, with the same accuracy and the same wall-clock
   (7.3 min each; we expected the 4-bit dequantization to slow it down, but on the L4 it did not). That frees enough of the L4
   to fine-tune a considerably larger model on it.
4. **Full fine-tuning fits a 4B model on an L4,** but only with 8-bit AdamW and in-backward optimizer steps. Even then it peaks at
   20.3 GB (22.4 GB reserved), which is right at the edge: one seed hit a fragmentation OOM and needed `expandable_segments`.
   It is 23% slower than LoRA per run and saves a full 8 GB checkpoint instead of a 66 MB (bf16) adapter.

## Compute
- **GPU:** NVIDIA L4 (Colab, 1.54 CU/h measured with only this session active).
- **Kept runs** (13 runs, training + eval): 3.37 GPU-hours ≈ **5.2 CU**.
- **Total session time billed** across 4 VMs (3 were lost after restarts of the local monitoring machine; see README execution notes): 4.4–6.7 h,
  i.e. **≈ 6.8–10.2 CU**. The range exists because we only know a lost VM's last-seen time, not its exact end time.
  Session `peft-head-to-head` was stopped at 13:36 UTC. (The account balance can't isolate our spend because another session was running.)
- Evidence: `figures/colab_run.txt` (nvidia-smi, per-run timing and result lines), `results/progress.log`, `results/poll.log`.

## Caveats
- **Easy task, near ceiling.** AG News gets 91–92% after one epoch on 4k examples. A harder task (reasoning, long-form
  generation) could separate the methods, especially full FT vs adapters.
- **One untuned LR per arm** (2e-4 for adapters, 1e-5 for full FT), fixed in advance. A per-arm LR sweep could change the ordering
  by a few tenths of a point.
- **The full-FT arm is not a textbook full fine-tune.** It uses bf16 weights, 8-bit AdamW with stochastic rounding, and no
  grad-norm clipping (impossible with in-backward optimizer steps). This is a documented deviation and is what "full FT on one
  L4" realistically means.
- **DoRA's speed penalty depends on the implementation** (PEFT's DoRA recomputes weight norms each step and is not merged for
  eval). A fused or merged DoRA would narrow the time gap. The accuracy result would not change.
- **"Accuracy per GPU-minute" counts training time only.** If you add eval time, full FT gains (fastest eval, no adapter
  overhead) and DoRA loses further. The QLoRA/LoRA tie is unchanged.
- **Runs span 4 Colab L4 VMs,** and seed 1236's adapter runs used the `expandable_segments` allocator setting. Every L4 ran at
  ~the same speed (per-arm train times vary by <3% across seeds), but VM-to-VM variance is not zero.
- The CIs combine seed and test-item resampling across only 3 seeds, so the seed component is coarse.

## Files
`scripts/train.py` (remote driver, one subprocess per run, resumable) · `scripts/analyze.py` (decision rule →
`results/analysis.json`, `results/summary.csv`) · `scripts/figures.py` · `results/runs.json`, `pred_*.json`, `loss_*.json`,
`compute.json`, logs · `figures/accuracy.png`, `accuracy_per_gpu_minute.png`, `cost.png`, `train_loss.png`, `table.png`,
`colab_run.txt`.
