# PEFT methods head-to-head on one L4 (Qwen3-4B, AG News)

**Status:** preregistered 2026-10-17, before any training run. **Finished 2026-10-17: INCONCLUSIVE** (H1 inconclusive, H2 supported). See `writeup.md`.

**GPU / CU used:** NVIDIA L4 on Colab (1.54 CU/h). Kept runs: 3.37 GPU-h ≈ 5.2 CU. Total billed across 4 VMs (3 lost to restarts of the local monitoring machine): ≈ 6.8–10.2 CU (`results/compute.json`).

## Question
If you fine-tune a 4B model on one 24 GB L4, which method should you pick: LoRA, DoRA, QLoRA (4-bit) or a full fine-tune?
We compare accuracy and cost (VRAM, wall-clock, compute units).

## Hypotheses
- **H1:** DoRA beats LoRA, by less than 1 accuracy point.
- **H2:** QLoRA matches LoRA within ±1 point, and uses much less peak VRAM (at most 0.60× LoRA's peak training VRAM).

## Setup (fixed in advance)
- **Model:** `Qwen/Qwen3-4B` (public HF). We only fall back to Qwen3-1.7B if the full fine-tune cannot run on L4 or A100.
- **Data:** the protocol from my [LoRA rank sweep](https://github.com/nankris/lora-rank-sweep). AG News (`fancyzhx/ag_news`),
  4,000 training items and 2,000 test items, both shuffled with data seed 1234. Same prompt, max length 192, and labels
  ` World` / ` Sports` / ` Business` / ` Sci/Tech`. Loss is computed on the answer tokens only.
- **Eval:** the predicted label is the argmax of the summed log-probs of the 4 label strings. Eval loss is the mean per-token
  cross-entropy of the gold label. A zero-shot baseline is also run.
- **Training (identical across arms):** 1 epoch = 250 steps at batch size 16. Linear warmup over 10 steps, then linear decay.
  Training order is fixed by the run seed, and gradient checkpointing is on for every arm. The model computes in bf16 autocast.
- **Arms:**
  | Arm | Base weights | Trainable | LR |
  |---|---|---|---|
  | LoRA r16 | bf16, frozen | LoRA r=16, α=32, dropout 0, all 7 linear projections | 2e-4 |
  | DoRA r16 | bf16, frozen | same as LoRA + `use_dora=True` | 2e-4 |
  | QLoRA r16 | NF4 4-bit + double quant, frozen | same as LoRA | 2e-4 |
  | Full FT | bf16, all params trainable | all ~4.0B params | 1e-5 |
  - All adapter arms use torch AdamW with weight decay 0 and global grad-norm clipping at 1.0.
  - A plain AdamW full fine-tune of 4B needs about 64 GB (weights + grads + fp32 states), so it cannot fit on an L4.
    The full-FT arm therefore uses the standard memory tricks for a 24 GB card: bf16 weights and torchao `AdamW8bit` with bf16
    stochastic rounding. The optimizer also steps inside the backward pass, so no full gradient copy is ever held.
    Because the full gradient is never materialized, global grad-norm clipping is impossible. This arm skips clipping, which
    is a documented deviation. Its LR (1e-5) is the usual full-FT default and was chosen in advance, not tuned.
- **Seeds:** 1234, 1235, 1236. The seed sets the training order and the adapter init. Runs go seed-major, so a crash still
  leaves complete seeds.
- **Logged per run:** test accuracy, eval loss, trainable params, peak VRAM (`torch.cuda.max_memory_allocated` and
  `max_memory_reserved` during training), training wall-clock, eval wall-clock, and all 2,000 predictions.
- **Hardware:** one Colab L4 (session `peft-head-to-head`). CU per arm = training GPU-hours × the measured session CU/hour rate.

## Decision rule (fixed)
The CIs are 95% two-level paired bootstrap CIs (10,000 draws). Each draw resamples seeds with replacement (the same seeds for
both arms) and test items with replacement (the same items for both arms). The statistic is the difference in mean accuracy
over those seeds × items, in percentage points.
- **H1** (Δ = DoRA − LoRA): **supported** if the CI lies inside (0, +1). **Contradicted** if the CI upper bound ≤ 0 (DoRA is not
  better) or the CI lower bound ≥ +1 (DoRA wins by more than 1 point). Anything else is inconclusive.
- **H2** (Δ = QLoRA − LoRA): **supported** if the CI lies inside [−1, +1] **and** mean peak allocated VRAM of QLoRA ≤ 0.60× LoRA.
  **Contradicted** if the CI lies wholly outside [−1, +1] **or** the VRAM ratio is > 0.60. Anything else is inconclusive.
- **Overall:** **PROVEN** if H1 and H2 are both supported. **DISPROVEN** if either is contradicted. Otherwise **INCONCLUSIVE**.
- **Headline metric:** "accuracy per GPU-minute" = (arm accuracy − zero-shot accuracy, in points) ÷ training wall-clock
  minutes on the L4, averaged over seeds. We also report raw accuracy, so the "cheap but worse" trade-off stays visible.

## Execution notes (added during the run, before any real results)
- Smoke test (64 train / 32 test items, not used for results) showed that the full-FT run's memory, including its torch.compile
  optimizer caches, was not freed in-process, so the next arms ran out of memory. Fix: `train.py` now acts as a driver that runs
  every (arm, seed) in a fresh subprocess. Training/eval protocol unchanged; peak VRAM now starts from zero for every run.
- The session slot was obtained 2026-10-17 06:57 UTC after ~1h of retries (account at its 2-session limit).
- That first VM disappeared between 07:33 and 08:20 UTC. The local monitoring machine had restarted and stopped the poller, and the VM
  most likely idled out. Its outputs (full_s1234 finished, lora_s1234 partway) were never fetched, so they are discarded
  unseen. A new `peft-head-to-head` L4 VM was started at 08:20 UTC and all runs restarted from scratch with the same code.
  `scripts/keepalive_fetch.sh` now polls and fetches every 5 min. CU for the lost VM (~1.54 CU/h × ~1.4 h) counts toward the total.
- VM2 was also lost (last fetch 08:59 UTC; gone by 09:24 UTC) after another restart of the local monitoring machine. Runs already fetched (zero_shot,
  full_s1234, lora_s1234) are kept. dora_s1234 was partway (step 76) and is discarded. `scripts/resume_vm.sh` started VM3 at
  09:25 UTC with the same package versions. It re-seeds the remote results directory with the fetched runs so `train.py` skips them, then
  continues with identical code. Wall-clock is still comparable across VMs because every one was an L4 at the same
  1.54 CU/h, but VM-to-VM speed variance is a caveat.
- full_s1236 hit a CUDA OOM at step 1 (11:17 UTC). The allocator reported 2.9 GB reserved-but-unallocated, i.e. fragmentation:
  the 20.3 GB peak is the same as the other full-FT seeds. Once every other run finished, the driver ran once more with
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (`scripts/chain_retry_full.py`) to retry just that run. This changes only
  the allocator, not the training math. The failed attempt stays in `runs.json` as `status: failed`.
- VM3 was lost too, after another restart of the local monitoring machine (last fetch 11:22 UTC, with lora_s1236 at step 126, which is discarded).
  `scripts/resume_vm.sh` started VM4 at 12:24 UTC with `LAUNCH=launch_expseg.py`, i.e. with `expandable_segments` for the
  remaining runs (full_s1236, lora_s1236, dora_s1236, qlora_s1236). It was re-seeded with all fetched runs. Allocator-only change.

## Files
`scripts/train.py` (remote, resumable via `results/runs.json`) · `scripts/launch.py` · `scripts/analyze.py` ·
`scripts/figures.py` · `results/` raw JSON/CSV + logs · `figures/` charts, table.png and colab_run.txt · `writeup.md`.
