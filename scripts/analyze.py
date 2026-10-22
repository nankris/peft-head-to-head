"""Local analysis for the PEFT comparison. Reads results/runs.json + pred_*.json + test_labels.json, applies the README decision rule,
writes results/summary.csv, results/analysis.json. Run: python scripts/analyze.py"""
import json, os, csv
import numpy as np

R = os.path.join(os.path.dirname(__file__), "..", "results")
runs = [r for r in json.load(open(f"{R}/runs.json")) if r.get("status", "ok") == "ok"]
y = np.array(json.load(open(f"{R}/test_labels.json")))
compute = json.load(open(f"{R}/compute.json")) if os.path.exists(f"{R}/compute.json") else {}
CU_PER_HR = compute.get("session_cu_per_hour")
zs = next(r for r in runs if r["run"] == "zero_shot")
ZS = 100 * zs["eval_accuracy"]
ARMS = ["lora", "dora", "qlora", "full"]
NAMES = {"lora": "LoRA r16", "dora": "DoRA r16", "qlora": "QLoRA 4-bit r16", "full": "Full fine-tune"}

def correct(name): return (np.array(json.load(open(f"{R}/pred_{name}.json"))) == y).astype(float)
by = {a: {r["seed"]: r for r in runs if r["arm"] == a} for a in ARMS}
seeds = sorted(set.intersection(*[set(by[a]) for a in ARMS if by[a]]))
C = {a: np.stack([correct(f"{a}_s{s}") for s in seeds]) for a in ARMS if by[a]}  # seeds x items

rng = np.random.default_rng(0); B = 10_000; S, N = len(seeds), len(y)
def boot(stat):
    out = np.empty(B)
    for b in range(B):
        si = rng.integers(0, S, S); ii = rng.integers(0, N, N)
        out[b] = stat(si, ii)
    return np.percentile(out, [2.5, 97.5])
def diff_ci(a, b):
    return boot(lambda si, ii: 100 * (C[a][si][:, ii].mean() - C[b][si][:, ii].mean()))
def acc_ci(a):
    return boot(lambda si, ii: 100 * C[a][si][:, ii].mean())

rows = []
for a in ARMS:
    if a not in C: continue
    rs = [by[a][s] for s in seeds]
    accs = np.array([100 * r["eval_accuracy"] for r in rs]); mins = np.array([r["train_seconds"] / 60 for r in rs])
    lo, hi = acc_ci(a)
    rows.append(dict(arm=a, method=NAMES[a], n_seeds=S, acc_mean=accs.mean(), acc_sd=accs.std(ddof=1) if S > 1 else 0.0,
                     acc_ci_lo=lo, acc_ci_hi=hi, eval_loss=np.mean([r["eval_loss"] for r in rs]),
                     trainable_params=rs[0]["trainable_params"], trainable_pct=100 * rs[0]["trainable_params"] / (by["lora"][seeds[0]]["total_params"] if a == "qlora" else rs[0]["total_params"]),  # 4-bit packing undercounts QLoRA totals
                     peak_alloc_gb=np.mean([r["peak_alloc_gb"] for r in rs]), peak_reserved_gb=np.mean([r["peak_reserved_gb"] for r in rs]),
                     train_min=mins.mean(), eval_min=np.mean([r["eval_seconds"] / 60 for r in rs]),
                     train_cu=(mins.mean() / 60 * CU_PER_HR) if CU_PER_HR else None,
                     gain_pts_per_gpu_min=np.mean((accs - ZS) / mins), final_train_loss=np.mean([r["final_train_loss_avg20"] for r in rs])))
with open(f"{R}/summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
row = {r["arm"]: r for r in rows}

# headline metric CI: bootstrap over seeds+items of (acc - zero-shot acc)/train-minutes (per-seed minutes)
czs = correct("zero_shot")
def eff_ci(a):
    mins = np.array([by[a][s]["train_seconds"] / 60 for s in seeds])
    return boot(lambda si, ii: np.mean(100 * (C[a][si][:, ii].mean(1) - czs[ii].mean()) / mins[si])).tolist()
for a in row: row[a]["gain_per_min_ci"] = eff_ci(a)

h1 = diff_ci("dora", "lora").tolist()
h1_d = row["dora"]["acc_mean"] - row["lora"]["acc_mean"]
h1_v = "supported" if (h1[0] > 0 and h1[1] < 1) else "contradicted" if (h1[1] <= 0 or h1[0] >= 1) else "inconclusive"
h2 = diff_ci("qlora", "lora").tolist()
h2_d = row["qlora"]["acc_mean"] - row["lora"]["acc_mean"]
vram_ratio = row["qlora"]["peak_alloc_gb"] / row["lora"]["peak_alloc_gb"]
inside = h2[0] >= -1 and h2[1] <= 1; outside = h2[0] > 1 or h2[1] < -1
h2_v = "supported" if (inside and vram_ratio <= 0.60) else "contradicted" if (outside or vram_ratio > 0.60) else "inconclusive"
overall = "PROVEN" if h1_v == h2_v == "supported" else "DISPROVEN" if "contradicted" in (h1_v, h2_v) else "INCONCLUSIVE"
extra = {}
if "full" in C:
    extra["full_minus_lora"] = dict(diff=row["full"]["acc_mean"] - row["lora"]["acc_mean"], ci=diff_ci("full", "lora").tolist())
extra["dora_minus_qlora"] = dict(diff=row["dora"]["acc_mean"] - row["qlora"]["acc_mean"], ci=diff_ci("dora", "qlora").tolist())
best = max(row, key=lambda a: row[a]["gain_pts_per_gpu_min"])
out = dict(seeds=seeds, zero_shot_acc=ZS, cu_per_hour=CU_PER_HR,
           H1=dict(diff=h1_d, ci=h1, verdict=h1_v), H2=dict(diff=h2_d, ci=h2, vram_ratio=vram_ratio, verdict=h2_v),
           overall=overall, extra=extra, best_gain_per_gpu_min=best, rows=rows)
json.dump(out, open(f"{R}/analysis.json", "w"), indent=2, default=float)
print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2, default=float))
for r in rows: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

# paired CI for the headline: QLoRA minus LoRA in gain-points per training GPU-minute (same seeds/items)
def eff_diff_ci(a, b):
    ma = np.array([by[a][s]["train_seconds"] / 60 for s in seeds]); mb = np.array([by[b][s]["train_seconds"] / 60 for s in seeds])
    f = lambda si, ii: np.mean(100 * (C[a][si][:, ii].mean(1) - czs[ii].mean()) / ma[si]) - np.mean(100 * (C[b][si][:, ii].mean(1) - czs[ii].mean()) / mb[si])
    return boot(f).tolist()
out["eff_qlora_minus_lora"] = dict(diff=row["qlora"]["gain_pts_per_gpu_min"] - row["lora"]["gain_pts_per_gpu_min"], ci=eff_diff_ci("qlora", "lora"))
json.dump(out, open(f"{R}/analysis.json", "w"), indent=2, default=float)
print("eff_qlora_minus_lora", out["eff_qlora_minus_lora"])
