# Study: PEFT head-to-head (LoRA / DoRA / QLoRA / full FT) on Qwen3-4B, AG News, one L4.
# Data prep, prompt and scoring are copied from the LoRA rank sweep (https://github.com/nankris/lora-rank-sweep).
# Writes /content/peft-results/{runs.json,progress.log,pred_*.json,loss_*.json,metadata.json}. Resumable: finished runs are skipped.
import gc, json, math, os, time, traceback
import torch, torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

OUT = os.environ.get("OUT", "/content/peft-results"); os.makedirs(OUT, exist_ok=True)
LOG = open(f"{OUT}/progress.log", "a", buffering=1)
def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a)
    print(s, flush=True); LOG.write(s + "\n")

MODEL = os.environ.get("MODEL", "Qwen/Qwen3-4B"); DATA_SEED = 1234
N_TRAIN, N_TEST = int(os.environ.get("N_TRAIN", 4000)), int(os.environ.get("N_TEST", 2000))  # overrides only for smoke test
BS, MAXLEN = 16, 192
ARMS = os.environ.get("ARMS", "full,lora,dora,qlora").split(",")
SEEDS = [int(s) for s in os.environ.get("SEEDS", "1234,1235,1236").split(",")]
LR = {"lora": 2e-4, "dora": 2e-4, "qlora": 2e-4, "full": 1e-5}
LABELS = [" World", " Sports", " Business", " Sci/Tech"]
TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
dev = "cuda"

tok = AutoTokenizer.from_pretrained(MODEL)
PAD = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
HEAD = tok("Classify the topic of this news article as World, Sports, Business, or Sci/Tech.\nArticle: ", add_special_tokens=False).input_ids
TAIL = tok("\nTopic:", add_special_tokens=False).input_ids
LAB_IDS = [tok(l, add_special_tokens=False).input_ids for l in LABELS]
MAXLAB = max(len(l) for l in LAB_IDS)
log("model", MODEL, "label token ids", LAB_IDS)

def prompt_ids(text):
    budget = MAXLEN - len(HEAD) - len(TAIL) - MAXLAB
    body = tok(text, add_special_tokens=False).input_ids[:budget]
    return HEAD + body + TAIL

ds = load_dataset("fancyzhx/ag_news")
train = ds["train"].shuffle(seed=DATA_SEED).select(range(N_TRAIN))
test = ds["test"].shuffle(seed=DATA_SEED).select(range(N_TEST))
train_ex = [(prompt_ids(t), y) for t, y in zip(train["text"], train["label"])]
test_ex = [(prompt_ids(t), y) for t, y in zip(test["text"], test["label"])]
json.dump([y for _, y in test_ex], open(f"{OUT}/test_labels.json", "w"))

def collate(pairs):
    """pairs: list of (prompt_ids, answer_ids). Left-pad so answers sit at the end; labels -100 except answer tokens."""
    L = max(len(p) + len(a) for p, a in pairs)
    ids = torch.full((len(pairs), L), PAD); att = torch.zeros((len(pairs), L), dtype=torch.long)
    lab = torch.full((len(pairs), L), -100)
    for i, (p, a) in enumerate(pairs):
        s = p + a; o = L - len(s); ids[i, o:] = torch.tensor(s); att[i, o:] = 1
        lab[i, L - len(a):] = torch.tensor(a)
    pos = (att.cumsum(-1) - 1).clamp(min=0)
    return ids.to(dev), att.to(dev), pos.to(dev), lab.to(dev)

K = MAXLAB + 1  # only compute logits for the last K positions (covers all answer tokens)
def answer_logits(model, ids, att, pos, lab):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(input_ids=ids, attention_mask=att, position_ids=pos, logits_to_keep=K).logits[:, :-1]
    return logits.float(), lab[:, -(K - 1):]

def answer_logprobs(model, ids, att, pos, lab):
    logits, tgt = answer_logits(model, ids, att, pos, lab); mask = tgt != -100
    lp = torch.log_softmax(logits, -1).gather(-1, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (lp * mask).sum(1), mask.sum(1)

@torch.no_grad()
def evaluate(model, name, ebs=16):
    model.eval(); correct = 0; losses = []; preds = []
    for i in range(0, len(test_ex), ebs):
        chunk = test_ex[i:i + ebs]
        s, n = answer_logprobs(model, *collate([(p, LAB_IDS[k]) for p, _ in chunk for k in range(4)]))
        s = s.view(len(chunk), 4); n = n.view(len(chunk), 4)
        for j, (_, y) in enumerate(chunk):
            preds.append(int(s[j].argmax().item())); correct += int(preds[-1] == y)
            losses.append((-s[j, y] / n[j, y]).item())  # mean per-token CE on gold answer
    json.dump(preds, open(f"{OUT}/pred_{name}.json", "w"))
    return correct / len(test_ex), sum(losses) / len(losses)

def load_model(arm, seed):
    torch.manual_seed(seed)
    if arm == "qlora":
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                               bnb_4bit_compute_dtype=torch.bfloat16)
        m = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=q, dtype=torch.bfloat16, device_map={"": 0})
    else:
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(dev)
    m.config.use_cache = False
    return m

def setup_full(m, lr_fn):
    """Full FT on 24 GB: bf16 weights + torchao AdamW8bit (bf16 stochastic rounding), optimizer stepped per-parameter
    inside backward (post-accumulate-grad hooks) so the full gradient is never held. No global grad clipping possible."""
    import torch._dynamo
    torch._dynamo.config.cache_size_limit = 256
    try: torch._dynamo.config.recompile_limit = 256
    except Exception: pass
    from torchao.optim import AdamW8bit
    opts = {}
    for p in m.parameters():
        p.requires_grad_(True)
        opts[p] = AdamW8bit([p], lr=lr_fn(0), weight_decay=0.0, bf16_stochastic_round=True)
        def hook(param, o=opts[p]):
            o.step(); o.zero_grad(set_to_none=True)
        p.register_post_accumulate_grad_hook(hook)
    return list(opts.values())

res_path = f"{OUT}/runs.json"
ONLY = os.environ.get("ONLY")  # set by the driver: run exactly one job in this process
if ONLY is None:
    # Driver: each (arm, seed) runs in a fresh subprocess so GPU memory (incl. torch.compile caches of the full-FT
    # optimizer) is fully released between runs and every run's peak-VRAM measurement starts from zero.
    import subprocess, sys
    for name in ["zero_shot"] + [f"{a}_s{s}" for s in SEEDS for a in ARMS]:
        runs = json.load(open(res_path)) if os.path.exists(res_path) else []
        if name in {r["run"] for r in runs if r.get("status", "ok") == "ok"}: continue
        log("driver: launching", name)
        rc = subprocess.run([sys.executable, __file__], env={**os.environ, "ONLY": name}).returncode
        if rc != 0: log(f"driver: {name} exited with code {rc}")
    log("DONE")
    import importlib.metadata as md
    json.dump(dict(gpu=torch.cuda.get_device_name(), model=MODEL, train_fingerprint=train._fingerprint, test_fingerprint=test._fingerprint,
                   versions={x: md.version(x) for x in ["torch", "transformers", "peft", "datasets", "bitsandbytes", "torchao"]}),
              open(f"{OUT}/metadata.json", "w"), indent=2)
    sys.exit(0)
runs = json.load(open(res_path)) if os.path.exists(res_path) else []
def save(): json.dump(runs, open(res_path, "w"), indent=2)
done = {r["run"] for r in runs if r.get("status", "ok") == "ok"}

if ONLY == "zero_shot" and "zero_shot" not in done:
    m = load_model("lora", 0); t0 = time.time(); acc, el = evaluate(m, "zero_shot")
    runs.append(dict(run="zero_shot", arm="zero_shot", eval_accuracy=acc, eval_loss=el, eval_seconds=time.time() - t0))
    save(); log("zero_shot", runs[-1]); del m; torch.cuda.empty_cache()

for seed in SEEDS:
    for arm in ARMS:
        name = f"{arm}_s{seed}"
        if name != ONLY or name in done: continue
        try:
            m = load_model(arm, seed)
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            steps = math.ceil(N_TRAIN / BS)
            lr_fn = lambda s: LR[arm] * min(1.0, (s + 1) / 10) * max(0.0, 1 - s / steps)
            if arm == "full":
                opts = setup_full(m, lr_fn); params = None
            else:
                m = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.0, target_modules=TARGETS,
                                                 task_type="CAUSAL_LM", use_dora=(arm == "dora")))
                params = [p for p in m.parameters() if p.requires_grad]
                opts = [torch.optim.AdamW(params, lr=lr_fn(0), weight_decay=0.0)]
            ntrain = sum(p.numel() for p in m.parameters() if p.requires_grad)
            ntotal = sum(p.numel() for p in m.parameters())
            log(f"{name}: trainable {ntrain} / {ntotal}; mem after load {torch.cuda.memory_allocated()/1e9:.2f} GB")
            g = torch.Generator().manual_seed(seed); order = torch.randperm(N_TRAIN, generator=g).tolist()
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            m.train(); losses = []
            for step in range(steps):
                for o in opts:
                    for gr in o.param_groups:
                        if torch.is_tensor(gr["lr"]): gr["lr"].fill_(lr_fn(step))  # torchao keeps lr as a tensor
                        else: gr["lr"] = lr_fn(step)
                batch = [train_ex[i] for i in order[step * BS:(step + 1) * BS]]
                logits, tgt = answer_logits(m, *collate([(p, LAB_IDS[y]) for p, y in batch]))
                loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), ignore_index=-100)
                loss.backward()  # full FT: optimizer steps happen inside this call
                if params is not None:
                    torch.nn.utils.clip_grad_norm_(params, 1.0); opts[0].step(); opts[0].zero_grad(set_to_none=True)
                losses.append(loss.item())
                if step % 25 == 0 or step == steps - 1:
                    log(f"{name} step {step + 1}/{steps} loss {loss.item():.4f} avg20 {sum(losses[-20:])/len(losses[-20:]):.4f} "
                        f"t={time.time()-t0:.0f}s peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB")
            torch.cuda.synchronize(); train_s = time.time() - t0
            peak, peak_res = torch.cuda.max_memory_allocated() / 1e9, torch.cuda.max_memory_reserved() / 1e9
            t1 = time.time(); acc, el = evaluate(m, name)
            json.dump(losses, open(f"{OUT}/loss_{name}.json", "w"))
            runs.append(dict(run=name, arm=arm, seed=seed, status="ok", lr=LR[arm], steps=steps, trainable_params=ntrain,
                             total_params=ntotal, train_seconds=train_s, peak_alloc_gb=peak, peak_reserved_gb=peak_res,
                             final_train_loss_avg20=sum(losses[-20:]) / len(losses[-20:]), eval_accuracy=acc, eval_loss=el,
                             eval_seconds=time.time() - t1))
            save(); log(name, runs[-1])
        except Exception as e:
            log(f"{name} FAILED: {e!r}\n{traceback.format_exc()}")
            runs.append(dict(run=name, arm=arm, seed=seed, status="failed", error=repr(e)[:500])); save()
        m = opts = params = None; gc.collect(); torch.cuda.empty_cache()
