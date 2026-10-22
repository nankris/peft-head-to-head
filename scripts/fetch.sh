#!/bin/bash
# Pull /content/peft-results (+ stdout log and nvidia-smi snapshot) from the Colab session into results/. Safe to re-run.
source /workspace/tools/setup.sh >/dev/null 2>&1
D="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"; T=$(mktemp -d)
cat > $T/pack.py <<'PY'
import subprocess
subprocess.run("nvidia-smi > /content/peft-results/nvidia_smi_end.txt; cp /content/train_stdout.log /content/peft-results/ 2>/dev/null; cd /content && tar czf peft-results.tgz peft-results", shell=True)
print("packed")
PY
for i in 1 2 3 4 5; do timeout 100 colab exec -s peft-head-to-head -f $T/pack.py --timeout 60 | grep -q packed && break; sleep 10; done
for i in 1 2 3 4 5; do timeout 200 colab download -s peft-head-to-head /content/peft-results.tgz $T/peft-results.tgz && break; sleep 10; done
tar xzf $T/peft-results.tgz -C $T && cp $T/peft-results/* $D/results/ && echo "fetched $(ls $T/peft-results | wc -l) files"
