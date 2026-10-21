#!/bin/bash
# Pull /content/out101 (+ stdout log and nvidia-smi snapshot) from the Colab session into results/. Safe to re-run.
source /workspace/tools/setup.sh >/dev/null 2>&1
D=/workspace/work/tig/101-peft-head-to-head; T=$(mktemp -d)
cat > $T/pack.py <<'PY'
import subprocess
subprocess.run("nvidia-smi > /content/out101/nvidia_smi_end.txt; cp /content/train_stdout.log /content/out101/ 2>/dev/null; cd /content && tar czf out101.tgz out101", shell=True)
print("packed")
PY
for i in 1 2 3 4 5; do timeout 100 colab exec -s b6-101 -f $T/pack.py --timeout 60 | grep -q packed && break; sleep 10; done
for i in 1 2 3 4 5; do timeout 200 colab download -s b6-101 /content/out101.tgz $T/out101.tgz && break; sleep 10; done
tar xzf $T/out101.tgz -C $T && cp $T/out101/* $D/results/ && echo "fetched $(ls $T/out101 | wc -l) files"
