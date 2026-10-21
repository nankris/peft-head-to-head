#!/bin/bash
# Bring up a fresh b6-101 VM after a VM loss: allocate, install deps, re-seed /content/out101 with every finished run
# already fetched into results/ (train.py skips those), launch the driver, then start the keepalive/fetch loop.
D=/workspace/work/tig/101-peft-head-to-head; R=$D/results
source /workspace/tools/setup.sh >/dev/null 2>&1
if ! colab sessions 2>&1 | grep -q b6-101; then
  bash $D/scripts/get_session.sh || exit 1
  date -u +%s >> $R/vm_starts.log
fi
N=$(ls $R/vm_starts.log >/dev/null 2>&1 && wc -l < $R/vm_starts.log)
timeout 900 colab exec -s b6-101 -f $D/scripts/setup_remote.py --timeout 900 > $R/setup_remote_vm$N.txt 2>&1
T=$(mktemp -d); mkdir $T/out101
cp $R/runs.json $R/pred_*.json $R/loss_*.json $R/progress.log $T/out101/ 2>/dev/null
(cd $T && tar czf seed.tgz out101)
colab upload -s b6-101 $T/seed.tgz /content/seed.tgz
colab upload -s b6-101 $D/scripts/train.py /content/train.py
echo 'import subprocess; print(subprocess.run("cd /content && tar xzf seed.tgz && ls out101 && echo RESUMED_FROM_FETCHED_RESULTS >> out101/progress.log", shell=True, capture_output=True, text=True).stdout)' > $T/unpack.py
colab exec -s b6-101 -f $T/unpack.py --timeout 60
colab exec -s b6-101 -f $D/scripts/${LAUNCH:-launch.py} --timeout 60
nohup bash $D/scripts/keepalive_fetch.sh > /dev/null 2>&1 &
echo resumed
