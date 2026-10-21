#!/bin/bash
# Every 5 min: poll the Colab run (also keeps the VM from idling out) and fetch outputs into results/. Stops on DONE.
D=/workspace/work/tig/101-peft-head-to-head
source /workspace/tools/setup.sh >/dev/null 2>&1
for i in $(seq 1 100); do
  out=$(timeout 120 colab exec -s b6-101 -f $D/scripts/poll.py --timeout 60 2>&1)
  echo "== $(date -u +%FT%TZ)" >> $D/results/poll.log; echo "$out" | tail -3 >> $D/results/poll.log
  bash $D/scripts/fetch.sh >> $D/results/fetch.log 2>&1
  echo "$out" | grep -qE "RETRY_DONE|DONE" && { echo DONE >> $D/results/poll.log; exit 0; }
  sleep 180
done
