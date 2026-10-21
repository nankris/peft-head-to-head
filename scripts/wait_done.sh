#!/bin/bash
# Poll the Colab run every 10 min until the driver logs DONE (or 4 h pass); appends to results/poll.log.
source /workspace/tools/setup.sh >/dev/null 2>&1
D=/workspace/work/tig/101-peft-head-to-head
for i in $(seq 1 24); do
  out=$(timeout 120 colab exec -s b6-101 -f $D/scripts/poll.py --timeout 60 2>&1)
  echo "== $(date -u +%FT%TZ)" >> $D/results/poll.log; echo "$out" | tail -4 >> $D/results/poll.log
  echo "$out" | grep -q "DONE" && { echo DONE; exit 0; }
  sleep 600
done
echo TIMEOUT
