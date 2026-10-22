#!/bin/bash
# Retry `colab new` every 5 min for up to 2 h (account allows only 2 concurrent GPU sessions).
source /workspace/tools/setup.sh >/dev/null 2>&1
R="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../results" && pwd)"
for i in $(seq 1 25); do
  echo "== attempt $i $(date -u +%FT%TZ)" >> $R/session_attempts.log
  colab usage >> $R/session_attempts.log 2>&1
  if colab new --gpu L4 -s peft-head-to-head >> $R/session_attempts.log 2>&1 && colab status -s peft-head-to-head >> $R/session_attempts.log 2>&1; then
    date -u +%s > $R/session_start_epoch; colab usage > $R/usage_after_alloc.txt 2>&1; echo OK >> $R/session_attempts.log; exit 0
  fi
  sleep 300
done
echo GAVE_UP >> $R/session_attempts.log; exit 1
