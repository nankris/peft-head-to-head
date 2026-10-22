# `colab exec -s peft-head-to-head -f chain_retry_full.py`: on the VM, wait for the driver's first DONE, then re-run the driver once with
# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True so the only failed run (full_s1236, OOM from fragmentation at step 1) is retried.
import subprocess
sh = r'''
while ! grep -q "DONE" /content/peft-results/progress.log; do sleep 30; done
echo "$(date +%H:%M:%S) chain: retrying failed runs with expandable_segments" >> /content/peft-results/progress.log
cd /content && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python train.py >> /content/train_stdout.log 2>&1
echo "$(date +%H:%M:%S) chain: RETRY_DONE" >> /content/peft-results/progress.log
'''
open("/content/chain.sh", "w").write(sh)
subprocess.Popen("nohup bash /content/chain.sh > /content/chain.log 2>&1 &", shell=True)
print("chained")
