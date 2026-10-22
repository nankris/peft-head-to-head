# `colab exec -s peft-head-to-head -f poll.py`: key lines of the progress log + nvidia-smi memory line
import os, subprocess
p = "/content/peft-results/progress.log"
if os.path.exists(p):
    L = open(p).readlines(); print("".join(l[:300] for l in L if "{'run'" in l or "FAILED" in l or "exited" in l or "DONE" in l)); print(L[-1][:300])
print(subprocess.run("nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv,noheader", shell=True, capture_output=True, text=True).stdout)
