# Smoke test: 64 train / 32 test items, one seed, all arms, into /content/smoke (not used in results).
import subprocess
subprocess.Popen("rm -rf /content/smoke; cd /content && OUT=/content/smoke N_TRAIN=64 N_TEST=32 SEEDS=1 nohup python train.py > /content/smoke_stdout.log 2>&1 &", shell=True)
print("smoke launched")
