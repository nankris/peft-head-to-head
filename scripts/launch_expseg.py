# VM4 launcher: same driver, but with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True from the start (allocator only;
# avoids the fragmentation OOM that hit full_s1236). The driver skips every run already marked ok in runs.json.
import subprocess
subprocess.Popen("cd /content && PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True nohup python train.py >> /content/train_stdout.log 2>&1 &", shell=True)
print("launched")
