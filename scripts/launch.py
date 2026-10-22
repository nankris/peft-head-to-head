# Run via `colab exec -s peft-head-to-head -f launch.py` after uploading train.py to /content: starts training in the background.
import subprocess
subprocess.Popen("cd /content && nohup python train.py >> /content/train_stdout.log 2>&1 &", shell=True)
print("launched")
