# Run via `colab exec -s b6-101 -f launch.py` after uploading train.py to /content: starts training in the background.
import subprocess
subprocess.Popen("cd /content && nohup python train.py >> /content/train_stdout.log 2>&1 &", shell=True)
print("launched")
