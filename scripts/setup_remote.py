# Run via `colab exec -s b6-101 -f setup_remote.py --timeout 900`: installs deps and records GPU info.
import subprocess
subprocess.run('pip install -q -U transformers peft datasets accelerate bitsandbytes torchao 2>&1 | tail -3', shell=True)
print(subprocess.run('nvidia-smi', shell=True, capture_output=True, text=True).stdout)
import importlib.metadata as md, torch
print(torch.cuda.get_device_name(), {x: md.version(x) for x in ["torch","transformers","peft","bitsandbytes","torchao"]})
