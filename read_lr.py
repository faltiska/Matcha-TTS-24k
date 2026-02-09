import torch
import sys

if len(sys.argv) < 2:
    print("Usage: python read_lr.py <checkpoint.ckpt>")
    sys.exit(1)

ckpt = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
lr_schedulers = ckpt.get('lr_schedulers', [])

if lr_schedulers:
    for i, scheduler in enumerate(lr_schedulers):
        lr = scheduler.get('_last_lr', 'N/A')[0]
        print(f"Scheduler {i}: {lr:.1e}")
else:
    print("No LR schedulers found in checkpoint")

print(f"Epoch: {ckpt.get('epoch', 'N/A')}")
