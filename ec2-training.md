# Test training on an EC2

The instance store is not the root disk and is wiped on every stop. The root disk is the EBS volume from
`--block-device-mappings` in step 2 (the AMI default would be only 75 GB and deleted on terminate).

Stopping the instance keeps the volume, so you pay only for storage (EBS) until the next session.

The guide works in any region, including one where the account has nothing yet (e.g. us-east-1, when eu-central-1
has no g5 capacity). Key pairs, security groups, instances, volumes, snapshots and AMIs are all per region, so
step 1 is done once per region. Everything is created with the name `matcha-training`, so the commands are the
same in every region.

What is reused from the production setup (`docker/BUILD_AND_DEPLOY.md`, `## EC2 Instance Setup`):
- The AMI family: Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023), user `ec2-user`. Step 2 resolves
  the latest AMI ID of the current region through SSM, so nothing is hard-coded.
- Your SSH-from-home rule (port 22 from 128.127.123.203/32), recreated in a new security group.
- The default VPC of the region; its subnets assign public IPs automatically.

Not reused: the production key pairs (`ec2-connect-key-fra`, `ec2-connect-key`). The training machine gets its own
SSH key (step 1.1), so nothing from production is shared with it.

Not needed: IAM role, ECR endpoints, `aws configure` on the instance. Keep AWS credentials off the training machine.

Run the commands from your local WSL shell unless they say "on the instance". Set the region in every new local shell;
otherwise the CLI uses your default region eu-west-1, where production runs:
```bash
export AWS_REGION=us-east-1
```

## 1. One-time setup

### 1.1 SSH key (local, once)

Create a dedicated key pair locally. The private key never leaves your machine; AWS only gets the public key:
```bash
ssh-keygen -t ed25519 -f ~/.ssh/matcha-training -C matcha-training
```
Press Enter for an empty passphrase, or set one and use `ssh-add`. This creates `~/.ssh/matcha-training` (private,
mode 600) and `~/.ssh/matcha-training.pub` (public).

### 1.2 Per region (once per region)

Check that the region offers g6e.2xlarge, and in which zones:
```bash
aws ec2 describe-instance-type-offerings --location-type availability-zone \
  --filters Name=instance-type,Values=g6e.2xlarge --query 'InstanceTypeOfferings[].Location' --output text
```

Import the public key under the same name in every region:
```bash
aws ec2 import-key-pair --key-name matcha-training --public-key-material fileb://$HOME/.ssh/matcha-training.pub
```

Look up the default VPC. 
```bash
VPC_ID=$(aws ec2 describe-vpcs --filters Name=is-default,Values=true --query 'Vpcs[0].VpcId' --output text)
echo "$VPC_ID"   # us-east-1: vpc-a78c76c3
```
If the output is `None`, the region has no default VPC; create one with
`aws ec2 create-default-vpc`
and do the lookup again.

Security group with only the SSH rule. Do not copy the production 8880-8881 rules:
```bash
aws ec2 create-security-group --group-name matcha-training --description "SSH for matcha training" --vpc-id "$VPC_ID"
aws ec2 authorize-security-group-ingress --group-name matcha-training --protocol tcp --port 22 --cidr 128.127.123.203/32
```

In eu-central-1 the `matcha-training` group already exists from the first setup; there, only import the key.
Instances launched earlier with `ec2-connect-key-fra` keep that key; a key pair is applied only at launch.

In any new shell, look the group ID up by name (the launch in step 2 needs `$SG`):
```bash
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=matcha-training \
     --query 'SecurityGroups[0].GroupId' --output text)
echo "$SG"
```

Your public IP changes periodically. When SSH times out, add the new one:
```bash
aws ec2 authorize-security-group-ingress --group-name matcha-training --protocol tcp --port 22 \
    --cidr "$(curl -s https://checkip.amazonaws.com)/32"
```

## 2. Launch the instance

`DeleteOnTermination=false` keeps the volume even if the instance is terminated by mistake.
```bash
INSTANCE_ID=$(aws ec2 run-instances \
  --image-id resolve:ssm:/aws/service/deeplearning/ami/x86_64/base-oss-nvidia-driver-gpu-amazon-linux-2023/latest/ami-id \
  --instance-type g6e.2xlarge \
  --key-name matcha-training \
  --security-group-ids "$SG" \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":150,"VolumeType":"gp3","Throughput":250,"DeleteOnTermination":false}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=matcha-training}]' \
                       'ResourceType=volume,Tags=[{Key=Name,Value=matcha-training}]' \
  --query 'Instances[0].InstanceId' --output text)
echo "$INSTANCE_ID"   # note it down:  i-0f4fe440d8be00680

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text    # note it down: ec2-98-92-185-226.compute-1.amazonaws.com
```

If the launch fails with `InsufficientInstanceCapacity`, run it again, or pick another zone from step 1.2 by adding
this line to the command (EC2 then uses the default subnet of that zone):
```bash
  --placement AvailabilityZone=us-east-1b \
```
The volume is created in the same zone as the instance and stays there.

Add the host to `~/.ssh/config`. The public DNS name changes on every stop/start, so update `HostName` each session (see step 9).
```
Host ec2-training-machine
    HostName ec2-3-89-147-7.compute-1.amazonaws.com
    User ec2-user
    IdentityFile ~/.ssh/matcha-training
    IdentitiesOnly yes
```
`IdentitiesOnly yes` stops ssh from offering your other keys first. Public IPs are reused across instances, so on a
new instance ssh may warn that the host key changed; remove the old entry with `ssh-keygen -R <hostname>`.

First login, and the tools the next steps need (rsync must exist on both ends):
```bash
ssh ec2-training-machine

sudo dnf install -y --disablerepo=cuda-rhel9-x86_64 rsync tmux htop git
exit
```
`--disablerepo` skips NVIDIA's CUDA repo, which the AMI configures but these packages do not need; its metadata
sometimes returns 404. Ignore the "newer release of Amazon Linux" notice: upgrading can change the kernel and driver.

## 3. Copy the code

Training needs only four things: the `matcha` package, the Hydra `configs` (`matcha/train.py` loads them from
`../configs`), and `pyproject.toml` + `uv.lock` for `uv sync`. `.git` is not copied: `.git/config` contains the
Bitbucket access token in the remote URL. The copies include your uncommitted changes.

`pyproject.toml` pins torch 2.15.0+cu134, so the instance gets the same build as your local `.venv`.
```bash
cd ~/projects/matcha-tts
ssh ec2-training-machine mkdir -p matcha-tts
rsync -av pyproject.toml uv.lock ec2-training-machine:matcha-tts/
rsync -av --delete --exclude __pycache__ matcha configs ec2-training-machine:matcha-tts/
```

`--exclude __pycache__` only skips local Python bytecode caches (about 0.7 MB); the instance builds its own.
`--delete` removes files on the instance that you deleted locally, so `matcha` and `configs` stay identical to yours.
Re-run the same commands after local code changes; rsync sends only the differences.
Training reads phoneme IDs from the CSVs, so eSpeak is not needed on the instance.

## 4. Copy the data

Copy the two CSVs and the precomputed mels (16 GB).
Nothing needs to be rebuilt on the instance. `--partial` lets you resume an interrupted upload by running the command again.

```bash
ssh ec2-training-machine mkdir -p matcha-tts/data/corpus-24k
rsync -av data/corpus-24k/train.csv data/corpus-24k/validate.csv ec2-training-machine:matcha-tts/data/corpus-24k/
rsync -av --partial --info=progress2 data/corpus-24k/mels ec2-training-machine:matcha-tts/data/corpus-24k/
ssh ec2-training-machine du -sh matcha-tts/data/corpus-24k/mels   # 16G, as locally
```

Training reads only the CSVs (phoneme IDs) and the mels (`.npy` and `.fine.npy`). The wavs (6.5 GB) are used only by
corpus tools and offline validation scripts (`precompute_mels`, `mcd_validate_*`, `utmos_validate`, ...).
If you want to run those on the instance later:
```bash
rsync -av --partial --info=progress2 data/corpus-24k/wav ec2-training-machine:matcha-tts/data/corpus-24k/
```

### 4.1 Resume from an earlier EC2 run (optional)

Local `logs/ec2/` mirrors the instance's `logs/`, with the trainer's own layout
(`train/<run_name>/runs/<start time>/`, with `checkpoints/`, `tensorboard/` and `train.log` inside), so a run has the
same relative path everywhere:
```
local:    logs/ec2/train/v24/runs/2026-10-28_11-35-09/checkpoints/checkpoint_epoch=114.ckpt
instance: logs/train/v24/runs/2026-10-28_11-35-09/checkpoints/checkpoint_epoch=114.ckpt
```
Locally it holds the TensorBoard events and `train.log` of every run, plus the checkpoint to resume from. Before
uploading, delete all other local checkpoints, so the upload sends only that one (1.2 GB).

To continue training, upload the whole run-name folder, so TensorBoard on the instance shows the history from the
first epoch:
```bash
ssh ec2-training-machine mkdir -p matcha-tts/logs/train
rsync -av --partial --info=progress2 logs/ec2/train/v24 ec2-training-machine:matcha-tts/logs/train/
ssh ec2-training-machine ls -l 'matcha-tts/logs/train/v24/runs/*/checkpoints'   # only the checkpoint to resume from
```

## 5. Python environment (on the instance)

```bash
ssh ec2-training-machine

curl -LsSf https://astral.sh/uv/install.sh | sh

cd ~/matcha-tts
uv venv .venv --python 3.13      # uv downloads Python 3.13 itself
source .venv/bin/activate
uv sync --active --locked        # --locked: install exactly uv.lock, fail if it does not match pyproject.toml
```
After every later upload of `pyproject.toml` and `uv.lock` (step 3), run `uv sync --active --locked` again; uv updates
only the packages that changed.

The AMI's `/etc/profile.d/dlami.sh` puts the system CUDA 13.2 libraries, including cuDNN 9.20, on `LD_LIBRARY_PATH`
in every login shell. torch then loads part of cuDNN from there and part from the venv (9.26), and every convolution
fails with `CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED` ("Unable to load any of {libcudnn_engines_runtime_compiled.so...}").
torch brings all its CUDA libraries as wheels in the venv, so clear the variable for all future shells (once):
```bash
cat >> ~/.bashrc <<'EOF'

# matcha: drop DLAMI CUDA paths; torch uses the CUDA/cuDNN wheels in the venv.
unset LD_LIBRARY_PATH
EOF
exit
```
Log in again (and start tmux sessions after this), then check.

Check the driver and torch:
```bash
nvidia-smi

cd ~/matcha-tts
source .venv/bin/activate
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
python -c "import torch; print(torch.backends.cudnn.version()); x=torch.randn(1,1,64,64,device='cuda'); print(torch.nn.Conv2d(1,1,3).cuda()(x).shape)"
```
Expected: `2.15.0+cu134 True NVIDIA <card name>`, then a cuDNN version and `torch.Size`.

`nvidia-smi` reports "CUDA Version: 13.2" (driver 595), lower than the cu134 build. That works: within CUDA 13, a
newer runtime runs on an older 13.x driver (CUDA minor version compatibility). Your local driver also reports 13.2,
and the same torch build runs there. Only a "requires a newer driver" error at runtime would mean otherwise.

Every new SSH session needs:
```bash
cd ~/matcha-tts && source .venv/bin/activate
```

## 6. Run training (on the instance)

### 6.1 Experiment config for EC2

`configs/experiment/v24-ec2.yaml` is a copy of `v24.yaml` for EC2. Edit it locally. Settings that differ:
```yaml
run_name: v24              # same as locally; on the instance, logs go to logs/train/v24/runs/<start time>/

data:
  num_workers: 7           
  max_frames_per_batch: 40000  
```

To resume (step 4.1), point `ckpt_path` at the checkpoint's path on the instance (relative to `~/matcha-tts`, where
you start training); for a new run, use `ckpt_path: null`:
```yaml
ckpt_path: logs/train/v24/runs/2026-10-28_11-35-09/checkpoints/checkpoint_epoch=114.ckpt
```

Upload it (re-run after every local edit):
```bash
rsync -av configs/experiment/v24-ec2.yaml ec2-training-machine:matcha-tts/configs/experiment/
```

The step 3 command (`rsync ... matcha configs`) uploads it too, together with any other code changes.
It uses `--delete`, so always edit configs locally: a file that exists only on the instance would be removed.

### 6.2 Start training

Run inside tmux so training survives a dropped SSH connection (`Ctrl+B, D` detaches, `tmux attach -t train` reconnects):
```bash
tmux new -s train
cd ~/matcha-tts && source .venv/bin/activate
python -m matcha.train +experiment=v24-ec2
```

It runs until you stop it with `Ctrl+C` (`max_epochs: -1`). Lightning stops cleanly; checkpoints are saved every 5 epochs.
Stop the instance afterwards (step 9.1), otherwise it keeps billing.

The first epoch includes torch.compile warm-up and the first read of the mels from disk; compare epochs 2+.
Validation runs every 5 epochs and is slower than a training epoch, so compare training epochs only.
The progress bar shows it/s and the epoch time. Compare with your local v24 epochs, about 4 minutes at 20000 frames.

### 6.3 Batch size

Larger batches give fewer optimizer steps per epoch, so compare time per epoch, not it/s.
They also change training dynamics (you may need a higher LR, `model.optimizer.lr`), so results are not directly
comparable with your local runs. For a pure speed test that does not matter.
Increase in steps and watch `memory.used` in `nvidia-smi`. Leave headroom: batch length varies across epochs.

## 7. Monitor (second SSH session)

GPU utilization, memory and power, every second:
```bash
ssh ec2-training-machine

watch -n 1 nvidia-smi
# or a compact rolling log: power (p), utilization (u), clocks (c), memory (m)
nvidia-smi dmon -s pucm
```

If GPU utilization stays well below 100%, the dataloader cannot keep up. Check the CPUs with `htop`.
If all 8 are busy, a bigger instance (g5.4xlarge) is the fix, not more workers.

TensorBoard through an SSH tunnel, then open http://localhost:6007 locally:
```bash
ssh -L 6007:localhost:6007 ec2-training-machine

cd ~/matcha-tts && source .venv/bin/activate
tensorboard --logdir logs/train --port 6007
```

Copy previous events to the remote machine to see them in tensorboard:
```bash
rsync -av --exclude '*.ckpt' logs/train/v24/runs/2026-10-09_22-34-40 ec2-training-machine:matcha-tts/logs/train/v24/runs/
```


## 8. Bring results back (local)

At the end of a session, stop training first (`Ctrl+C` in tmux). `Ctrl+C` does not save a checkpoint; checkpoints are
written only every 5 epochs, so stop right after one appears, or the epochs since the last one are lost:
```bash
ssh ec2-training-machine 'ls -lt matcha-tts/logs/train/*/runs/*/checkpoints/ | head'
```

Then download the TensorBoard events and logs of all runs, and only the latest checkpoint:
```bash
rsync -av --exclude '*.ckpt' ec2-training-machine:matcha-tts/logs/ logs/ec2/

CKPT=$(ssh ec2-training-machine 'cd matcha-tts/logs && ls -t train/*/runs/*/checkpoints/*.ckpt | head -1')
echo "$CKPT"
rsync -av --partial --info=progress2 "ec2-training-machine:matcha-tts/logs/$CKPT" "logs/ec2/$CKPT"
```
The local copies of older checkpoints are not removed; delete them yourself once the new one has arrived.
Uploads to AWS are free, but data transfer out of AWS is billed per GB (about 1.2 GB per checkpoint).

## 9. Stop, restart, clean up

All commands below need `export AWS_REGION=<the region of the instance>`. In eu-west-1 they would only fail with
"instance not found", since instance IDs are per region, but setting it avoids any confusion with the production machine.

Forgot the instance ID?
```bash
INSTANCE_ID=$(aws ec2 describe-instances --filters Name=tag:Name,Values=matcha-training \
  Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
echo "$INSTANCE_ID"
```
`None` almost always means `AWS_REGION` is not set in this shell, so the CLI searched eu-west-1. Export it and retry.

### 9.1 Pause: stop the instance (use this between sessions)

A stopped instance is not billed for compute. Only the 150 GB volume is billed, exactly as if the instance were deleted.
Venv, data and logs stay on the volume, so the next session starts in a minute.
```bash
aws ec2 stop-instances --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-stopped --instance-ids "$INSTANCE_ID"
```

Resume, then put the new DNS name into `HostName` in `~/.ssh/config`:
```bash
aws ec2 start-instances --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text  # note it down: ec2-63-177-62-194.eu-central-1.compute.amazonaws.com
```

A stopped instance does not reserve its GPU. If `start-instances` fails with `InsufficientInstanceCapacity`,
try again later.

To try another instance type on the same volume, change the type while stopped (the zone stays the same).
The new type must be offered in the volume's zone (step 1.2); `start-instances` can still fail on capacity:
```bash
aws ec2 modify-instance-attribute --instance-id "$INSTANCE_ID" --instance-type '{"Value":"g6e.2xlarge"}'
```

### 9.2 Delete the instance, keep the storage

Terminating costs the same as stopping, so only do this when you are done with the instance itself.
The root volume survives (`DeleteOnTermination=false`) and becomes an unattached volume:
```bash
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID"
aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID"
aws ec2 describe-volumes --filters Name=tag:Name,Values=matcha-training \
  --query 'Volumes[].[VolumeId,State,AvailabilityZone]' --output text   # State: available
```

### 9.3 Delete the storage (only when you no longer need the data)

This deletes the venv, data and logs on AWS.
```bash
aws ec2 describe-volumes --filters Name=tag:Name,Values=matcha-training \
  --query 'Volumes[].[VolumeId,State]' --output text

# only volumes in state "available" (not attached to an instance) can be deleted
VOL_ID=$(aws ec2 describe-volumes --filters Name=tag:Name,Values=matcha-training Name=status,Values=available \
  --query 'Volumes[0].VolumeId' --output text)
aws ec2 delete-volume --volume-id "$VOL_ID"
```
If the listing shows more than one available volume, the lookup takes the first;
set `VOL_ID` by hand to the one you mean.

When you are finished with EC2 training altogether, also remove the security group and the key pair (in each region
you used). The security group can only be deleted after the instance is terminated (9.2):
```bash
aws ec2 delete-security-group --group-name matcha-training
aws ec2 delete-key-pair --key-name matcha-training
```

### 10 Conclusion

I have tested on an L4 (g6), and it is as slow as rtx 4000 sff ada.
I tested on a L40s (g6e) and got 1:54 minutes per epoch.
On a rtx pro 6000 (g7e) I got some trouble with cudnn but with cudnn disabled, I got about 2:15 minutes per epoch.
I also tested on a rtx pro 4500 (g7), 2:30 minutes per epoch.
On an A10g (g5), it takes around 3:50 minutes / epoch.
Training with the same code on my RTX 4000 sff ada takes 6:30 minutes / epoch (6:00 minutes if I disable ECC).

A10g - g5.2xlarge (equiv of rtx A4500)
L40s - g6e.2xlarge (equiv of rtx 6000 ada)
H100 p5.4xlarge (not found in us-east or eu-central)
RTX Pro 4500 - g7.2xlarge
RTX Pro 6000 - g7e.2xlarge 
V100 - p3.2xlarge (no longer supported by CUDA 13)

These are On-Demand Linux prices per hour, from the AWS Price List API:

| Instance    | GPU          | us-east-1 | eu-central-1 |
|-------------|--------------|-----------|--------------|
| g5.2xlarge  | A10G         | $1.21     | $1.52        |
| g6e.2xlarge | L40S         | $2.24     | $2.80        |
| g7.2xlarge  | RTX PRO 4500 | $2.52     | $3.15        |
| g7e.2xlarge | RTX PRO 6000 | $3.36     | $5.72        |
