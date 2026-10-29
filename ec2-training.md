# Test training on an EC2 g7e.2xlarge in eu-central-1

g7e.2xlarge: 1x NVIDIA RTX PRO 6000 Server (Blackwell, 96 GB VRAM), 8 vCPU, 64 GiB RAM, 1.9 TB local NVMe (wiped on stop).
eu-central-1 offers it in zones 1a and 1b, and the G vCPU quota there is 768, so no quota request is needed.

One instance with a 150 GB root volume holds code, venv, data and logs. Stopping the instance keeps the volume,
so you pay only for storage (EBS) until the next session.

What is reused from the production setup (`docker/BUILD_AND_DEPLOY.md`, `## EC2 Instance Setup`):
- The AMI family: Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023), user `ec2-user`. The current
  release lists G7e as supported.
- The Frankfurt key pair `ec2-connect-key-fra`, already in eu-central-1. Its private key is
  `/mnt/e/Programare/eVoiceReader/Resources/ec2-connect-key-frankfurt.ppk` (PuTTY format), converted once below.
- Your SSH-from-home rule (port 22 from 128.127.123.203/32). Security groups are per region and eu-central-1 has only
  the default group with no inbound rules, so the rule is recreated in a new group.
- The default VPC in eu-central-1 (`vpc-b60da3df`); its subnets assign public IPs automatically.

Not needed: IAM role, ECR endpoints, `aws configure` on the instance. Keep AWS credentials off the training machine.

Run the commands from your local WSL shell unless they say "on the instance". Set the region in every new local shell;
otherwise the CLI uses your default region eu-west-1, where production runs:
```bash
export AWS_REGION=eu-central-1
```

## 1. One-time setup

Convert the Frankfurt key to OpenSSH format, the way `ec2-connect-key-ireland.pem` is used:
```bash
puttygen /mnt/e/Programare/eVoiceReader/Resources/ec2-connect-key-frankfurt.ppk \
         -O private-openssh -o ~/.ssh/ec2-connect-key-frankfurt.pem
chmod 600 ~/.ssh/ec2-connect-key-frankfurt.pem
```

Check that it is the private key of `ec2-connect-key-fra`; the two fingerprints must be equal:
```bash
openssl pkcs8 -in ~/.ssh/ec2-connect-key-frankfurt.pem -topk8 -nocrypt -outform DER | openssl sha1 -c
aws ec2 describe-key-pairs --key-names ec2-connect-key-fra --query 'KeyPairs[0].KeyFingerprint' --output text
```
If they differ, import the Ireland public key instead and use `--key-name ec2-connect-key` with the Ireland `.pem` below:
```bash
ssh-keygen -y -f ~/.ssh/ec2-connect-key-ireland.pem > ~/.ssh/ec2-connect-key.pub
aws ec2 import-key-pair --key-name ec2-connect-key --public-key-material fileb://$HOME/.ssh/ec2-connect-key.pub
```

Security group with only the SSH rule. Do not copy the production 8880-8881 rules:
```bash
aws ec2 create-security-group --group-name matcha-training --description "SSH for matcha training" --vpc-id vpc-b60da3df
aws ec2 authorize-security-group-ingress --group-name matcha-training --protocol tcp --port 22 --cidr 128.127.123.203/32
```

In any new shell, look the group ID up by name (the launch in step 2 needs `$SG`):
```bash
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=matcha-training \
     --query 'SecurityGroups[0].GroupId' --output text)
echo "$SG"
```

My sg is: sg-0883d2ddbaedf06d7

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
  --instance-type g7e.2xlarge \
  --key-name ec2-connect-key-fra \
  --security-group-ids "$SG" \
  --placement AvailabilityZone=eu-central-1a \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":150,"VolumeType":"gp3","Throughput":250,"DeleteOnTermination":false}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=matcha-training}]' \
                       'ResourceType=volume,Tags=[{Key=Name,Value=matcha-training}]' \
  --query 'Instances[0].InstanceId' --output text)
echo "$INSTANCE_ID"   # note it down

aws ec2 wait instance-running --instance-ids "$INSTANCE_ID"
aws ec2 describe-instances --instance-ids "$INSTANCE_ID" \
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text
```

My instance id is: i-00564a7cd9a0022a4
My instance is: ec2-52-59-248-26.eu-central-1.compute.amazonaws.com

If the launch fails with `InsufficientInstanceCapacity`, run it again with `AvailabilityZone=eu-central-1b`.

Add the host to `~/.ssh/config`. The public DNS name changes on every stop/start, so update `HostName` each session (see step 9).
```
Host g7e
    HostName ec2-52-59-248-26.eu-central-1.compute.amazonaws.com
    User ec2-user
    IdentityFile ~/.ssh/ec2-connect-key-frankfurt.pem
```

First login, and the tools the next steps need (rsync must exist on both ends):
```bash
ssh g7e
sudo dnf install -y rsync tmux htop git
exit
```

## 3. Copy the code

Training needs only four things: the `matcha` package, the Hydra `configs` (`matcha/train.py` loads them from
`../configs`), and `pyproject.toml` + `uv.lock` for `uv sync`. `.git` is not copied: `.git/config` contains the
Bitbucket access token in the remote URL. The copies include your uncommitted changes.

```bash
cd ~/projects/matcha-tts
ssh g7e mkdir -p matcha-tts
rsync -av pyproject.toml uv.lock g7e:matcha-tts/
rsync -av --delete --exclude __pycache__ matcha configs g7e:matcha-tts/
```

`--exclude __pycache__` only skips local Python bytecode caches (about 0.7 MB); the instance builds its own.
`--delete` removes files on the instance that you deleted locally, so `matcha` and `configs` stay identical to yours.
Re-run the same commands after local code changes; rsync sends only the differences.
Training reads phoneme IDs from the CSVs, so eSpeak is not needed on the instance.

## 4. Copy the data

Copy the two CSVs and the precomputed mels (16 GB).
Nothing needs to be rebuilt on the instance. `--partial` lets you resume an interrupted upload by running the command again.

```bash
ssh g7e mkdir -p matcha-tts/data/corpus-24k
rsync -av data/corpus-24k/train.csv data/corpus-24k/validate.csv g7e:matcha-tts/data/corpus-24k/
rsync -av --partial --info=progress2 data/corpus-24k/mels g7e:matcha-tts/data/corpus-24k/
ssh g7e du -sh matcha-tts/data/corpus-24k/mels   # 16G, as locally
```

Training reads only the CSVs (phoneme IDs) and the mels (`.npy` and `.fine.npy`). The wavs (6.5 GB) are used only by
corpus tools and offline validation scripts (`precompute_mels`, `mcd_validate_*`, `utmos_validate`, ...).
If you want to run those on the instance later:
```bash
rsync -av --partial --info=progress2 data/corpus-24k/wav g7e:matcha-tts/data/corpus-24k/
```

## 5. Python environment (on the instance)

```bash
ssh g7e

curl -LsSf https://astral.sh/uv/install.sh | sh
uv --version   # Amazon Linux already has ~/.local/bin on PATH, so the installer does not create ~/.local/bin/env

cd ~/matcha-tts
uv venv .venv --python 3.13      # uv downloads Python 3.13 itself
source .venv/bin/activate
uv sync --active
```

Check the driver and torch. torch is pinned to `+cu132`, so `nvidia-smi` must report "CUDA Version" 13.2 or higher:
```bash
nvidia-smi

python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```
Expected: `2.12.1+cu132 True NVIDIA RTX PRO 6000 ... (12, 0)`. If the driver is too old, relax the pins in `pyproject.toml` on the instance only.

Every new SSH session needs:
```bash
cd ~/matcha-tts && source .venv/bin/activate
```

## 6. Run training (on the instance)

### 6.1 Experiment config for the g7e

Edit `configs/experiment/v24-g7e.yaml` locally (a copy of `v24.yaml`). Settings worth changing:
```yaml
run_name: v24-g7e          # logs go to logs/train/v24-g7e, not mixed with v24
ckpt_path: null            # from scratch; v24 points at a local checkpoint. Set a path to resume (see step 4).

data:
  num_workers: 7           # g7e.2xlarge has 8 vCPUs, leave one for the main process
  max_frames_per_batch: 20000   # same as local v24 for a direct speed comparison; see 6.3 for larger batches
```

Upload it (re-run after every local edit):
```bash
rsync -av configs/experiment/v24-g7e.yaml g7e:matcha-tts/configs/experiment/
```

The step 3 command (`rsync ... matcha configs`) uploads it too, together with any other code changes.
It uses `--delete`, so always edit configs locally: a file that exists only on the instance would be removed.

### 6.2 Start training

Run inside tmux so training survives a dropped SSH connection (`Ctrl+B, D` detaches, `tmux attach -t train` reconnects):
```bash
tmux new -s train
cd ~/matcha-tts && source .venv/bin/activate
python -m matcha.train +experiment=v24-g7e
```

It runs until you stop it with `Ctrl+C` (`max_epochs: -1`). Lightning stops cleanly; checkpoints are saved every 5 epochs.
Stop the instance afterwards (step 9.1), otherwise it keeps billing.

The first epoch includes torch.compile warm-up and the first read of the mels from disk; compare epochs 2+.
Validation runs every 5 epochs and is slower than a training epoch, so compare training epochs only.
The progress bar shows it/s and the epoch time. Compare with your local v24 epochs, about 4 minutes at 20000 frames.

For v25, do the same with a copy `v25-g7e.yaml`. v25 inherits from `v24` (not `v24-g7e`), so set `run_name`,
`ckpt_path: null` and the `data` settings in the copy as well.

### 6.3 Larger batches

Locally, `max_frames_per_batch: 20000` uses about 18 GB of VRAM; the 96 GB card fits roughly 4x more.
Change it in `v24-g7e.yaml`, e.g. to 60000, upload, and start again.

Larger batches give fewer optimizer steps per epoch, so compare time per epoch, not it/s.
They also change training dynamics (you may need a higher LR, `model.optimizer.lr`), so results are not directly
comparable with your local runs. For a pure speed test that does not matter.
Increase in steps (60000, 80000) and watch `memory.used` in `nvidia-smi`. Leave headroom: batch length varies across epochs.

## 7. Monitor (second SSH session)

GPU utilization, memory and power, every second:
```bash
ssh g7e

watch -n 1 nvidia-smi
# or a compact rolling log: power (p), utilization (u), clocks (c), memory (m)
nvidia-smi dmon -s pucm
```

If GPU utilization stays well below 100%, the dataloader cannot keep up. Check the CPUs with `htop`.
If all 8 are busy, a bigger instance (g7e.4xlarge) is the fix, not more workers.

TensorBoard through an SSH tunnel, then open http://localhost:6006 locally:
```bash
ssh -L 6007:localhost:6007 g7e

cd ~/matcha-tts && source .venv/bin/activate
tensorboard --logdir logs/train --port 6007
```

## 8. Bring results back (local)

```bash
rsync -av --partial g7e:matcha-tts/logs/ logs/ec2/
```

Checkpoints are 1.2 GB each; add `--exclude '*.ckpt'` if you only need the TensorBoard logs.

## 9. Stop, restart, clean up

All commands below need `export AWS_REGION=eu-central-1`. In eu-west-1 they would only fail with "instance not found",
since instance IDs are per region, but setting it avoids any confusion with the production machine.

Forgot the instance ID?
```bash
INSTANCE_ID=$(aws ec2 describe-instances --filters Name=tag:Name,Values=matcha-training \
  Name=instance-state-name,Values=pending,running,stopping,stopped \
  --query 'Reservations[0].Instances[0].InstanceId' --output text)
echo "$INSTANCE_ID"
```

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
  --query 'Reservations[0].Instances[0].PublicDnsName' --output text
```

A stopped instance does not reserve its GPU. If `start-instances` fails with `InsufficientInstanceCapacity`,
try again later, or use 9.3 to move the volume to the other zone.

To try another GPU on the same volume, change the type while stopped (the zone stays the same), e.g. the L40S:
```bash
aws ec2 modify-instance-attribute --instance-id "$INSTANCE_ID" --instance-type '{"Value":"g6e.4xlarge"}'
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

### 9.3 New instance from the kept storage

An existing volume cannot be the root disk of a new instance directly. Make a snapshot of it, register the snapshot
as an AMI, and launch from that AMI. The new instance gets an exact copy, with venv, data and logs.
This also works in another zone (e.g. eu-central-1b when 1a has no capacity).
```bash
VOL_ID=$(aws ec2 describe-volumes --filters Name=tag:Name,Values=matcha-training Name=status,Values=available \
  --query 'Volumes[0].VolumeId' --output text)

SNAP_ID=$(aws ec2 create-snapshot --volume-id "$VOL_ID" --description "matcha-training root" \
  --tag-specifications 'ResourceType=snapshot,Tags=[{Key=Name,Value=matcha-training}]' \
  --query SnapshotId --output text)
aws ec2 wait snapshot-completed --snapshot-ids "$SNAP_ID"   # can take a while for 150 GB

AMI_ID=$(aws ec2 register-image --name "matcha-training-$(date +%Y%m%d-%H%M)" \
  --architecture x86_64 --root-device-name /dev/xvda --virtualization-type hvm --ena-support --boot-mode uefi-preferred \
  --block-device-mappings "[{\"DeviceName\":\"/dev/xvda\",\"Ebs\":{\"SnapshotId\":\"$SNAP_ID\",\"VolumeType\":\"gp3\",\"Throughput\":250,\"DeleteOnTermination\":false}}]" \
  --query ImageId --output text)
echo "$AMI_ID"
```

Then run the launch command from step 2 with `--image-id "$AMI_ID"` instead of the `resolve:ssm:...` value,
and without the `--block-device-mappings` line (the AMI already defines the volume). Once the new instance works,
delete the old volume (9.4); the snapshot and AMI can be deleted too, or kept as a backup.

### 9.4 Delete the storage (only when you no longer need the data)

This deletes the venv, data and logs on AWS. Copy back anything you want first (step 8).
```bash
aws ec2 describe-volumes --filters Name=tag:Name,Values=matcha-training \
  --query 'Volumes[].[VolumeId,State]' --output text
aws ec2 delete-volume --volume-id <vol-id>      # only volumes in state "available"
```

Snapshots and AMIs from 9.3, if any:
```bash
aws ec2 describe-images --owners self --filters 'Name=name,Values=matcha-training-*' \
  --query 'Images[].[ImageId,Name]' --output text
aws ec2 deregister-image --image-id <ami-id>
aws ec2 describe-snapshots --owner-ids self --filters Name=tag:Name,Values=matcha-training \
  --query 'Snapshots[].[SnapshotId,StartTime]' --output text
aws ec2 delete-snapshot --snapshot-id <snap-id>
```

When you are finished with EC2 training altogether, also remove the security group:
```bash
aws ec2 delete-security-group --group-name matcha-training
```
