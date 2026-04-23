# MatchaTTS Production Deployment - summary
Do these for each release.

1. Replace *26.04.23-1* with the current tag.
2. Replace *logs/train/v17/checkpoint_epoch=603.ckpt* with the path to the image you want to release. 

3. Build the image and test it locally
```bash
# Linux
python -m matcha.utils.prepare_ckpt_for_release logs/train/v17/checkpoint_epoch=603.ckpt
export TAG=26.04.23-1
export REGISTRY=678811077621.dkr.ecr.eu-west-1.amazonaws.com
export IMAGE_NAME=evie/matcha
docker buildx build -f docker/Dockerfile -t $REGISTRY/$IMAGE_NAME:$TAG .
docker run -p 8000:8000 --gpus all --name matcha 678811077621.dkr.ecr.eu-west-1.amazonaws.com/$IMAGE_NAME:$TAG
```
Test with Postman locally.

4Push the image to ECR and clean up local environment:
```bash
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $REGISTRY
docker push $REGISTRY/$IMAGE_NAME:$TAG
docker container remove matcha
docker image prune -f
docker container prune -f
docker builder prune -f
```

5Log into the remote EC2 machine
```bash
ssh -i ~/.ssh/ec2-connect-key-ireland.pem ec2-user@ec2-34-247-83-140.eu-west-1.compute.amazonaws.com
```

6Pull the image and do a rolling update:
```bash
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $REGISTRY
export TAG=26.04.23-1
export REGISTRY=678811077621.dkr.ecr.eu-west-1.amazonaws.com
export IMAGE_NAME=evie/matcha
docker pull $REGISTRY/$IMAGE_NAME:$TAG
docker service update --update-delay 120s --image $REGISTRY/$IMAGE_NAME:$TAG matcha
```
Test with Postman in EC2

7Clean up the remove environment.
```bash
docker container prune
docker image prune -a
```

---

# MatchaTTS Production Deployment - full docs

## Prerequisites

```bash
sudo apt install unzip
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install
aws configure
# It will ask for: 
# - Region: eu-west-1
# - Access Key ID: use the key called "AWS CLI Access Key" from truekey
```


### Local development
```
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo apt install docker.io
sudo systemctl restart docker
```
Install buildx from Docker's official repo (not in Ubuntu's default apt)
```
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /usr/share/keyrings/docker.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list
sudo apt update && sudo apt install docker-buildx-plugin
```

### In AWS
- AWS EC2 g4dn.xlarge instance (T4 GPU, 16GB VRAM)
- Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)
- Docker with NVIDIA runtime configured
- AWS ECR access configured

## AWS Resources
- ECR Repository: https://eu-west-1.console.aws.amazon.com/ecr/private-registry/repositories?region=eu-west-1
- EC2 Instances: https://eu-west-1.console.aws.amazon.com/ec2/home?region=eu-west-1#Instances
- SSH Keys: https://eu-west-1.console.aws.amazon.com/ec2/home?region=eu-west-1#KeyPairs
- Security Groups: https://eu-west-1.console.aws.amazon.com/ec2/home?region=eu-west-1#SecurityGroups

## Building and testing

Test the Docker image locally with GPU support.
See: https://docker-desktop.io/docs/docker/gpu

```bash
# Linux
export TAG=26.04.23-1
export REGISTRY=678811077621.dkr.ecr.eu-west-1.amazonaws.com
export IMAGE_NAME=evie/matcha

# Prepare the checkpoint and copy it to the docker folder
python -m matcha.utils.prepare_ckpt_for_release logs/train/v17/checkpoint_epoch=603.ckpt

# Build docker image
docker buildx build -f docker/Dockerfile -t $REGISTRY/$IMAGE_NAME:$TAG .

# Run it and do a quick test with Postman
docker run -p 8000:8000 --gpus all --name matcha 678811077621.dkr.ecr.eu-west-1.amazonaws.com/$IMAGE_NAME:$TAG

# Log into container
docker exec -it matcha /bin/bash
```

## Cleanup docker caches
Docker holds huge amounts of data. Check it with: 
```
docker system df
```

You can reclaim the space, but the next build command will be slow and will download some of those files again:
```
docker system prune -a --volumes -f
```

## Push to ECR


```bash
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $REGISTRY
docker push $REGISTRY/$IMAGE_NAME:$TAG

# Stop and remove the container I started for testing the image. 
docker container remove matcha

# Remove dangling images only (untagged intermediate layers from previous builds)
docker image prune -f

# Remove stopped containers
docker container prune -f

# Remove unused build cache (keeps downloaded base layers)
docker builder prune -f
```

## EC2 Instance Setup

### SSH Access
Log into EC2 using ssh (Linux/Mac).
Username: `ec2-user`

May need to allow inbound access on port 22 for my public IP in Security Groups in EC2.
Public IP changes periodically.

```bash
cp /mnt/e/Programare/eVoiceReader/Resources/ec2-connect-key-ireland.pem ~/.ssh/
ssh -i ~/.ssh/ec2-connect-key-ireland.pem ec2-user@ec2-34-247-83-140.eu-west-1.compute.amazonaws.com
```

### One-Time Docker Swarm Setup

1. **Configure Docker daemon for GPU access:**
```bash
sudo nano /etc/docker/daemon.json
```
Add:
```json
{
    "runtimes": {
        "nvidia": {
            "args": [],
            "path": "nvidia-container-runtime"
        }
    },
    "default-runtime": "nvidia",
    "node-generic-resources": [
        "NVIDIA-GPU=0"
    ]
}
```

2. **Enable Swarm resource in NVIDIA runtime:**
```bash
sudo nano /etc/nvidia-container-runtime/config.toml
```
Uncomment this line:
```toml
swarm-resource = "DOCKER_RESOURCE_GPU"
```

3. **Restart Docker:**
```bash
sudo systemctl restart docker.service
```

4. **Configure AWS and initialize Swarm:**
```bash
sudo yum update -y
sudo yum clean all
aws configure
docker swarm init
```

### ECR Access from EC2 (if not using aws configure)

Attach IAM role to EC2 instance with `AmazonEC2ContainerRegistryReadOnly` policy.

Create VPC Interface Endpoints:
- `com.amazonaws.eu-west-1.ecr.api`
- `com.amazonaws.eu-west-1.ecr.dkr`

Both in EC2 VPC, with their own security group allowing:
- Inbound: Port 443 from EC2 security group
- EC2 security group must allow outbound to endpoints on port 443

## Deployment

### Initial Deployment
Log into the remote EC2 machine:
```bash
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $REGISTRY
ssh -i ~/.ssh/ec2-connect-key-ireland.pem ec2-user@ec2-34-247-83-140.eu-west-1.compute.amazonaws.com
```

Create the service (only needs to be done once):
```bash
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $REGISTRY
export TAG=26.04.23-1
export REGISTRY=678811077621.dkr.ecr.eu-west-1.amazonaws.com
export IMAGE_NAME=evie/matcha

docker pull $REGISTRY/$IMAGE_NAME:$TAG

# Create service with 3 replicas
docker service create \
  --name matcha \
  --replicas 2 \
  --publish 8881:8000 \
  --env CHECKPOINT_PATH=/app/models/checkpoint.ckpt \
  --env MAX_TEXT_LENGTH=500 \
  --generic-resource "NVIDIA-GPU=0" \
  --update-delay 120s \
  --update-parallelism 1 \
  --update-order stop-first \
  $REGISTRY/$IMAGE_NAME:$TAG
```

### Rolling Updates (Zero Downtime)
```bash
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $REGISTRY
ssh -i ~/.ssh/ec2-connect-key-ireland.pem ec2-user@ec2-34-247-83-140.eu-west-1.compute.amazonaws.com
```

```bash
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $REGISTRY
export TAG=26.04.23-1
export REGISTRY=678811077621.dkr.ecr.eu-west-1.amazonaws.com
export IMAGE_NAME=evie/matcha

docker pull $REGISTRY/$IMAGE_NAME:$TAG

# Update service (rolling update with 20s delay)
docker service update --update-delay 120s --image $REGISTRY/$IMAGE_NAME:$TAG matcha

# Remove stopped containers
docker container prune

# Reclaim space occupied by images not used by any container
docker image prune -a
```

## Monitoring and Management

```bash
# Watch GPU usage
watch -d -n 0.3 nvidia-smi

# View service logs
docker service logs -f matcha

# Check service status
docker service ps matcha

# Scale service
docker service update matcha --replicas 3

# List containers
docker container ls

# Exec into container
docker exec -it <container_id> bash

# Rollback to previous version
docker service rollback matcha

# Remove service
docker service rm matcha

# Leave swarm (cleanup)
docker swarm leave --force
```

## Testing

Access the API:
- Health check: `http://<ec2-public-dns>:8000/health`
- API endpoint: `http://<ec2-public-dns>:8000/api/v1/speak`

Ensure Security Group allows inbound traffic on port 8000 from your IP.

## Troubleshooting

### Service won't start
```bash
docker service logs matcha
docker service ps matcha --no-trunc
```

### GPU not available
```bash
# Verify NVIDIA runtime
docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi

# Check daemon config
```bash
cat /etc/docker/daemon.json
```


## Notes

- Each replica runs 1 uvicorn worker with the full model
- Swarm's built-in load balancer distributes requests across replicas
- Rolling updates start new container before stopping old one (brief VRAM spike)
- Health checks ensure traffic only goes to healthy containers
- `--update-order start-first` ensures zero downtime during updates