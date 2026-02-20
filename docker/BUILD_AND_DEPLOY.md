# MatchaTTS Production Deployment

## Prerequisites

### Local development
```
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo apt install docker.io
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

## Local Testing (Optional)

Test the Docker image locally with GPU support.
See: https://docker-desktop.io/docs/docker/gpu

```bash
# Linux
export TAG=v1.0.0
docker buildx build -f docker/Dockerfile -t 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/matcha-tts:$TAG .
docker run -p 8000:8000 --gpus all --name matcha 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/matcha-tts:$TAG

# Log into container
docker exec -it matcha /bin/bash

# Cleanup
docker container remove matcha
docker image remove 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/matcha-tts:$TAG
```

## Build and Push to ECR

```powershell
# PowerShell (WSL)
$TAG="v1.0.0"
$REGISTRY="678811077621.dkr.ecr.eu-west-1.amazonaws.com"
$IMAGE_NAME="evie/matcha-tts"

# Build image (copy checkpoint into docker/ first)
cp /path/to/checkpoint.ckpt docker/checkpoint.ckpt
docker buildx build -f docker/Dockerfile -t "$REGISTRY/$IMAGE_NAME:$TAG" .

# Login to ECR
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $REGISTRY

# Push image
docker push "$REGISTRY/$IMAGE_NAME:$TAG"
```

```bash
# Linux/Mac
export TAG=v1.0.0
export REGISTRY=678811077621.dkr.ecr.eu-west-1.amazonaws.com
export IMAGE_NAME=evie/matcha-tts

# Build image (copy checkpoint into docker/ first)
cp /path/to/checkpoint.ckpt docker/checkpoint.ckpt
docker buildx build -f docker/Dockerfile -t $REGISTRY/$IMAGE_NAME:$TAG .
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $REGISTRY
docker push $REGISTRY/$IMAGE_NAME:$TAG
```

## EC2 Instance Setup

### SSH Access
Log into EC2 using PuTTY (Windows) or ssh (Linux/Mac).
Username: `ec2-user`

May need to allow inbound access on port 22 in Security Groups.

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
        "gpu=GPU-all"
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

```bash
export TAG=v1.0.0
export REGISTRY=678811077621.dkr.ecr.eu-west-1.amazonaws.com
export IMAGE_NAME=evie/matcha-tts

# Login and pull
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $REGISTRY
docker pull $REGISTRY/$IMAGE_NAME:$TAG

# Create service with 3 replicas
docker service create \
  --name matcha \
  --replicas 3 \
  --publish 8000:8000 \
  --env CHECKPOINT_PATH=/app/models/checkpoint.ckpt \
  --env MAX_TEXT_LENGTH=500 \
  --generic-resource "NVIDIA-GPU=0" \
  --update-delay 20s \
  --update-parallelism 1 \
  --update-order start-first \
  $REGISTRY/$IMAGE_NAME:$TAG
```

### Rolling Updates (Zero Downtime)

```bash
export TAG=v1.0.1
export REGISTRY=678811077621.dkr.ecr.eu-west-1.amazonaws.com
export IMAGE_NAME=evie/matcha-tts

# Login and pull new version
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $REGISTRY
docker pull $REGISTRY/$IMAGE_NAME:$TAG

# Update service (rolling update with 20s delay)
docker service update --update-delay 20s --image $REGISTRY/$IMAGE_NAME:$TAG matcha
```

### Alternative: Simple Docker Run (with downtime)

If you don't need rolling updates:

```bash
export TAG=v1.0.0
export REGISTRY=678811077621.dkr.ecr.eu-west-1.amazonaws.com
export IMAGE_NAME=evie/matcha-tts

# Login and pull
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin $REGISTRY
docker pull $REGISTRY/$IMAGE_NAME:$TAG

# Stop old, start new
docker stop matcha
docker container rm matcha
docker run -d --restart unless-stopped --gpus all -p 8000:8000 --name matcha $REGISTRY/$IMAGE_NAME:$TAG

# View logs
docker logs -f matcha
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
cat /etc/docker/daemon.json
```

### Out of memory errors
With 3 replicas, each loads ~3GB model = ~9-12GB VRAM total.
Reduce replicas if needed:
```bash
docker service update matcha --replicas 2
```

## Notes

- Each replica runs 1 uvicorn worker with the full model
- Swarm's built-in load balancer distributes requests across replicas
- Rolling updates start new container before stopping old one (brief VRAM spike)
- Health checks ensure traffic only goes to healthy containers
- `--update-order start-first` ensures zero downtime during updates