If you want to test the docker image locally, with GPU support, use docker.
See instructions for enabling CUDA on docker: https://docker-desktop.io/docs/docker/gpu
Use these commands from the project root:
```
set / export TAG=25.09.21-2 

docker build -f docker\gpu\Dockerfile -t 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokoro:%TAG% .
```
Run it with:
```
docker run -e NUM_WORKERS=1 -p 8880:8880 --name kokoro 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokoro:%TAG%
```

You can log into the container with
```
docker exec -it kokoro /bin/bash
```

For repetitive operations, copy and paste this:
```
docker container remove kokoro
docker build -f docker\gpu\Dockerfile -t 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokoro:%TAG% .
docker run -e NUM_WORKERS=1 -p 8880:8880 --name kokoro 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokoro:%TAG%
```

Other commands:
```
docker container remove kokoro
docker images
docker image remove 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokoro:%TAG% .
```

Push the image to the ECR repo:

See docs for pushing to ECR: https://docs.aws.amazon.com/AmazonECR/latest/userguide/getting-started-cli.html
This is my private repo: https://eu-west-1.console.aws.amazon.com/ecr/private-registry/repositories?region=eu-west-1
I can get the AWS docker push / pull commands from there.

SSH keys for logging into EC2 containers: https://eu-west-1.console.aws.amazon.com/ec2/home?region=eu-west-1#KeyPairs

Start Rancher, then run these commands:
```
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 678811077621.dkr.ecr.eu-west-1.amazonaws.com
docker push 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokoro:%TAG%
```
The last command will print the pull command that you have to execute on the EC2 machine

To run the app on an EC2 instance, it must have x86 CPUs and nVidia cGPUs, for example g4dn.xlarge.
It's best to use an instance that already has nVidia drivers for example: "Deep Learning Base OSS Nvidia Driver GPU AMI (Amazon Linux 2023)".
I can use swarm as explained here: https://blog.container-solutions.com/rolling-updates-with-docker-swarm

Get the instance name from: https://eu-west-1.console.aws.amazon.com/ec2/home?region=eu-west-1#Instances
Log in using ssh:
```
ssh -i ~/.ssh/ec2-connect-key-ireland.pem ec2-user@ec2-34-251-240-148.eu-west-1.compute.amazonaws.com
```
May need to allow inbound access on port 22 for my current public IP in Security Groups:
https://eu-west-1.console.aws.amazon.com/ec2/home?region=eu-west-1#SecurityGroups

I can update it with
```
aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 678811077621.dkr.ecr.eu-west-1.amazonaws.com
docker pull 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokoro:$TAG
```

I can run it with:
```
docker stop kokoro
docker container rm kokoro
docker run -d -e NUM_WORKERS=2 -e LOGURU_LEVEL=WARNING --restart unless-stopped --gpus all -p 8880:8880 --name kokoro 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokoro:$TAG
docker logs -f kokoro
```

When the web app shows as ready, Open postman and test it.

------------------------------------------------------------------------------------------------------

Alternate method of running.


If I want to run with docker swarm, first, I have to enable GPU access as follows:

1. Add this to /etc/docker/daemon.json:
    ```
    sudo nano /etc/docker/daemon.json
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
2. Uncomment this line in /etc/nvidia-container-runtime/config.toml:
    ```
    sudo nano /etc/nvidia-container-runtime/config.toml
    ...
    # swarm-resource = "DOCKER_RESOURCE_GPU"
    ...
    ```
3. Then restart docker
    ```
    sudo systemctl restart docker.service
    ```
    
4. To have access to pull images from ECR, I need to add a role to the EC2 instance.
   The role must have at least AmazonEC2ContainerRegistryReadOnly.
   Then I have to create two Interface Endpoints, one for com.amazonaws.eu-west-1.ecr.api and one for com.amazonaws.eu-west-1.ecr.dkr
   both in the EC2 VPC, for the instance Security Group, and the subnet used by the Security Group.
   Do not put the endpoints in the EC2 security group, all ecr commands will time out.
   The endpoint needs its own security group, which must allow inbound traffic from your EC2 instance's security group on port 443 (HTTPS).
   The EC2 instance's security group must allow outbound traffic to the VPC endpoints on port 443 (or all outbound traffic).

   One-time setup:
    ```
    sudo dnf update -y
    sudo dnf clean all
    aws configure
    docker swarm init
    ```

   Initial deployment:
    ```
    aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 678811077621.dkr.ecr.eu-west-1.amazonaws.com
    docker pull 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokoro:$TAG
    docker service create --name kokoro --replicas 1 --publish 8880:8880 --generic-resource "NVIDIA-GPU=0" 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokoro:$TAG
    ```

5. Rolling updates:
    ``` 
    aws ecr get-login-password --region eu-west-1 | docker login --username AWS --password-stdin 678811077621.dkr.ecr.eu-west-1.amazonaws.com
    docker pull 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokoro:$TAG
    docker service update --update-delay 20s --image 678811077621.dkr.ecr.eu-west-1.amazonaws.com/evie/kokoro:$TAG kokoro
    ```
    
6. As needed, use these commands:
    ```
    watch -d -n 0.3 nvidia-smi
    docker service logs -f kokoro
    docker service update kokoro --replicas 3
    docker service rm kokoro
    docker swarm leave --force
    docker container ls
    docker exec -it <container_id> bash
    ```

My current EC2 machine is ec2-34-251-240-148.eu-west-1.compute.amazonaws.com
I can access the web ui from home (I have an inbound rule by IP address): http://ec2-34-251-240-148.eu-west-1.compute.amazonaws.com:8880/web/
Synthesis for one of the long sentences listed here https://thejohnfox.com/2021/08/65-long-sentences-in-literature/. For example, a sentence
of 100 words, 500 chars, takes 1.5 sec in total.
