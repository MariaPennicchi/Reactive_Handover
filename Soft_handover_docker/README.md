# Soft Handover Docker Container Installation 

## Important Note About the MoveIt Workspace

The `moveit_ws` folder **is not included in this GitHub repository** because it is too large.  
To use MoveIt, you will need to download the workspace yourself, following the official instructions:

- [MoveIt Installation Guide](https://moveit.ros.org/install/)

After downloading the workspace, make sure to place it in the `Soft_handover_docker/moveit_ws` directory, and then proceed with the regular installation steps below.

## 1) Open a terminal
Navigate into the `Soft_handover_docker` project folder.

## 2) Build the Docker image
Run the following command:

```bash
docker build -t <image_name>:<tag> -f Docker/Dockerfile .
```

## 3) Create a Docker container
Run:

```bash
docker run -td -i --privileged --net=host \
  --name=<container_name_softh> \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -e DISPLAY=$DISPLAY \
  -e "QT_X11_NO_MITSHM=1" \
  -h $HOSTNAME \
  -v /home/isarlab/Documenti/DockerShared/:/root/Archive/Shared \
  -v /dev:/dev \
  -v /dev/bus/usb:/dev/bus/usb \
  --device=/dev/bus/usb \
  --device-cgroup-rule='c 189:* rmw' \
  -v /etc/udev/rules.d:/etc/udev/rules.d \
  -v $XAUTHORITY:/root/.Xauthority:rw \
  --runtime=nvidia \
  --gpus all \
  --env="NVIDIA_DRIVER_CAPABILITIES=all" \
  <image_name>
```

## 4) Verify container creation
Check if the container exists:

```bash
docker ps -a
```

## 5) Start and access the container
Start the container:

```bash
docker start <container_name>
```

Open an interactive shell inside:

```bash
docker exec -it <container_name> bash
```

## 6) First-time container startup note 
When starting the container for the first time, disable the Markdown plugin in PyCharm, otherwise the IDE may crash.

## 7) One-time setup inside the container

The first time you start the container, go to the softh_ws workspace and run:

```bash
source /workspace/softh_ws/devel/setup.bash
```
