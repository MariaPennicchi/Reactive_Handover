# Intel RealSense Docker Container Installation

## 1) Install Intel RealSense D415 drivers
Make sure you install the proper drivers and the `librealsense2` SDK compatible with your operating system.

## 2) Open a terminal
Navigate into the `Real_sense_docker` project folder.

## 3) Build the Docker image
Run the following command:

```bash
docker build -t <image_name>:<tag> -f Dockerfile .
```

## 4) Create a Docker container
Run:

```bash
docker run -td -i --privileged --net=host \
  --name=<container_name_real_sense> \
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

## 5) Verify container creation
Check if the container exists:

```bash
docker ps -a
```

## 6) Start and access the container
Start the container:

```bash
docker start <container_name>
```

Open an interactive shell inside:

```bash
docker exec -it <container_name> bash
```
