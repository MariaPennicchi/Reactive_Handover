FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

## for apt to be noninteractive
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true

# ------------------------------------------------------------------------------------
# Basic dependencies
# ------------------------------------------------------------------------------------

RUN echo "tzdata tzdata/Areas select Europe" > /tmp/preseed.txt; \
    echo "tzdata tzdata/Zones/Europe select Rome" >> /tmp/preseed.txt; \
    debconf-set-selections /tmp/preseed.txt && \
    apt-get update && \
    apt-get install --no-install-recommends -y tzdata git wget cmake make build-essential g++ \
     python3 python3-pip python3-dev python3-numpy python3-venv curl nano && \
    rm -rf /var/lib/apt/lists/*

# ------------------------------------------------------------------------------------
# CUDNN
# ------------------------------------------------------------------------------------

COPY Docker/install_cudnn.sh /tmp/install_cudnn.sh
RUN /bin/bash -c "chmod +x /tmp/install_cudnn.sh && \
    bash /tmp/install_cudnn.sh"

# ------------------------------------------------------------------------------------
# OPENCV
# ------------------------------------------------------------------------------------

COPY Docker/install_Opencv.sh /tmp/install_Opencv.sh
RUN /bin/bash -c "chmod +x /tmp/install_Opencv.sh && \
    bash /tmp/install_Opencv.sh"

# ------------------------------------------------------------------------------------
# Pytorch
# ------------------------------------------------------------------------------------

RUN /bin/bash -c "pip install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118"
 
# ------------------------------------------------------------------------------------
# Python requirements 
# ------------------------------------------------------------------------------------
   
COPY requirements.txt ./
RUN /bin/bash -c "pip install -r requirements.txt"

# ------------------------------------------------------------------------------------
# Pycharm 
# ------------------------------------------------------------------------------------

ARG PYCHARM_VERSION=2024.3.3
ARG pycharm_local_dir=.PyCharmCE${PYCHARM_VERSION}

WORKDIR /opt/pycharm

RUN echo "Preparing PyCharm ${PYCHARM_VERSION} ..." && \
	export pycharm_source=https://download.jetbrains.com/python/pycharm-community-${PYCHARM_VERSION}.tar.gz && \
	curl -fsSL $pycharm_source -o /opt/pycharm/installer.tgz && \
	tar --strip-components=1 -xzf installer.tgz && \ 
	rm installer.tgz

# ------------------------------------------------------------------------------------
# ROS
# ------------------------------------------------------------------------------------

ENV WORKSPACE_DIR_SH "/workspace/softh_ws"
ENV WORKSPACE_DIR_MI "/workspace/moveit_ws"

RUN apt-get update && apt-get install sudo

RUN echo "deb http://packages.ros.org/ros/ubuntu $(grep '^UBUNTU_CODENAME=' /etc/os-release | cut -d= -f2) main" > /etc/apt/sources.list.d/ros-latest.list

RUN /bin/bash -c "curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | sudo apt-key add -"

# Prerequisites ROS
RUN apt-get update \
    && apt-get install --no-install-recommends -y  python3-rosdep python3-rosinstall \
    python3-rosinstall-generator python3-wstool \
    python3-catkin-tools ros-noetic-rviz\
    ros-noetic-moveit ros-noetic-vision-msgs ros-noetic-franka-ros && \
    rm -rf /var/lib/apt/lists/*


# Set ROS system level
RUN  /bin/bash -c "echo \"source /opt/ros/noetic/setup.bash\" >> ~/.bashrc"

# ------------------------------------------------------------------------------------
# Prepare SOFT HANDOVER Workspace ROS
# ------------------------------------------------------------------------------------

COPY softh_ws/ /workspace/softh_ws/


RUN /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    catkin init && \
    catkin build && \
    source ${WORKSPACE_DIR_SH}/devel/setup.bash && \
    echo \"source ${WORKSPACE_DIR_SH}/devel/setup.bash\" >> ~/.bashrc"

# ------------------------------------------------------------------------------------
# Prepare MOVE IT Workspace ROS
# ------------------------------------------------------------------------------------

WORKDIR ${WORKSPACE_DIR_MI}
RUN /bin/bash -c "mkdir src && \ 
                  source /opt/ros/noetic/setup.bash && catkin init"  
COPY moveit_ws/ ${WORKSPACE_DIR_MI}

RUN /bin/bash -c "rosdep init && rosdep update --include-eol-distros && apt update && \ 
                  rosdep install -y --from-paths . --ignore-src --rosdistro noetic && \
                  catkin config --extend /opt/ros/noetic --cmake-args -DCMAKE_BUILD_TYPE=Release && \
                  catkin build && \
                  source ${WORKSPACE_DIR_MI}/devel/setup.bash && \
                  echo \"source ${WORKSPACE_DIR_MI}/devel/setup.bash\" >> ~/.bashrc"
                  
WORKDIR ${WORKSPACE_DIR_SH}

    
