FROM ros:noetic-ros-base-focal

## for apt to be noninteractive
ENV DEBIAN_FRONTEND noninteractive
ENV DEBCONF_NONINTERACTIVE_SEEN true


ENV WORKSPACE_DIR "/workspace/handover_ws"

## Depends
RUN echo "tzdata tzdata/Areas select Europe" > /tmp/preseed.txt; \
    echo "tzdata tzdata/Zones/Europe select Rome" >> /tmp/preseed.txt; \
    debconf-set-selections /tmp/preseed.txt && \
    apt-get update && \
    apt-get install --no-install-recommends -y tzdata git wget cmake make build-essential && \
    rm -rf /var/lib/apt/lists/*

# Prerequisites ROS
RUN apt-get update \
    && apt-get install --no-install-recommends -y  python3-rosdep python3-rosinstall \
    python3-rosinstall-generator python3-wstool \
    python3-catkin-tools ros-noetic-pcl-ros ros-noetic-rviz\
    ros-noetic-perception && \
    rm -rf /var/lib/apt/lists/*


# Set ROS system level
RUN  /bin/bash -c "echo \"source /opt/ros/noetic/setup.bash\" >> ~/.bashrc"

# Prepare Workspace ROS
WORKDIR ${WORKSPACE_DIR}

RUN mkdir ./src \
    && /bin/bash -c "source /opt/ros/noetic/setup.bash && \
    catkin init && \
    catkin build && \
    echo \"source ${WORKSPACE_DIR}/devel/setup.bash\" >> ~/.bashrc"

# Clone the realsense repo and chekout

RUN /bin/bash -c "cd ./src && \
git clone https://github.com/IntelRealSense/realsense-ros"

RUN /bin/bash -c "cd ./src/realsense-ros && git checkout ros1-legacy"

# Install Packages depenencies
RUN /bin/bash -c "apt-get update && \
    rosdep install --from-paths src --ignore-src -r -y && \
    apt-get install libboost-all-dev"

# compile the realsense packages
RUN /bin/bash -c "source ${WORKSPACE_DIR}/devel/setup.bash && \
 catkin build -DCATKIN_ENABLE_TESTING=False -DCMAKE_BUILD_TYPE=Release"



