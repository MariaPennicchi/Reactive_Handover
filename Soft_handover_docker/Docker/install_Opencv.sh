#!/bin/bash
#
#Install prerequisites Video/Audio Libs - FFMPEG, GSTREAMER, x264 etc.
apt-get update && \
    apt-get install -y \
        libdc1394-22 libdc1394-22-dev libxine2-dev libv4l-dev v4l-utils \
        libfaac-dev libmp3lame-dev libvorbis-dev \
        libxvidcore-dev x264 libx264-dev libfaac-dev libmp3lame-dev libtheora-dev \
        libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
        libavcodec-dev libavformat-dev libswscale-dev libavresample-dev && \
    rm -rf /var/lib/apt/lists/*


#Parallelism library C++ for CPU and optimization libraries for OpenCV
apt-get update && \
    apt-get install -y libtbb-dev \
    libatlas-base-dev \
    gfortran openexr libtbb2 && \
    rm -rf /var/lib/apt/lists/*

# Additional libraries
apt-get update && \
    apt-get install -y libprotobuf-dev protobuf-compiler \
        libgoogle-glog-dev libgflags-dev libpng-dev libjpeg-dev \
        libgphoto2-dev libeigen3-dev libhdf5-dev libtiff-dev \
        doxygen libgtk-3-dev && \
    rm -rf /var/lib/apt/lists/*


# Detect Python 3 version
PYTHON3_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d'.' -f1,2)
PYTHON3_MAJOR=$(echo "$PYTHON3_VERSION" | cut -d'.' -f1)
PYTHON3_MINOR=$(echo "$PYTHON3_VERSION" | cut -d'.' -f2)


if [[ -z "$PYTHON3_VERSION" ]]; then
    echo "Error: Python 3 not found."
    exit 1
fi

echo "Detected Python 3 version: $PYTHON3_VERSION"
echo "Prefix for Opencv make: ${PYTHON3_MAJOR}${PYTHON3_MINOR}"

# Construct paths based on detected version
PYTHON3_EXECUTABLE="/usr/bin/python3.$PYTHON3_MINOR"  # or /usr/bin/python3 if minor version is not needed
PYTHON3_PACKAGES_PATH="/usr/lib/python3.$PYTHON3_MINOR/dist-packages"
PYTHON3_INCLUDE_DIR="/usr/include/python3.$PYTHON3_MINOR"
OPENCV_PYTHON3_INSTALL_PATH="/usr/local/lib/python3.$PYTHON3_MINOR/dist-packages"
PYTHON3_NUMPY_INCLUDE_DIRS="/usr/local/lib/python3.$PYTHON3_MINOR/dist-packages/numpy/core/include"

echo "STARTED SCRIPT IN -> $(ls -la)"
# Setting Up the Folder
CV_ROOT_DIR="/root/opencv_build"
mkdir ${CV_ROOT_DIR}

# Clone the repositories
export OPENCV_CONTRIB_HASH "f852576142dec4c99fbeb89902129192cda3e7b6"
cd ${CV_ROOT_DIR} && git clone https://github.com/opencv/opencv_contrib.git && \
    cd opencv_contrib && \
    git checkout ${OPENCV_CONTRIB_HASH} && \
    echo "Cloned opencv_contrin in -> $(pwd)"

export OPENCV_HASH "b1cf5501233405de3ea5926d1d688e421b337458"
cd ${CV_ROOT_DIR} && git clone https://github.com/opencv/opencv.git && \
    cd opencv && \
    git checkout ${OPENCV_HASH} && \
    echo "Cloned opencv in -> $(pwd)"


cd "${CV_ROOT_DIR}/opencv" && \
    mkdir build && \
    cd build && \
    echo "cmake in -> $(pwd)" &&  \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_C_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D WITH_TBB=ON \
    -D ENABLE_FAST_MATH=1 \
    -D CUDA_FAST_MATH=1 \
    -D WITH_CUBLAS=1 \
    -D WITH_CUDA=ON \
    -D BUILD_opencv_cudacodec=OFF \
    -D WITH_CUDNN=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D CUDA_ARCH_BIN=${CCAP} \
    -D WITH_V4L=ON \
    -D WITH_QT=OFF \
    -D WITH_OPENGL=ON \
    -D WITH_GSTREAMER=ON \
    -D OPENCV_ENABLE_NONFREE=ON \
    -D OPENCV_EXTRA_MODULES_PATH=/root/opencv_build/opencv_contrib/modules \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D PYTHON_VERSION="${PYTHON3_MAJOR}${PYTHON3_MINOR}" \
    -D PYTHON3_EXECUTABLE="$PYTHON3_EXECUTABLE" \
    -D PYTHON3_PACKAGES_PATH="$PYTHON3_PACKAGES_PATH" \
    -D PYTHON3_INCLUDE_DIR="$PYTHON3_INCLUDE_DIR" \
    -D OPENCV_PYTHON3_INSTALL_PATH="$OPENCV_PYTHON3_INSTALL_PATH" \
    -D PYTHON3_NUMPY_INCLUDE_DIRS="$PYTHON3_NUMPY_INCLUDE_DIRS" \
    -D BUILD_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    ..

# Build OpenCv
cd "${CV_ROOT_DIR}/opencv/build" && make -j7
# Install OpenCv
cd "${CV_ROOT_DIR}/opencv/build" && make install