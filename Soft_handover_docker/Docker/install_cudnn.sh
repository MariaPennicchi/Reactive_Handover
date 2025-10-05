#!/bin/bash


# CUDA CHECK
nvcc_output=$(nvcc --version 2>&1)

if [[ $? -eq 0 ]]; then  # Check if nvcc executed successfully
  echo "CUDA is installed."
  echo "$nvcc_output" # Print the version information.
else
  echo "CUDA is NOT installed or nvcc failed."
  echo "$nvcc_output" # Print error information.
  exit 1  # Exit with failure code (1)
fi


# INSTALLATION
CUDNN_TAR_FILE="cudnn-linux-x86_64-8.7.0.84_cuda11-archive.tar.xz"
wget https://developer.download.nvidia.com/compute/redist/cudnn/v8.7.0/local_installers/11.8/${CUDNN_TAR_FILE}
tar -xvf ${CUDNN_TAR_FILE}
mv cudnn-linux-x86_64-8.7.0.84_cuda11-archive cudnn

# copy the following files into the cuda toolkit directory.
cp -P cudnn/include/* /usr/local/cuda-11.8/include/ && \
 cp -P cudnn/lib/libcudnn* /usr/local/cuda-11.8/lib64/ && \
 chmod a+r /usr/local/cuda-11.8/lib64/libcudnn*

# Remove the folder
rm -r cudnn && rm ${CUDNN_TAR_FILE}

# VERIFY CUDNN INSTALLATION
cudnn_version_header="/usr/local/cuda/include/cudnn_version.h"  # Path to the header file

# Extract MAJOR, MINOR, and PATCHLEVEL
CUDNN_MAJOR=$(grep CUDNN_MAJOR "$cudnn_version_header" | awk '{print $3}')
CUDNN_MINOR=$(grep CUDNN_MINOR "$cudnn_version_header" | awk '{print $3}')
CUDNN_PATCHLEVEL=$(grep CUDNN_PATCHLEVEL "$cudnn_version_header" | awk '{print $3}')

# Construct the version string
CUDNN_VERSION="${CUDNN_MAJOR}.${CUDNN_MINOR}.${CUDNN_PATCHLEVEL}"

# Display the version
echo "$CUDNN_VERSION"