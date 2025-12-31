FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

# Set working directory
WORKDIR /usr/src

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    ffmpeg \
    cmake \
    g++ \
    wget \
    unzip \
    git \
    build-essential pkg-config \
    libswscale-dev libswresample-dev libavdevice-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV and ViTPose from source
RUN git clone https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && \
    git checkout v1.3.9 && \
    pip install -e . && \
    cd .. && \
    git clone https://github.com/ViTAE-Transformer/ViTPose.git && \
    cd ViTPose && \
    pip install -v -e .

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Uninstall opencv-python in the original image if any
RUN pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless

# Define OpenCV version
ENV OPENCV_VERSION=4.11.0

RUN cd /usr/src && \
    wget -O opencv.zip https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip && \
    unzip opencv.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip && \
    unzip opencv_contrib.zip && \
    rm opencv.zip opencv_contrib.zip

# # Install cuDNN system package to ensure compatibility (uncomment this if using OpenCV with CUDA support)
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     libcudnn9-dev-cuda-12 \
#     && rm -rf /var/lib/apt/lists/*

# Download and build OpenCV from source (CPU only)
RUN cd /usr/src/opencv-${OPENCV_VERSION} && mkdir build && cd build && \
    cmake  \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D with_OPENMP=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_GSTREAMER=ON \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_DOCS=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_opencv_java=OFF \
        -D OPENCV_ENABLE_NONFREE=ON \
        -D BUILD_opencv_python3=ON \
        -D PYTHON3_EXECUTABLE=$(command -v python3) \
        -D PYTHON3_PACKAGES_PATH=$(python3 -c "import sysconfig; print(sysconfig.get_paths()['purelib'])") \
        -D OPENCV_EXTRA_MODULES_PATH=/usr/src/opencv_contrib-${OPENCV_VERSION}/modules \
        .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Clean up
WORKDIR /usr/src
RUN rm -rf opencv-${OPENCV_VERSION} opencv_contrib-${OPENCV_VERSION}    

WORKDIR /yolov9-main
# Set default command to launch interactive bash shell
CMD ["/bin/bash"]


