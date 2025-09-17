FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-devel

# Set working directory
WORKDIR /yolov9-main

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
    build-essential pkg-config \
    libswscale-dev libswresample-dev libavdevice-dev \
    libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
    gstreamer1.0-plugins-base gstreamer1.0-plugins-good gstreamer1.0-libav \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update && apt-get install -y git

# # Uninstall opencv-python in the original image if any
RUN pip uninstall -y opencv-python opencv-python-headless

# Download and build OpenCV from source
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.12.0.zip && \
    unzip opencv.zip && \
    mkdir -p opencv-4.12.0/build && \
    cd opencv-4.12.0/build && \
    cmake  \
        -D CMAKE_BUILD_TYPE=Release \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D WITH_FFMPEG=ON \
        -D WITH_GSTREAMER=ON \
        -D BUILD_EXAMPLES=OFF -D BUILD_TESTS=OFF -D BUILD_DOCS=OFF \
        -D BUILD_opencv_python3=ON \
        -D PYTHON3_EXECUTABLE=$(which python3) \
        -D OPENCV_PYTHON3_INSTALL_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])") \
        .. && \
    make -j$(nproc) && \
    make install && \
    cd /yolov9-main && \
    rm -rf opencv.zip opencv-4.12.0

# Copy requirements file
COPY requirements.txt .


RUN cd .. && \
    git clone https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && \
    git checkout v1.3.9 && \
    pip install -e . && \
    cd .. && \
    git clone https://github.com/ViTAE-Transformer/ViTPose.git && \
    cd ViTPose && \
    pip install -v -e .

# Install Python dependencies
RUN pip install -r requirements.txt

# Set default command to launch interactive bash shell
CMD ["/bin/bash"]


