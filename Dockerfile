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
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Set default command to launch interactive bash shell
CMD ["/bin/bash"]