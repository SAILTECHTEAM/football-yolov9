# Dockerfile

# Use NVIDIA's PyTorch base image
FROM nvcr.io/nvidia/pytorch:21.11-py3

# Set environment variables if necessary
ENV DEBIAN_FRONTEND=noninteractive
ENV QT_QPA_PLATFORM=xcb

# Create directories inside the container where volumes will be mounted
RUN mkdir -p /coco /yolov9

# Set a working directory (optional, based on where your main script resides)
WORKDIR /yolov9

# Install necessary dependencies
RUN apt-get update && apt-get install -y \
    vim \
    wget \
    libgl1 \
    libglib2.0-0 \
    libxkbcommon-x11-0 \
    libxcb-xinerama0 \
    libxcb-xinput0 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-render-util0 \
    libxcb-xfixes0 \
    libxcb-shm0 \
    libxcb-randr0 \
    libxcb-shape0 \
    libxcb-glx0 \
    software-properties-common \
    x11-apps \
    mesa-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/* 

RUN add-apt-repository universe && apt-get update

# Remove existing cv2 if it's installed from another source
RUN rm -rf /opt/conda/lib/python3.8/site-packages/cv2

RUN pip3 install shapely

# Reinstall OpenCV via pip
RUN pip3 install opencv-python seaborn \
    && pip3 install git+https://github.com/developer0hye/onepose.git



# You could also add your own script to install other Python packages
# Example (optional): Copy requirements.txt and install Python packages
# COPY requirements.txt /yolov9/
# RUN pip install -r requirements.txt

# You can also add an entry point to run a script by default (optional)
# Example: ENTRYPOINT ["python", "train.py"]
