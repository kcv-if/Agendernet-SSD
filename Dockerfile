FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04
RUN apt update && \
    apt install -y build-essential cmake git nano\
    libopenblas-dev liblapack-dev  \
    libx11-dev libgtk-3-dev \
    python3 python3-dev python3-pip python3-tk \
    graphviz

# Install python package
RUN pip3 install numpy pandas scipy \
    scikit-learn scikit-image pydot \
    numba filterpy matplotlib opencv-python\
    six tqdm tensorflow-gpu==1.4.1 keras==2.1.3

# Build dlib
RUN git clone https://github.com/davisking/dlib.git
WORKDIR /dlib
RUN python3 setup.py install

