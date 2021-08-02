FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04
# https://github.com/ceccocats/tkDNN/tree/master/docker
LABEL maintainer "Francesco Gatti"

ADD nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb /tmp/trt.deb
RUN apt-get update && dpkg -i /tmp/trt.deb && rm /tmp/trt.deb && apt-get update
RUN apt install -y libnvinfer7=7.0.0-1+cuda10.2 libnvinfer-dev=7.0.0-1+cuda10.2
RUN DEBIAN_FRONTEND=noninteractive apt install -y git wget libeigen3-dev libyaml-cpp-dev
RUN cd /tmp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.17.3/cmake-3.17.3-Linux-x86_64.sh && \
    chmod +x cmake-3.17.3-Linux-x86_64.sh && \
    ./cmake-3.17.3-Linux-x86_64.sh --prefix=/usr/local --exclude-subdir --skip-license && \
    rm ./cmake-3.17.3-Linux-x86_64.sh

RUN echo "INSTALL OPENCV"
RUN apt-get install -y build-essential \
    unzip \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libv4l-dev \
    libxvidcore-dev \
    libx264-dev \
    libgtk-3-dev \
    libatlas-base-dev \
    gfortran \
    libgstreamer1.0-dev \
    libgstreamer-plugins-base1.0-dev \
    libdc1394-22-dev \
    libavresample-dev
RUN cd && wget https://github.com/opencv/opencv/archive/4.3.0.tar.gz && tar -xf 4.3.0.tar.gz && rm *.tar.gz
RUN cd && wget https://github.com/opencv/opencv_contrib/archive/4.3.0.tar.gz && tar -xf 4.3.0.tar.gz && rm *.tar.gz
RUN cd && \
    cd opencv-4.3.0 && mkdir build && cd build && \
    cmake -D CMAKE_BUILD_TYPE=RELEASE \
    -D CMAKE_INSTALL_PREFIX=/usr/local \
    -D INSTALL_PYTHON_EXAMPLES=OFF \
    -D INSTALL_C_EXAMPLES=OFF \
    -D OPENCV_EXTRA_MODULES_PATH='~/opencv_contrib-4.3.0/modules' \
    -D BUILD_EXAMPLES=OFF \
    -D WITH_CUDA=ON \
    -D CUDA_ARCH_BIN=7.2 \
    -D CUDA_ARCH_PTX="" \
    -D ENABLE_FAST_MATH=ON \
    -D CUDA_FAST_MATH=ON \
    -D WITH_CUBLAS=ON \
    -D WITH_LIBV4L=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_GSTREAMER_0_10=OFF \
    -D WITH_TBB=ON \
    ../ && make -j12 && make install
RUN apt clean
# end of https://github.com/ceccocats/tkDNN/tree/master/docker

RUN DEBIAN_FRONTEND=noninteractive apt update && apt-get install -y git wget libeigen3-dev

#install new cmake version
RUN apt remove -y cmake
RUN wget https://cmake.org/files/v3.15/cmake-3.15.0.tar.gz && tar -zxvf cmake-3.15.0.tar.gz
RUN cd cmake-3.15.0 && ./bootstrap && make install

RUN cmake --version

# Source https://github.com/dusty-nv/jetson-containers/blob/master/Dockerfile.ros.foxy
# compile yaml-cpp-0.6, which is needed (but is not in the 18.04 apt repo)
RUN git clone --branch yaml-cpp-0.6.0 https://github.com/jbeder/yaml-cpp yaml-cpp-0.6 && \
    cd yaml-cpp-0.6 && \
    mkdir build && \
    cd build && \
    cmake -DBUILD_SHARED_LIBS=ON -D CMAKE_INSTALL_PREFIX=/usr/local/libyaml-cpp .. && \
    make $MAKEFLAGS && \
    make install
    
WORKDIR /
RUN ls
RUN git clone https://github.com/ceccocats/tkDNN.git

# adapt scripts to work with python
COPY ./tkdnn_python/DetectionNN.h /tkDNN/include/tkDNN/DetectionNN.h
COPY ./tkdnn_python/utils.h /tkDNN/include/tkDNN/utils.h
COPY ./tkdnn_python/utils.cpp /tkDNN/src/utils.cpp
COPY ./tkdnn_python/darknetRT.cpp /tkDNN/demo/demo/darknetRT.cpp
COPY ./tkdnn_python/darknetRT.h /tkDNN/demo/demo/darknetRT.h

ADD data /model
WORKDIR /tkDNN
RUN mkdir build
COPY yolo4tiny.cpp tests/darknet/yolo4tiny.cpp


RUN sed -i 's/10, false)/10)/' tests/test_rtinference/rtinference.cpp
RUN cd build && cmake .. -D CMAKE_INSTALL_PREFIX=/usr/local/tkDNN && \
    make && \
    make install

RUN git clone https://git.hipert.unimore.it/fgatti/darknet.git
RUN cd darknet && make 
RUN cd darknet && mkdir layers debug

