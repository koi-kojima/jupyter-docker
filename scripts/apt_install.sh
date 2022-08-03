#! /usr/bin/env bash

PYTHON_VER=3.10
sed -i -e 's%http://[^ ]\+%mirror://mirrors.ubuntu.com/mirrors.txt%g' /etc/apt/sources.list
apt-get update --fix-missing -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    autoconf \
    autoconf-archive \
    automake \
    ccache \
    cmake \
    curl \
    gmp-ecm \
    extra-cmake-modules \
    ffmpeg \
    file \
    g++ \
    gcc \
    git \
    gphoto2 \
    gosu \
    less \
    libavcodec-dev \
    libavformat-dev \
    libboost-all-dev \
    libdc1394-22 \
    libdc1394-22-dev \
    libeigen3-dev \
    libeigen3-doc \
    libfaac-dev \
    libgflags-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libgtk-3-dev \
    libjpeg-dev \
    liblapacke-dev \
    libleptonica-dev \
    libmp3lame-dev \
    libogre-1.9-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libosmesa6-dev \
    libpng-dev \
    libsdl2-dev \
    libswscale-dev \
    libtbb-dev \
    libtesseract-dev \
    libtheora-dev \
    libtiff5-dev \
    libtool \
    libv4l-dev \
    libvorbis-dev \
    libxine2-dev \
    libxvidcore-dev \
    locales \
    mesa-utils \
    neofetch \
    nkf \
    patchelf \
    pkg-config \
    python3-opengl \
    python3-vtk7 \
    "python${PYTHON_VER}-dev" \
    qt5-default \
    ssh \
    sudo \
    swig \
    tcl-vtk7 \
    tesseract-ocr \
    tesseract-ocr-jpn \
    tree \
    unzip \
    v4l-utils \
    vim \
    vtk7 \
    wget \
    x264 \
    xvfb \
    yasm \
    zip \
    zlib1g-dev \
    zsh \
    --no-install-recommends
apt-get clean
rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

