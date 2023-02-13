# syntax = docker/dockerfile:1

FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 as python_build
ARG PYTHON_VER=${PYTHON_VER:-3.10.9}
RUN <<EOF
sed -i -e 's%http://[^ ]\+%mirror://mirrors.ubuntu.com/mirrors.txt%g' /etc/apt/sources.list
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    cmake \
    curl \
    gdb \
    lcov \
    libbz2-dev \
    libffi-dev \
    libgdbm-compat-dev \
    libgdbm-dev \
    liblzma-dev \
    libncurses5-dev \
    libreadline6-dev \
    libsqlite3-dev \
    libssl-dev \
    lzma \
    lzma-dev \
    pkg-config \
    tk-dev \
    uuid-dev \
    zlib1g-dev
apt-get clean
rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
EOF

# Install Python
RUN <<EOF
set -eu
mkdir -p /python
curl -L -o /python/python.tar.xz https://www.python.org/ftp/python/${PYTHON_VER}/Python-${PYTHON_VER}.tar.xz
cd /python
tar -xf python.tar.xz
rm python.tar.xz
cd Python-${PYTHON_VER}
./configure --enable-optimizations --enable-shared --prefix=/python/build >/dev/null
sed -i -e 's/-L. -lpython$(LDVERSION)/libpython$(LDVERSION).a/' Makefile
make && make altinstall
EOF

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04 as opencv_base
ARG OPEN_CV_VERSION=${OPEN_CV_VERSION:-4.7.0}
ARG PYTHON_VER=${PYTHON_VER:-3.10.9}

# Install libraries
RUN <<EOF
sed -i -e 's%http://[^ ]\+%mirror://mirrors.ubuntu.com/mirrors.txt%g' /etc/apt/sources.list
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    automake \
    build-essential \
    ccache \
    cmake \
    curl \
    extra-cmake-modules \
    ffmpeg \
    g++ \
    gcc \
    git \
    gmp-ecm \
    libavcodec-dev \
    libavformat-dev \
    libboost-dev \
    libbz2-dev \
    libdb-dev \
    libeigen3-dev \
    libfaac-dev \
    libffi-dev \
    libgdbm-dev \
    libgflags-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libjpeg-dev \
    liblapacke-dev \
    libleptonica-dev \
    liblzma-dev \
    libmp3lame-dev \
    libncursesw5-dev \
    libogre-1.9-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libosmesa6-dev \
    libpng-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libswscale-dev \
    libtbb-dev \
    libtheora-dev \
    libtiff5-dev \
    libtool \
    libv4l-dev \
    libvorbis-dev \
    libxine2-dev \
    libxvidcore-dev \
    locales \
    mesa-utils \
    patchelf \
    pkg-config \
    swig \
    tcl-vtk7 \
    unzip \
    uuid-dev \
    v4l-utils \
    yasm \
    zip \
    zlib1g-dev \
    --no-install-recommends
apt-get clean
rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
EOF

# Install Python
ENV PATH=/python/build/bin:$PATH
COPY --from=python_build ["/python", "/python"]
RUN <<EOF
ln -s $(find /python/build/bin -name "python*" | grep -v "config") /usr/local/bin/python3
python3 -m pip install numpy numba
python3 -c "import numpy"
EOF

# Install Python Libs
RUN <<EOF
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y python3-dev
apt-get clean
rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
EOF

# Download OpenCV
RUN <<EOF
mkdir -p /opencv/opencv-build
curl -L -o /opencv/opencv-${OPEN_CV_VERSION}.zip "https://github.com/opencv/opencv/archive/${OPEN_CV_VERSION}.zip"
curl -L -o /opencv/opencv_contrib-${OPEN_CV_VERSION}.zip "https://github.com/opencv/opencv_contrib/archive/${OPEN_CV_VERSION}.zip"
unzip -q /opencv/opencv-${OPEN_CV_VERSION}.zip -d /opencv
unzip -q /opencv/opencv_contrib-${OPEN_CV_VERSION}.zip -d /opencv
rm /opencv/opencv-${OPEN_CV_VERSION}.zip /opencv/opencv_contrib-${OPEN_CV_VERSION}.zip
mkdir -p /opencv/opencv-${OPEN_CV_VERSION}/build
EOF

FROM opencv_base as opencv

# Build OpenCV
RUN <<EOF
set -eu
PYTHON_MAJOR_MINOR=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
cd /opencv/opencv-${OPEN_CV_VERSION}/build
CC=gcc CXX=g++ cmake \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=/opencv/opencv-build \
    -D OPENCV_EXTRA_MODULES_PATH=/opencv/opencv_contrib-${OPEN_CV_VERSION}/modules \
    -D OPENCV_GENERATE_PKGCONFIG=ON \
    -D BUILD_opencv_apps=ON \
    -D BUILD_opencv_calib3d=ON \
    -D BUILD_opencv_core=ON \
    -D BUILD_opencv_dnn=ON \
    -D BUILD_opencv_features2d=ON \
    -D BUILD_opencv_flann=ON \
    -D BUILD_opencv_highgui=ON \
    -D BUILD_opencv_imgcodecs=ON \
    -D BUILD_opencv_imgproc=ON \
    -D BUILD_opencv_java_bindings_generator=OFF \
    -D BUILD_opencv_js_bindings_generator=OFF \
    -D BUILD_opencv_ml=ON \
    -D BUILD_opencv_objdetect=ON \
    -D BUILD_opencv_photo=ON \
    -D BUILD_opencv_video=ON \
    -D BUILD_opencv_videoio=ON \
    -D BUILD_JAVA=OFF \
    -D BUILD_opencv_python2=OFF \
    -D BUILD_opencv_python3=ON \
    -D WITH_FFMPEG=ON \
    -D WITH_GPHOTO2=OFF \
    -D WITH_GIGEAPI=ON \
    -D WITH_GSTREAMER=ON \
    -D WITH_GTK=OFF \
    -D WITH_INTELPERC=OFF \
    -D WITH_LIBV4L=ON \
    -D WITH_OPENCL=ON \
    -D WITH_OPENCLAMDBLAS=OFF \
    -D WITH_OPENCLAMDFFT=OFF \
    -D WITH_OPENCL_SVM=OFF \
    -D WITH_OPENEXR=ON \
    -D WITH_OPENGL=ON \
    -D WITH_PTHREADS_PF=OFF \
    -D WITH_QT=OFF \
    -D WITH_UNICAP=OFF \
    -D CUDA_NVCC_FLAGS=--expt-relaxed-constexpr \
    -D CUDA_FAST_MATH=ON \
    -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
    -D CUDA_HOST_COMPILER=/usr/bin/gcc \
    -D CUDA_ARCH_BIN="6.1 7.5 8.6" \
    -D WITH_CUBLAS=ON \
    -D WITH_CUDA=ON \
    -D WITH_CUFFT=ON \
    -D WITH_NVCUVID=ON \
    -D OPENCV_DNN_CUDA=ON \
    -D BUILD_opencv_cudaarithm=ON \
    -D BUILD_opencv_cudabgsegm=ON \
    -D BUILD_opencv_cudacodec=ON \
    -D BUILD_opencv_cudafeatures2d=ON \
    -D BUILD_opencv_cudafilters=ON \
    -D BUILD_opencv_cudaimgproc=ON \
    -D BUILD_opencv_cudalegacy=ON \
    -D BUILD_opencv_cudaobjdetect=ON \
    -D BUILD_opencv_cudaoptflow=ON \
    -D BUILD_opencv_cudastereo=ON \
    -D BUILD_opencv_cudawarping=ON \
    -D BUILD_opencv_cudev=ON \
    -D PYTHON_DEFAULT_EXECUTABLE=python3 \
    -D PYTHON3_INCLUDE_DIR=$(python3 -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())') \
    -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python3 -c 'import numpy; print(numpy.get_include())') \
    -D PYTHON3_LIBRARIES=$(find /python/build -name "libpython3.so") \
    -D PYTHON3_PACKAGES_PATH=$(python3 -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())') \
    -D BUILD_TESTS=OFF \
    -D BUILD_PERF_TESTS=OFF \
    -D BUILD_EXAMPLES=OFF \
    -D INSTALL_PYTHON_EXAMPLES=ON \
    .. \
&& make -j $(($(nproc) + 1)) \
&& make install

python3 -m pip install /opencv/opencv-${OPEN_CV_VERSION}/build/python_loader
python3 -c "import cv2"
EOF

FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 as release
# Install libraries
RUN <<EOF
sed -i -e 's%http://[^ ]\+%mirror://mirrors.ubuntu.com/mirrors.txt%g' /etc/apt/sources.list
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y \
    automake \
    build-essential \
    ccache \
    cmake \
    curl \
    extra-cmake-modules \
    ffmpeg \
    g++ \
    gcc \
    git \
    gmp-ecm \
    libavcodec-dev \
    libavformat-dev \
    libboost-dev \
    libbz2-dev \
    libdb-dev \
    libeigen3-dev \
    libfaac-dev \
    libffi-dev \
    libgdbm-dev \
    libgflags-dev \
    libgl1-mesa-glx \
    libglfw3 \
    libjpeg-dev \
    liblapacke-dev \
    libleptonica-dev \
    liblzma-dev \
    libmp3lame-dev \
    libncursesw5-dev \
    libogre-1.9-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    libosmesa6-dev \
    libpng-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libswscale-dev \
    libtbb-dev \
    libtheora-dev \
    libtiff5-dev \
    libtool \
    libv4l-dev \
    libvorbis-dev \
    libxine2-dev \
    libxvidcore-dev \
    locales \
    mesa-utils \
    patchelf \
    pkg-config \
    swig \
    tcl-vtk7 \
    unzip \
    uuid-dev \
    v4l-utils \
    yasm \
    zip \
    zlib1g-dev \
    file \
    git \
    less \
    locales \
    neofetch \
    nkf \
    ssh \
    sudo \
    gosu \
    tree \
    vim \
    --no-install-recommends
apt-get clean
rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
EOF

# SSH setting
RUN <<EOF 
sed -i \
    -e "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/" \
    -e "s/#PermitEmptyPasswords no/PermitEmptyPasswords yes/" \
    /etc/ssh/sshd_config
sed -i -e "s/root:x:/root::/g" /etc/passwd
mkdir /run/sshd
EOF

# Install Python
ENV PATH=/python/build/bin:$PATH
COPY --from=opencv ["/python/build", "/python/build"]
RUN <<EOF
ln -s $(find /python/build/bin -name "python*" | grep -v "config") /usr/local/bin/python3
ln -s $(find /python/build/bin -name "pip*") /usr/local/bin/pip3
EOF

# Install OpenCV
COPY --from=opencv ["/opencv", "/opencv"]

ENV NOTEBOOK_DIR=/root/notebooks
ENV DEFAULT_TEMPLATE_DIR=${NOTEBOOK_DIR}/templates

# Install Python libraries
RUN <<EOF
python3 -m pip install torch torchvision torchdata torchtext --extra-index-url https://download.pytorch.org/whl/cu117
python3 -m pip install \
    classy_vision \
    hydra-core \
    japanize-matplotlib \
    jupyterlab \
    lightning \
    lz4 \
    pandas \
    pylint autopep8 \
    scikit-learn \
    scipy \
    seaborn \
    sympy \
    tensorboardX \
    timm \
    torchinfo \
    torchmetrics \
    tqdm \
    umap-learn \
    wandb \
    --no-cache-dir
EOF

# Setup Jupyter lab
COPY --chmod=755 ["./scripts/jupyter_setting.sh", "/install_scripts/"]
RUN /install_scripts/jupyter_setting.sh

COPY ./jupyter_config/ /root/.jupyter/lab/user-settings/@jupyterlab/
COPY --chmod=755 ["./check_gpu.py", "./mnist*.py", "./qmnist.py", "${DEFAULT_TEMPLATE_DIR}/"]
EXPOSE 8888
CMD ["jupyter", "lab", "--no-browser"]
