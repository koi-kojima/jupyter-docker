ARG CUDA_MAJOR_VERSION=${CUDA_MAJOR_VERSION:-11}
ARG CUDA_MINOR_VERSION=${CUDA_MINOR_VERSION:-1}
FROM nvidia/cuda:${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION}-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ARG OPEN_CV_VERSION=${OPEN_CV_VERSION:-4.5.3}

RUN adduser -q --gecos "" --disabled-password dev
ENV PATH /home/dev/conda/bin:$PATH

RUN sed -i -e 's%http://[^ ]\+%mirror://mirrors.ubuntu.com/mirrors.txt%g' /etc/apt/sources.list && \
    apt-get update --fix-missing -qq && apt-get install -y \
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
    python3.9-dev \
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
    --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*
EXPOSE 22

# SSH setting
RUN sed -i -e "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/" \
           -e "s/#PermitEmptyPasswords no/PermitEmptyPasswords yes/" \
           /etc/ssh/sshd_config && \
    sed -i -e "s/root:x:/root::/g" -e "s/dev:x:/dev::/g" /etc/passwd && \
    mkdir /run/sshd

# Install Miniconda
RUN curl -L -sS -o /home/dev/miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh" \
    && sudo -u dev --login /bin/bash /home/dev/miniconda.sh -b -p /home/dev/conda \
    && rm /home/dev/miniconda.sh \
    && ln -s /home/dev/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && sudo -u dev --login /home/dev/conda/bin/conda update conda --quiet --yes >/dev/null \
    && sudo -u dev --login /home/dev/conda/bin/conda create --yes -n research -c conda-forge python=3.9 numpy \
    && sudo -u dev --login /home/dev/conda/bin/conda clean --yes --index-cache >/dev/null \
    && echo ". /home/dev/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate research" >> ~/.bashrc \
    && ln -s /home/dev/conda/envs/research/bin/python /usr/bin/python
SHELL ["/bin/bash", "-l", "-c"]

USER dev
WORKDIR /home/dev
ENV HOME /home/dev
ENV PATH $HOME/.local/bin:$PATH
# Install OpenCV
RUN CC=gcc && CXX=g++ && mkdir ~/opencv && cd ~/opencv \
    && curl -L -O "https://github.com/opencv/opencv/archive/${OPEN_CV_VERSION}.zip" \
    && curl -L -o opencv_contrib-${OPEN_CV_VERSION}.zip "https://github.com/opencv/opencv_contrib/archive/${OPEN_CV_VERSION}.zip" \
    && unzip -q ${OPEN_CV_VERSION}.zip \
    && unzip -q opencv_contrib-${OPEN_CV_VERSION}.zip \
    && cd opencv-${OPEN_CV_VERSION} \
    && mkdir build \
    && conda activate research && cd ~/opencv/opencv-${OPEN_CV_VERSION}/build \
    && cmake -D CMAKE_BUILD_TYPE=Release \
             -D CMAKE_INSTALL_PREFIX=/home/dev/conda/envs/research \
             -D OPENCV_EXTRA_MODULES_PATH=/home/dev/opencv/opencv_contrib-${OPEN_CV_VERSION}/modules \
             -D OPENCV_GENERATE_PKGCONFIG=ON \
             -D BUILD_opencv_apps=ON \
             -D BUILD_opencv_calib3d=ON \
             -D BUILD_opencv_core=ON \
             -D BUILD_opencv_features2d=ON \
             -D BUILD_opencv_flann=ON \
             -D BUILD_opencv_highgui=ON \
             -D BUILD_opencv_imgcodecs=ON \
             -D BUILD_opencv_imgproc=ON \
             -D BUILD_opencv_ml=ON \
             -D BUILD_opencv_objdetect=ON \
             -D BUILD_opencv_photo=ON \
             -D BUILD_opencv_stitching=ON \
             -D BUILD_opencv_superres=ON \
             -D BUILD_opencv_ts=ON \
             -D BUILD_opencv_video=ON \
             -D BUILD_opencv_videoio=ON \
             -D BUILD_opencv_videostab=ON \
             -D WITH_1394=ON \
             -D WITH_EIGEN=ON \
             -D WITH_FFMPEG=ON \
             -D WITH_GDAL=OFF \
             -D WITH_GPHOTO2=ON \
             -D WITH_GIGEAPI=ON \
             -D WITH_GSTREAMER=ON \
             -D WITH_GTK=OFF \
             -D WITH_INTELPERC=OFF \
             -D WITH_IPP=ON \
             -D WITH_IPP_A=OFF \
             -D WITH_JASPER=ON \
             -D WITH_JPEG=ON \
             -D WITH_LIBV4L=ON \
             -D WITH_OPENCL=ON \
             -D WITH_OPENCLAMDBLAS=OFF \
             -D WITH_OPENCLAMDFFT=OFF \
             -D WITH_OPENCL_SVM=OFF \
             -D WITH_OPENEXR=ON \
             -D WITH_OPENGL=ON \
             -D WITH_OPENMP=OFF \
             -D WITH_OPENNI=OFF \
             -D WITH_PNG=ON \
             -D WITH_PTHREADS_PF=OFF \
             -D WITH_PVAPI=ON \
             -D WITH_QT=OFF \
             -D WITH_TBB=ON \
             -D WITH_TIFF=ON \
             -D WITH_UNICAP=OFF \
             -D WITH_V4L=ON \
             -D WITH_VTK=ON \
             -D WITH_WEBP=ON \
             -D WITH_XIMEA=OFF \
             -D WITH_XINE=OFF \
             -D CUDA_NVCC_FLAGS=--expt-relaxed-constexpr \
             -D CUDA_FAST_MATH=ON \
             -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
             -D CUDA_HOST_COMPILER=/usr/bin/gcc \
             -D CUDA_ARCH_BIN="6.1 7.5 8.6" \
             -D WITH_CUBLAS=ON \
             -D WITH_CUDA=ON \
             -D WITH_CUFFT=ON \
             -D WITH_NVCUVID=ON \
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
             -D PYTHON3_INCLUDE_DIR=`python -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())'` \
             -D PYTHON3_NUMPY_INCLUDE_DIRS=`python -c 'import numpy; print(numpy.get_include())'` \
             -D PYTHON3_LIBRARIES=`find /home/dev/conda/envs/research/lib -name 'libpython*.so'` \
             -D BUILD_TESTS=OFF \
             -D BUILD_PERF_TESTS=OFF \
             -D BUILD_EXAMPLES=OFF \
             .. \
    && make -j $(($(nproc) + 1)) \
    && make install \
    && rm ~/opencv/${OPEN_CV_VERSION}.zip ~/opencv/opencv_contrib-${OPEN_CV_VERSION}.zip

# Install Python library
RUN conda config --add channels conda-forge \ 
    && conda install --yes -n research -c pytorch -c conda-forge \
    pytorch torchvision cudatoolkit=11.1 \
    mamba \
    scipy \
    sympy \
    pandas \
    openpyxl \
    pytables \
    seaborn \
    scikit-learn \
    scikit-image \
    jedi jupyterlab jupyterlab-git ipywidgets \
    && conda config --add channels conda-forge \
    && conda activate research \
# Pip Install
    && pip install japanize-matplotlib torchinfo --no-cache-dir \
    && conda clean -y --all &>/dev/null && chown -hR dev:dev /home/dev/conda/ \
    && echo ". /home/dev/conda/etc/profile.d/conda.sh" >> ~/.bashrc \
    && echo "conda activate research" >> ~/.bashrc
ENV PATH $PATH:/home/dev/conda/envs/research/bin

# Make symbolic link to a lib. Required for OpenCV and ffmpeg?
RUN ln -s /home/dev/conda/envs/research/lib/libopenh264.so /home/dev/conda/envs/research/lib/libopenh264.so.5

# Install Jupyter lab extensions
ENV NOTEBOOK_DIR /home/dev/notebooks
ENV DEFAULT_TEMPLATE_DIR ${NOTEBOOK_DIR}/templates
ENV HOST_TEMPLATE_DIR ${NOTEBOOK_DIR}/host/templates
RUN jupyter lab --generate-config \
    && jupyter_lab_config=$(jupyter --config-dir)/jupyter_lab_config.py \
    && mkdir -p ${DEFAULT_TEMPLATE_DIR} \
    && echo "c.ContentsManager.allow_hidden = True" >> ${jupyter_lab_config} \
    && echo "c.FileContentsManager.allow_hidden = True" >> ${jupyter_lab_config} \
    && echo "c.ServerApp.tornado_settings = {'shell_command': ['bash']}" >> ${jupyter_lab_config} \
    && sed -i \
    -e "s/# c.ServerApp.ip = 'localhost'/c.ServerApp.ip = '0.0.0.0'/" \
    -e "s/# c.ServerApp.allow_root = False/c.ServerApp.allow_root = True/" \
    -e "s/# c.ServerApp.allow_remote_access = False/c.ServerApp.allow_remote_access = True/" \
    -e "s:# c.ServerApp.root_dir = '':c.ServerApp.root_dir = '$NOTEBOOK_DIR':" \
    ${jupyter_lab_config} \
    && mkdir -p ~/.jupyter/lab/user-settings/@jupyterlab 
COPY --chown=dev:dev ["./check_gpu.py", "./mnist.py", "/home/dev/notebooks/templates/"]
COPY --chown=dev:dev ./jupyter_config/ /home/dev/.jupyter/lab/user-settings/@jupyterlab/

# Laucher
ENV LAUNCH_SCRIPT_DIR /home/dev/.local/bin
ENV LAUNCH_SCRIPT_PATH ${LAUNCH_SCRIPT_DIR}/run_jupyter.sh
COPY --chown=dev:dev ./run_jupyter.sh /home/dev/.local/bin/

RUN chmod +x ${LAUNCH_SCRIPT_PATH}
USER root
EXPOSE 8888
CMD ["run_jupyter.sh"]
