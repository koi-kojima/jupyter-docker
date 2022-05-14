ARG CUDA_MAJOR_VERSION=${CUDA_MAJOR_VERSION:-11}
ARG CUDA_MINOR_VERSION=${CUDA_MINOR_VERSION:-3}

FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04 as opencv
ARG CUDA_MAJOR_VERSION=${CUDA_MAJOR_VERSION:-11}
ARG CUDA_MINOR_VERSION=${CUDA_MINOR_VERSION:-3}

ARG OPEN_CV_VERSION=${OPEN_CV_VERSION:-4.5.5}

RUN umask 000 && mkdir /home/dev 
WORKDIR /home/dev
ENV HOME /home/dev
ENV PATH ${HOME}/conda/bin:$PATH
ENV PATH $HOME/.local/bin:$PATH

COPY --chmod=755 ["./apt_install.sh", "./setup_miniforge.sh", "/install_scripts/"]

RUN umask 000 && /install_scripts/apt_install.sh

# Install miniforge
RUN umask 000 && /install_scripts/setup_miniforge.sh
SHELL ["/bin/bash", "-l", "-c"]

# Install OpenCV
RUN umask 000 && mkdir ${HOME}/opencv ${HOME}/opencv/opencv-build && cd ${HOME}/opencv \
    && curl -L -o opencv-${OPEN_CV_VERSION}.zip "https://github.com/opencv/opencv/archive/${OPEN_CV_VERSION}.zip" \
    && curl -L -o opencv_contrib-${OPEN_CV_VERSION}.zip "https://github.com/opencv/opencv_contrib/archive/${OPEN_CV_VERSION}.zip" \
    && unzip -q opencv-${OPEN_CV_VERSION}.zip \
    && unzip -q opencv_contrib-${OPEN_CV_VERSION}.zip \
    && rm opencv-${OPEN_CV_VERSION}.zip opencv_contrib-${OPEN_CV_VERSION}.zip
RUN umask 000 \
    && cd ${HOME}/opencv/opencv-${OPEN_CV_VERSION} \
    && mkdir build && cd build \
    && conda activate \
    && CC=gcc CXX=g++ cmake \
             -D CMAKE_BUILD_TYPE=Release \
             -D CMAKE_INSTALL_PREFIX=${HOME}/opencv/opencv-build \
             -D OPENCV_EXTRA_MODULES_PATH=${HOME}/opencv/opencv_contrib-${OPEN_CV_VERSION}/modules \
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
             -D WITH_FFMPEG=ON \
             -D WITH_GPHOTO2=ON \
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
             -D PYTHON3_INCLUDE_DIR=$(python -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())') \
             -D PYTHON3_NUMPY_INCLUDE_DIRS=$(python -c 'import numpy; print(numpy.get_include())') \
             -D PYTHON3_LIBRARIES=$(find ${HOME}/conda/lib -name 'libpython*.so') \
             -D BUILD_TESTS=OFF \
             -D BUILD_PERF_TESTS=OFF \
             -D BUILD_EXAMPLES=OFF \
             -D INSTALL_PYTHON_EXAMPLES=ON \
             .. \
    && make -j $(($(nproc) + 1)) \
    && make install
RUN cd ${HOME}/opencv/opencv-${OPEN_CV_VERSION}/build/python_loader \
    && sed -i -e "/],/a include_package_data=True," -e "/],/a zip_safe=False," setup.py \
    && sed -i -e "s:opencv-${OPEN_CV_VERSION}/build:opencv-build:" cv2/config.py \
    && sed -i -e "s:opencv-${OPEN_CV_VERSION}/build/lib/python3:opencv-build/lib/python3.9/site-packages/cv2/python-3.9:" cv2/config-3.9.py \
    && python setup.py bdist_wheel \
    && mv dist/*.whl ${HOME}/opencv/opencv-build/

FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
ARG CUDA_MAJOR_VERSION=${CUDA_MAJOR_VERSION:-11}
ARG CUDA_MINOR_VERSION=${CUDA_MINOR_VERSION:-3}

ENV HOME /home/dev
ENV PATH ${HOME}/conda/bin:$PATH
ENV PATH $HOME/.local/bin:$PATH

COPY --chmod=755 ["./apt_install.sh", "./setup_miniforge.sh", "/install_scripts/"]

RUN umask 000 && \
    mkdir /home/dev && \
    /install_scripts/apt_install.sh && \
    /install_scripts/setup_miniforge.sh
WORKDIR /home/dev

SHELL ["/bin/bash", "-l", "-c"]

# SSH abd sudo setting
RUN sed -i -e "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/" \
           -e "s/#PermitEmptyPasswords no/PermitEmptyPasswords yes/" \
           /etc/ssh/sshd_config && \
    sed -i -e "s/root:x:/root::/g" /etc/passwd && \
    mkdir /run/sshd && \
    echo "dev ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/dev && \
    chmod 0440 /etc/sudoers.d/dev

# Copy initial scripts
RUN sed -i -e "s/#force_color_prompt=yes/force_color_prompt=yes/g" /root/.bashrc \
    && cp /root/.bashrc /root/.profile ${HOME}/ \
    && chmod 777 ${HOME}/.bashrc ${HOME}/.profile

# Install Python library
# Default channel is required to install latest version of torchvision.
RUN umask 000 && conda config --append channels defaults \
    && conda activate \
    && mamba install --yes -c pytorch -c nvidia \
       pytorch torchvision av torchdata cudatoolkit=${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION} \
       pytorch-lightning \
       torchmetrics \
# The new version of grpc cause library error in launching tensorboard.
       grpcio=1.42.0 \
       scipy \
       sympy \
       pandas \
       openpyxl \
       pytables \
       h5py \
       seaborn \
       scikit-learn \
       scikit-image \
       tqdm \
       umap-learn \
       jedi jupyterlab nodejs jupyterlab-git ipywidgets \
       pylint autopep8 \
    && conda config --remove channels defaults \
# Pip Install
    && pip install japanize-matplotlib torchinfo pytorchvideo --no-cache-dir \
    && mamba clean -y --all &> /dev/null
ENV PATH $PATH:${HOME}/conda/bin

COPY --from=opencv ["${HOME}/opencv/opencv-build", "${HOME}/opencv/opencv-build/"]
RUN umask 000 \
    && conda activate \
    && pip install ${HOME}/opencv/opencv-build/*.whl --no-cache-dir

# Make symbolic link to a lib. Required for OpenCV and ffmpeg?
RUN ln -s ${HOME}/conda/lib/libopenh264.so ${HOME}/conda/lib/libopenh264.so.5

# Install Jupyter lab extensions
ENV NOTEBOOK_DIR ${HOME}/notebooks
ENV DEFAULT_TEMPLATE_DIR ${NOTEBOOK_DIR}/templates
RUN umask 000 \
    && jupyter lab --generate-config \
    # && jupyter notebook --generate-config \
    && jupyter_lab_config=$(jupyter --config-dir)/jupyter_lab_config.py \
    && jupyter_notebook_config=$(jupyter --config-dir)/jupyter_notebook_config.py \
    && mkdir -p ${DEFAULT_TEMPLATE_DIR}\
    && echo "c.ContentsManager.allow_hidden = True" >> ${jupyter_lab_config} \
    && echo "c.FileContentsManager.allow_hidden = True" >> ${jupyter_lab_config} \
    && echo "c.ServerApp.terminado_settings = {'shell_command': ['/usr/bin/bash']}" >> ${jupyter_lab_config} \
    && sed -i \
    -e "s/# c.\(.*\).ip = 'localhost'/c.\1.ip = '0.0.0.0'/" \
    -e "s/# c.\(.*\).allow_root = False/c.\1.allow_root = True/" \
    -e "s/# c.\(.*\).allow_remote_access = False/c.\1.allow_remote_access = True/" \
    -e "s:# c.\(.*\).root_dir = '':c.\1.root_dir = '$NOTEBOOK_DIR':" \
    -e "s/# c.\(.*\).allow_hidden = False/c.\1.allow_hidden = True/" \
    -e "s/# c.\(.*\).open_browser = True/c.\1.open_browser = False/" \
    ${jupyter_lab_config} \
    # ${jupyter_notebook_config} \
    && mkdir -p $(jupyter --config-dir)/lab/user-settings/@jupyterlab 
COPY --chmod=777 ["./check_gpu.py", "./mnist.py", "${DEFAULT_TEMPLATE_DIR}/"]
COPY --chmod=777 ./jupyter_config/ ${HOME}/.jupyter/lab/user-settings/@jupyterlab/

# Laucher
ENV LAUNCH_SCRIPT_DIR ${HOME}/.local/bin
ENV LAUNCH_SCRIPT_PATH ${LAUNCH_SCRIPT_DIR}/run_jupyter.sh
COPY --chmod=777 ["./run_jupyter.sh", "./entry.sh", "${LAUNCH_SCRIPT_DIR}/"]

RUN chmod +x ${LAUNCH_SCRIPT_PATH} \
    && chmod 777 -R ${HOME}/.jupyter \
    && chmod 777 -R ${HOME}/.local \
    && ln -s ${NOTEBOOK_DIR} /work
# For Jupyter
EXPOSE 8888
# For SSH
EXPOSE 22
CMD ["run_jupyter.sh"]
ENTRYPOINT ["/home/dev/.local/bin/entry.sh"]

