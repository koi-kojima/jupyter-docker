ARG CUDA_RELEASE_VERSION=${CUDA_RELEASE_VERSION:-11.3.1}
FROM nvidia/cuda:${CUDA_RELEASE_VERSION}-base-ubuntu20.04
ARG CUDA_MAJOR_VERSION=${CUDA_MAJOR_VERSION:-11}
ARG CUDA_MINOR_VERSION=${CUDA_MINOR_VERSION:-3}

ENV HOME /root
ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i -e 's%http://[^ ]\+%mirror://mirrors.ubuntu.com/mirrors.txt%g' /etc/apt/sources.list && \
    apt-get update --fix-missing -qq && apt-get install -y \
    curl \
    ca-certificates \
    ffmpeg \
    file \
    git \
    less \
    locales \
    neofetch \
    nkf \
    opencv-data \
    ssh \
    sudo \
    gosu \
    tree \
    unzip \
    zip \
    --no-install-recommends \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# SSH setting
RUN sed -i -e "s/#PermitRootLogin prohibit-password/PermitRootLogin yes/" \
           -e "s/#PermitEmptyPasswords no/PermitEmptyPasswords yes/" \
           /etc/ssh/sshd_config && \
    sed -i -e "s/root:x:/root::/g" /etc/passwd && \
    mkdir /run/sshd

COPY --chmod=755 ["./scripts/setup_miniforge.sh", "./scripts/jupyter_setting.sh", "/install_scripts/"]

ENV PATH $HOME/conda/bin:$PATH
ENV PATH $HOME/.local/bin:$PATH
# Install Miniconda
RUN /install_scripts/setup_miniforge.sh
SHELL ["/bin/bash", "-l", "-c"]

# Install Python library
RUN conda config --append channels defaults \
    && conda activate \
    && mamba install --yes -c pytorch \
       pytorch torchvision av torchdata cudatoolkit=${CUDA_MAJOR_VERSION}.${CUDA_MINOR_VERSION} \
       pytorch-lightning \
       torchmetrics \
# The new version of grpc cause library error in launching tensorboard.
       grpcio=1.42.0 \
       scipy \
       sympy \
       pandas \
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
    && pip install japanize-matplotlib torchinfo pytorchvideo opencv-python opencv-contrib-python --no-cache-dir \
    && mamba clean -y --all &> /dev/null

# Install Jupyter lab extensions
ENV NOTEBOOK_DIR ${HOME}/notebooks
ENV DEFAULT_TEMPLATE_DIR ${NOTEBOOK_DIR}/templates
RUN /install_scripts/jupyter_setting.sh

COPY ./jupyter_config/ $HOME/.jupyter/lab/user-settings/@jupyterlab/
COPY --chmod=755 ["./check_gpu.py", "./mnist*.py", "${DEFAULT_TEMPLATE_DIR}/"]
COPY --chmod=755 ./scripts/run_jupyter.sh $HOME/.local/bin/

EXPOSE 8888
EXPOSE 22
CMD ["run_jupyter.sh"]
