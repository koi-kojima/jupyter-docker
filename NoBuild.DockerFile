# syntax = docker/dockerfile:1

ARG CUDA_VERSION=11.6.2
# --------------------------------------------------------------------------------
FROM condaforge/mambaforge:22.9.0-2 as installer
ARG PYTHON_VER=3.10
ARG CUDA_VERSION

# Update conda and install base
RUN <<EOF
mamba install --yes python=${PYTHON_VER} numpy=1.21
mamba clean -ya
EOF

# Install Python library
# The new version of grpc cause library error in launching tensorboard.
RUN <<EOF
conda config --append channels defaults
mamba install --yes -c pytorch -c nvidia \
    pytorch-cuda=$(echo "${CUDA_VERSION}" | cut -d'.' -f 1-2) torchvision av torchdata \
    pytorch-lightning \
    torchmetrics \
    torchtext \
    grpcio=1.42.0 \
    scipy \
    sympy \
    pandas \
    seaborn \
    scikit-learn \
    tqdm \
    umap-learn \
    lz4 \
    jedi jupyterlab nodejs jupyterlab-git ipywidgets \
    pylint autopep8
conda config --remove channels defaults
mamba clean -y --all
EOF

# Pip Install
RUN <<EOF
pip install \
    japanize-matplotlib \
    torchinfo \
    pytorchvideo \
    opencv-python \
    opencv-contrib-python \
    hydra-core \
    wandb \
    timm \
    classy_vision \
    --no-cache-dir
EOF

# --------------------------------------------------------------------------------
FROM ubuntu:jammy as image

RUN <<EOF
# sed -i -e 's%http://[^ ]\+%mirror://mirrors.ubuntu.com/mirrors.txt%g' /etc/apt/sources.list
DEBIAN_FRONTEND=noninteractive \
apt-get update --fix-missing && apt-get install -y \
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
    vim \
    zip \
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

# GPU Setting and Environments
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV HOME="/root"
ENV CONDA_DIR="/opt/conda"
ENV PATH=${CONDA_DIR}/bin:$PATH
ENV NOTEBOOK_DIR=${HOME}/notebooks
ENV DEFAULT_TEMPLATE_DIR=${NOTEBOOK_DIR}/templates

# Copy Python
RUN echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ${HOME}/.bashrc
COPY --from=installer /opt/conda /opt/conda

# Setup Jupyter lab
COPY --chmod=755 ["./scripts/jupyter_setting.sh", "/install_scripts/"]
RUN /install_scripts/jupyter_setting.sh

COPY ./jupyter_config/ $HOME/.jupyter/lab/user-settings/@jupyterlab/
COPY --chmod=755 ["./check_gpu.py", "./mnist*.py", "./qmnist.py", "${DEFAULT_TEMPLATE_DIR}/"]
COPY --chmod=755 <<EOF $HOME/.local/bin/run_jupyter.sh
${CONDA_DIR}/bin/jupyter lab --no-browser
EOF

EXPOSE 8888
EXPOSE 22
CMD ["run_jupyter.sh"]
