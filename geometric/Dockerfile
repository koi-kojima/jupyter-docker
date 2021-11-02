FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04

ARG MINIFORGE_NAME=Mambaforge
ARG MINIFORGE_VERSION=4.10.3-7
ARG TINI_VERSION=v0.18.0
ARG TARGETPLATFORM=linux/amd64

ENV CONDA_DIR=/opt/conda
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH=${CONDA_DIR}/bin:${PATH}
ENV DEBIAN_FRONTEND=noninteractive

# 1. Install just enough for conda to work
# 2. Keep $HOME clean (no .wget-hsts file), since HSTS isn't useful in this context
# 3. Install miniforge from GitHub releases
# 4. Apply some cleanup tips from https://jcrist.github.io/conda-docker-tips.html
#    Particularly, we remove pyc and a files. The default install has no js, we can skip that
# 5. Activate base by default when running as any *non-root* user as well
#    Good security practice requires running most workloads as non-root
#    This makes sure any non-root users created also have base activated
#    for their interactive shells.
# 6. Activate base by default when running as root as well
#    The root user is already created, so won't pick up changes to /etc/skel
RUN apt-get update > /dev/null && \
    apt-get install --no-install-recommends --yes \
        wget bzip2 ca-certificates \
        automake curl opencv-data unzip vim \
        g++ gcc make \
        libgmp-dev libmpfr-dev libgmpxx4ldbl libboost-all-dev \
        libgl1-mesa-glx libglfw3 \
        git > /dev/null && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    TARGETARCH="$(echo ${TARGETPLATFORM} | cut -d / -f 2)"; case ${TARGETARCH} in "ppc64le") TARGETARCH="ppc64el" ;; *) ;; esac ; \
    wget --no-hsts --quiet https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini-${TARGETARCH} -O /usr/local/bin/tini && \
    chmod +x /usr/local/bin/tini && \
    wget --no-hsts --quiet https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/${MINIFORGE_NAME}-${MINIFORGE_VERSION}-Linux-$(uname -m).sh -O /tmp/miniforge.sh && \
    /bin/bash /tmp/miniforge.sh -b -p ${CONDA_DIR} && \
    rm /tmp/miniforge.sh && \
    conda clean -tipsy && \
    find ${CONDA_DIR} -follow -type f -name '*.a' -delete && \
    find ${CONDA_DIR} -follow -type f -name '*.pyc' -delete && \
    conda clean -afy && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> /etc/skel/.bashrc && \
    echo ". ${CONDA_DIR}/etc/profile.d/conda.sh && conda activate base" >> ~/.bashrc

ENTRYPOINT ["tini", "--"]

# Update CMake
RUN cd /root && \
    curl -o cmake_script.sh -L https://github.com/Kitware/CMake/releases/download/v3.21.4/cmake-3.21.4-linux-x86_64.sh && \
    chmod +x ./cmake_script.sh && \
    mkdir /opt/cmake && \
    ./cmake_script.sh --skip-license --prefix=/opt/cmake %% \
    rm ./cmake_script.sh

ENV PATH=/opt/cmake/bin:${PATH}

# Downgrade Python
RUN mamba install python=3.8 -y

# Install PyTorch Geometric and PyYAML
RUN mamba install -y -c pytorch \
    pytorch==1.4.0 \
    torchvision==0.5.0 \
    cudatoolkit=10.1 \
    numba \
    scipy \
    seaborn \
    scikit-learn \
    scikit-image \
    tensorboard \
    h5py && \
    mamba clean -y --all

RUN  pip install torch-scatter==1.4.0 torch-sparse==0.4.3 torch-cluster==1.4.5 torch-spline-conv==1.1.1 \
    nose \
    japanize-matplotlib\
    torchinfo \
    opencv-python \
     --no-cache-dir

# Install PyTorch Geometric
RUN cd /root && \
    git clone -n https://github.com/rusty1s/pytorch_geometric.git && \
    cd pytorch_geometric && \
    git checkout 082fb83 && \
    sed -i 30,31d setup.py && \
    python setup.py install

# Install PyMesh
RUN cd /root && \
    pip install git+https://github.com/PyMesh/PyMesh.git@93d182c0a7cee446e89fac74033347d900054af4

# Install PD-MeshNet
RUN cd /root && \
    git clone --depth=1 https://github.com/MIT-SPARK/PD-MeshNet.git meshnet && \
    cd meshnet && \
    pip install -e .

COPY test_code.sh /root
