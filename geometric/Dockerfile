ARG PARENT_VERSION
# Set parent image version. See https://github.com/koi-kojima/jupyter-docker/pkgs/container/jupyter
FROM ghcr.io/koi-kojima/jupyter:${PARENT_VERSION}

# Install dependency. (User root)
RUN apt-get update --fix-missing -qq && apt-get install -y --no-install-recommends \
    libgmp-dev libmpfr-dev libgmpxx4ldbl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

USER dev
# Install PyTorch Geometric and PyYAML
RUN conda activate research && \
    mamba install -c pyg -c conda-forge -y pyg tensorboard && \
    pip install nose --no-cache-dir && \
    mamba clean -y --all

# Install PyMesh
ARG PYMESH_BRANCH=${PYMESH_BRANCH:-main}
RUN conda activate research && \
    git clone --depth=1 -b $PYMESH_BRANCH https://github.com/PyMesh/PyMesh.git pymesh && \
    cd pymesh && \
    git submodule update --init --recursive --recommend-shallow --depth 1 && \
    ./setup.py build && \
    ./setup.py install && \
    python -c "import pymesh; pymesh.test()"

# Install PD-MeshNet
RUN conda activate research && \
    git clone --depth=1 https://github.com/MIT-SPARK/PD-MeshNet.git meshnet && \
    cd meshnet && \
    pip install -e .

