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
    mamba install -c pyg -c conda-forge -y pyg && \
    pip install PyYAML nose --no-cache-dir && \
    mamba clean -y --all

# Install PyMesh
ARG PYMESH_BRANCH=${PYMESH_BRANCH:-main}
RUN conda activate research && \
    git clone --single-branch -b $PYMESH_BRANCH https://github.com/PyMesh/PyMesh.git pymesh && \
    cd pymesh && \
    git submodule update --init && \
    ./setup.py build && \
    ./setup.py install && \
    python -c "import pymesh; pymesh.test()"

