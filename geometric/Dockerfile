ARG PARENT_VERSION
# Set parent image version. See https://github.com/koi-kojima/jupyter-docker/pkgs/container/jupyter
FROM ghcr.io/koi-kojima/jupyter:${PARENT_VERSION}

USER dev
# Install PyTorch Geometric and PyRAML
RUN mamba install pyg -c pyg -c conda-forge && \
    pip install PyRAML nose --no-cache-dir && \
    mamba clean -y --all

# Install PyMesh
ARG PYMESH_BRANCH=${PYMESH_BRANCH:main}
RUN git clone --single-branch -b $PYMESH_BRANCH https://github.com/PyMesh/PyMesh.git pymesh && \
    cd pymesh && \
    git submodule update --init && \
    ./setup.py bdist_wheel && \
    rm -rf build_3.7 third_party/build && \
    python docker/patches/patch_wheel.py dist/pymesh2*.whl && \
    pip install dist/pymesh2*.whl && \
    python -c "import pymesh; pymesh.test()"

