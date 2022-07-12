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

COPY --chmod=755 ["./setup_miniforge.sh", "/install_scripts/"]

ENV PATH $HOME/conda/bin:$PATH
ENV PATH $HOME/.local/bin:$PATH
# Install Miniconda
RUN /install_scripts/setup_miniforge.sh
SHELL ["/bin/bash", "-l", "-c"]

# Install Python library
RUN conda config --append channels defaults \
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
RUN conda activate \
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
    && mkdir -p $(jupyter --config-dir)/lab/user-settings/@jupyterlab \
    && ln -s ${NOTEBOOK_DIR} /work

COPY ./jupyter_config/ $HOME/.jupyter/lab/user-settings/@jupyterlab/
COPY --chmod=755 ["./check_gpu.py", "./mnist*.py", "${DEFAULT_TEMPLATE_DIR}/"]
COPY --chmod=755 ./run_jupyter.sh $HOME/.local/bin/

EXPOSE 8888
EXPOSE 22
CMD ["run_jupyter.sh"]
