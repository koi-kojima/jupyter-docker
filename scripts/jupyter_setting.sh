#! /usr/bin/env bash

JUPYTER_TOKEN=password_jupyter
set -eux
# conda activate
${CONDA_DIR}/bin/jupyter lab --generate-config
jupyter_lab_config="$(${CONDA_DIR}/bin/jupyter --config-dir)/jupyter_lab_config.py"
mkdir -p "${DEFAULT_TEMPLATE_DIR}"

echo "c.ContentsManager.allow_hidden = True" >> ${jupyter_lab_config}
echo "c.FileContentsManager.allow_hidden = True" >> ${jupyter_lab_config}
echo "c.ServerApp.terminado_settings = {'shell_command': ['/usr/bin/bash']}" >> ${jupyter_lab_config}
sed -i \
    -e "s/# c.\(.*\).ip = 'localhost'/c.\1.ip = '0.0.0.0'/" \
    -e "s/# c.\(.*\).allow_root = False/c.\1.allow_root = True/" \
    -e "s/# c.\(.*\).allow_remote_access = False/c.\1.allow_remote_access = True/" \
    -e "s:# c.\(.*\).root_dir = '':c.\1.root_dir = '$NOTEBOOK_DIR':" \
    -e "s/# c.\(.*\).allow_hidden = False/c.\1.allow_hidden = True/" \
    -e "s/# c.\(.*\).open_browser = True/c.\1.open_browser = False/" \
    ${jupyter_lab_config}
# -e "s/# c.ServerApp.token = '<generated>'/c.ServerApp.token = '$JUPYTER_TOKEN'/"

mkdir -p "$(${CONDA_DIR}/bin/jupyter --config-dir)/lab/user-settings/@jupyterlab"
ln -s ${NOTEBOOK_DIR} /work
