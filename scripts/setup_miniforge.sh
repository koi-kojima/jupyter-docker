#! /usr/bin/env bash
set -eux

PYTHON_VER=3.10

curl -L -sS -o ${HOME}/miniforge.sh "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"
/bin/bash ${HOME}/miniforge.sh -b -p ${HOME}/conda
rm ${HOME}/miniforge.sh
ln -s ${HOME}/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
${HOME}/conda/bin/conda update conda --quiet --yes > /dev/null
${HOME}/conda/bin/conda install --yes python=${PYTHON_VER} numpy=1.21
${HOME}/conda/bin/conda clean --yes --index-cache > /dev/null
echo ". ${HOME}/conda/etc/profile.d/conda.sh" >> ${HOME}/.bashrc
echo "conda activate" >> ${HOME}/.bashrc
ln -s ${HOME}/conda/bin/python "/usr/bin/python${PYTHON_VER}"
ln -s ${HOME}/conda/bin/python /usr/bin/python
