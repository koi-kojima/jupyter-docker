#!/usr/bin/env bash
USER_ID=${LOCAL_UID:-1000}
GROUP_ID=${LOCAL_GID:-1000}
PYTHONPATH=${PYTHONPATH:-/home/dev/.local/lib/python3.9/site-packages}

usermod -u $USER_ID -o -m dev -d /home/dev
sudo groupmod -g $GROUP_ID dev
sudo \
	-u dev \
	-E PYTHONPATH=${PYTHONPATH} \
	-E LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
	/home/dev/conda/envs/research/bin/jupyter lab --no-browser

