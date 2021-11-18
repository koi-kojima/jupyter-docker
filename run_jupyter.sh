#!/usr/bin/env bash
set -e

if [ -z "${LOCAL_UID}" ] && [ -z "${LOCAL_GID}" ] ; then
	/usr/sbin/sshd
	echo "Launch jupyter lab as root"
	/home/dev/conda/envs/research/bin/jupyter lab --no-browser
else
	if [ -z "${LOCAL_UID}" ] ; then
	  USER_ID=$(id dev --user)
	else
	  USER_ID=${LOCAL_UID:-1000}
	fi
	if [ -z "${LOCAL_GID}" ] ; then
	  GROUP_ID=$(id dev --group)
	else
	  GROUP_ID=${LOCAL_GID:-1000}
	fi
	PYTHONPATH=${PYTHONPATH:-/home/dev/.local/lib/python3.9/site-packages}

	# Open SSH access
	/usr/sbin/sshd

	if [ "${USER_ID}" != "$(id dev --user)" ] ; then
	  usermod --uid $USER_ID --non-unique dev
	fi
	if [ "${GROUP_ID}" != "$(id dev --group)" ] ; then
	  sudo groupmod --gid $GROUP_ID --non-unique dev
	fi

	echo "Launch jupyter lab as User ${USER_ID}:${GROUP_ID}"
	sudo \
		-u dev \
		-E PYTHONPATH=${PYTHONPATH} \
		-E LD_LIBRARY_PATH=${LD_LIBRARY_PATH} \
		/home/dev/conda/envs/research/bin/jupyter lab --no-browser
fi

