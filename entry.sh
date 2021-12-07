#! /usr/bin/env bash
set -e

USER_ID=${LOCAL_UID:-1000}
GROUP_ID=${LOCAL_GID:-1000}

# Start sshd as root
/usr/sbin/sshd

# Add the dev user
groupadd --gid ${GROUP_ID} dev
useradd --home-dir /home/dev --uid ${USER_ID} --gid ${GROUP_ID} --shell /bin/bash dev
sed -i -e "s/dev:x:/dev::/g" /etc/passwd

exec gosu dev "$@"
