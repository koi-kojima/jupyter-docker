#! /usr/bin/env bash
set -e

USER_ID=${LOCAL_UID:-1000}
GROUP_ID=${LOCAL_GID:-1000}

# Add the dev user
adduser --quiet --gecos "" --disabled-password --uid ${USER_ID} --home /home/dev dev > /dev/null
sed -i -e "s/dev:x:/dev::/g" /etc/passwd
# chown -R dev:dev /home/dev

exec "$@"

