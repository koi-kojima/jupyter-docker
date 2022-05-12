#! /usr/bin/env bash
set -e

USER_ID=${LOCAL_UID:-1000}
GROUP_ID=${LOCAL_GID:-1000}

# Start sshd as root
/usr/sbin/sshd

# Add the dev user
if ! id dev &>/dev/null ; then
  echo "Creating dev user with UID=${USER_ID}, GID=${GROUP_ID}"
  groupadd --gid ${GROUP_ID} dev
  useradd --home-dir /home/dev --uid ${USER_ID} --gid ${GROUP_ID} --shell /bin/bash dev
  chown -R dev:dev ~/conda/pkgs/url*
  chown -R dev:dev ~/conda/conda-meta
  sed -i -e "s/dev:x:/dev::/g" /etc/passwd
else
  echo "User dev already exists."
fi

exec gosu dev "$@"
