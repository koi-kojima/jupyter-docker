set -ue

docker run -it --rm -P \
  -e LOCAL_UID=$(id -u) \
  -e LOCAL_GID=$(id -g) \
  --gpu all \
  ghcr.io/koi-kojima/jupyter:v8.3-cuda

