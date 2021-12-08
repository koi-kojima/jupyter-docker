set -ue

docker run -it --rm -p 8888 \
  -e LOCAL_UID=$(id -u) \
  -e LOCAL_GID=$(id -g) \
  --gpu all \
  ghcr.io/koi-kojima/jupyter:v8.5-cuda

