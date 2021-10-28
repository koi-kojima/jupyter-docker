# Jupyter Docker Geometric

## Installed libraries
* PyTorch Geometric
* PyMesh

## Docker image
`ghcr.io/koi-kojima/jupyter/geometric`

See https://github.com/koi-kojima/jupyter-docker/pkgs/container/jupyter%2Fgeometric for other versions.

## Build

```bash
# Version of this image.
VERSION=??
# See https://github.com/koi-kojima/jupyter-docker/pkgs/container/jupyter
# Example: PARENT_VERSION=v8.1-cuda
PARENT_VERSION=??
docker build --tag ghcr.io/koi-kojima/jupyter/geometric:${VERSION} --build-arg PARENT_VERSION=${PARENT_VERSION} .
```

