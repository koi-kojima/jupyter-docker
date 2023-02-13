# jupyter-docker

Docker image of jupyter lab with PyTorch

## Tag List

See https://github.com/users/koi-kojima/packages/container/package/jupyter

## Installed Libraries

* Python 3.10
* Calculation
  * [numpy](https://numpy.org/doc/stable/user/index.html)
  * [scipy](https://docs.scipy.org/doc/scipy/reference/)
  * [sympy](https://www.sympy.org/en/index.html)
* Table
  * [pandas](https://pandas.pydata.org/docs/index.html)
* Graph
  * [matplotlib](https://matplotlib.org/stable/api/index.html)
  * [seaborn](https://seaborn.pydata.org/)
  * [japanize-matplotlib](https://github.com/uehara1414/japanize-matplotlib)
* Jupyter
  * [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/)
    * Version 3.x
* Image
  * [OpenCV](https://docs.opencv.org/master/)
    * Built for GPU
  * [Pillow](https://pillow.readthedocs.io/en/stable/)
* Machine Learning
  * [scikit-learn](https://scikit-learn.org/stable/user_guide.html)
  * [PyTorch(with CUDA)](https://pytorch.org/)
    * [torchvision](https://pytorch.org/vision/stable/index.html)
    * [lightning](https://www.pytorchlightning.ai/)
    * [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/)
    * [torchinfo](https://github.com/TylerYep/torchinfo)
    * timm
    * classy_vision
  * [umap-learn](https://umap-learn.readthedocs.io/en/latest/)
  * tensorboardX
* ML manager
  * hydra-core
  * wandb

For all installation, see Pip.Dockerfile.

## How to use

### Jupyter Lab

Run this image without command. Jupyter launches with port 8888.
Publish this or other port to access the instance.

### SSH

You can access this container via SSH.
Run `/usr/sbin/sshd` to start SSH server.
Root has no password.

## Improvements

* Font of terminal in Jupyter Lab
  * Changed to DejaVu Sans Mono or something.
* Allow root in default.
  * no need to add "--allow-root"
* Show all files in file tree
  * Hidden files will be shown.
* Dark theme
* Allow access from outside of machine.
  * Use with care because it allows access from outside of your machine.
* SSH server
  * Start by executing `/usr/sbin/sshd`.
* User permission
  * Sorry, this improvement was removed.
  * For VS Code user, this doesn't affect user experience.
