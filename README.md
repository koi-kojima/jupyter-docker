# jupyter-docker

Docker image of jupyterlab with PyTorch

## Tag List

See https://github.com/users/koi-kojima/packages/container/package/jupyter

Postfix "cuda" means opencv is built with cuda.
This image is based on cuda11.3 and cudnn8 from nvidia image, so PyTorch can use gpus in the image.

## Installed Libraries

* Python 3.9
* [Anaconda](https://github.com/conda-forge/miniforge)
  * This image uses MiniForge distributiuon with mamba.
* Numelic
  * [numpy](https://numpy.org/doc/stable/user/index.html)
  * [scipy](https://docs.scipy.org/doc/scipy/reference/)
  * [sympy](https://www.sympy.org/en/index.html)
* Table
  * [pandas](https://pandas.pydata.org/docs/index.html)
  * [PyTables](https://pypi.org/project/tables/)
  * [h5py](https://docs.h5py.org/en/stable/)
* Graph
  * [matplotlib](https://matplotlib.org/stable/api/index.html)
  * [seaborn](https://seaborn.pydata.org/)
  * [japanize-matplotlib](https://github.com/uehara1414/japanize-matplotlib)
* Jupyter
  * [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/)
    * Version 3.x
    * NodeJS is also installed via conda.
  * [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/)
  * [jupyterlab-git](https://github.com/jupyterlab/jupyterlab-git)
  * [jedi](https://github.com/davidhalter/jedi)
* Image
  * [OpenCV](https://docs.opencv.org/master/)
    * With CUDA("-cuda" postfix version)
  * [Pillow](https://pillow.readthedocs.io/en/stable/)
  * [scikit-image](https://scikit-image.org/)
  * [PyAV](https://github.com/PyAV-Org/PyAV)
* Machine Learning
  * [scikit-learn](https://scikit-learn.org/stable/user_guide.html)
  * [PyTorch(with CUDA)](https://pytorch.org/)
    * [torchvision](https://pytorch.org/vision/stable/index.html)
    * [Pytorch Lightning](https://www.pytorchlightning.ai/)
    * [TorchMetrics](https://torchmetrics.readthedocs.io/en/stable/)

## How to use

### Jupyter Lab
Edit [`docker_run.sh`](https://github.com/koi-kojima/jupyter-docker/blob/main/docker_run.sh) to change the image version and add volume mounts.
Then run the script to create container.

Execute `docker ps` to see the port to access jupyter in docker.
**Be careful not to bind sensitive files such as /etc. The user in container can access and modify the file.**

### SSH
Edit [`docker_run.sh`](https://github.com/koi-kojima/jupyter-docker/blob/main/docker_run.sh) to expose port 22.
Then run the script and connect to the container.
Username is "dev" and no password is required.
**Be careful not to bind sensitive files such as /etc. The user in container can access and modify the file.**

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
  * "-cpu\_cv" image will start ssh server in default.
* User permission
  * When the lab starts, the script will execute `useradd` command to change id of user in container. It will solve permission problem if you bind local file into container. ("-cuda" only)
