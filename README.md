# jupyter-docker

Docker image of jupyterlab with PyTorch

## Tag List

See https://github.com/users/koi-kojima/packages/container/package/jupyter

Postfix "cuda" means opencv is built with cuda. "cpu_cv" means the image uses prebuild opencv provided via [pip](https://pypi.org/project/opencv-python/).
Both images are based on cuda11.1 and cudnn8 from nvidia image, so PyTorch can use gpus in both images.

## Installed Libraries

* Python 3.9
* [Anaconda](https://github.com/conda-forge/miniforge)
* Numelic
  * [numpy](https://numpy.org/doc/stable/user/index.html)
  * [scipy](https://docs.scipy.org/doc/scipy/reference/)
* Table
  * [pandas](https://pandas.pydata.org/docs/index.html)
  * [PyTables](https://pypi.org/project/tables/)
  * [openpyxl](https://openpyxl.readthedocs.io/en/stable/)
* Graph
  * [matplotlib](https://matplotlib.org/stable/api/index.html)
  * [seaborn](https://seaborn.pydata.org/)
  * [japanize-matplotlib](https://github.com/uehara1414/japanize-matplotlib)
* Jupyter
  * [Jupyter Lab](https://jupyterlab.readthedocs.io/en/stable/)
  * [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/)
  * [jupyterlab-git](https://github.com/jupyterlab/jupyterlab-git)
  * [jedi](https://github.com/davidhalter/jedi)
* Image
  * [OpenCV](https://docs.opencv.org/master/)
    * With CUDA("-cuda" postfix version)
    * Without CUDA("-cpu\_cv")
  * [Pillow](https://pillow.readthedocs.io/en/stable/)
  * [scikit-image](https://scikit-image.org/)
* Machine Learning
  * [scikit-learn](https://scikit-learn.org/stable/user_guide.html)
  * [PyTorch(with CUDA)](https://pytorch.org/)
  * [torchvision](https://pytorch.org/vision/stable/index.html)
  * torchaudio("-cpu\_cv" only)

## How to use

### Jupyter Lab
```
docker run -it \
           --gpus all \
           --name "jupyter.$USER" \
           --mount type=bind,source=/path/to/local,target=/home/dev/notebooks
           --env LOCAL_UID=$(id $USER --user) \
           --env LOCAL_GID=$(id $USER --group) \
           -p 8888 \
           ghcr.io/koi-kojima/jupyter:v7.9-cuda
```

Execute `docker ps` to see the port to access jupyter in docker.

### SSH
`docker run -d --gpus all --name "ssh.$USER" -p 8888 ghcr.io/koi-kojima/jupyter:v7.9-cuda /usr/sbin/sshd -D`

DO NOT run with `-it`.
The sshd process doesn't accept any inputs (even Ctrl-D or Ctrl-C) so you can't do anything.

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
  * Start by executing `/usr/sbin/sshd` or `/usr/sbin/sshd -D`.
  * The second is useful if you want to start container with the command.
  * "-cpu\_cv" image will start ssh server in default.
* User permission
  * When the lab starts, the script will execute `usermod` command to change id of user in container. It will solve permission problem if you bind local file into container. ("-cuda" only)
