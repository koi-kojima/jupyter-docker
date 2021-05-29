# jupyter-docker
Docker image of jupyterlab with PyTorch

## Tag List
See https://github.com/users/koi-kojima/packages/container/package/jupyter

Postfix "cuda" means opencv is built with cuda. "cpu_cv" means the image uses prebuild opencv provided via [pip](https://pypi.org/project/opencv-python/).
Both images are based on cuda11.1 and cudnn8 from nvidia image, so PyTorch can use gpus in both images.

## Installed Libraries
* Python 3.9
* Anaconda
* Numelic
  * numpy
  * scipy
* Table
  * pandas
  * [PyTables](https://pypi.org/project/tables/)
  * openpyxl
* Graph
  * matplotlib
  * seaborn
  * japanize-matplotlib
* Jupyter
  * Jupyter Lab
  * ipywidgets
  * jupyterlab-git
  * jedi
* Image
  * OepnCV
    * With CUDA("-cuda" postfix version)
    * Without CUDA("-cpu_cv")
  * Pillow
  * scikit-image
* Machine Learning
  * scikit-learn
  * PyTorch(with CUDA)
  * torchvision
  * torchaudio("-cpu_cv" only)

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
  * "-cpu_cv" image will start ssh server in default.
* User permission
  * When the lab starts, the script will execute `usermod` command to change id of user in container. It will solve permission problem if you bind local file into container. ("-cuda" only)

