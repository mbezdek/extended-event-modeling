# extended-event-modeling


## Installation
Follow instructions to install conda: https://www.anaconda.com/products/individual
### Install packages
```conda env create -f environment.yml```

```conda activate sem-37```
### Install interactive extensions for Jupyter notebook/lab
```jupyter labextension install @jupyter-widgets/jupyterlab-manager```

```jupyter nbextension enable --py widgetsnbextension```
### Install ffmpeg (optional)
#### We need ffmpeg for scikit-video to work

Download this release: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.7z

Unzip file using: https://www.7-zip.org/

Remember where you install ffmpeg, then adding that path to PATH, example: https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/
