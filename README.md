# extended-event-modeling


## Installation
### Install anaconda
Follow instructions to install conda: https://www.anaconda.com/products/individual
### Install packages
```conda env create -f environment.yml```\
```conda activate sem-pysot-37```


### Install pysot for tracking
```git clone https://github.com/STVIR/pysot``` \
```cd pysot``` \
```python setup.py build_ext --inplace```

For Windows users, you might need Microsoft Visual Studio 2017 to be able to build: \
```python setup.py build_ext --inplace```

After installing MVS 2017, open Cross Tools Command Prompt VS 2017 and run: \
```set DISTUTILS_USE_SDK=1``` \
```cd /path/to/pysot``` \
```python setup.py build_ext --inplace```

If there is a runtime error: \
`RuntimeError (RuntimeError: CuDNN error: CUDNN_STATUS_SUCCESS)` \
Try remove instead of uninstall \
```conda remove cudnn cudatoolkit pytorch torchvision && conda install cuda92 pytorch=0.4.1 -c pytorch```

For GPU running \
```conda install -c anaconda tensorflow-gpu=2``` \
```conda install -c pytorch cuda92``` 


### Install SEM from github repository (be aware of changes)
Using `python -m pip` to avoid confusing if the system has multiple Python versions. \
```python -m pip install git+https://github.com/nicktfranklin/SEM2```


### Install interactive extensions for Jupyter notebook/lab
```jupyter labextension install @jupyter-widgets/jupyterlab-manager``` \
```jupyter nbextension enable --py widgetsnbextension```


### Install ffmpeg (optional)
#### We need ffmpeg for scikit-video to work
Download this release: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.7z \
Unzip file using: https://www.7-zip.org/ \
Remember where you install ffmpeg, then adding that path to PATH, example: https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/


## Running
### Drawing segmentations
```python load_and_draw.py -c configs/config.ini```
### Tracking
```python tracking/tracking.py -c configs/config_tracking.ini```
### Segmenting
```python segmenting/segmenting.py -c configs/config_segmenting.ini```
