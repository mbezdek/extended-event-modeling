# extended-event-modeling
This repository manages code for:
- object tracking
- skeleton projecting (mapping from kinect camera to C1 or C2)
- feature generating (e.g. hand-object distance)
- SEM: training and inference (segmenting)
- Plotting scripts

## Installation

### Install anaconda
Follow instructions to install conda: https://www.anaconda.com/products/individual
### Install packages
Note: Because of an attempt to use one environment file that can run on Windows, OSX, Linux, 
this version of `environment.yml` is constructed to be generic (e.g. tensorflow=2 
instead of tensorflow=2.1.0 because conda for OSX only has tensorflow 2.0.0 version) \
```conda env create -f environment.yml```\
```conda activate sem-pysot-37```

### Install pysot for tracking
```git clone https://github.com/STVIR/pysot``` \
```cd pysot``` \
```python setup.py build_ext --inplace```

For Windows users, you might need Microsoft Visual Studio 2017 to be able to build: \
```python setup.py build_ext --inplace``` \
After installing MVS 2017, open Cross Tools Command Prompt VS 2017 and run: \
```set DISTUTILS_USE_SDK=1``` \
```cd /path/to/pysot``` \
```python setup.py build_ext --inplace```

### Install SEM from github repository
Using `python -m pip` to avoid confusing if the system has multiple Python versions. \
Be aware of changes because this command will pull the latest version of SEM \
```python -m pip install git+https://github.com/nicktfranklin/SEM2``` \

### For GPU running (local machine has a GPU card or running on cloud)
```conda install -c anaconda tensorflow-gpu=2``` \
```conda install -c pytorch cuda92``` 

### Install interactive extensions for Jupyter notebook/lab (optional)
Recommended for experiments and visualization \
```jupyter labextension install @jupyter-widgets/jupyterlab-manager``` \
```jupyter nbextension enable --py widgetsnbextension```

### Install ffmpeg (optional)
#### We need ffmpeg for scikit-video to work
Download this release: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.7z \
Unzip file using: https://www.7-zip.org/ \
Remember where you install ffmpeg, then adding that path to PATH, example: https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/

## Running
### Drawing segmentations
```python load_and_draw.py -c configs/config_draw_segmentation.ini```
### Tracking
```python tracking/tracking.py -c configs/config_tracking.ini``` \
#### Trouble shooting
If there is a runtime error: \
`RuntimeError (RuntimeError: CuDNN error: CUDNN_STATUS_SUCCESS)` \
Try remove (instead of uninstall) and then re-install \
```conda remove cudnn cudatoolkit pytorch torchvision && conda install cuda92 pytorch=0.4.1 -c pytorch```
### Segmenting
```python segmenting/segmenting.py -c configs/config_segmenting.ini```
#### Trouble shooting
If there is a runtime error: \
`AttributeError: Can't pickle local object 'processify.<locals>.process_func'` \
Go to installation of SEM, e.g.: \
`C:\Users\nguye\AppData\Roaming\Python\Python37\site-packages\sem\sem.py` \
Search `def sem_run` and command out processify decorator:
```
# @processify
def sem_run(x, sem_init_kwargs=None, run_kwargs=None):
```
