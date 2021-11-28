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
Install environment to run tracking algorithm \
```conda env create -f environment_tracking.yml```\
Install environment to run SEM \
```conda env create -f environment_sem_Windows.yml```\
or ```conda env create -f environment_sem_MacOS.yml```\

### Install pysot for tracking
```cd ..```
```git clone https://github.com/STVIR/pysot``` \
```cd pysot``` \
```python setup.py build_ext --inplace```\
```export PYTHONPATH=/absolute/path/to/pysot```

For Windows users, you might need Microsoft Visual Studio 2017 to be able to build: \
```python setup.py build_ext --inplace``` \
After installing MVS 2017, open Cross Tools Command Prompt VS 2017 and run: \
```set DISTUTILS_USE_SDK=1``` \
```cd /path/to/pysot``` \
```python setup.py build_ext --inplace```


### Install SEM from github repository 
Be aware of changes because this command will pull the latest version of SEM \
```git clone git@github.com:NguyenThanhTan/SEM2.git  ``` \
Export the Path to SEM \
```export PYTHONPATH="${PYTHONPATH}:/Users/{USERNAME}/Documents/SEM2"```\




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

## Running Scripts
### Drawing segmentations
```python load_and_draw.py -c configs/config_draw_segmentation.ini```
### Tracking
```python tracking/tracking.py -c configs/config_tracking.ini``` \
#### Trouble shooting
If there is a runtime error: \
`RuntimeError (RuntimeError: CuDNN error: CUDNN_STATUS_SUCCESS)` \
Try remove (instead of uninstall) and then re-install \
```conda remove cudnn cudatoolkit pytorch torchvision && conda install cuda92 pytorch=0.4.1 -c pytorch```
### Train SEM and Evaluate
```python run_sem_pretrain.py -c configs/config_run_sem.ini```
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
## Visualize Results
We use panel to develop some visualization dashboards useful to understand SEM's results
### Input Features
```panel serve input_viz_refactored.ipynb --port 6300 --allow-websocket-origin=localhost:6300 --allow-websocket-origin=127.0.0.1:6300```
### Output Diagnostic Results
```panel serve output_tabs_viz.ipynb --allow-websocket-origin=localhost:6100 --port 6100 --allow-websocket-origin=127.0.0.1:6100```
### Overall Metrics
```panel serve matrix_viz.ipynb --port 6200 --allow-websocket-origin=localhost:6200 --allow-websocket-origin=127.0.0.1:6200```
