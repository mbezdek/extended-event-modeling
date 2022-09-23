# extended-event-modeling
This repository manages code for:
- Object tracking
- Feature generation (e.g. hand-object distance, velocity of joints, etc.)
- Training and validating SEM
- Diagnostic scripts

Submitted manuscript: https://psyarxiv.com/pt6hx/ \
OSF repository: https://osf.io/39qwz/

## Installation

### Install anaconda
Follow instructions to install conda: https://www.anaconda.com/products/individual
### Install packages
Install environment to run tracking algorithm \
```conda env create -f environment_tracking.yml```\
Install environment to run SEM \
```conda env create -f environment_sem_Windows.yml```\
or ```conda env create -f environment_sem_MacOS.yml```

### Install pysot for tracking
```cd ..```\
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
```git clone git@github.com:NguyenThanhTan/SEM2.git  ``` \
Export the Path to SEM \
```export PYTHONPATH="${PYTHONPATH}:/Users/{USERNAME}/Documents/SEM2"```




### For GPU running (local machine has a GPU card or running on cloud)
```conda install -c anaconda tensorflow-gpu=2``` \
```conda install -c pytorch cuda92``` 

### Install interactive extensions for Jupyter notebook/lab (optional)
Recommended for experiments and visualization \
```jupyter labextension install @jupyter-widgets/jupyterlab-manager``` \
```jupyter nbextension enable --py widgetsnbextension```

### Install ffmpeg (optional)
Download this release: https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.7z \
Unzip file using: https://www.7-zip.org/ \
Remember after you install ffmpeg, add that path to PATH, example: https://www.architectryan.com/2018/03/17/add-to-the-path-on-windows-10/

## Data preparation

From semi-annotated object locations (one frame every ten seconds), we use Siamese to track forward and backward in time, 
then use an algorithm to merge these tracks. Script to run tracking algorithm:

```python tracking/tracking.py -c configs/config_tracking.ini```

Tracking annotations are in ```output/tracking_kinect/```. These annotations are necessary to compute object
appearance/disappearance features, and object-hand features.

Skeleton annotations are in ```data/clean_skeleton_data/```. These annotations are necessary to compute motion features.

Videos are in ```data/small_videos/```. These videos are necessary to compute optical flow features.

### Compute individual features

Compute how many objects appear/disappear for each frame: \
```python individual_features/appear_feature.py -c configs/config_appear.ini```

Compute objects' distance to the hand for each frame: \
```python individual_features/object_hand_features.py -c configs/config_objhand_features.ini```

Compute velocity, acceleration, distance to the trunk, inter-hand velocity/acceleration, etc.: \
```python individual_features/skelfeatures.py -c configs/config_skelfeatures.ini```

Compute optical flow and pixel difference features for each frame: \
```python individual_features/vidfeatures.py -c configs/config_vidfeatures.ini```

Output features will be saved in ```output/{feature_name}/```

Instead of running 4 commands above, you can run (if using slurm):

```sbatch compute_indv_features.sh```

Check {feature_name}_complete*.txt (e.g. objhand_complete_sep_09.txt) to see which videos (runs) were successfully computed.
Check {feature_name}_error*.txt (e.g. objhand_error_sep_09.txt) for errors.

### Preprocess features and create input to SEM

For each run, this script below will:

- Load individual raw features from the "Compute individual features" step and *preprocess* (standardize/smooth for skel, 
drop irrelevant feature columns, extract embeddings based on categories of nearest objects, etc.)
- Resample features (3Hz) and combine them into a dataframe, save the combined dataframe and individual dataframes into a pickle
  file. (the combined dataframe is SEM's input)

```python preprocess_indv_run.py -c configs/config_preprocess.ini```

You can create a text file with a list of runs to preprocess, this can be a list of completed runs for all individual features in
the "Compute individual features" step. Then, run this script to parallel the above script on slurm:

```python parallel_preprocess_indv_run.py -c configs/config_preprocess.ini``` 

Output will be saved in ```output/preprocessed_features/*.pkl```

Check preprocessed_runs_{}.txt (e.g. preprocessed_runs_sep_09.txt) to see which runs were preprocessed. 
Check filtered_skel_{}.txt (e.g. filtered_skel_sep_09_0.8_0.8.txt) to see a list of runs that have corrupted skeleton.

Run the script below to compute one PCA matrix for each feature, and a PCA matrix for all features together.

```python compute_pca_all_runs.py - c configs/config_pca.ini --run clean_skel_sep_09.txt```

`clean_skel_sep_09.txt` is the difference between `preprocessed_complete*.txt` and `filtered_skel*.txt`. To get
`clean_skel_sep_09.txt`, run: \
`python diff_two_list.py {preprocessed_list} {filtered_skel_list} clean_skel_sep_09.txt`

Output PCA matrices will be saved in ```output/*.pkl```

## Run SEM

The script below will load preprocessed features, apply PCA transformation, and train SEM.\
```python run_sem_pretrain.py -c configs/config_run_sem.ini```

SEM's states (prediction error, hidden activations, schema activations, etc.) will be saved in `output/run_sem/{tag}/*_diagnostic_{epoch}.pkl`. 
This information will be useful for visualization and diagnose.\
SEM's input and output vectors will be saved in `output/run_sem/{tag}/*_inputdf_{epoch}.pkl`. This information is useful for 
visualization and diagnose.

## Recreate statistics and figures

All scripts to generate statistics and figures are in `generate_statistics_and_figure.ipynb`

## Diagnose

This panel will visualize SEM's metrics (prediction error, mutual information, biserial, #events, etc.) across tags, 
useful for model comparison: \
```panel serve matrix_viz.ipynb```

This panel will zoom-in deeper, visualizing SEM's states (posteriors, segmentation, PCA features, etc.) across epoch 
for each video: \
`panel serve output_tabs_viz.ipynb`

This panel will compare SEM's event schemas and human action categories:\
`panel serve sankey.ipynb`

To visualize input and output projected into original feature space, run:\
`python draw_video.py {run} {tag} {epoch}` \
or add a list of (run tag epoch) in `runs_to_draw.txt` and run: \
`python draw_multiple_videos.py`


