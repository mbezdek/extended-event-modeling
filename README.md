# extended-event-modeling

This repository manages code to:

- Generate object locations for all frames from semi-labels (1 frame is labeled every 10s)
- Compute features (for each frame) useful for modeling (e.g. hand-object distance, velocity of joints, etc.)
- Training and validating SEM (skip the previous two steps and jump here if you use your own features or use preprocessed features from the OSF repository)
- Scripts to generate statistics and figures from SEMs' outputs
- Model diagnostic scripts

Submitted modeling manuscript: https://psyarxiv.com/pt6hx/ \
OSF repository: https://osf.io/39qwz/

## Generate object locations for all frames from semi-labels
You can skip this step if you use preprocessed features from the OSF repository.
### Installation
#### Packages requirements
Install environment to run tracking algorithm \
(a CUDA-enabled device is required to perform object tracking) \
```conda env create -f environment_tracking.yml```

#### Install tracking model (pysot)

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

### Perform automated object tracking on unlabeled video frames, using labeled frames

This script tracks object locations on video frames between the subset of video frames that were manually labeled: \
```python src/tracking/tracking_to_correct_label.py -c configs/config_tracking_to_correct_label.ini --run $run --track_tag $tag 2>&1 | tee "logs/$run$tag.log"```

Running the tracking algorithm takes about 10 hours for a single activity run. It's recommended to use parallel computing to run the tracking algorithm on multiple runs at the same time.

## Compute features useful for modeling
You can skip this step if you use your own features or want to use preprocessed features from the OSF repository.

### Packages requirements
Install environment in Windows/Linux: \
```conda env create -f environment_feature_wd_or_linux.yml``` \
or Mac OSX+: ```conda env create -f environment_feature_osx.yml``` \
Installation was tested on a Mac M1 computer, OS version 12.6, and took about 10 minutes.

### Data preparation

Tracking annotations (object locations) are in ```output/tracking/``` after running the step in the previous section. These annotations are necessary to compute object appearance/disappearance
features, and object-hand features.

Skeleton annotations are in ```data/clean_skeleton_data/```. These annotations are necessary to compute motion features.

Videos are in ```data/small_videos/```. These videos are necessary to compute optical flow features.

Depth information (to calculate object-hand distance) is in `output/depth/`

Make sure to download these from `https://osf.io/39qwz/` to run the following steps

### Compute individual features

Compute how many objects appear/disappear for each frame: \
```python src/individual_features/appear_feature.py -c configs/config_appear.ini```

Compute objects' distance to the hand for each frame: \
```python src/individual_features/object_hand_features.py -c configs/config_objhand_features.ini```

Compute velocity, acceleration, distance to the trunk, inter-hand velocity/acceleration, etc.: \
```python src/individual_features/skel_features.py -c configs/config_skel_features.ini```

Compute optical flow and pixel difference features for each frame: \
```python src/individual_features/optical_features.py -c configs/config_optical_features.ini```

Output features will be saved in ```output/{feature_name}/```

The computation of features should take about 5 to 10 minutes for a single activity video.

Instead of running 4 commands above, you can run (if using slurm):

```sbatch src/individual_features/compute_indv_features.sh```

Check {feature_name}_complete*.txt (e.g. objhand_complete_sep_09.txt) to see which videos (runs) were successfully computed. Check
{feature_name}_error*.txt (e.g. objhand_error_sep_09.txt) for potential errors.

### Preprocess features and create input to SEM

For each run, this script below will:

- Load individual raw features from the "Compute individual features" step and *preprocess* (standardize/smooth for skel, drop
  irrelevant feature columns, extract embeddings based on categories of nearest objects, etc.)
- Resample features (3Hz) and combine them into a dataframe, save the combined dataframe and individual dataframes into a pickle
  file. (the combined dataframe is SEM's input)

```python src/preprocess_features/preprocess_indv_run.py -c configs/config_preprocess.ini```

You can create a text file with a list of runs to preprocess, this can be a list of completed runs for all individual features in
the "Compute individual features" step. Then, run this script to parallel the above script on slurm:

```python src/preprocess_features/parallel_preprocess_indv_run.py -c configs/config_preprocess.ini```

Output will be saved in ```output/preprocessed_features/*.pkl```

Check preprocessed_runs_{}.txt (e.g. preprocessed_runs_sep_09.txt) to see which runs were preprocessed. Check filtered_skel_
{}.txt (e.g. filtered_skel_sep_09_0.8_0.8.txt) to see a list of runs that have corrupted skeleton.

Run the script below to compute one PCA matrix for each feature, and a PCA matrix for all features together.

```python src/preprocess_features/compute_pca_all_runs.py - c configs/config_pca.ini --run clean_skel_sep_09.txt```

`clean_skel_sep_09.txt` is the difference between `preprocessed_complete*.txt` and `filtered_skel*.txt`. To get
`clean_skel_sep_09.txt`, run: \
`python src/preprocess_features/diff_two_list.py {preprocessed_list} {filtered_skel_list} output/clean_skel_sep_09.txt`

Output PCA matrices will be saved in ```output/*_{feature_name}_pca.pkl```

## Training and validating SEM
If you use your own features, or preprocessed features from the OSF repository, you can skip the previous steps and just start here. Please make sure to download preprocessed features from `https://osf.io/39qwz/` for all 149 videos. There are preprocessed features for 4 videos in this repository, which are used for demonstration purposes.

### Packages requirements
Install environment in Windows/Linux: \
```conda env create -f environment_sem_wd_or_linux.yml``` \
or Mac OSX+: ```conda env create -f environment_sem_osx.yml``` \
Installation was tested on a Mac M1 chip, OS version 12.6, and took about 10 minutes. For M2 chip, change ray[default] to 1.8.0 in environment_sem_osx.yml.

### Install SEM from github repository

```git clone git@github.com:NguyenThanhTan/SEM2.git  ``` \
Export the Path to SEM \
```export PYTHONPATH="${PYTHONPATH}:/Users/{USERNAME}/Documents/SEM2"```

## Run SEM

The script below will load preprocessed features, apply PCA transformation, and train SEM.\
```python src/train_eval_inference/run_sem_pretrain.py -c configs/config_run_sem.ini```

The file configs/config_run_sem.ini can be modified to perform SEM on a single run rather than a set of runs (e.g., for demo purposes). \
Simply change the `train` and `valid` parameters, such as:

Replace:

```
train=output/train_sep_09.txt
valid=output/valid_sep_09.txt
```

With:

```
train=2.4.9_kinect
valid=1.1.3_kinect
```

It should take about two minutes for SEM to process a single activity run.

SEM's states (prediction error, hidden activations, schema activations, etc.) will be saved
in `output/run_sem/{tag}/*_diagnostic_{epoch}.pkl`. This information will be useful for visualization and diagnose.\
SEM's input and output vectors will be saved in `output/run_sem/{tag}/*_inputdf_{epoch}.pkl`. This information is useful for
visualization and diagnose.

## Recreate statistics and figures

All scripts to generate statistics and figures are in `generate_statistics_and_figure.ipynb`

## Diagnose

This panel will visualize SEM's metrics (prediction error, mutual information, biserial, #events, etc.) across tags, useful for
model comparison: \
```panel serve matrix_viz.ipynb```

This panel will zoom-in deeper, visualizing SEM's states (posteriors, segmentation, PCA features, etc.) across epoch for each
video: \
`panel serve output_tabs_viz.ipynb`

This panel will compare SEM's event schemas and human action categories:\
`panel serve sankey.ipynb`

To visualize input and output projected into original feature space, run:\
`python draw_video.py {run} {tag} {epoch}` \
or add a list of (run tag epoch) in `runs_to_draw.txt` and run: \
`python draw_multiple_videos.py`


