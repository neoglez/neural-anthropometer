[![Training](https://img.shields.io/badge/-Training-green.svg)](https://colab.research.google.com/github/neoglez/neural-anthropometer/blob/main/notebook/train_neural_antropomter.ipynb) [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/neoglez/neural-anthropometer/blob/main/notebook/train_neural_antropomter.ipynb)

# neural-anthropometer
A Neural Anthropometer Learning from Body Dimensions Computed on Human 3D Meshes.
Accepted to the [IEEE Symposium Series on Computational Intelligence (IEEE SSCI 2021)](https://attend.ieee.org/ssci-2021/)

[Yansel Gonzalez Tejeda](https://github.com/neoglez) and [Helmut A. Mayer](https://www.cosy.sbg.ac.at/~helmut/helmut.html)

[[Project page - TBD]](http://example.com)

[![arXiv](https://img.shields.io/badge/arXiv-2110.04064-green)](https://arxiv.org/abs/2110.04064)

<p style="display: flex; flex-direction: column;">
<img src="/img/InferenceResults.jpg">
<img src="/img/NeuralAnthropometerApproachOverview.jpg">
<img src="/img/ShoulderWidthMaleSubject.jpg">
<img src="/img/FemaleSubjectArmLength.jpg">
</p>

## Contents
* [1. Clone and install Neural-Anthropometer](https://github.com/neoglez/neural-anthropometer#1-clone-and-install-neural-anthropometer)
* [2. Download Neural-Anthropometer dataset](https://github.com/neoglez/neural-anthropometer#2-download-neural-anthropometer-dataset)
* [3. Or create your own synthetic data](https://github.com/neoglez/neural-anthropometer#3-or-create-your-own-synthetic-data)
* [4. Training and evaluating The Neural Anthropometer](https://github.com/neoglez/neural-anthropometer#4-training-and-evaluating-the-neural-anthropometer)
* [5. Storage info](https://github.com/neoglez/neural-anthropometer#5-storage-info)
* [6. Uninstalling](https://github.com/neoglez/neural-anthropometer#6-Uninstalling)
* [7. Citation](https://github.com/neoglez/neural-anthropometer#7-citation)
* [8. License](https://github.com/neoglez/neural-anthropometer#8-license)
* [9. Acknowledgements](https://github.com/neoglez/neural-anthropometer#9-acknowledgements)

## 1. Clone and install Neural-Anthropometer

Be aware that we did not list all dependencies in setup.py. Therefore, you will have to install the libraries depending on the functionality you want to obtain.

### 1.1. HBM, Vedo and Trimesh

You need to install these three libraries:

``` shell

git clone http://github.com/neoglez/hbm.git
cd hbm
pip install .

conda install -c conda-forge vedo

pip install trimesh sklearn matplotlib
```

### 1.2. Install the Neural-Anthropometer

``` shell

git clone http://github.com/neoglez/neural-anthropometer.git
cd neural-anthropometer
pip install .
```

## 2. Download Neural-Anthropometer dataset


Download from our cloud (see bellow).

| Dataset  |  Download Link     | sha256sum      |  Password |
|----------|:-------------:|---------------:|---------------:|
| Neural-Anthropometer (full) |  [NeuralAnthropometer.tar.gz](https://cloudlogin03.world4you.com/index.php/s/5uD3bt1n207k8ko) | 7fe685fa21988a5dfcf567cdc18bee208f99e37ef44bef1d239aa720e219c47e | na-dataset |

Unpack the dataset and place it directly under the folder `neural-anthropometer`.

``` shell
tar -xf neural-anthropometer.tar.gz
mv dataset/*  ~/your-path/neural-anthropometer/dataset/

```

The general structure of the folders must be:

``` shell
neural-anthropometer/dataset/
---------------------  annotations/ # json annotations with calvis
----------------------------------  female/
----------------------------------  male/
---------------------  human_body_meshes/ # generated meshes
---------------------------------------- pose0/ # meshes in pose 0
-------------------------------------------- female/
-------------------------------------------- male/
---------------------------------------- pose1/ # meshes in pose 1
-------------------------------------------- female/
-------------------------------------------- male/
---------------------  synthetic_images/ # synthetic greyscale images (200x200x1)
---------------------------------------- 200x200/
---------------------------------------------- pose0/
------------------------------------------------ female/
------------------------------------------------ male/
---------------------------------------------- pose1/
------------------------------------------------ female/
------------------------------------------------ male/


```

## 3. Or create your own synthetic data

Be aware that we did not list the dependencies in setup.py. Therefore, you will have to install the libraries depending on the functionality you want to obtain.

### 3.1. Preparation

Please consider that in all cases, we install dependencies into a conda environment. The code was tested under ubuntu 20.04 with python 3.8.

* Install pytorch. We recommend using CUDA. CPU will run as well but it will take much longer.

You need to install these libraries:

``` shell

pip install chumpy
```

#### 3.1.1. SMPL data

You need to download SMPL data from http://smpl.is.tue.mpg.de in order to run the synthetic data generation code. Once you register and agree on SMPL license terms, you will have access to downloads. Note that there are several downloads. [We use SMPL version 1.0.0](https://download.is.tue.mpg.de/download.php?domain=smpl&sfile=SMPL_python_v.1.0.0.zip) (python version is not relevant).

Download and unpack.

```

unzip  SMPL_python_v.1.0.0.zip
```

You will have the following files:

```
basicModel_f_lbs_10_207_0_v1.0.0.pkl
basicmodel_m_lbs_10_207_0_v1.0.0.pkl
```

Place the basic models (two files) under `neural-antropometer/datageneration/data` folder.

``` shell
mv smpl/models/*.pkl ~/your-path/neural-anthropometer/datageneration/data/

```

The folder structure should be as follows.

```

neural-anthropometer/datageneration/data/
------------------------------------- basicModel_f_lbs_10_207_0_v1.0.0.pkl
------------------------------------- basicmodel_m_lbs_10_207_0_v1.0.0.pkl
```

#### 3.1.2. Mesh synthesis

To synthesize the meshes, open and run `generate/generate_6000_meshes_with_smpl_total_random.py` in your preferred IDE (we use VSCode/Spyder).

#### 3.1.3. Synthetic images with Blender

Building Blender is a painful process. We used Blender 2.91.0 Alpha **with python 3.7**. Wheel mercifully provided by 
https://github.com/TylerGubala/blenderpy/releases/tag/v2.91a0

**if you are working already under python 3.7, skip this step!**

create conda environment with python 3.7, install dependencies (pytorch) and The Neural Anthropometer as described in 1.

```
conda deactivate
conda create -n napy37 python=3.7
conda activate napy37
pip install mathutils trimesh scipy matplotlib
conda install -c conda-forge vedo
cd /your-path/hbm
pip install .
cd /your-path/neural-anthropometer
pip install .
```

To install bpy, follow the instructions given at
https://github.com/TylerGubala/blenderpy/releases/tag/v2.91a0


Open and run `synthesize/synthesize_na_200x200_grayscale_images.py` in your preferred IDE (we use VSCode/Spyder).

The process takes several minutes.

**if you are working under python 3.7, skip this step!**

Return to default environment:

```
conda deactivate
conda activate your-na-default
```

### 3.2. Annotating with Sharmeam (SHoulder width, ARM length and insEAM) and Calvis

You need to install shapely, rtree and Calvis:

``` shell
pip install shapely rtree
git clone https://github.com/neoglez/calvis
cd calvis
pip install .
```

#### 3.2.1. Calculating eight Human Body Dimensions (HBDs): shoulder width, right and left arm length, inseam; chest, waist and pelvis circumference, and height.

Open and run `annotate/annotate_with_Sharmeam_and_Calvis.py` in your preferred IDE (we use VSCode/Spyder).
The process takes several hours.

#### 3.2.2. Optional: visualize the eight Human Body Dimensions (HBDs): shoulder width, right and left arm length, inseam; chest, waist and pelvis circumference, and height.

To visualize at which points Sharmeam and Calvis are calculating the HBDs, open and run `neural-antropometer/display/display_subject_sharmean_and_calvis_with_vedo_and_trimesh.py` or directly display it with colab.

Note: To display the meshes in the browser, we use k3d backend. Install it with

``` shell

conda install -c conda-forge k3d
```

## 4. Training and evaluating The Neural Anthropometer

At this point you should have the input (synthetic images) and the supervision signal (human body dimensions annotations). Here, we provide code to train and evaluate The Neural Anthropometer on the synthetic data to predict given the input eight human body dimensions: shoulder width, right and left arm length, chest, waist and pelvis circumference and height.

Both training and inference can be directly displayed in colab (work in progress).

### 4.1. Preparation

#### 4.1.1. Requirements
* Install [pytorch](https://pytorch.org/). We recommend using [CUDA](https://developer.nvidia.com/cuda-downloads). CPU will run as well but it will take much longer.
* Install scikit-learn, SciPy and its image processing routines

``` shell

pip install scikit-learn
pip install anaconda scipy
pip install scikit-image
```

### 4.2. Training

*Tested on Linux (Ubuntu 20.04) with cuda 10.2 on a GeForce GTX 1060 6GB graphic card*

To train and evaluate The Neural Anthropometer, open and run `experiments/experiment_1_input_all_test_all_save_results.py` in your preferred IDE.

### 4.3. Inference

To perform inference with The Neural Anthropometer, open and run `experiments/load_and_make_inference_na_and_make_grid.py` in your preferred IDE.

By running the above script, a 4-instance minibatch will be displayed. We generate the figure with matplotlib and latex.
In the generated figure the instances (synthetic pictures) and inference results (tables with HBDs) are overlapped, making the figure look messy and broken. Just maximize the window and the figure will be displayed correctly.

Important: as of Matplotlib 3.2.1, you also need the package cm-super (see https://github.com/matplotlib/matplotlib/issues/16911).

On linux, install it with:

```
sudo apt install cm-super
```

Abbreviations used in the figure:
| Abbreviation | Human Body Dimension (HBD) |
|---|---|
| CC | Chest Circumference |
| H | Height |
| I | Inseam |
| LAL | Left Arm Length |
| PC | Pelvis Circumference  |
| RAL | Right Arm Length |
| SW | Shoulder Width |
| WC | Waist Circumference |


## 5. Storage info

| Dataset  | `.tar.gz` file   | 12000 Meshes | 12000 (200x200x1) Synthetic images | Annotations | Total |
| --------:|-------------:|-----------------------------------:|------------:|------:|------------|
| Neural Anthropometer | 1.9 GB  | 4.9 GB  | 160.6 MB   |   4.4 MB   | ~5 GB |

## 6. Uninstalling

```
pip uninstall neural_anthropometer
```

## 7. Citation
If you use this code, please cite the following:

```
@INPROCEEDINGS{9660069,
  author={Gonzalez Tejeda, Yansel and Mayer, Helmut A.},
  booktitle={2021 IEEE Symposium Series on Computational Intelligence (SSCI)}, 
  title={A Neural Anthropometer Learning from Body Dimensions Computed on Human 3D Meshes}, 
  year={2021},
  volume={},
  number={},
  pages={1-8},
  doi={10.1109/SSCI50451.2021.9660069}}
```

## 8. License
Please check the [license terms](https://github.com/neoglez/neural-anthropometer/blob/master/LICENSE.md) before downloading and/or using the code, the models and the data.

## 9. Acknowledgements
The [SMPL team](https://smpl.is.tue.mpg.de/) for providing us with the learned human body templates and the SMPL code.


The [vedo team](https://github.com/marcomusy/vedo) (specially Marco Musy) and the [trimesh team](https://github.com/mikedh/trimesh) (specially Michael Dawson-Haggerty) for the great visualization and intersection libraries.
