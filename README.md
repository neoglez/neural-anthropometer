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
* [6. Citation](https://github.com/neoglez/neural-anthropometer#6-citation)
* [7. License](https://github.com/neoglez/neural-anthropometer#7-license)
* [8. Acknowledgements](https://github.com/neoglez/neural-anthropometer#8-acknowledgements)

## 1. Clone and install Neural-Anthropometer

Be aware that we did not list all dependencies in setup.py. Therefore, you will have to install the libraries depending on the functionality you want to obtain.

### 1.1. HBM, Vedo and Trimesh

You need to install these three libraries:

``` shell

git clone http://github.com/neoglez/hbm.git
cd hbm
pip install .

conda install -c conda-forge vedo

pip install trimesh
pip install sklearn
pip install matplotlib
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
mv dataset  /your-path/neural-anthropometer/

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

* Install [pytorch](https://pytorch.org/) with [CUDA](https://developer.nvidia.com/cuda-downloads) support.

You need to install these libraries:

``` shell

pip install chumpy
```

#### 3.1.1. SMPL data

You need to download SMPL data from http://smpl.is.tue.mpg.de and https://www.di.ens.fr/willow/research/surreal/data/ in order to run the synthetic data generation code. Once you agree on SMPL license terms and have access to downloads, you will have the following three files:

```
basicModel_f_lbs_10_207_0_v1.0.0.pkl
basicmodel_m_lbs_10_207_0_v1.0.0.pkl
smpl_data.npz
```

Place the basic models (two files) under `neural-antropometer/datageneration/data` folder.

``` shell
mkdir datageneration/data
cp your/path/models/*.pkl datageneration/data

```
The folder structure should be as follows.

```

datageneration/data/
---------------- basicModel_f_lbs_10_207_0_v1.0.0.pkl
---------------- basicmodel_m_lbs_10_207_0_v1.0.0.pkl
```

#### 3.1.2. Mesh synthesis

To synthesize the meshes, open and run `generate_6000_meshes_with_smpl_total_random.py` in your preferred IDE (we use Spyder).

#### 3.1.3. Synthetic images with Blender

Building Blender is a painful process. We used Blender 2.91.0 Alpha. Wheel mercifully provided by 
https://github.com/TylerGubala/blenderpy/releases/tag/v2.91a0

Follow the instructions given at
https://github.com/TylerGubala/blenderpy/releases/tag/v2.91a0

Open and run `synthesize_na_200x200_grayscale_images.py` in your preferred IDE (we use Spyder).

The process takes several minutes.

### 3.2. Annotating with Sharmeam (SHoulder width, ARM length and insEAM) and Calvis

You need to install Calvis:

``` shell

git clone https://github.com/neoglez/calvis
cd calvis
pip install .
```

#### 3.2.1. Calculating eight Human Body Dimensions (HBDs): shoulder width, right and left arm length, inseam; chest, waist and pelvis circumference, and height.

Open and run `annotate_with_Sharmeam_and_Calvis.py` in your preferred IDE (we use Spyder).
The process takes several hours.

#### 3.2.2. Optional: visualize the eight Human Body Dimensions (HBDs): shoulder width, right and left arm length, inseam; chest, waist and pelvis circumference, and height.

To visualize at which points Sharmeam and Calvis are calculating the HBDs, open and run `neural-antropometer/display/display_subject_sharmean_and_calvis_with_vedo_and_trimesh.py` or directly display it with colab.

Note: To display the meshes in the browser, we use k3d backend. Install it with

``` shell

conda install -c conda-forge k3d
```

## 4. Training and evaluating The Neural Anthropometer

At this point you should have the input (synthetic images) and the supervision signal (human body dimensions annotations). Here, we provide code to train and evaluate The Neural Anthropometer on the synthetic data to predict given the input eight human body dimensions: shoulder width, right and left arm length, chest, waist and pelvis circumference and height.

Both training and inference can be directly displayed in colab.

### 4.1. Preparation

#### 4.1.1. Requirements
* Install [pytorch](https://pytorch.org/) with [CUDA](https://developer.nvidia.com/cuda-downloads) support.
* Install scikit-learn, SciPy and its image processing routines

``` shell

conda install scikit-learn 
conda install -c anaconda scipy
conda install -c anaconda scikit-image
```

### 4.2. Training

*Tested on Linux (Ubuntu 20.04) with cuda 10.2 on a GeForce GTX 1060 6GB graphic card*

To train and evaluate The Neural Anthropometer, open and run `experiments/experiment_1_input_all_test_all_save_results.py` in your preferred IDE.

### 4.3. Inference

To perform inference with The Neural Anthropometer, open and run `experiments/load_and_make_inference_na_and_make_grid.py` in your preferred IDE.


## 5. Storage info

| Dataset  | `.tar.gz` file   | 12000 Meshes | 12000 (200x200x1) Synthetic images | Annotations | Total |
| --------:|-------------:|-----------------------------------:|------------:|------:|------------|
| Neural Anthropometer | 1.9 GB  | 4.9 GB  | 160.6 MB   |   4.4 MB   | ~5 GB |

## 6. Citation
If you use this code, please cite the following:

```
@misc{ygtham2021na,
	title={A Neural Anthropometer Learning from Body Dimensions Computed on Human 3D Meshes.},
	author={Gonzalez Tejeda, Yansel and Mayer, Helmut A.},
	year={2021},
	eprint={2110.04064},
	archivePrefix={arXiv},
	primaryClass={cs.CV},
	note={To appear in the Proceedings of the IEEE Symposium Series on Computational Intelligence (IEEE SSCI 2021)},
}
```

## 7. License
Please check the [license terms](https://github.com/neoglez/neural-anthropometer/blob/master/LICENSE.md) before downloading and/or using the code, the models and the data.

## 8. Acknowledgements
The [SMPL team](https://smpl.is.tue.mpg.de/) for providing us with the learned human body templates and the SMPL code.


The [vedo team](https://github.com/marcomusy/vedo) (specially Marco Musy) and the [trimesh team](https://github.com/mikedh/trimesh) (specially Michael Dawson-Haggerty) for the great visualization and intersection libraries.
