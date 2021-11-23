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
* [1. Download the Neuaral-Anthropometer (NA) dataset](https://github.com/neoglez/neural-anthropometer#1-download-neural-anthropometer-dataset)
* [2. or Create your own synthetic data](https://github.com/neoglez/neural-anthropometer#2-or-create-your-own-synthetic-data)
* [3. Training models](https://github.com/neoglez/neural-anthropometer#3-training-models)
* [4. Storage info](https://github.com/neoglez/#4-storage-info)
* [Citation](https://github.com/neoglez/neural-anthropometer#citation)
* [License](https://github.com/neoglez/neural-anthropometer#license)
* [Acknowledgements](https://github.com/neoglez/neural-anthropometer#acknowledgements)

## 1. Download Neural-Anthropometer dataset


You can check [Storage info](https://github.com/neoglez/neural-anthropometer#4-storage-info) for how much disk space they require and can do partial download.
Download from our cloud (see bellow).
| Dataset  |  Download Link     | sha256sum      |  Password |
|----------|:-------------:|---------------:|---------------:|
| Neural-Anthropometer (full) |  [NeuralAnthropometer.tar.gz](https://cloudlogin03.world4you.com/index.php/s/5uD3bt1n207k8ko) | 7fe685fa21988a5dfcf567cdc18bee208f99e37ef44bef1d239aa720e219c47e | na-dataset |

The general structure of the folders is as follows:

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

## 2. or Create your own synthetic data
### 2.1. Preparation

Please consider that in all cases, we install dependencies into a conda environment. The code was tested under ubuntu 20.04 with python 3.8.

#### 2.1.1. SMPL data

You need to download SMPL data from http://smpl.is.tue.mpg.de and https://www.di.ens.fr/willow/research/surreal/data/ in order to run the synthetic data generation code. Once you agree on SMPL license terms and have access to downloads, you will have the following three files:

```
basicModel_f_lbs_10_207_0_v1.0.0.pkl
basicmodel_m_lbs_10_207_0_v1.0.0.pkl
smpl_data.npz
```

Place these three files under `datageneration/data` folder.


``` shell

smpl_data/
--------- smpl_data.npz # 2.5GB
 # trans*           [T x 3]     - (T: number of frames in MoCap sequence)
 # pose*            [T x 72]    - SMPL pose parameters (T: number of frames in MoCap sequence)
 # maleshapes       [1700 x 10] - SMPL shape parameters for 1700 male scans
 # femaleshapes     [2103 x 10] - SMPL shape parameters for 2103 female scans 
 # regression_verts [232]
 # joint_regressor  [24 x 232]
```

#### 2.1.2. Neural Anthropometer
``` shell

git clone http://github.com/neoglez/neural-anthropometer.git
cd neural-anthropometer
pip install .
```

#### 2.1.3. Mesh synthesis

To synthesize the meshes, open and run `train_Neural-Anthropometer_cross_validation.py` in your preferred IDE.

#### 2.1.4. Synthetic images with Blender

Building Blender is a painful process. That is why we recommend to download and install the version that we used. The provided code was tested with [Blender2.78](http://download.blender.org/release/Blender2.78/blender-2.78a-linux-glibc211-x86_64.tar.bz2).

Just open the Scripting view and load (or copy and paste) the script `synthesize_cmu_200x200_grayscale_images.py`
Change the path correspondingly at `cmu_dataset_path = os.path.abspath("/home/youruser/YourCode/calvis/CALVIS/dataset/cmu/")` and run the script.
The process takes several minutes.

#### 2.1.5. Vedo and Trimesh

You need to install these two libraries (if you do not have them):

``` shell

conda install -c conda-forge vedo
pip install trimesh
```

### 2.2. Annotating with Sharmeam (SHoulder width, ARM length and insEAM) and Calvis

#### 2.1.2. Calculating shoulder width, right and left arm length and inseam.
Run the script `.py`
The process takes several hours.

#### 2.1.3. Visualize shoulder width, right and left arm length and inseam.
To visualize at which points Sharmeam is calculating the body measurements, follow the code in `display_one_by_one_8_subjects_Sharmeam_with_vedo_and_trimesh.py` or directly display it with colab `display_one_by_one_8_subjects_Sharmeam_with_vedo_and_trimesh.ipynb`

Note: To display the meshes in the browser, we use k3d backend. Install it with

``` shell

conda install -c conda-forge k3d
```

## 3. Training and evaluating The Neural Anthropometer

At this point you should have the input (synthetic images) and the supervision signal (human body dimensions annotations). Here, we provide code to train and evaluate The Neural Anthropometer on the synthetic data to predict given the input eight human body dimensions: shoulder width, right and left arm length, chest, waist and pelvis circumference and height.

### 3.1. Preparation

#### 3.1.1. Requirements
* Install [pytorch](https://pytorch.org/) with [CUDA](https://developer.nvidia.com/cuda-downloads) support.
* Download [The Neural Anthropometer](https://github.com/neoglez/neural-anthropometer)
* Install scikit-learn, SciPy and its image processing routines

``` shell

conda install scikit-learn 
conda install -c anaconda scipy
conda install -c anaconda scikit-image
```

### 3.2. Training

*Tested on Linux (Ubuntu 20.04) with cuda 10.2 on a GeForce GTX 1060 6GB graphic card*
To train and evaluate The Neural Anthropometer, open and run `train_Neural-Anthropometer_cross_validation.py` in your preferred IDE.

### 3.2. Inference

To perform inference with The Neural Anthropometer, open and run `train_Neural-Anthropometer_cross_validation.py` in your preferred IDE.


## 4. Storage info

| Dataset     | 12000 Meshes | 12000 (200x200x1) Synthetic images | Annotations | Total |
| -----------:|---------:|------------:|----------------------------------:|------------:|------:|
| Neural Anthropometer      | 4.9 GB    | 160.6 MB      |   4.4 MB   | 5 GB |

## Citation
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

## License
Please check the [license terms](https://github.com/neoglez/neural-anthropometer/blob/master/LICENSE.md) before downloading and/or using the code, the models and the data.

## Acknowledgements
The [SMPL team](https://smpl.is.tue.mpg.de/) for providing us with the learned human body templates and the SMPL code.


The [vedo team](https://github.com/marcomusy/vedo) (specially Marco Musy) and the [trimesh team](https://github.com/mikedh/trimesh) (specially Michael Dawson-Haggerty) for the great visualization and intersection libraries.
