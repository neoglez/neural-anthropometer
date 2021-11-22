#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: neoglez
"""
from smpl import SMPL

import numpy as np
import os
import torch
from utils import pose2
import math
import json
from datetime import datetime

dataset_path = os.path.abspath("../dataset")
dataset_meshes_path = os.path.join(dataset_path, "human_body_meshes/")
dataset_meshes_path_pose0 = os.path.join(dataset_meshes_path, "pose0/")
dataset_meshes_path_pose1 = os.path.join(dataset_meshes_path, "pose1/")
female_meshes_path_pose0 = os.path.join(dataset_meshes_path_pose0, "female/")
male_meshes_path_pose0 = os.path.join(dataset_meshes_path_pose0, "male/")
female_meshes_path_pose1 = os.path.join(dataset_meshes_path_pose1, "female/")
male_meshes_path_pose1 = os.path.join(dataset_meshes_path_pose1, "male/")
dataset_meshes_path_length = len(dataset_meshes_path)
dataset_annotation_path = os.path.join(dataset_path, "annotations/")
json_log_dir = os.path.join(dataset_path, "log/")
json_log_path = os.path.join(json_log_dir, "synthezing.json")

for d in [dataset_meshes_path,
          dataset_meshes_path_pose0,
          female_meshes_path_pose0,
          male_meshes_path_pose0,
          dataset_meshes_path_pose1,
          female_meshes_path_pose1,
          male_meshes_path_pose1,
          dataset_annotation_path,
          json_log_dir]:
    os.makedirs(d, exist_ok=True)

SMPL_basicModel_f_lbs_path = ("/media/neoglez/Data2/privat/PhD_Uni_Salzburg"
              "/DATASETS/smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl")
SMPL_basicModel_m_lbs_path = ("/media/neoglez/Data2/privat/PhD_Uni_Salzburg"
              "/DATASETS/smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
model_files = {
    'female': SMPL_basicModel_f_lbs_path,
    'male': SMPL_basicModel_m_lbs_path
  }
num_betas = 10
batch_size = 1

num_male_meshes = 3000
num_female_meshes = 3000

low = -3
hig = 3
betas = np.random.uniform(low, hig,(
     batch_size, num_betas, num_female_meshes + num_female_meshes))

# Pose 2: Usually people are required to totally
# lower their arms. However, we observe that if we impose this in
# general for all 3D scans, and due to the fact that we are working
# with LBS (which produce artifacts as we know) inter-penetrations
# occur at the pelvis level with the hands. Therefore, we lower the
# arms 'only' 45 degrees; this setting does not influence the upper
# torso volume and so it does not have a significant impact in the
# measurement/calculation while avoiding inter-penetrations.
zero_pose = torch.FloatTensor(np.zeros((1,72)))
pose1 = torch.FloatTensor(pose2(batch_size=batch_size))

already_synthesized_female_meshes = 0
already_synthesized_male_meshes = 0

padding_f = int(math.log10(num_female_meshes)) + 1
padding_m = int(math.log10(num_male_meshes)) + 1
padding = None

str_log_start = "Started synthesizing meshes at {:%d %B %Y %H:%M:%S}".format(
    datetime.now()
)
print(str_log_start)

# get faces only once
faces = None

for i,_ in enumerate(range(betas.shape[2])):
    gender = np.random.choice(["female", "male"])
    gender = (
        "male"
        if already_synthesized_female_meshes == num_female_meshes
        else gender
    )

    # init SMPL
    smpl = SMPL(model_files[gender])
    # synthesize
    subject_betas = torch.FloatTensor(betas[:, :, i])
    # if you want to synthesize the mesh in the zero pose
    zero_posemodel = smpl.forward(zero_pose, subject_betas)
    posed_model = smpl.forward(pose1, subject_betas)
    if faces is None:
        faces = smpl.faces.numpy()
    
    zero_posemodel = zero_posemodel[-1, :, :].cpu().numpy()
    posed = posed_model[-1, :, :].cpu().numpy()
    
    if gender == "female":
        padding = padding_f
        already_synthesized_female_meshes += 1
        already_synthesized = already_synthesized_female_meshes
    else:
        padding = padding_m
        already_synthesized_male_meshes += 1
        already_synthesized = already_synthesized_male_meshes

    # the name/path for this mesh
    outmesh_path = (
        dataset_meshes_path_pose0
        + gender
        + "/"
        + "subject_mesh_%0.*d.obj" % (padding, already_synthesized)
    )
    ## Write to an .obj file
    with open(outmesh_path, "w") as fp:
        for v in zero_posemodel:
            fp.write("v %f %f %f\n" % (v[0], v[1], v[2]))

        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp.write("f %d %d %d\n" % (f[0], f[1], f[2]))
    
    outmesh_path1 = (
        dataset_meshes_path_pose1
        + gender
        + "/"
        + "subject_mesh_%0.*d.obj" % (padding, already_synthesized)
    )
        ## Write to an .obj file
    with open(outmesh_path1, "w") as fp1:
        for v in posed:
            fp1.write("v %f %f %f\n" % (v[0], v[1], v[2]))

        for f in faces + 1:  # Faces are 1-based, not 0-based in obj files
            fp1.write("f %d %d %d\n" % (f[0], f[1], f[2]))
    
    annotations_path = (
        dataset_annotation_path
        + gender
        + "/"
        + "subject_mesh_%0.*d_anno.json" % (padding, already_synthesized)
    )
    
    ## Print message
    print("..Output mesh in zero pose saved to: {}".format(outmesh_path))
    print("..Output mesh in pose 1 saved to: {}".format(outmesh_path1))

    if not os.path.exists(os.path.dirname(annotations_path)):
        try:
            os.makedirs(os.path.dirname(annotations_path))
        except OSError as exc:  # Guard against race condition
            import errno

            if exc.errno != errno.EEXIST:
                raise

    with open(annotations_path, "w") as fp:
        # Write the betas. Since "the betas" is a matrix, we have to
        # 'listifyit'.
        json.dump(
            {"betas": subject_betas.tolist(), "human_dimensions": {}},
            fp,
            sort_keys=True,
            indent=4,
            ensure_ascii=False,
        )
    
    ## Print message
    print("..Output annotations saved to: {}".format(annotations_path))

str_log_end = "finished synthesizing meshes at {:%d %B %Y %H:%M:%S}".format(
    datetime.now()
)
print(str_log_end)
with open(json_log_path, "w") as fp:
    json.dump(
        {"started": str_log_start, "ended": str_log_end},
        fp,
        sort_keys=True,
        indent=4,
        ensure_ascii=False,
    )
