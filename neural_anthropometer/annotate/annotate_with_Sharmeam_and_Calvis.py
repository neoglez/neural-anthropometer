# -*- coding: utf-8 -*-
"""
@author: neoglez
"""

import numpy as np
import os
import locale
from calvis import Calvis
import neural_anthropometer as na
import time
import json
from datetime import datetime

locale.setlocale(locale.LC_NUMERIC, "C")

rootDir = os.path.join("..", "..", "dataset")
rootDir = os.path.abspath(rootDir)

smpl_models = os.path.join(rootDir,"..", "datageneration", "data")

SMPL_basicModel_f_lbs_path = os.path.join(smpl_models,
                                       "basicModel_f_lbs_10_207_0_v1.0.0.pkl")
SMPL_basicModel_m_lbs_path = os.path.join(smpl_models,
                                       "basicmodel_m_lbs_10_207_0_v1.0.0.pkl")
basicModel = {
    'female': SMPL_basicModel_f_lbs_path,
    'male': SMPL_basicModel_m_lbs_path,
    'f': SMPL_basicModel_f_lbs_path,
    'm': SMPL_basicModel_m_lbs_path
  }

dataset = na.NeuralAnthropometerBasic(rootDir)
json_log_path = dataset.json_log_path

sharmeam = na.Sharmeam()
calvis = Calvis()


epsilon = 65
m = 0.005
N = 55

str_log_start = ("Started to calculate human body dimensions from 3D human"
                 " meshes at {:%d %B %Y %H:%M:%S}").format(datetime.now())
print(str_log_start)

dataset_length = len(dataset)

for i, meshi in enumerate(dataset, 0):

    # meshpath
    meshpath = meshi['pose0_file']

    sharmeam.clear()
    sharmeam.mesh_path(meshpath)
    sharmeam.load_mesh(basicModel, gender=meshi['person_gender'])
    sharmeam.load_trimesh()
    
    # shoulder width
    sw = sharmeam.shoulder_width()
    
    # right arm length
    ral = sharmeam.right_arm_lenth()

    # left arm length
    lal = sharmeam.left_arm_lenth()

    # inseam
    ins = sharmeam.inseam()

    height = sharmeam.height()

    # start calculating with calvis
    calvis.calvis_clear()
    calvis.mesh_path(meshpath)
    calvis.load_trimesh()
    calvis.fit_SMPL_model_to_mesh(basicModel, gender=meshi['person_gender'])

    calvis.segmentation(N=N)

    calvis.assemble_mesh_signatur(m=m)

    calvis.assemble_slice_statistics()

    cc = calvis.chest_circumference()
    ccslice_2D, to_3D = cc.to_planar()

    wc = calvis.waist_circumference()
    wcslice_2D, to_3D = wc.to_planar()

    pc = calvis.pelvis_circumference()
    pcslice_2D, to_3D = pc.to_planar()

    # Print info
    print("Shoulder width is: %s" % sw)
    print("Right arm length is: %s" % ral)
    print("Left arm length is: %s" % lal)
    print("Inseam is: %s" % ins)
    print("Chest circunference length is: %s" % ccslice_2D.length)
    print("Waist circunference length is: %s" % wcslice_2D.length)
    print("Pelvis circunference length is: %s" % pcslice_2D.length)
    print("Height is: %s" % height)

    annotation_file = meshi['annotation_file']

    with open(annotation_file, "r") as fp:
        data = json.load(fp)
        betas = np.array([beta for beta in data["betas"]])

    with open(annotation_file, "w") as fp:
        # Write the betas. Since "the betas" is a matrix, we have to
        # 'listifyit'.
        json.dump(
            {
                "betas": betas.tolist(),
                "human_dimensions": {
                    "shoulder_width": sw,
                    "right_arm_length": ral,
                    "left_arm_length": lal,
                    "inseam": ins,
                    "chest_circumference": ccslice_2D.length,
                    "waist_circumference": wcslice_2D.length,
                    "pelvis_circumference": pcslice_2D.length,
                    "height": height
                },
            },
            fp,
            sort_keys=True,
            indent=4,
            ensure_ascii=False,
        )
    print("Saved annotations file in %s" % annotation_file)
    print("%s percent finished" % (i / dataset_length * 100))

finish_time = time.time()


str_log_end = ("finished calculate human body dimensions from 3D human meshes"
               " at {:%d %B %Y %H:%M:%S}").format(datetime.now())
print(str_log_end)
with open(json_log_path, "w") as fp:
    json.dump(
        {"started": str_log_start, "ended": str_log_end},
        fp,
        sort_keys=True,
        indent=4,
        ensure_ascii=False,
    )
