# -*- coding: utf-8 -*-
"""
@author: yansel
"""
import os
import locale
from calvis import Calvis
na = __import__("neural-anthropometer")
import torch

try:
    from vtkplotter import (
        Plotter,
        trimesh2vtk,
        settings,
        write,
        Text2D,
        Lines,
        Line,
        exportWindow
    )
except:
    from vedo import (
        Plotter,
        trimesh2vedo,
        settings,
        write,
        Text2D,
        Lines,
        Line,
        exportWindow
    )

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
    'male': SMPL_basicModel_m_lbs_path
  }

dataset = na.NeuralAnthropometerBasic(rootDir)

subsampler = torch.utils.data.SubsetRandomSampler(range(0, len(dataset)))

# Define data loaders for training and testing data in this fold
loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, sampler=subsampler
)

json_log_path = dataset.json_log_path

sharmean = na.Sharmean()
item_id = None
# if we want a particular subject
want_subject = True
if want_subject:
    # put here the item id (which probably not the subject number)
    item_id = 11
    subject = dataset.__getitem__(item_id)
    # normalize
    meshi = subject.copy()
    meshi["person_gender"] = [subject["person_gender"]]
    meshi["pose0_file"] = [subject["pose0_file"]]
    meshi["mesh_name"] = [subject["mesh_name"]]
else:
    # meshpath
    # just random
    subject = next(iter(loader))
    meshi = subject



# plotter
settings.embedWindow(backend=False)
vp = Plotter(shape=(1, 3), size=(800, 500), bg="w")
# vp.sharecam = False
settings.useDepthPeeling = False


epsilon = 65
m = 0.005
N = 55

meshpath = meshi["pose0_file"][0]
gender = meshi["person_gender"][0]

sharmean.clear()
sharmean.mesh_path(meshpath)
sharmean.load_mesh(basicModel, gender=gender)
sharmean.load_trimesh()

# shoulder width
sw = sharmean.shoulder_width()
sw_subcurve = sharmean.return_shoulder_width_subcurve()
# assemble the subcurves
sw_actor = (
    Lines(sw_subcurve[:, 0, :], endPoints=sw_subcurve[:, 1, :]).lw(5).c("g")
)

# change pose
sharmean.pose_model_pose2()

# right arm length
ral = sharmean.right_arm_lenth()
ral_subcurve = sharmean.return_right_arm_subcurve()
# assemble the subcurves
ral_actor = (
    Lines(ral_subcurve[:, 0, :], endPoints=ral_subcurve[:, 1, :]).lw(5).c("m")
)

# left arm length
lal = sharmean.left_arm_lenth()
lal_subcurve = sharmean.return_left_arm_subcurve()
# assemble the subcurves
lal_actor = (
    Lines(lal_subcurve[:, 0, :], endPoints=lal_subcurve[:, 1, :]).lw(5).c("b")
)

# inseam
ins = sharmean.inseam()
# array of two points
inseam_line = sharmean.return_inseam_line()
inseam_actor = Line(inseam_line).lw(5).c("c")

height = sharmean.height()

calvis = Calvis()
calvis.calvis_clear()
calvis.mesh_path(meshpath)
calvis.load_trimesh()
calvis.fit_SMPL_model_to_mesh(basicModel, gender=gender)

calvis.segmentation(N=N)

calvis.assemble_mesh_signatur(m=m)

calvis.assemble_slice_statistics()

cc = calvis.chest_circumference()
ccslice_2D, to_3D = cc.to_planar()
cc_actor = trimesh2vedo(cc).unpack()[0].c('black').lw(5)

wc = calvis.waist_circumference()
wcslice_2D, to_3D = wc.to_planar()
wc_actor = trimesh2vedo(wc).unpack()[0].c('indigo').lw(5)

pc = calvis.pelvis_circumference()
pcslice_2D, to_3D = pc.to_planar()
pc_actor = trimesh2vedo(pc).unpack()[0].c('maroon').lw(5)

# Print info
text_info = (
    "Shoulder width (green subcurve) is {:.2f} cm.\n"
    "Right arm length (magenta subcurve) is {:.2f} cm.\n"
    "Left arm length (blue subcurve) is {:.2f} cm.\n"
    "Inseam (cyan line) is {:.2f} cm.\n"
    "Chest circunference length (black curve) is {:.2f} m.\n"
    "Waist circunference length  (indigo curve) is {:.2f} m.\n"
    "Pelvis circunference length  (maroon curve) is {:.2f} m.\n"
    "Height is {:.2f} m.\n"
).format(
    sw * 100,
    ral * 100,
    lal * 100,
    ins * 100,
    ccslice_2D.length,
    wcslice_2D.length,
    pcslice_2D.length,
    height,
)

text = Text2D(text_info, pos=6, s=1)
text0 = Text2D("Subject {} in Pose 2".format(subject), pos=3, s=1)
text1 = Text2D("Subject {} in Pose 1".format(subject), pos=3, s=1)
text2 = Text2D(
    "Human Body Dimensions (HBD) for subject {}".format(subject), pos=3, s=1
)

human = vp.load(meshpath)

vp.show(
    sharmean.mesh.alpha(0.4),
    sw_actor,
    ral_actor,
    lal_actor,
    inseam_actor,
    text0,
    at=0,
)
vp.show(human.c("y").alpha(0.4), cc_actor, wc_actor, pc_actor, text1, at=1)
vp.show(text, text2, at=2)
vp.show(interactive=1)
