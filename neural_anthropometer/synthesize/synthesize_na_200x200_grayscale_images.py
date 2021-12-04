import sys
import os
import bpy
from mathutils import Matrix
from NeuralAnthropometerDataset import NeuralAnthropometerBasic
import time
import json

dataset_path = os.path.join("..", "..", "dataset")
dataset_path = os.path.abspath(dataset_path)
dataset_meshes_path = os.path.join(dataset_path, "human_body_meshes/")
dataset_images_path = os.path.join(dataset_path, "synthetic_images/200x200/")

dataset_meshes_path_pose0 = os.path.join(dataset_meshes_path, "pose0/")
dataset_meshes_path_pose1 = os.path.join(dataset_meshes_path, "pose1/")

dataset_images_path_pose0 = os.path.join(dataset_images_path, "pose0/")
dataset_images_path_pose1 = os.path.join(dataset_images_path, "pose1/")

female_meshes_path_pose0 = os.path.join(dataset_meshes_path_pose0, "female/")
male_meshes_path_pose0 = os.path.join(dataset_meshes_path_pose0, "male/")

female_meshes_path_pose1 = os.path.join(dataset_meshes_path_pose1, "female/")
male_meshes_path_pose1 = os.path.join(dataset_meshes_path_pose1, "male/")

female_images_path_pose0 = os.path.join(dataset_images_path_pose0, "female/")
male_images_path_pose0 = os.path.join(dataset_images_path_pose0, "male/")

female_images_path_pose1 = os.path.join(dataset_images_path_pose1, "female/")
male_images_path_pose1 = os.path.join(dataset_images_path_pose1, "male/")

dataset = NeuralAnthropometerBasic(dataset_path)

json_log_dir = dataset.json_log_dir
json_log_path = os.path.join(json_log_dir, "synthezing_images.json")

try:
    for d in [
        dataset_images_path,
        dataset_images_path_pose0,
        dataset_images_path_pose1,
        female_images_path_pose0,
        male_images_path_pose0,
        female_images_path_pose1,
        male_images_path_pose1,
    ]:
        os.makedirs(d, exist_ok=False)
except:
    pass


##############################################################################
# Only tested on Blender 2.91.0 Alpha
# Wheel downloaded and installed from
# https://github.com/TylerGubala/blenderpy/releases/tag/v2.91a0
##############################################################################
# init scene
scene = bpy.data.scenes["Scene"]
# blender < v 2.80
# scene.render.engine = "BLENDER_RENDER"
scene.render.engine = "CYCLES"
# scene.render.engine = "BLENDER_EEVEE"
# set camera properties and initial position
cam_ob = bpy.data.objects["Camera"]
# scene.objects.active = cam_ob
cam_ob.matrix_world = Matrix(
    (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 0.08715556561946869, -0.9961947202682495, -3.0),
        (0.0, 0.9961947202682495, 0.08715556561946869, 0.0),
        (0.0, 0.0, 0.0, 1.0),
    )
)

cam_ob.data.lens = 60
# Clipping is related to the depth of field (DoF). Rendering will start and end
# at this depth values.
cam_ob.data.clip_start = 0.1
cam_ob.data.clip_end = 100
# (default) cam_ob.data.sensor_fit = AUTO
cam_ob.data.sensor_width = 32
# (default) cam_ob.data.sensor_hight = 18
cam_ob.data.type = "ORTHO"
cam_ob.data.ortho_scale = 2.5
# blender < v 2.80
# cam_ob.data.draw_size = 0.5

# delete the default cube (which held the material) if any

try:
    bpy.data.objects["Cube"].select_set(True)
    bpy.ops.object.delete()
except:
    print("No default cube found")

# position the lamp in front of the camara as simulating a "real lamp"
# hanging from the celling
lamp_objects = [o for o in bpy.data.objects if o.type == "LAMP"]

# if there are no lamps we add one
if len(lamp_objects) == 0:
    # Create new lamp datablock
    # blender < v 2.80
    lamp_data = bpy.data.lights.new(name="Lamp", type="POINT")
    # lamp_data = bpy.ops.object.light_add(type='POINT')

    # Create new object with our lamp datablock
    lamp = bpy.data.objects.new(name="Lamp", object_data=lamp_data)

    # Link lamp object to the scene so it'll appear in this scene
    # Links object to the master collection of the scene.
    scene.collection.objects.link(lamp)

    # link light object
    bpy.context.collection.objects.link(lamp)
else:
    lamp = lamp_objects[0]

lamp.matrix_world = Matrix(
    (
        (1.0, 0.0, 0.0, 0.10113668441772461),
        (0.0, 1.0, 0.0, -0.8406344056129456),
        (0.0, 0.0, 1.0, 1.4507088661193848),
        (0.0, 0.0, 0.0, 1.0),
    )
)

# 200 pixels in the x direction
bpy.context.scene.render.resolution_x = 200
# 200 pixels in the y direction
bpy.context.scene.render.resolution_y = 200
# "Percentage scale for render resolution"
bpy.context.scene.render.resolution_percentage = 100

# save a grayscale(BlackWhite) png image
scene.render.image_settings.color_mode = "BW"
scene.render.image_settings.file_format = "PNG"
# set the background color to white (R, G, B)
bpy.data.worlds["World"].node_tree.nodes["Background"].inputs[
    0
].default_value = (1, 1, 1, 1)

start_time = time.time()
print("Started to render pictures from humans at %s" % start_time)

# Iterating over the dataset returns a dictionary as follows:
# sample = {
#   # Just the mesh name, e.g., subject_mesh_0013.obj.
#   # Be aware that the mesh name is not unique. There are
#   # four! (female, male, pose0, pose1) meshes for each subject
#   # with the same mesh name.
#   "mesh_name": self.meshes[idx],
#   "meshfile": mesh file path,
#   "person_gender": "male" if gender == 1 else "female",
#   "mesh_pose": pose,  # 0 for zero pose or 1 pose one.
#   "annotations": annotations/human body dimensions in json format,
#   "annotation_file": annotation file path,
#   # This is a kind of normalization. One of these two keys
#   # equals the key "meshfile" depending on which pose the
#   # current mesh is in. The idea is that if you iterate over the
#   # dataset and you get a mesh which is in pose 0, you can know
#   # also the location of this mesh in pose 1, and vice versa.
#   "pose0_file": mesh path in pose zero,
#   "pose1_file": mesh path in pose one,
# }

for i, meshi in enumerate(dataset, 0):
    meshpath = meshi["meshfile"]
    gender = meshi["person_gender"]
    imagename = meshi["mesh_name"][:-3] + "png"
    pose = meshi["mesh_pose"]
    if gender == "male" and pose == 0:
        savepng = os.path.join(male_images_path_pose0, imagename)
    elif gender == "male" and pose == 1:
        savepng = os.path.join(male_images_path_pose1, imagename)
    elif gender == "female" and pose == 0:
        savepng = os.path.join(female_images_path_pose0, imagename)
    elif gender == "female" and pose == 1:
        savepng = os.path.join(female_images_path_pose1, imagename)

    # Deselect everthing
    bpy.ops.object.select_all(action="DESELECT")

    bpy.ops.import_scene.obj(filepath=meshpath)
    # select this object
    imported_object = bpy.data.objects[meshi["mesh_name"][:-4]]
    imported_object.select_set(True)

    # print('Imported name: ', obj_object.name)
    imported_object.data.use_auto_smooth = (
        False  # autosmooth creates artifacts
    )
    # change Base Color
    bpy.data.materials["Material"].node_tree.nodes["Principled BSDF"].inputs[
        0
    ].default_value = (0.111, 0.111, 0.111, 1)
    # change Subsurface Color
    bpy.data.materials["Material"].node_tree.nodes["Principled BSDF"].inputs[
        3
    ].default_value = (0, 0, 0, 1)
    # change Emission
    bpy.data.materials["Material"].node_tree.nodes["Principled BSDF"].inputs[
        17
    ].default_value = (0, 0, 0, 1)

    imported_object.active_material = bpy.data.materials["Material"]
    # update scene, if needed
    layer = bpy.context.view_layer
    layer.update()

    # blender < v2.80
    # scene.render.use_antialiasing = True
    # bpy.context.scene.display.render_aa = 'OFF'
    scene.render.filepath = savepng
    # disable render output
    logfile = "/dev/null"
    open(logfile, "a").close()
    old = os.dup(1)
    sys.stdout.flush()
    os.close(1)
    os.open(logfile, os.O_WRONLY)

    # Render
    bpy.ops.render.render(write_still=True)

    # now delete this mesh and update
    bpy.ops.object.delete()
    # print('Object %s deleted.' % obj_object.name)
    to_remove = [block for block in bpy.data.meshes if block.users == 0]
    for block in to_remove:
        bpy.data.meshes.remove(block)

    layer = bpy.context.view_layer
    layer.update()

    # disable output redirection
    os.close(1)
    os.dup(old)
    os.close(old)

    print(
        "Png image for {} with gender {} in pose {} saved"
        " under {}".format(meshi["mesh_name"], gender, pose, savepng)
    )

finish_time = time.time()
str1 = "Started to render pictures from humans at {}".format(
    time.asctime(time.localtime(start_time))
)

print(str1)

str2 = "Finished to render pictures from humans at {}".format(
    time.asctime(time.localtime(finish_time))
)
print(str2)

with open(json_log_path, "w") as fp:
    json.dump(
        {"started": str1, "ended": str2},
        fp,
        sort_keys=False,
        indent=4,
        ensure_ascii=False,
    )

elapsed_time = finish_time - start_time
print("Total time needed was %s seconds" % elapsed_time)
