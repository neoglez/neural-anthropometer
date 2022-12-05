# -*- coding: utf-8 -*-
"""
@author: yansel
"""

from __future__ import print_function, division
import os
import numpy as np
import json
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms


def read_meshes_from_directory(dir):
    meshes = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            meshes.append(fname)

    return meshes


def read_images_from_directory(dir):
    images = []
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in sorted(fnames):
            images.append(fname)

    return images


def load_annotation_data(data_file_path):
    with open(data_file_path, "r") as data_file:
        return json.load(data_file)


def read_from_obj_file(inmesh_path):
    resolution = 6890

    vertices = np.zeros([resolution, 3], dtype=float)
    faces = np.zeros([resolution - 1, 3], dtype=int)

    with open(inmesh_path, "r") as fp:
        meshdata = np.genfromtxt(fp, usecols=(1, 2, 3))
        vertices = meshdata[:resolution, :]
        faces = meshdata[resolution:, :].astype("int")
    return {"vertices": vertices, "faces": faces}


###############################################################################
#        NA basic dataset. Defines basic init. logic.
###############################################################################


class NeuralAnthropometerBasic(Dataset):
    """
    NeuralAnthropometer basic dataset.
    It just defines basic initialization logic.
    It can be used to iterate over the 3D meshes and if they already
    exist also over annotations.
    """

    def __init__(self, root_dir="../dataset", transform=None):
        """
        Args:
            root_dir (string): Directory containing 2D data and annotation.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.dataset_meshes_path = os.path.join(
            self.root_dir, "human_body_meshes/"
        )
        self.dataset_meshes_path_pose0 = os.path.join(
            self.dataset_meshes_path, "pose0/"
        )
        self.dataset_meshes_path_pose1 = os.path.join(
            self.dataset_meshes_path, "pose1/"
        )
        self.female_meshes_path_pose0 = os.path.join(
            self.dataset_meshes_path_pose0, "female/"
        )
        self.male_meshes_path_pose0 = os.path.join(
            self.dataset_meshes_path_pose0, "male/"
        )
        self.female_meshes_path_pose1 = os.path.join(
            self.dataset_meshes_path_pose1, "female/"
        )
        self.male_meshes_path_pose1 = os.path.join(
            self.dataset_meshes_path_pose1, "male/"
        )

        self.dataset_annotation_path = os.path.join(
            self.root_dir, "annotations/"
        )
        self.json_log_dir = os.path.join(self.root_dir, "log/")
        #  Log for the calculation of human body dimensions to assemble
        #  the dataset.
        self.json_log_path = os.path.join(self.json_log_dir, "dataset.json")

        # these are mandatory
        for directory in [
            self.root_dir,
            self.dataset_meshes_path,
            self.dataset_meshes_path_pose0,
            self.dataset_meshes_path_pose1,
            self.female_meshes_path_pose0,
            self.male_meshes_path_pose0,
            self.female_meshes_path_pose1,
            self.male_meshes_path_pose1,
        ]:
            if not os.path.isdir(directory):
                raise ValueError(directory + " does not exist!")
        # The dataset is the concatenated meshes found on female and male
        # directories. The list contains the meshes filenames.
        meshes = []
        # holds meshes in pose 0 or 1
        mesh_poses = []
        # holds mesh genders: female => 0, male => 1
        mesh_genders = []
        # holds annotation files
        annotation_files = []
        pose0_female_meshes = read_meshes_from_directory(
            self.female_meshes_path_pose0
        )
        pose0_male_meshes = read_meshes_from_directory(
            self.male_meshes_path_pose0
        )
        pose1_female_meshes = read_meshes_from_directory(
            self.female_meshes_path_pose1
        )
        pose1_male_meshes = read_meshes_from_directory(
            self.male_meshes_path_pose1
        )

        # consistency tests
        assert len(pose0_female_meshes) == len(pose1_female_meshes)
        assert len(pose1_female_meshes) == len(pose0_male_meshes)
        assert len(pose0_male_meshes) == len(pose1_male_meshes)

        for m in pose0_female_meshes:
            meshes.append(m)
            annotation_files.append(m[:13] + m[13:-4] + "_anno.json")
            mesh_poses.append(0)
            mesh_genders.append(0)
        for m in pose0_male_meshes:
            meshes.append(m)
            annotation_files.append(m[:13] + m[13:-4] + "_anno.json")
            mesh_poses.append(0)
            mesh_genders.append(1)
        for m in pose1_female_meshes:
            meshes.append(m)
            annotation_files.append(m[:13] + m[13:-4] + "_anno.json")
            mesh_poses.append(1)
            mesh_genders.append(0)
        for m in pose1_male_meshes:
            meshes.append(m)
            annotation_files.append(m[:13] + m[13:-4] + "_anno.json")
            mesh_poses.append(1)
            mesh_genders.append(1)

        self.meshes = meshes
        self.mesh_genders = mesh_genders
        self.mesh_poses = mesh_poses
        self.annotation_files = annotation_files

    def __len__(self):
        return len(self.meshes)

    def __getitem__(self, idx):
        meshfile = ""
        gender = self.mesh_genders[idx]
        pose = self.mesh_poses[idx]
        pose0file = ""
        pose1file = ""
        if gender == 1 and pose == 0:
            meshfile = os.path.join(
                self.male_meshes_path_pose0, self.meshes[idx]
            )
        elif gender == 1 and pose == 1:
            meshfile = os.path.join(
                self.male_meshes_path_pose1, self.meshes[idx]
            )
        elif gender == 0 and pose == 0:
            meshfile = os.path.join(
                self.female_meshes_path_pose0, self.meshes[idx]
            )
        elif gender == 0 and pose == 1:
            meshfile = os.path.join(
                self.female_meshes_path_pose1, self.meshes[idx]
            )

        if gender == 0:
            annofile = os.path.join(
                self.dataset_annotation_path,
                "female",
                self.annotation_files[idx],
            )
            pose0file = os.path.join(
                self.female_meshes_path_pose0, self.meshes[idx]
            )
            pose1file = os.path.join(
                self.female_meshes_path_pose1, self.meshes[idx]
            )
        elif gender == 1:
            annofile = os.path.join(
                self.dataset_annotation_path,
                "male",
                self.annotation_files[idx],
            )
            pose0file = os.path.join(
                self.male_meshes_path_pose0, self.meshes[idx]
            )
            pose1file = os.path.join(
                self.male_meshes_path_pose1, self.meshes[idx]
            )

        sample = {
            # Just the mesh name, e.g., subject_mesh_0013.obj.
            # Be aware that the mesh name is not unique. There are
            # four! (female, male, pose0, pose1) meshes for each subject
            # with the same mesh name.
            "mesh_name": self.meshes[idx],
            "meshfile": meshfile,
            "person_gender": "male" if gender == 1 else "female",
            "mesh_pose": pose,  # 0 for zero pose or 1 pose one.
            "annotations": load_annotation_data(annofile),
            "annotation_file": annofile,
            # This is a kind of normalization. One of these two keys
            # equals the key "meshfile" depending on which pose the
            # current mesh is in. The idea is that if you iterate over the
            # dataset and you get a subject mesh which is in pose 0, you can
            # know also the location of this subject mesh in pose 1, and
            # vice versa.
            "pose0_file": pose0file,
            "pose1_file": pose1file,
            # A string describing the subject, gender and the pose. Useful for
            # logging or displaying information.
            "subject_string": "{} subject {} in pose {}".format(
                "Male" if gender == 1 else "Female",
                self.meshes[idx][13:-4],
                pose,
            ),
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def getFemaleIndxs(self):
        indexes = [i for i, g in enumerate(self.mesh_genders) if g == 0]
        return indexes

    def getMaleIndxs(self):
        indexes = [i for i, g in enumerate(self.mesh_genders) if g == 1]
        return indexes

    def getFemalePose0Indxs(self):
        indexes = [
            i
            for i, g in enumerate(self.mesh_genders)
            if g == 0 and self.mesh_poses[i] == 0
        ]
        return indexes

    def getMalePose0Indxs(self):
        indexes = [
            i
            for i, g in enumerate(self.mesh_genders)
            if g == 1 and self.mesh_poses[i] == 0
        ]
        return indexes

    def getFemalePose1Indxs(self):
        indexes = [
            i
            for i, g in enumerate(self.mesh_genders)
            if g == 0 and self.mesh_poses[i] == 1
        ]
        return indexes

    def getMalePose1Indxs(self):
        indexes = [
            i
            for i, g in enumerate(self.mesh_genders)
            if g == 1 and self.mesh_poses[i] == 1
        ]
        return indexes

    def getSameElementsIndxsForPose1(self, pose0_idxs):
        """
        Given indices of meshes in pose 0, return the indices
        of these meshes in pose 1

        Parameters
        ----------
        pose0_idxs : array
            Indices of subject meshes in pose 0.

        Returns
        -------
        indexes : Corresponding indices of subject meshes in pose 1.

        """
        # Just add 6000, see the constructor
        for e in pose0_idxs:
            if e < 0 and e > 5999:
                raise Exception(
                    "Element indices of meshes in pose 0 must be"
                    "between 0 and 5999"
                )
        # pose0_female_meshes = np.asanyarray(self.meshes)[:3000]
        # pose0_male_meshes = np.asanyarray(self.meshes)[3000:6000]
        # pose1_female_meshes = np.asanyarray(self.meshes)[6000:9000]
        # pose1_male_meshes = np.asanyarray(self.meshes)[9000:]
        pose1_elements = np.asanyarray(pose0_idxs) + 6000
        return pose1_elements

    def getHumanBodyDimensionsNames(self):
        """
        Return the a list which the human body dimensions names in the order
        returned by the dataset

        Returns
        -------
        list

        """
        return [
            'chest_circumference','height', 'inseam', 'left_arm_length',
            'pelvis_circumference', 'right_arm_length', 'shoulder_width',
            'waist_circumference'
            ]
    
    def getDefaultHumanBodyDimensionsMetricUnit(self):
        """
        Return the default metric unit in which the human body dimensions
        are expressed.

        Returns
        -------
        str

        """
        return 'meter'

###############################################################################
#        NeuralAntropometer 200 x 200 Synthetic Images Dataset.
###############################################################################


class NeuralAnthropometerSyntheticImagesDataset(NeuralAnthropometerBasic):
    """
    2D 200x200 images synthesized with blender plus
    annotations generated automatically by NeuralAntropometer.
    """

    def __init__(
        self, root_dir="../dataset", transform=None, load_images=True
    ):
        """
 Args:
            root_dir (string): Directory with all the 3D body models(meshes)
            and the annotations.
            png_dir (string): Directory with all images.
            transform (callable, optional): Optional transform to be applied
                on a sample.

        Parameters
        ----------
        root_dir : String, optional
            Dataset root directory. Default is "../dataset".
        transform : callable, optional
            Optional transform to be applied on a sample.
        load_images : Boolean, optional
            If true images are loaded as part of the sample. Otherwise the
            sample entry is None. The default is True.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        None.

        """
        super().__init__(root_dir, transform)
        self.image_dir = os.path.join(self.root_dir, "synthetic_images")
        self.image_200x200_dir = os.path.join(self.image_dir, "200x200")
        self.dataset_images_path_pose0 = os.path.join(
            self.image_200x200_dir, "pose0"
        )
        self.dataset_images_path_pose1 = os.path.join(
            self.image_200x200_dir, "pose1"
        )
        self.female_images_path_pose0 = os.path.join(
            self.dataset_images_path_pose0, "female"
        )
        self.male_images_path_pose0 = os.path.join(
            self.dataset_images_path_pose0, "male"
        )
        self.female_images_path_pose1 = os.path.join(
            self.dataset_images_path_pose1, "female"
        )
        self.male_images_path_pose1 = os.path.join(
            self.dataset_images_path_pose1, "male"
        )

        for directory in [
            self.image_dir,
            self.image_200x200_dir,
            self.dataset_images_path_pose0,
            self.dataset_images_path_pose1,
            self.female_images_path_pose0,
            self.male_images_path_pose0,
            self.female_images_path_pose1,
            self.male_images_path_pose1,
        ]:
            if not os.path.isdir(directory):
                raise ValueError(directory + " does not exist!")
        self.load_images = load_images
        # The dataset is the concatenated images found on female and male
        # directories. The list contains the image filenames.
        images = []
        pose0_female_images = read_images_from_directory(
            self.female_images_path_pose0
        )
        pose0_male_images = read_images_from_directory(
            self.male_images_path_pose0
        )
        pose1_female_images = read_meshes_from_directory(
            self.female_images_path_pose1
        )
        pose1_male_images = read_meshes_from_directory(
            self.male_images_path_pose1
        )

        # consistency tests
        assert len(pose0_female_images) == len(pose1_female_images)
        assert len(pose1_female_images) == len(pose0_male_images)
        assert len(pose0_male_images) == len(pose1_male_images)

        for img in pose0_female_images:
            images.append(img)
            # annotation, pose and gender are in the parent class
        for img in pose0_male_images:
            images.append(img)
        for img in pose1_female_images:
            images.append(img)
        for img in pose1_male_images:
            images.append(img)

        self.images = images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # augment the sample with image information
        sample = super().__getitem__(idx)
        gender = self.mesh_genders[idx]
        pose = self.mesh_poses[idx]
        imagefile = ""
        pose0file = ""
        pose1file = ""

        if gender == 0:
            pose0file = os.path.join(
                self.female_images_path_pose0, self.images[idx]
            )
            pose1file = os.path.join(
                self.female_images_path_pose1, self.images[idx]
            )
        elif gender == 1:
            pose0file = os.path.join(
                self.male_images_path_pose0, self.images[idx]
            )
            pose1file = os.path.join(
                self.male_images_path_pose1, self.images[idx]
            )

        if pose == 0:
            imagefile = pose0file
        elif pose == 1:
            imagefile = pose1file

        image = None
        if self.load_images:
            image = Image.open(imagefile)
        sample["image"] = image
        sample["imagefile"] = imagefile

        if self.transform:
            sample = self.transform(sample)

        return sample


###############################################################################
class NeuralAnthropometerSyntheticImagesDatasetTrainTest(
    NeuralAnthropometerSyntheticImagesDataset
):
    """
    The test set has 600 female and 600 male ids equally chosen from pose 0 and
    pose 1. That is 300 female ids in pose 0 and 300 male ids in pose 0; and 300
    female ids in pose 1 and 300 male ids in pose 1.
    The train set has 5400 female and 5400 male images equally chosen from pose 0
    and pose 1. That is 2700 female images in pose 0 and 2700 male images in
    pose 0; and 2700 female ids in pose 1 and 2700 male ids in pose 1.
    """

    def __init__(self, root_dir, train=True, transform=None):
        """
        Args:
            root_dir (string): Directory with all the 3D body models(meshes),
                images and annotations.
            train (boolean): If True, load elements from the train set,
                otherwise from the test set.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__(root_dir, transform)
        # this should not be changed
        self.train_test_split_dir = os.path.join(self.root_dir, "train_test")
        self.split_file = "train_test_split.json"
        self.train_test_split_file = os.path.join(
            self.train_test_split_dir, self.split_file
        )
        # print(self.train_test_split_file)
        with open(self.train_test_split_file) as json_file:
            data = json.load(json_file)

        self.test_female_pose0 = data["test"]["female"]["pose0"]
        self.test_female_pose1 = data["test"]["female"]["pose1"]
        self.test_male_pose0 = data["test"]["male"]["pose0"]
        self.test_male_pose1 = data["test"]["male"]["pose1"]

        self.train_female_pose0 = data["train"]["female"]["pose0"]
        self.train_female_pose1 = data["train"]["female"]["pose1"]
        self.train_male_pose0 = data["train"]["male"]["pose0"]
        self.train_male_pose1 = data["train"]["male"]["pose1"]
        self.train = train

        img_numpy = np.asanyarray(self.images)
        mesh_numpy = np.asanyarray(self.meshes)
        gender_numpy = np.asanyarray(self.mesh_genders)
        pose_numpy = np.asanyarray(self.mesh_poses)
        anno_numpy = np.asanyarray(self.annotation_files)

        if self.train == True:
            must_delete = (
                self.test_female_pose0
                + self.test_female_pose1
                + self.test_male_pose0
                + self.test_male_pose1
            )
        else:
            must_delete = (
                self.train_female_pose0
                + self.train_female_pose1
                + self.train_male_pose0
                + self.train_male_pose1
            )

        must_delete.sort()

        must_delete = np.asanyarray(must_delete)

        self.images = np.delete(img_numpy, must_delete).tolist()
        # also the meshes to keep everything synchronized
        self.meshes = np.delete(mesh_numpy, must_delete).tolist()
        self.mesh_genders = np.delete(gender_numpy, must_delete).tolist()
        self.mesh_poses = np.delete(pose_numpy, must_delete).tolist()
        self.annotation_files = np.delete(anno_numpy, must_delete).tolist()


###############################################################################

###############################################################################
if __name__ == "__main__":

    import matplotlib.pyplot as plt

    rootDir = "../dataset/"
    rootDir = os.path.abspath(rootDir)

    dataset = NeuralAnthropometerBasic(rootDir)

    indx = np.random.randint(0, len(dataset))
    # print("NeuralAntropometerBasic element with index {} follows:".format(indx))
    # print(dataset[indx])

    dataset = NeuralAnthropometerSyntheticImagesDataset(rootDir)
    print("SyntheticImagesDataset length is {}".format(len(dataset)))
    indx = np.random.randint(0, len(dataset))
    # print("SyntheticImagesDataset element with index {} follows:".format(indx))
    # print(dataset[indx])
    sample = dataset[indx]
    img = sample["image"]
    to_tensor = transforms.ToTensor()
    img_t = to_tensor(img)
    # plt.imshow(img)

    print("Instanciate Train Dataset")
    dataset = NeuralAnthropometerSyntheticImagesDatasetTrainTest(rootDir)
    print("Train Dataset length is {}".format(len(dataset)))
    indx = np.random.randint(0, len(dataset))
    # print("Train Dataset element with index {} follows:".format(indx))
    # print(dataset[indx])
    # sample = dataset[indx]
    # img = sample['image']
    # to_tensor = transforms.ToTensor()
    # img_t = to_tensor(img)
    # plt.imshow(img)

    print("Instanciate Test Dataset")
    dataset = NeuralAnthropometerSyntheticImagesDatasetTrainTest(
        rootDir, train=False
    )
    print("Test Dataset length is {}".format(len(dataset)))
    indx = np.random.randint(0, len(dataset))
    # print(" Test Dataset element with index {} follows:".format(indx))
    # print(dataset[indx])
    # sample = dataset[indx]
    # img = sample['image']
    # to_tensor = transforms.ToTensor()
    img_t = to_tensor(img)
    plt.imshow(img)
