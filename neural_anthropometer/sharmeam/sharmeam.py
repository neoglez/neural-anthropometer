import trimesh
import trimesh.viewer
import numpy as np
import vedo
import os
import pickle
import hbm
import locale
from scipy.spatial import cKDTree
import neural_anthropometer as na

locale.setlocale(locale.LC_NUMERIC, "C")


class Sharmeam:
    """
    A method to calculate SHoulder width, ARM length (right and left) and
    insEAM from 3D human body meshes.
    """

    def __init__(self, epsilon=65):
        # mesh path on file system
        self.meshpath = None
        # mesh loaded with vtkplotter a.k.a. vedo
        self.mesh = None
        # mesh loaded with trimesh library
        self.trimesh = None
        # O(bject) female model
        self.female_model = None
        # O(bject) male model
        self.male_model = None
        # O(bject) human model initialized for the concrete mesh
        self.human_model = None
        # Joint locations for this concrete human model
        self.joints_location = None
        # Fast access to gender
        self.gender = None
        # Percentage of the height to establish the y coordinate of the third
        # point to define the cutting plane (the other two points are the right
        # and left shoulder landmarks).
        self.epsilon = epsilon
        #################cache#########
        self.sw = None
        self.ral = None
        self.lal = None
        self.ins = None
        ############subcurves and lines for displaying
        self.sw_subcurve = None
        self.ral_subcurve = None
        self.lal_subcurve = None
        self.ins_line = None

    def clear(self):
        """
        Reset all state variables to its default value.
        """
        self.__init__()

    def mesh_path(self, meshpath):
        self.meshpath = meshpath
        return self

    def load_mesh(self, basicModel, gender, pose=2):
        """
        Load the mesh with vedo/vtkplotter

        Parameters
        ----------
        basicModel : Object
            SMPL models. The Object must contain two properties: 'female' and
            'male'. The values of these properties are the path where the SMPL
            models where saved.
        gender : str
            'Gender' of the model: one of 'female' or 'male'
        pose : int, optional
            The default is 2. Pose the mesh in the pose Nr.2, i.e., lower the
            arms 45 degrees. When pose=1, the mesh is not posed (zero pose).

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        # meshpath must have been set (for consistency)
        if self.meshpath is None:
            raise Exception("The meshpath must be set before loading the mesh")

        self.fit_model_to_mesh(model=basicModel, gender=gender)

        if pose == 2:
            self.pose_model_pose2()

        human = self.human_model.return_as_obj_format()
        self.human_model.calculate_stature_from_mesh()
        ###################################################################
        posed_verts = np.array(
            [
                line.split()[1:]
                for line in human.split("\n")
                if line.startswith("v ")
            ],
            dtype="f8",
        )

        faces = np.array(
            [
                line.split()[1:]
                for line in human.split("\n")
                if line.startswith("f ")
            ],
            dtype="int",
        )
        # remember that faces are 1-indexed in obj files therefore we have
        # to one from every element.
        faces = faces - 1
        self.mesh = vedo.shapes.Mesh([posed_verts, faces]).flag("pose_1")
        return self

    def load_trimesh(self, mode=1):
        """
        Load/cast the mesh with/to trimesh library

        Parameters
        ----------
        mode : TYPE, optional
            The default is 1: cast to trimesh via vedo2trimesh.
            If mode=2 load trimesh via trimesh.load_mesh(self.meshpath)

        Returns
        -------
        sharmeam self

        """
        if mode not in [1, 2]:
            raise Exception("Argument mode must be 1 or 2")
        try:
            self.trimesh = (
                vedo.vtk2trimesh(self.mesh)
                if mode == 1
                else trimesh.load_mesh(self.meshpath)
            )
        except:
            self.trimesh = (
                vedo.vedo2trimesh(self.mesh)
                if mode == 1
                else trimesh.load_mesh(self.meshpath)
            )

        return self

    def points_on_segment_query(self, curve, pointA, pointB):
        """
        Search pointA and pointB on the polyline. If both points are found, the
        segment containing these points are returned

        Parameters
        ----------
        curve : numpy ndarray of shape mx2x3
            The polygonal curve to be splitted. The curve is represented as m lines
            with start and end point.
        pointA : numpy ndarray of shape 1x3
            Point A to "cut" the curve
        pointB : numpy ndarray of shape 1x3
            Point B to "cut" the curve

        Returns
        -------
        ndarray of shape 2x3,
            segment  containing point A
        ndarray of shape 2x3
            segment index containing point B

        """
        segment_containing_A = None
        segment_containing_B = None

        # reshape the array to stack the points
        points3D = np.vstack((curve[:, 0, :], curve[:, 1, :]))
        ckdtree = cKDTree(points3D)

        pointA_index = ckdtree.query(pointA)[1]
        pointB_index = ckdtree.query(pointB)[1]

        segment_containing_A = pointA_index % len(curve)
        segment_containing_B = pointB_index % len(curve)

        return (segment_containing_A, segment_containing_B)

    def shoulder_width_subcurve(self, subcurveA, subcurveB):
        """
        Returns which of the two subcurves represents the shoulder width. We
        assume the mesh has LSA orientation, therefore the shoulder width is
        represented by the curve with minimum z coordinate.

        Parameters
        ----------
        subcurveA : numpy ndarray of shape nx2x3
            Polygonal curve. The curve is represented as m lines
            with start and end point.
        subcurveB : numpy ndarray of shape mx2x3
            Polygonal curve. The curve is represented as m lines
            with start and end point.

        Returns
        -------
        integer
            1 if the subcurve A represents the shoulder width or 2 if the
            subcurveB represents the shoulder width

        """
        return (
            1
            if (
                np.vstack((subcurveA[:, 0, :], subcurveA[:, 1, :])).min(
                    axis=0
                )[2]
                < np.vstack((subcurveB[:, 0, :], subcurveB[:, 1, :])).min(
                    axis=0
                )[2]
            )
            else 2
        )

    def arm_length_subcurve(self, subcurveA, subcurveB, side="right"):
        """
        Returns which of the two subcurves represents the corresponding (right
        or left) arm length. We assume the mesh has LSA orientation, therefore
        the arm legth is represented by the curve with minimum (side = right)
        or maximum (side = "left") x coordinate.

        Parameters
        ----------
        subcurveA : numpy ndarray of shape nx2x3
            Polygonal curve. The curve is represented as m lines
            with start and end point.
        subcurveB : numpy ndarray of shape mx2x3
            Polygonal curve. The curve is represented as m lines
            with start and end point.
        side : str
            One of the two sides "right" or "left"

        Returns
        -------
        integer
            1 if the subcurve A represents the arm length or 2 if the subcurveB
            represents the arm length.

        """
        if side not in ["right", "left"]:
            raise Exception(
                'side must be "right" or "left". Received {}'.format(side)
            )

        if side == "right":
            return (
                1
                if (
                    np.vstack((subcurveA[:, 0, :], subcurveA[:, 1, :])).min(
                        axis=0
                    )[0]
                    < np.vstack((subcurveB[:, 0, :], subcurveB[:, 1, :])).min(
                        axis=0
                    )[0]
                )
                else 2
            )

        if side == "left":
            return (
                1
                if (
                    np.vstack((subcurveA[:, 0, :], subcurveA[:, 1, :])).min(
                        axis=0
                    )[0]
                    > np.vstack((subcurveB[:, 0, :], subcurveB[:, 1, :])).min(
                        axis=0
                    )[0]
                )
                else 2
            )

    def calculate_curve_length(self, curve):
        """
        Calculate the length of a curve

        Parameters
        ----------
        curve : numpy ndarray of shape mx2x3
            The polygonal curve to be splitted. The curve is represented as m
            lines with start and end point.

        Returns
        -------
        Float

        """
        return np.linalg.norm(curve[:, 0, :] - curve[:, 1, :], axis=1).sum()
    
    def fit_model_to_mesh(self, model, gender):
        """
        

        Parameters
        ----------
        basicModel : Object
            SMPL models. The Object must contain two properties: 'female' and
            'male'. The values of these properties are the path where the SMPL
            models where saved.
        gender : str
            'Gender' of the model: one of 'female' or 'male'

        Raises
        ------
        Exception
            Argument gender must be 'female' or 'male'.

        Returns
        -------
        None.

        """

        if gender not in ["female", "male"]:
            raise Exception("Argument gender must be 'female' or 'male'")

        self.gender = gender
        basicModel_f_lbs_path = model["female"]
        basicModel_m_lbs_path = model["male"]

        try:
            # Load pkl created in python 2.x with python 2.x
            female_model = pickle.load(open(basicModel_f_lbs_path, "rb"))
            male_model = pickle.load(open(basicModel_m_lbs_path, "rb"))
        except:
            # Load pkl created in python 2.x with python 3.x
            female_model = pickle.load(
                open(basicModel_f_lbs_path, "rb"), encoding="latin1"
            )
            male_model = pickle.load(
                open(basicModel_m_lbs_path, "rb"), encoding="latin1"
            )

        ####################################################################
        # Initialize the joints regressor as dense array (for clarity).    #
        ####################################################################

        k_joints_predictor = female_model.get("J_regressor").A

        new_female_joint_regressor = hbm.KJointPredictor(k_joints_predictor)

        k_joints_predictor = male_model.get("J_regressor").A

        new_male_joint_regressor = hbm.KJointPredictor(k_joints_predictor)

        ####################################################################
        # Initialize the Osmpl female and male template.                   #
        ####################################################################
        new_female_template = hbm.OSmplTemplate(
            female_model.get("v_template"),
            female_model.get("f"),
            female_model.get("weights"),
            female_model.get("shape_blend_shapes"),
            new_female_joint_regressor,
            male_model.get("kintree_table"),
            female_model.get("posedirs"),
        )

        new_male_template = hbm.OSmplTemplate(
            male_model.get("v_template"),
            male_model.get("f"),
            male_model.get("weights"),
            male_model.get("shape_blend_shapes"),
            new_male_joint_regressor,
            male_model.get("kintree_table"),
            male_model.get("posedirs"),
        )

        ####################################################################
        # Once we have the template we instanciate the complete model.     #
        ####################################################################
        self.female_model = hbm.OSmplWithPose(
            new_female_template,
            female_model.get("shapedirs").x,
            female_model.get("posedirs"),
            None,
        )
        self.male_model = hbm.OSmplWithPose(
            new_male_template,
            male_model.get("shapedirs").x,
            male_model.get("posedirs"),
            None,
        )

        # select gender
        self.human_model = (
            self.male_model if gender == "male" else self.female_model
        )
        self.human_model.read_from_obj_file(self.meshpath)

    def pose_model_pose1(self):
        # Pose the model, in this case is just the zero pose
        pose = np.zeros(72)
        self.human_model.update_body()
        self.human_model.apply_pose_blend_shapes(pose)

        # perfom the skinning transformation
        self.human_model.apply_linear_blend_skinning()

        return self

    def pose_model_pose2(self):
        # Usually people are required to totally
        # lower their arms. However, we observe that if we impose this in
        # general for all 3D scans, and due to the fact that we are working
        # with LBS (which produce artifacts as we know) inter-penetrations
        # occur at the pelvis level with the hands. Therefore, we lower the
        # arms 'only' 45 degrees; this setting does not influence the upper
        # torso volume and so it does not have a significant impact in the
        # measurement/calculation while avoiding inter-penetrations.
        leftShoulderRotationVector = np.pi / 3 * np.array([0, 0, -1])
        rightShoulderRotationVector = np.pi / 3 * np.array([0, 0, 1])

        leftWristRotationVector = np.pi / 4.8 * np.array([0, 0, -1])
        rightWristRotationVector = np.pi / 4.8 * np.array([0, 0, 1])

        # The body parts
        left_part_s_index = 18
        right_part_s_index = 19

        left_part_w_index = 20
        right_part_w_index = 21

        # Pose the model, in this case lower the arms
        # 45 degrees.
        pose = np.zeros(72)

        left_part_pose_s_element_index_end = left_part_s_index * 3 - 3
        left_part_pose_s_element_index_begin = (
            left_part_pose_s_element_index_end - 3
        )

        right_part_pose_s_element_index_end = right_part_s_index * 3 - 3
        right_part_pose_s_element_index_begin = (
            right_part_pose_s_element_index_end - 3
        )

        left_part_pose_w_element_index_end = left_part_w_index * 3 - 3
        left_part_pose_w_element_index_begin = (
            left_part_pose_w_element_index_end - 3
        )

        right_part_pose_w_element_index_end = right_part_w_index * 3 - 3
        right_part_pose_w_element_index_begin = (
            right_part_pose_w_element_index_end - 3
        )

        pose[
            left_part_pose_s_element_index_begin:left_part_pose_s_element_index_end
        ] = [elem for elem in leftShoulderRotationVector]
        pose[
            right_part_pose_s_element_index_begin:right_part_pose_s_element_index_end
        ] = [elem for elem in rightShoulderRotationVector]
        pose[
            left_part_pose_w_element_index_begin:left_part_pose_w_element_index_end
        ] = [elem for elem in leftWristRotationVector]
        pose[
            right_part_pose_w_element_index_begin:right_part_pose_w_element_index_end
        ] = [elem for elem in rightWristRotationVector]

        self.human_model.apply_pose_blend_shapes(pose)

        # perfom the skinning transformation
        self.human_model.apply_linear_blend_skinning()

        return self

    def shoulder_width(self):
        # return the cachedone
        if self.sw is not None:
            return self.sw

        # Joints and intersections
        joints_location = (
            self.human_model.return_posed_template_joint_locations()
        )

        inverted_joint_names = dict(
            (v, k)
            for k, v in self.human_model.mean_template_shape.joint_names.items()
        )

        rshoulder_xyz = joints_location[inverted_joint_names["R_Shoulder"]]
        lshoulder_xyz = joints_location[inverted_joint_names["L_Shoulder"]]

        posedm = self.mesh

        rsray_origins = np.array(rshoulder_xyz)
        lsray_origins = np.array(lshoulder_xyz)

        rsdirectionVector = np.array(
            [rshoulder_xyz[0], posedm.ybounds()[1], posedm.zbounds()[0]]
        )
        lsdirectionVector = np.array(
            [lshoulder_xyz[0], posedm.ybounds()[1], posedm.zbounds()[0]]
        )
        rsray_direction_outside_mesh = np.copy(rsdirectionVector)
        lsray_direction_outside_mesh = np.copy(lsdirectionVector)
        rspts = posedm.intersectWithLine(
            rsray_origins, rsray_direction_outside_mesh
        )
        lspts = posedm.intersectWithLine(
            lsray_origins, lsray_direction_outside_mesh
        )

        #######################################################################
        # Construct the plane to slide the body to obtain the subcurves.
        # We need three points: two are the shoulder joints themself and the third
        # point is a function of the height. The idea is to relate the two body
        # dimensions. To achieve that, we introduce a parameter epsilon, being the
        # percentage of the height at which the y coordinate of the third point
        # is located. For example if epsilon = 15 that means that the y
        # coordinate will be at a hight of 15 % of the human mesh height starting
        # from the floor. The other (x, z) coordinates are the AABB front
        # coordinates.
        # Height is the distance from the top of the head to he floor. For
        # implementation details see the description on calulate_stature_from_mesh()
        self.human_model.calculate_stature_from_mesh()
        height = self.human_model.dimensions.stature
        point3_y_coordinate = posedm.ybounds()[0] + (
            self.epsilon / 100 * height
        )
        point3 = [
            posedm.xbounds()[1] / 2,
            point3_y_coordinate,
            posedm.zbounds()[1],
        ]
        points = np.asanyarray([rspts[0], lspts[0], point3], dtype="float")

        p0, p1, p2 = points
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]
        vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]

        u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
        point = np.array(p0)
        normal = np.array(u_cross_v)

        shoulder_plane_origin = point
        shoulder_plane_normal = normal
        # intersect
        sslice = self.trimesh.section(
            plane_origin=shoulder_plane_origin,
            plane_normal=shoulder_plane_normal,
        )

        # print(type(mslice))
        # Can be used to debug by asserting the sum of the two subsurves (see
        # beneath) is close to this length.
        shoulder_intersection_boundary_length = 0
        if sslice is None:
            shoulder_intersection_boundary_length = 0
            raise Exception("Plane did not cut the human mesh")
        else:
            slice_2D, to_3D = sslice.to_planar()
            shoulder_intersection_boundary_length = slice_2D.length
            vertices = sslice.vertices

        # nodes contains segment connection information (topology)
        nodes = sslice.entities[0].nodes

        # curve is the collection of ordered vertices (correspondingly
        # segments) that form the curve. In other words, if you stich the
        # segments together you get a polyline curve.
        curve = vertices[nodes]

        # Two points are right and left shoulder points in the "skin"
        pointA = points[0]
        pointB = points[1]

        sA_index, sB_index = self.points_on_segment_query(
            curve, pointA, pointB
        )

        if sA_index is not None and sB_index is not None:

            # We already garantied that the points are in distint segments.
            # Now assume w.l.o.g. that point A is always in the segment with
            # the lowest index in the nodes array.

            if sA_index > sB_index:
                sA_index, sB_index = sB_index, sA_index
                pointA, pointB = pointB, pointA

            lA, lB = curve[sA_index], curve[sB_index]

            # now add the vertices and the segments

            new_vertices = np.append(
                vertices, np.vstack((lA[0], lB[0])), axis=0
            )
            pointA_vertex_index = len(vertices)
            pointB_vertex_index = pointA_vertex_index + 1

            # we have to append 4 segments in total because every point divide
            # one segment in two parts
            node1 = np.array([nodes[sA_index][0], pointA_vertex_index])
            node2 = np.array([pointA_vertex_index, nodes[sA_index][1]])
            node3 = np.array([nodes[sB_index][0], pointB_vertex_index])
            node4 = np.array([pointB_vertex_index, nodes[sB_index][1]])

            # replace the two nodes by the four nodes at the right place
            subcurveA = np.append(
                node1.reshape(1, 2), nodes[:sA_index], axis=0
            )
            subcurveA = np.append(subcurveA, nodes[sB_index + 1 :], axis=0)
            subcurveA = np.append(subcurveA, node4.reshape((1, 2)), axis=0)

            subcurveB = np.append(
                node2.reshape(1, 2), nodes[sA_index + 1 : sB_index], axis=0
            )
            subcurveB = np.append(subcurveB, node3.reshape((1, 2)), axis=0)

            pathA = new_vertices[subcurveA]
            pathB = new_vertices[subcurveB]

            # calculate shoulder width selecting the right subcurve
            shoulder_width = None
            if self.shoulder_width_subcurve(pathA, pathB) == 1:
                shoulder_width = self.calculate_curve_length(pathA)
                # set the curve so it can be display if wanted
                self.sw_subcurve = pathA
            else:
                shoulder_width = self.calculate_curve_length(pathB)
                self.sw_subcurve = pathB
        else:
            raise Exception("Intersection points where not found in the curve")

        self.sw = shoulder_width
        return self.sw

    def right_arm_lenth(self):
        # return the cachedone
        if self.ral is not None:
            return self.ral

        joints_location = (
            self.human_model.return_posed_template_joint_locations()
        )
        posedm = self.mesh

        inverted_joint_names = dict(
            (v, k)
            for k, v in self.human_model.mean_template_shape.joint_names.items()
        )

        rshoulder_xyz = joints_location[inverted_joint_names["R_Shoulder"]]
        relbow_xyz = joints_location[inverted_joint_names["R_Elbow"]]
        rwrist_xyz = joints_location[inverted_joint_names["R_Wrist"]]

        # cast a ray with origin in the joints
        rsray_origins = np.array(rshoulder_xyz)
        reray_origins = np.array(relbow_xyz)
        rwray_origins = np.array(rwrist_xyz)
        #######################################################################
        # Direction vectors: points where the ray should be cast to.
        rsdirectionVector = np.array(
            [rshoulder_xyz[0], posedm.ybounds()[1], posedm.zbounds()[0]]
        )
        # The direction vectors have their origin at the corresponding wrist
        # joint (left/right) and are perpendicular to the analogous side of
        # the bounding box (YZ-plane).

        redirectionVector = np.array(
            [posedm.xbounds()[0], relbow_xyz[1], relbow_xyz[2]]
        )

        rwdirectionVector = np.array(
            [posedm.xbounds()[0], rwrist_xyz[1], rwrist_xyz[2]]
        )

        rwdirectionVector = np.array(
            [posedm.xbounds()[0], rwrist_xyz[1], rwrist_xyz[2]]
        )

        #######################################################################
        # Outside of mesh: cast to numpy
        rsray_direction_outside_mesh = np.copy(rsdirectionVector)
        reray_direction_outside_mesh = np.copy(redirectionVector)
        rwray_direction_outside_mesh = np.copy(rwdirectionVector)

        #######################################################################
        # Intersection with lines
        rspts = posedm.intersectWithLine(
            rsray_origins, rsray_direction_outside_mesh
        )
        repts = posedm.intersectWithLine(
            reray_origins, reray_direction_outside_mesh
        )
        rwpts = posedm.intersectWithLine(
            rwray_origins, rwray_direction_outside_mesh
        )

        #######################################################################
        trimesh_posedm = self.trimesh

        #######################################################################
        # We asume that the three points (shoulder, elbow and wrist are not
        # collinear)

        points = np.asanyarray([rspts[0], repts[0], rwpts[0]], dtype="float")

        p0, p1, p2 = points
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]
        vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]

        u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
        point = np.array(p0)
        normal = np.array(u_cross_v)

        right_arm_plane_origin = point
        right_arm_plane_normal = normal

        sslice = trimesh_posedm.section(
            plane_origin=right_arm_plane_origin,
            plane_normal=right_arm_plane_normal,
        )

        # print(type(mslice))
        # Can be used to debug by asserting the sum of the two subsurves (see
        # beneath) is close to this length.
        # print(type(mslice))
        arm_intersection_boundary_length = 0
        vertices = sslice.vertices
        if sslice is None:
            arm_intersection_boundary_length = 0
            # raise error because the plane does not intersects the mesh.
            raise Exception("The plane does not intersects the mesh")
        elif len(sslice.entities) > 1:
            # The plane intersects the mesh and returns more than one closed
            # curve. Therefore, we have to search for the right curve: the one
            # than contains the three shoulder, elbow and wrists landmarks. In the
            # praxis this can be an expensive operation. Thus, we assume that the
            # right curve is the one that contains at least one of the three
            # landmarks.
            for p in points:
                found = na.search_point_on_Path3D(sslice, p, atol=1e-2)
                if found is not None:
                    break
            if found is not None:
                # nodes contains segment connection information (topology)
                nodes = sslice.entities[found].nodes
                arm_intersection_boundary_length = sslice.entities[
                    found
                ].length(vertices)
            else:
                raise Exception("Right curve was not found")
        else:
            slice_2D, to_3D = sslice.to_planar()
            arm_intersection_boundary_length = slice_2D.length

            # nodes contains segment connection information (topology)
            nodes = sslice.entities[0].nodes
        #######################################################################

        # curve is the collection of ordered vertices (correspondingly segments)
        # that form the curve. In other words, if you stich the segments together
        # you get a polyline curve.
        curve = vertices[nodes]

        # Two points are right shoulder and right wrist points in the skin
        pointA = points[0]
        pointB = points[2]

        sA_index, sB_index = self.points_on_segment_query(
            curve, pointA, pointB
        )

        if sA_index is not None and sB_index is not None:

            # We already garantied that the points are in distint segments.
            # Now assume w.l.o.g. that point A is always in the segment with the
            # lowest index in the nodes array.

            if sA_index > sB_index:
                sA_index, sB_index = sB_index, sA_index
                pointA, pointB = pointB, pointA

            lA, lB = curve[sA_index], curve[sB_index]

            # now add the vertices and the segments

            new_vertices = np.append(
                vertices, np.vstack((lA[0], lB[0])), axis=0
            )
            pointA_vertex_index = len(vertices)
            pointB_vertex_index = pointA_vertex_index + 1

            # we have to append 4 segments in total because every point divide one
            # segment in two parts
            node1 = np.array([nodes[sA_index][0], pointA_vertex_index])
            node2 = np.array([pointA_vertex_index, nodes[sA_index][1]])
            node3 = np.array([nodes[sB_index][0], pointB_vertex_index])
            node4 = np.array([pointB_vertex_index, nodes[sB_index][1]])

            # replace the two nodes by the four nodes at the right place
            subcurveA = np.append(
                node1.reshape(1, 2), nodes[:sA_index], axis=0
            )
            subcurveA = np.append(subcurveA, nodes[sB_index + 1 :], axis=0)
            subcurveA = np.append(subcurveA, node4.reshape((1, 2)), axis=0)

            subcurveB = np.append(
                node2.reshape(1, 2), nodes[sA_index + 1 : sB_index], axis=0
            )
            subcurveB = np.append(subcurveB, node3.reshape((1, 2)), axis=0)

            pathA = new_vertices[subcurveA]
            pathB = new_vertices[subcurveB]

            # calculate right_arm length selecting the right subcurve
            right_arm_length = None

            if self.arm_length_subcurve(pathA, pathB, side="right") == 1:
                right_arm_length = self.calculate_curve_length(pathA)
                self.ral_subcurve = pathA
            else:
                right_arm_length = self.calculate_curve_length(pathB)
                self.ral_subcurve = pathB
        else:
            raise Exception("Intersection points where not found in the curve")

        self.ral = right_arm_length
        return self.ral

    def left_arm_lenth(self):
        # return the cachedone
        if self.lal is not None:
            return self.lal

        joints_location = (
            self.human_model.return_posed_template_joint_locations()
        )
        posedm = self.mesh

        inverted_joint_names = dict(
            (v, k)
            for k, v in self.human_model.mean_template_shape.joint_names.items()
        )

        lshoulder_xyz = joints_location[inverted_joint_names["L_Shoulder"]]
        lelbow_xyz = joints_location[inverted_joint_names["L_Elbow"]]
        lwrist_xyz = joints_location[inverted_joint_names["L_Wrist"]]

        # cast a ray with origin in the joints
        lsray_origins = np.array(lshoulder_xyz)
        leray_origins = np.array(lelbow_xyz)
        lwray_origins = np.array(lwrist_xyz)

        #######################################################################
        # Direction vectors: points where the ray should be cast to.
        lsdirectionVector = np.array(
            [lshoulder_xyz[0], posedm.ybounds()[1], posedm.zbounds()[0]]
        )

        # The direction vectors have their origin at the corresponding wrist
        # joint (left/right) and are perpendicular to the analogous side of
        # the bounding box (YZ-plane).
        ledirectionVector = np.array(
            [posedm.xbounds()[1], lelbow_xyz[1], lelbow_xyz[2]]
        )
        lwdirectionVector = np.array(
            [posedm.xbounds()[1], lwrist_xyz[1], lwrist_xyz[2]]
        )
        #######################################################################
        # Outside of mesh: cast to numpy
        lsray_direction_outside_mesh = np.copy(lsdirectionVector)
        leray_direction_outside_mesh = np.copy(ledirectionVector)
        lwray_direction_outside_mesh = np.copy(lwdirectionVector)

        #######################################################################
        # Intersection with lines

        lspts = posedm.intersectWithLine(
            lsray_origins, lsray_direction_outside_mesh
        )

        lepts = posedm.intersectWithLine(
            leray_origins, leray_direction_outside_mesh
        )

        lwpts = posedm.intersectWithLine(
            lwray_origins, lwray_direction_outside_mesh
        )

        trimesh_posedm = self.trimesh

        #######################################################################
        # We asume that the three points (shoulder, elbow and wrist are not
        # collinear)

        points = np.asanyarray([lspts[0], lepts[0], lwpts[0]], dtype="float")

        p0, p1, p2 = points
        x0, y0, z0 = p0
        x1, y1, z1 = p1
        x2, y2, z2 = p2

        ux, uy, uz = u = [x1 - x0, y1 - y0, z1 - z0]
        vx, vy, vz = v = [x2 - x0, y2 - y0, z2 - z0]

        u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
        point = np.array(p0)
        normal = np.array(u_cross_v)

        left_arm_plane_origin = point
        left_arm_plane_normal = normal

        sslice = trimesh_posedm.section(
            plane_origin=left_arm_plane_origin,
            plane_normal=left_arm_plane_normal,
        )

        # print(type(mslice))
        # Can be used to debug by asserting the sum of the two subsurves (see
        # beneath) is close to this length.
        # print(type(mslice))
        arm_intersection_boundary_length = 0
        vertices = sslice.vertices
        if sslice is None:
            arm_intersection_boundary_length = 0
            # raise error because the plane does not intersects the mesh.
            raise Exception("The plane does not intersects the mesh")
        elif len(sslice.entities) > 1:
            # The plane intersects the mesh and returns more than one closed
            # curve. Therefore, we have to search for the right curve: the one
            # than contains the three shoulder, elbow and wrists landmarks. In the
            # praxis this can be an expensive operation. Thus, we assume that the
            # right curve is the one that contains at least one of the three
            # landmarks.
            for p in points:
                found = na.search_point_on_Path3D(sslice, p, atol=1e-2)
                if found is not None:
                    break
            if found is not None:
                # nodes contains segment connection information (topology)
                nodes = sslice.entities[found].nodes
                arm_intersection_boundary_length = sslice.entities[
                    found
                ].length(vertices)
            else:
                raise Exception("Right curve was not found")
        else:
            slice_2D, to_3D = sslice.to_planar()
            arm_intersection_boundary_length = slice_2D.length

            # nodes contains segment connection information (topology)
            nodes = sslice.entities[0].nodes

        #######################################################################

        # curve is the collection of ordered vertices (correspondingly
        # segments) that form the curve. In other words, if you stich the
        # segments together you get a polyline curve.
        curve = vertices[nodes]

        # Two points are left shoulder and left wrist points in the skin
        pointA = points[0]
        pointB = points[2]

        sA_index, sB_index = self.points_on_segment_query(
            curve, pointA, pointB
        )
        if sA_index is not None and sB_index is not None:

            # We already garantied that the points are in distint segments.
            # Now assume w.l.o.g. that point A is always in the segment with
            # the lowest index in the nodes array.

            if sA_index > sB_index:
                sA_index, sB_index = sB_index, sA_index
                pointA, pointB = pointB, pointA

            lA, lB = curve[sA_index], curve[sB_index]

            # now add the vertices and the segments

            new_vertices = np.append(
                vertices, np.vstack((lA[0], lB[0])), axis=0
            )
            pointA_vertex_index = len(vertices)
            pointB_vertex_index = pointA_vertex_index + 1

            # we have to append 4 segments in total because every point divide
            # one segment in two parts
            node1 = np.array([nodes[sA_index][0], pointA_vertex_index])
            node2 = np.array([pointA_vertex_index, nodes[sA_index][1]])
            node3 = np.array([nodes[sB_index][0], pointB_vertex_index])
            node4 = np.array([pointB_vertex_index, nodes[sB_index][1]])

            # replace the two nodes by the four nodes at the right place
            subcurveA = np.append(
                node1.reshape(1, 2), nodes[:sA_index], axis=0
            )
            subcurveA = np.append(subcurveA, nodes[sB_index + 1 :], axis=0)
            subcurveA = np.append(subcurveA, node4.reshape((1, 2)), axis=0)

            subcurveB = np.append(
                node2.reshape(1, 2), nodes[sA_index + 1 : sB_index], axis=0
            )
            subcurveB = np.append(subcurveB, node3.reshape((1, 2)), axis=0)

            pathA = new_vertices[subcurveA]
            pathB = new_vertices[subcurveB]

            # calculate left_arm length selecting the right subcurve
            left_arm_length = None

            if self.arm_length_subcurve(pathA, pathB, side="left") == 1:
                left_arm_length = self.calculate_curve_length(pathA)
                self.lal_subcurve = pathA
            else:
                left_arm_length = self.calculate_curve_length(pathB)
                self.lal_subcurve = pathB
        else:
            raise Exception("Intersection points where not found in the curve")

        self.lal = left_arm_length
        return self.lal

    def inseam(self):
        # return the cached-one
        if self.ins is not None:
            return self.ins
        joints_location = (
            self.human_model.return_posed_template_joint_locations()
        )

        inverted_joint_names = dict(
            (v, k)
            for k, v in self.human_model.mean_template_shape.joint_names.items()
        )
        posedm = self.mesh

        pelvis_xyz = joints_location[inverted_joint_names["Pelvis"]]
        # cast a ray with origin in the pelvis and direction to the floor
        crotchray_origins = np.array(pelvis_xyz)

        # The direction vectors have their origin at the pelvis joint (to
        # simulate crotch) and are perpendicular to the floor (XZ-plane).
        floor_xyz = [pelvis_xyz[0], posedm.ybounds()[0], pelvis_xyz[2]]
        crotchdirectionVector = np.array(floor_xyz)

        cray_direction_outside_mesh = np.copy(crotchdirectionVector)

        #######################################################################
        # Intersection with lines
        cpts = posedm.intersectWithLine(
            crotchray_origins, cray_direction_outside_mesh
        )

        # In the case of crotch point the ray can intersects the mesh in more
        # than one point. We asumme the crotch to begin at perineum and
        # therefore, at the last intersection point.
        crotch_xyz = np.array(cpts[-1])
        floor_xyz = np.array(floor_xyz)
        # inseam (crotch height in our case) is the euclidean distance between
        # the above defined crotch point and the floor.
        inseam = np.linalg.norm(crotch_xyz - floor_xyz)
        self.ins = inseam
        self.ins_line = np.array([crotch_xyz, floor_xyz])
        return self.ins

    def height(self):
        # Height is the distance from the top of the head to he floor. For
        # implementation details see the description on
        # calulate_stature_from_mesh()
        self.human_model.calculate_stature_from_mesh()
        # This is the first human body dimension we can calculate
        height = self.human_model.dimensions.stature
        return height

    def return_shoulder_width_subcurve(self):
        return self.sw_subcurve

    def return_right_arm_subcurve(self):
        return self.ral_subcurve

    def return_left_arm_subcurve(self):
        return self.lal_subcurve

    def return_inseam_line(self):
        return self.ins_line


if __name__ == "__main__":

    import torch

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
        # put here the item id (which is probably not the subject number)
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

    meshpath = meshi["pose0_file"]
    gender = meshi["person_gender"][0]
    sharmean = Sharmeam()
    sharmean.clear()
    sharmean.mesh_path(meshpath[0])
    sharmean.load_mesh(basicModel, gender=gender, pose=2)
    sharmean.load_trimesh()
    # shoulder width
    sw = sharmean.shoulder_width()
    ral = sharmean.right_arm_lenth()
    lal = sharmean.left_arm_lenth()
    ins = sharmean.inseam()
    height = sharmean.height()
    text_info = (
        "HBD for {}:\n"
        "Shoulder width (green subcurve) is {:.2f} cm.\n"
        "Right arm length (magenta subcurve) is {:.2f} cm.\n"
        "Left arm length (blue subcurve) is {:.2f} cm.\n"
        "Inseam (cyan line) is {:.2f} cm.\n"
        "Height is {:.2f} m.\n"
    ).format(
        subject["subject_string"],
        sw * 100,
        ral * 100,
        lal * 100,
        ins * 100,
        height,
    )
    print(text_info)
