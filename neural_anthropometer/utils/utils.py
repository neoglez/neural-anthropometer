import numpy as np
from scipy.spatial import cKDTree
from trimesh.path.segments import split as split_segments
import trimesh.util as util


def pose2(batch_size=1):
    """
    returns pose (thetas) for the pose 1: arms lowered.

    Returns
    -------
    numpy ndarray of size batch_sizeX72

    """
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

    poses = np.zeros((batch_size, 72))
    poses[:] = pose
    return poses


def shoulder_width_subcurve(subcurveA, subcurveB):
    """
    Returns which of the two subcurves represents the shoulder width. We assume
    the mesh has LSA orientation, therefore the shoulder width is represented
    by the curve with minimum z coordinate.

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
        1 if the subcurve A represents the shoulder width or 2 if the subcurveB
        represents the shoulder width

    """
    return (
        1
        if (
            np.vstack((subcurveA[:, 0, :], subcurveA[:, 1, :])).min(axis=0)[2]
            < np.vstack((subcurveB[:, 0, :], subcurveB[:, 1, :])).min(axis=0)[
                2
            ]
        )
        else 2
    )


def arm_length_subcurve(subcurveA, subcurveB, side="right"):
    """
    Returns which of the two subcurves represents the corresponding (right or
    left) arm length. We assume
    the mesh has LSA orientation, therefore the arm legth is represented
    by the curve with minimum (side = right)  or maximum (side = "left")
    x coordinate.

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


def calculate_curve_length(curve):
    """
    Calculate the length of a curve

    Parameters
    ----------
    curve : numpy ndarray of shape mx2x3
        The polygonal curve to be splitted. The curve is represented as m lines
        with start and end point.

    Returns
    -------
    Float

    """
    return np.linalg.norm(curve[:, 0, :] - curve[:, 1, :], axis=1).sum()


def points_on_segment_query(curve, pointA, pointB):
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


def search_points_curve(curve, pointA, pointB, atol=1e-5):
    """
    search pointA and pointB on the polyline. If both points are found, the
    segment indexes containing these points are returned

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
    tuple of length 2,
        segment index containing point A, segment index containing point B
    """
    segment_containing_A = None
    segment_containing_B = None

    # search in which line (line index) is point A
    for i, line in enumerate(curve):
        # Query using split. If the point lies on the line two segments are
        # returned otherwise the original segments are returned.
        # reshape the segment and the point
        segment = line[np.newaxis, :, :]
        pA = np.array([pointA])
        pB = np.array([pointB])

        if split_segments(segment, pA, atol).shape[0] == 2:
            segment_containing_A = i
        if split_segments(segment, pB, atol).shape[0] == 2:
            segment_containing_B = i
        # early return
        if (
            segment_containing_A is not None
            and segment_containing_B is not None
        ):
            return (segment_containing_A, segment_containing_B)

    if segment_containing_A is None:
        return (None, segment_containing_B)
    if segment_containing_B is None:
        return (segment_containing_A, None)
    return (segment_containing_A, segment_containing_B)


def search_point_on_Path3D(curve, pointA, atol=1e-5):
    """
    Search pointA on a Path3D curve.

    Parameters
    ----------
    curve : Object of type trimesh.path.path.Path,
            see
            https://trimsh.org/trimesh.path.path.html#trimesh.path.path.Path3D
            The polygonal curve to be splitted. The curve may have more than
            one entity, which usually means that this object contains more than
            one closed curve.
    pointA : numpy ndarray of shape 1x3
            Point A to search for.

    Returns
    -------
    Interger: entity index that contains the point. If the point is not found,
    None is returned.
    """

    # search in which line (line index) is point A
    if len(curve.entities) == 1:
        if not isinstance(curve.entities):
            raise Exception(
                (
                    "Curve entities must be of type "
                    "trimesh.path.entities.Line"
                )
            )
        raise Exception(("Curve has only one entity"))
    for i, line in enumerate(curve.entities):

        nodes = line.nodes
        # now we have the segments: every segment has shape (2, 3), e.g., begin
        # and end 3D points.
        vertices = curve.vertices[nodes]
        point = np.asanyarray(pointA, dtype=np.float64)
        segments = np.asanyarray(vertices, dtype=np.float64)
        # reshape to a flat 2D (n, dimension) array
        seg_flat = segments.reshape((-1, segments.shape[2]))
        # find the length of every segment
        length = ((segments[:, 0, :] - segments[:, 1, :]) ** 2).sum(
            axis=1
        ) ** 0.5

        # a mask to remove segments we split at the end
        keep = np.ones(len(segments), dtype=bool)
        # append new segments to a list
        new_seg = []
        # find the distance from point to every segment endpoint
        pair = ((seg_flat - point) ** 2).sum(axis=1).reshape((-1, 2)) ** 0.5

        # point is on a segment if it is on a vertex
        # or the sum length is equal to the actual segment length
        on_seg = np.logical_and(
            util.isclose(length, pair.sum(axis=1), atol=atol),
            util.isclose(pair, 0.0, atol=atol).any(axis=1),
        )
        if on_seg.any():
            # found!
            return i
    # unfortunately not found...
    return None
