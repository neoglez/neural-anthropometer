"""
@author: neoglez
Code adapted from
https://github.com/ahmedosman/STAR/blob/master/star/pytorch/star.py
"""

from __future__ import print_function, division
import numpy as np

try:
    import cPickle as pickle
except:
    import pickle as pickle

import torch
import torch.nn as nn


def quat_feat(theta):
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_sin * normalized, v_cos - 1], dim=1)
    return quat


def rodrigues(theta):
    """Convert axis-angle representation to rotation matrix.
    Args:
        theta: size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)
    return quat2mat(quat)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: size = [B, 4] 4 <===>(w, x, y, z)
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = (
        norm_quat[:, 0],
        norm_quat[:, 1],
        norm_quat[:, 2],
        norm_quat[:, 3],
    )

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack(
        [
            w2 + x2 - y2 - z2,
            2 * xy - 2 * wz,
            2 * wy + 2 * xz,
            2 * wz + 2 * xy,
            w2 - x2 + y2 - z2,
            2 * yz - 2 * wx,
            2 * xz - 2 * wy,
            2 * wx + 2 * yz,
            w2 - x2 - y2 + z2,
        ],
        dim=1,
    ).view(B, 3, 3)
    return rotMat


class SMPL(nn.Module):
    def __init__(self, model_file):
        super(SMPL, self).__init__()
        try:
            # Load pkl created in python 2.x with python 2.x
            smpl_model = pickle.load(open(model_file, "rb"))
        except:
            # Load pkl created in python 2.x with python 3.x
            smpl_model = pickle.load(open(model_file, "rb"), encoding="latin1")

        J_regressor = smpl_model["J_regressor"].tocoo()
        row = J_regressor.row
        col = J_regressor.col
        data = J_regressor.data
        i = torch.LongTensor([row, col])
        v = torch.FloatTensor(data)
        J_regressor_shape = [24, 6890]
        self.register_buffer(
            "J_regressor",
            torch.sparse.FloatTensor(i, v, J_regressor_shape).to_dense(),
        )
        self.register_buffer(
            "weights", torch.FloatTensor(smpl_model["weights"])
        )
        self.register_buffer(
            "posedirs", torch.FloatTensor(smpl_model["posedirs"])
        )
        self.register_buffer(
            "v_template", torch.FloatTensor(smpl_model["v_template"])
        )
        self.register_buffer(
            "shapedirs", torch.FloatTensor(np.array(smpl_model["shapedirs"]))
        )
        self.register_buffer(
            "faces", torch.from_numpy(smpl_model["f"].astype(np.int64))
        )
        self.register_buffer(
            "kintree_table",
            torch.from_numpy(smpl_model["kintree_table"].astype(np.int64)),
        )
        id_to_col = {
            self.kintree_table[1, i].item(): i
            for i in range(self.kintree_table.shape[1])
        }
        self.register_buffer(
            "parent",
            torch.LongTensor(
                [
                    id_to_col[self.kintree_table[0, it].item()]
                    for it in range(1, self.kintree_table.shape[1])
                ]
            ),
        )

        self.pose_shape = [24, 3]
        self.beta_shape = [10]
        self.translation_shape = [3]

        self.pose = torch.zeros(self.pose_shape)
        self.beta = torch.zeros(self.beta_shape)
        self.translation = torch.zeros(self.translation_shape)

        self.verts = None
        self.J = None
        self.R = None

        self.joints_idx = [e for e in np.arange(24)]

    def forward(self, pose, beta):
        device = pose.device
        batch_size = pose.shape[0]
        v_template = self.v_template[None, :]
        shapedirs = self.shapedirs.view(-1, 10)[None, :].expand(
            batch_size, -1, -1
        )
        beta = beta[:, :, None]
        v_shaped = torch.matmul(shapedirs, beta).view(-1, 6890, 3) + v_template
        # batched sparse matmul not supported in pytorch
        J = []
        for i in range(batch_size):
            J.append(torch.matmul(self.J_regressor, v_shaped[i]))
        J = torch.stack(J, dim=0)
        # input it rotmat: (bs,24,3,3)
        if pose.ndimension() == 4:
            R = pose
        # input it rotmat: (bs,72)
        elif pose.ndimension() == 2:
            pose_cube = pose.view(-1, 3)  # (batch_size * 24, 1, 3)
            R = rodrigues(pose_cube).view(batch_size, 24, 3, 3)
            R = R.view(batch_size, 24, 3, 3)
        else:
            raise Exception("Pose does not have the right shape. "
                             "Pose shape must (bs, 24, 3, 3) or (bs, 72). "
                             "Found pose with dimension {}".format(
                                 pose.ndimension()))
            
        I_cube = torch.eye(3)[None, None, :].to(device)
        # I_cube = torch.eye(3)[None, None, :].expand(
        #    theta.shape[0], R.shape[1]-1, -1, -1)
        lrotmin = (R[:, 1:, :] - I_cube).view(batch_size, -1)
        posedirs = self.posedirs.view(-1, 207)[None, :].expand(
            batch_size, -1, -1
        )
        v_posed = v_shaped + torch.matmul(posedirs, lrotmin[:, :, None]).view(
            -1, 6890, 3
        )
        J_ = J.clone()
        J_[:, 1:, :] = J[:, 1:, :] - J[:, self.parent, :]
        G_ = torch.cat([R, J_[:, :, :, None]], dim=-1)
        pad_row = (
            torch.FloatTensor([0, 0, 0, 1])
            .to(device)
            .view(1, 1, 1, 4)
            .expand(batch_size, 24, -1, -1)
        )
        G_ = torch.cat([G_, pad_row], dim=2)
        G = [G_[:, 0].clone()]
        for i in range(1, 24):
            G.append(torch.matmul(G[self.parent[i - 1]], G_[:, i, :, :]))
        G = torch.stack(G, dim=1)

        rest = torch.cat(
            [J, torch.zeros(batch_size, 24, 1).to(device)], dim=2
        ).view(batch_size, 24, 4, 1)
        zeros = torch.zeros(batch_size, 24, 4, 3).to(device)
        rest = torch.cat([zeros, rest], dim=-1)
        rest = torch.matmul(G, rest)
        G = G - rest
        T = (
            torch.matmul(
                self.weights, G.permute(1, 0, 2, 3).contiguous().view(24, -1)
            )
            .view(6890, batch_size, 4, 4)
            .transpose(0, 1)
        )
        rest_shape_h = torch.cat(
            [v_posed, torch.ones_like(v_posed)[:, :, [0]]], dim=-1
        )
        v = torch.matmul(T, rest_shape_h[:, :, :, None])[:, :, :3, 0]
        return v

    def get_joints(self, vertices):
        """
        This method is used to get the joint locations from the SMPL mesh
        Input:
            vertices: size = (B, 6890, 3)
        Output:
            3D joints: size = (B, 38, 3)
        """
        joints = torch.einsum("bik,ji->bjk", [vertices, self.J_regressor])
        joints = joints[:, [e for e in np.arange(24)]]
        return joints
