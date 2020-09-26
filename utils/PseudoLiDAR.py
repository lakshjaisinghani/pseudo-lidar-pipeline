#!/usr/bin/env python
import numpy as np


class PseudoLiDAR:

    def __init__(self, calib_dir, sparsity):
        
        self.T, self.P = self.get_trans_proj(calib_dir)
        self.sparsity = sparsity

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        return data

    def cart2hom(self, pts_3d):
            ''' Input: nx3 points in Cartesian
                Oupput: nx4 points in Homogeneous by pending 1
            '''
            n = pts_3d.shape[0]
            pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
            return pts_3d_hom

    def inverse_rigid_trans(self, Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
        return inv_Tr

    def get_trans_proj(self, calib_dir):

        calib_velo_to_cam = self.read_calib_file(calib_dir + "calib_velo_to_cam.txt")
        calib_cam_to_cam  = self.read_calib_file(calib_dir + "calib_cam_to_cam.txt")

        r = calib_velo_to_cam["R"].reshape(3, 3)
        t = calib_velo_to_cam["T"].reshape(3, 1)

        # make transformation matrix
        T = np.concatenate((r, t),axis = 1)
        T = np.vstack([T, [0, 0, 0, 1]])

        # get cam-to-cam projection matrix
        P = calib_cam_to_cam["P_rect_02"].reshape(3, 4)

        # TODO: research -> not sure
        # P = P[:3, :3]
        # P = np.matmul(P, np.eye(3, 4))

        return T, P

    def project_PL(self, depth_img):
        
        rows, cols = depth_img.shape
        c, r = np.meshgrid(np.arange(cols), np.arange(rows))
        points = np.stack([c, r, depth_img])
        points = points.reshape((3, -1))
        uv_depth = points.T

        # Camera intrinsics and extrinsics
        c_u = self.P[0, 2]
        c_v = self.P[1, 2]
        f_u = self.P[0, 0]
        f_v = self.P[1, 1]
        b_x = self.P[0, 3] / (-f_u)  # relative
        b_y = self.P[1, 3] / (-f_v)

        # image to cam transform
        n = uv_depth.shape[0]
        x = ((uv_depth[:, 0] - c_u) * uv_depth[:, 2]) / f_u + b_x
        y = ((uv_depth[:, 1] - c_v) * uv_depth[:, 2]) / f_v + b_y
        pts_3d_cam = np.zeros((n, 3))
        pts_3d_cam[:, 0] = x
        pts_3d_cam[:, 1] = y
        pts_3d_cam[:, 2] = uv_depth[:, 2]


        T_inv =self.inverse_rigid_trans(self.T)

        pts_3d_cam = self.cart2hom(pts_3d_cam)  # nx4

        # cam to velodyne transform
        cloud = np.matmul(pts_3d_cam, np.transpose(T_inv))

        max_high = 1 # meters

        valid = (cloud[:, 0] >= 0) & (cloud[:, 2] < max_high)
        cloud  = cloud[valid]

        if self.sparsity:
            return cloud[0::self.sparsity]
        else:
            return cloud