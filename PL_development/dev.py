#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

class Transformer:

    def __init__(self, data_dir):

        #data directories
        self.CALIB_DIR = data_dir
        self.PNTS_DIR  = data_dir + "point_cloud.bin"

        # image constraints
        self.width = 1242
        self.height = 375

    def get_dirs(self):
        return self.VELOD_DIR 
    
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

    def get_trans_proj(self):

        calib_velo_to_cam = self.read_calib_file(self.CALIB_DIR + "calib_velo_to_cam.txt")
        calib_cam_to_cam  = self.read_calib_file(self.CALIB_DIR + "calib_cam_to_cam.txt")

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

    def mse(self, im1, im2):
        err  = np.sum((im1 + im2) ** 2)
        err /= float(im1.shape[0] * im1.shape[1])

        return err

    def test_3d_to_2d(self, points):

        T, P = self.get_trans_proj()

        # scan velodyne points
        if points is None:
            points = (np.fromfile(self.PNTS_DIR, dtype=np.float32)).reshape(-1, 4)
            points = points[:, :3]
            print(points.shape)
        else:
            points = points[:, :3]

        
        pixel_coords = []
        depth_array = np.zeros((self.width, self.height))
        tmp_pnts = []

        # transform points
        for pnt in points:

            # calculate velo distance
            x = pnt[0]
            y = pnt[1]
            z = pnt[2]
            dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            
            # make point homogeneous
            pnt = pnt.reshape(3, 1)
            pnt = np.vstack((pnt, [1]))

            # velo-to-cam
            xyz_pnt = np.matmul(T, pnt)

            # cam-to-cam
            uv_pnt  = np.matmul(P, xyz_pnt)
            uv_pnt = (uv_pnt/uv_pnt[2]).reshape(1, 3)[0] # scale adjustment

            if uv_pnt[0] >= 0 and uv_pnt[0] < self.width and \
                        uv_pnt[1] >= 0 and uv_pnt[1] < self.height and dist <= 120 and x > 0:
                
                # TODO: optimise and calculate velo distance
                # z = xyz_pnt[0]
                # y = xyz_pnt[1]
                # x = xyz_pnt[2]
                # dist_cam = np.sqrt(x ** 2 + y ** 2 + z ** 2)

                depth_array[int(uv_pnt[0])][int(uv_pnt[1])] = xyz_pnt[2]


        # depth
        depth_img = np.transpose(depth_array)

        return depth_img
    
    def plot(self, img):
        # display result image
        fig=plt.figure(figsize=(13, 3))
        fig.add_subplot(2, 1, 1)
        plt.imshow(img, cmap='gist_gray')
        plt.title("Projected points ")

        fig.add_subplot(2, 1, 2)
        plt.imshow(img, cmap='gist_gray', interpolation='nearest')
        plt.title("Projected points (interpolated)")
        plt.show()

    
