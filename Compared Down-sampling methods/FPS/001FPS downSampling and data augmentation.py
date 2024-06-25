# -*- coding: utf-8 -*-

import numpy as np
import math
import time
import sys
import os
import numba
from numba import jit


def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist

##Class: FarthestSampler
class FarthestSampler:
    def __init__(self):
        pass
    def _calc_distances(self, p0, points):
        return ((p0 - points) ** 2).sum(axis=1)  #Returns the sum of squared Euclidean distances between the set of sample points and other points
    def _call__(self, pts, k):  #PTS is the input point cloud,  K is the number of downsampling
        farthest_pts = np.zeros((k, 4), dtype=np.float32) #  The first three columns are coordinates xyz, and the fourth column is labels
        farthest_pts[0] = pts[np.random.randint(len(pts))]
        distances = self._calc_distances(farthest_pts[0,:3], pts[:,:3])
        for i in range(1, k):
            farthest_pts[i] = pts[np.argmax(distances)]
            distances = np.minimum( distances, self._calc_distances(farthest_pts[i,:3], pts[:,:3]))
        return farthest_pts

'''
    input:  point clouds in the format of the TXT by xyzLabel
	output:  TXT dataset after FPS downSampling  in the form of XYZLabel
'''


if __name__ == '__main__':
    path = r'.../merge'  #your merged points directory path
    saved_path = '..../FPS_TXT/'  #save path of FPS points
    Filelist = get_filelist(path)
    for z in range(0, 10): # do 10 times data augmentation
        for file in Filelist:
            print(file)
            points = np.loadtxt(file,dtype=float,delimiter=' ')

            pcd_array=np.array(points)
            print("pcd_array.shape:", pcd_array.shape)
            sample_count = 1024  # Fixed number of points after FPS downsampling


            # Perform FPS Downsampling for center point set and edge point set respectively
            FPS = FarthestSampler()  #
            sample_points =FPS._call__(pcd_array[:, 0:4],sample_count ) #FPS downSampling
            print("sample_points.shape:", sample_points.shape)

            #Obtain the file name under the path and remove the suffix
            file_name = file.split("\\")[-1]
            file_nameR = file_name[0:len(file_name) - 9]
            print("file_nameR:", file_nameR)

            np.savetxt(saved_path + file_nameR + str(z) + ".txt", sample_points, fmt='%.6f')  # Save 3DEPS TXT in the form of XYZLabel
            print(saved_path + file_nameR + str(z) + ".txt")
