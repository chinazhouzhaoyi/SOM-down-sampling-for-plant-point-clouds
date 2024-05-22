"""
paper: Unsupervised shape-aware SOM down-sampling for plant point clouds
file:  Algorithm3_3D_SOM/SOM3D_Plant_data_making.py
about: main entrance for using 3D SOM for point cloud downsampling, targeting a single 3D topology shape
author: Zhaoyi Zhou
date: 2024-5-16
"""
import os
import numpy as np
import warnings
import pickle
import glob
from tqdm import tqdm
import open3d as o3d
import som3D 
import operator
from numpy import (array, unravel_index, nditer, linalg, random, subtract, max,
                   power, exp, pi, zeros, ones, arange, outer, meshgrid, dot,
                   logical_and, mean, std, cov, argsort, linspace, transpose,
                   einsum, prod, nan, sqrt, hstack, diff, argmin, multiply,
                   nanmean, nansum)
import argparse
warnings.filterwarnings('ignore')


def get_file_names(folder):
    '''
    Get an absolute path list of all files in the specified folder
    '''
    file_names = glob.glob(folder + '/*')
    return file_names

def pc_normalize(pc):
    """
    Normalize point cloud with coordinate origin (0, 0, 0)
    """
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def read_pcd(pcd_path):
    """
    Read point cloud data in PCD format.

    Args:
    Pcd_path (str): The path to the PCD file.

    Returns:
    List [np. ndarray]: A list composed of point cloud data, each element is a one-dimensional numpy array containing 4 elements, representing the x, y, z coordinates and labels of the point.

    Raises:
    AssertionError: If there is no POINTS line in the PCD file or the POINTS line cannot resolve the number of points, an exception is thrown.
    """
    lines = []
    num_points = None

    with open(pcd_path, 'r') as f:
        for line in f:
            lines.append(line.strip())
            if line.startswith('POINTS'):
                num_points = int(line.split()[-1])
    assert num_points is not None

    points = []
    for line in lines[-num_points:]:
        #Only the first 4 dimensions are taken, and the data in the 5th dimension can be omitted here
        x, y, z, label = list(map(float, line.split()))[:4]  
        points.append((np.array([x, y, z, label])))

    return points



def farthest_point_sample(point, npoint):  
    """
    select npoint points from the input point cloud using the farthest point sampling algorithm.

    Args:
    Point (np. ndarray): The input point cloud data has a shape of [N, D], where N is the number of points and D is the dimension of each point.
    Npoint (int): The number of points that need to be sampled.

    Returns:
    Np.ndarray: The sampled point cloud data has a shape of [npoint, D].
    """
    N, D = point.shape
    xyz = point[:, :3] #take out Rthe xyz coordinates
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)# randomly pick a number from(0,N)
    for i in range(npoint):
        centroids[i] = farthest   #centroids(npoint,) the index of the farthest point
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)  #the next index of the farthest point
    point = point[centroids.astype(np.int32)] #obtain sampling points based on the farthest point index
    return point

def SOM_point_sample(data, resampleNum, sigma):
    """
    Perform 3D SOM (Self Organizing Mapping) point sampling on the given dataset and return the sampled weight array.

    Args:
    Data (np. ndarray): The input point cloud data has a shape of [N, D], where N is the number of points and D is the dimension of each point.
    ResampleNum (int): The number of points that need to be sampled.
    sigma (float): The sigma parameter of the Gaussian kernel.

    Returns:
    Np.ndarray: The sampled weight array, with a shape of (resampleNum, D).
    """
    organs = []
    organs.append(data)
    num = int(np.power(resampleNum + 1, 1 / 3))
    if num * num * num < resampleNum:
                num = num + 1
                assignNum = [num * num * num]
    else: assignNum = [resampleNum]
    print("AssignNum:", assignNum)
    weight_centroid = som3D.getSkeleton(organs, num, sigma) 
    weights = []
    for o, i in enumerate(weight_centroid):
        for s in i:
            for p in s:
               for xyz in p:  # returns xyz
                weights.append(xyz)
    weights = np.array(weights)
    if assignNum[0] > resampleNum:
        weights = farthest_point_sample(weights, resampleNum)
    return weights


class plantDataLoader():
    def __init__(self, args):
        self.data_path = args.data_path
        self.output_path = args.output_path
        self.npoints = args.num_point
        self.sigma = args.sigma
        self.SOMform = args.use_som_sample
        self.label_sample = args.label_sample
        self.suffix = args.suffix
        self.expand_data= args.expand_data
        self.expand_num = args.expand_num
        self.normalize = args.normalize
        

        self.datapath = get_file_names(self.data_path)
        print('The size of %s data is %d' % ("dataset", len(self.datapath)))
        if self.expand_data == False:
            self.expand_num = 1
            self.expand_data = True
        if self.expand_data:
            for exNum in range(0, self.expand_num):
                for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
                        fn = self.datapath[index] 
                        dataname = fn.split('/')[-1]
                        if self.expand_num == 1: savename = dataname.split('.')[0] + '.txt'
                        else: savename = dataname.split('.')[0] + '_' + str(exNum) + '.txt'
                        sn = os.path.join(self.output_path, savename) 
                        print("Savepath:", sn)
                        suffix = dataname.split('.')[-1]
                        if suffix == "txt": P = np.loadtxt(fn).astype(np.float32)
                        elif suffix == "pcd": P = read_pcd(fn) 
                        else: print(print("Error: suffix not support"))
                        P = np.array(P)
                        P = P[:, :4]
                        h, w = P.shape
                        if self.normalize:
                            Pnorm = pc_normalize(P[:, :3])
                        else: Pnorm = P[:, :3]
                    
                        point_set = np.zeros((h, w))
                        point_set[:, :3] = Pnorm
                        point_set[:, -1] = P[:, -1]
                        if self.label_sample:# The sampling results include xyz coordinates and labels

                                    if self.SOMform:
                                    #3D SOM sampling
                                        point_sample=np.zeros((self.npoints,4))
                                        SOM_sample = SOM_point_sample(point_set[:,:3], self.npoints, self.sigma)
                                        print("SOM_sample.shape:", SOM_sample.shape)
                                        point_sample[:, :3] = SOM_sample
                                        # print(SOM_sample.shape)
                                        for j in range(self.npoints):
                                            Dstore = []
                                            sampler = SOM_sample[j, :3]
                                            minID=np.argmin(np.sum((point_set[:, :3]-sampler)**2, axis=1)) 
                                            # Take the label of the nearest point in the original point cloud corresponding to the point sampled 
                                            # by SOM as the label of the sampled point
                                            point_sample[j, -1] = point_set[minID, -1]
                                        print('The shape after sampling:', point_sample.shape)
                                    else:
                                    #FPS sampling
                                        point_sample = farthest_point_sample(point_set[:, :4], self.npoints)

                                        print('The shape after sampling:', point_sample.shape)
                                    print("Savepath:", sn)
                                    np.savetxt(sn, point_sample, fmt='%.8f %.8f %.8f %d')
                                    print("Save successfully")

                        else: #The sampling results only include xyz coordinates
                                    if self.SOMform:
                                            point_sample = SOM_point_sample(point_set[:, :3], self.npoints, self.sigma) 
                                            print('The shape after sampling:', point_sample.shape)
                                    else:
                                            point_sample = farthest_point_sample(point_set[:, :3], self.npoints)
                                            print('The shape after sampling:', point_sample.shape)
                                    print("Savepath", sn)
                                    np.savetxt(sn, point_sample, fmt='%.8f %.8f %.8f')
                                    print("Save successfully")

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu',  type=bool, default=True, help='use cpu mode')
    # parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=125, help='Point sampler Number, recommended [64 125]')
    parser.add_argument('--sigma', type=float, default=1.2, help='Control parameters for activating domain radius, recommended range[1.2, 1.5]')
    parser.add_argument('--use_som_sample', type=bool, default=True, help='use som sampling instead of FPS sampling') 
    parser.add_argument('--label_sample', type=bool, default=False, help='sample with instance label instead of merely points') 
    parser.add_argument('--data_path', type=str, default='./data', help='specify your point cloud path')
    parser.add_argument('--output_path', type=str, default='./output', help='specify your point cloud path')
    parser.add_argument('--suffix', type=str, default='txt', help='specify the suffix of your input file by txt or pcd')
    parser.add_argument('--normalize',  type=bool, default=False, help='Normalize point cloud with coordinate origin (0, 0, 0)')
    parser.add_argument('--expand_data',  type=bool, default=False, help='expand the sapmling number by repeating the samplig, recommended False')
    parser.add_argument('--expand_num',  type=int, default=10, help='repeating the samplig expaned times')
    
    return parser.parse_args()


if __name__ == '__main__':
    args=parse_args()
    plantDataLoader(args=args)