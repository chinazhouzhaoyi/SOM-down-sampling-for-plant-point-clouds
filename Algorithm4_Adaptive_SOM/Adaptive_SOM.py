"""
paper: Unsupervised shape-aware SOM down-sampling for plant point clouds
file:  Algorithm4_Adaptive_SOM/Adaptive_SOM.py
about: Adaptive shape-aware down-sampling codes
author: Zhaoyi Zhou
date: 2024-5-18
"""
import os
import numpy as np
import warnings
import pickle
import glob
from tqdm import tqdm
import open3d as o3d
import sys
sys.path.insert(0, '../Algorithm1_1D_SOM')
sys.path.insert(0, '../Algorithm2_2D_SOM')
sys.path.insert(0, '../Algorithm3_3D_SOM')
import som1D 
import som2D 
import som3D 
from collections import Counter
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

def SOM1D_point_sample(data, resampleNum, sigma):
    """
    Perform 1D SOM (Self Organizing Mapping) point sampling on the given dataset and return the sampled weight array.

    Args:
    Data (np. ndarray): The input point cloud data has a shape of [N, D], where N is the number of points and D is the dimension of each point.
    ResampleNum (int): The number of points that need to be sampled.
    sigma (float): The sigma parameter of the Gaussian kernel.

    Returns:
    Np.ndarray: The sampled weight array, with a shape of (resampleNum, D).
    """
    organs = []
    organs.append(data)
    assignNum = [resampleNum]
    #Use a dictionary to receive the weight vectors corresponding to the competition layer neurons obtained,
    # as well as the centroid obtained from the mean
    accept_dict = som1D.getSkeleton(organs, assignNum, sigma)  
    weight_centroid = accept_dict["weights"] 
    weights = []
    for o, i in enumerate(weight_centroid):
        for s in i: 
            weights.append(s[0])
    weights = np.array(weights)
    # print()
    return weights

def SOM2D_point_sample(data, resampleNum, sigma):
    """
    Perform 2D SOM (Self Organizing Mapping) point sampling on the given dataset and return the sampled weight array.

    Args:
    Data (np. ndarray): The input point cloud data has a shape of [N, D], where N is the number of points and D is the dimension of each point.
    ResampleNum (int): The number of points that need to be sampled.
    sigma (float): The sigma parameter of the Gaussian kernel.

    Returns:
    Np.ndarray: The sampled weight array, with a shape of (resampleNum, D).
    """
    organs = []
    organs.append(data)
    assignNum = [resampleNum]
    print("AssignNum:", assignNum)
    weight_centroid = som2D.getSkeleton(organs, assignNum, sigma)  #
    weights = []
    #Weights are saved in the order of competing layer neurons
    for o, i in enumerate(weight_centroid):
        for s in i:  
            for p in s: #p returns xyz
                weights.append(p)
    weights = np.array(weights)
    return weights

def SOM3D_point_sample(data, resampleNum, sigma):
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
    return weights

def ComPerCent(new_counts, nodesNum):
    """
    Allocate nodes evenly based on the number of nodes new_counts, and return a list of allocated nodes.

    Args:
    New_counts: A list of the number of ins points, of type int.
    NodesNum: Total number of nodes, type int.

    Returns:
    LeafPer: A list of allocated nodes of type np.ndarray, with elements of type int.

    """
    leafnodesNum = nodesNum
    new_counts_array = new_counts
    ans = sum(new_counts_array)
    new_counts_array = np.array(new_counts_array)
    leafPer = (leafnodesNum * new_counts_array / ans)
    # Preliminary adjustment of ins shape allocation nodes
    for i in range(len(leafPer)):
        leafPer[i] = (round(leafPer[i])) 
        if leafPer[i] < 3:
            leafPer[i] = 3

    sum_leafNode = sum(leafPer)
    max_index = np.argmax(leafPer)
    max_number = np.max(leafPer)

    if sum_leafNode < leafnodesNum:
        leafPer[max_index] = leafPer[max_index] + leafnodesNum - sum_leafNode

    else:
        leafPer[max_index] = leafPer[max_index] + (leafnodesNum - sum_leafNode)
    leafPer = leafPer.astype(np.int32) 
    return leafPer

def adjustPerNum(orangPerNum, olist):
    """
    Computer the square and cubic values upwards for 2D and 3D structures, respectively
    Args:
    OrangPerNum (list): A list, where each element represents a number of ins.
    Olist (list): A list
    List: The adjusted list of allocated nodes.
    """
    adjustNum = orangPerNum
    for i in range(len(adjustNum)):
        if olist[i][0][-1] == 2:  #2D structure
            if np.sqrt(adjustNum[i]) % 1 != 0:
                Num = int(np.sqrt(adjustNum[i])) + 1
                adjustNum[i] = Num * Num
        if olist[i][0][-1] == 3: #3D structure
            if np.power(adjustNum[i], 1 / 3) % 1 != 0:
                Num = int(np.power(adjustNum[i], 1 / 3)) + 1
                adjustNum[i] = Num * Num * Num
    adjustNum = adjustNum.astype(np.int32)
    return adjustNum


class plantDataLoader():
    def __init__(self, args):
        self.data_path = args.data_path
        self.output_path = args.output_path
        self.npoints = args.num_point
        self.sigma = args.sigma
        self.SOMform = args.use_som_sample
        self.suffix = args.suffix
        self.normalize = args.normalize
        

        self.datapath = get_file_names(self.data_path)
        print('The size of %s data is %d' % ("dataset", len(self.datapath)))
        
        for index in tqdm(range(len(self.datapath)), total=len(self.datapath)):
            fn = self.datapath[index] 
            dataname = fn.split('/')[-1]
            savename = dataname.split('.')[0] + '.txt'
            sn = os.path.join(self.output_path, savename) 
            suffix = dataname.split('.')[-1]
            if suffix == "txt": P = np.loadtxt(fn).astype(np.float32)
            elif suffix == "pcd": P = read_pcd(fn) 
            else: print(print("Error: suffix not support"))
            P = np.array(P) # x y z insLabel shapeLabel
            h, w = P.shape
            if self.normalize: Pnorm = pc_normalize(P[:, :3])
            else: Pnorm = P[:, :3]
            point_set = np.zeros((h, w))
            point_set[:, :3] = Pnorm
            point_set[:, 3:] = P[:, 3:]
            index = np.lexsort([P[:, -2]])
            sorted_data = P[index, :]  #
            insLabel = sorted_data[:, -2]
            counts = Counter(insLabel)
            o_list = []
            new_counts = []
            Num = 0
            for i in range(len(counts)): 
                # print(counts[i])
                if counts[i] > 10:  # Exclude noise insLabel
                    new_counts.append(counts[i])
                    o_list.append(sorted_data[Num:Num + counts[i], :])
                Num += counts[i]
            orangPerNum = ComPerCent(new_counts, self.npoints)  # 
            print("OrangPerNum:", orangPerNum)
            adjustNum = adjustPerNum(orangPerNum, o_list)# Computer the square and cubic values upwards for 2D and 3D structures, respectively
            print("Adjust orangPerNum:", adjustNum)
            orangPerNum = ComPerCent(new_counts, self.npoints)
            
            # The sampling results include xyz coordinates and ins labels
            if self.SOMform:
                    point_sample=np.zeros((self.npoints,4))
                    if o_list[0][0][-1] == 1: Som_sample = SOM1D_point_sample(o_list[0][:, :3], adjustNum[0], self.sigma)
                    elif o_list[0][0][-1] == 2: Som_sample = SOM2D_point_sample(o_list[0][:, :3], adjustNum[0], self.sigma)
                    elif o_list[0][0][-1] == 3: Som_sample = SOM3D_point_sample(o_list[0][:, :3], adjustNum[0], self.sigma)
                    if Som_sample.shape[0] > orangPerNum[0]:
                        Som_sample = farthest_point_sample(Som_sample, orangPerNum[0])
                    if(len(o_list) > 1):
                        for o in range(1, len(o_list)):
                            if o_list[o][0][-1] == 1: Som_s = SOM1D_point_sample(o_list[o][:, :3], adjustNum[o], self.sigma)
                            elif o_list[o][0][-1] == 2: Som_s = SOM2D_point_sample(o_list[o][:, :3], adjustNum[o], self.sigma)
                            elif o_list[o][0][-1] == 3: Som_s = SOM3D_point_sample(o_list[o][:, :3], adjustNum[o], self.sigma)
                            if Som_s.shape[0] > orangPerNum[o]:
                                Som_s = farthest_point_sample(Som_s, orangPerNum[o])
                            print(Som_s.shape)
                            Som_sample = np.concatenate((Som_sample, Som_s), axis=0)
                    point_sample[:, :3] = Som_sample
                    print("Som_sample.shape: ", Som_sample.shape)
                    for j in range(self.npoints):
                        Dstore = []
                        sampler = Som_sample[j, :3]
                        minID=np.argmin(np.sum((point_set[:, :3]-sampler)**2, axis=1)) 
                        # Take the label of the nearest point in the original point cloud corresponding to the point sampled 
                        # by SOM as the label of the sampled point
                        point_sample[j, -1] = point_set[minID, -1]
                    print('The shape after sampling:', point_sample.shape)
            else:
                    #FPS sampling
                    Fps_sample = farthest_point_sample(o_list[0][:, :4], orangPerNum[0])
                    if(len(o_list) > 1):
                        for o in range(1, len(o_list)):
                            Fps_s = farthest_point_sample(o_list[o][:, :4], orangPerNum[o])
                            Fps_sample = np.concatenate((Fps_sample, Fps_s), axis=0)
                    point_sample = Fps_sample
                    print('The shape after sampling:', point_sample.shape)
                    savename = dataname.split('.')[0] + 'FPS.txt'
                    sn = os.path.join(self.output_path, savename) 
        print("Savepath:", sn)
        np.savetxt(sn, point_sample, fmt='%.8f %.8f %.8f %d')

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu',  type=bool, default=True, help='use cpu mode')
    # parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--num_point', type=int, default=1024, help='Point sampler Number, recommended 1024')
    parser.add_argument('--sigma', type=float, default=1.2, help='Control parameters for activating domain radius, recommended range[1.2, 1.5]')
    parser.add_argument('--use_som_sample', type=bool, default=True, help='use som sampling instead of FPS sampling') 
    parser.add_argument('--data_path', type=str, default='./cluster_out', help='specify your point cloud path after clustering')
    parser.add_argument('--output_path', type=str, default='./output', help='specify your point cloud path')
    parser.add_argument('--suffix', type=str, default='txt', help='specify the suffix of your input file by txt or pcd')
    parser.add_argument('--normalize',  type=bool, default=False, help='Normalize point cloud with coordinate origin (0, 0, 0)')
    
    return parser.parse_args()


if __name__ == '__main__':
    args=parse_args()
    plantDataLoader(args=args)