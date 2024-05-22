'''
paper: Unsupervised shape-aware SOM down-sampling for plant point clouds
file:  Algorithm3_3D_SOM/som3D.py
about: main entrance for using 3D SOM for point cloud downsampling, targeting a single 3D topology shape 
author: Zhaoyi Zhou
date: 2023-5-16
'''
import utils3D as utils
import numpy as np

def trainSom(data, x, y, z, epochs, sigma, verbose=False):
    """
    trains a som given the input data

    :param data: input, must be [Nx3]
    :param x: number of nodes in the x direction
    :param y: number of nodes in the y direction
    :param z: number of nodes in the z direction
    :param epochs: number of training iterations
    :param sigma: sigma for the gaussian kernel
    :param verbose: if True prints information during training
    :return: result of the som training
    """
    som = utils.MiniSom(x, y, z, 3, sigma, learning_rate=.5,
                         random_seed=1)  # 源码sigma=0.5, learning_rate=0.3, random_seed=1
    # som.random_weights_init(data)   #random init
    som.pca_weights_init(data)  # recommended
    som.train_random(data, epochs, verbose=verbose)
    weight = som._weights  
    #Output quantization error and topology error, if necessary
    #print("Quantization error: ", som.quantization_error(data))
    #print("Topology error: ", som.topographic_error(data))
    return weight

def getSkeleton(organs, Num, sigma):
    """
    for input 3D structure, returns the nodes of the skeleton
    
    Args:
    Organs: List [np. ndarray], a list containing multiple organ point clouds, each of which is a numpy array with a shape of (n, 3), where n is the number of points in the point cloud and 3 represents the three-dimensional coordinates of the points (x, y, z)
    AssignNum: List [int], an integer list of the same length as organs, representing how many skeleton nodes each organ's point cloud is assigned to
    Sigma: float, standard deviation of Gaussian kernel function
    Returns:
    Weight vectors as the sampled points
    """

    weights_vector = []
    for i, o in enumerate(organs): 
        oshape = o.shape
        weight = trainSom(o, x=Num, y=Num, z=Num, epochs=int(5 * oshape[0]), sigma=sigma)
        weights_vector.append(weight)

    return weights_vector


