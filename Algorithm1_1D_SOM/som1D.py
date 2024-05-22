'''
paper: Unsupervised shape-aware SOM down-sampling for plant point clouds
file:  Algorithm1_1D_SOM/som1D.py
about: main entrance for using 1D SOM for point cloud downsampling 
author: Zhaoyi Zhou
date: 2023-5-16
'''
import utils1D as utils
import numpy as np

def trainSom(data, x, y, epochs, sigma, verbose=False):
    """
    trains a som given the input data

    :param data: input, must be [Nx3]
    :param x: number of nodes in the x direction
    :param y: number of nodes in the y direction
    :param epochs: number of training iterations
    :param sigma: sigma for the gaussian kernel
    :param verbose: if True prints information during training
    :return: result of the som training
    """
    som = utils.MiniSom(x, y, 3, sigma, learning_rate=0.5, random_seed=1)
    # som.random_weights_init(data)   #random init
    som.uniform_weights_init(data) # recommened
    # som.FPS_weights_init(data,x)  #FPS init
    # som.pca_weights_init(data) #The size of the competition layer here is points * 1, while PCA requires a competition layer size of at least 2 * 2, so it is not applicable
    som.train_random(data, epochs, verbose=verbose) 
    winmap = som.win_map(data)  # winmap returns the index of winning nodes

    key_list = winmap.keys()
    print("Valid cluster number: ", len(key_list))

    if len(key_list) < x:
        print("Do refinement operation")
        som.train_cutOutliers(data, epochs, x, random_order=True, verbose=False)
        winmap = som.win_map(data)  # 设置winmap返回的是索引
        key_list = winmap.keys()
        print("Number of effective clusters after refinement operation", len(key_list))

    #Output quantization error and topology error, if necessary
    #print("Quantization error: ", som.quantization_error(data))
    #print("Topology error: ", som.topographic_error(data))
    weight = som._weights
    # my_dict = {"winmap": winmap, "weight": weight}
    my_dict = {"weight": weight}
    return my_dict


def somSkeleton(data, x, y, epochs, sigma):  
    """
    trains a som and decodes the winning units into skeleton nodes
    :param data: input, must be [Nx3]
    :param x: number of nodes in the x direction
    :param y: number of nodes in the y direction
    :param epochs: number of training iterations
    :param sigma: sigma for the gaussian kernel
    :return: the nodes that will form the skeleton
    """
    accept_dict = trainSom(data, x, y, epochs, sigma)
    # winmap = accept_dict["winmap"]
    weight = accept_dict["weight"]

    # skeleton = []
    # for w in winmap: 
    #     w_array = np.asarray(winmap[w])  # 
    #     w_mean = np.mean(w_array, axis=0) 
    #     skeleton.append(w_mean)
    # my_dict = {"skeleton_nodes": skeleton, "weight": weight}
    my_dict = {"weight": weight}
    return my_dict




def getSkeleton(organs, assignNum, sigma):
    """
    for each organ, returns the nodes of the skeleton
    We consider the entire point cloud as an organ
    
    Args:
    Organs: List [np. ndarray], a list containing multiple organ point clouds, each of which is a numpy array with a shape of (n, 3), where n is the number of points in the point cloud and 3 represents the three-dimensional coordinates of the points (x, y, z)
    AssignNum: List [int], an integer list of the same length as organs, representing how many skeleton nodes each organ's point cloud is assigned to
    Sigma: float, standard deviation of Gaussian kernel function
    Returns:
    Out_dict (dict): A dictionary containing skeleton nodes and weights, 
    where the value corresponding to the "skeleton nodes" key is a list, and each element in the list is a numpy array, indicating that the point cloud is assigned to the nearest weight vector, and then the clustered point cloud is averaged to obtain the skeleton nodes; 
    The value corresponding to the "weights" key is also a list, and each element in the list is a numpy array, representing the weight vector

    """
    skeletons = []
    weights = []

    for i, o in enumerate(organs): 
        epoch = len(o)
        accept_dict = somSkeleton(o, x=assignNum[i], y=1, epochs=int(5 * epoch), sigma=sigma) 

        # skeleton = accept_dict["skeleton_nodes"]
        weight = accept_dict["weight"]
        # skeletons.append(skeleton)
        weights.append(weight)
    # out_dict = {"skeleton_nodes": skeletons, "weights": weights}
    out_dict = {"weights": weights}
    return out_dict

