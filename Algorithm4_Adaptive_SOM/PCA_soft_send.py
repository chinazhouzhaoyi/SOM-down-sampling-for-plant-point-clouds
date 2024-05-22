"""
paper: Unsupervised shape-aware SOM down-sampling for plant point clouds
file:  Algorithm4_Adaptive_SOM/PCA_soft_send.py
about: Filtering codes
author: Yongchang Wei, Zhaoyi Zhou
date: 2024-5-21
"""
import open3d as o3d
import os
import numpy as np
from scipy.spatial import KDTree
from sklearn.neighbors import NearestNeighbors
from os import listdir, path
import argparse

def PCA(data, correlation=False, sort=True):
    mean_data = np.mean(data, axis=0)
    normal_data = data - mean_data
    H = np.dot(normal_data.T, normal_data)
    eigenvectors, eigenvalues, _ = np.linalg.svd(H)

    if sort:
        sort = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[sort]
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors

def search_normals(radius_neighbor_idx, points):
    normals = []  
    alphas = []
    neighbor_num = []
    # -------------search normals---------------
    # Searching for points within the radius
    for i in range(len(radius_neighbor_idx)):
        neighbor_idx = radius_neighbor_idx[i]  
        neighbor_data = points[neighbor_idx] 
        eigenvalues, eigenvectors = PCA(neighbor_data)
        lambdas = np.sqrt(eigenvalues)
        # lambdas = eigenvalues
        alpha1 = (lambdas[0] - lambdas[1]) / lambdas[0]
        alpha2 = (lambdas[1] - lambdas[2]) / lambdas[0]
        alpha3 = lambdas[2] / lambdas[0]
        alphas.append(np.array([alpha1, alpha2, alpha3]))
        normals.append(eigenvectors[:, 2])  # The direction corresponding to the minimum eigenvalue is the normal direction
        neighbor_num.append(np.array(len(neighbor_idx)))

    return normals, alphas, neighbor_num

def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def loadDataFile(path):
    data = np.loadtxt(path)

    return data


def main(args):
    path_str = args.data_path 
    Filelist = get_filelist(path_str)
    txts = [f for f in listdir(path_str)
            if f.endswith('.txt') and path.isfile(path.join(path_str, f))]
    for file in Filelist:
        for txt in txts:
            with open(os.path.join(path_str, txt), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # lines = lines[11:] #if pcd file
            points = np.loadtxt(lines)[:, 0:3]  # 
            print('Total points number is:', points.shape[0])
            print(txt)

            # Computer the main direction of point clouds by PCA
            w, v = PCA(points) 
            point_cloud_vector = v[:, 0] 
            print('The main orientation of this pointcloud is: ', point_cloud_vector)

            nn = NearestNeighbors(n_neighbors=5).fit(points)
            distances, idx = nn.kneighbors(points)
            distances = np.sort(distances, axis=0)
            distances = distances[:,1]
            mean_distance = np.mean(distances)

            axis = o3d.geometry.TriangleMesh.create_coordinate_frame().rotate(v, center=(0, 0, 0))

            #Recurrent calculation of the normal vector for each point
            leafsize = 32 # Switch to the minimum number of violent searches

            KDTree_radius = args.radius * mean_distance# set the radius of KDTree
            desired_neighbor_count = 500

            tree = KDTree(points, leafsize=leafsize)  # build KDTree
            radius_neighbor_idx = tree.query_ball_point(points, KDTree_radius)  # obtain the neighbor idx within radius to each point
            temp_radius = KDTree_radius

            # Preliminary likelihood of shapes
            normals, alphas, neighbor_num = search_normals(radius_neighbor_idx, points)
            max_values_per_row_test = np.argmax(alphas, axis=1)
            zero_indices = np.where(max_values_per_row_test == 0)
            one_indices = np.where(max_values_per_row_test == 1)
            two_indices = np.where(max_values_per_row_test == 2)

            alphas = np.array(alphas, dtype=np.float64)
            ratio_l1_l2 = []
            ratio01 = args.ratio01

            for i in range(len(zero_indices[0])):

                # Compare the second and fourth elements
                index_of_max_value = 1 if alphas[zero_indices[0][i]][1] > alphas[zero_indices[0][i]][2] else 2
                # if index_of_max_value == 1:
                if(alphas[zero_indices[0][i]][0] / alphas[zero_indices[0][i]][1] < ratio01 and alphas[zero_indices[0][i]][0] / alphas[zero_indices[0][i]][1] > 1.0):
                    max_values_per_row_test[zero_indices[0][i]] = 1

            point_with_labels = np.column_stack((points, max_values_per_row_test))
            output_folder = args.output_path
            txt_name = txt.split('.')[0] +'.txt'
            outpath = os.path.join(output_folder, txt_name)
            np.savetxt(outpath, point_with_labels, fmt='%.8f %.8f %.8f %d')
            
            #----visualization-----
            # color = np.zeros((len(max_values_per_row_test), 4))
            # for i in range(len(max_values_per_row_test)):
            #     if max_values_per_row_test[i] == 0:
            #         # LawnGreen
            #         color[i][0] = 0.18431
            #         color[i][1] = 0.3098
            #         color[i][2] = 0.3098
            #         color[i][3] = 1
            #     if max_values_per_row_test[i] == 1:
            #         # LawnGreen
            #         color[i][0] = 0.41176
            #         color[i][1] = 0.41176
            #         color[i][2] = 0.41176
            #         color[i][3] = 1
            #     if max_values_per_row_test[i] == 2:
            #         # LawnGreen
            #         color[i][0] = 0.09804
            #         color[i][1] = 0.09804
            #         color[i][2] = 0.43922
            #         color[i][3] = 1


            # print(KDTree_radius)
            # #
            # KDTree_radius = KDTree_radius - 0.001
            # point_cloud = o3d.geometry.PointCloud()
            # point_cloud.points = o3d.utility.Vector3dVector(points)
            # point_cloud.colors = o3d.utility.Vector3dVector(color[:, :3])

            # o3d.visualization.draw_geometries([point_cloud])

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu',  type=bool, default=True, help='use cpu mode')
    # parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--radius', type=float, default=12.76, help='Control parameters for KDTree radius, recommended 12.76')
    parser.add_argument('--ratio01', type=float, default=2.5, help='Control parameters for likelihood of shapes, recommended 2.5')
    parser.add_argument('--data_path', type=str, default='./data', help='specify your point cloud path')
    parser.add_argument('--output_path', type=str, default='./filter_out', help='specify your point cloud path after filtering')
    parser.add_argument('--suffix', type=str, default='txt', help='specify the suffix of your input file by txt')
    
    return parser.parse_args()

if __name__ == '__main__':
    args=parse_args()
    main(args=args)