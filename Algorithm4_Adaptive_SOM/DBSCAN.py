"""
paper: Unsupervised shape-aware SOM down-sampling for plant point clouds
file:  Algorithm4_Adaptive_SOM/PCA_soft_send.py
about: Filtering codes
author: Yongchang Wei, Zhaoyi Zhou
date: 2024-5-21
"""
import sklearn
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import time
import os
import numpy as np
from os import listdir, path
import argparse

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

def main(agrs):
    path_str = args.data_path
    Filelist = get_filelist(path_str)
    txts = [f for f in listdir(path_str)
            if f.endswith('.txt') and path.isfile(path.join(path_str, f))]
    for file in Filelist:
        for txt in txts:
            with open(os.path.join(path_str, txt), 'r', encoding='utf-8') as f:
                lines = f.readlines()
                # lines = lines[11:] # if pcd file

            points = np.loadtxt(lines, delimiter=' ')[:, 0:3]  # 导入txt数据到np.array，这里只需导入前3列
            labels1 = np.loadtxt(lines, delimiter=' ', dtype=int)[:, 3]
            print('total points number is:', points.shape[0])

            alpha1 = []
            alpha2 = []
            alpha3 = []


            for i in range(len(points)):
                if(labels1[i] == 0):
                    alpha1.append(points[i])
                if(labels1[i] == 1):
                    alpha2.append(points[i])
                if (labels1[i] == 2):
                    alpha3.append(points[i])

            alpha1 = np.array(alpha1)
            alpha2 = np.array(alpha2)
            alpha3 = np.array(alpha3)

            print(sorted(sklearn.neighbors.VALID_METRICS['brute']))
            all_points = []
            all_points = np.array(all_points)

            # 标签迭代
            label = 0

            nn = NearestNeighbors(n_neighbors=5).fit(alpha1)
            distances, idx = nn.kneighbors(alpha1)
            distances = np.sort(distances, axis=0)
            distances = distances[:, 1]
            mean_distance = np.mean(distances)
            mean_distance = args.radius * mean_distance

            start = time.perf_counter()
            db = DBSCAN(eps=mean_distance, min_samples=5).fit(alpha1)
            end = time.perf_counter()
            print('Running time: %s Seconds' % (end - start))
            labels = db.labels_
            # print(labels)
            cluster_races = labels[np.argmax(labels)]
            print(max(labels))

            races = np.arange(cluster_races + 1, dtype=np.int64)


            for i in range(len(races)):
                mid_cluster_points_index = []
                for j in range(len(alpha1)):
                    if (labels[j] == i):
                        mid_cluster_points_index.append(j)
                mid_cluster_points = alpha1[mid_cluster_points_index]
                mid_cluster_points = np.array(mid_cluster_points)
                new_clounm = np.full((mid_cluster_points.shape[0], 1), label)
                pca_clounm = np.full((mid_cluster_points.shape[0], 1), 1)
                label = label + 1
                mid_cluster_points = np.concatenate((mid_cluster_points, new_clounm), axis=1)
                mid_cluster_points = np.concatenate((mid_cluster_points, pca_clounm), axis=1)

                if all_points.shape[0] == 0:
                    all_points = mid_cluster_points
                else:
                    all_points = np.row_stack((all_points, mid_cluster_points))




            start = time.perf_counter()

            nn = NearestNeighbors(n_neighbors=5).fit(alpha2)
            distances, idx = nn.kneighbors(alpha1)
            distances = np.sort(distances, axis=0)
            distances = distances[:, 1]
            mean_distance = np.mean(distances)
            mean_distance = args.radius * mean_distance

            db = DBSCAN(eps=mean_distance, min_samples=5).fit(alpha2)
            end = time.perf_counter()
            print('Running time: %s Seconds' % (end - start))
            labels = db.labels_
            cluster_races = labels[np.argmax(labels)]
            print(max(labels))

            races = np.arange(cluster_races + 1, dtype=np.int64)


            for i in range(len(races)):
                mid_cluster_points_index = []
                for j in range(len(alpha2)):
                    if (labels[j] == i):
                        mid_cluster_points_index.append(j)
                mid_cluster_points = alpha2[mid_cluster_points_index]
                mid_cluster_points = np.array(mid_cluster_points)
                new_clounm = np.full((mid_cluster_points.shape[0], 1), label)
                pca_clounm = np.full((mid_cluster_points.shape[0], 1), 2)
                label = label + 1
                mid_cluster_points = np.concatenate((mid_cluster_points, new_clounm), axis=1)
                mid_cluster_points = np.concatenate((mid_cluster_points, pca_clounm), axis=1)

                if all_points.shape[0] == 0:
                    all_points = mid_cluster_points
                else:
                    all_points = np.row_stack((all_points, mid_cluster_points))



            start = time.perf_counter()
            nn = NearestNeighbors(n_neighbors=5).fit(alpha3)
            distances, idx = nn.kneighbors(alpha3)
            distances = np.sort(distances, axis=0)
            distances = distances[:, 1]
            mean_distance = np.mean(distances)
            mean_distance = args.radius * mean_distance

            db = DBSCAN(eps=mean_distance, min_samples=5).fit(alpha3)
            end = time.perf_counter()
            print('Running time: %s Seconds' % (end - start))
            labels = db.labels_
            cluster_races = labels[np.argmax(labels)]
            print(max(labels))

            races = np.arange(cluster_races + 1, dtype=np.int64)


            for i in range(len(races)):
                mid_cluster_points_index = []
                for j in range(len(alpha3)):
                    if (labels[j] == i):
                        mid_cluster_points_index.append(j)
                mid_cluster_points = alpha3[mid_cluster_points_index]
                mid_cluster_points = np.array(mid_cluster_points)
                new_clounm = np.full((mid_cluster_points.shape[0], 1), label)
                pca_clounm = np.full((mid_cluster_points.shape[0], 1), 3)
                label = label + 1
                mid_cluster_points = np.concatenate((mid_cluster_points, new_clounm), axis=1)
                mid_cluster_points = np.concatenate((mid_cluster_points, pca_clounm), axis=1)

                if all_points.shape[0] == 0:
                    all_points = mid_cluster_points
                else:
                    all_points = np.row_stack((all_points, mid_cluster_points))
            output_folder = args.output_path
            txt_name = txt.split('.')[0] +'.txt'
            outpath = os.path.join(output_folder, txt_name)
            np.savetxt(outpath, all_points, fmt='%.8f %.8f %.8f %d %d')

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu',  type=bool, default=True, help='use cpu mode')
    # parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--radius', type=float, default=12.76, help='Control parameters for DBSCAN eps, recommended 12.76')
    parser.add_argument('--data_path', type=str, default='./filter_out', help='specify your point cloud path after filtering')
    parser.add_argument('--output_path', type=str, default='./cluster_out', help='specify your point cloud path after clustering')
    parser.add_argument('--suffix', type=str, default='txt', help='specify the suffix of your input file by txt')
    
    return parser.parse_args()

if __name__ == '__main__':
    args=parse_args()
    main(args)
