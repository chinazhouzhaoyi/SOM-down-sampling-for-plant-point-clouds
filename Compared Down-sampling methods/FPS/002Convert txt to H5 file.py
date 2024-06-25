#!/usr/bin/env python3


import os
import sys
import numpy as np
import h5py
from tqdm import tqdm


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]


def loadDataFile(path):
    data = np.loadtxt(path)
    point_xyz = data[:, 0:3]
    ins_label = (data[:, 3]).astype(int)
    find_index = np.where(ins_label >= 1)
    sem_label = np.zeros((data.shape[0]), dtype=int)

    return point_xyz, ins_label, sem_label


def change_scale(data):
    # Move the center of the point cloud to the coordinate origin and limit the coordinates of all points to (0,1)
    xyz_min = np.min(data[:, 0:3], axis=0)
    xyz_max = np.max(data[:, 0:3], axis=0)
    xyz_move = xyz_min + (xyz_max - xyz_min) / 2
    data[:, 0:3] = data[:, 0:3] - xyz_move
    # scale
    scale = np.max(data[:, 0:3])

    return data[:, 0:3] / scale



def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist


'''
    input:  FPS TXT dataset in the form of XYZLabel
            Txt file for training sample names
            Txt file for testing sample names
	output: h5 File for training or testing
'''



if __name__ == "__main__":
    root = '..../FPS_TXT/'  # your 3DEPS TXT dataset directory path

    #  Reading training or testing data
    plant_classes = {}
    plant_classes['test'] = [line.rstrip() for line in open(os.path.join(root, "data_ids", 'test_ids.txt'))]  # assign test_ids.txt or train_ids.txt
    datapath = [(os.path.join(root, plant_classes['test'][i] + '.txt')) for i in range(len(plant_classes['test']))]  #
    print('The size of %s data is %d' % ("dataset", len(datapath)))

    save_path = '.../3DEPS_test_norm.h5'  # save path of h5 File

    num_sample = 1024  # Number of Downsampling
    files_count = len(datapath)
    DATA_ALL = []

    for index in tqdm(range(len(datapath)), total=len(datapath)):
        fn = datapath[index]  # fn is a single sample path

        current_data, current_ins_label, current_sem_label = loadDataFile(fn)

        for i in range(len(current_ins_label)):
            basename = os.path.basename(fn)

            #-----Assigning semantic labels needs to be determined based on the actual instance labels assigned to plants
            #for example
            # if (basename[0] == 'b' and current_ins_label[i] >= 1): # class1 instance of leaves
            #     current_sem_label[i] = 1
            # if (basename[0] == 'b' and current_ins_label[i] == 0): # class1 instance of stem
            #     current_sem_label[i] = 0
            # if (basename[0] == 'm' and current_ins_label[i] >= 1): # class2 instance of leaves
            #     current_sem_label[i] = 3
            # if (basename[0] == 'm' and current_ins_label[i] == 0): #class2 instance of stem
            #     current_sem_label[i] = 2
            # if (basename[0] == 'c' and current_ins_label[i] >=2): # class3 instance of leaves
            #     current_sem_label[i] = 5
            # if (basename[0] == 'c' and (current_ins_label[i] == 1 )): #class3 instance of stem
            #     current_sem_label[i] = 4

            if (basename[0] == 'b' and current_ins_label[i] == 0): # plant class1 instance of old organs
                current_sem_label[i] = 0
            if (basename[0] == 'b' and current_ins_label[i] == 1): # plant class1 instance of new organs
                current_sem_label[i] = 1
            if (basename[0] == 'm' and current_ins_label[i] == 0): # plant class2 instance of old organs
                current_sem_label[i] = 2
            if (basename[0] == 'm' and current_ins_label[i] == 1): # plant class2 instance of new organs
                current_sem_label[i] = 3


        change_data = change_scale(current_data)  # normalization

        data_label = np.column_stack((current_data, current_ins_label, current_sem_label))
        DATA_ALL.append(data_label)
    print(np.asarray(DATA_ALL).shape)
    output = np.vstack(DATA_ALL)
    output = output.reshape(files_count, num_sample, 5)

    print(output.shape)

    if not os.path.exists(save_path):
        with h5py.File(save_path, 'w') as f:

            f['data'] = output[:, :, 0:3]
            f['pid'] = output[:, :, 3]  # instance label
            f['seglabel'] = output[:, :, 4]  # semantic label
            f['obj'] = np.zeros(output[:, :, 4].shape) - 1


