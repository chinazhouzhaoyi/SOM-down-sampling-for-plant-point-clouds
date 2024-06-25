"""

"""
import os
from os import listdir, path
import numpy as np
import random

'''
    input:  Center points PCD Dataset and Edge points PCD Dataset
	output: Merged points in the format of the TXT by xyzLabel
'''

path_center = r'...\core'  # your core points directory path
path_edge = r'...\edge'   # your edge points directory path
save_path = r'...\merge'  # save path of merged points

txt_cs = [f for f in listdir(path_center)
        if f.endswith('.txt') and path.isfile(path.join(path_center, f))]

txt_es = [f for f in listdir(path_edge)
        if f.endswith('.txt') and path.isfile(path.join(path_edge, f))]

#Selection of Random Seeds
i = 0

for txt_c, txt_e in zip(txt_cs, txt_es):
        c_temp = []
        e_temp = []
        end_temp = []
        i = i + 1 # Variation of seeds
        with open(os.path.join(path_center, txt_c), 'r') as f:
                index = []
                lines = f.readlines()
                size_c = len(lines) # Get the length of the input data
                np.random.seed(i)
                indexs = np.random.randint(11, size_c, 4096, int)# np.random.randint(start_Num, end_Num, N, int)   Randomly pick N points from (start_Num, end_Num)
                #Since PCD descriptions account for 10 lines, strat_ Num=11.
                for index in indexs:
                        c_temp.append(lines[index])

        with open(os.path.join(path_edge, txt_e), 'r') as f:
                index = []
                lines = f.readlines()
                size_e = len(lines)
                np.random.seed(i)
                indexs = np.random.randint(11, size_e, 4096, int)

                for index in indexs:
                        e_temp.append(lines[index])

        end_temp = c_temp + e_temp    #Merge the extracted edge and center points
        # Store merged points in the format of the TXT by xyzLabel
        with open(os.path.join(save_path, os.path.splitext(txt_e)[0] + "end.txt"), 'w') as f:
                f.write(''.join(end_temp[0:]))