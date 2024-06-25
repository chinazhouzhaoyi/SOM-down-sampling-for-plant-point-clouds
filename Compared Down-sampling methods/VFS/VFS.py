import open3d as o3d
import numpy as np


#@jit()
def distance_point3d(p0, p1):
    d = (p0[0] - p1[0]) ** 2 + (p0[1] - p1[1]) ** 2 + (p0[2] - p0[2]) ** 2
    return math.sqrt(d)


def get_filelist(path):
    Filelist = []
    for home, dirs, files in os.walk(path):
        for filename in files:
            Filelist.append(os.path.join(home, filename))
    return Filelist


if __name__ == '__main__':
    path = './VFS/data'
    saved_path='./VFS_512point/'
    Filelist = get_filelist(path)
    expand_num = 1
    sample_count = 512
    voxel_size = 0.05 #voxel down sample dense
    for z in range(0, expand_num): 
        for file in Filelist:
            print(file)
            # ply = o3d.io.read_point_cloud(file)
            points = np.loadtxt(file,dtype=float,delimiter=' ')
            # points = np.loadtxt(file)
            # points=data_f[10:]
            # read_sample = o3d.io.read_point_cloud(file)
            # points = read_sample.points
            # o3d.visualization.draw_geometries([read_sample])
            # P = np.array(P)
            # pcd_array = np.asarray(points)
            pcd_array=np.array(points)
            print("pcd_array.shape:", pcd_array.shape)

            pcd = o3d.geometry.PointCloud()
            pcd.points= o3d.utility.Vector3dVector(pcd_array)

            # voxel sampling
            print("Downsample the point cloud with a voxel of 0.05")
            downpcd = pcd.voxel_down_sample(voxel_size=voxel_size)  # voxel_down_sample
            o3d.visualization.draw_geometries([downpcd])
            P=downpcd.points

            sample_point = np.zeros((sample_count, pcd_array.shape[1])) 
            file_name=file.split("\\")[-1]
            file_nameR=file_name
            print("file_nameR:",file_nameR)
            for i in range(sample_count):
                sample_point[i] = pcd_array[sampled_points_index[i]]
                
            # np.savetxt(saved_path +file_nameR+str(z) +".txt", sample_point,fmt='%.6f') 
            # print(saved_path +file_nameR+str(z) +".txt")

            # np.savetxt(saved_path + file_nameR , sample_point,
            #            fmt='%.6f')
            # print(saved_path + file_nameR + ".txt")
