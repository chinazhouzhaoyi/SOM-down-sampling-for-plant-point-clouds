#include <pcl/surface/gp3.h>
#include <boost/math/special_functions/round.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/surface/mls.h>        //Least Squares Smoothing Class Definition Header Files
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/surface/grid_projection.h>
#include <iostream>
#include <string.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <time.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/PolygonMesh.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/io/vtk_io.h>
#include <pcl/features/boundary.h>




using namespace pcl;
using namespace pcl::io;
using namespace pcl::console;
using namespace std;
void getFiles(string path, vector<string>& files)
{
	//File handle
	__int64 hFile = 0;
	//File information reading structure
	struct __finddata64_t  fileinfo;  //
	string p;  //string class
	if ((hFile = _findfirst64(p.assign(path).append("/*.pcd").c_str(), &fileinfo)) == -1)
	{
		cout << "No file is found\n" << endl;
	}
	else
	{
		do
		{
			files.push_back(p.assign(path).append("/").append(fileinfo.name));
		} while (_findnext64(hFile, &fileinfo) == 0);  //Find the next one, return 0 if successful, otherwise -1
		_findclose(hFile);
	}
}







int main(int argc, char** argv)
{
    /*
	input: PCD Dataset, Each PCD point cloud data format is PointXYZL
	output:Center points in PCD format and Edge points in PCD format by PointXYZL
	*/


	vector<string> files;
	char* filePath = "...\\PCD_Point_clouds"; //File path

	////Obtain all files under this path
	getFiles(filePath, files);
	char str[30];
	int size = files.size();
	for (int i = 0; i < size; i++)
	{
		pcl::PointCloud<pcl::PointXYZL>::Ptr cloudOrign(new pcl::PointCloud<pcl::PointXYZL>);
		pcl::PCDReader reader;
		// Replace the path below with the path where you saved your file;
		string path = files[i];
		reader.read(path, *cloudOrign); // Remember to download the file first!

		pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_b(new pcl::PointCloud<pcl::PointXYZL>);
		pcl::PointCloud<pcl::PointXYZL>::Ptr cloud_c(new pcl::PointCloud<pcl::PointXYZL>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointXYZ point;
		for (int i = 0; i < cloudOrign->size(); i++) {
			point.x = cloudOrign->points[i].x;
			point.y = cloudOrign->points[i].y;
			point.z = cloudOrign->points[i].z;
			cloud->push_back(point);
		}

		//Calculate Normals
		pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normEst;
		pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
		pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
		tree->setInputCloud(cloud);
		normEst.setInputCloud(cloud);
		normEst.setSearchMethod(tree);
		normEst.setKSearch(20);// K-neighborhood number
		normEst.compute(*normals);
		//Judge edge points
		pcl::PointCloud<pcl::Boundary> boundaries;
		pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundEst;
		tree2->setInputCloud(cloud);
		boundEst.setInputCloud(cloud);
		boundEst.setInputNormals(normals);
		boundEst.setSearchMethod(tree2);
		boundEst.setKSearch(20);// K-neighborhood number
		boundEst.setAngleThreshold(M_PI / 2);
		boundEst.compute(boundaries);
		//Extracting Edge Points ,then Reorganizing Point Clouds
		cloud_b->width = cloudOrign->points.size();
		cloud_b->height = 1;
		cloud_b->points.resize(cloud_b->width * cloud_b->height);
		//Extracting Core Points ,then Reorganizing Point Clouds
		cloud_c->width = cloudOrign->points.size();
		cloud_c->height = 1;
		cloud_c->points.resize(cloud_c->width * cloud_c->height);
		int j = 0;
		int k = 0;
		for (int i = 0; i < cloudOrign->points.size(); i++)
		{
			if (boundaries.points[i].boundary_point != 0)
			{
				cloud_b->points[j].x = cloudOrign->points[i].x;
				cloud_b->points[j].y = cloudOrign->points[i].y;
				cloud_b->points[j].z = cloudOrign->points[i].z;
				cloud_b->points[j].label = cloudOrign->points[i].label;
				j++;
			}
			else
			{
				cloud_c->points[k].x = cloudOrign->points[i].x;
				cloud_c->points[k].y = cloudOrign->points[i].y;
				cloud_c->points[k].z = cloudOrign->points[i].z;
				cloud_c->points[k].label = cloudOrign->points[i].label;
				k++;
			}
			continue;
		}
		cloud_b->width = j;
		cloud_b->points.resize(cloud_b->width * cloud_b->height);
		cloud_c->width = k;
		cloud_c->points.resize(cloud_c->width * cloud_c->height);
		cout << "********" << i << "********" << endl;
		cout << "Original number of points" << cloudOrign->size() << endl;
		cout << "number of edge points" << cloud_b->size() << endl;
		cout << "number of core points" << cloud_c->size() << endl;

        //Path for storing edge points and center points separately
		string path_e = "F:\\zzy\\VFS+3DEPSdata\\3DEPS_hebing\\edge\\";
		string path_c = "F:\\zzy\\VFS+3DEPSdata\\3DEPS_hebing\\core\\";
		//Read the original PCD file name
		string::size_type pidx_e = path.rfind('/', path.length());
		string::size_type pidx_c = path.rfind('.', path.length());
		string filename_e = path.substr(pidx_e + 1, pidx_c);
		//Recombine storage paths
		path_e = path_e + filename_e + "_e.txt";
		path_c = path_c + filename_e + "_c.txt";
		cout << "path_e:" << path_e << endl;
		cout << "path_c:" << path_c << endl;
		savePCDFile<pcl::PointXYZL>(path_e, *cloud_b); //Save PCD in the form of xyzLabel
		savePCDFile<pcl::PointXYZL>(path_c, *cloud_c); //Save PCD in the form of xyzLabel
	}

	return (0);
}