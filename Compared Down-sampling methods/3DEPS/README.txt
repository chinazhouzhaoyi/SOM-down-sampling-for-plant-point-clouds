Step 001:批量执行分离边缘点和中心点操作
输入：带标签的PCD点云数据集，格式PointXYZL
输出：带标签的边缘点PCD数据集，格式PointXYZL
 及     带标签的中心点PCD数据集，格式PointXYZL

Step 002:随机抽取部分边缘点、中心点，合并成新的txt文件
输入：带标签的边缘点PCD数据集，格式PointXYZL
 及     带标签的中心点PCD数据集，格式PointXYZL
输出：合并的边缘点与中心点txt文件集合，格式XYZLabel


Step 003:按比例合成指定数目的点云，并数据增强
输入：合并的边缘点与中心点txt文件集合，格式XYZLabel
输出：3DEPS采样后的txt数据集合，格式XYZLabel

Step 004: 将txt文件转换成h5文件，例如训练.h5 , 测试.h5
输入：3DEPS采样后的txt数据集合，格式XYZLabel
	训练样本文件名的txt
	测试样本文件名的txt
输出：h5 File for training or testing
