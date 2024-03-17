# SOM-down-sampling-for-plant-point-clouds
```
[ISPRS P&RS] Unsupervised shape-aware SOM down-sampling for plant point clouds
Dawei Li, Zhaoyi Zhou, and Yongchang Wei
```
The code  and dataset will be released soon.

## Introduction
Observation of the external 3D shape/structure and some measurable phenotypic traits is of great significance to screening excellent varieties and improving crop yield in agriculture. The dense crop point clouds scanned by 3D sensors not only may include imaging noise, but also contain a large number of redundant points that will put high burden on storage and slow down the speed of algorithm for point cloud segmentation, classification, and other following processing steps.   
  
To reduce the complexity of point cloud data and meanwhile better represent the structure under limited resources, this paper presents a new Self-organizing Map (SOM)-based down-sampling strategy that is tailored for plant (or plant-like) point clouds. Our SOM-based sampling works in a purely unsupervised manner and precisely controls the number of points after down-sampling. It obtains shape-aware sampling on irregular plant point clouds by automatically encoding preliminary semantics to different organ types (e.g., stems are sampled as “lines”, and leaves are sampled as folded curved shaped in “surfaces”).   
  
Extensive experiments on a multi-species plant dataset were conducted using several popular deep 3D-segmentation networks as the downstream task unit, respectively. The segmentation performance of the SOM-processed dataset outperformed several other mainstream down-sampling strategies. Our SOM strategy with 1D neuron layer can be further generalized to 2D and 3D versions, and also can be extended to a more adaptive framework that automatically picks the most suitable version of SOM for each corresponding local shape component. The proposed strategy also showed good potential in serving different applications including point cloud skeleton extraction, crop main stem length measurement; and presented satisfactory results on point cloud datasets from other domains, indicating its high applicability and good data domain adaptation.

## Method

## Installation and usage

## Citation
If you find our work useful in your research, please consider citing:
