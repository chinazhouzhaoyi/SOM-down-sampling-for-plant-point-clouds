
Step 001: FPS downsampling and data enhancement of point clouds under the path

Input: txt point cloud file, format XYZLabel

Output: FPS sampled txt data set, format XYZLabel



Step 002: Convert the txt file to an h5 file, such as training. h5, testing. h5

Input: 1.FPS sampled txt data set, format XYZLabel

2.Txt of training sample file name or Test sample file name txt

Output: h5 File for training or testing

Note: Changes need to be made to the semantic label assigned based on the instance label
