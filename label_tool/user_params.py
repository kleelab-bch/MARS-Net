'''
Author: Junbong Jang
7/28/2020

This is a file to set parameters.
Specify canny_std_multiplier param, dataset name, image path, 
and save paths for edge and segmentation results
'''

# -------- User parameters -------------------
canny_std_multiplier = 1.8  # For canny algorithm, values from 1 to 2 usually works the best
denoise_kernel_size = 5  # For blurring, 5 usually works the best

# '040119_PtK1_S01_01_phase_3_DMSO_nd_03', '040119_PtK1_S01_01_phase_2_DMSO_nd_02', '040119_PtK1_S01_01_phase_2_DMSO_nd_01', '040119_PtK1_S01_01_phase_ROI2', '040119_PtK1_S01_01_phase'
a_dataset = '040119_PtK1_S01_01_phase'
img_root_path = '../assets/' + a_dataset + '/img/'  # original image folder path
saved_edge_path = 'generated_edge/' + a_dataset   # folder to save edges
saved_segment_path = 'generated_segmentation/' + a_dataset  # folder to save segmentations
saved_edge_user_params_path = 'generated_explore_edge/' + a_dataset