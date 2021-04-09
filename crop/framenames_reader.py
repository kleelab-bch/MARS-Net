import numpy as np

temp_data = np.load('./crop_results/crop_round1_VGG16/040119_PtK1_S01_01_phase_2_DMSO_nd_01_frame2_split0_repeat0_train_mask.npz')

temp_img = temp_data['arr_0']
temp_mask = temp_data['arr_1']
print(temp_img.shape)
print(temp_mask.shape)