% entropy(imread('teacher.png'))
% entropy(imread('student.png'))
% entropy(imread('mask.png'))
% entropy(imread('mask_expand.png'))
% entropy(imread('img.png'))
% entropy(imread('img_expand.png'))

dataset = '040119_PtK1_S01_01_phase_3_DMSO_nd_03';
teacher_path = ['../vUnet/average_hist/predict_wholeframe_teacher/', dataset, '/34_0_', dataset, '/'];
student_path = ['../vUnet/average_hist/predict_wholeframe_student/', dataset, '/34_0_', dataset, '/'];
img_path = ['../assets/', dataset, '/img/'];
mask_path = ['../assets/', dataset, '/mask/'];
processed_mask_path = 'img_proc/processed_images';

teacher_struct = dir(fullfile(teacher_path, '*.png'));
student_struct = dir(fullfile(student_path, '*.png'));
img_struct = dir(fullfile(img_path, '*.png'));
mask_struct = dir(fullfile(mask_path, '*.png'));
processed_mask_struct = dir(fullfile(processed_mask_path, '*.png'));

folder_path = student_struct.folder;
img_names = {student_struct.name};

[mean_entropy, entropy_list] = calc_mean_entropy(folder_path, img_names);
histogram(entropy_list(:,1))
xlabel('entropy')
ylabel('counts')
title('Entropy of predicted images by student model')