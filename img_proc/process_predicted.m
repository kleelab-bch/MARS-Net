% Author Junbong Jang
% Date: March 2020

%dataset = {'040119_PtK1_S01_01_phase_3_DMSO_nd_03'; '040119_PtK1_S01_01_phase_2_DMSO_nd_02'; '040119_PtK1_S01_01_phase_2_DMSO_nd_01'; '040119_PtK1_S01_01_phase_ROI2'; '040119_PtK1_S01_01_phase'};
%model_names = {'ABCD';'ABCE'; 'ABDE'; 'ACDE'; 'BCDE'};
%root_path_list = {'../models/results/predict_wholeframe_round1_unet/'; '../models/results/predict_wholeframe_round1_VGG19_dropout/';};
%frame_list = {'1','2','6','10','22','34'};


%dataset= {'1121-1'; '1121-3'; '1121-4'; '1121-5'; '1121-6'};
%model_names = {'ABCD';'ABCE'; 'ABDE'; 'ACDE'; 'BCDE'};
%root_path_list = {'../models/results/predict_wholeframe_round1_mDia_raw_unet/'; '../models/results/predict_wholeframe_round1_mDia_raw_VGG19_dropout/';};
%frame_list = {'1','2','6','10','22','34'};

%dataset = {'Paxilin-HaloTMR-TIRF3'; 'Paxilin-HaloTMR-TIRF4'; 'Paxilin-HaloTMR-TIRF5'; 'Paxilin-HaloTMR-TIRF6'; 'Paxilin-HaloTMR-TIRF7'; 'Paxilin-HaloTMR-TIRF8'};
%model_names = {'ABCDE'; 'ABCDF'; 'ABCEF'; 'ABDEF'; 'ACDEF'; 'BCDEF'};
%root_path_list = {'../models/results/predict_wholeframe_round1_paxillin_TIRF_normalize_cropped_unet_patience_10/';
%                    '../models/results/predict_wholeframe_round1_paxillin_TIRF_normalize_cropped_VGG19_dropout_patience_10/';};
%frame_list = {'1','2','6','10','22'};


%dataset = {'040119_PtK1_S01_01_phase_3_DMSO_nd_03'; '040119_PtK1_S01_01_phase_2_DMSO_nd_02'; '040119_PtK1_S01_01_phase_2_DMSO_nd_01';
%            '040119_PtK1_S01_01_phase_ROI2'; '040119_PtK1_S01_01_phase'; '1121-1'; '1121-3'; '1121-4'; '1121-5'; '1121-6'};
%model_names = {'A';'B'; 'C'; 'D'; 'E';'F';'G'; 'H'; 'I'; 'J'};
%dataset = {'Paxilin-HaloTMR-TIRF3'; 'Paxilin-HaloTMR-TIRF4'; 'Paxilin-HaloTMR-TIRF5'; 'Paxilin-HaloTMR-TIRF6'; 'Paxilin-HaloTMR-TIRF7'; 'Paxilin-HaloTMR-TIRF8'};
%model_names = {'K';'L'; 'M'; 'N'; 'O'; 'P'};
%root_path_list = {'../models/results/predict_wholeframe_round1_generalist_VGG19_dropout/';'../models/results/predict_wholeframe_round1_generalist_unet/';
%            '../models/results/predict_wholeframe_round1_specialist_VGG19_dropout/';'../models/results/predict_wholeframe_round1_specialist_unet/';};


%dataset = {'Paxilin-HaloTMR-TIRF4';'Paxilin-HaloTMR-TIRF7';'Paxilin-HaloTMR-TIRF8'};
%model_names = { 'N';'N';'O'};
%root_path_list = {'../models/results/predict_wholeframe_round1_generalist_VGG19_dropout/'};
%root_path_list = {'../img_proc/generated/'};
%frame_list = {'2'};

%dataset = {'Paxilin-HaloTMR-TIRF3'};
%model_names = {'K'};
%root_path_list = {'../models/results/predict_wholeframe_round1_generalist_unet/'};
%frame_list = {'2'};

%dataset = {'Paxilin-HaloTMR-TIRF5'}
%model_names = {'ABCEF'}
%root_path_list = {'../models/results/predict_wholeframe_round1_paxillin_TIRF_normalize_cropped_VGG19_dropout_patience_10/'};
%frame_list = {'22'};


dataset = {'101018_part_A'};
model_names = {'BCDE'};
root_path_list = {'../models/results/predict_wholeframe_round1_cryptic_VGG19_dropout_mm_patience_10_overfit/'};
frame_list = {'10'};

repeat_index = 0;
for root_path_index = 1 : length(root_path_list)
    for data_index = 1 : length(dataset)
        for frame_index = 1 : length(frame_list)
            frame_num = frame_list{frame_index};
            root_path = root_path_list{root_path_index};
            disp(['@@@', root_path, ' ', num2str(data_index), ' ', frame_num])
            model_name = model_names{data_index, 1};

            mask_path = [root_path, dataset{data_index, 1}, '/frame', frame_num, '_', model_name, '_repeat' num2str(repeat_index), '/'];
%            mask_path = [root_path, dataset{data_index, 1}, '/predict_generalist_VGG19_dropout/'];
            filesStructure = dir(fullfile(mask_path, '*.png'));
            allFileNames = {filesStructure(:).name};

            generate_path = [root_path, dataset{data_index, 1}, '/processed_frame', frame_num, '_', model_name, '_repeat' num2str(repeat_index), '/'];
%            generate_path = [root_path, dataset{data_index, 1}, '/processed_predict_generalist_VGG19_dropout/'];

            mkdir(generate_path);
            for k = 1 : length(allFileNames)
                fileName = allFileNames{k};
                filePath = [mask_path, fileName];
                I = imread(filePath);

                process_image(I, [generate_path, fileName]);
            end
        end
    end
end

function process_image(I, save_path)

    I_binary = im2bw(I, 0.5);
%    I_binary = imbinarize(I);
    [row, col] = size(I_binary);

    artifact_percentage = ceil(row * col * 0.05);
    I_final = bwareaopen(I_binary, artifact_percentage);  % remove floating artifacts
    I_final = imfill(I_final,'holes'); % fill hole

%   The following image processing can fill up correct background to be foreground so use with caution
    fill_hole_percentage = ceil(row * col * 0.20);
    I_final = bwareaopen(~I_final, fill_hole_percentage);
    I_final = imfill(~I_final,'holes');

    imwrite(I_final*255, save_path)
end
