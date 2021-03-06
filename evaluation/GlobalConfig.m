classdef GlobalConfig
    properties

    % blue, orange, green, yellow, red, skyblue, violet, pink, gray
%    graph_colors = [[0, 0.4470, 0.7410];
%                    [0.8500, 0.3250, 0.0980];
%                    [0.4660, 0.6740, 0.1880];
%                    [0.9290, 0.6940, 0.1250];
%                    [0.6350, 0.0780, 0.1840];
%                    [0.3010, 0.7450, 0.9330];
%                    [0.4940, 0.1840, 0.5560];
%                    [247,129,191]/255;
%                    [153,153,153]/255;];
        graph_colors = [[0, 0.4470, 0.7410]; [0.8500, 0.3250, 0.0980]];  % blue, orange

        % blue, yellow, pink, orange, green, red
%        graph_colors = [[0, 0.4470, 0.7410];
%                        [0.9290, 0.6940, 0.1250];
%                        [247,129,191]/255;
%                        [0.8500, 0.3250, 0.0980];
%                        [0.4660, 0.6740, 0.1880];
%                        [0.6350, 0.0780, 0.1840];];


        root_path = '/media/bch_drive/Public/JunbongJang/Segmentation/evaluation/';
        img_root_path = '/media/bch_drive/Public/JunbongJang/Segmentation/assets/';

        mask_type = '';
        img_type = '';
        dataset_list = {};
        fold_name_list = {};
        total_prediction_path_num = 0;
        chosen_frame_index = 1; % 1 represent 1 frame, and 4 represents 10 frames given [1,2,6,10,22,34] for bargraph
        repeat_max = 1;

        dataset_split_list = [];
        dataset_interval_list = [];

        dice_ylim = [0,0.8];
        f1_ylim = [0.84, 0.95];
        recall_ylim = [0.86, 0.92];
        precision_ylim = [0.93, 0.985];

        dice_bar_ylim = [0.99,1];
        f1_bar_ylim = [0.9, 0.95];
        recall_bar_ylim = [0.86, 0.925];
        precision_bar_ylim = [0.935, 0.98];

        box_ylim = [-0.05, 1.15]
%        box_ylim = [0.4, 1.1]

        violin_diff_ylim = [-0.05, 0.05];
        violin_compare_ylim = [-0.05, 1.15];

%% --------------------------------------------------------------
        round_num = 1;
%        round_num = 2;

        max_dist_pixel = 3  % 3 for phase-contrast, 5 for paxillin
%% --------------------------------------------------------------
%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_specialist_unet/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_generalist_unet/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_specialist_VGG19_dropout/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_generalist_VGG19_dropout/'};


%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_cryptic_VGG19_dropout/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_cryptic_VGG19_dropout_patience_10/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_cryptic_VGG19_dropout_sm/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_cryptic_VGG19_dropout_mm_patience_10/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_cryptic_VGG19_dropout_mm_patience_10_overfit/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_cryptic_VGG19_dropout_overfit/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_cryptic_VGG16/';}

%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_cryptic_all_VGG19_dropout_patience_10_overfit/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_cryptic_all_VGG19_dropout_mm_patience_10_overfit/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_cryptic_all_heq_VGG19_dropout_mm_patience_10_overfit/';}

%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_cryptic_VGG19_dropout/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_cryptic_VGG19D_temporal_context_residual/';}

%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_unet/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_VGG19_dropout/';}
%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_unet/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_single_micro_unet/';}
%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_unet/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_unet_imagenet_pretrained/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_VGG19/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_VGG19_imagenet_pretrained/'}
%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_VGG19_dropout/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_single_micro_VGG19_dropout/'};
%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_single_micro_VGG19_dropout/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_single_micro_VGG19D_se/'};
%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_VGG19_dropout/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_single_micro_VGG19D/'};
%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_single_micro_VGG19_dropout/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_single_micro_VGG19D_temporal_attn_v3/'};
%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_single_micro_VGG19_dropout/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_single_micro_VGG19D_temporal_context_residual/'};

%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_VGG19_dropout_input64/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_VGG19_dropout_input80/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_VGG19_dropout_input96/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_VGG19_dropout/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_VGG19_dropout_input192/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_VGG19_dropout_input256_crop200/';}

%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_mDia_raw_unet/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_mDia_raw_VGG19_dropout/';}

%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_paxillin_TIRF_normalize_cropped_unet_patience_10/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_paxillin_TIRF_normalize_cropped_VGG19_dropout_patience_10/';};
%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_paxillin_WF_normalize/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_paxillin_TIRF_normalize/';};


        prediction_path_list = {'../../models/results/predict_wholeframe_round1_unet/';
                                 '../../models/results/predict_wholeframe_round1_unet_marsnet/';
                                 '../../models/results/predict_wholeframe_round1_unet_imagenet_pretrained_1024/';};
%        prediction_path_list = {'../../models/results/predict_wholeframe_round1_unet/';
%                                 '../../models/results/predict_wholeframe_round1_VGG16_no_pretrain/';
%                                 '../../models/results/predict_wholeframe_round1_VGG19_no_pretrain/';
%                                 '../../models/results/predict_wholeframe_round1_unet_imagenet_pretrained_marsnet/';
%                                 '../../models/results/predict_wholeframe_round1_VGG16/';
%                                 '../../models/results/predict_wholeframe_round1_VGG19/';
%                                 '../../models/results/predict_wholeframe_round1_VGG16_batchnorm/';
%                                  '../../models/results/predict_wholeframe_round1_VGG19_batchnorm/';
%                                  '../../models/results/predict_wholeframe_round1_VGG16_dropout/';
%                                  '../../models/results/predict_wholeframe_round1_VGG19_dropout/';
%                                  '../../models/results/predict_wholeframe_round1_VGG19_batchnorm_dropout/'};


%         prediction_path_list = {'../../models/results/predict_wholeframe_round1_unet/';
%                                 '../../models/results/predict_wholeframe_round1_unet_imagenet_pretrained_marsnet/';
%                                 '../../models/results/predict_wholeframe_round1_VGG16/';
%                                 '../../models/results/predict_wholeframe_round1_VGG19/';
%                                 '../../models/results/predict_wholeframe_round1_VGG16_dropout/';
%                                 '../../models/results/predict_wholeframe_round1_VGG19_dropout/';
%                                 '../../models/results/predict_wholeframe_round1_Res50V2/';
%                                 '../../models/results/predict_wholeframe_round1_EFF_B7_no_preprocessing/'};

%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_spheroid_unet/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_spheroid_VGG19_no_pretrain/';
%                                '/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_spheroid_VGG19/'};
%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_spheroid_test_VGG19_upscaled/'};

%        prediction_path_list = {'/media/bch_drive/Public/JunbongJang/Segmentation/models/results/predict_wholeframe_round1_organoid_VGG19_dropout_crop_even/'};
%% --------------------------------------------------------------
%        dice_folder_name = 'round1_specialist_unet_round1_generalist_unet_round1_specialist_VGG19_dropout_round1_generalist_VGG19_dropout';

%        dice_folder_name = 'round1_paxillin_TIRF_normalize_cropped_unet_patience_10_round1_paxillin_TIRF_normalize_cropped_VGG19_dropout_patience_10';
%         dice_folder_name = 'round1_paxillin_TIRF_normalize_unet_round1_paxillin_TIRF_normalize_VGG19_dropout';

%         dice_folder_name = 'round1_mDia_raw_unet_round1_mDia_raw_VGG19_dropout';

%         dice_folder_name = 'round1_VGG19_dropout_input64_round1_VGG19_dropout_input80_round1_VGG19_dropout_input96_round1_VGG19_dropout_round1_VGG19_dropout_input192_round1_VGG19_dropout_input256_crop200';
%         dice_folder_name = 'round1_unet_round1_VGG16_no_pretrain_round1_VGG19_no_pretrain_round1_VGG16_round1_VGG19_round1_VGG16_batchnorm_round1_VGG19_batchnorm_round1_VGG16_dropout_round1_VGG19_dropout';
%         dice_folder_name = 'round1_unet_round1_VGG16_round1_VGG19_round1_VGG16_dropout_round1_VGG19_dropout_round1_Res50V2_round1_EFF_B7_no_preprocessing';

%% --------------------------------------------------------------
%        display_names = {'Specialist U-Net'; 'Generalist U-Net'; 'Specialist VGG19-U-Net'; 'Generalist VGG19-U-Net'};
%        display_names = {'U-Net'; 'U-Net pretrained'; 'VGG19'; 'VGG19 pretrained'};
%        display_names = {'VGG19D-U-Net'; 'VGG19D-U-Net temporal'};

%        display_names = {'cryptic VGG19D'; 'cryptic VGG19D temporal'};
%        display_names = {'cryptic VGG19D'; 'cryptic VGG19D pat10'; 'cryptic VGG19D sm'; 'cryptic VGG19D mm pat10';'cryptic VGG19D mm pat10 overfit';'cryptic VGG19D overfit';'cryptic VGG16'};
%        display_names = {'cryptic_all VGG19D pat10'; 'cryptic_all VGG19D mm pat10'; 'cryptic_all heq VGG19D mm pat10'};
%        display_names = {'VGG19D-U-Net'; 'VGG19D-U-Net input256'};
%        display_names = {'U-Net';'VGG19-U-Net Dropout'};
        display_names = {'U-Net';'U-Net marsnet';'U-Net pretrained marsnet'};
%        display_names = {'WF norm'; 'TIRF norm'};
%        display_names = {'Teacher TIRF'; 'Student TIRF'};

%        display_names = {'spheroid_unet'; 'spheroid_VGG19_no_pretrain'; 'spheroid_VGG19'};
%        display_names = {'spheroid_VGG19_marsnet'};

%        display_names = {'64';'80';'96';'128'; '192'; '256'};
%        display_names = {'U-Net'; 'U-Net pretrained';};
%        display_names = {'U-Net';
%                         'VGG16P-U-Net';
%                         'VGG19P-U-Net';
%                         'U-Net P';
%                         'VGG16-U-Net';
%                         'VGG19-U-Net';
%                         'VGG16B-U-Net';
%                         'VGG19B-U-Net';
%                         'VGG16D-U-Net';
%                         'VGG19D-U-Net';
%                         'VGG19D&B-U-Net'};

%        display_names = {'U-Net no pretrain';
%                         'VGG16-U-Net';
%                         'VGG19-U-Net';
%                         'VGG16-U-Net dropout';
%                         'VGG19-U-Net dropout';
%                         'Res50V2-U-Net';
%                         'EFF-B7-U-Net'};
%        display_names = {'U-Net NP';
%                         'U-Net P';
%                         'VGG16-U-Net';
%                         'VGG19-U-Net';
%                         'VGG16-U-Net dropout';
%                         'VGG19-U-Net dropout';
%                         'Res50V2-U-Net';
%                         'EFF-B7-U-Net'};

%        display_names = {'spheroid_test_VGG19_upscaled';};

%% --------------------------------------------------------------
% In the frame_list matrix,
% each column corresponds to different number of traning frames and
% each row separated by ; corresponds to the different models
%        frame_list = [1;1;1];
%        frame_list = [2;2];
%        frame_list = [10;10;10;10];

%         frame_list = [34;34];
        frame_list =[10;10;10];

%        frame_list = [10;10;10;10;10;10;10];
%        frame_list = [10;10;10;10;10;10];
%        frame_list =[1,2,6,10,22,34;
%                    1,2,6,10,22,34];
%        frame_list =[1,2,6,10,22;
%                    1,2,6,10,22];
%        frame_list = [2 ;
%                      200];

%        frame_list = [10;10;10;10;10;
%                      10;10;10;10;10;10];
%        frame_list = [10;10;10;10;
%                      10;10;10;10];

%        frame_list = [1,2,6,10,22,34;
%                     1,2,6,10,22,34;
%                     1,2,6,10,22,34;
%                     1,2,6,10,22,34;
%                     1,2,6,10,22,34;
%                     1,2,6,10,22,34;
%                     1,2,6,10,22,34;
%                     1,2,6,10,22,34];

%        frame_list = [1];

%% --------------------------------------------------------------
    end  % properties end

    methods
        function GlobalConfigObj = GlobalConfig()
            close all;

            addpath(genpath([GlobalConfigObj.root_path, 'npy-matlab-master']));
            addpath(genpath([GlobalConfigObj.root_path, 'Violinplot-Matlab-master']));
            addpath(genpath([GlobalConfigObj.root_path, 'subaxis']));
            addpath(genpath([GlobalConfigObj.root_path, 'evaluation_f1/extended-berkeley-segmentation-benchmark-master-linux']));

            GlobalConfigObj.total_prediction_path_num = size(GlobalConfigObj.prediction_path_list, 1);

            set(gca, 'FontName', 'Arial');
        end

        function GlobalConfigObj = update_config(self, prediction_path)
            GlobalConfigObj = self;
            if strcmp(prediction_path, '')
                prediction_path = self.prediction_path_list{1};
            end

            if self.round_num == 1
                if contains(prediction_path, 'generalist') || contains(prediction_path, 'specialist')
                    GlobalConfigObj.repeat_max = 1;
                    GlobalConfigObj.img_root_path = '/media/bch_drive/Public/JunbongJang/Segmentation/assets/test_generalist/';
                    GlobalConfigObj.mask_type = '/mask/';
                    GlobalConfigObj.img_type = '/img/';

%                    GlobalConfigObj.dataset_list = {'040119_PtK1_S01_01_phase_3_DMSO_nd_03'; '040119_PtK1_S01_01_phase_2_DMSO_nd_02'; '040119_PtK1_S01_01_phase_2_DMSO_nd_01'; '040119_PtK1_S01_01_phase_ROI2'; '040119_PtK1_S01_01_phase'};
%                    GlobalConfigObj.fold_name_list = {'A';'B'; 'C'; 'D'; 'E'};

%                    GlobalConfigObj.dataset_list = {'1121-1'; '1121-3'; '1121-4'; '1121-5'; '1121-6'};
%                    GlobalConfigObj.fold_name_list = {'F';'G'; 'H'; 'I'; 'J'};

%                    GlobalConfigObj.dataset_list = {'Paxilin-HaloTMR-TIRF3'; 'Paxilin-HaloTMR-TIRF4'; 'Paxilin-HaloTMR-TIRF5';
%                                          'Paxilin-HaloTMR-TIRF6'; 'Paxilin-HaloTMR-TIRF7'; 'Paxilin-HaloTMR-TIRF8'};
%                    GlobalConfigObj.fold_name_list = {'K'; 'L'; 'M'; 'N'; 'O'; 'P'};

                    GlobalConfigObj.dataset_list = {'040119_PtK1_S01_01_phase_3_DMSO_nd_03';
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_02';
                                          '040119_PtK1_S01_01_phase_2_DMSO_nd_01'; '040119_PtK1_S01_01_phase_ROI2';
                                          '040119_PtK1_S01_01_phase'; '1121-1'; '1121-3'; '1121-4'; '1121-5'; '1121-6';
                                          'Paxilin-HaloTMR-TIRF3'; 'Paxilin-HaloTMR-TIRF4'; 'Paxilin-HaloTMR-TIRF5';
                                          'Paxilin-HaloTMR-TIRF6'; 'Paxilin-HaloTMR-TIRF7'; 'Paxilin-HaloTMR-TIRF8'};
                    GlobalConfigObj.fold_name_list = {'A'; 'B'; 'C'; 'D'; 'E'; 'F'; 'G'; 'H'; 'I'; 'J'; 'K'; 'L'; 'M'; 'N'; 'O'; 'P'};

                    % For phase, mDia, Paxillin
                    GlobalConfigObj.dataset_split_list = [5, 10, 16];
%                    GlobalConfigObj.dataset_interval_list = [2,10,1];  % for violin plot all
                    GlobalConfigObj.dataset_interval_list = [1,1,1];  % for violin plot individual
                    % For one dataset only
%                    GlobalConfigObj.dataset_split_list = [length(GlobalConfigObj.dataset_list)];
%                    GlobalConfigObj.dataset_interval_list = [1];

                    GlobalConfigObj.max_dist_pixel = 5;
                    % blue, pink, orange, green
                    GlobalConfigObj.graph_colors = [[0, 0.4470, 0.7410];
                                                    [247,129,191]/255;
                                                    [0.8500, 0.3250, 0.0980];
                                                    [0.4660, 0.6740, 0.1880];];


                elseif contains(prediction_path, 'cryptic_all')
                    GlobalConfigObj.repeat_max = 1;
                    GlobalConfigObj.img_root_path = '/media/bch_drive/Public/JunbongJang/Segmentation/assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/';
                    GlobalConfigObj.mask_type = '/mask/';
                    GlobalConfigObj.img_type = '/img/';
                    GlobalConfigObj.dataset_list = {'101018_all'};
                    GlobalConfigObj.fold_name_list = {'A'};
                    GlobalConfigObj.max_dist_pixel = 5;
                    GlobalConfigObj.dataset_split_list = [length(GlobalConfigObj.dataset_list)];
                    GlobalConfigObj.dataset_interval_list = [1];

                    % blue, yellow, pink, orange, green, red
                    GlobalConfigObj.graph_colors = [[0, 0.4470, 0.7410];
                                    [0.9290, 0.6940, 0.1250];
                                    [247,129,191]/255;
                                    [0.8500, 0.3250, 0.0980];
                                    [0.4660, 0.6740, 0.1880];
                                    [0.6350, 0.0780, 0.1840];];

                elseif contains(prediction_path, 'cryptic_')
                    GlobalConfigObj.repeat_max = 1;
                    GlobalConfigObj.img_root_path = '/media/bch_drive/Public/JunbongJang/Segmentation/assets/Cryptic Lamellipodia/CellMask-05152014-Control-1/';
%                    GlobalConfigObj.mask_type = '/mask_cropped/';
%                    GlobalConfigObj.img_type = '/img_cropped/';
                    GlobalConfigObj.mask_type = '/mask/';
                    GlobalConfigObj.img_type = '/img/';
                    GlobalConfigObj.dataset_list = {'101018_part_E'; '101018_part_D'; '101018_part_C'; '101018_part_B'; '101018_part_A'; };
                    GlobalConfigObj.fold_name_list = {'ABCD';'ABCE'; 'ABDE'; 'ACDE'; 'BCDE'};
                    GlobalConfigObj.max_dist_pixel = 5;
                    GlobalConfigObj.dataset_split_list = [length(GlobalConfigObj.dataset_list)];
                    GlobalConfigObj.dataset_interval_list = [1];

                    % blue, pink, skyblue, red, green, light yellow, gray, yellow, orange, violet
                    GlobalConfigObj.graph_colors = [[0, 0.4470, 0.7410];
                                    [247,129,191]/255;
                                    [0.3010, 0.7450, 0.9330];
                                    [0.6350, 0.0780, 0.1840];
                                    [0.4660, 0.6740, 0.1880];
                                    [255,255,153]/255;
                                    [153,153,153]/255;
                                    [0.9290, 0.6940, 0.1250];
                                    [0.8500, 0.3250, 0.0980];
                                    [0.4940, 0.1840, 0.5560]];

                elseif contains(prediction_path, 'organoid_')
                    GlobalConfigObj.repeat_max = 1;
                    GlobalConfigObj.img_root_path = '/media/bch_drive/Public/JunbongJang/Organoid/generated/segmentation_train/';
                    GlobalConfigObj.mask_type = '/mask/';
                    GlobalConfigObj.img_type = '/img_bright/';
%                    GlobalConfigObj.dataset_list = {'Lu-17_48h_z_ex1'; 'Lu-17_48h_z_ex2'; 'Lu-17_48h_z_ex4';
%                                                  'Lu-24_48h_z_ex1';'Lu-24_48h_z_ex2';'Lu-24_48h_z_ex3';'Lu-24_48h_z_ex4';
%                                                  'Lu-24_724_z_ex1'; 'Lu-24_724_z_ex2'; 'Lu-24_724_z_ex4';
%                                                  'Lu-28_724_z_ex2'; 'Lu-28_724_z_ex4'};
%                    GlobalConfigObj.fold_name_list = {'A'; 'B'; 'C'; 'D'; 'E'; 'F'; 'G'; 'H'; 'I'; 'J'; 'K'; 'L'};
                    GlobalConfigObj.dataset_list = { 'Lu-17_48h_z_ex4'};
                    GlobalConfigObj.fold_name_list = {'C'};
                    GlobalConfigObj.max_dist_pixel = 5;
                    GlobalConfigObj.dataset_split_list = [length(GlobalConfigObj.dataset_list)];
                    GlobalConfigObj.dataset_interval_list = [1];

                elseif contains(prediction_path, 'spheroid_test')
                    GlobalConfigObj.repeat_max = 1;
                    GlobalConfigObj.img_root_path = '/media/bch_drive/Public/JunbongJang/Segmentation/assets/Spheroid/test/';
                    GlobalConfigObj.mask_type = '/mask/';
                    GlobalConfigObj.img_type = '/';
                    GlobalConfigObj.dataset_list = {'Cdc42_5uM'; 'Cdc42_10uM'; 'DMSO'; 'DMSO_2'; 'FAK'; 'FAK_2';
                                                    'Rac_5uM'; 'Rac_10uM'; 'Rac_20uM'; 'Rho_5uM'; 'Rho_10uM'; 'Rho_20uM'};
                    GlobalConfigObj.fold_name_list = {'train'; 'train'; 'train'; 'train'; 'train'; 'train';
                                                      'train'; 'train'; 'train'; 'train'; 'train'; 'train'; };
                    GlobalConfigObj.max_dist_pixel = 3;

                    % blue, dark gray, orange
                    GlobalConfigObj.graph_colors = [[0, 0.4470, 0.7410]];
                    GlobalConfigObj.dataset_split_list = [length(GlobalConfigObj.dataset_list)];
                    GlobalConfigObj.dataset_interval_list = [1];


                elseif contains(prediction_path, 'spheroid')
                    GlobalConfigObj.repeat_max = 1;
                    GlobalConfigObj.img_root_path = '/media/bch_drive/Public/JunbongJang/Segmentation/assets/Spheroid/';
                    GlobalConfigObj.mask_type = '/mask/';
                    GlobalConfigObj.img_type = '/img/';
                    GlobalConfigObj.dataset_list = {'1'; '2'; '3'; '4'; '5'; '6'; '7'; '8'; '9'; '10'; '11'; '12'; '13'; '14'; '15'; '16'; '17'; '18'; '19'; '20'; '21'; '22'; '23'};
                    GlobalConfigObj.fold_name_list = {'A'; 'B'; 'C'; 'D'; 'E'; 'F'; 'G'; 'H'; 'I'; 'J'; 'K'; 'L'; 'M'; 'N'; 'O'; 'P'; 'Q'; 'R'; 'S'; 'T'; 'U'; 'V'; 'W'};
                    GlobalConfigObj.max_dist_pixel = 3;

                    % blue, dark gray, orange
                    GlobalConfigObj.graph_colors = [[0, 0.4470, 0.7410];
                                                    [153,153,153]/255;
                                                    [0.8500, 0.3250, 0.0980];];
                    GlobalConfigObj.dataset_split_list = [length(GlobalConfigObj.dataset_list)];
                    GlobalConfigObj.dataset_interval_list = [1];

                elseif contains(prediction_path, '_paxillin_WF')
                    GlobalConfigObj.repeat_max = 1;
                    GlobalConfigObj.img_root_path = '/media/bch_drive/Public/JunbongJang/Segmentation/assets/';
                    GlobalConfigObj.mask_type = '/mask_old/';
                    GlobalConfigObj.img_type = '/img/';
                    GlobalConfigObj.dataset_list = {'Paxilin-HaloTMR-TIRF3-WF_TMR_M'; 'Paxilin-HaloTMR-TIRF4-WF_TMR_M'; 'Paxilin-HaloTMR-TIRF5-WF_TMR_M';
                                                    'Paxilin-HaloTMR-TIRF6-WF_TMR_M'; 'Paxilin-HaloTMR-TIRF7-WF_TMR_M'; 'Paxilin-HaloTMR-TIRF8-WF_TMR_M'};
                    GlobalConfigObj.fold_name_list = {'ABCDE'; 'ABCDF'; 'ABCEF'; 'ABDEF'; 'ACDEF'; 'BCDEF'};
                    GlobalConfigObj.max_dist_pixel = 5;

                elseif contains(prediction_path, '_paxillin_TIRF')
                    GlobalConfigObj.repeat_max = 1;
                    GlobalConfigObj.img_root_path = '/media/bch_drive/Public/JunbongJang/Segmentation/assets/';
                    GlobalConfigObj.mask_type = '/mask/';
                    GlobalConfigObj.img_type = '/img/';
                    GlobalConfigObj.dataset_list = {'Paxilin-HaloTMR-TIRF3'; 'Paxilin-HaloTMR-TIRF4'; 'Paxilin-HaloTMR-TIRF5';
                                                    'Paxilin-HaloTMR-TIRF6'; 'Paxilin-HaloTMR-TIRF7'; 'Paxilin-HaloTMR-TIRF8'};
                    GlobalConfigObj.fold_name_list = {'ABCDE'; 'ABCDF'; 'ABCEF'; 'ABDEF'; 'ACDEF'; 'BCDEF'};
                    GlobalConfigObj.max_dist_pixel = 5;
                    % For one dataset only
                    GlobalConfigObj.dataset_split_list = [length(GlobalConfigObj.dataset_list)];
                    GlobalConfigObj.dataset_interval_list = [1];

                    if contains(prediction_path, '_cropped')
                        GlobalConfigObj.img_type = '/img_cropped/'
                        GlobalConfigObj.mask_type = '/mask_cropped/'
                    end

                elseif contains(prediction_path, '_mDia')
                    GlobalConfigObj.repeat_max = 1;
                    GlobalConfigObj.img_root_path = '/media/bch_drive/Public/JunbongJang/Segmentation/assets/mDia_chauncey/';
                    GlobalConfigObj.mask_type = '/mask/';
                    GlobalConfigObj.img_type = '/raw/';
                    GlobalConfigObj.dataset_list = {'1121-1'; '1121-3'; '1121-4'; '1121-5'; '1121-6'};
                    GlobalConfigObj.fold_name_list = {'ABCD';'ABCE'; 'ABDE'; 'ACDE'; 'BCDE'};
                    GlobalConfigObj.max_dist_pixel = 5;
                    % For one dataset only
                    GlobalConfigObj.dataset_split_list = [length(GlobalConfigObj.dataset_list)];
                    GlobalConfigObj.dataset_interval_list = [1];
                
                elseif contains(prediction_path, '_VGG') || contains(prediction_path, '_unet') || contains(prediction_path, '_Res') || contains(prediction_path, '_Dense') || contains(prediction_path, '_EFF_B7')
                    GlobalConfigObj.repeat_max = 1;
                    GlobalConfigObj.img_root_path = '/media/bch_drive/Public/JunbongJang/Segmentation/assets/';
                    GlobalConfigObj.mask_type = '/mask_fixed/';
                    GlobalConfigObj.img_type = '/img/';
%                    GlobalConfigObj.dataset_list = {'040119_PtK1_S01_01_phase_3_DMSO_nd_03'};
%                    GlobalConfigObj.fold_name_list = {'ABCD'};
                    GlobalConfigObj.dataset_list = {'040119_PtK1_S01_01_phase_3_DMSO_nd_03'; '040119_PtK1_S01_01_phase_2_DMSO_nd_02'; '040119_PtK1_S01_01_phase_2_DMSO_nd_01'; '040119_PtK1_S01_01_phase_ROI2'; '040119_PtK1_S01_01_phase'};
                    GlobalConfigObj.fold_name_list = {'ABCD';'ABCE'; 'ABDE'; 'ACDE'; 'BCDE'};
%                    GlobalConfigObj.fold_name_list = {'A'; 'B'; 'C'; 'D'; 'E'};
                    GlobalConfigObj.max_dist_pixel = 3;

                    % blue, skyblue, red, green, yellow, orange, pink, violet
                    GlobalConfigObj.graph_colors = [[0, 0.4470, 0.7410];
                                    [0.3010, 0.7450, 0.9330];
                                    [0.6350, 0.0780, 0.1840];
                                    [0.4660, 0.6740, 0.1880];
                                    [0.9290, 0.6940, 0.1250];
                                    [0.8500, 0.3250, 0.0980];
                                    [247,129,191]/255;
                                    [0.4940, 0.1840, 0.5560]];

                    % blue, pink, skyblue, red, green, light yellow, gray, yellow, orange, violet
%                    GlobalConfigObj.graph_colors = [[0, 0.4470, 0.7410];
%                                    [247,129,191]/255;
%                                    [0.3010, 0.7450, 0.9330];
%                                    [0.6350, 0.0780, 0.1840];
%                                    [0.4660, 0.6740, 0.1880];
%                                    [255,255,153]/255;
%                                    [153,153,153]/255;
%                                    [0.9290, 0.6940, 0.1250];
%                                    [0.8500, 0.3250, 0.0980];
%                                    [0.4940, 0.1840, 0.5560]];

                    % blue, pink, darkgray, skyblue, red, green, light yellow, gray, yellow, orange, violet
%                    GlobalConfigObj.graph_colors = [[0, 0.4470, 0.7410];
%                                    [247,129,191]/255;
%                                    [53,53,53]/255;
%                                    [0.3010, 0.7450, 0.9330];
%                                    [0.6350, 0.0780, 0.1840];
%                                    [0.4660, 0.6740, 0.1880];
%                                    [255,255,153]/255;
%                                    [153,153,153]/255;
%                                    [0.9290, 0.6940, 0.1250];
%                                    [0.8500, 0.3250, 0.0980];
%                                    [0.4940, 0.1840, 0.5560]];

                    % blue, yellow, pink, orange, green, red
%                    GlobalConfigObj.graph_colors = [[0, 0.4470, 0.7410];
%                                    [0.9290, 0.6940, 0.1250];
%                                    [247,129,191]/255;
%                                    [0.8500, 0.3250, 0.0980];
%                                    [0.4660, 0.6740, 0.1880];
%                                    [0.6350, 0.0780, 0.1840];];


                    % orange, green
%                    GlobalConfigObj.graph_colors = [[0.8500, 0.3250, 0.0980];
%                                                    [0.4660, 0.6740, 0.1880]];
                    % For one dataset only
                    GlobalConfigObj.dataset_split_list = [length(GlobalConfigObj.dataset_list)];
                    GlobalConfigObj.dataset_interval_list = [1];
                end


            elseif self.round_num == 2
                if contains(prediction_path, '_VGG')
                    GlobalConfigObj.repeat_max = 1;
                    GlobalConfigObj.img_root_path = '/media/bch_drive/Public/JunbongJang/Segmentation/assets/';
                    GlobalConfigObj.mask_type = '/mask_fixed/';
                    GlobalConfigObj.img_type = '/img/';
                    GlobalConfigObj.dataset_list = {'040119_PtK1_S01_01_phase_3_DMSO_nd_03'};
                    GlobalConfigObj.fold_name_list = {'ABCD'};
                    GlobalConfigObj.max_dist_pixel = 3;


                elseif contains(prediction_path, 'paxillin_TIRF')
                    GlobalConfigObj.repeat_max = 1;
                    GlobalConfigObj.img_root_path = '/media/bch_drive/Public/JunbongJang/Segmentation/assets/';
                    GlobalConfigObj.mask_type = '/mask/';
                    GlobalConfigObj.img_type = '/img/';
                    GlobalConfigObj.dataset_list = {'Paxilin-HaloTMR-TIRF3'};
                    GlobalConfigObj.fold_name_list = {'ABCDE'};
                    GlobalConfigObj.max_dist_pixel = 5;
                end
            end


            % --------------------------------- Testing -------------------------------------
%            size(GlobalConfigObj.prediction_path_list)
%            size(GlobalConfigObj.display_names)
%            size(GlobalConfigObj.dataset_list)
%            size(GlobalConfigObj.fold_name_list)

            assert(GlobalConfigObj.total_prediction_path_num > 0)
            assert(size(GlobalConfigObj.prediction_path_list, 1) == size(GlobalConfigObj.display_names, 1))
            assert(size(GlobalConfigObj.dataset_list, 1) == size(GlobalConfigObj.fold_name_list, 1))
            assert(GlobalConfigObj.chosen_frame_index <= size(GlobalConfigObj.frame_list, 2))

        end

        function concatenated_display_name = concat_display_names(self)
            % concatenate display name to name the folder
            for display_index = 1 : self.total_prediction_path_num
                if display_index == 1
                    concatenated_display_name = self.display_names{display_index};
                else
                    concatenated_display_name = [concatenated_display_name, '_', self.display_names{display_index}];
                end
            end
        end

    end  % methods end


    methods(Static)
        function ylim_vector = find_graph_ylim(mean_matrix, error_ci_matrix, upper_offset_bool)
            min_data = min(mean_matrix,[],'all');
            max_data = max(mean_matrix,[],'all');

            if ndims(mean_matrix) == 2
                [x,y]=find(mean_matrix==min_data);
                min_data = min_data - error_ci_matrix(x,y,1);

                [x,y]=find(mean_matrix==max_data);
                max_data = max_data + error_ci_matrix(x,y,2);
            elseif ndims(mean_matrix) == 3
                [x,y,z] = ind2sub(size(mean_matrix),find(mean_matrix == min_data));
                min_data = min_data - error_ci_matrix(x,y,z,1);

                [x,y,z] = ind2sub(size(mean_matrix),find(mean_matrix == max_data));
                max_data = max_data + error_ci_matrix(x,y,z,2);
            else
                error('find_graph_ylim: wrong dimensions');
            end

            bottom_n = fix(min_data/0.005);
            lower_bound = bottom_n*0.005;

            top_n = fix(max_data/0.005);
            upper_bound = (top_n+1)*0.005;

            if upper_offset_bool == 1
                upper_bound_offset = (upper_bound-lower_bound)/6;
                upper_bound = upper_bound + upper_bound_offset;
            end

            if upper_bound > 1
                upper_bound=1;
            end
            if lower_bound < 0
                lower_bound = 0;
            end

            ylim_vector = [lower_bound, upper_bound];
        end
    end

end  % class ends
