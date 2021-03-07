% Junbong Jang
% Modified on 1/2/2021
% For comparing two models across several frames

addpath('..');
constants_struct = GlobalConfig().update_config('');
repeat_index = 1
first_model_index = 1;
second_model_index = 2;

for data_index  = 1 : length(constants_struct.dataset_list)
    dataset_name = constants_struct.dataset_list{data_index, 1}
    fold_name = constants_struct.fold_name_list{data_index, 1}

    mask_path = [constants_struct.img_root_path, dataset_name, constants_struct.mask_type]; % directory containing masked (labeled) images
    img_path = [constants_struct.img_root_path, dataset_name, constants_struct.img_type]; % directory containing raw (original) images

    for frame_index = 1 : size(constants_struct.frame_list, 2)
        frame_num = constants_struct.frame_list(1, frame_index);

        disp([data_index, frame_num, repeat_index]);
        prediction_path_list = {[constants_struct.prediction_path_list{first_model_index}, dataset_name, '/frame', num2str(frame_num), '_' fold_name , '_repeat', num2str(repeat_index-1), '/'];
                                [constants_struct.prediction_path_list{second_model_index}, dataset_name, '/frame', num2str(frame_num), '_' fold_name , '_repeat', num2str(repeat_index-1), '/']};

        save_path = [constants_struct.root_path, 'evaluation_f1/results/', constants_struct.concat_display_names(), ...
                    '/repeat', num2str(repeat_index), '/compare_', dataset_name, '_', fold_name, '/'];

        overlap_compare(prediction_path_list, mask_path, img_path, frame_num, constants_struct.max_dist_pixel, save_path);

    end
end

