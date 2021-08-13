% Junbong Jang
% Modified on 11/23/2020
% For comparing two models across several frames

addpath('..');
constants_struct = GlobalConfig().update_config('');

for prediction_num = 1 : length(constants_struct.prediction_path_list)
    prediction_path_root = constants_struct.prediction_path_list{prediction_num}
    constants_struct = constants_struct.update_config(prediction_path_root);

    for data_index  = 1 : length(constants_struct.dataset_list)
        dataset_name = constants_struct.dataset_list{data_index, 1}
        fold_name = constants_struct.fold_name_list{data_index, 1}

        mask_path = [constants_struct.img_root_path, dataset_name, constants_struct.mask_type]; % directory containing masked (labeled) images
        img_path = [constants_struct.img_root_path, dataset_name, constants_struct.img_type]; % directory containing raw (original) images

        for frame_index = 1 : size(constants_struct.frame_list, 2)
            frame_num = constants_struct.frame_list(prediction_num, frame_index);
%            visualize_bool = frame_index == size(constants_struct.frame_list, 2);  % if last frame
            visualize_bool = 0;

            for repeat_index = 1: constants_struct.repeat_max
                disp([prediction_num, data_index, frame_num, repeat_index]);
                prediction_path = [prediction_path_root, dataset_name, '/frame', num2str(frame_num), '_' fold_name , '_repeat', num2str(repeat_index-1), '/'];
                save_path = [constants_struct.root_path, 'evaluation_f1/results/', constants_struct.concat_display_names(), ...
                            '/repeat', num2str(repeat_index), '/', ...
                             constants_struct.display_names{prediction_num} ,'_', dataset_name, '_', fold_name, '/']
                 constants_struct.concat_display_names()

%                overlap_mask_prediction(prediction_path, mask_path, img_path, frame_num, constants_struct.max_dist_pixel, visualize_bool, save_path);
%                overlap_mask_only(mask_path, img_path, save_path);
                overlap_prediction_only(prediction_path, img_path, save_path);
%                draw_edge_evolution(prediction_path, img_path, frame_num, save_path);
            end

        end
    end
end

