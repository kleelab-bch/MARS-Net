% visualize the bar with error for each groups

addpath('..');
constants_struct = GlobalConfig().update_config('')
frame_list = constants_struct.frame_list;

saved_folder = [constants_struct.root_path, 'evaluation_dice/dice_coeff_compare/', constants_struct.dice_folder_name];
mkdir(saved_folder);

%% get the calculated measurements and average them
for fold_index = 1 : length(constants_struct.fold_name_list)
    dice_coef_mean_across_frame = zeros(constants_struct.total_prediction_path_num, size(frame_list,2));
    dice_coef_errorci = zeros(constants_struct.total_prediction_path_num, size(frame_list,2), 2);
    for model_num = 1:constants_struct.total_prediction_path_num
        frame_index = 1;
        for a_frame = frame_list(1,:)

            repetition_combined_model_dice_coff = [];
            for repeat_index = 1 : constants_struct.repeat_max
                folder_path = [saved_folder '/frame' num2str(a_frame), '_', constants_struct.fold_name_list{fold_index, 1}, '_repeat', num2str(repeat_index-1)];
                try
                    disp([folder_path '/model',  num2str(model_num-1), '_dice_value.npy'])
                    model_dice_coef = readNPY([folder_path '/model',  num2str(model_num-1), '_dice_value.npy']);
                    repetition_combined_model_dice_coff = [repetition_combined_model_dice_coff, model_dice_coef];
                catch ME
                    disp('skip!!!!!!')
                end
            end
            [mean_dice_coef, errci_dice_coef] = mean_errci(repetition_combined_model_dice_coff);
            dice_coef_mean_across_frame(model_num, frame_index) = mean_dice_coef;
            dice_coef_errorci(model_num, frame_index, :) = errci_dice_coef(1, 1, :);

            frame_index = frame_index + 1;
        end
    end
    save(fullfile(saved_folder, [constants_struct.fold_name_list{fold_index, 1}, '_dice_coef_across_frames.mat']), 'dice_coef_mean_across_frame', 'dice_coef_errorci');
end

%% visaluze average across all datasets to compare two models
disp('combine ----------------')
dice_coef_mean_across_model_datasets_frames = zeros(constants_struct.total_prediction_path_num, size(frame_list,2));
dice_coef_errorci_across_model_datasets_frames = zeros(constants_struct.total_prediction_path_num, size(frame_list,2), 2);

dice_coef_mean_across_datasets_frames = zeros(constants_struct.total_prediction_path_num, length(constants_struct.fold_name_list),size(frame_list,2));
dice_coef_errorci_across_datasets_frames = zeros(constants_struct.total_prediction_path_num, length(constants_struct.fold_name_list),size(frame_list,2),2);
for model_num = 1:constants_struct.total_prediction_path_num
    model_dice_coef_mean_across_frame = zeros(length(constants_struct.fold_name_list),size(frame_list,2));
    model_dice_coef_errorci_across_frame = zeros(length(constants_struct.fold_name_list),size(frame_list,2),2);

    for fold_index = 1 : length(constants_struct.fold_name_list)
        loaded_data = load(fullfile(saved_folder, [constants_struct.fold_name_list{fold_index, 1}, '_dice_coef_across_frames.mat']));
        model_dice_coef_mean_across_frame(fold_index,:) = loaded_data.dice_coef_mean_across_frame(model_num, :);
        model_dice_coef_errorci_across_frame(fold_index,:,:) = loaded_data.dice_coef_errorci(model_num, :, :);
    end
    dice_coef_mean_across_model_datasets_frames(model_num, :) = mean(model_dice_coef_mean_across_frame, 1);
    dice_coef_errorci_across_model_datasets_frames(model_num, :, 1) = mean(model_dice_coef_errorci_across_frame(:, :, 1), 1);
    dice_coef_errorci_across_model_datasets_frames(model_num, :, 2) = mean(model_dice_coef_errorci_across_frame(:, :, 2), 1);

    dice_coef_mean_across_datasets_frames(model_num, :, :) = model_dice_coef_mean_across_frame;
    dice_coef_errorci_across_datasets_frames(model_num, :, :, :) = model_dice_coef_errorci_across_frame;
end

%draw_line_with_error(dice_coef_mean_across_model_datasets_frames, dice_coef_errorci_across_model_datasets_frames, 'across datasets and frames', 'Dice', saved_folder, frame_list(1,:), constants_struct.dice_ylim, constants_struct.display_names);
draw_line_with_error_extra_lines(dice_coef_mean_across_model_datasets_frames, dice_coef_errorci_across_model_datasets_frames, ...
                                dice_coef_mean_across_datasets_frames, dice_coef_errorci_across_datasets_frames, 'across datasets and frames', ...
                                'Dice Coefficient', saved_folder, frame_list(1,:), constants_struct.graph_colors, constants_struct.dice_ylim, constants_struct.display_names);
round(dice_coef_mean_across_model_datasets_frames, 5)

draw_barplot(dice_coef_mean_across_model_datasets_frames(:,constants_struct.chosen_frame_index), dice_coef_errorci_across_model_datasets_frames(:,constants_struct.chosen_frame_index,:), saved_folder, ...
              'Dice Coefficient', constants_struct.graph_colors, constants_struct.dice_bar_ylim, constants_struct.display_names);



%% visualize line plots for each dataset across frames
% total_dice_coef_mean_across_frame = zeros(length(constants_struct.fold_name_list),size(frame_list,2));
% total_dice_coef_errorci = zeros(length(constants_struct.fold_name_list),size(frame_list,2),2);
% for fold_index = 1 : length(constants_struct.fold_name_list)
%     loaded_data = load(fullfile(saved_folder, [constants_struct.fold_name_list{fold_index, 1}, '_dice_coef_across_frames.mat']));
%     total_dice_coef_mean_across_frame(fold_index,:) = loaded_data.dice_coef_mean_across_frame(1, :)
%     total_dice_coef_errorci(fold_index,:,:) = loaded_data.dice_coef_errorci(1, :, :)
% end
% draw_line_with_error(total_dice_coef_mean_across_frame, total_dice_coef_errorci, 'Model Comparison', 'Dice', saved_folder, frame_list(1), constants_struct.dice_ylim, constants_struct.display_names);
%
% mean_dice_coef_across_frame_of_all_movies = mean(total_dice_coef_mean_across_frame,1)


%% visualize two line plots to compare two models of a dataset
% loaded_data = load(fullfile(saved_folder, [constants_struct.fold_name_list{1, 1}, '_dice_coef_across_frames.mat']));
% draw_boxplotwitherror_self_training(loaded_data.dice_coef_mean_across_frame, loaded_data.dice_coef_errorci, 'Model Comparison', saved_folder, [constants_struct.fold_name_list{1, 1}, '_dice_coef_errorplot'], frame_list,  constants_struct.display_names);
% loaded_data.dice_coef_mean_across_frame