
addpath('..');
constants_struct = GlobalConfig().update_config('');

saved_folder = [constants_struct.root_path, 'evaluation_f1/results/', constants_struct.concat_display_names(),'/'];

f1_across_frame_across_dataset_across_predictions = [];
precision_across_frame_across_dataset_across_predictions = [];
recall_across_frame_across_dataset_across_predictions = [];
dice_across_frame_across_dataset_across_predictions = [];


for prediction_num = 1:constants_struct.total_prediction_path_num
    prediction_path_root = constants_struct.prediction_path_list{prediction_num};
    constants_struct = constants_struct.update_config(prediction_path_root);

    f1_across_frame_across_dataset = [];
    precision_across_frame_across_dataset = [];
    recall_across_frame_across_dataset = [];
    dice_across_frame_across_dataset = [];

    dataset_split_index = 1;
    for data_index = 1 : length(constants_struct.dataset_list)
%        disp([prediction_num, data_index])
        dataset_name = constants_struct.dataset_list{data_index, 1};
        fold_name = constants_struct.fold_name_list{data_index, 1};

        disp(['aggregate model ', num2str(prediction_num), '  dataset ' num2str(data_index)]);
        [f1_across_frames, f1_mean_across_frame, f1_errorci] = visualize_across_frames(dataset_name, fold_name, prediction_num, constants_struct.frame_list(prediction_num, :), constants_struct.repeat_max, saved_folder, constants_struct.display_names, 'F_score');
        [precision_across_frames, precision_mean_across_frame, precision_errorci] = visualize_across_frames(dataset_name, fold_name, prediction_num, constants_struct.frame_list(prediction_num, :), constants_struct.repeat_max, saved_folder, constants_struct.display_names, 'precision');
        [recall_across_frames, recall_mean_across_frame, recall_errorci] = visualize_across_frames(dataset_name, fold_name, prediction_num, constants_struct.frame_list(prediction_num, :), constants_struct.repeat_max, saved_folder, constants_struct.display_names, 'recall');
        [dice_across_frames, dice_mean_across_frame, dice_errorci] = visualize_across_frames(dataset_name, fold_name, prediction_num, constants_struct.frame_list(prediction_num, :), constants_struct.repeat_max, saved_folder, constants_struct.display_names, 'dice');

        % Store data for total datasets
        f1_mean_across_frame_across_dataset(prediction_num, data_index, :) = f1_mean_across_frame(:);
        f1_errorci_across_frame_across_dataset(prediction_num, data_index, :, :) = f1_errorci(:, :);
        precision_mean_across_frame_across_dataset(prediction_num, data_index, :) = precision_mean_across_frame(:);
        precision_errorci_across_frame_across_dataset(prediction_num, data_index, :, :) = precision_errorci(:, :);
        recall_mean_across_frame_across_dataset(prediction_num, data_index, :) = recall_mean_across_frame(:);
        recall_errorci_across_frame_across_dataset(prediction_num, data_index, :, :) = recall_errorci(:, :);
        dice_mean_across_frame_across_dataset(prediction_num, data_index, :) = dice_mean_across_frame(:);
        dice_errorci_across_frame_across_dataset(prediction_num, data_index, :, :) = dice_errorci(:, :);

        % combine data
        f1_across_frame_across_dataset = [f1_across_frame_across_dataset, f1_across_frames(constants_struct.chosen_frame_index, (1:constants_struct.dataset_interval_list(dataset_split_index):end))];
        precision_across_frame_across_dataset = [precision_across_frame_across_dataset, precision_across_frames(constants_struct.chosen_frame_index, (1:constants_struct.dataset_interval_list(dataset_split_index):end))];
        recall_across_frame_across_dataset = [recall_across_frame_across_dataset, recall_across_frames(constants_struct.chosen_frame_index, (1:constants_struct.dataset_interval_list(dataset_split_index):end))];
        dice_across_frame_across_dataset = [dice_across_frame_across_dataset, dice_across_frames(constants_struct.chosen_frame_index, (1:constants_struct.dataset_interval_list(dataset_split_index):end))];

        if data_index == constants_struct.dataset_split_list(dataset_split_index)
            dataset_split_index = dataset_split_index + 1;
        end

    end
    f1_across_frame_across_dataset_across_predictions = [f1_across_frame_across_dataset_across_predictions; f1_across_frame_across_dataset];
    precision_across_frame_across_dataset_across_predictions = [precision_across_frame_across_dataset_across_predictions; precision_across_frame_across_dataset];
    recall_across_frame_across_dataset_across_predictions = [recall_across_frame_across_dataset_across_predictions; recall_across_frame_across_dataset];
    dice_across_frame_across_dataset_across_predictions = [dice_across_frame_across_dataset_across_predictions; dice_across_frame_across_dataset];
end

% ------------------- Multiple Line plots across frames for each dataset --------
% draw_line_with_error(precision_mean_across_frame_across_dataset, precision_errorci_across_frame_across_dataset, 'Datasets', 'Precision', saved_folder, constants_struct.frame_list, constants_struct.fold_name_list);
% draw_line_with_error(recall_mean_across_frame_across_dataset, recall_errorci_across_frame_across_dataset, 'Datasets', 'Recall', saved_folder, constants_struct.frame_list, constants_struct.fold_name_list);
% draw_line_with_error(f1_mean_across_frame_across_dataset, f1_errorci_across_frame_across_dataset, 'Datasets', 'F1', saved_folder, constants_struct.frame_list, constants_struct.fold_name_list);

% ------------------- Two Line plot across frames and dataset -----------------------------
draw_violinplot(f1_across_frame_across_dataset_across_predictions.', saved_folder, 'F1', constants_struct.graph_colors, constants_struct.display_names);
draw_violinplot(recall_across_frame_across_dataset_across_predictions.', saved_folder, 'Recall', constants_struct.graph_colors, constants_struct.display_names);
draw_violinplot(precision_across_frame_across_dataset_across_predictions.', saved_folder, 'Precision', constants_struct.graph_colors, constants_struct.display_names);
draw_violinplot(dice_across_frame_across_dataset_across_predictions.', saved_folder, 'Dice', constants_struct.graph_colors, constants_struct.display_names);

%draw_boxplot(f1_across_frame_across_dataset_across_predictions.', saved_folder, 'F1', constants_struct.graph_colors, constants_struct.display_names);
%draw_boxplot(recall_across_frame_across_dataset_across_predictions.', saved_folder, 'Recall', constants_struct.graph_colors, constants_struct.display_names);
%draw_boxplot(precision_across_frame_across_dataset_across_predictions.', saved_folder, 'Precision', constants_struct.graph_colors, constants_struct.display_names);

draw_line_for_models(f1_mean_across_frame_across_dataset, f1_errorci_across_frame_across_dataset, constants_struct.total_prediction_path_num, 'F1', constants_struct.frame_list, constants_struct.chosen_frame_index, constants_struct.display_names, constants_struct.graph_colors, constants_struct.f1_ylim, constants_struct.f1_bar_ylim, saved_folder);
draw_line_for_models(precision_mean_across_frame_across_dataset, precision_errorci_across_frame_across_dataset, constants_struct.total_prediction_path_num, 'Precision', constants_struct.frame_list, constants_struct.chosen_frame_index, constants_struct.display_names, constants_struct.graph_colors, constants_struct.precision_ylim, constants_struct.precision_bar_ylim, saved_folder);
draw_line_for_models(recall_mean_across_frame_across_dataset, recall_errorci_across_frame_across_dataset, constants_struct.total_prediction_path_num, 'Recall', constants_struct.frame_list, constants_struct.chosen_frame_index, constants_struct.display_names, constants_struct.graph_colors, constants_struct.recall_ylim, constants_struct.recall_bar_ylim, saved_folder);
%draw_line_for_models(dice_mean_across_frame_across_dataset, dice_errorci_across_frame_across_dataset, constants_struct.total_prediction_path_num, 'Dice', constants_struct.frame_list, constants_struct.chosen_frame_index, constants_struct.display_names, constants_struct.graph_colors, constants_struct.dice_ylim, constants_struct.dice_bar_ylim, saved_folder);


function draw_line_for_models(mean_across_frame_across_dataset, errorci_across_frame_across_dataset, total_prediction_path_num, metric_type, frame_list, chosen_frame_index, display_names, graph_colors, ylim, bar_ylim, saved_folder)
    frame_list = frame_list(1,:);
    mean_dataset_across_frame = zeros(total_prediction_path_num, length(frame_list));
    errorci_dataset_across_frame = zeros(total_prediction_path_num, length(frame_list), 2);

    for prediction_num = 1:total_prediction_path_num
%        mean_across_frame_across_dataset: first index is model, second is dataset, third is frames
        mean_dataset_across_frame(prediction_num, :) = mean(mean_across_frame_across_dataset(prediction_num, :, :), 2);
        errorci_dataset_across_frame(prediction_num, :, 1) = mean(errorci_across_frame_across_dataset(prediction_num, :, :, 1), 2);
        errorci_dataset_across_frame(prediction_num, :, 2) = mean(errorci_across_frame_across_dataset(prediction_num, :, :, 2), 2);
    end

    draw_line_with_error_extra_lines(mean_dataset_across_frame, errorci_dataset_across_frame, mean_across_frame_across_dataset, ...
    errorci_across_frame_across_dataset, 'across datasets and frames', metric_type, saved_folder, frame_list, graph_colors, ylim, display_names);

    draw_barplot(mean_dataset_across_frame(:,chosen_frame_index), errorci_dataset_across_frame(:,chosen_frame_index,:), saved_folder, ...
              metric_type, graph_colors, bar_ylim, display_names);
end
