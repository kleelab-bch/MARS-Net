%%
% Junbong Jang
% 10/6/2020
% draw multiple violin plot comparing two models with f1, precision, recall and diec

addpath('evaluation_f1')
addpath('evaluation_dice')

constants_struct = GlobalConfig().update_config('');

%% For comparing two model at one specific frame
%chosen_first_model_index = 1;
%chosen_second_model_index = 3;
prediction_counter = 1;
for prediction_counter = 1:length(constants_struct.prediction_path_list)-1
    chosen_first_model_index = prediction_counter
    chosen_second_model_index = chosen_first_model_index + 1
    name_pair = [constants_struct.display_names{chosen_first_model_index}, '_', constants_struct.display_names{chosen_second_model_index}]
    root_save_path = [constants_struct.root_path, 'generated/violin_compare/', constants_struct.concat_display_names(), '/', name_pair, '/'];
    first_model_constants = constants_struct.update_config(constants_struct.prediction_path_list{chosen_first_model_index});
    second_model_constants = constants_struct.update_config(constants_struct.prediction_path_list{chosen_second_model_index});

    frame_index = constants_struct.chosen_frame_index;

    data_repeat_combined_first_model_f1 = [];
    data_repeat_combined_first_model_precision = [];
    data_repeat_combined_first_model_recall = [];
    data_repeat_combined_first_model_dice_coeff = [];
    data_repeat_combined_second_model_f1 = [];
    data_repeat_combined_second_model_precision = [];
    data_repeat_combined_second_model_recall = [];
    data_repeat_combined_second_model_dice_coeff = [];

    data_split_repeat_combined_first_model_f1 = [];
    data_split_repeat_combined_first_model_precision = [];
    data_split_repeat_combined_first_model_recall = [];
    data_split_repeat_combined_first_model_dice_coeff = [];
    data_split_repeat_combined_second_model_f1 = [];
    data_split_repeat_combined_second_model_precision = [];
    data_split_repeat_combined_second_model_recall = [];
    data_split_repeat_combined_second_model_dice_coeff = [];
    dataset_split_index = 1;

    for data_index  = 1 : length(constants_struct.dataset_list)

        first_model_frame = num2str(constants_struct.frame_list(chosen_first_model_index, frame_index));
        second_model_frame = num2str(constants_struct.frame_list(chosen_second_model_index, frame_index));

        saved_path = [root_save_path, first_model_constants.dataset_list{data_index, 1}, '_', ...
                     second_model_constants.dataset_list{data_index, 1}, '_', first_model_frame, '_', second_model_frame, '/'];
        mkdir(saved_path);

        repeat_combined_first_model_f1 = [];
        repeat_combined_first_model_precision = [];
        repeat_combined_first_model_recall = [];
        repeat_combined_first_model_dice_coeff = [];
        repeat_combined_second_model_f1 = [];
        repeat_combined_second_model_precision = [];
        repeat_combined_second_model_recall = [];
        repeat_combined_second_model_dice_coeff = [];

        for repeat_index = 1 : constants_struct.repeat_max
            % get f1 data
            try
                f1_folder_first = [constants_struct.root_path, 'evaluation_f1/results/', constants_struct.concat_display_names(), ...
                                    '/repeat', num2str(repeat_index), '/', ...
                                     constants_struct.display_names{chosen_first_model_index} ,'_', first_model_constants.dataset_list{data_index, 1}, '_', first_model_constants.fold_name_list{data_index, 1}, '/'];
                f1_folder_second = [constants_struct.root_path, 'evaluation_f1/results/', constants_struct.concat_display_names(), ...
                                '/repeat', num2str(repeat_index), '/', ...
                                 constants_struct.display_names{chosen_second_model_index} ,'_', second_model_constants.dataset_list{data_index, 1}, '_', second_model_constants.fold_name_list{data_index, 1}, '/'];

                first_model_loaded_data = load([f1_folder_first, 'Recall_Precision_F_score_frame', first_model_frame,'.mat']);
                second_model_loaded_data = load([f1_folder_second, 'Recall_Precision_F_score_frame', second_model_frame,'.mat']);

                % ----------- combine data -------------
                if repeat_index == 1
                    repeat_combined_first_model_f1 = first_model_loaded_data.model_F_score(1:constants_struct.dataset_interval_list(dataset_split_index):end);
                    repeat_combined_first_model_precision = first_model_loaded_data.model_precision(1:constants_struct.dataset_interval_list(dataset_split_index):end);
                    repeat_combined_first_model_recall = first_model_loaded_data.model_recall(1:constants_struct.dataset_interval_list(dataset_split_index):end);

                    repeat_combined_second_model_f1 = second_model_loaded_data.model_F_score(1:constants_struct.dataset_interval_list(dataset_split_index):end);
                    repeat_combined_second_model_precision = second_model_loaded_data.model_precision(1:constants_struct.dataset_interval_list(dataset_split_index):end);
                    repeat_combined_second_model_recall = second_model_loaded_data.model_recall(1:constants_struct.dataset_interval_list(dataset_split_index):end);
                else
                    repeat_combined_first_model_f1 = repeat_combined_first_model_f1 + first_model_loaded_data.model_F_score(1:constants_struct.dataset_interval_list(dataset_split_index):end);
                    repeat_combined_first_model_precision = repeat_combined_first_model_precision + first_model_loaded_data.model_precision(1:constants_struct.dataset_interval_list(dataset_split_index):end);
                    repeat_combined_first_model_recall = repeat_combined_first_model_recall + first_model_loaded_data.model_recall(1:constants_struct.dataset_interval_list(dataset_split_index):end);

                    repeat_combined_second_model_f1 = repeat_combined_second_model_f1 + second_model_loaded_data.model_F_score(1:constants_struct.dataset_interval_list(dataset_split_index):end);
                    repeat_combined_second_model_precision = repeat_combined_second_model_precision + second_model_loaded_data.model_precision(1:constants_struct.dataset_interval_list(dataset_split_index):end);
                    repeat_combined_second_model_recall = repeat_combined_second_model_recall + second_model_loaded_data.model_recall(1:constants_struct.dataset_interval_list(dataset_split_index):end);
                end

                % if you uncomment this, comment out upper and lower block of codes
    %            repeat_combined_first_model_f1 = [repeat_combined_first_model_f1, first_model_loaded_data.model_F_score(1:constants_struct.dataset_interval_list(dataset_split_index):end)];
    %            repeat_combined_first_model_precision = [repeat_combined_first_model_precision, first_model_loaded_data.model_precision(1:constants_struct.dataset_interval_list(dataset_split_index):end) ];
    %            repeat_combined_first_model_recall = [repeat_combined_first_model_recall, first_model_loaded_data.model_recall(1:constants_struct.dataset_interval_list(dataset_split_index):end) ];
    %
    %            repeat_combined_second_model_f1 = [repeat_combined_second_model_f1, second_model_loaded_data.model_F_score(1:constants_struct.dataset_interval_list(dataset_split_index):end) ];
    %            repeat_combined_second_model_precision = [repeat_combined_second_model_precision, second_model_loaded_data.model_precision(1:constants_struct.dataset_interval_list(dataset_split_index):end) ];
    %            repeat_combined_second_model_recall = [repeat_combined_second_model_recall, second_model_loaded_data.model_recall(1:constants_struct.dataset_interval_list(dataset_split_index):end) ];


            catch ME
                disp(['skip ', num2str(prediction_counter), ' ', num2str(data_index), ' ', num2str(repeat_index)])
            end

        end

        % adding and averaging the technical replicate is necessary to reduce its statistical importance relative to biological replicates
        repeat_combined_first_model_f1 = repeat_combined_first_model_f1 / constants_struct.repeat_max;
        repeat_combined_first_model_precision = repeat_combined_first_model_precision / constants_struct.repeat_max;
        repeat_combined_first_model_recall = repeat_combined_first_model_recall / constants_struct.repeat_max;

        repeat_combined_second_model_f1 = repeat_combined_second_model_f1 / constants_struct.repeat_max;
        repeat_combined_second_model_precision = repeat_combined_second_model_precision / constants_struct.repeat_max;
        repeat_combined_second_model_recall = repeat_combined_second_model_recall / constants_struct.repeat_max;


        %% -------------- violinplot for comparing two methods at one frame and data -----------------------
%        violinplot_two_models(repeat_combined_first_model_f1, repeat_combined_second_model_f1,constants_struct.violin_compare_ylim, saved_path, 'F1', 'Model Comparison', constants_struct.display_names{chosen_first_model_index}, constants_struct.display_names{chosen_second_model_index});
%        violinplot_two_models(repeat_combined_first_model_precision, repeat_combined_second_model_precision, constants_struct.violin_compare_ylim, saved_path, 'Precision', 'Model Comparison', constants_struct.display_names{chosen_first_model_index}, constants_struct.display_names{chosen_second_model_index});
%        violinplot_two_models(repeat_combined_first_model_recall, repeat_combined_first_model_recall, constants_struct.violin_compare_ylim, saved_path, 'Recall', 'Model Comparison', constants_struct.display_names{chosen_first_model_index}, constants_struct.display_names{chosen_second_model_index});
%        violinplot_two_models(repeat_combined_first_model_dice_coeff, repeat_combined_second_model_dice_coeff, constants_struct.violin_compare_ylim, saved_path, 'Dice', 'Model Comparison', constants_struct.display_names{chosen_first_model_index}, constants_struct.display_names{chosen_second_model_index});
%
%        draw_histogram(repeat_combined_first_model_f1, repeat_combined_second_model_f1, saved_path, 'F1', 'Histogram', constants_struct.display_names{chosen_first_model_index}, constants_struct.display_names{chosen_second_model_index})
%        violinplot_diff(repeat_combined_second_model_f1 - repeat_combined_first_model_f1, ...
%                         repeat_combined_second_model_precision - repeat_combined_first_model_precision, ...
%                         repeat_combined_second_model_recall - repeat_combined_first_model_recall, ...
%                         constants_struct.violin_diff_ylim, saved_path, 'Performance Difference', 'Model Comparison');
        %% -------------------------------------------------------------------------------------------------

        data_split_repeat_combined_first_model_f1 = [data_split_repeat_combined_first_model_f1, repeat_combined_first_model_f1];
        data_split_repeat_combined_first_model_precision = [data_split_repeat_combined_first_model_precision, repeat_combined_first_model_precision];
        data_split_repeat_combined_first_model_recall = [data_split_repeat_combined_first_model_recall, repeat_combined_first_model_recall];
        data_split_repeat_combined_first_model_dice_coeff = [data_split_repeat_combined_first_model_dice_coeff, repeat_combined_first_model_dice_coeff];
        data_split_repeat_combined_second_model_f1 = [data_split_repeat_combined_second_model_f1, repeat_combined_second_model_f1];
        data_split_repeat_combined_second_model_precision = [data_split_repeat_combined_second_model_precision, repeat_combined_second_model_precision];
        data_split_repeat_combined_second_model_recall = [data_split_repeat_combined_second_model_recall, repeat_combined_second_model_recall];
        data_split_repeat_combined_second_model_dice_coeff = [data_split_repeat_combined_second_model_dice_coeff, repeat_combined_second_model_dice_coeff];

        % per dataset summary for evaluating multi-modal model
        if data_index == constants_struct.dataset_split_list(dataset_split_index)
            saved_path = [root_save_path, num2str(dataset_split_index) ,'_dataset_summary_', first_model_frame, '_', second_model_frame, '/'];
            mkdir(saved_path);

            %% violinplot for comparing two methods at one frame across datasets
%            violinplot_two_models(data_split_repeat_combined_first_model_f1, data_split_repeat_combined_second_model_f1, constants_struct.violin_compare_ylim, saved_path, 'F1', 'Model Comparison', constants_struct.display_names{chosen_first_model_index}, constants_struct.display_names{chosen_second_model_index});
%            violinplot_two_models(data_split_repeat_combined_first_model_precision, data_split_repeat_combined_second_model_precision, constants_struct.violin_compare_ylim, saved_path, 'Precision', 'Model Comparison', constants_struct.display_names{chosen_first_model_index}, constants_struct.display_names{chosen_second_model_index});
%            violinplot_two_models(data_split_repeat_combined_first_model_recall, data_split_repeat_combined_second_model_recall, constants_struct.violin_compare_ylim, saved_path, 'Recall', 'Model Comparison', constants_struct.display_names{chosen_first_model_index}, constants_struct.display_names{chosen_second_model_index});
%            violinplot_two_models(data_split_repeat_combined_first_model_dice_coeff, data_split_repeat_combined_second_model_dice_coeff, constants_struct.violin_compare_ylim, saved_path, 'Dice', 'Model Comparison', constants_struct.display_names{chosen_first_model_index}, constants_struct.display_names{chosen_second_model_index});
%
%            draw_histogram(data_split_repeat_combined_first_model_f1, data_split_repeat_combined_second_model_f1, saved_path, 'F1', 'Histogram', constants_struct.display_names{chosen_first_model_index}, constants_struct.display_names{chosen_second_model_index})
%
%            violinplot_diff(data_split_repeat_combined_second_model_f1 - data_split_repeat_combined_first_model_f1, ...
%                             data_split_repeat_combined_second_model_precision - data_split_repeat_combined_first_model_precision, ...
%                             data_split_repeat_combined_second_model_recall - data_split_repeat_combined_first_model_recall, ...
%                             constants_struct.violin_diff_ylim, saved_path, 'Performance Difference', 'Model Comparison');

            data_repeat_combined_first_model_f1 = [data_repeat_combined_first_model_f1, data_split_repeat_combined_first_model_f1];
            data_repeat_combined_first_model_precision = [data_repeat_combined_first_model_precision, data_split_repeat_combined_first_model_precision];
            data_repeat_combined_first_model_recall = [data_repeat_combined_first_model_recall, data_split_repeat_combined_first_model_recall];
            data_repeat_combined_first_model_dice_coeff = [data_repeat_combined_first_model_dice_coeff, data_split_repeat_combined_first_model_dice_coeff];
            data_repeat_combined_second_model_f1 = [data_repeat_combined_second_model_f1, data_split_repeat_combined_second_model_f1];
            data_repeat_combined_second_model_precision = [data_repeat_combined_second_model_precision, data_split_repeat_combined_second_model_precision];
            data_repeat_combined_second_model_recall = [data_repeat_combined_second_model_recall, data_split_repeat_combined_second_model_recall];
            data_repeat_combined_second_model_dice_coeff = [data_repeat_combined_second_model_dice_coeff, data_split_repeat_combined_second_model_dice_coeff];

            data_split_repeat_combined_first_model_f1 = [];
            data_split_repeat_combined_first_model_precision = [];
            data_split_repeat_combined_first_model_recall = [];
            data_split_repeat_combined_first_model_dice_coeff = [];
            data_split_repeat_combined_second_model_f1 = [];
            data_split_repeat_combined_second_model_precision = [];
            data_split_repeat_combined_second_model_recall = [];
            data_split_repeat_combined_second_model_dice_coeff = [];
            dataset_split_index = dataset_split_index + 1;
        end

    end

    % get dice data
    dice_folder_first = [constants_struct.root_path, 'evaluation_dice/dice_generated/round1_', constants_struct.display_names{chosen_first_model_index}];
    dice_folder_second = [constants_struct.root_path, 'evaluation_dice/dice_generated/round1_', constants_struct.display_names{chosen_second_model_index}];
    data_repeat_combined_first_model_dice_coeff = readNPY([dice_folder_first '/dice_value.npy']);
    data_repeat_combined_second_model_dice_coeff = readNPY([dice_folder_second '/dice_value.npy']);

    % all datasets summary
    saved_path = [root_save_path, 'all_dataset_summary_', first_model_frame, '_', second_model_frame, '/'];
    mkdir(saved_path);

    draw_boxplot([data_repeat_combined_first_model_dice_coeff, data_repeat_combined_second_model_dice_coeff], saved_path, 'Dice', constants_struct.graph_colors, constants_struct.display_names)

    %% violinplot for comparing two methods at one frame across datasets
    violinplot_two_models(data_repeat_combined_first_model_f1, data_repeat_combined_second_model_f1, saved_path, 'F1', 'Model Comparison', constants_struct.display_names{chosen_first_model_index}, constants_struct.display_names{chosen_second_model_index}, constants_struct.graph_colors);
    violinplot_two_models(data_repeat_combined_first_model_precision, data_repeat_combined_second_model_precision, saved_path, 'Precision', 'Model Comparison', constants_struct.display_names{chosen_first_model_index}, constants_struct.display_names{chosen_second_model_index}, constants_struct.graph_colors);
    violinplot_two_models(data_repeat_combined_first_model_recall, data_repeat_combined_second_model_recall, saved_path, 'Recall', 'Model Comparison', constants_struct.display_names{chosen_first_model_index}, constants_struct.display_names{chosen_second_model_index}, constants_struct.graph_colors);
    violinplot_two_models(data_repeat_combined_first_model_dice_coeff, data_repeat_combined_second_model_dice_coeff, saved_path, 'Dice', 'Model Comparison', constants_struct.display_names{chosen_first_model_index}, constants_struct.display_names{chosen_second_model_index}, constants_struct.graph_colors);

    violinplot_diff(data_repeat_combined_second_model_f1 - data_repeat_combined_first_model_f1, ...
                     data_repeat_combined_second_model_precision - data_repeat_combined_first_model_precision, ...
                     data_repeat_combined_second_model_recall - data_repeat_combined_first_model_recall, ...
                     constants_struct.violin_diff_ylim, saved_path, 'Performance Difference', 'Model Comparison');
end