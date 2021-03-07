% Author: Junbong Jang
% Date 2/22/2021
% For evaluating vality of Dice with respect to F1
% Throug this code, we found that Dice evaluation had some error in edge extraction part.

constants_struct = GlobalConfig().update_config('')
frame_list = constants_struct.frame_list;

dice_load_folder = [constants_struct.root_path, 'evaluation_dice/dice_coeff_compare/', constants_struct.dice_folder_name];
f1_load_folder = [constants_struct.root_path, 'evaluation_f1/results/', constants_struct.concat_display_names()];
saved_folder = [constants_struct.root_path, 'generated/f1_dice_combined/', constants_struct.concat_display_names(), '/'];
mkdir(saved_folder);


all_model_dice_coff = [];
all_F1_values = [];

for prediction_num = 1:constants_struct.total_prediction_path_num

    prediction_path_root = constants_struct.prediction_path_list{prediction_num};
    constants_struct = constants_struct.update_config(prediction_path_root);

    for fold_index = 1 : length(constants_struct.fold_name_list)
        dataset_name = constants_struct.dataset_list{fold_index, 1};
        fold_name = constants_struct.fold_name_list{fold_index, 1};
        frame_index = 1;
        for a_frame = frame_list(1,:)

            repeat_combined_model_dice_coff = [];
            repeat_combined_F1_values = [];
            repeat_combined_image_list = [];
            repeat_combined_F1_image_list = [];
            for repeat_index = 1 : constants_struct.repeat_max
                % get Dice
                folder_path = [dice_load_folder '/frame' num2str(a_frame), '_', fold_name, '_repeat', num2str(repeat_index-1)];
                try
                    model_dice_coef = readNPY([folder_path '/model',  num2str(prediction_num-1), '_dice_value.npy']);
                    model_image_list = load([folder_path '/model',  num2str(prediction_num-1), '_image_list.mat']);
                    model_image_list = model_image_list.('image_list');
                    repeat_combined_model_dice_coff = [repeat_combined_model_dice_coff, model_dice_coef];
                    repeat_combined_image_list = [repeat_combined_image_list; model_image_list];
                catch ME
                    disp('Dice skip@@@@@@')
                end

                % get F1
                try
                    fpr_score = load([f1_load_folder, '/repeat', num2str(repeat_index), '/', ...
                                    constants_struct.display_names{prediction_num} ,'_', dataset_name, '_', fold_name, ...
                                    '/Recall_Precision_F_score_frame', num2str(a_frame), '.mat']);
                    repeat_combined_F1_values = [repeat_combined_F1_values, fpr_score.('model_F_score')];
                    repeat_combined_F1_image_list = [repeat_combined_F1_image_list; fpr_score.('image_list')];
                catch ME
                    disp(['F1 skip@@@@@@', num2str(prediction_num),' ', num2str(fold_index), ' ', num2str(a_frame), ' ', num2str(repeat_index)])
                end
            end

            frame_index = frame_index + 1;
            save(fullfile(saved_folder, [constants_struct.display_names{prediction_num}, '_', fold_name, '_', num2str(a_frame), '.mat']), 'repeat_combined_F1_values', 'repeat_combined_model_dice_coff', 'repeat_combined_image_list', 'repeat_combined_F1_image_list');

            draw_line_graph(repeat_combined_F1_values, repeat_combined_model_dice_coff, saved_folder, [constants_struct.display_names{prediction_num}, '_', fold_name, '_', num2str(a_frame)])

            all_F1_values = [all_F1_values, repeat_combined_F1_values];
            all_model_dice_coff = [all_model_dice_coff, repeat_combined_model_dice_coff];

         end
    end
end

draw_line_graph(all_F1_values, all_model_dice_coff, saved_folder, 'All')

% ------- Draw Linear regression prediction
function draw_line_graph(F1_values, dice_values, saved_folder, save_name_prefix)
    mdl = fitlm(F1_values,dice_values)
    line_slope = mdl.Coefficients{'x1','Estimate'};
    line_intercept = mdl.Coefficients{'(Intercept)','Estimate'};
    %sort to calculate r2
    [sorted_F1_values,I] = sort(F1_values);
    sorted_dice_values = dice_values(I);
    %calc r2
    disp('@@@@@@@')
    R = corrcoef(sorted_dice_values,sorted_F1_values*line_slope+line_intercept);
    Rsq = R(1,2).^2

    figure;
    output_fig_size = [1500, 900]; % width and height in inches
    fig_res = 300;
    set(gcf,'paperunits','inches','paperposition',[0 0 output_fig_size/fig_res]);

    hold on
    plot([0,1],[0,1],'--')  % diagonal line
    scatter(F1_values, dice_values,'MarkerEdgeColor',[0.8500, 0.3250, 0.0980])
    plot(F1_values, F1_values*line_slope+line_intercept,'--','Color',[0.8500, 0.3250, 0.0980])  % predicted line

    xlabel('F1');
    ylabel('Dice');
    xlim([0.6, 1]);
    ylim([0.6, 1]);

    dim = [0.78 0.92 0 0];
    str = ['R^2: ', num2str( round(Rsq, 3) ), newline, 'Slope:', num2str( round(line_slope, 3) ), newline, 'y-interc:', num2str( round(line_intercept, 3) )];
    annotation('textbox',dim,'String',str,'FitBoxToText','on', 'EdgeColor','none');

    axis square;
    box on

    saveas(gcf, fullfile(saved_folder, [save_name_prefix, '_F1_Dice_scatter.fig']));
    print(fullfile(saved_folder, [save_name_prefix, '_F1_Dice_scatter.svg']),'-dsvg',['-r',num2str(fig_res)]);
end
