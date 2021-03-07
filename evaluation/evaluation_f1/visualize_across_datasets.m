% Compare two models at one specific frame across datasets

addpath('..');
constants_struct = global_constants();
saved_folder = [constants_struct.root_path, 'evaluation_f1/results/'];

model1_train_frames = constants_struct.model1_train_frames;
model2_train_frames = constants_struct.model2_train_frames;

model1_F_total = [];
model2_F_total = [];
for cell_index  = 1 : length(constants_struct.dataset_list)
    cell_name = constants_struct.dataset_list{cell_index, 1};
    saved_path = [saved_folder, cell_name, '/', num2str(model1_train_frames),'_', num2str(model2_train_frames) , '/'];
    loaded_data = load([saved_path, 'Recall_Precision_F_score.mat']);
    model1_F_total = [model1_F_total, loaded_data.model1_F_score];
    model2_F_total = [model2_F_total, loaded_data.model2_F_score];
end

violinplot_helper(model1_F_total, model2_F_total, saved_folder, 'F1', ['frame ' num2str(model1_train_frames),' ', num2str(model2_train_frames), ' across datasets'], constants_struct.condition_one, constants_struct.condition_two)
hypothesis_testing(model1_F_total, model2_F_total);