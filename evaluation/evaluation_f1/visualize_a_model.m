
addpath('..');
combined_data = [model_F_score; model_precision; model_recall];

% blue, orange, green
graph_colors = [[0, 0.4470, 0.7410];
                [0.8500, 0.3250, 0.0980];
                [0.4660, 0.6740, 0.1880];];

draw_violinplot(combined_data.', "./results", 'a_model', graph_colors);