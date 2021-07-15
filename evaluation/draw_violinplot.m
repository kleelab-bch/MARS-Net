function draw_violinplot(data, saved_folder, metric_type, graph_colors, display_names)
    size(data)
    figure;
    output_fig_size = [1500, 850];
    fig_res = 300;
    set(gcf,'paperunits','inches','paperposition',[0 0 output_fig_size/fig_res]);

    label = ones(size(data));
    for a_model_index = 1:size(data,2)
        label(:, a_model_index) = label(:, a_model_index) * a_model_index;
    end
    vs = violinplot(data, label);

    for a_model_index = 1:size(data,2)
        vs(1,a_model_index).ViolinColor = graph_colors(a_model_index,:);
    end


    xticklabels({});
    ylabel(metric_type);
    min_data = min(data,[],'all');
    max_data = max(data,[],'all');
    top_offset = (max_data-min_data)/6;
%    top_offset = (max_data-min_data)/20;
    bottom_offset = (max_data-min_data)/20;
    
    ylim([min_data-bottom_offset, max_data+top_offset]);
    if size(data,2) == 4
        xlim([0.4,size(data,2)+0.6]);
    end
    axis square;
    box on;
    set(gca, 'fontsize', 10, 'fontname', 'Arial');

    saveas(gcf, [saved_folder, '/violinplot_', metric_type, '.fig']);
    print([saved_folder, '/violinplot_', metric_type, '.svg'],'-dsvg',['-r',num2str(fig_res)]);
    close;
end