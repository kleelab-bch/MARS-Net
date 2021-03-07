function draw_boxplot(data, saved_folder, metric_type, graph_colors, display_names)
    size(data)
    figure;
    output_fig_size = [1452, 834];
    fig_res = 300;
    set(gcf,'paperunits','inches','paperposition',[0 0 output_fig_size/fig_res]);

    hold on
    for a_col=1:size(data,2)
        scatter(ones(size(data(:,a_col))).*(a_col+(rand(size(data(:,a_col)))-0.5)/(8/size(data,2))), data(:,a_col), 2, ...
       'filled', 'MarkerFaceColor', graph_colors(a_col,:), 'MarkerFaceAlpha',0.2);
    end

    boxplot(data, 'boxstyle', 'outline',  'Symbol', '', 'Labels', {});

    % change box color
    h = findobj(gca,'Tag','Box');
    for j=1:length(h)
        color_j = 1 + length(h) - j;
        patch(get(h(j),'XData'),get(h(j),'YData'),graph_colors(color_j,:),'FaceAlpha',0.4);
    end
    % change median color
    lines = findobj(gcf, 'type', 'line', 'Tag', 'Median');
    set(lines, 'Color', 'black');


    hold off

    xticklabels({});
    ylabel(metric_type);
    min_data = min(data,[],'all');
    max_data = max(data,[],'all');
    top_offset = (max_data-min_data)/6;
    ylim([min_data-0.05, max_data+top_offset]);
    axis square;
    box on;
    set(gca, 'fontsize', 10, 'fontname', 'Arial');

    saveas(gcf, [saved_folder, '/boxplot_', metric_type, '.fig']);
    print([saved_folder, '/boxplot_', metric_type, '.svg'],'-dsvg',['-r',num2str(fig_res)]);
    close;
end