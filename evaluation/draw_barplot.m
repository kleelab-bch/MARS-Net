function draw_barplot(mean_across_frame, errorci, saved_folder, metric_type, graph_colors, graph_ylim, display_names)

    figure;
    output_fig_size = [1500, 850];
    fig_res = 300;
    set(gcf,'paperunits','inches','paperposition',[0 0 output_fig_size/fig_res]);
%    figure('Renderer', 'painters', 'Position', [0 0 700 300])
    hold on

%    h = barwitherr(errorci,mean_across_frame);    % Deprecated
    category_list = 1:size(mean_across_frame,1);
    for i = category_list
        bar(i, mean_across_frame(i), 'facecolor', graph_colors(i,:));
    end

    er = errorbar(category_list, mean_across_frame,errorci(:,:,1),errorci(:,:,2),'LineWidth', 1);
    er.Color = [0 0 0];
    er.LineStyle = 'none';
    hold off

    set(gca,'XTickLabel', {});  % display_names
%    legend(display_names, 'Location', 'SouthEastOutside', 'Box','off');
    ylabel(metric_type);
%    title('Comparison of models');

    ylim(GlobalConfig.find_graph_ylim(mean_across_frame, errorci, 1));

    axis square;
    box on;
    set(gca, 'fontsize', 10, 'fontname', 'Arial');
    set(gca,'XLim',[0, size(mean_across_frame,1)+1])

%    saveas(gcf, [saved_folder, '/Barplot_', metric_type, '.png']);
    saveas(gcf, [saved_folder, '/barplot_', metric_type, '.fig']);
    print([saved_folder, '/barplot_', metric_type, '.svg'],'-dsvg',['-r',num2str(fig_res)]);
    close;
end