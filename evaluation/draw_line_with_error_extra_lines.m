function draw_line_with_error_extra_lines(all_means, all_ci, extra_means, extra_ci, title_prefix, metric_name, saved_folder, frame_array, graph_colors, graph_ylim, display_names)
    % Created by Junbong Jang
    % 11/10/2020

    %Draw the box plot based on the mean standard confidence interval.
    %all_means: saved the mean values in different frames on model comparison
    %all_ci   : saved the confidence interval in different frames on model comparison
    %saved_folder: the path to saved the results.
    %saved_name  : the file name.

    %% Plot data
    total_plots = [];
    [models, folds, frame] = size(extra_means);

    figure;
    output_fig_size = [1800, 900]; % width and height in inches
    fig_res = 300;
    set(gcf,'paperunits','inches','paperposition',[0 0 output_fig_size/fig_res]);
%    figure('Renderer', 'painters', 'Position', [0 0 700 300])

    hold on
    for i = 1 : models
        main_line_width = 2;
        %% Plot extra lines if there are only two models to compare
        if models <= 3
            for j = 1 : folds
                p = plot(frame_array, reshape(extra_means(i, j, :), size(all_means(i, :))), ...
                'Color', [graph_colors(i, :), 0.3], 'LineWidth', 1);
            end
        end
        main_plot = errorbar(frame_array,all_means(i, :),squeeze(all_ci(i, :, 1)),squeeze(all_ci(i, :, 2)), '-', ...
         'Color', [graph_colors(i, :), 0.5], 'LineWidth', main_line_width);  % 'Marker', '+', 'MarkerSize', 8,
        total_plots = [total_plots, main_plot];
    end


    %% decorations
    display_names = replace(display_names, '_', '\_');
    legend(total_plots, display_names, 'Location', 'SouthEastOutside','Box','off');
    xlabel('Number of training frames');
    ylabel(metric_name);
    ylim(GlobalConfig.find_graph_ylim(extra_means, extra_ci, 1));

%    title([ 'Compare ', metric_name, ' ', title_prefix],'fontweight','bold','fontsize',24);
    xticks(frame_array);

    cur_ax = gca;
    cur_ax.XAxis.LineWidth = 1;
    cur_ax.YAxis.LineWidth = 1;
    set(gca,'XLim',[0 frame_array(end)+1])
%    set(gca,'XTickLabel',get(gca,'XTickLabel'));
%    set(gca,'YTickLabel',get(gca,'YTickLabel'));
    set(gca, 'fontsize', 10, 'fontname', 'Arial');
    xtickangle(45);

    axis square;
    box on

    saveas(gcf, fullfile(saved_folder, ['Compare ', metric_name, ' ', title_prefix, ' line.fig']));
%    saveas(gcf, fullfile(saved_folder, ['Compare ', metric_name, ' ', title_prefix, ' line.png']));

    print(fullfile(saved_folder, ['Compare ', metric_name, ' ', title_prefix, ' line.svg']),'-dsvg',['-r',num2str(fig_res)]);

    close;
end