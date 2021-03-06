function violinplot_two_models(model1, model2, saved_path, metric_type, title_suffix, condition_one, condition_two, graph_colors)
    figure
    set(gcf,'Visible','off');

    output_fig_size = [1150, 850];
    fig_res = 300;
    set(gcf,'paperunits','inches','paperposition',[0 0 output_fig_size/fig_res]);
    
    label = [ones(length(model1), 1); 2*ones(length(model2), 1)];
    vs = violinplot([model1, model2], label);

    for a_model_index = 1:2
        vs(1,a_model_index).ViolinColor = graph_colors(a_model_index,:);
    end
    
    %% hypothesis testing
    [p,h,stats] = hypothesis_testing(model1, model2);
    [kstest_h,kstest_p,kstest_k] = kstest2(model1,model2,'Alpha',0.01);
    [ttest_h,ttest_p,ttest_ci,ttest_stats] = ttest(model1, model2);
    [lillie_h,lillie_p] = lillietest(model1-model2);
    [shapiro_h, shaprio_p, shapiro_W] = swtest(model1-model2);

    %% Text decorations
    ylabel(metric_type, 'fontsize', 14);
    disp('violinplot_two_models')
    min_data = min([model1, model2],[],'all');
    max_data = max([model1, model2],[],'all');
    top_offset = (max_data-min_data)/20;
    bottom_offset = (max_data-min_data)/6;  % 20
    ylim([min_data-bottom_offset, max_data+top_offset]);

    xticklabels({condition_one, condition_two});
    set(gca,'XTickLabel', get(gca,'XTickLabel') ,'fontsize',14, 'fontname', 'Arial')

%    title([metric_type, ' ', title_suffix], 'fontsize', 16);

    dim = [0.45 0.4 0 0]; % x, y, width, height
    str = [condition_one, ' mean: ', num2str( round(mean(model1), 3) ), newline, ...
        condition_two, ' mean: ', num2str( round(mean(model2), 3) ), newline, ...
        'Wilcoxon test: ', num2str(p), ' signedrank: ', num2str( round(stats.signedrank, 3) ), newline, ...
        'ks test: ', num2str(kstest_p), ' k:', num2str(kstest_k), newline, ...
        't-test: ', num2str(ttest_p), '  normality test: ', num2str(lillie_p), ' ', num2str(shaprio_p)];
    annotation('textbox',dim,'String',str,'FitBoxToText','on', 'EdgeColor','none', 'fontsize', 7);

    box on;
    set(gca, 'fontsize', 10, 'fontname', 'Arial');

    
    saveas(gcf, [saved_path, title_suffix, ' ', metric_type, '.png']);
    print([saved_path, title_suffix, ' ', metric_type, '.svg'],'-dsvg',['-r',num2str(fig_res)]);
    close;
end