function violinplot_two_models(model1, model2, ylim_vector, saved_path, metric_type, title_suffix, condition_one, condition_two)
    figure
    set(gcf,'Visible','off');
    label = [ones(length(model1), 1); 2*ones(length(model2), 1)];
    vs = violinplot([model1, model2], label);
    
    %% hypothesis testing
    [p,h,stats] = hypothesis_testing(model1, model2);
    [kstest_h,kstest_p,kstest_k] = kstest2(model1,model2,'Alpha',0.01);
    [ttest_h,ttest_p,ttest_ci,ttest_stats] = ttest(model1, model2);
    [lillie_h,lillie_p] = lillietest(model1-model2);
    [shapiro_h, shaprio_p, shapiro_W] = swtest(model1-model2);

    %% Text decorations
    ylabel(metric_type, 'fontsize', 14);
    ylim(ylim_vector);
    
    xticklabels({condition_one, condition_two});
    set(gca,'XTickLabel', get(gca,'XTickLabel') ,'fontsize',14, 'fontname', 'Arial')

%    title([metric_type, ' ', title_suffix], 'fontsize', 16);

    dim = [.15 0.27 0 0];
    str = [condition_one, ' mean: ', num2str( round(mean(model1), 3) ), newline, ...
        condition_two, ' mean: ', num2str( round(mean(model2), 3) ), newline, ...
        'Wilcoxon test: ', num2str(p), ' signedrank: ', num2str( round(stats.signedrank, 3) ), newline, ...
        'ks test: ', num2str(kstest_p), ' ', num2str(kstest_k), newline, ...
        't-test: ', num2str(ttest_p), '  normality test: ', num2str(lillie_p), ' ', num2str(shaprio_p)];
    annotation('textbox',dim,'String',str,'FitBoxToText','on', 'EdgeColor','none');

    box on;
    
    saveas(gcf, [saved_path, title_suffix, ' ', metric_type, '.png']);
end