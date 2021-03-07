function violinplot_two_models(model1, model2, ylim_vector, saved_path, metric_type, title_suffix, condition_one, condition_two)
    figure
    set(gcf,'Visible','off');
    test = [model1, model2];
    label = [ones(length(model1), 1); 2*ones(length(model2), 1)];
    vs = violinplot(test, label);
    
    %% hypothesis testing
    [p,h,stats] = hypothesis_testing(model1, model2);
    [kstest_h,kstest_p,kstest_k] = kstest2(model1,model2,'Alpha',0.01)
    
    %% Text decorations
    ylabel(metric_type, 'fontsize', 14);
    ylim(ylim_vector);
    
    xticklabels({condition_one, condition_two});
    set(gca,'XTickLabel', get(gca,'XTickLabel') ,'fontsize',14, 'fontname', 'Arial')

%    title([metric_type, ' ', title_suffix], 'fontsize', 16);
    
    dim = [.15 0.27 0 0];
    str = [condition_one, ' mean: ', num2str( round(mean(model1), 3) ), newline, ...
        condition_two, ' mean: ', num2str( round(mean(model2), 3) ), newline, ...
        'Wilcoxon signed-rank test', newline, ...
        'p: ', num2str(p), ' signedrank: ', num2str( round(stats.signedrank, 3) ), newline, ...
        'ks test: ', num2str(kstest_p), ' ' num2str(kstest_k)]; % '  zval: ', num2str( round(stats.zval, 3) ), '
    annotation('textbox',dim,'String',str,'FitBoxToText','on', 'EdgeColor','none');

    box on;
    
    saveas(gcf, [saved_path, title_suffix, ' ', metric_type, '.png']);
end