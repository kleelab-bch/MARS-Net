function violinplot_diff(f1_dff, prec_diff, recall_diff, ylim_vector, saved_path, metric_type, title_suffix)
    figure
    set(gcf,'Visible','off');
    test = [f1_dff, prec_diff, recall_diff];
    label = [ones(length(f1_dff), 1); 2*ones(length(prec_diff), 1); 3*ones(length(recall_diff), 1)];
    vs = violinplot(test, label);
    plot(get(gca, 'XLim'), [0 0], '--');  % dashed reference line at 0
    
    %% hypothesis testing
%    [p1,h1,stats1] = hypothesis_testing(zeros(length(f1_dff), 1), f1_dff);
%    [p2,h2,stats2] = hypothesis_testing(zeros(length(prec_diff), 1), prec_diff);
%    [p3,h3,stats3] = hypothesis_testing(zeros(length(recall_diff), 1), recall_diff);
%    [p4,h4,stats4] = hypothesis_testing(zeros(length(dice_diff), 1), dice_diff);
%    hypothesis_testing_result = [newline, 'Wilcoxon signed-rank test', newline, ...
%                                'p: ', num2str(round(p1,3)), '  zval: ', num2str( round(stats1.zval, 3) ), newline, ...
%                                'p: ', num2str(round(p2,3)), '  zval: ', num2str( round(stats2.zval, 3) ), newline, ...
%                                'p: ', num2str(round(p3,3)), '  zval: ', num2str( round(stats3.zval, 3) ), newline, ...
%                                'p: ', num2str(round(p4,3)), '  zval: ', num2str( round(stats4.zval, 3) )];
    hypothesis_testing_result='';


    %% Text decorations
    ylabel(metric_type, 'fontsize', 14);
%    ylim(ylim_vector);
    
    xticklabels({'F1', 'Precision', 'Recall'});
    set(gca,'XTickLabel', get(gca,'XTickLabel') ,'fontsize',14, 'fontname', 'Arial')

%    title([metric_type, ' ', title_suffix], 'fontsize', 16);

    dim = [.65 0.92 0 0];
    str = ['F1 mean: ', num2str( round(mean(f1_dff), 3) ), newline, ...
            'Precision mean: ', num2str( round(mean(prec_diff), 3) ), newline, ...
            'Recall mean: ', num2str( round(mean(recall_diff), 3) ), newline, ...
             hypothesis_testing_result];
    annotation('textbox',dim,'String',str,'FitBoxToText','on', 'EdgeColor','none');

    box on;
    
    saveas(gcf, [saved_path, title_suffix, ' ', metric_type, '.png']);
end