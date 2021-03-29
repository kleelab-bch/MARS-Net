function draw_histogram(model1, model2, saved_path, metric_type, title_suffix, condition_one, condition_two)
    figure
    set(gcf,'Visible','off');

    nbins = ceil(sqrt(size(model1,2)));
    if nbins < 10
        nbins = 10;
    end
    nbins = int8(nbins);
    model_difference = model1-model2;

    [lillie_h,lillie_p] = lillietest(model_difference);


    hold on
    hist(model_difference);

    prob_dist = fitdist(model_difference','Kernel');
    x = min(model_difference):.005:max(model_difference);
    plot(x, pdf(prob_dist,x),'k--','LineWidth',2);

    xlabel(metric_type, 'fontsize', 14);
    ylabel('Frequency', 'fontsize', 14);

    dim = [.15 0.9 0 0];
    str = ['N=', num2str(size(model1,2)), newline, ...
           'Normality p=', num2str(lillie_p)];
    annotation('textbox',dim,'String',str,'FitBoxToText','on', 'EdgeColor','none');

    box on;
    saveas(gcf, [saved_path, title_suffix, ' ', metric_type, '.png']);
end