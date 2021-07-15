function [mean_value, errci] = mean_errci(list_of_values)
    list_of_values(isnan(list_of_values)) = 0;
    if size(list_of_values, 1) > 1
        mean_value = mean(bootstrp(1000,@mean,list_of_values));
        ci = bootci(1000, @mean, list_of_values);

        errci(1) = mean_value - ci(1);  % low
        errci(2) = ci(2) - mean_value;  % high
    else
        disp('Only one element, so skip bootstrapping')
        mean_value = list_of_values;
        ci = list_of_values;
        errci(1) = mean_value;
        errci(2) = mean_value;
    end
end