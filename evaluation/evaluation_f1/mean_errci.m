function [mean_value, errci] = mean_errci(list_of_values)
    list_of_values(isnan(list_of_values)) = 0;
    mean_value = mean(bootstrp(1000,@mean,list_of_values));
    ci = bootci(1000, @mean, list_of_values);

    errci(1) = mean_value - ci(1);  % low
    errci(2) = ci(2) - mean_value;  % high
end