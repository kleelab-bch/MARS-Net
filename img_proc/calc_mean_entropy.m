function [mean_entropy, entropy_list] = calc_mean_entropy(folder_path, img_names) 
    total_entropy = 0;
    entropy_list = zeros(length(img_names), 1, 'double');
    for n = 1 : length(img_names)
        img_name = img_names{1,n}
        entropy_list(n,1) = entropy(imread(fullfile(folder_path, img_name)));
        total_entropy = total_entropy + entropy_list(n,1);
    end
    mean_entropy = total_entropy / length(img_names)
end