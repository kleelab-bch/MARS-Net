function back = extract_edge(prediction_sim, ground_truth)
    if nargin < 2
        ground_truth = 1;
    end
    if ground_truth
        prediction_sim = imgaussfilt(prediction_sim,3);
    end
        
    
    prediction_sim = im2bw(prediction_sim, 0.5);
    prediction_sim = imfill(prediction_sim, 'holes');
    prediction_sim = find_max(prediction_sim);  %  find biggest cell object

    [B,L] = bwboundaries(prediction_sim,'noholes');
    boundary = B{1}; 

    size_b = size(prediction_sim);
    back = zeros(size_b(1),size_b(2));
    for j = 1:size(boundary,1)
        back(boundary(j,1), boundary(j,2)) = 255;
    end
    back = find_max(back);
end