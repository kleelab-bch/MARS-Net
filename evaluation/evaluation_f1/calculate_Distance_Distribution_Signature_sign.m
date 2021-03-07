function [DDS_Db_based_B_gt, DDS_Db_based_B_bs] = calculate_Distance_Distribution_Signature_sign(ground_truth, bin_segmentation, grouth_truth_region, c_threshold)

if nargin < 4
    c_threshold = 70;
end
[row, col] = size(ground_truth);

%remove the boundary of binary image.
ground_truth = ground_truth(2:row-1, 2:col-1);
bin_segmentation = bin_segmentation(2:row-1, 2:col-1);
grouth_truth_region = grouth_truth_region(2:row-1, 2:col-1);

%get the index of image.
[row_gt, col_gt] = find(ground_truth == 255);
[row_bs, col_bs] = find(bin_segmentation == 255);

R = length(row_gt);
S = length(row_bs);
%Calculate Db(B_gt, B_bs)
DDS_Db_based_B_gt = [];
for i = 1 : R
    current_p = [row_gt(i), col_gt(i)];
    [min_dist, index] = calculate_mindist_withlocation(current_p, [row_bs, col_bs], c_threshold);
    if grouth_truth_region(row_bs(index), col_bs(index)) == 0
        sign = -1;
    else
        sign = 1;
    end
    DDS_Db_based_B_gt = [DDS_Db_based_B_gt, sign * min_dist];
end

%Calculate Db(B_bs, B_gt)
DDS_Db_based_B_bs = [];
for i = 1 : S
    current_p = [row_bs(i), col_bs(i)];
    min_dist = calculate_mindist(current_p, [row_gt, col_gt], c_threshold);
    DDS_Db_based_B_bs = [DDS_Db_based_B_bs, min_dist];
end
end