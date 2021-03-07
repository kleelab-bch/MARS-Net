% Junbong Jang
% 11/11/2020

addpath('..');
constants_struct = GlobalConfig().update_config('');


dataset_name = 'Paxilin-HaloTMR-TIRF3';
patch_name = [dataset_name, '-150.png'];
img_path = [constants_struct.img_root_path, dataset_name, '/img/'];
mask_path = [constants_struct.img_root_path, dataset_name, constants_struct.mask_type];
fold_name = 'ABCDE';
%fold_name = 'ABCEF';
%fold_name = 'ABDEF';
%fold_name = 'BCDEF';
save_path = [constants_struct.root_path, 'evaluation_f1/results/', constants_struct.concat_display_names(), ...
                        '/repeat1/test/'];

for prediction_num = 1:length(constants_struct.prediction_path_list)
    disp('----------------')
    prediction_path_root = constants_struct.prediction_path_list{prediction_num};
    prediction_path = [prediction_path_root, dataset_name, '/frame4_', fold_name, '_repeat0/'];

    raw_img = imread([img_path, patch_name]);
    [row, col, color_channels] = size(raw_img);
    if color_channels == 3
        img = im2uint8(rgb2gray(raw_img));
    else
        img = im2uint8(raw_img);
    end

    mask = double(imread([mask_path, patch_name]));
    img_predicted = imread([prediction_path, patch_name]);

    %% Extracting the Boundary
    mask_edge = extract_edge(mask, 0);
    predicted_edge = extract_edge(img_predicted);

    %% cropping the edges of images
    img = img(30:row, 30:col);
    mask_edge = mask_edge(30:row, 30:col);
    predicted_edge = predicted_edge(30:row, 30:col);

%    mkdir(save_path)
%    visualize_boundary_overlay_mask_prediction(img, mask_edge, predicted_edge);
%    saveas(gcf, [save_path, 'mask_prediction_overlay_', num2str(prediction_num), '_', patch_name]);

    %% --------- Caclulate the performance -----------------
    mask_edge = mask_edge/255;
    predicted_edge = predicted_edge/255;

    % Estimating the parameter of the threshold
    [row, col] = size(mask_edge)
    % dig_length is necessary because correspondPixels use maxDist*dig_length as threshold
    dig_length = sqrt(row^2 + col^2)
    max_pixel = 13.0;
    maxDist = max_pixel / dig_length

    [match1_U,match2_U] = correspondPixels(predicted_edge, mask_edge, maxDist);

    precision = double(sum(match1_U(:) > 0)) / double(sum(predicted_edge(:)));
    recall = double(sum(match2_U(:) > 0)) / double(sum(mask_edge(:)));
    F_score=2*recall*precision/(precision+recall);
    disp(['precision: ', num2str(precision)])
    disp(['recall: ', num2str(recall)])
    disp(['F score: ', num2str(F_score)])

end


function visualize_boundary_overlay_mask_prediction(img, mask, img_model)
    img_gt_model = uint8(cat(3, img, img, img));
    img_gt_model(:, :, 1) = img_gt_model(:, :, 1) + uint8(mask);
    img_gt_model(:, :, 2) = img_gt_model(:, :, 2) + uint8(img_model);
%    img_gt_model = imresize(img_gt_model, 5);  % Increase the resolution of the images

    figure(1);
    imshow(img_gt_model);
end