%% Created by Chauncey
% modified by Junbong Jang
% 6/2/2020

function overlap_mask_prediction(prediction_path, mask_path, img_path, frame_num, max_dist_pixel, visualize_bool, save_path)
    mkdir(save_path);

    image_format = '.png';
    image_path_list = dir([prediction_path, ['*', image_format]]);
    if length(image_path_list) > 0
        orig_image_name = image_path_list(1, 1).name;
        mask_path_list = dir([mask_path, ['*', image_format]]);

        model_precision = [];
        model_recall = [];
        model_F_score = [];
        model_dice = [];
        image_list = [];
        for frame_index = 1: length(mask_path_list)
            %% Load image, mask, predicted image
            mask_name = mask_path_list(frame_index, 1).name;
            image_name = convert_image_name(orig_image_name, mask_name, image_format);
            image_list = [image_list; image_name];

            raw_img = imread([img_path, image_name]);
            [row, col, color_channels] = size(raw_img);
            if color_channels == 3
                img = im2uint8(rgb2gray(raw_img));
            else
                img = im2uint8(raw_img);
            end

            mask = double(imread([mask_path, mask_name]));
            img_predicted = imread([prediction_path, image_name]);

            %% cropping images
            img = img(30:row, 30:col);
            mask = mask(30:row, 30:col);
            img_predicted = img_predicted(30:row, 30:col);

            %% Extracting the Boundary for F1, precision and recall
            mask_edge = extract_edge(mask, 0);
            predicted_edge = extract_edge(img_predicted);

            % processing the images for dice
            mask = process_image(mask);
            img_predicted = process_image(img_predicted);

            %% Visualization of prediction
            if visualize_bool
                visualize_boundary_overlay_mask_prediction(img, mask_edge, predicted_edge);
                saveas(gcf, [save_path, 'mask_prediction_overlay_', image_name]);
            end

            %% --------- Caclulate the performance -----------------
            mask_edge = mask_edge/255;
            predicted_edge = predicted_edge/255;

            % Estimating the parameter of the threshold
            if frame_index == 1
                [row, col] = size(mask_edge);
                % dig_length is necessary because correspondPixels use maxDist*dig_length as threshold
                dig_length = sqrt(row^2 + col^2);
                maxDist = max_dist_pixel / dig_length;
            end

            [match1_U,match2_U] = correspondPixels(predicted_edge, mask_edge, maxDist);
            
            %Precision
            precision = double(sum(match1_U(:) > 0)) / double(sum(predicted_edge(:)));
            model_precision = [model_precision, precision];
            %Recall
            recall = double(sum(match2_U(:) > 0)) / double(sum(mask_edge(:)));
            model_recall = [model_recall, recall];
            %F_score
            if precision+recall == 0
                F_score = 0;
            else
                F_score = 2*recall*precision/(precision+recall); %%F_score=2*1/((1/Precision)+(1/Recall));
            end
            model_F_score = [model_F_score, F_score];
            %Dice
            model_dice = [model_dice, dice(mask, img_predicted)];

        end
        save([save_path, 'Recall_Precision_F_score_frame', num2str(frame_num), '.mat'], 'model_recall', 'model_precision', 'model_F_score', 'model_dice', 'image_list');
    end
end

function I_binary = process_image(I)
    I_binary = im2bw(I, 0.5);
end


function visualize_boundary_overlay_mask_prediction(img, mask, img_model)
    img_gt_model = uint8(cat(3, img, img, img));
    img_gt_model(:, :, 1) = img_gt_model(:, :, 1) + uint8(mask);
    img_gt_model(:, :, 2) = img_gt_model(:, :, 2) + uint8(img_model);
%    img_gt_model = imresize(img_gt_model, 5);  % Increase the resolution of the images

    figure(1);
    imshow(img_gt_model);
end


function imlegend(colorArr, labelsArr)
    hold on;
    for ii = 1:length(labelsArr)
      % Make a new legend entry for each label. 'color' contains a 0->255 RGB triplet
      scatter([],[],1, colorArr(ii,:), 'filled', 'DisplayName', labelsArr{ii});
    end
    hold off;
    lgnd = legend();
    set(lgnd,'color',[127 127 127]/255);
end


function new_image_name = convert_image_name(image_name, mask_name, image_format)
    % assume image code is the same between original image and mask image

    if length(image_name) < 8 && length(mask_name) < 8 % since image names can be 005, 010, 015, ... 200.png
        new_image_name = image_name;
    else
        seq_code = extractBetween(mask_name,length(mask_name)-7,length(mask_name)-4);
        new_image_name = strjoin([extractBetween(image_name,1,length(image_name)-8), seq_code, image_format], '');
    end
end

        
        
