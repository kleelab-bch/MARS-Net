%% Created by Junbong Jang
% 1/2/2021

function overlap_compare(prediction_path_list, mask_path, img_path, frame_num, max_dist_pixel, save_path)
    mkdir(save_path);

    image_format = '.png';
    image_path_list = dir([prediction_path_list{1}, ['*', image_format]]);
    if length(image_path_list) > 0
        orig_image_name = image_path_list(1, 1).name;
        mask_path_list = dir([mask_path, ['*', image_format]]);

        model_precision = [];
        model_recall = [];
        model_F_score = [];
        for frame_index = 1: length(mask_path_list)
            frame_index
            %% Load image, mask, predicted image
            mask_name = mask_path_list(frame_index, 1).name;
            image_name = convert_image_name(orig_image_name, mask_name, image_format);

            raw_img = imread([img_path, image_name]);
            [row, col, color_channels] = size(raw_img);
            if color_channels == 3
                img = im2uint8(rgb2gray(raw_img));
            else
                img = im2uint8(raw_img);
            end

            mask = double(imread([mask_path, mask_name]));
            img_predicted_1 = imread([prediction_path_list{1}, image_name]);
            img_predicted_2 = imread([prediction_path_list{2}, image_name]);

            %% cropping the edges of images
            img = img(30:row, 30:col);
            mask = mask(30:row, 30:col);
            img_predicted_1 = img_predicted_1(30:row, 30:col);
            img_predicted_2 = img_predicted_2(30:row, 30:col);

            %% Extracting the Boundary
            mask_edge = extract_edge(mask, 0);
            predicted_edge_1 = extract_edge(img_predicted_1);
            predicted_edge_2 = extract_edge(img_predicted_2);


            %% Visualization of prediction
            visualize_boundary_overlay_mask_prediction(img, mask_edge, predicted_edge_1, predicted_edge_2);
            saveas(gcf, [save_path, 'mask_prediction_overlay_', image_name]);

        end
    end
end


function visualize_boundary_overlay_mask_prediction(img, mask, img_model1, img_model2)
    se = strel('diamond',1);
    mask = imdilate(mask,se);
    img_model1 = imdilate(img_model1,se);
    img_model2 = imdilate(img_model2,se);

%    img = imlocalbrighten(img);

    img_gt_model = uint8(cat(3, img, img, img));
    img_gt_model(:, :, 1) = img_gt_model(:, :, 1) + uint8(mask);
    img_gt_model(:, :, 2) = img_gt_model(:, :, 2) + uint8(img_model1);
    img_gt_model(:, :, 3) = img_gt_model(:, :, 3) + uint8(img_model2);
%    img_gt_model = imresize(img_gt_model, 5);  % Increase the resolution of the images

    figure(1);
    imshow(img_gt_model);
    set(gca,'position',[0 0 1 1]);


    width = 100;
    height = 4;
    col_pos = size(img_gt_model,1)-20; % bottom
%    row_pos = size(img_gt_model,2)-(width+10); % right
%    row_pos = 10; % left
    row_pos = size(img_gt_model,2)/2-width/2; % middle
    rectangle('Position',[row_pos,col_pos,width,height],'FaceColor', 'white', 'LineWidth',1,'LineStyle','none')

%    text(10+width/2, size(img_gt_model,1)-30,'\color{white} 32.5 \mum','FontSize',12, 'HorizontalAlignment','center')
end


function new_image_name = convert_image_name(image_name, mask_name, image_format)
    % assume image code is the same between original image and mask image
    seq_code = extractBetween(mask_name,length(mask_name)-7,length(mask_name)-4);
    new_image_name = strjoin([extractBetween(image_name,1,length(image_name)-8), seq_code, image_format], '');
end


