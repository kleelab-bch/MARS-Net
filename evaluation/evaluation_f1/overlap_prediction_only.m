%% Created by Chauncey
% modified by Junbong Jang
% 6/2/2020

function overlap_prediction_only(prediction_path, img_path, save_path)
    mkdir(save_path);

    image_format = '.png';
    image_path_list = dir([prediction_path, ['*', image_format]]);
    if length(image_path_list) > 0
        for frame_index = 1: length(image_path_list)
            %% Load image, predicted image
            image_name = image_path_list(frame_index, 1).name

            raw_img = imread([img_path, image_name]);
            [row, col, color_channels] = size(raw_img);
            if color_channels == 3
                img = im2uint8(rgb2gray(raw_img));
            else
                img = im2uint8(raw_img);
            end
            img_predicted = imread([prediction_path, image_name]);

%            size(img)
%            size(img_predicted)
            %% cropping images
%            img = img(30:row, 30:col);
%            img_predicted = img_predicted(30:row, 30:col);

            %% Extracting the Boundary
            predicted_edge = extract_edge(img_predicted);

            %% Visualization of prediction
            visualize_boundary_overlay_prediction(img, predicted_edge);
            saveas(gcf, [save_path, 'prediction_only_overlay_', image_name]);
        end
    end
end

function visualize_boundary_overlay_prediction(img, img_model)
    se = strel('diamond',5);
    img_model = imdilate(img_model,se);

    img_gt_model = uint8(cat(3, img, img, img));
    img_gt_model(:, :, 1) = img_gt_model(:, :, 1) + uint8(img_model);
%    img_gt_model = imresize(img_gt_model, 5);  % Increase the resolution of the images

    figure(1);
    imshow(img_gt_model);
    axis off;
    set(gca,'LooseInset',get(gca,'TightInset'));
    set(gca,'position',[0 0 1 1]);

    width = 100;
    height = 4;
    col_pos = size(img_gt_model,1)-20; % bottom
    row_pos = 20; % left
    rectangle('Position',[row_pos,col_pos,width,height],'FaceColor', 'white', 'LineWidth',1,'LineStyle','none')

end
