% Visualize the edges of manual labels overlaid on top of the image

function overlap_mask_only(mask_path, img_path, save_path)
    mkdir(save_path);

    image_path_list = dir([img_path, '*.png']);
    mask_path_list = dir([mask_path, '*.png']);
    for frame_index = 1: length(image_path_list)
        frame_index
        %% Load image, mask image
        img_name = image_path_list(frame_index, 1).name;
        mask_name = mask_path_list(frame_index, 1).name;

        raw_img = imread([img_path, img_name]);
        [row, col, color_channels] = size(raw_img);
        if color_channels == 3
            img = im2uint8(rgb2gray(raw_img));
        else
            img = im2uint8(raw_img);
        end

        mask = double(imread([mask_path, mask_name]));
        mask_edge = extract_edge(mask, 0);

        visualize_boundary_overlay_mask(img, mask_edge);
        saveas(gcf, [save_path, 'mask_overlay_', mask_name]);
    end
end


function visualize_boundary_overlay_mask(img, mask)
    se = strel('diamond',2);
    mask = imdilate(mask,se);

    img_mask = uint8(cat(3, img, img, img));
    img_mask(:, :, 1) = img_mask(:, :, 1) + uint8(mask);

    figure(1);
    imshow(img_mask);
end