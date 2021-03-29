% Visualize the edges of manual labels overlaid on top of the image

function overlap_mask_only(mask_path, img_path, save_path)
    mkdir(save_path);

    image_format = '.png';
    image_path_list = dir([img_path, '*.png']);
    orig_image_name = image_path_list(1, 1).name;
    mask_path_list = dir([mask_path, '*.png']);
    for frame_index = 1: length(mask_path_list)
        frame_index
        %% Load image, mask image
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


function new_image_name = convert_image_name(image_name, mask_name, image_format)
    % assume image code is the same between original image and mask image

    if length(image_name) < 8 && length(mask_name) < 8 % since image names can be 005, 010, 015, ... 200.png
        new_image_name = mask_name;
    else
        seq_code = extractBetween(mask_name,length(mask_name)-7,length(mask_name)-4);
        new_image_name = strjoin([extractBetween(image_name,1,length(image_name)-8), seq_code, image_format], '');
    end
end
