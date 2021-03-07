% 1/4/2021
% Draw edge evolution over an image.


function draw_edge_evolution(mask_path, img_path, frame_num, save_path)
    mkdir(save_path);

    img_file_names = dir([img_path, '*.png']);
    mask_file_names = dir([mask_path, '*.png']);

    img = imread([img_path, img_file_names(1, 1).name]);
    img = img(30:end, 30:end);
    [row, col] = size(img);

    img = normalizeIntensity(img, row, col);
    imgRGB(:,:,1) = img;
    imgRGB(:,:,2) = img;
    imgRGB(:,:,3) = img;

    color_grad = jet(length(mask_file_names));
    for_loop_counter = 1;
    frame_list = [];
    for iFrame = 1: length(mask_file_names)
        frame_list = [frame_list, iFrame];
        % Extract the edge from mask
        patch_name = mask_file_names(iFrame, 1).name;
        mask = imread([mask_path, patch_name]);
        mask = mask(30:end, 30:end);
        mask_edge = extract_edge(mask, 0);
        mask_edge = imresize(mask_edge, size(imgRGB,1,2));
        mask_edge = normalizeIntensity(mask_edge, row, col);

        % Visualization of Edge Evolution
        red_mask = mask_edge*color_grad(for_loop_counter,1);
        green_mask = mask_edge*color_grad(for_loop_counter,2);
        blue_mask = mask_edge*color_grad(for_loop_counter,3);
        for_loop_counter = for_loop_counter + 1;

%        if mod(iFrame,170) == 1
%        end
        imgRGB = darken_along_edge(imgRGB, mask_edge);
        imgRGB(:, :, 1) = imgRGB(:, :, 1) + red_mask;
        imgRGB(:, :, 2) = imgRGB(:, :, 2) + green_mask;
        imgRGB(:, :, 3) = imgRGB(:, :, 3) + blue_mask;
        imshow(imgRGB)
    end

    set(gca,'position',[0 0 1 1]);
    width = 100;
    height = 4;
    col_pos = size(imgRGB,1)-36; % bottom
    row_pos = size(imgRGB,2)-(width+10); % right
%    row_pos = 20; % left
%    row_pos = size(imgRGB,2)/2-width/2; % middle
    rectangle('Position',[row_pos,col_pos,width,height],'FaceColor', 'white', 'LineWidth',1,'LineStyle','none') % bottom middle
%    text(10+width/2, size(imgRGB,1)-30,'\color{white} 32.5 \mum','FontSize',12, 'HorizontalAlignment','center')

    colormap(color_grad);
%    colorbar('YTick', linspace(0, 1, length(frame_list(1:20:end))), 'YTickLabel', frame_list(1:20:end));
     colorbar('YTick', [], 'YTickLabel', []); % 'YTick', linspace(0, 1, length(frame_list)), 'YTickLabel', frame_list
    saveas(gcf, [save_path, 'edge_evolution_', num2str(frame_num), '.png']);
end


function imgRGB = darken_along_edge(imgRGB, mask)
    % make the edge dark on the image so that edge's color do not merge with previous edges
    [rows, cols] = size(mask);
    for row = 1:rows
        for col = 1:cols
            if mask(row, col) == 1
                imgRGB(row, col, 1) = 0;
                imgRGB(row, col, 2) = 0;
                imgRGB(row, col, 3) = 0;
            end
        end
    end

end


function normalized_img = normalizeIntensity(img, row, col)
    maxIm = max(img(:));
    minIm = min(img(:));
    normalized_img = (double(img) - double(repmat(minIm, row, col))) ./ double(maxIm - minIm);
end
