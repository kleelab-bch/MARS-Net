function visualize_boundary_overlap(img, mask, img_model1, img_model2)
    img_gt_model1 = uint8(cat(3, img, img, img));
    img_gt_model2 = uint8(cat(3, img, img, img));
    img_model1_model2 = uint8(cat(3, img, img, img));
%    se = strel('square',3);
%         BW2_mask = imdilate(mask,se);
%         BW2_img_model1 = imdilate(img_model1, se);
%         BW2_img_model2 = imdilate(img_model2, se);

    img_gt_model1(:, :, 1) = img_gt_model1(:, :, 1) + uint8(mask);
    img_gt_model1(:, :, 2) = img_gt_model1(:, :, 2) + uint8(img_model1);

    img_gt_model2(:, :, 1) = img_gt_model2(:, :, 1) + uint8(mask);
    img_gt_model2(:, :, 3) = img_gt_model2(:, :, 3) + uint8(img_model2);

    img_model1_model2(:, :, 2) = img_model1_model2(:, :, 2) + uint8(img_model1);
    img_model1_model2(:, :, 3) = img_model1_model2(:, :, 3) + uint8(img_model2);

%    img3(:, :, 2) = img3(:, :, 2) + uint8(BW2_img_model1)*255;
%    img3(:, :, 3)  = img3(:, :, 3) + uint8(BW2_img_model2);
%    % color mask edge as black
%    [row, col, channels] = size(img3);
%    for i_row = 1:row
%     for i_col = 1:col
%         if BW2_mask(i_row, i_col) > 1
%             img3(i_row, i_col, :) = 0;
%         end
%     end
%    end

%    Increase the resolution of the image
%    img_gt_model1 = imresize(img_gt_model1, 5);
%    img_gt_model2 = imresize(img_gt_model2, 5);
%    img_model1_model2 = imresize(img_model1_model2, 5);

    figure(1);
    subplot(1,3,1);
    subaxis(1,3,1, 'Spacing', 0.01, 'Padding', 0, 'Margin', 0);
    imshow(img_gt_model1);

    subplot(1,3,2);
    subaxis(1,3,2, 'Spacing', 0.01, 'Padding', 0, 'Margin', 0);
    imshow(img_gt_model2);

    subplot(1,3,3);
    subaxis(1,3,3, 'Spacing', 0.01, 'Padding', 0, 'Margin', 0);
    imshow(img_model1_model2);
%         imlegend([1 0 0; 0 1 0; 0 0 1; 1 1 1], {'Round1'; 'Round2'; 'Round3'; 'Overlap'});
end