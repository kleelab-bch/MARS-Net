% Author Junbong Jang
% Last Modified Date 2/12/2020

% \\research.wpi.edu\leelab\Chauncey\vUnet_Segmentation\vUnet_PhaseContrast_final\svUnet_pairwise_jj\average_hist\predict_wholeframe\040119_PtK1_S01_01_phase_2_DMSO_nd_03\6_0_040119_PtK1_S01_01_phase_2_DMSO_nd_03
dataset = {'040119_PtK1_S01_01_phase_2_DMSO_nd_03'};
dice_coefficient_cell = cell(41,2);
counter = 1;
for n = 1 : length(dataset)
    folder = dataset(n)
	mask_path = strcat('../DataSet_expand/', folder, '/mask/');
	mask_path = mask_path{1}
	predict_path = strcat('../DataSet_expand/', folder, '_pred/mask/');
    generate_path = strcat('../DataSet_expand/', folder, '_compare/');
	predict_path = predict_path{1}
    generate_path = generate_path{1}
    filesStructure = dir(predict_path);
	allFileNames = {filesStructure(:).name};
    
	for k = 1 : length(allFileNames)
		fileName = allFileNames{k}
		imageFileBool = contains(fileName,'.png');
		if imageFileBool
            % get original iamge
			filePath = strcat(mask_path, strrep(fileName,'predict',''))
			I = imread(filePath);
			I_binary = imbinarize(I);
            [row, col] = size(I_binary)
            I_binary = I_binary(30:row-31, 30:col-31)
            % get predicted image
			predFilePath = strcat(predict_path, fileName)
			I2 = imread(predFilePath);
			I2_binary = imbinarize(I2);
			%I2_binary = imresize(I2_binary,size(I_binary));
            
            
            %% calculate similarity
            %
            I2_final = I2_binary;
			similarity = dice(I_binary, I2_final)
            figure(1), clf
            hAx = axes;
            C = imfuse(I_binary, I2_final); % where gcf is created
            imshow(C, 'Parent', hAx, 'Border','Loose');
			title(hAx, ['Dice Index = ' num2str(similarity)], 'FontSize', 14, 'Color','b');
            imlegend([1 0 1; 0 1 0; 1 1 1], {'Predict'; 'True'; 'Overlap'});
			saveas(gcf, strcat(generate_path, '/compare-',fileName));
            
            dice_coefficient_cell{counter,1} = fileName;
            dice_coefficient_cell{counter,2} = num2str(similarity);
            counter = counter + 1;
            
		end
    end
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