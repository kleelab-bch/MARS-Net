function model_name = dataset_to_model(a_dataset)
    model_name = '';
    if strcmp(a_dataset, '040119_PtK1_S01_01_phase')
        model_name = 'BCDE';
    elseif strcmp(a_dataset, '040119_PtK1_S01_01_phase_ROI2')
        model_name = 'ACDE';
    elseif strcmp(a_dataset, '040119_PtK1_S01_01_phase_2_DMSO_nd_01')
        model_name = 'ABDE';
    elseif strcmp(a_dataset, '040119_PtK1_S01_01_phase_2_DMSO_nd_02')
        model_name = 'ABCE';
    elseif strcmp(a_dataset, '040119_PtK1_S01_01_phase_3_DMSO_nd_03')
        model_name = 'ABCD';

    end
end