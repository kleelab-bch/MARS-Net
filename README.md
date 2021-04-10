# MARS-Net
Deep learning-based segmentation pipeline for profiling cellular morphodynamics from multiple types of live cell microscopy  
To learn more about MARS-Net, read the [paper](https://www.biorxiv.org/content/10.1101/191858v3)

## Run Demo
You can quickly one of our segment live cell movie using the demo in Google Colab:
* hi  
<!-- end of the list -->
To test our pipeline from the scratch, the user needs to crop images, and train the models before running inference on trained models which can take several hours.  
So this allows users to segment using trained MARS-Net and U-Net on our live cell movies to see that the MARS-Net is better than U-Net.

## Pipeline
The pipeline consists of label tool, segmentation modeling, and morphodynamics profiling.    
There is no installation procedure except for downloading or installing the software requirements below

#### Software Requirements
This pipeline has been tested on Ubuntu 16.04
* Please download Matlab 2019b
    * Other versions might work but we didn't test our pipeline on other versions.
* [Correspondence Algorithm](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark) developed by University of California Berkeley Segmentation Benchmark for F1, precision and recall evaluation.
* [Windowing and Protrusion package](https://github.com/DanuserLab/Windowing-Protrusion) developed by Gaudenz Danuser lab (UT Southwestern) for Morphoydynamics profiling.  
* For training and segmenting the cell boundary, Python v3.6.8, TensorFlow (v1.15 or v2.3), and Keras v2.3.1 .  
    * Tensorflow v2.3 on RTX Titan GPU with CUDA 10.1
    * Tensorflow v1.15 on GTX 1080Ti GPU with CUDA 10.0 
    
### Label Tool
Tool to facilitate labelling raw images semi-automatically
In the folder label_tool folder,
* python explore_edge_extraction_user_params.py
Do parameter searching for canny_multiplider and denoise kernel size
1. Specify parameters in 
    * user_params.py  
1. Edge extraction Step
    * python extract_edge.py  
1. Manual Fix Step
    * Among the generated edges, connect fragmented edges and delete the wrong edges
1. Post processing step to fill the extracted edges
    * python segment_edge.py  


### Segmentation Model Training and Prediction
In models folder,  
* Set User parameters for datasets, round_nums, and etc.
    * UserParams.py
*Cropping in crop folder
    * crop_augment_split.py
* Training & Prediction in models folder
    * train.py
    * prediction.py

### Evaluation
In evaluation folder,  
Before running code, please install [Correspondence Algorithm](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark)  
and edit parameters GlobalConfig.m
* F1_constants.m
* dice_constants.m
* Evaluation_dice 
* Calculate_dice_coeff_self_training.py
* For Self-Training result
* Calculate_dice_coeff_compare.py
* For Two Model comparison result
* Evaluation_f1
* For Edge evolution and precision, recall, and f1 calculation
* Run in Linux, doesn’t work in Windows 10 because MEX file was compiled for Linux https://www.mathworks.com/help/matlab/matlab_external/platform-compatibility.html
* matlab -nodisplay -nosplash -nodesktop -r "run('draw_overlap_boundary.m');exit;"
* Violin Plot
* violin_compare.m  
For reproduction


### Morphodynamics
* Download protrusion and windowing package at 
* Add directory and subfolders of the downloaded package to the path 
* Type movieSelectorGUI in the command window in MATLAB
* Create new movie and perform windowing.