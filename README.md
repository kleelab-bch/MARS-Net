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
    * Doesn’t work on Windows OS because it was compiled for Linux
* [Windowing and Protrusion package](https://github.com/DanuserLab/Windowing-Protrusion) developed by Gaudenz Danuser lab (UT Southwestern) for Morphoydynamics profiling.  
* For training and segmenting the cell boundary, Python v3.6.8, TensorFlow (v1.15 or v2.3), and Keras v2.3.1 .  
    * Tensorflow v2.3 on RTX Titan GPU with CUDA 10.1
    * Tensorflow v1.15 on GTX 1080Ti GPU with CUDA 10.0 
    
### Label Tool
Tool to facilitate labelling raw images semi-automatically
In the folder label_tool folder
1. For parameter searching for canny_multiplider and denoise kernel size
    * python explore_edge_extraction_user_params.py
1. Specify parameters in 
    * UserParams.py  
1. Edge extraction Step
    * python extract_edge.py  
1. Manual Fix Step
    * Among the generated edges, connect fragmented edges and delete the wrong edges
1. Post processing step to fill the extracted edges
    * python segment_edge.py  


### Segmentation Model Training and Prediction
In models folder,  
* Set User parameters for datasets and etc. in the
    * UserParams.py
* Cropping in crop folder
    * crop_augment_split.py
* Training & Prediction in models folder
    * train_mars.py
    * prediction.py

### Evaluation
To replicate the evaluation results such as bar graphs, line graphs, bubble plots, and violin plots,
In evaluation folder,  
Before running code, please install [Correspondence Algorithm](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark)  
* Edit parameters 
    * evaluation/GlobalConfig.m
* To evaluate F1, precision and recall 
    * evaluation/evaluation_f1/Evaluation_f1.m
* For Edge evolution
    * evaluation/evaluation_f1/run_overlap_compare.m
* Violin Plot
    * evaluation/violin_compare.m  
<!-- end of the list -->

To replicate SEG-Grad-CAM results
* set appropriate settings in UserParams.py
* SegGradCAM/main.py


### Morphodynamics
* Download protrusion and windowing package at 
* Add directory and subfolders of the downloaded package to the path 
* Type movieSelectorGUI in the command window in MATLAB
* Create new movie and perform windowing.