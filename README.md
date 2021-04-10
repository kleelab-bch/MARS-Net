# MARS-Net
Deep learning-based segmentation pipeline for profiling cellular morphodynamics from multiple types of live cell microscopy  
To learn more about MARS-Net, read the [paper](https://www.biorxiv.org/content/10.1101/191858v3)
<div text-align="center">
  <img width="300" src="./assets/MARS-Net_logo.png" alt="MARS-Net Logo">
</div>

## Run Demo
You can quickly segment one of our live cell movie in this demo  
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kleelab-bch/MARS-Net/blob/master/run_MARS_Net_demo.ipynb) 
<!-- end of the list -->
To test our pipeline from the scratch, the user needs to crop images, and train the models before running inference to segment movies which can take several hours.  
This demo allows users to see the segmentation performance of MARS-Net and U-Net already trained on our live cell movies.
## Software Requirements
MARS-Net pipeline has been tested on Ubuntu 16.04
* Please download MATLAB 2019b
    * Other versions might work but we didn't test our pipeline on other versions.
* [Correspondence Algorithm](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark) developed by University of California Berkeley Segmentation Benchmark for F1, precision and recall evaluation.
* [Windowing and Protrusion package](https://github.com/DanuserLab/Windowing-Protrusion) developed by Gaudenz Danuser lab (UT Southwestern) for Morphoydynamics profiling.  
* For training and segmenting the cell boundary, Python v3.6.8, TensorFlow (v1.15 or v2.3), and Keras v2.3.1 .  
    * Tensorflow v2.3 on RTX Titan GPU with CUDA 10.1
    * Tensorflow v1.15 on GTX 1080Ti GPU with CUDA 10.0 

## Pipeline
The pipeline consists of label tool, segmentation modeling, and morphodynamics profiling.    
There is no installation procedure except for downloading our code from Github or installing the software requirements above.

* Before running the pipeline, please specify the following parameters in UserParams.py
    * strategy_type
    * dataset_folders
    * img_folders
    * mask_folders
    * frame_list
    * dataset_names 
    * model_names 
    * REPEAT_MAX

The example phase contrast movie with its labeled mask is in the assets folder.

### Label Tool
Tool to facilitate labelling raw images semi-automatically
In the folder label_tool folder
1. To determine hysteresis thresholding for canny detector and kernel size for blurring
    * python explore_edge_extraction_user_params.py
    * Compare results in generated_explore_edge folder
1. Edge extraction Step
    * Python extract_edge.py 
1. Manual Fix Step
    * The generated edges in generated_edge folder, connect fragmented edges and delete the wrong edges
    * We used [ImageJ](https://imagej.nih.gov/ij/download.html) or [GIMP](https://www.gimp.org/) for manual fix after overlaying edge over the original image
1. Post processing step to fill the extracted edges
    * python segment_edge.py will save results in generated_segmentation folder


### Segmentation Model Training and Prediction
This section is for training deep learning models from scratch and segmenting the live cell movies 
* Put your live cell movies into the assets folder. Our pipeline assumes leave-one-movie-out cross validation so please provide multiple movies.
* To crop patches, run
    * crop/crop_augment_split.py
* To Train, run
    * models/train_mars.py
* To segment live cell movies, run
    * models/prediction.py

### Evaluation
This section is for replicating our evaluation results including bar graphs, line graphs, and violin plots,
In evaluation folder,  
Before running code, please install [Correspondence Algorithm](https://github.com/davidstutz/extended-berkeley-segmentation-benchmark)  
* Edit parameters in evaluation/GlobalConfig.m
    * prediction_path_list
    * display_names
    * frame_list
    * root_path
    * img_root_path
* Before visualizing the evaluated results, calculate F1, precision and recall from the segmented movies, run
    * evaluation/evaluation_f1/run_overlap_mask_prediction.m
* To draw bar graphs and line graphs across different training frames, run
    * evaluation/evaluation_f1/visualize_across_frames_datasets.m
* To draw edge evolution, run
    * evaluation/evaluation_f1/run_overlap_compare.m
* To draw violin plot, run
    * evaluation/violin_compare.m  

<!-- end of the list -->
Unlike MATLAB code above, learning curves and bubble plots are drawn using Python
* To draw learning curve, run
    * evaluation/draw_learning_curve.py
* To draw bubble plot, run
    * evaluation/bubble_training_curve.ipynb
<!-- end of the list -->

To replicate SEG-Grad-CAM results, run
* SegGradCAM/main.py


### Morphodynamics
* For single cell cropping the segmented movie, run
    * rotate_crop_img.py
* Download [Windowing and Protrusion package](https://github.com/DanuserLab/Windowing-Protrusion)
* Add directory and sub directory of the downloaded package to the MATLAB path 
* Type movieSelectorGUI in the command window in MATLAB
* Create new movie and perform windowing on the segmented movie from the previous step.
* For details, refer to [Windowing and Protrusion package](https://github.com/DanuserLab/Windowing-Protrusion) Github page.