""" 1) orig, gt, predicted, sgc for:
1.1) 1 px
1.2) class
1.3) roi

2) plot same things for all act in a folder

3) Subtraction experiment

4) mask out a piece of sgc output (input: any mask of 0 and 1s,
e.g biased mask, roi, everything outside roi)
to see how much saliency is inside

"""
import os
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

from seggradcam import SegGradCAM, BiasRoI, SuperRoI, ClassRoI, PixelRoI
import numpy as np


class SegGradCAMplot(SegGradCAM):
    def __init__(self, seggradcam, next_dict, image_id, image_name, ground_truth_mask, pred_image=None, gt=None,
                n_classes=None, outfolder=None):

        # SegGradCAM.__init__(self, seggradcam) #, trainparam) #?

        # self.model = seggradcam.input_model
        self.gt_mask = ground_truth_mask
        self.cls = seggradcam.cls  # class
        # prop_from_layer is the layer with logits prior to the last activation function
        # self.prop_from_layer = seggradcam.prop_from_layer
        self.prop_to_layer = seggradcam.prop_to_layer  # an intermediate layer, typically of the bottleneck layers

        self.roi = seggradcam.roi  # M, a set of pixel indices of interest in the output mask.
        self.orig_image = self.roi.image
        self.pred_image = pred_image

        # self.normalize = seggradcam.normalize  # [True, False] normalize the saliency map L_c
        # self.abs_w = seggradcam.abs_w  # if True, absolute function is applied to alpha_c
        # self.posit_w = seggradcam.posit_w  # if True, ReLU is applied to alpha_c
        # self.alpha_c = seggradcam.alpha_c  # alpha_c, weights for importance of feature maps
        # self.A = seggradcam.A  # A, feature maps from the intermediate prop_to_layer
        # self.grads_val = seggradcam.grads_val  # gradients of the logits y with respect to all pixels of each feature map 𝐴^𝑘
        self.cam = seggradcam.cam
        self.n_classes = n_classes
        self.outfolder = outfolder
        self.image_name = image_name

        # only for TexturedMNIST
        self.next_dict = next_dict
        self.image_id = image_id
        self.gt = gt

        timestamp = datetime.now()
        self.timestr = timestamp.strftime("%m%d%H%M")
        # different for cityscapes&mnist ?
        if self.orig_image.shape[0] == 1:
            self.ximg = self.orig_image[0,...] # ---
            self.cmap_orig = 'gray'
        else:
            self.ximg = self.orig_image
            self.cmap_orig = None


    def defaultScales(self):
        classes_cmap = plt.get_cmap('Spectral', self.n_classes)
        scale_fig = 10
        fonts = 40
        scatter_size = 330 * scale_fig
        return classes_cmap, scale_fig, fonts, scatter_size

    def explainBase(self, title1, title1bias, start_save_name, pixel=False):
        """"""
        classes_cmap, scale_fig, fonts, scatter_size = self.defaultScales()
        fonts = int(fonts / 3)
        scatter_size = int(scatter_size / 3)
        plt.figure(figsize=(scale_fig, scale_fig))

        # move last channel to first channel for visualization
        # new_img = np.moveaxis(self.ximg, -1, 0)
        # new_img = np.moveaxis(new_img, -1, 0)
        # plt.imshow(new_img, vmin=0, vmax=1, cmap=self.cmap_orig)
        # class contour
        X, Y = self.roi.meshgrid()

        if pixel:
            classroi = ClassRoI(self.gt_mask, self.orig_image, self.cls)
            roi_contour1 = classroi.roi
        else:
            roi_contour1 = self.roi.roi

        if self.pred_image is not None:
            plt.contour(X, Y, self.pred_image, colors='blue')
        plt.contour(X, Y, roi_contour1, colors='green')

        # plt.title(title1, fontsize=fonts)
        # biased texture contour
        if self.next_dict and self.image_id:
            biasroi = BiasRoI(self.next_dict, self.image_id)
            plt.contour(X, Y, biasroi.biased_mask, colors='magenta')
            if biasroi.biased_mask.any() != 0:
                plt.title(title1bias, fontsize=fonts)
        plt.imshow(self.cam, cmap='hot') # , alpha=0.6, vmin=0,vmax=1,
        plt.axis('off')

        tick_values = np.arange(start=0.0, stop=np.amax(self.cam)+0.01, step=np.amax(self.cam)/2)
        jet = plt.colorbar(fraction=0.046, pad=0.04, ticks=tick_values)
        jet.set_label(label="Importance", size=fonts)

        jet.ax.set_yticklabels(tick_values, size=fonts) # [0, 0.2, 0.4, 0.6, 0.8, 1]
        if pixel:
            plt.scatter(self.roi.j, self.roi.i, color='white', s=scatter_size)  # j then i or i,j ?

        plt.savefig(os.path.join(self.outfolder, self.image_name + '_' + start_save_name + str(self.cls) + '_' + self.prop_to_layer + ".png"))
        plt.close()

    def explainClass(self):
        """Plot seg-grad-cam explanation for a selected class channel"""
        title1 = 'Seg-Grad-CAM for class %d' % (self.cls)
        title1bias = 'Seg-Grad-CAM for class %d \n& biased texture in magenta' % (self.cls)
        start_save_name = 'class'
        self.explainBase(title1, title1bias, start_save_name)

    def explainRoi(self):
        """Plot seg-grad-cam explanation for a region of interest"""
        title1 = 'Seg-Grad-CAM for RoI(pink) of class %d' % (self.cls)
        title1bias = 'Seg-Grad-CAM for RoI(pink) of class %d \n& biased texture in magenta' % (self.cls)
        start_save_name = 'roi_cl'
        self.explainBase(title1, title1bias, start_save_name)

    def explainPixel(self):
        """Plot seg-grad-cam explanation for a selected single pixel"""
        i, j = self.roi.i, self.roi.j
        title1 = 'Seg-Grad-CAM for pixel [%d,%d]. Class %d' % (i, j, self.cls)
        title1bias = 'Seg-Grad-CAM for pixel [%d,%d], class %d \n& biased texture in magenta' % (i, j, self.cls)
        start_save_name = 'pixel' + str(i) + '_' + str(j) + '_cl'
        self.explainBase(title1, title1bias, start_save_name, pixel=True)

    def baseGtPrediction(self, title2, start_save_name_tetra, pixel=False):

        if self.next_dict and self.image_id:
            self.gt = self.next_dict[1][self.image_id]
        if type(self.gt) == 'NoneType':  # .all()==None:
            # raise error
            print("Provide the groundtruth mask")

        classes_cmap, scale_fig, fonts, scatter_size = self.defaultScales()
        plt.figure(figsize=(30 * scale_fig, 15 * scale_fig))
        plt.subplot(221)
        plt.imshow(self.ximg, vmin=0, vmax=1, cmap=self.cmap_orig)
        plt.title('Input image', fontsize=fonts)
        X, Y = self.roi.meshgrid()
        # biased texture contour
        if self.next_dict and self.image_id:
            biasroi = BiasRoI(self.next_dict, self.image_id)
            plt.contour(X, Y, biasroi.biased_mask, colors='magenta')
            if biasroi.biased_mask.any() != 0:
                plt.title('Input image & biased texture', fontsize=fonts)
        if pixel:
            i, j = self.roi.i, self.roi.j
            plt.scatter(j, i, color='red', s=scatter_size)
        plt.tick_params(labelsize=fonts)

        plt.subplot(222)
        plt.axis('off')
        plt.imshow(self.ximg, vmin=0, vmax=1, cmap=self.cmap_orig)
        if pixel:
            plt.scatter(j, i, color='white', s=scatter_size)
        else:
            # class contour
            plt.contour(X, Y, self.roi.roi, colors='pink')

        plt.title(title2, fontsize=fonts)
        plt.imshow(self.cam, cmap='jet',  # vmin=0,vmax=1,
                   alpha=0.6)
        jet = plt.colorbar(fraction=0.046, pad=0.04, ticks=[0, 0.2, 0.4, 0.6, 0.8, 1])
        jet.set_label(label="Importance", size=fonts)
        jet.ax.set_yticklabels([0, 0.2, 0.4, 0.6, 0.8, 1], size=fonts)

        plt.subplot(223)
        plt.axis('off')
        plt.imshow(np.argmax(self.gt, axis=-1), vmin=0, vmax=self.n_classes - 1, cmap=classes_cmap)
        plt.title('Groundtruth mask', fontsize=fonts)
        if pixel:
            plt.scatter(j, i, color='white', s=scatter_size)

        plt.subplot(224)
        plt.axis('off')
        plt.imshow(np.argmax(self.gt_mask, axis=-1), vmin=0, vmax=self.n_classes - 1, cmap=classes_cmap)
        # plt.colorbar(fraction=0.046, pad=0.04,cmap=cm)
        cbar = plt.colorbar(cmap=classes_cmap, fraction=0.046, pad=0.04, ticks=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        cbar.set_label(label="Classes", size=fonts)
        # categories =['Void','Flat','Const-\nruction','Object','Nature','Sky','Human','Vehicle']
        cbar.ax.tick_params(labelsize=fonts)
        # cbar.ax.set_yticklabels(categories, size=fonts)

        if pixel:
            plt.scatter(j, i, color='white', s=scatter_size)

        plt.title('Predicted mask', fontsize=fonts)
        plt.tight_layout(pad=0.1, w_pad=15, h_pad=5.0)

        plt.savefig(os.path.join(self.outfolder,
                                 start_save_name_tetra + str(self.cls) + '_to_act' + self.prop_to_layer.split('_')[1] +'_'+
                                 self.timestr + ".png"))
        plt.close()

    def classGtPrediction(self):
        """Plot 4 images: original, ground truth, predicted mask, seg-grad-cam explanations for a selected class channel"""
        title2 = 'Seg-Grad-CAM for class %d' % (self.cls)
        start_save_name_tetra = 'tetra_class'
        self.baseGtPrediction(title2, start_save_name_tetra)

    def roiGtPrediction(self):
        """Plot 4 images: original, ground truth, predicted mask, seg-grad-cam explanations for a region of interest"""
        title2 = 'Seg-Grad-CAM for RoI(pink) of class %d' % (self.cls)
        start_save_name_tetra = 'tetra_roi'
        self.baseGtPrediction(title2, start_save_name_tetra)

    def pixelGtPrediction(self):
        """Plot 4 images: original, ground truth, predicted mask, seg-grad-cam explanations for a selected single pixel"""
        i, j = self.roi.i, self.roi.j
        title2 = 'Seg-Grad-CAM for pixel [%d,%d]. Class %d' % (i, j, self.cls)
        start_save_name_tetra = 'tetra_pixel' + str(i) + '_' + str(j) + '_cl'
        self.baseGtPrediction(title2, start_save_name_tetra, pixel=True)

