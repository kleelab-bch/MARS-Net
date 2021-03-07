'''
Author: Junbong Jang
7/28/2020

This is for parameter tuning for the edge extraction
'''

from extract_edge import *

if __name__ == "__main__":
    if not os.path.exists(user_params.saved_edge_user_params_path):
        os.makedirs(user_params.saved_edge_user_params_path)
        
    # get all of dataset's images
    img_list = glob.glob(user_params.img_root_path + '*' + '.png')
    
    # Get an image and its name
    img_path = img_list[0]
    img = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
    img_name = img_path[len(user_params.img_root_path):]
    print(img_name, img.shape)
    print()

    # parameter grid search
    canny_std_multiplier_range = np.arange(0.5, 2.6, 0.1)
    denoise_kernel_size_range = np.arange(2,11,1)
    print('canny_std_multiplier_range: ', canny_std_multiplier_range)
    print('denoise_kernel_size_range: ', denoise_kernel_size_range)
    print()
    print()
    
    for canny_std_multiplier in canny_std_multiplier_range:
        for denoise_kernel_size in denoise_kernel_size_range:
            canny_std_multiplier = round(canny_std_multiplier, 2)
            print('canny: ' + str(canny_std_multiplier) + ' & denoise kernel size: ' + str(denoise_kernel_size))
            guided_edge = extract_edge(img, 'guided', canny_std_multiplier, denoise_kernel_size, debug_mode=False)
            overlaid_img = overlay_edge_over_img(img, guided_edge, save_path='')
            
            save_path = user_params.saved_edge_user_params_path + '/canny_' + str(canny_std_multiplier) + '_denoise_' + str(denoise_kernel_size) + '.png'
            # image save
            plt.imshow(overlaid_img)
            plt.axis('off')
            plt.title('canny: ' + str(canny_std_multiplier) + ' & denoise kernel size: ' + str(denoise_kernel_size))
            plt.tight_layout(pad=0.2) # remove the padding between images
            plt.savefig(save_path,  bbox_inches = 'tight', pad_inches = 0, dpi=240) # remove up and down paddings
            plt.close()