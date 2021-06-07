'''
Junbong Jang
6/7/2021

CAM for image classifier
'''

def visualize_feature_activation_map(feature_map, image_path, image_name, save_path):
    # reference https://codeocean.com/capsule/0685076/tree/v1
    # reduce feature map depth from 512 to 1
    averaged_feature_map = np.zeros(feature_map.shape[:2], dtype=np.float64)
    for i in range(feature_map.shape[0]):
        for j in range(feature_map.shape[1]):
            for k in range(feature_map.shape[2]):
                averaged_feature_map[i, j] = averaged_feature_map[i, j] + feature_map[i, j, k]
    # get image to overlay
    image = cv2.imread(image_path + image_name)
    width, height, channels = image.shape

    # generate heatmap
    cam = averaged_feature_map / np.max(averaged_feature_map)
    cam = cv2.resize(cam, (height, width))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

    # overlay heatmap on original image
    alpha = 0.6
    heatmap_img = cv2.addWeighted(image, alpha, heatmap, 1 - alpha, 0)
    cv2.imwrite(f'{save_path}heatmap_{image_name}', heatmap_img)