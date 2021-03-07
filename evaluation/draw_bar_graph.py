'''
Author Junbong Jang
Date 12/25/2020

Plot bar graph for dice / f1 evaluation
not used, I used Matlab instead
'''

import sys
sys.path.append('../')
from UserParams import UserParams

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import statistics


def bar_graph(loaded_data, metric_type):
    ax = plt.gca()
    sns.barplot(x="day", y=metric_type, data=loaded_data)
    plt.title(f'{metric_type} in each cluster')
    # plt.xlabel()

    plt.tight_layout()
    plt.savefig(root_save_path + f'{metric_type} bar.png')
    plt.close()



if __name__ == "__main__":
    constants = UserParams('eval')
    root_path = f'./evaluation_dice/dice_coeff_compare/{constants.get_eval_folder_name()}/'
    constants.update_eval_config(0)

    mean_per_prediction = {}
    for repeat_index in range(constants.REPEAT_MAX):
        for model_index in range(len(constants.model_names)):
            for prediction_path_index, prediction_path in enumerate(constants.predict_path_list):
                print(repeat_index,model_index,prediction_path_index)
                a_frame = 6
                save_suffix = '/frame{}_{}_repeat{}/'.format(str(a_frame), constants.model_names[model_index], str(repeat_index))
                saved_path = root_path + save_suffix + f'model{prediction_path_index}_dice_value.npy'
                if os.path.isfile(saved_path):
                    loaded_data = np.load(saved_path)
                    loaded_data = loaded_data.tolist()[0]
                    if prediction_path_index in mean_per_prediction:
                        mean_per_prediction[prediction_path_index] = mean_per_prediction[prediction_path_index] + loaded_data
                    else:
                        mean_per_prediction[prediction_path_index] = loaded_data
                else:
                    print('skipped')

    print(mean_per_prediction)

    for prediction_path_index, prediction_path in enumerate(constants.predict_path_list):
        print(statistics.mean(mean_per_prediction[prediction_path_index]) )

                # bar_graph(loaded_data, 'dice')
