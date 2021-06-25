'''
Junbong Jang
6/17/2021

To draw precision recall curve
'''

import sys
sys.path.append('..')
from UserParams import UserParams

import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score, auc, roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
import math
from tqdm import tqdm


def plot_precision_recall_curve(y_true, y_pred, save_base_path):
    lr_precision, lr_recall, _ = precision_recall_curve(y_true, y_pred)
    lr_f1, lr_auc = f1_score(y_true, y_pred), auc(lr_recall, lr_precision)

    no_skill = len(y_true[y_true == 1]) / len(y_true)
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    plt.plot(lr_recall, lr_precision, marker='.',
             label=f'F1={round(lr_f1,3)} auc={round(lr_auc,3)}')

    plt.title('Precision-Recall curve')
    plt.xlabel('Recall', fontsize='large')
    plt.ylabel('Precision', fontsize='large')

    plt.xlim(left=0)
    plt.ylim(bottom=no_skill)
    plt.legend()
    plt.grid()
    plt.savefig(save_base_path + 'precision_recall_curve.png')
    plt.close()


def plot_precision_recall_curve_at_thresholds(y_true, precision_list, recall_list, save_base_path):
    no_skill = len(y_true[y_true == 1]) / len(y_true)
    # include both endpoints
    precision_list = precision_list + [1,no_skill]
    recall_list = recall_list + [0,1]

    # sort them
    recall_sort_index = np.argsort(recall_list)
    precision_list = [precision_list[i] for i in recall_sort_index]
    recall_list = [recall_list[i] for i in recall_sort_index]

    no_skill_auc = auc([0, 1], [no_skill, no_skill])
    lr_auc = auc(recall_list, precision_list)

    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label=f'No Skill AUC={round(no_skill_auc, 3)}', lw=2)
    plt.plot(recall_list, precision_list, marker='.',
             label=f'AUC={round(lr_auc, 3)}', lw=2)

    plt.title('Precision-Recall curve', fontsize='x-large')
    plt.xlabel('Recall', fontsize='large')
    plt.ylabel('Precision', fontsize='large')

    plt.xlim(left=0, right=1.02)
    plt.ylim(bottom= math.floor(no_skill*100)/100)
    plt.legend()
    plt.grid()
    print(save_base_path)
    plt.savefig(save_base_path + 'precision_recall_curve_at_thresholds.png')
    plt.close()


def calc_precision_recall_thresholds(y_true, y_pred):
    # threshold y_pred, given boolean y_true
    precision_list = []
    recall_list = []
    f1_list = []
    print(y_true.shape, y_true.dtype, y_pred.shape, y_pred.dtype)
    for threshold in np.linspace(0,1,11):
        y_true_temp, y_pred_temp = np.copy(y_true), np.copy(y_pred)
        print(threshold)
        y_pred_temp[y_pred_temp>threshold] = 1
        y_pred_temp[y_pred_temp<=threshold] = 0

        precision_list.append(precision_score(y_true_temp, y_pred_temp))
        recall_list.append(recall_score(y_true_temp, y_pred_temp))
        f1_list.append(f1_score(y_true_temp, y_pred_temp))

    return precision_list, recall_list, f1_list

if __name__ == "__main__":
    constants = UserParams('predict')
    model_index = 0
    frame = 2
    repeat_index = 0
    model_name = constants.model_names[model_index]
    dataset_folder = constants.dataset_folders[model_index]
    dataset_name = constants.dataset_names[model_index]

    save_path = constants.get_save_prediction_path(dataset_name, model_name, frame, repeat_index)
    load_path = '../models/' + save_path
    y_true = np.load(load_path + 'class_list_true.npy', allow_pickle=True)
    y_pred = np.load(load_path + 'class_list_pred.npy', allow_pickle=True)

    precision_list, recall_list, f1_list = calc_precision_recall_thresholds(y_true, y_pred)
    print('precision', precision_list)
    print('recall', recall_list)
    print('f1', f1_list)
    plot_precision_recall_curve_at_thresholds(y_true, precision_list, recall_list, save_path)