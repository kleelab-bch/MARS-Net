'''
Author: Junbong Jang
Modified Date: 9/20/2020

'''

import matplotlib
matplotlib.use('agg')
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
sys.path.append('..')
from UserParams import UserParams

from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Arial']

font = {'size':'11', 'weight':'normal',}
matplotlib.rc('font', **font)


def traning_curve_compare_two_models(color_list, list_max, frame_num, ylim_bottom_top):
    constants = UserParams('eval')
    constants.update_eval_config(0)
    save_folder = f'generated/training_curves/{constants.get_eval_folder_name()}'
    if os.path.isdir(save_folder) == 0:
        os.makedirs(save_folder)

    graph_line_dict = {}
    my_dpi = 300
    f, ax = plt.subplots(figsize=(1800/my_dpi, 900/my_dpi), dpi=my_dpi)
    for repeat_index in range(constants.REPEAT_MAX):

        for strategy_index, (round_num, strategy_type) in enumerate(zip(constants.round_num, constants.strategy_type)):
            for model_name in constants.model_names:
                print('------------------')
                print(repeat_index, model_name, strategy_type)
                folder_path = '../models/results/history_round{}_{}'.format(round_num, strategy_type)

                try:
                    data = np.load(
                        folder_path + '/history_frame{}_{}_repeat{}.npy'.format(frame_num, model_name, repeat_index),
                        allow_pickle=True, encoding="bytes")
                    data_dict = data.ravel()[0]

                    key = 'loss'
                    if list_max >  len(data_dict[key]):
                        loss_x_coord = list(range(1, len(data_dict[key]) + 1))
                    else:
                        loss_x_coord = list(range(1, list_max+1))

                    line_loss, = ax.plot(loss_x_coord, data_dict[key][:list_max], label=key, color=color_list[strategy_index])

                    key = 'val_loss'
                    ax.plot(loss_x_coord, data_dict[key][:list_max], linestyle='--', label=key, color=color_list[strategy_index])

                    graph_line_dict[str(strategy_index)+'_loss'] = line_loss
                except:
                    print('skipped')

        assert len(constants.strategy_type) == 2

        ax.set_ylim(ylim_bottom_top)
        set_plot_aspect_ratio(ax)
        ax.set_ylabel('Loss')
        ax.set_xlabel('Epoch')
        ax.legend((graph_line_dict['0_loss'], graph_line_dict['1_loss']), ('U-Net', 'VGG19-U-Net Dropout'), loc='upper right', frameon=False)
        # f.suptitle(f'Training Curves repeat {repeat_index}')
        # f.savefig('{}/training_curve_frame{}_repeat{}.png'.format(save_folder, frame_num, repeat_index))
        # plt.close()
    plt.tight_layout()
    f.savefig('{}/training_curve_frame{}.svg'.format(save_folder, frame_num))


def traning_curve_compare_two_models_aggregate(color_list, list_max, frame_num, ylim_bottom_top):
    constants = UserParams('eval')
    constants.update_eval_config(0)
    save_folder = f'generated/training_curves/{constants.get_eval_folder_name()}'
    if os.path.isdir(save_folder) == 0:
        os.makedirs(save_folder)

    loss_list = {}
    val_loss_list = {}

    for list_index in range(0,list_max):
        for repeat_index in range(constants.REPEAT_MAX):
            for strategy_index, (round_num, strategy_type) in enumerate(zip(constants.round_num, constants.strategy_type)):
                constants.update_eval_config(strategy_index)
                for model_name in constants.model_names:
                    print('------------------')
                    print(repeat_index, model_name, strategy_type)
                    folder_path = '../models/results/history_round{}_{}'.format(round_num, strategy_type)

                    try:
                        data = np.load(
                            folder_path + '/history_frame{}_{}_repeat{}.npy'.format(frame_num, model_name, repeat_index),
                            allow_pickle=True, encoding="bytes")
                        data_dict = data.ravel()[0]

                        if len(data_dict['loss']) > list_index:
                            if list_index in loss_list.keys():
                                loss_list[f'{strategy_index}_{list_index}'].append(data_dict['loss'][list_index])
                            else:
                                loss_list[f'{strategy_index}_{list_index}'] = [data_dict['loss'][list_index]]

                        if len(data_dict['val_loss']) > list_index:
                            if list_index in val_loss_list.keys():
                                val_loss_list[f'{strategy_index}_{list_index}'].append(data_dict['val_loss'][list_index])
                            else:
                                val_loss_list[f'{strategy_index}_{list_index}'] = [data_dict['val_loss'][list_index]]

                    except:
                        print('skipped')

    my_dpi = 300
    f, ax = plt.subplots(figsize=(1800/my_dpi, 900/my_dpi), dpi=my_dpi)
    for strategy_index, (round_num, strategy_type) in enumerate(zip(constants.round_num, constants.strategy_type)):
        avg_loss_list = []
        avg_val_loss_list = []
        for list_index in range(0,list_max):
            if f'{strategy_index}_{list_index}' in loss_list.keys():
                avg_loss_list.append(sum(loss_list[f'{strategy_index}_{list_index}']) / len(loss_list[f'{strategy_index}_{list_index}']))

            if f'{strategy_index}_{list_index}' in val_loss_list.keys():
                avg_val_loss_list.append(sum(val_loss_list[f'{strategy_index}_{list_index}']) / len(val_loss_list[f'{strategy_index}_{list_index}']))

        x_coord = list(range(1, len(avg_loss_list) + 1))
        if 'unet' in strategy_type:
            strategy_type = 'U-Net'
        elif 'VGG19_dropout' in strategy_type:
            strategy_type = 'VGG19-U-Net Dropout'
        ax.plot(x_coord, avg_loss_list, label=strategy_type, color=color_list[strategy_index])
        ax.plot(x_coord, avg_val_loss_list, linestyle='--', color=color_list[strategy_index])

    ax.set_ylim(ylim_bottom_top)
    set_plot_aspect_ratio(ax)
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    ax.legend(loc='upper right', frameon=False)

    plt.tight_layout()
    f.savefig('{}/aggreagate_training_curve_frame{}.svg'.format(save_folder, frame_num))


def training_curve_per_model(frame_num):
    constants = UserParams('train')

    folder_path = '../models/results/history_round{}_{}'.format(constants.round_num, constants.strategy_type)
    save_folder = 'generated/training_curves/{}'.format(constants.strategy_type)
    if os.path.isdir(save_folder) == 0:
        os.makedirs(save_folder)

    loss_epochs_list = []
    times_list = []
    for repeat_index in range(constants.REPEAT_MAX):
        for model_name in constants.model_names:
            print('------------------')
            print(repeat_index, model_name, frame_num)
            data = np.load(
                folder_path + '/history_frame{}_{}_repeat{}.npy'.format(frame_num, model_name, repeat_index),
                allow_pickle=True, encoding="bytes")
            data_dict = data.ravel()[0]

            f, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True)

            for key in data_dict:
                if 'val_' in key:
                    alpha = 0.3
                else:
                    alpha = 1.0

                if key == 'loss' or key == 'val_loss':
                    loss_x_coord = list(range(1, len(data_dict[key]) + 1))
                    ax1.plot(loss_x_coord, data_dict[key], label=key, alpha=alpha)
                    ax1.set_xlabel('Epoch')
                    ax1.legend(loc='center right', frameon=False)

                elif key == 'dice_coef' or key == 'val_dice_coef':
                    x_coord = list(range(1, len(data_dict[key]) + 1))
                    ax2.plot(x_coord, data_dict[key], label=key, alpha=alpha)
                    ax2.set_xlabel('Epoch')
                    ax2.legend(loc='center right', frameon=False)
            loss_epochs_list.append(len(loss_x_coord))
            times_list = times_list + data_dict['times']
            print('Epochs:', len(loss_x_coord))
            print('Average number of seconds: ', sum(data_dict['times']) / len(data_dict['times']))

            f.suptitle('Training Curves')
            f.savefig('{}/training_curve_frame{}_{}_repeat{}.png'.format(save_folder, frame_num, model_name,
                                                                         repeat_index))
            plt.close()

    print('------------------------------')
    print('Average number of Epoch: ', sum(loss_epochs_list) / len(loss_epochs_list))
    print('Total Average number of seconds: ', sum(times_list) / len(times_list))


def set_plot_aspect_ratio(input_ax):
    ratio = 1.0
    xleft, xright = input_ax.get_xlim()
    ybottom, ytop = input_ax.get_ylim()
    input_ax.set_aspect(abs((xright - xleft) / (ybottom - ytop)) * ratio)


def test_open_data():
    # strategy_type = 'single_micro_VGG19_dropout'
    # model_names=['A', 'B', 'C', 'D', 'E']
    strategy_type = 'VGG19_dropout'
    model_names = ['ABCD','ABCE', 'ABDE', 'ACDE', 'BCDE']
    frame = 34
    for model_name in model_names:
        root_path = '../models/results/history_round1_{}/history_frame{}_{}_repeat0.npy'.format(strategy_type, frame, model_name)
        data = np.load(root_path, allow_pickle=True, encoding="bytes")
        data_dict = data.ravel()[0]
        print('Total training in hours: ', sum(data_dict['times'])/3600)



if __name__ == "__main__":
    # blue, orange, green, yellow, red, skyblue, violet
    # ['#0071bc', '#d85218', '#76ab2f', '#ecb01f', '#a1132e', '#4cbded', '#7d2e8d']
    color_list = ['#0071bc', '#d85218']
    ylim_bottom_top = (0.01, 0.12)
    list_max = 100
    frame_num = 2

    # traning_curve_compare_two_models(color_list, list_max, frame_num, ylim_bottom_top)
    # traning_curve_compare_two_models_aggregate(color_list, list_max, frame_num, ylim_bottom_top)
    # training_curve_per_model(frame_num)
    test_open_data()