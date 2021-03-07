import sys
sys.path.append('..')

import numpy as np
from UserParams import UserParams
import umap
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.utils import shuffle

import plotly.express as px
import plotly.offline as pyo
pyo.init_notebook_mode()


def combine_feature_vectors():
    # combine feature vectors
    constants = UserParams('predict')
    root_prediciton_path = "../models/results/predict_wholeframe_round{}_{}/".format(constants.round_num,
                                                                                     constants.strategy_type)

    combined_feature_vectors = None
    group_id_list = []
    feature_id_list = []

    for model_index in range(len(constants.model_names)):
        model_name = constants.model_names[model_index]
        for dataset_index in range(len(constants.dataset_names)):
            dataset_name = constants.dataset_names[dataset_index]

            # phase 5 movies
            # mDia 5 movies
            # TIRF 6 movies
            dataset_group = 0
            if dataset_index >= 5 and dataset_index < 10:
                dataset_group = 1
            elif dataset_index >= 10:
                dataset_group = 2

            for frame in constants.frame_list:
                print(model_index, dataset_group, frame)
                repeat_index = 0
                save_path = root_prediciton_path + '{}/frame{}_{}_repeat{}/style_feature_vector.npy'.format(
                    dataset_name, str(frame), model_name, str(repeat_index))
                loaded_feature_vectors = np.load(save_path, allow_pickle=True, encoding="bytes")

                if combined_feature_vectors is None:
                    combined_feature_vectors = loaded_feature_vectors
                else:
                    combined_feature_vectors = np.concatenate((combined_feature_vectors, loaded_feature_vectors),
                                                              axis=0)

                group_id_list = group_id_list + [f"{dataset_group}" for x in range(loaded_feature_vectors.shape[0])]
                feature_id_list = feature_id_list + [f"{dataset_index}_{x}" for x in
                                                     range(loaded_feature_vectors.shape[0])]

    combined_feature_vectors, group_id_list, feature_id_list = shuffle(combined_feature_vectors, group_id_list,
                                                                       feature_id_list, random_state=0)
    print(combined_feature_vectors.shape)
    print(len(feature_id_list))

    return combined_feature_vectors, group_id_list, feature_id_list


def visualize_umap(combined_feature_vectors, group_id_list, feature_id_list):
    # UMAP data and then visualize
    for n in range(8, 19, 2):
        for d in range(25, 81, 5):
            d = d / 100
            embedding = umap.UMAP(n_neighbors=n, min_dist=d, random_state=42).fit_transform(combined_feature_vectors)
            print(embedding.shape, n, d)
            x = embedding[:, 0]
            y = embedding[:, 1]

            feature_data_dict = {'x': x, 'y': y, 'group_id': group_id_list, 'feature_id': feature_id_list}
            feature_df = pd.DataFrame.from_dict(feature_data_dict)

            fig = px.scatter(feature_df, x="x", y="y", color="group_id", hover_data=['feature_id'],
                             width=600, height=560,
                             color_discrete_sequence=px.colors.qualitative.D3,
                             title=f"UMAP of the style vectors at neighbors={n} mindist={d}",
                             labels={'group_id': 'Dataset Type'})

            fig.update_layout(yaxis={'visible': False, 'showticklabels': False},
                              xaxis={'visible': False, 'showticklabels': False})

            fig.write_image(f'generated/style_umap_n{n}_d{d}.png')


if __name__ == "__main__":
    combined_feature_vectors, group_id_list, feature_id_list = combine_feature_vectors()
    visualize_umap(combined_feature_vectors, group_id_list, feature_id_list)
