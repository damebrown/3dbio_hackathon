import matplotlib.pyplot as plt
import pickle

import pandas as pd
import numpy as np
import preproccessing
from preproccessing.utils import create_num_to_name_dict
import random
import preproccessing.preproccess as prcs
import seaborn as sbn


####################################################################################################
# Code to replace the name of the structures in the cluster files with their corresponding number  #
####################################################################################################

def triangle_to_symmetric_matrix(matrix_path, dict_path=False):
    df = pd.read_csv(matrix_path, sep=',', header=None)
    orig_matrix = df.values[:, :df.values.shape[0]]
    # orig_matrix = np.array([[123,107,113,103],[140,110,108, np.nan],[136,106, np.nan, np.nan],[115, np.nan, np.nan, np.nan]])
    new_matrix = np.zeros((orig_matrix.shape[0], orig_matrix.shape[0]))
    for i in range(orig_matrix.shape[0]):
        left_side = [orig_matrix[k][j] for k, j in zip(range(0, i, 1), range(i, 0, -1))]
        right_side = [orig_matrix[i][j] for j in range(0, orig_matrix.shape[0]) if
                      not pd.isna(orig_matrix[i][j])]
        new_matrix[i] = np.array(left_side + right_side)
    if dict_path:
        num_to_name_dict = create_num_to_name_dict(dict_path, False)
        matches_df = pd.DataFrame(new_matrix)
        matches_df['struct_name'] = pd.Series(
            [num_to_name_dict[str(num)] for num in range(1, new_matrix.shape[0] + 1)])
        return matches_df, new_matrix.astype(int)
    return new_matrix.astype(int)
    # create_num_to_name_dict("name_num_dict")


# replace_names_cluster_files("output_0.8.clstr", "name_num_dict")
# triangle_to_symmetric_matrix("correspond_1000_first_batch_08.csv",
#                              "preproccessing/struct_to_index_first_1000.txt")


def get_key(val, dict):
    for key, value in dict.items():
        if val in value:
            return key
    return None


def func(samples):
    alons_clusters_dict = prcs.parse_DCHit()
    false2true_index_dict = preproccessing.utils.create_num_to_name_dict(
        "preproccessing/struct_to_index_second_1000.txt", False)
    # false to true index (like indexes file)
    samples = np.c_[samples, np.zeros(len(samples))].astype(int)
    for i, sample in enumerate(samples):
        x = int(false2true_index_dict[str(i + 1)])
        key = get_key(x, alons_clusters_dict)
        samples[i, -1] = int(key)
    return samples


def plot_heatmap(X: np.array, y: np.array, title: str, save_path: str = None):
    mat_temp = np.concatenate((X, np.array(y).reshape(200, 1)), axis = 1)
    k = np.array(sorted(mat_temp, key = lambda x: x[200]))
    mat_temp = mat_temp[np.argsort(mat_temp[:, 200])]
    mat_temp = k
    mat_drop_column = np.delete(mat_temp, 200, 1)
    plt.figure()
    sbn.heatmap(mat_drop_column, cmap = "YlGnBu")
    plt.title(title)
    plt.savefig(save_path)
    plt.show()


def plot_heatmap_max(X: np.array, title: str, save_path: str = None):
    plt.figure()
    sbn.heatmap(X, cmap = "YlGnBu")
    plt.title(title)
    plt.savefig(save_path)
    plt.show()


# yitzhack's clusters
files = [
    "representatives_1000_MinibatchKmeans_200_Tsne_2_Standard.pickle",
    "representatives_1000_MinibatchKmeans_200_Tsne_2_MinMax.pickle",
    "representatives_1000_MinibatchKmeans_200_None_2_MinMax.pickle",
    "representatives_1000_MinibatchKmeans_200_None_2_Standard.pickle"]
for k, path in enumerate(files):
    f = open(path, 'rb')
    file = pickle.load(f)
    smp = np.array(file)
    answers = func(smp).astype(int)
    yitz_labels = np.unique(answers[:, 0])
    alon_labels = np.unique(answers[:, 1])

    mat = np.zeros((len(yitz_labels) + 1, len(alon_labels) + 1))
    mat[0] = np.array([0] + list(yitz_labels))
    mat[:, 0] = np.array([0] + list(alon_labels))
    for i, y in enumerate(yitz_labels):
        for j, a in enumerate(alon_labels):
            for ans in answers:
                if ans[0] == y and ans[1] == a:
                    mat[i + 1, j + 1] += 1
    mat = mat.astype(int)
    m = mat[1:, 1:]
    arr = []
    max_arr = []
    for j, r in enumerate(m):
        arr.append(r / sum(r))
        max_arr.append(max(r / sum(r)))
    plot_heatmap(arr, max_arr, 'Probabilites of cluster matching', "data_files\Figures\\figs" + str(k))
    # plot_heatmap_max(max_arr, 'Maximum Probabilites of cluster matching', "data_files\Figures\\figsmax"+ str(i))
    print(np.average(max_arr))

