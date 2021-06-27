import matplotlib.pyplot as plt
import pickle

import pandas as pd
import numpy as np
import preproccessing
from preproccessing.utils import create_num_to_name_dict
import preproccessing.preproccess as preproccess
import seaborn as sbn

FILES = [
    # "data_files\\labels\\representatives_1000_MinibatchKmeans_200_Tsne_2_Standard.pickle",
    "data_files\\labels\\representatives_1000_MinibatchKmeans_200_Tsne_2_MinMax.pickle",
    "data_files\\labels\\representatives_1000_MinibatchKmeans_200_None_2_MinMax.pickle",
    # "data_files\\labels\\representatives_1000_MinibatchKmeans_200_None_2_Standard.pickle"
]

titles = ['None', 'Tsne']


####################################################################################################
# Code to replace the name of the structures in the cluster files with their corresponding number  #
####################################################################################################

def triangle_to_symmetric_matrix(matrix_path, dict_path = False):
    df = pd.read_csv(matrix_path, sep = ',', header = None)
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
    """
    this function gets a value and a dictionary and returns the key in which this value appears as {key: value}.
    if {key: value} doesn't exist in dict, returns None
    """
    for key, value in dict.items():
        if val in value:
            return key
    return None


def find_clusters(samples):
    '''
    this function recieves samples and using the dict from parse_DCHit andd the false2true_index_dict (taken from the file
    "preproccessing/struct_to_index_second_1000.txt") finds the right index in of the sample in alons clusters dict
    # samples is an np.array of the samples from the pickle file
    returns samples with another column with the right key
    '''
    alons_clusters_dict = preproccess.parse_DCHit()
    # false to true index (like indexes file)
    false2true_index_dict = preproccessing.utils.create_num_to_name_dict(
        "preproccessing/struct_to_index_second_1000.txt", False)
    # adding another column to samples in order to place the labels in the matrix
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
    df = pd.DataFrame({"Max": X},
                      index=list(range(1, 201)))
    sbn.heatmap(df, cmap="YlGnBu")
    plt.title(title)
    plt.savefig(save_path)
    plt.show()


def parse_pickle_files(files):
    '''
    this function takes as argument a list of pathes to pickle files and then parses them and returns the average
    value of the top probabilites for each file as a list
    '''
    # iterating over the files
    probabilities_avg = []
    for k, path in enumerate(files):
        pickle_file = pickle.load(open(path, 'rb'))
        np_samples = np.array(pickle_file)
        answers = find_clusters(np_samples).astype(int)
        yitz_labels = np.unique(answers[:, 0])
        alon_labels = np.unique(answers[:, 1])
        # correlation_matrix is used to count the amount of occurences of every tuple of clusters from the two
        # different cluster methods
        correlation_matrix = np.zeros((len(yitz_labels) + 1, len(alon_labels) + 1))
        # adding the labels to the matrix as first row and first column
        correlation_matrix[0] = np.array([0] + list(yitz_labels))
        correlation_matrix[:, 0] = np.array([0] + list(alon_labels))
        for i, y in enumerate(yitz_labels):
            for j, a in enumerate(alon_labels):
                # iterating over the tuples of labels
                for ans in answers:
                    # checking that this tupleof answers exists
                    if ans[0] == y and ans[1] == a:
                        # incrementing by 1 the count of this tuple
                        correlation_matrix[i + 1, j + 1] += 1
        # after calculating the corrolations between the two different clusterings, now we will compute the
        # probabilites of the most frequent tuples
        correlation_matrix = correlation_matrix.astype(int)
        # making a copy to work on:
        copy_dist_matrix = correlation_matrix[1:, 1:]
        probabilites_arr = []
        max_arr = []
        for row in copy_dist_matrix:
            probabilites_arr.append(row / sum(row))
            max_arr.append(max(row / sum(row)))
        plot_heatmap(probabilites_arr, max_arr, 'Probabilities of cluster matching, dim reduction: ' + titles[k], "data_files\\Figures\\figs" + str(k))
        # plot_heatmap_max(max_arr, 'Maximum Probabilites of cluster matching', "data_files\Figures\\figsmax"+ str(i))
        probabilities_avg.append(np.average(max_arr))
    # return the average of all of the file's probabilites
        plot_heatmap_max(np.array(max_arr), "max probability for same cluster heatmap, avg= " + "{:.2f}".format(np.average(max_arr)),"data_files\\Figures\\figs_max" + str(k))
        break
    return probabilities_avg


# avg = parse_pickle_files(FILES)
# print(avg)
