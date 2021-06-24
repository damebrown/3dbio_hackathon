import pandas as pd
import numpy as np
from preproccessing.utils import create_num_to_name_dict
import random
import preproccessing


####################################################################################################
# Code to replace the name of the structures in the cluster files with their corresponding number  #
####################################################################################################

def triangle_to_symmetric_matrix(matrix_path, dict_path):
    df = pd.read_csv(matrix_path, sep = ',', header = None)
    orig_matrix = df.values[:, :df.values.shape[0]]
    # orig_matrix = np.array([[123,107,113,103],[140,110,108, np.nan],[136,106, np.nan, np.nan],[115, np.nan, np.nan, np.nan]])
    new_matrix = np.zeros((orig_matrix.shape[0], orig_matrix.shape[0]))
    for i in range(orig_matrix.shape[0]):
        left_side = [orig_matrix[k][j] for k, j in zip(range(0, i, 1), range(i, 0, -1))]
        right_side = [orig_matrix[i][j] for j in range(0, orig_matrix.shape[0]) if
                      not pd.isna(orig_matrix[i][j])]
        new_matrix[i] = np.array(left_side + right_side)

    num_to_name_dict = create_num_to_name_dict(dict_path, False)
    matches_df = pd.DataFrame(new_matrix)
    matches_df['struct_name'] = pd.Series(
        [num_to_name_dict[str(num)] for num in range(1, new_matrix.shape[0] + 1)])
    return matches_df, new_matrix.astype(int)
    # create_num_to_name_dict("name_num_dict")


# replace_names_cluster_files("output_0.8.clstr", "name_num_dict")
triangle_to_symmetric_matrix("correspond_1000_first_batch_08.csv",
                             "preproccessing/struct_to_index_first_1000.txt")


def get_key(val, dict):
    for key, value in dict.items():
        if val in value:
            return key
    return None


def get_indices(samples):
    f = open("preproccessing\\indexes.txt", 'r')
    samples = np.c_[samples, np.zeros(len(samples))]
    for i in range(len(samples)):
        samples[i, -1] = f.readline()
    return samples


# 100 samples, 1 cluster, 1 alon's index
def func(samples):
    # samples = get_indices(samples)
    clusters_dict = preproccessing.preproccess.parse_DCHit()
    samples = np.c_[samples, np.zeros(len(samples))]
    f = open("preproccessing\\indexes.txt", 'r')
    for i, sample in enumerate(samples):
        samples[i, -1] = int(get_key(int(f.readline().strip('\n')), clusters_dict))
    f.close()
    return samples


mat = []
for x in range(1000):
    mat.append([random.randint(1, 101)])
# f = open("plot_random_1000_{}_{}_{}_{}_{}.pickle", 'rb')
# file = pickle.load(f)
smp = np.array(mat)
answers = func(smp).astype(int)
# print(answers)
dictionary = {}
for answer in answers:
    if answer[0] in dictionary.keys():
        dictionary[answer[0]].append(answer[1])
    else:
        dictionary[answer[0]] = [answer[1]]
print(dictionary)
# f.close()
