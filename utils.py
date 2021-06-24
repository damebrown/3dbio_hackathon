import pandas as pd
import numpy as np


####################################################################################################
# Code to replace the name of the structures in the cluster files with their corresponding number  #
####################################################################################################

def triangle_to_symmetric_matrix(matrix_path):
    df = pd.read_csv(matrix_path, sep=',', header=None)
    orig_matrix = df.values[:, :df.values.shape[0]]
    # orig_matrix = np.array([[123,107,113,103],[140,110,108],[136,106],[115]])
    new_matrix = np.zeros((orig_matrix.shape[0], orig_matrix.shape[0]))
    for i in range(orig_matrix.shape[0]):
        left_side = [orig_matrix[k][j] for k, j in zip(range(0, i, 1), range(i, 0, -1))]
        right_side = [orig_matrix[i][j] for j in range(0, orig_matrix.shape[0]) if
                      not pd.isna(orig_matrix[i][j])]
        new_matrix[i] = np.array(left_side + right_side)

    # create_num_to_name_dict("name_num_dict")


# replace_names_cluster_files("output_0.8.clstr", "name_num_dict")
triangle_to_symmetric_matrix("")
