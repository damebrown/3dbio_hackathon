import pandas as pd
import numpy as np
from preproccessing.utils import create_num_to_name_dict


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
triangle_to_symmetric_matrix("correspond_1000_first_batch_08.csv",
                             "preproccessing/struct_to_index_first_1000.txt")
