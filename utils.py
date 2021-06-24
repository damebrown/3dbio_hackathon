import pandas as pd
import numpy as np

####################################################################################################
# Code to replace the name of the structures in the cluster files with their corresponding number  #
####################################################################################################
def create_num_to_name_dict(dict_path):
    file_dict = open(dict_path, 'r')
    name_to_num = {}
    for line in file_dict.readlines():
        num, name = line.strip().split(",")
        name_to_num[name] = num
    file_dict.close()
    return name_to_num


def replace_names_cluster_files(clusters_path, dict_path):
    new_file = open("replaced_" + clusters_path, 'w')
    orig_file = open(clusters_path, 'r')
    name_to_num = create_num_to_name_dict(dict_path)
    for line in orig_file.readlines():
        if '...' in line:
            name = line.split("...")[0].split(">")[1]
            new_line = name_to_num[name] + "\n"
            new_file.write(new_line)
        else:
            new_file.write(line)
    orig_file.close()
    new_file.close()


def triangle_to_symmetric_matrix(matrix_path):
    df = pd.read_csv(matrix_path, sep=',', header=None)
    orig_matrix = df.values()
    # orig_matrix = np.array([[123,107,113,103],[140,110,108],[136,106],[115]])
    new_matrix = np.zeros((orig_matrix.shape[0],orig_matrix.shape[0]))
    for i in range(orig_matrix.shape[0]):
        left_side = [orig_matrix[k][j] for k,j in zip(range(0,i,1),range(i,0,-1))]
        new_matrix[i] = np.array(left_side + orig_matrix[i])
    print(new_matrix)
    return new_matrix




# create_num_to_name_dict("name_num_dict")
# replace_names_cluster_files("output_0.8.clstr", "name_num_dict")
triangle_to_symmetric_matrix("")
