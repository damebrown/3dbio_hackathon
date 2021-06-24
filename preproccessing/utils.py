import os


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
    new_file = open(clusters_path + "output0.9_replaced.clstr", 'w')
    orig_file = open(clusters_path + "output0.9.clstr", 'r')
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
    return "output0.9_replaced.clstr"


def rename_structures(directory_path):
    pass