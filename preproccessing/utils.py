import os


####################################################################################################
# Code to replace the name of the structures in the cluster files with their corresponding number  #
####################################################################################################
def create_num_to_name_dict(dict_path, num_to_name=True):
    """
    creates a dictionary by loading a txt file that maps indexes to the actual number of a structure
    (in some runs of the alignment code we had to renumber the structures to allow for an efficient
    implementation of the algorithm

    """
    file_dict = open(dict_path, 'r')
    name_to_num = {}
    for line in file_dict.readlines():
        num, name = line.strip().split(",")
        if num_to_name:
            name_to_num[name] = num
        else:
            name_to_num[num] = name
    file_dict.close()
    return name_to_num


def replace_names_cluster_files(clusters_path, dict_path):
    """
    replaces the names of the nanobodies in the cluster files to their number

    """
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
    """
    renames pdb files in the given directory
    so that they are in chronological order. Creates a txt file with a mapping of
    the original number of the structure and its new index
    """
    print(os.listdir(directory_path))
    dict_file = open("struct_to_index_second_1000.txt", 'w')
    results = "/cs/usr/linoytsaban_14/PycharmProjects/3dbio_hackathon/pdbs_5_from_200_renamed/"
    for i,file in enumerate(os.listdir(directory_path)):
        print(i)
        struct_num = file.split(".pdb")[0]
        dict_file.write(str(i+1)+","+struct_num+"\n")
        os.rename(directory_path+file, results+str(i+1)+".pdb")
    dict_file.close()

