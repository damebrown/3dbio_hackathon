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
    new_file = open("replaced_"+clusters_path, 'w')
    orig_file = open(clusters_path, 'r')
    name_to_num = create_num_to_name_dict(dict_path)
    for line in orig_file.readlines():
        if '...' in line:
            name = line.split("...")[0].split(">")[1]
            new_line = line.replace(name, name_to_num[name])
            new_file.write(new_line)
        else:
            new_file.write(line)
    orig_file.close()
    new_file.close()



# create_num_to_name_dict("name_num_dict")
# replace_names_cluster_files("output_0.8.clstr","name_num_dict")
