# 3dbio_hackathon

Authors:

	- Alon Fridman
	
	- Daniel Brown
	
	- Edan Patt
	
	- Linoy Tsaban
	
	- Yitzchak Vaknin
	
Description:

    This program includes a pipeline for structural clustering of proteins. 
    It contains scripts for preprocessing the output of TrainedNanoNet as well as the output from CD-Hit.

Files:

	- cpp_code(directory):
	    Containes the code that creates the adjacancy matrix by correspondence size (specificly correspondece_calc.cpp).
	    The relevant file is "correspondence_calc.py". The main function revcives an argument of epsilon 
	    which is the distance treshold and calculates the corespondence between each two nanobodies(based on the pdb files in the directory 
	    "/cs/labs/dina/linoytsaban_14/pdbs1/pdbs1/") and writes the correspondence matrix to a csv file. 
	    This version of the code calculates the correspondence matrix for the 200 chosen representatives. 
	    To run it for different pdbs, change the paths in the variabls file1 and file 2 in the main function and
	    adujst the value of i and j in the loops accordingly. 
	    ** The output matrix is triangular since we calculated each match once for each pair(for efficency), in the utils 
        file there is a script(which is called during the clustering pipeline) that converts the matrix into a symmetrical one.
	    
	- data_files(directory):
	    
	    - databases (directory): contains the correspondence matrices for different runs of the 
	      cpp code on different groups of pdb files.
	    
	    - Figures (directory): an empty directory. 
	      Used in order to save outputs(plots, heatmaps etc.) of different clustring attempts. 
	      
	    - labels (directory): contains the clustring results of our pipline for different runs. 
	    
	    - name_num_dict: Index for each sequence (names from CD Hit results) in given fasta file.
	    
	    - sequences.zip: Sequences from given fasta file.

    - preproccessing (directory):
        
        - CDHits_results (directory) - Containes the CDHits results files.
        
        - HackatonUtils.py - given by course staff.
        
        - preproccess.py - parsing CD-Hit results and creating the appropriate PDB's (1000 from biggest cluster, or, 5 from each of the 200 biggest clusters - also 1000 PDB's)
        
        - indexes.txt - the real indexes of the nanobodies from the original fasta file relative to our clustering (the first row is the nanobody's index of the first PDB's we created, etc.)
        
        - utils.py - util for precessing (creating name-number dictionary and more parsing for CD-Hit results)
        
	- cluster_pipeline.py:
	    Creates the cluster pipline.

	- main_script.py:
	    Containes different type of clustering expirments. 
	    the script loads the correspondence matrix from a csv file and runs a clustering alogrithm. 
	    The different clustring methods can be used by running one of the following experiment 
	    functions that apply them - models_with_k_experiment, DBScan_experiment, mini_batch_experiment.
	    
	    To run the main function, please give as argument the path of the csv file containing the 
	    correspondence matrix to apply the pipeline. In the databases directory you will find a 
	    number of correspondence matrices created using different nanobody representatives. 
	    
	    Also, the cuurent version of the main function contains a call to the function 
	    "mini_batch_experiment" which runs the pipeline using the mini batch k means method.
	    As mentioned, there other clustering methods that can be experimented with by calling a different 
	    experiment method instead.

	- utils.py: 
	    Helper functions.