#!/bin/bash

#either the $PYTHONPATH must be set to include alib, vnep_approx, evaluation_acm_ccr_2019 or 
#you execute this from within the virtual environment in which these packages were installed

set -e			#to exit upon failure
shopt -s nullglob	#for looping over files correctly, even if there are none

mkdir -p log/ && mkdir -p input && mkdir -p output

function move_logs_and_output() {
	mkdir -p $1
	echo "make sure to check the logs for errors and that the generated output is correct"
	for file in output/*; 	do mv $file input/; done
	for file in log/*; 	do mv $file $1/ ; done
}

#STUDY TREEWIDTH OF RANDOM GRAPHS
#The following command generates approx. 4.47M many random graphs and computes the treewidth
#using Tamaki's algorithm.
#Please note that to run this command, you should increase the thread count and have enough memory (probably more than 32 GB)

python -m evaluation_acm_ccr_2019.cli execute_treewidth_computation_experiment treewidth_computation_acm_ccr_2019.yml --threads 2 --timeout 5400 --remove_intermediate_solutions
move_logs_and_output log_treewidth_computation

#EXTRACT UNDIRECTED GRAPH STORAGE
#The following takes all the graphs of treewidth 2 to 4 and stores the respective graphs
#in an extra graph storage container. This graph storage container will then be used 
#for the generation of scenarios having a specific treewidth.
python -m evaluation_acm_ccr_2019.cli create_undirected_graph_storage_from_treewidth_experiments input/treewidth_computation_acm_ccr_2019_results_aggregated_results.pickle input/treewidth_computation_acm_ccr_2019_undirected_graph_storage_treewidth_2_to_4.pickle 2 4
move_logs_and_output log_undirected_graph_storage

#PLOTTING RESULTS
#Please refer to the create_plots.sh for instructions.

