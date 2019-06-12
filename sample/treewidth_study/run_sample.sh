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

#compute treewidths of random graphs according to parameters of the yml file
python -m evaluation_acm_ccr_2019.cli execute_treewidth_computation_experiment --threads 2 sample_treewidth_computation.yml --timeout 5400 --remove_intermediate_solutions
move_logs_and_output log_treewidth_computation


python -m evaluation_acm_ccr_2019.cli treewidth_plot_computation_results sample_treewidth_computation.yml input/sample_treewidth_computation_results_aggregated_results.pickle ./plots/ --output_filetype pdf
move_logs_and_output log_treewidth_plot_pdf

python -m evaluation_acm_ccr_2019.cli treewidth_plot_computation_results sample_treewidth_computation.yml input/sample_treewidth_computation_results_aggregated_results.pickle ./plots/ --output_filetype png
move_logs_and_output log_treewidth_plot_png

