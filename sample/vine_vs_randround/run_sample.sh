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
python -m evaluation_acm_ccr_2019.cli execute_treewidth_computation_experiment --threads 4 sample_treewidth_computation.yml --timeout 5400 --remove_intermediate_solutions
move_logs_and_output log_treewidth_computation

#extract undirected graph storage
python -m evaluation_acm_ccr_2019.cli create_undirected_graph_storage_from_treewidth_experiments input/sample_treewidth_computation_results_aggregated_results.pickle input/sample_undirected_graph_storage.pickle 2 3 
move_logs_and_output log_undirected_graph_storage


#generate scenarios
python -m vnep_approx.cli generate_scenarios sample_scenarios.pickle sample_scenario_generation.yml --threads 4
move_logs_and_output log_scenario_generation 


#run randomized rounding algorithm using separation lp with dynvmp
python -m vnep_approx.cli start_experiment sample_rr_seplp_dynvmp_execution.yml 0 10000 --concurrent 4 --overwrite_existing_intermediate_solutions --keep_temporary_scenarios --remove_intermediate_solutions
move_logs_and_output log_seplp_dynvmp_execution

#run vine algorithm
python -m vnep_approx.cli start_experiment sample_vine_execution.yml 0 10000 --concurrent 4 --overwrite_existing_intermediate_solutions --remove_temporary_scenarios --remove_intermediate_solutions
move_logs_and_output log_vine_execution


#extract data to be plotted
move_logs_and_output log_reduction_rr_seplp_dynvmp
python -m evaluation_acm_ccr_2019.cli reduce_to_plotdata_rr_seplp_optdynvmp sample_scenarios_results_seplp_dynvmp.pickle

python -m evaluation_acm_ccr_2019.cli reduce_to_plotdata_vine sample_scenarios_ViNE_results.pickle
move_logs_and_output log_reduction_vine

#generate plots in folder ./plots
mkdir -p ./plots

python -m evaluation_acm_ccr_2019.cli evaluate_separation_randround_vs_vine sample_scenarios_results_seplp_dynvmp_reduced.pickle sample_scenarios_ViNE_results_reduced.pickle ./plots/ --request_sets "[[20,30], [40,50]]" --output_filetype pdf --papermode
move_logs_and_output log_plot_pdf

python -m evaluation_acm_ccr_2019.cli evaluate_separation_randround_vs_vine sample_scenarios_results_seplp_dynvmp_reduced.pickle sample_scenarios_ViNE_results_reduced.pickle ./plots/ --request_sets "[[20,30], [40,50]]" --output_filetype png --non-papermode
move_logs_and_output log_plot_png

rm gurobi.log


