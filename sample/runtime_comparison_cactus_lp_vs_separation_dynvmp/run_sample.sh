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


#generate scenarios
python -m vnep_approx.cli generate_scenarios sample_scenarios.pickle sample_scenarios.yml
move_logs_and_output log_scenario_generation

#run randomized rounding algorithm using cactus formulation
python -m vnep_approx.cli start_experiment sample_randround_execution.yml 0 10000 --concurrent 2 --overwrite_existing_intermediate_solutions --remove_intermediate_solutions 
move_logs_and_output log_cactus_execution

#run randomized rounding algorithm using separation lp with dynvmp
python -m vnep_approx.cli start_experiment sample_rr_seplp_optdynvmp.yml 0 10000 --concurrent 2 --overwrite_existing_intermediate_solutions --remove_temporary_scenarios --remove_intermediate_solutions
move_logs_and_output log_seplp_dynvmp_execution

#extract data to be plotted
python -m evaluation_ifip_networking_2018.cli reduce_to_plotdata_randround_pickle sample_scenarios_results_cactus.pickle
move_logs_and_output log_cactus_reduction_to_plotdata

python -m evaluation_acm_ccr_2019.cli reduce_to_plotdata_rand_round_pickle sample_scenarios_results_seplp_dynvmp.pickle
move_logs_and_output log_seplp_dynvmp_reduction_to_plotdata

#generate plots in folder ./plots
mkdir -p ./plots

python -m evaluation_acm_ccr_2019.cli evaluate_separation_vs_cactus_lp sample_scenarios_results_seplp_dynvmp_reduced.pickle sample_scenarios_results_cactus_reduced.pickle  ./plots/ --output_filetype png --request_sets "[[20,30],[40,50]]"
move_logs_and_output log_plot_png

python -m evaluation_acm_ccr_2019.cli evaluate_separation_vs_cactus_lp sample_scenarios_results_seplp_dynvmp_reduced.pickle sample_scenarios_results_cactus_reduced.pickle  ./plots/ --output_filetype pdf --request_sets "[[20,30],[40,50]]"
move_logs_and_output log_plot_pdf

rm gurobi.log


