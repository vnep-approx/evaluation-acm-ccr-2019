#!/bin/bash

#either the $PYTHONPATH must be set to include alib, vnep_approx, evaluation_acm_ccr_2019 or 
#you execute this from within the virtual environment in which these packages were installed

set -e			#to exit upon failure
shopt -s nullglob	#for looping over files correctly, even if there are none

function move_logs_and_output() {
	mkdir -p $1
	echo "make sure to check the logs for errors and that the generated output is correct"
	for file in output/*; 	do mv $file input/; done
	for file in log/*; 	do mv $file $1/ ; done
}

mkdir -p log/ && mkdir -p input && mkdir -p output

#The following shall just give an idea of how the results were obtained.
#Running this script, will either take very long (few processes) or will have very 
#large RAM demands (we executed all these experiments on a server with 256 GB of RAM). 

#GENERATE SCENARIOS
python -m vnep_approx.cli generate_scenarios treewidth_scenarios_acm_ccr_2019.pickle scenario_generation_treewidth_acm_ccr_2019.yml --threads 4
move_logs_and_output log_scenario_generation 

#RUN RANDOMIZED ROUNDING USING SEPARATION LP WITH DYNVMP
python -m vnep_approx.cli start_experiment randround_seplp_dynvmp_execution_acm_ccr_2019.yml.yml 0 10000 --concurrent 4 --overwrite_existing_intermediate_solutions --keep_temporary_scenarios --remove_intermediate_solutions
move_logs_and_output log_seplp_dynvmp_execution

#RUN VINE ALGORITHMS
python -m vnep_approx.cli start_experiment vine_execution_acm_ccr_2019.yml 0 10000 --concurrent 4 --overwrite_existing_intermediate_solutions --remove_temporary_scenarios --remove_intermediate_solutions
move_logs_and_output log_vine_execution

#EXTRACT DATA TO BE PLOTTED
python -m evaluation_acm_ccr_2019.cli reduce_to_plotdata_rr_seplp_optdynvmp treewidth_scenarios_acm_ccr_2019.pickle
move_logs_and_output log_reduction_rr_seplp_dynvmp

python -m evaluation_acm_ccr_2019.cli reduce_to_plotdata_vine treewidth_scenarios_acm_ccr_2019_vine_results.pickle
move_logs_and_output log_reduction_vine

#PLOT RESULTS
#Please see create_plots.sh for instructions.
mkdir -p ./plots

python -m evaluation_acm_ccr_2019.cli evaluate_separation_randround_vs_vine sample_scenarios_results_seplp_dynvmp_reduced.pickle sample_scenarios_ViNE_results_reduced.pickle ./plots/ --request_sets "[[20,30], [40,50]]" --output_filetype pdf --papermode
move_logs_and_output log_plot_pdf

python -m evaluation_acm_ccr_2019.cli evaluate_separation_randround_vs_vine sample_scenarios_results_seplp_dynvmp_reduced.pickle sample_scenarios_ViNE_results_reduced.pickle ./plots/ --request_sets "[[20,30], [40,50]]" --output_filetype png --non-papermode
move_logs_and_output log_plot_png

rm gurobi.log


