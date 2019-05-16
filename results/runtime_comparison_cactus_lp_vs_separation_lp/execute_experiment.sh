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


#GENERATE SCENARIOS
#for the evaluation of the cactus LP and the separation based LP using DynVMP, the same instances as for the
#evaluation presented at IFIP Networking 2018 were used. The procedure to generate (similar) instances can be 
#found at: https://github.com/vnep-approx/evaluation-ifip-networking-2018


#RUN RANDOMIZED ROUNDING USING CACTUS LP FORMULATION
#Here, the results of the evaluation of IFIP Networking were re-used and included in this repository.
#See https://github.com/vnep-approx/evaluation-ifip-networking-2018 for more information

#RUN RANDOMIZED ROUNDING USING SEPLP DYNVMP LP
python -m vnep_approx.cli start_experiment acm_ccr_2019_rr_seplp_dynvmp_execution_ifip_networking_2018_scenarios.yml 0 10000 --concurrent 2 --overwrite_existing_intermediate_solutions --remove_temporary_scenarios --remove_intermediate_solutions
move_logs_and_output log_seplp_dynvmp_execution

#EXTRACT DATA TO BE PLOTTED
python -m evaluation_ifip_networking_2018.cli reduce_to_plotdata_randround_pickle sample_scenarios_results_cactus.pickle
move_logs_and_output log_cactus_reduction_to_plotdata

python -m evaluation_acm_ccr_2019.cli reduce_to_plotdata_rand_round_pickle sample_scenarios_results_seplp_dynvmp.pickle
move_logs_and_output log_seplp_dynvmp_reduction_to_plotdata



