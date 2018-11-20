#!/bin/bash

python -m evaluation_acm_ccr_2019.cli evaluate_results ifip_networking_separation_lp_solutions_after_refactor_reduced.pickle ifip_networking_evaluation_solutions_rand_round_reduced.pickle ./output/ --exclude_generation_parameters "{'number_of_requests': [20]}"
