# MIT License
#
# Copyright (c) 2016-2018 Matthias Rost
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#

import pickle

import numpy as np
import alib
import vnep_approx
import pickle
import os

from alib import util
import matplotlib
from matplotlib import pyplot as plt
import logging

from vnep_approx.to_delete import ccr_2018_eval

from collections import namedtuple

logger = logging.getLogger(__name__)


def extract_comparison_data_randround_vs_separation(separation_lp_pickle,
                                                    reduced_randround_pickle,
                                                    output_pickle):
    print("Input is {} {} {}".format(separation_lp_pickle, reduced_randround_pickle, output_pickle))

    print("Reading reduced randround pickle{}..".format(reduced_randround_pickle))
    rrr_pickle = None
    with open(reduced_randround_pickle, "r") as f:
        rrr_pickle = pickle.load(f)
    print("Done: {}".format(rrr_pickle))

    result = {}

    rrr_sol_dict = rrr_pickle.algorithm_scenario_solution_dictionary['RandomizedRoundingTriumvirate']
    list_of_scenario_indices_with_runtime = []
    for scenario_index, value in rrr_sol_dict.iteritems():
        rand_round_solution = rrr_sol_dict[scenario_index][0]
        total_runtime = rand_round_solution.meta_data.time_preprocessing + \
                        rand_round_solution.meta_data.time_optimization + \
                        rand_round_solution.meta_data.time_postprocessing
        LPobjValue = rrr_sol_dict[scenario_index][0].meta_data.status.objValue
        print("Runtime of scenario with index {} is {}.".format(scenario_index, total_runtime))
        result[scenario_index] = {"RR": (scenario_index, total_runtime, LPobjValue)}

    sep_pickle = None
    with open(separation_lp_pickle, "r") as f:
        sep_pickle = pickle.load(f)
    print("Done: {}".format(sep_pickle))

    sep_sol_dict = sep_pickle.algorithm_scenario_solution_dictionary['SeparationLPDynVMP']
    list_of_scenario_indices_with_runtime = []
    for scenario_index, value in rrr_sol_dict.iteritems():
        sep_sol = sep_sol_dict[scenario_index][0]
        result[scenario_index]['SEP'] = sep_sol
        print sep_sol

    with open(output_pickle, "w") as f:
        pickle.dump(result, f)

    with open(output_pickle, "r") as f:
        foo = pickle.load(f)

def reduce_separation_lp_dynvmp_pickle(separation_lp_dynvmp_result_input_pickle_name,
                                       separation_lp_dynvmp_result_output_name = None):

    sep_lp_dynvmp_reducer = SeparationLPDynVMPReducer()
    sep_lp_dynvmp_reducer.reduce_seplp(separation_lp_dynvmp_result_input_pickle_name,
                                       separation_lp_dynvmp_result_output_name)



class SeparationLPDynVMPReducer(object):

    def __init__(self):
        pass

    def reduce_seplp(self,
                     separation_lp_dynvmp_result_input_pickle_name,
                     separation_lp_dynvmp_result_output_name=None):

        separation_lp_dynvmp_solutions_input_pickle_path = os.path.join(util.ExperimentPathHandler.INPUT_DIR,
                                                             separation_lp_dynvmp_result_input_pickle_name)

        reduced_separatio_lp_dynvmp_solutions_output_pickle_path = None
        if separation_lp_dynvmp_result_output_name is None:
            file_basename = os.path.basename(separation_lp_dynvmp_solutions_input_pickle_path).split(".")[0]
            reduced_separatio_lp_dynvmp_solutions_output_pickle_path = os.path.join(util.ExperimentPathHandler.OUTPUT_DIR,
                                                                          file_basename + "_reduced.pickle")
        else:
            reduced_separatio_lp_dynvmp_solutions_output_pickle_path = os.path.join(util.ExperimentPathHandler.OUTPUT_DIR,
                                                                          separation_lp_dynvmp_result_input_pickle_name)

        logger.info("\nWill read from ..\n\t{} \n\t\tand store reduced data into\n\t{}\n".format(
            separation_lp_dynvmp_solutions_input_pickle_path, reduced_separatio_lp_dynvmp_solutions_output_pickle_path))

        logger.info("Reading pickle file at {}".format(separation_lp_dynvmp_solutions_input_pickle_path))
        with open(separation_lp_dynvmp_solutions_input_pickle_path, "rb") as f:
            sss = pickle.load(f)

        sss.scenario_parameter_container.scenario_list = None
        sss.scenario_parameter_container.scenario_triple = None

        for alg, scenario_solution_dict in sss.algorithm_scenario_solution_dictionary.iteritems():
            logger.info(".. Reducing results of algorithm {}".format(alg))
            for sc_id, ex_param_solution_dict in scenario_solution_dict.iteritems():
                logger.info("   .. handling scenario {}".format(sc_id))
                for ex_id, solution in ex_param_solution_dict.iteritems():
                    compressed = self.reduce_single_solution(solution)
                    ex_param_solution_dict[ex_id] = compressed

        logger.info("Writing result pickle to {}".format(reduced_separatio_lp_dynvmp_solutions_output_pickle_path))
        with open(os.path.join(reduced_separatio_lp_dynvmp_solutions_output_pickle_path),
                  "w") as f:
            pickle.dump(sss, f)
        logger.info("All done.")

    def reduce_single_solution(self, solution):
        return solution
        #pass: reduce nothing at the moment.



def extract_parameter_range(scenario_parameter_space_dict, key):
    if not isinstance(scenario_parameter_space_dict, dict):
        return None
    for generator_name, value in scenario_parameter_space_dict.iteritems():
        if generator_name == key:
            return [key], value
        if isinstance(value, list):
            if len(value) != 1:
                continue
            value = value[0]
            result = extract_parameter_range(value, key)
            if result is not None:
                path, values = result
                return [generator_name, 0] + path, values
        elif isinstance(value, dict):
            result = extract_parameter_range(value, key)
            if result is not None:
                path, values = result
                return [generator_name] + path, values
    return None

def lookup_scenarios_having_specific_values(scenario_parameter_space_dict, path, value):
    current_path = path[:]
    current_dict = scenario_parameter_space_dict
    while len(current_path) > 0:
        if isinstance(current_path[0], basestring):
            current_dict = current_dict[current_path[0]]
            current_path.pop(0)
        elif current_path[0] == 0:
            current_path.pop(0)
    # print current_dict
    return current_dict[value]


def lookup_scenario_parameter_room_dicts_on_path(scenario_parameter_space_dict, path):
    current_path = path[:]
    current_dict_or_list = scenario_parameter_space_dict
    dicts_on_path = []
    while len(current_path) > 0:
        dicts_on_path.append(current_dict_or_list)
        if isinstance(current_path[0], basestring):
            current_dict_or_list = current_dict_or_list[current_path[0]]
            current_path.pop(0)
        elif isinstance(current_path[0], int):
            current_dict_or_list = current_dict_or_list[int(current_path[0])]
            current_path.pop(0)
        else:
            raise RuntimeError("Could not lookup dicts.")
    return dicts_on_path



def evaluate_baseline_and_randround(dc_seplp_dynvmp,
                                    seplp_dynvmp_algorithm_id,
                                    seplp_dynvmp_execution_config,
                                    dc_randround,
                                    randround_algorithm_id,
                                    randround_execution_config,
                                    exclude_generation_parameters=None,
                                    parameter_filter_keys=None,
                                    show_plot=False,
                                    save_plot=True,
                                    overwrite_existing_files=True,
                                    forbidden_scenario_ids=None,
                                    papermode=True,
                                    output_path="./",
                                    output_filetype="png"):
    """ Main function for evaluation, creating plots and saving them in a specific directory hierarchy.
    A large variety of plots is created. For heatmaps, a generic plotter is used while for general
    comparison plots (ECDF and scatter) an own class is used. The plots that shall be generated cannot
    be controlled at the moment but the respective plotters can be easily adjusted.

    :param dc_seplp_dynvmp: unpickled datacontainer of baseline experiments (e.g. MIP)
    :param seplp_dynvmp_algorithm_id: algorithm id of the baseline algorithm
    :param seplp_dynvmp_execution_config: execution config (numeric) of the baseline algorithm execution
    :param dc_randround: unpickled datacontainer of randomized rounding experiments
    :param randround_algorithm_id: algorithm id of the randround algorithm
    :param randround_execution_config: execution config (numeric) of the randround algorithm execution
    :param exclude_generation_parameters:   specific generation parameters that shall be excluded from the evaluation.
                                            These won't show in the plots and will also not be shown on axis labels etc.
    :param parameter_filter_keys:   name of parameters according to which the results shall be filtered
    :param show_plot:               Boolean: shall plots be shown
    :param save_plot:               Boolean: shall the plots be saved
    :param overwrite_existing_files:   shall existing files be overwritten?
    :param forbidden_scenario_ids:     list / set of scenario ids that shall not be considered in the evaluation
    :param papermode:                  nicely layouted plots (papermode) or rather additional information?
    :param output_path:                path to which the results shall be written
    :param output_filetype:            filetype supported by matplotlib to export figures
    :return: None
    """

    if forbidden_scenario_ids is None:
        forbidden_scenario_ids = set()

    if exclude_generation_parameters is not None:
        for key, values_to_exclude in exclude_generation_parameters.iteritems():
            parameter_filter_path, parameter_values = extract_parameter_range(
                dc_seplp_dynvmp.scenario_parameter_container.scenarioparameter_room, key)

            parameter_dicts_baseline = lookup_scenario_parameter_room_dicts_on_path(
                dc_seplp_dynvmp.scenario_parameter_container.scenarioparameter_room, parameter_filter_path)
            parameter_dicts_seplpdynvmp = lookup_scenario_parameter_room_dicts_on_path(
                dc_randround.scenario_parameter_container.scenarioparameter_room, parameter_filter_path)

            for value_to_exclude in values_to_exclude:

                if value_to_exclude not in parameter_values:
                    raise RuntimeError("The value {} is not contained in the list of parameter values {} for key {}".format(
                        value_to_exclude, parameter_values, key
                    ))

                #add respective scenario ids to the set of forbidden scenario ids
                forbidden_scenario_ids.update(set(lookup_scenarios_having_specific_values(
                    dc_seplp_dynvmp.scenario_parameter_container.scenario_parameter_dict, parameter_filter_path, value_to_exclude)))

            #remove the respective values from the scenario parameter room such that these are not considered when
            #constructing e.g. axes
            parameter_dicts_baseline[-1][key] = [value for value in parameter_dicts_baseline[-1][key] if
                                                 value not in values_to_exclude]
            parameter_dicts_seplpdynvmp[-1][key] = [value for value in parameter_dicts_seplpdynvmp[-1][key] if
                                                  value not in values_to_exclude]

    sep_lp_dynvmp_data_set = {scenario_index:
                                  dc_seplp_dynvmp.algorithm_scenario_solution_dictionary[seplp_dynvmp_algorithm_id][
                                      scenario_index][seplp_dynvmp_execution_config]
                              for scenario_index in
                              dc_seplp_dynvmp.algorithm_scenario_solution_dictionary[
                                  seplp_dynvmp_algorithm_id].keys() if scenario_index not in forbidden_scenario_ids}

    randround_data_set = {scenario_index:
                                  dc_randround.algorithm_scenario_solution_dictionary[randround_algorithm_id][
                                      scenario_index][randround_execution_config]
                          for scenario_index in
                          dc_randround.algorithm_scenario_solution_dictionary[
                                  randround_algorithm_id].keys() if scenario_index not in forbidden_scenario_ids}



    plot_comparison_separation_dynvmp_vs_lp(sep_lp_dynvmp_data_set=sep_lp_dynvmp_data_set,
                                            randround_data_set=randround_data_set,
                                            dc_seplp_dynvmp=dc_seplp_dynvmp,
                                            dc_randround=dc_randround,
                                            forbidden_scenario_ids=forbidden_scenario_ids)






def plot_comparison_separation_dynvmp_vs_lp(sep_lp_dynvmp_data_set,
                                            randround_data_set,
                                            dc_seplp_dynvmp,
                                            dc_randround,
                                            forbidden_scenario_ids):

    logger.info(sep_lp_dynvmp_data_set)

    scenarioparameter_room = dc_seplp_dynvmp.scenario_parameter_container.scenarioparameter_room

    scenario_parameter_dict = dc_seplp_dynvmp.scenario_parameter_container.scenario_parameter_dict

    filter_path_number_of_requests, list_number_of_requests = extract_parameter_range(scenarioparameter_room,
                                                                                      "number_of_requests")

    logger.info(list_number_of_requests)


    fix, ax = plt.subplots(figsize=(5, 3.5))

    colors = ['g', 'b','r','k']
    linestyles = ['-', ':']

    with_td = matplotlib.lines.Line2D([], [], color='#333333', linestyle=linestyles[0], label=r"incl. $\mathcal{T}_r$ comp.", linewidth=2)
    wo_td = matplotlib.lines.Line2D([], [], color='#333333', linestyle=linestyles[1], label=r"excl. $\mathcal{T}_r$ comp.", linewidth=2.75)


    second_legend_handlers = []

    max_observed_value = 0

    for request_number_index, number_of_requests in enumerate(list_number_of_requests):
        #do the code!
        scenario_ids_of_requests = lookup_scenarios_having_specific_values(scenario_parameter_dict, filter_path_number_of_requests, number_of_requests)

        speedups_real = []
        speedups_wotd = []  # without tree decomposition
        relative_speedup_sep_lp_wo_td = []

        for scenario_id in scenario_ids_of_requests:
            seplp_with_decomposition = sep_lp_dynvmp_data_set[scenario_id].time_preprocessing + sep_lp_dynvmp_data_set[scenario_id].time_optimization
            seplp_without_decomposition = seplp_with_decomposition - sum(sep_lp_dynvmp_data_set[scenario_id].tree_decomp_runtimes)

            randround_lp_runtime = randround_data_set[scenario_id].meta_data.time_preprocessing + \
                                   randround_data_set[scenario_id].meta_data.time_optimization + \
                                   randround_data_set[scenario_id].meta_data.time_postprocessing

            relative_speedup_sep_lp_wo_td.append(seplp_with_decomposition / seplp_without_decomposition)

            speedups_real.append(randround_lp_runtime / seplp_with_decomposition)
            speedups_wotd.append(randround_lp_runtime / seplp_without_decomposition)

        speedup_real = sorted(speedups_real)
        speedup_wotd = sorted(speedups_wotd)


        logger.info("Relative when excluding tree decomposition computation {} requests:\n"
                    "mean: {}\n".format(number_of_requests,
                                        np.mean(relative_speedup_sep_lp_wo_td)))


        logger.info("Relative speedup compared to cactus LP for {} requests:\n"
                    "with tree decomposition (mean): {}\n"
                    "without tree decomposition (mean): {}".format(number_of_requests,
                                                                   np.mean(speedups_real),
                                                                   np.mean(speedups_wotd)))



        max_observed_value = np.maximum(max_observed_value, speedup_real[-1])
        yvals = np.arange(1, len(speedup_real) + 1) / float(len(speedup_real))
        yvals = np.insert(yvals, 0, 0.0, axis=0)
        yvals = np.append(yvals, [1.0])
        speedup_real.append(max_observed_value)
        speedup_real.insert(0, 0.5)
        ax.semilogx(speedup_real, yvals, color=colors[request_number_index], linestyle=linestyles[0],
                    linewidth=2, alpha=0.75)

        max_observed_value = np.maximum(max_observed_value, speedup_wotd[-1])
        yvals = np.arange(1, len(speedup_wotd) + 1) / float(len(speedup_wotd))
        yvals = np.insert(yvals, 0, 0.0, axis=0)
        yvals = np.append(yvals, [1.0])
        speedup_wotd.append(max_observed_value)
        speedup_wotd.insert(0, 0.5)
        ax.semilogx(speedup_wotd, yvals, color=colors[request_number_index], linestyle=linestyles[1],
                    linewidth=2.75, alpha=0.75)

        second_legend_handlers.append(matplotlib.lines.Line2D([], [], color=colors[request_number_index], alpha=0.75, linestyle="-",
                                                              label=("{}".format(number_of_requests)).ljust(3), linewidth=2))

    first_legend = plt.legend(handles=[with_td, wo_td], loc=4, fontsize=14, title="", handletextpad=.35,
                              borderaxespad=0.1, borderpad=0.2, handlelength=0.6666)
    first_legend.get_frame().set_alpha(1.0)
    first_legend.get_frame().set_facecolor("#FFFFFF")
    plt.setp(first_legend.get_title(), fontsize=14)
    plt.gca().add_artist(first_legend)
    # ax.tick_params(labelright=True)

    # print second_legend_handlers

    second_legend = plt.legend(handles=second_legend_handlers, loc=2, fontsize=14, title="#requests", handletextpad=.35,
                               borderaxespad=0.175, borderpad=0.2, handlelength=1.75)
    #plt.gca().add_artist(second_legend)
    plt.setp(second_legend.get_title(), fontsize=14)

    second_legend.get_frame().set_alpha(1.0)
    second_legend.get_frame().set_facecolor("#FFFFFF")

    # first_legend = plt.legend(title="Bound($\mathrm{MIP}_{\mathrm{MCF}})$", handles=root_legend_handlers, loc=(0.225,0.0125), fontsize=14, handletextpad=0.35, borderaxespad=0.175, borderpad=0.2)
    # plt.setp(first_legend.get_title(), fontsize='15')
    # plt.gca().add_artist(first_legend)
    # plt.setp("TITLE", fontsize='15')

    ax.set_title("LP Runtime Comparison", fontsize=17)
    ax.set_xlabel(r"speedup: time($\mathsf{LP}_{\mathsf{Cactus}}$) / time($\mathsf{LP}_{\mathsf{DynVMP}}$)",
                  fontsize=16)
    ax.set_ylabel("ECDF", fontsize=16)

    ax.set_xlim(0.4, max_observed_value * 1.15)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(15.5)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(15.5)

    ax.set_xticks([0.5, 1, 5, 20, 60, ], minor=False)
    ax.set_xticks([2, 3, 4, 10, 30, 40], minor=True)

    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], minor=False)
    ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)

    # ax.set_yticks([x*0.1 for x in range(1,10)], minor=True)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xticklabels([], minor=True)

    ax.grid(True, which="both", linestyle=":", color='k', alpha=0.7, linewidth=0.33)
    plt.tight_layout()
    plt.savefig("ecdf_speedup_cactus_lp_vs_separation_dynvmp.pdf")
