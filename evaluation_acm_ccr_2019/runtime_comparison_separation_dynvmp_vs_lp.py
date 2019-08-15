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


import numpy as np
import alib
import vnep_approx
import os

try:
    import cPickle as pickle
except ImportError:
    import pickle

import matplotlib
from matplotlib import pyplot as plt
import logging


logger = logging.getLogger(__name__)


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
                                    forbidden_scenario_ids=None,
                                    output_path="./",
                                    output_filetype="png",
                                    request_sets=[[40,60],[80,100]]):
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
    :param forbidden_scenario_ids:     list / set of scenario ids that shall not be considered in the evaluation
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
                                            request_sets=request_sets,
                                            output_path=output_path,
                                            output_filetype=output_filetype)






def plot_comparison_separation_dynvmp_vs_lp(sep_lp_dynvmp_data_set,
                                            randround_data_set,
                                            dc_seplp_dynvmp,
                                            request_sets,
                                            output_path,
                                            output_filetype):

    logger.info(sep_lp_dynvmp_data_set)

    scenarioparameter_room = dc_seplp_dynvmp.scenario_parameter_container.scenarioparameter_room

    scenario_parameter_dict = dc_seplp_dynvmp.scenario_parameter_container.scenario_parameter_dict

    filter_path_number_of_requests, list_number_of_requests = extract_parameter_range(scenarioparameter_room,
                                                                                      "number_of_requests")

    logger.info(list_number_of_requests)


    fix, ax = plt.subplots(figsize=(5, 3.5))

    def get_color(value):
        return plt.cm.inferno(value)

    colors = [get_color(0.5),get_color(0.0), get_color(0.75), get_color(0.25)] #get_color(0.7),
    #colors = [get_color(0.75), get_color(0.55), get_color(0.35), get_color(0.0)]
    linestyles = ['-', ':']

    with_td = matplotlib.lines.Line2D([], [], color='#333333', linestyle=linestyles[0], label=r"incl.  $\mathcal{T}_r$ comp.", linewidth=2)
    wo_td = matplotlib.lines.Line2D([], [], color='#333333', linestyle=linestyles[1], label=r"excl. $\mathcal{T}_r$ comp.", linewidth=2.75)


    second_legend_handlers = []

    max_observed_value = 0

    for request_number_index, number_of_requests_ in enumerate(request_sets):
        scenario_ids_to_consider = set()
        for number_of_requests in number_of_requests_:
            #do the code!
            scenario_ids_of_requests = lookup_scenarios_having_specific_values(scenario_parameter_dict, filter_path_number_of_requests, number_of_requests)
            scenario_ids_to_consider = scenario_ids_to_consider.union(scenario_ids_of_requests)

        speedups_real = []
        speedups_wotd = []  # without tree decomposition
        relative_speedup_sep_lp_wo_td = []

        for scenario_id in scenario_ids_to_consider:
            seplp_with_decomposition = sep_lp_dynvmp_data_set[scenario_id].lp_time_preprocess + sep_lp_dynvmp_data_set[scenario_id].lp_time_optimization
            seplp_without_decomposition = seplp_with_decomposition - (sep_lp_dynvmp_data_set[scenario_id].lp_time_tree_decomposition.mean * sep_lp_dynvmp_data_set[scenario_id].lp_time_tree_decomposition.value_count)

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
                    linewidth=2.75, alpha=1)

        max_observed_value = np.maximum(max_observed_value, speedup_wotd[-1])
        yvals = np.arange(1, len(speedup_wotd) + 1) / float(len(speedup_wotd))
        yvals = np.insert(yvals, 0, 0.0, axis=0)
        yvals = np.append(yvals, [1.0])
        speedup_wotd.append(max_observed_value)
        speedup_wotd.insert(0, 0.5)
        ax.semilogx(speedup_wotd, yvals, color=colors[request_number_index], linestyle=linestyles[1],
                    linewidth=2.75, alpha=1)

        if len(number_of_requests_) == 2:
            second_legend_handlers.append(matplotlib.lines.Line2D([], [], color=colors[request_number_index], alpha=1, linestyle="-",
                                                                  label=("{} & {}".format(number_of_requests_[0], number_of_requests_[1])).ljust(3), linewidth=2.5))
        else:
            second_legend_handlers.append(
                matplotlib.lines.Line2D([], [], color=colors[request_number_index], alpha=1, linestyle="-",
                                        label=("{}".format(number_of_requests_[0])).ljust(
                                            3), linewidth=2.5))

    first_legend = plt.legend(handles=[with_td, wo_td], loc=4, fontsize=14, title="", handletextpad=.35,
                              borderaxespad=0.1, borderpad=0.2, handlelength=1)
    first_legend.get_frame().set_alpha(1.0)
    first_legend.get_frame().set_facecolor("#FFFFFF")
    plt.setp(first_legend.get_title(), fontsize=15)
    plt.gca().add_artist(first_legend)
    # ax.tick_params(labelright=True)

    # print second_legend_handlers

    second_legend = plt.legend(handles=second_legend_handlers, loc=2, fontsize=14, title="#requests", handletextpad=.35,
                               borderaxespad=0.175, borderpad=0.2, handlelength=2)
    #plt.gca().add_artist(second_legend)
    plt.setp(second_legend.get_title(), fontsize=15)

    second_legend.get_frame().set_alpha(1.0)
    second_legend.get_frame().set_facecolor("#FFFFFF")

    # first_legend = plt.legend(title="Bound($\mathrm{MIP}_{\mathrm{MCF}})$", handles=root_legend_handlers, loc=(0.225,0.0125), fontsize=14, handletextpad=0.35, borderaxespad=0.175, borderpad=0.2)
    # plt.setp(first_legend.get_title(), fontsize='15')
    # plt.gca().add_artist(first_legend)
    # plt.setp("TITLE", fontsize='15')

    ax.set_title("Cactus LP Runtime Comparison", fontsize=17)
    ax.set_xlabel(r"Speedup: time($\mathsf{LP}_{\mathsf{Cactus}}$) / time($\mathsf{LP}_{\mathsf{DynVMP}}$)",
                  fontsize=16)
    ax.set_ylabel("ECDF", fontsize=16)

    ax.set_xlim(0.4, max_observed_value * 1.15)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(14)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(14)

    ax.set_xticks([0.5, 1, 5, 20, 60, ], minor=False)
    ax.set_xticks([2, 3, 4, 10, 30, 40], minor=True)

    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], minor=False)
    ax.set_yticks([0.1, 0.3, 0.5, 0.7, 0.9], minor=True)

    # ax.set_yticks([x*0.1 for x in range(1,10)], minor=True)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.set_xticklabels([], minor=True)

    ax.grid(True, which="both", linestyle=":", color='k', alpha=0.7, linewidth=0.33)
    plt.tight_layout()
    file_to_write = os.path.join(output_path, "ecdf_speedup_cactus_lp_vs_separation_dynvmp." + output_filetype)
    plt.savefig(file_to_write)


def plot_comparison_separation_dynvmp_vs_lp_orig(sep_lp_dynvmp_data_set,
                                            randround_data_set,
                                            dc_seplp_dynvmp):

    logger.info(sep_lp_dynvmp_data_set)

    scenarioparameter_room = dc_seplp_dynvmp.scenario_parameter_container.scenarioparameter_room

    scenario_parameter_dict = dc_seplp_dynvmp.scenario_parameter_container.scenario_parameter_dict

    filter_path_number_of_requests, list_number_of_requests = extract_parameter_range(scenarioparameter_room,
                                                                                      "number_of_requests")

    logger.info(list_number_of_requests)


    fix, ax = plt.subplots(figsize=(5, 3.5))

    def get_color(value):
        return plt.cm.inferno(value)

    colors = [get_color(0.75), get_color(0.55),get_color(0.35),get_color(0.0)]
    linestyles = ['-', ':']

    with_td = matplotlib.lines.Line2D([], [], color='#333333', linestyle=linestyles[0], label=r"incl.  $\mathcal{T}_r$ comp.", linewidth=2)
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
            seplp_with_decomposition = sep_lp_dynvmp_data_set[scenario_id].lp_time_preprocess + sep_lp_dynvmp_data_set[scenario_id].lp_time_optimization
            seplp_without_decomposition = seplp_with_decomposition - (sep_lp_dynvmp_data_set[scenario_id].lp_time_tree_decomposition.mean * sep_lp_dynvmp_data_set[scenario_id].lp_time_tree_decomposition.value_count)

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
                    linewidth=2.75, alpha=1)

        max_observed_value = np.maximum(max_observed_value, speedup_wotd[-1])
        yvals = np.arange(1, len(speedup_wotd) + 1) / float(len(speedup_wotd))
        yvals = np.insert(yvals, 0, 0.0, axis=0)
        yvals = np.append(yvals, [1.0])
        speedup_wotd.append(max_observed_value)
        speedup_wotd.insert(0, 0.5)
        ax.semilogx(speedup_wotd, yvals, color=colors[request_number_index], linestyle=linestyles[1],
                    linewidth=2.75, alpha=1)

        second_legend_handlers.append(matplotlib.lines.Line2D([], [], color=colors[request_number_index], alpha=1, linestyle="-",
                                                              label=("{}".format(number_of_requests)).ljust(3), linewidth=2.5))

    first_legend = plt.legend(handles=[with_td, wo_td], loc=4, fontsize=11, title="", handletextpad=.35,
                              borderaxespad=0.1, borderpad=0.2, handlelength=2.5)
    first_legend.get_frame().set_alpha(1.0)
    first_legend.get_frame().set_facecolor("#FFFFFF")
    plt.setp(first_legend.get_title(), fontsize=12)
    plt.gca().add_artist(first_legend)
    # ax.tick_params(labelright=True)

    # print second_legend_handlers

    second_legend = plt.legend(handles=second_legend_handlers, loc=2, fontsize=11, title="#requests", handletextpad=.35,
                               borderaxespad=0.175, borderpad=0.2, handlelength=2)
    #plt.gca().add_artist(second_legend)
    plt.setp(second_legend.get_title(), fontsize=12)

    second_legend.get_frame().set_alpha(1.0)
    second_legend.get_frame().set_facecolor("#FFFFFF")

    # first_legend = plt.legend(title="Bound($\mathrm{MIP}_{\mathrm{MCF}})$", handles=root_legend_handlers, loc=(0.225,0.0125), fontsize=14, handletextpad=0.35, borderaxespad=0.175, borderpad=0.2)
    # plt.setp(first_legend.get_title(), fontsize='15')
    # plt.gca().add_artist(first_legend)
    # plt.setp("TITLE", fontsize='15')

    ax.set_title("Cactus LP Runtime Comparison", fontsize=17)
    ax.set_xlabel(r"Speedup: Time($\mathsf{LP}_{\mathsf{Cactus}}$) / Time($\mathsf{LP}_{\mathsf{DynVMP}}$)",
                  fontsize=16)
    ax.set_ylabel("ECDF", fontsize=16)

    ax.set_xlim(0.4, max_observed_value * 1.15)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(13)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(13)

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
