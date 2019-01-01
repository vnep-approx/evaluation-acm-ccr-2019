# MIT License
#
# Copyright (c) 2016-2018 Matthias Rost, Elias Doehne, Alexander Elvers
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

"""This is the evaluation and plotting module.

This module handles all plotting related evaluation.
"""
import itertools
import os
import pickle
import sys
from collections import namedtuple
from itertools import combinations, product
from time import gmtime, strftime

import matplotlib
import matplotlib.patheffects as PathEffects
import yaml
from matplotlib import font_manager
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from alib import solutions, util
from vnep_approx import vine, treewidth_model
from evaluation_acm_ccr_2019 import plot_data

REQUIRED_FOR_PICKLE = solutions  # this prevents pycharm from removing this import, which is required for unpickling solutions

OUTPUT_PATH = None
OUTPUT_FILETYPE = "png"

logger = util.get_logger(__name__, make_file=False, propagate=True)


class HeatmapPlotType(object):
    ViNE = 0  # a plot only for OfflineViNEResult data
    RandRoundSepLPDynVMP = 1  # a plot only for RandRoundSepLPOptDynVMPCollectionResult data
    SeparationLP = 2  # a plot only for SeparationLPSolution data
    VALUE_RANGE = [0, 1, 2]


"""
Collection of heatmap plot specifications. Each specification corresponds to a specific plot and describes all essential
information:
- name:                 the title of the plot
- filename:             prefix of the files to be generated
- plot_type:            A HeatmapPlotType describing which data is required as input.             
- vmin and vmax:        minimum and maximum value for the heatmap
- cmap:                 the colormap that is to be used for the heatmap
- lookup_function:      which of the values shall be plotted. the input is a tuple consisting of a baseline and a randomized rounding
                        solution. The function must return a numeric value or NaN
- metric filter:        after having applied the lookup_function (returning a numeric value or NaN) the metric_filter is 
                        applied (if given) and values not matching this function are discarded.
- rounding_function:    the function that is applied for displaying the mean values in the heatmap plots
- colorbar_ticks:       the tick values (numeric) for the heatmap plot   

"""

heatmap_specification_runtime = dict(
    name="ViNE: Mean Runtime [s]",
    filename="mean_runtime",
    vmin=0,
    vmax=11,
    colorbar_ticks=[x for x in range(0, 11, 20)],
    cmap="Greys",
    plot_type=HeatmapPlotType.ViNE,
    lookup_function=lambda vine_result_dict, vine_settings: np.average([
        vine_result.total_runtime
        for vine_result in vine_result_dict[vine_settings]
    ]),
    rounding_function=lambda x: int(round(x)),
)

heatmap_specification_embedding_ratio = dict(
    name="ViNE: Acceptance Ratio [%]",
    filename="embedding_ratio",
    vmin=0.0,
    vmax=100.0,
    colorbar_ticks=[x for x in range(0, 101, 20)],
    cmap="Greens",
    plot_type=HeatmapPlotType.ViNE,
    lookup_function=lambda vine_result_dict, vine_settings: 100 * np.average([
        vine_result.embedding_ratio
        for vine_result in vine_result_dict[vine_settings]
    ]),
)

heatmap_specification_average_node_load = dict(
    name="ViNE: Avg. Node Load [%]",
    filename="avg_node_load",
    vmin=0.0,
    vmax=100,
    colorbar_ticks=[x for x in range(0, 100, 10)],
    cmap="Oranges",
    plot_type=HeatmapPlotType.ViNE,
    lookup_function=lambda vine_result_dict, vine_settings: np.average([
        compute_average_node_load(vine_result)
        for vine_result in vine_result_dict[vine_settings]
    ]),
)

heatmap_specification_average_edge_load = dict(
    name="ViNE: Avg. Edge Load [%]",
    filename="avg_edge_load",
    vmin=0.0,
    vmax=100,
    colorbar_ticks=[x for x in range(0, 100, 10)],
    cmap="Purples",
    plot_type=HeatmapPlotType.ViNE,
    lookup_function=lambda vine_result_dict, vine_settings: np.average([
        compute_average_edge_load(vine_result)
        for vine_result in vine_result_dict[vine_settings]
    ]),
)

heatmap_specification_max_node_load = dict(
    name="ViNE: Max. Node Load [%]",
    filename="max_node_load",
    vmin=0.0,
    vmax=100,
    colorbar_ticks=[x for x in range(0, 101, 20)],
    cmap="Oranges",
    plot_type=HeatmapPlotType.ViNE,
    lookup_function=lambda vine_result_dict, vine_settings: max(
        compute_max_node_load(vine_result)
        for vine_result in vine_result_dict[vine_settings]
    )
)

heatmap_specification_max_edge_load = dict(
    name="ViNE: Max. Edge Load [%]",
    filename="max_edge_load",
    vmin=0.0,
    vmax=100,
    colorbar_ticks=[x for x in range(0, 101, 20)],
    cmap="Purples",
    plot_type=HeatmapPlotType.ViNE,
    lookup_function=lambda vine_result_dict, vine_settings: max(
        compute_max_edge_load(vine_result)
        for vine_result in vine_result_dict[vine_settings]
    )
)

heatmap_specification_max_load = dict(
    name="ViNE: MaxLoad (Edge and Node)",
    filename="max_load",
    vmin=0.0,
    vmax=100,
    colorbar_ticks=[x for x in range(0, 101, 20)],
    cmap="Reds",
    plot_type=HeatmapPlotType.ViNE,
    lookup_function=lambda vine_result_dict, vine_settings: max(
        compute_max_load(vine_result)
        for vine_result in vine_result_dict[vine_settings]
    )
)

heatmap_specification_avg_max_node_load_dynvmp = dict(
    name="DynVMP: Max node load",
    filename="max_node_load",
    vmin=0.0,
    vmax=100,
    colorbar_ticks=[x for x in range(0, 101, 20)],
    cmap="Reds",
    plot_type=HeatmapPlotType.RandRoundSepLPDynVMP,
    lookup_function=lambda dynvmp_result, dynvmp_variant_tuple: 100.0 * np.mean(dynvmp_result.max_node_loads[dynvmp_variant_tuple])
)
heatmap_specification_avg_max_edge_load_dynvmp = dict(
    name="DynVMP: Max edge load",
    filename="max_edge_load",
    vmin=0.0,
    vmax=100,
    colorbar_ticks=[x for x in range(0, 101, 20)],
    cmap="Reds",
    plot_type=HeatmapPlotType.RandRoundSepLPDynVMP,
    lookup_function=lambda dynvmp_result, dynvmp_variant_tuple: 100.0 * np.mean(dynvmp_result.max_edge_loads[dynvmp_variant_tuple])
)

heatmap_specification_mean_profit_dynvmp = dict(
    name="DynVMP: Mean Profit",
    filename="mean_profit",
    vmin=0.0,
    vmax=100,
    colorbar_ticks=[x for x in range(0, 101, 20)],
    cmap="Reds",
    plot_type=HeatmapPlotType.RandRoundSepLPDynVMP,
    lookup_function=lambda dynvmp_result, dynvmp_variant_tuple: np.mean(dynvmp_result.profits[dynvmp_variant_tuple])
)

heatmap_specification_lp_runtime_dynvmp = dict(
    name="DynVMP: LP runtime",
    filename="lp_runtime",
    vmin=0.0,
    vmax=100,
    colorbar_ticks=[x for x in range(0, 101, 20)],
    cmap="Blues",
    plot_type=HeatmapPlotType.RandRoundSepLPDynVMP,
    lookup_function=lambda dynvmp_result, dynvmp_variant_tuple: dynvmp_result.lp_time_optimization
)

global_heatmap_specfications = [
    # heatmap_specification_obj,
    heatmap_specification_runtime,
    heatmap_specification_embedding_ratio,
    heatmap_specification_average_node_load,
    heatmap_specification_average_edge_load,
    heatmap_specification_max_node_load,
    heatmap_specification_max_edge_load,
    heatmap_specification_max_load,
    heatmap_specification_avg_max_node_load_dynvmp,
    heatmap_specification_avg_max_edge_load_dynvmp,
    heatmap_specification_mean_profit_dynvmp,
    heatmap_specification_lp_runtime_dynvmp,
    # heatmap_specification_avg_load,
    # heatmap_specification_nu_real_req,
    # heatmap_specification_embedding_ratio_cleaned,
    # heatmap_specification_runtime_randround_preprocessing,
    # heatmap_specification_runtime_randround_optimization,
    # heatmap_specification_runtime_randround_postprocessing,
    # heatmap_specification_comparison_baseline_rr_mdk,
    # heatmap_specification_comparison_baseline_rr_heuristic,
    # heatmap_specification_comparison_baseline_rr_min_load,
    # heatmap_specification_comparison_baseline_rr_max_profit,
    # heatmap_specification_runtime_randround_runtime,
    # heatmap_specification_runtime_mdk_runtime,
]

heatmap_specifications_per_type = {
    plot_type_item: [
        heatmap_specification for heatmap_specification in global_heatmap_specfications
        if heatmap_specification['plot_type'] == plot_type_item
    ]
    for plot_type_item in [HeatmapPlotType.ViNE,
                           HeatmapPlotType.RandRoundSepLPDynVMP]
}

"""
Axes specifications used for the heatmap plots.
Each specification contains the following elements:
- x_axis_parameter: the parameter name on the x-axis
- y_axis_parameter: the parameter name on the y-axis
- x_axis_title:     the legend of the x-axis
- y_axis_title:     the legend of the y-axis
- foldername:       the folder to store the respective plots in
"""
heatmap_axes_specification_resources = dict(
    x_axis_parameter="node_resource_factor",
    y_axis_parameter="edge_resource_factor",
    x_axis_title="Node Resource Factor",
    y_axis_title="Edge Resource Factor",
    foldername="AXES_RESOURCES"
)

heatmap_axes_specification_requests_treewidth = dict(
    x_axis_parameter="treewidth",
    y_axis_parameter="number_of_requests",
    x_axis_title="Treewidth",
    y_axis_title="Number of Requests",
    foldername="AXES_TREEWIDTH_vs_NO_REQ"
)

heatmap_axes_specification_requests_edge_load = dict(
    x_axis_parameter="number_of_requests",
    y_axis_parameter="edge_resource_factor",
    x_axis_title="Number of Requests",
    y_axis_title="Edge Resource Factor",
    foldername="AXES_NO_REQ_vs_EDGE_RF"
)

heatmap_axes_specification_requests_node_load = dict(
    x_axis_parameter="number_of_requests",
    y_axis_parameter="node_resource_factor",
    x_axis_title="Number of Requests",
    y_axis_title="Node Resource Factor",
    foldername="AXES_NO_REQ_vs_NODE_RF"
)

global_heatmap_axes_specifications = (
    heatmap_axes_specification_requests_edge_load,
    heatmap_axes_specification_requests_treewidth,
    heatmap_axes_specification_resources,
    heatmap_axes_specification_requests_node_load,
)


def compute_average_node_load(result_summary):
    logger.warn("In the function compute_average_node_load the single universal node type 'univerval' is assumed."
                "This should be fixed in the future and might yield wrong results when considering more general "
                "resource types. Disregard this warning if you know what you are doing.")
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        if x == "universal":
            cum_loads.append(result_summary.load[(x, y)])
    return np.mean(cum_loads)


def compute_average_edge_load(result_summary):
    logger.warn("In the function compute_average_edge_load the single universal node type 'univerval' is assumed."
                "This should be fixed in the future and might yield wrong results when considering more general "
                "resource types. Disregard this warning if you know what you are doing.")
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        if x != "universal":
            cum_loads.append(result_summary.load[(x, y)])
    return np.mean(cum_loads)


def compute_max_node_load(result_summary):
    logger.warn("In the function compute_max_node_load the single universal node type 'univerval' is assumed."
                "This should be fixed in the future and might yield wrong results when considering more general "
                "resource types.  Disregard this warning if you know what you are doing.")
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        if x == "universal":
            cum_loads.append(result_summary.load[(x, y)])
    return max(cum_loads)


def compute_max_edge_load(result_summary):
    logger.warn("In the function compute_max_edge_load the single universal node type 'univerval' is assumed."
                "This should be fixed in the future and might yield wrong results when considering more general "
                "resource types. Disregard this warning if you know what you are doing.")
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        if x != "universal":
            cum_loads.append(result_summary.load[(x, y)])
    return max(cum_loads)


def compute_avg_load(result_summary):
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        cum_loads.append(result_summary.load[(x, y)])
    return np.mean(cum_loads)


def compute_max_load(result_summary):
    cum_loads = []
    for (x, y) in result_summary.load.keys():
        cum_loads.append(result_summary.load[(x, y)])
    return max(cum_loads)


def get_title_for_filter_specifications(filter_specifications):
    result = "\n".join(
        [filter_specification['parameter'] + "=" + str(filter_specification['value']) + "; " for filter_specification in
         filter_specifications])
    return result[:-2]


def extract_parameter_range(scenario_parameter_space_dict, key, min_recursion_depth=0):
    if not isinstance(scenario_parameter_space_dict, dict):
        return None
    for generator_name, value in scenario_parameter_space_dict.iteritems():
        if generator_name == key and min_recursion_depth <= 0:
            return [key], value
        if isinstance(value, list):
            if len(value) != 1:
                continue
            value = value[0]
            result = extract_parameter_range(value, key, min_recursion_depth=min_recursion_depth - 1)
            if result is not None:
                path, values = result
                return [generator_name, 0] + path, values
        elif isinstance(value, dict):
            result = extract_parameter_range(value, key, min_recursion_depth=min_recursion_depth - 1)
            if result is not None:
                path, values = result
                return [generator_name] + path, values
    return None


def extract_generation_parameters(scenario_parameter_dict, scenario_id):
    if not isinstance(scenario_parameter_dict, dict):
        return None

    results = []

    for generator_name, value in scenario_parameter_dict.iteritems():
        if isinstance(value, set) and generator_name != "all" and scenario_id in value:
            return [[generator_name]]
        if isinstance(value, list):
            if len(value) != 1:
                continue
            value = value[0]
            result = extract_generation_parameters(value, scenario_id)
            if result is not None:
                for atomic_result in result:
                    results.append([generator_name] + atomic_result)
        elif isinstance(value, dict):
            result = extract_generation_parameters(value, scenario_id)
            if result is not None:
                for atomic_result in result:
                    results.append([generator_name] + atomic_result)

    if results == []:
        return None
    else:
        # print "returning {}".format(results)
        return results


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


def load_reduced_pickle(reduced_pickle):
    with open(reduced_pickle, "rb") as f:
        data = pickle.load(f)
    return data


class AbstractPlotter(object):
    ''' Abstract Plotter interface providing functionality used by the majority of plotting classes of this module.
    '''

    def __init__(self,
                 output_path,
                 output_filetype,
                 scenario_solution_storage,
                 algorithm_id,
                 execution_id,
                 show_plot=False,
                 save_plot=True,
                 overwrite_existing_files=False,
                 forbidden_scenario_ids=None,
                 paper_mode=True
                 ):
        self.output_path = output_path
        self.output_filetype = output_filetype
        self.scenario_solution_storage = scenario_solution_storage

        self.algorithm_id = algorithm_id
        self.execution_id = execution_id

        self.scenario_parameter_dict = self.scenario_solution_storage.scenario_parameter_container.scenario_parameter_dict
        self.scenarioparameter_room = self.scenario_solution_storage.scenario_parameter_container.scenarioparameter_room
        self.all_scenario_ids = set(scenario_solution_storage.algorithm_scenario_solution_dictionary[self.algorithm_id].keys())

        self.show_plot = show_plot
        self.save_plot = save_plot
        self.overwrite_existing_files = overwrite_existing_files
        if not forbidden_scenario_ids:
            self.forbidden_scenario_ids = set()
        else:
            self.forbidden_scenario_ids = forbidden_scenario_ids
        self.paper_mode = paper_mode

    def _construct_output_path_and_filename(self, title, filter_specifications=None):
        filter_spec_path = ""
        filter_filename = "no_filter.{}".format(OUTPUT_FILETYPE)
        if filter_specifications:
            filter_spec_path, filter_filename = self._construct_path_and_filename_for_filter_spec(filter_specifications)
        base = os.path.normpath(OUTPUT_PATH)
        date = strftime("%Y-%m-%d", gmtime())
        output_path = os.path.join(base, date, OUTPUT_FILETYPE, "general_plots", filter_spec_path)
        filename = os.path.join(output_path, title + "_" + filter_filename)
        return output_path, filename

    def _construct_path_and_filename_for_filter_spec(self, filter_specifications):
        filter_path = ""
        filter_filename = ""
        for spec in filter_specifications:
            filter_path = os.path.join(filter_path, (spec['parameter'] + "_" + str(spec['value'])))
            filter_filename += spec['parameter'] + "_" + str(spec['value']) + "_"
        filter_filename = filter_filename[:-1] + "." + OUTPUT_FILETYPE
        return filter_path, filter_filename

    def _obtain_scenarios_based_on_filters(self, filter_specifications=None):
        allowed_scenario_ids = set(self.all_scenario_ids)
        sps = self.scenarioparameter_room
        spd = self.scenario_parameter_dict
        if filter_specifications:
            for filter_specification in filter_specifications:
                filter_path, _ = extract_parameter_range(sps, filter_specification['parameter'])
                filter_indices = lookup_scenarios_having_specific_values(spd, filter_path,
                                                                         filter_specification['value'])
                allowed_scenario_ids = allowed_scenario_ids & filter_indices

        return allowed_scenario_ids

    def _obtain_scenarios_based_on_axis(self, axis_path, axis_value):
        spd = self.scenario_parameter_dict
        return lookup_scenarios_having_specific_values(spd, axis_path, axis_value)

    def _show_and_or_save_plots(self, output_path, filename):
        plt.tight_layout()
        if self.save_plot:
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            print "saving plot: {}".format(filename)
            plt.savefig(filename)
        if self.show_plot:
            plt.show()

        plt.close()

    def plot_figure(self, filter_specifications):
        raise RuntimeError("This is an abstract method")


class SingleHeatmapPlotter(AbstractPlotter):

    def __init__(self,
                 output_path,
                 output_filetype,
                 scenario_solution_storage,
                 algorithm_id,
                 execution_id,
                 heatmap_plot_type,
                 algorithm_sub_parameter,
                 list_of_axes_specifications=global_heatmap_axes_specifications,
                 list_of_metric_specifications=None,
                 show_plot=False,
                 save_plot=True,
                 overwrite_existing_files=False,
                 forbidden_scenario_ids=None,
                 paper_mode=True
                 ):
        super(SingleHeatmapPlotter, self).__init__(output_path, output_filetype, scenario_solution_storage,
                                                   algorithm_id, execution_id, show_plot, save_plot,
                                                   overwrite_existing_files, forbidden_scenario_ids, paper_mode)
        if heatmap_plot_type is None or heatmap_plot_type not in HeatmapPlotType.VALUE_RANGE:
            raise RuntimeError("heatmap_plot_type {} is not a valid input. Must be of type HeatmapPlotType.".format(heatmap_plot_type))
        self.heatmap_plot_type = heatmap_plot_type

        if not list_of_axes_specifications:
            raise RuntimeError("Axes need to be provided.")
        self.list_of_axes_specifications = list_of_axes_specifications

        if not list_of_metric_specifications:
            self.list_of_metric_specifications = heatmap_specifications_per_type[self.heatmap_plot_type]
        else:
            for metric_specification in list_of_metric_specifications:
                if metric_specification.plot_type != self.heatmap_plot_type:
                    raise RuntimeError("The metric specification {} does not agree with the plot type {}.".format(metric_specification, self.heatmap_plot_type))
            self.list_of_metric_specifications = list_of_metric_specifications

        self.algorithm_sub_parameter = algorithm_sub_parameter

    def _construct_output_path_and_filename(self, metric_specification,
                                            heatmap_axes_specification,
                                            filter_specifications=None):
        filter_spec_path = ""
        filter_filename = "no_filter.{}".format(OUTPUT_FILETYPE)
        if filter_specifications:
            filter_spec_path, filter_filename = self._construct_path_and_filename_for_filter_spec(filter_specifications)

        base = os.path.normpath(OUTPUT_PATH)
        date = strftime("%Y-%m-%d", gmtime())
        axes_foldername = heatmap_axes_specification['foldername']
        sub_param_string = get_alg_variant_string(self.heatmap_plot_type, self.algorithm_sub_parameter)

        output_path = os.path.join(base, date, OUTPUT_FILETYPE, axes_foldername, sub_param_string, filter_spec_path)
        fname = "__".join(str(x) for x in [
            metric_specification['filename'],
            filter_filename,
        ])
        filename = os.path.join(output_path, fname)
        return output_path, filename

    def plot_figure(self, filter_specifications):
        for axes_specification in self.list_of_axes_specifications:
            for metric_specfication in self.list_of_metric_specifications:
                self.plot_single_heatmap_general(metric_specfication, axes_specification, filter_specifications)

    def _lookup_solutions(self, scenario_ids):
        solution_dicts = [self.scenario_solution_storage.get_solutions_by_scenario_index(x) for x in scenario_ids]
        result = [x[self.algorithm_id][self.execution_id] for x in solution_dicts]
        if self.heatmap_plot_type == HeatmapPlotType.ViNE:
            # result should be a list of dicts mapping vine_settings to lists of ReducedOfflineViNEResultCollection instances
            if result and self.algorithm_sub_parameter not in result[0]:
                return None
        elif self.heatmap_plot_type == HeatmapPlotType.RandRoundSepLPDynVMP:
            # result should be a list of ReducedRandRoundSepLPOptDynVMPCollectionResult instances
            if result and self.algorithm_sub_parameter not in result[0].profits:
                return None
        return result

    def plot_single_heatmap_general(self,
                                    heatmap_metric_specification,
                                    heatmap_axes_specification,
                                    filter_specifications=None):
        # data extraction

        sps = self.scenarioparameter_room
        spd = self.scenario_parameter_dict

        output_path, filename = self._construct_output_path_and_filename(heatmap_metric_specification,
                                                                         heatmap_axes_specification,
                                                                         filter_specifications)

        logger.debug("output_path is {};\t filename is {}".format(output_path, filename))

        if not self.overwrite_existing_files and os.path.exists(filename):
            logger.info("Skipping generation of {} as this file already exists".format(filename))
            return

        # check if filter specification conflicts with axes specification
        if filter_specifications is not None:
            for filter_specification in filter_specifications:
                if (heatmap_axes_specification['x_axis_parameter'] == filter_specification['parameter'] or
                        heatmap_axes_specification['y_axis_parameter'] == filter_specification['parameter']):
                    logger.debug("Skipping generation of {} as the filter specification conflicts with the axes specification.")
                    return

        path_x_axis, xaxis_parameters = extract_parameter_range(
            sps,
            heatmap_axes_specification['x_axis_parameter'],
            min_recursion_depth=2,
        )
        path_y_axis, yaxis_parameters = extract_parameter_range(
            sps,
            heatmap_axes_specification['y_axis_parameter'],
            min_recursion_depth=2,
        )

        # for heatmap plot
        xaxis_parameters.sort()
        yaxis_parameters.sort()

        # all heatmap values will be stored in X
        X = np.zeros((len(yaxis_parameters), len(xaxis_parameters)))
        column_labels = yaxis_parameters
        row_labels = xaxis_parameters

        min_number_of_observed_values = 10000000000000
        max_number_of_observed_values = 0
        observed_values = np.empty(0)

        for x_index, x_val in enumerate(xaxis_parameters):
            # all scenario indices which has x_val as xaxis parameter (e.g. node_resource_factor = 0.5
            scenario_ids_matching_x_axis = lookup_scenarios_having_specific_values(spd, path_x_axis, x_val)
            for y_index, y_val in enumerate(yaxis_parameters):
                scenario_ids_matching_y_axis = lookup_scenarios_having_specific_values(spd, path_y_axis, y_val)

                filter_indices = self._obtain_scenarios_based_on_filters(filter_specifications)
                scenario_ids_to_consider = (scenario_ids_matching_x_axis &
                                            scenario_ids_matching_y_axis &
                                            filter_indices) - self.forbidden_scenario_ids

                solutions = self._lookup_solutions(scenario_ids_to_consider)
                if not solutions:
                    logger.info("Found no solutions for algorithm variant {}".format(self.algorithm_sub_parameter))
                    return

                values = [heatmap_metric_specification['lookup_function'](solution, self.algorithm_sub_parameter)
                          for solution in solutions]

                if 'metric_filter' in heatmap_metric_specification:
                    values = [value for value in values if heatmap_metric_specification['metric_filter'](value)]

                observed_values = np.append(observed_values, values)

                if len(values) < min_number_of_observed_values:
                    min_number_of_observed_values = len(values)
                if len(values) > max_number_of_observed_values:
                    max_number_of_observed_values = len(values)

                logger.debug("values are {}".format(values))
                m = np.nanmean(values)
                logger.debug("mean is {}".format(m))

                if 'rounding_function' in heatmap_metric_specification:
                    rounded_m = heatmap_metric_specification['rounding_function'](m)
                else:
                    rounded_m = float("{0:.1f}".format(round(m, 2)))

                X[y_index, x_index] = rounded_m

        if min_number_of_observed_values == max_number_of_observed_values:
            solution_count_string = "{} values per square".format(min_number_of_observed_values)
        else:
            solution_count_string = "between {} and {} values per square".format(min_number_of_observed_values,
                                                                                 max_number_of_observed_values)

        fig, ax = plt.subplots(figsize=(5, 4))
        if self.paper_mode:
            ax.set_title(heatmap_metric_specification['name'], fontsize=17)
        else:
            title = heatmap_metric_specification['name'] + "\n"
            if filter_specifications:
                title += get_title_for_filter_specifications(filter_specifications) + "\n"
            title += solution_count_string + "\n"
            title += "min: {:.2f}; mean: {:.2f}; max: {:.2f}".format(np.nanmin(observed_values),
                                                                     np.nanmean(observed_values),
                                                                     np.nanmax(observed_values))

            ax.set_title(title)

        heatmap = ax.pcolor(X,
                            cmap=heatmap_metric_specification['cmap'],
                            vmin=heatmap_metric_specification['vmin'],
                            vmax=heatmap_metric_specification['vmax'])

        for x_index in range(X.shape[1]):
            for y_index in range(X.shape[0]):
                plt.text(x_index + .5,
                         y_index + .45,
                         X[y_index, x_index],
                         verticalalignment="center",
                         horizontalalignment="center",
                         fontsize=17.5,
                         fontname="Courier New",
                         # family="monospace",
                         color='w',
                         path_effects=[PathEffects.withStroke(linewidth=4, foreground="k")]
                         )

        if not self.paper_mode:
            fig.colorbar(heatmap, label=heatmap_metric_specification['name'] + ' - mean in blue')
        else:
            ticks = heatmap_metric_specification['colorbar_ticks']
            tick_labels = [str(tick).ljust(3) for tick in ticks]
            cbar = fig.colorbar(heatmap)
            cbar.set_ticks(ticks)
            cbar.set_ticklabels(tick_labels)
            # for label in cbar.ax.get_yticklabels():
            #    label.set_fontproperties(font_manager.FontProperties(family="Courier New",weight='bold'))

            cbar.ax.tick_params(labelsize=15.5)

        ax.set_yticks(np.arange(X.shape[0]) + 0.5, minor=False)
        ax.set_xticks(np.arange(X.shape[1]) + 0.5, minor=False)

        ax.set_xticklabels(row_labels, minor=False, fontsize=15.5)
        ax.set_xlabel(heatmap_axes_specification['x_axis_title'], fontsize=16)
        ax.set_ylabel(heatmap_axes_specification['y_axis_title'], fontsize=16)
        ax.set_yticklabels(column_labels, minor=False, fontsize=15.5)

        self._show_and_or_save_plots(output_path, filename)
        plt.close(fig)


def _construct_filter_specs(scenario_parameter_space_dict, parameter_filter_keys, maxdepth=3):
    parameter_value_dic = dict()
    for parameter in parameter_filter_keys:
        _, parameter_values = extract_parameter_range(scenario_parameter_space_dict,
                                                      parameter)
        parameter_value_dic[parameter] = parameter_values
    # print parameter_value_dic.values()
    result_list = [None]
    for i in range(1, maxdepth + 1):
        for combi in combinations(parameter_value_dic, i):
            values = []
            for element_of_combi in combi:
                values.append(parameter_value_dic[element_of_combi])
            for v in product(*values):
                _filter = []
                for (parameter, value) in zip(combi, v):
                    _filter.append({'parameter': parameter, 'value': value})
                result_list.append(_filter)

    return result_list


def get_alg_variant_string(plot_type, algorithm_sub_parameter):
    if plot_type == HeatmapPlotType.ViNE:
        vine.ViNESettingsFactory.check_vine_settings(algorithm_sub_parameter)
        is_splittable = algorithm_sub_parameter.edge_embedding_model == vine.ViNEEdgeEmbeddingModel.SPLITTABLE
        is_load_balanced_objective = (
                algorithm_sub_parameter.lp_objective in
                [vine.ViNELPObjective.ViNE_LB_DEF, vine.ViNELPObjective.ViNE_LB_INCL_SCENARIO_COSTS]
        )
        is_cost_objective = (
                algorithm_sub_parameter.lp_objective in
                [vine.ViNELPObjective.ViNE_COSTS_DEF, vine.ViNELPObjective.ViNE_LB_INCL_SCENARIO_COSTS]
        )
        is_random_rounding_procedure = algorithm_sub_parameter.rounding_procedure == vine.ViNERoundingProcedure.RANDOMIZED
        return "vine_{}{}{}{}".format(
            "mcf" if is_splittable else "sp",
            "_lb" if is_load_balanced_objective else "",
            "_cost" if is_cost_objective else "",
            "_rand" if is_random_rounding_procedure else "_det",
        )
    elif plot_type == HeatmapPlotType.RandRoundSepLPDynVMP:
        lp_mode, rounding_mode = algorithm_sub_parameter
        if lp_mode == treewidth_model.LPRecomputationMode.NONE:
            lp_str = "recomp_none"
        elif lp_mode == treewidth_model.LPRecomputationMode.RECOMPUTATION_WITHOUT_SEPARATION:
            lp_str = "recomp_no_sep"
        elif lp_mode == treewidth_model.LPRecomputationMode.RECOMPUTATION_WITH_SINGLE_SEPARATION:
            lp_str = "recomp_single_sep"
        else:
            raise ValueError()
        if rounding_mode == treewidth_model.RoundingOrder.RANDOM:
            rounding_str = "round_rand"
        elif rounding_mode == treewidth_model.RoundingOrder.STATIC_REQ_PROFIT:
            rounding_str = "round_static_profit"
        elif rounding_mode == treewidth_model.RoundingOrder.ACHIEVED_REQ_PROFIT:
            rounding_str = "round_achieved_profit"
        else:
            raise ValueError()

        return "dynvmp__{}__{}".format(
            lp_str,
            rounding_str,
        )
    else:
        raise ValueError("Unexpected HeatmapPlotType {}".format(plot_type))


def evaluate_baseline_and_randround(datacontainer,
                                    baseline_algorithm_id,
                                    baseline_execution_config,
                                    heatmap_plot_type,
                                    exclude_generation_parameters=None,
                                    parameter_filter_keys=None,
                                    show_plot=False,
                                    save_plot=True,
                                    overwrite_existing_files=True,
                                    forbidden_scenario_ids=None,
                                    papermode=True,
                                    maxdepthfilter=2,
                                    output_path="./",
                                    output_filetype="png"):
    """ Main function for evaluation, creating plots and saving them in a specific directory hierarchy.
    A large variety of plots is created. For heatmaps, a generic plotter is used while for general
    comparison plots (ECDF and scatter) an own class is used. The plots that shall be generated cannot
    be controlled at the moment but the respective plotters can be easily adjusted.

    :param heatmap_plot_type:
    :param datacontainer: unpickled datacontainer of baseline experiments (e.g. MIP)
    :param baseline_algorithm_id: algorithm id of the baseline algorithm
    :param baseline_execution_config: execution config (numeric) of the baseline algorithm execution
    :param exclude_generation_parameters:   specific generation parameters that shall be excluded from the evaluation.
                                            These won't show in the plots and will also not be shown on axis labels etc.
    :param parameter_filter_keys:   name of parameters according to which the results shall be filtered
    :param show_plot:               Boolean: shall plots be shown
    :param save_plot:               Boolean: shall the plots be saved
    :param overwrite_existing_files:   shall existing files be overwritten?
    :param forbidden_scenario_ids:     list / set of scenario ids that shall not be considered in the evaluation
    :param papermode:                  nicely layouted plots (papermode) or rather additional information?
    :param maxdepthfilter:             length of filter permutations that shall be considered
    :param output_path:                path to which the results shall be written
    :param output_filetype:            filetype supported by matplotlib to export figures
    :return: None
    """

    if forbidden_scenario_ids is None:
        forbidden_scenario_ids = set()

    if exclude_generation_parameters is not None:
        for key, values_to_exclude in exclude_generation_parameters.iteritems():
            parameter_filter_path, parameter_values = extract_parameter_range(
                datacontainer.scenario_parameter_container.scenarioparameter_room, key)

            parameter_dicts_baseline = lookup_scenario_parameter_room_dicts_on_path(
                datacontainer.scenario_parameter_container.scenarioparameter_room, parameter_filter_path)

            for value_to_exclude in values_to_exclude:

                if value_to_exclude not in parameter_values:
                    raise RuntimeError("The value {} is not contained in the list of parameter values {} for key {}".format(
                        value_to_exclude, parameter_values, key
                    ))

                # add respective scenario ids to the set of forbidden scenario ids
                forbidden_scenario_ids.update(set(lookup_scenarios_having_specific_values(
                    datacontainer.scenario_parameter_container.scenario_parameter_dict, parameter_filter_path, value_to_exclude)))

            # remove the respective values from the scenario parameter room such that these are not considered when
            # constructing e.g. axes
            parameter_dicts_baseline[-1][key] = [value for value in parameter_dicts_baseline[-1][key] if
                                                 value not in values_to_exclude]

    if parameter_filter_keys is not None:
        filter_specs = _construct_filter_specs(datacontainer.scenario_parameter_container.scenarioparameter_room,
                                               parameter_filter_keys,
                                               maxdepth=maxdepthfilter)
    else:
        filter_specs = [None]

    plotters = []
    # initialize plotters for each valid vine setting...

    for algorithm_sub_parameter in iterate_algorithm_sub_parameters(heatmap_plot_type):
        baseline_plotter = SingleHeatmapPlotter(output_path=output_path,
                                                output_filetype=output_filetype,
                                                scenario_solution_storage=datacontainer,
                                                algorithm_id=baseline_algorithm_id,
                                                algorithm_sub_parameter=algorithm_sub_parameter,
                                                execution_id=baseline_execution_config,
                                                heatmap_plot_type=heatmap_plot_type,
                                                show_plot=show_plot,
                                                save_plot=save_plot,
                                                overwrite_existing_files=overwrite_existing_files,
                                                forbidden_scenario_ids=forbidden_scenario_ids,
                                                paper_mode=papermode)

        plotters.append(baseline_plotter)

    for filter_spec in filter_specs:
        for plotter in plotters:
            plotter.plot_figure(filter_spec)


def iterate_algorithm_sub_parameters(plot_type):
    if plot_type == HeatmapPlotType.ViNE:
        for (edge_embedding_model, lp_objective, rounding_procedure) in itertools.product(
                vine.ViNEEdgeEmbeddingModel,
                vine.ViNELPObjective,
                vine.ViNERoundingProcedure,
        ):
            yield vine.ViNESettingsFactory.get_vine_settings(
                edge_embedding_model=edge_embedding_model,
                lp_objective=lp_objective,
                rounding_procedure=rounding_procedure,
            )
    elif plot_type == HeatmapPlotType.RandRoundSepLPDynVMP:
        for sub_param in itertools.product(
                treewidth_model.LPRecomputationMode,
                treewidth_model.RoundingOrder,
        ):
            yield sub_param
