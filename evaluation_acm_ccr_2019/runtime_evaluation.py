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
from itertools import combinations, product
from time import gmtime, strftime

import matplotlib.patches as mpatches
import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np

from alib import solutions, util
from vnep_approx import vine, treewidth_model

try:
    import cPickle as pickle
except ImportError:
    import pickle

REQUIRED_FOR_PICKLE = solutions  # this prevents pycharm from removing this import, which is required for unpickling solutions

OUTPUT_PATH = None
OUTPUT_FILETYPE = "png"

BOX_WIDTH = 0.5
BOX_SEPARATION_WITHIN_GROUP = 0.6
BOX_SEPARATION_BETWEEN_GROUPS = 0.3

PLOT_TITLE_FONTSIZE = 18
AXIS_LABEL_FONTSIZE = 16
LEGEND_TITLE_FONTSIZE = 16
LEGEND_LABEL_FONTSIZE = 15
AXIS_TICKLABEL_FONTSIZE = 14.8

logger = util.get_logger(__name__, make_file=False, propagate=True)


def get_list_of_vine_settings():
    result = []
    for (edge_embedding_model, lp_objective, rounding_procedure) in itertools.product(
            vine.ViNEEdgeEmbeddingModel,
            vine.ViNELPObjective,
            vine.ViNERoundingProcedure,
    ):
        if lp_objective == vine.ViNELPObjective.ViNE_LB_INCL_SCENARIO_COSTS or lp_objective == vine.ViNELPObjective.ViNE_COSTS_INCL_SCENARIO_COSTS:
            continue
        result.append(vine.ViNESettingsFactory.get_vine_settings(
            edge_embedding_model=edge_embedding_model,
            lp_objective=lp_objective,
            rounding_procedure=rounding_procedure,
        ))
    return result


def get_list_of_rr_settings():
    result = []
    for sub_param in itertools.product(
            treewidth_model.LPRecomputationMode,
            treewidth_model.RoundingOrder,
    ):
        if sub_param[0] == treewidth_model.LPRecomputationMode.RECOMPUTATION_WITH_SINGLE_SEPARATION:
            continue
        result.append(sub_param)
    return result


def get_alg_variant_string(alg_id, algorithm_sub_parameter):
    if alg_id == vine.OfflineViNEAlgorithmCollection.ALGORITHM_ID:
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
    elif alg_id == treewidth_model.RandRoundSepLPOptDynVMPCollection.ALGORITHM_ID:
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
        raise ValueError("Unexpected HeatmapPlotType {}".format(alg_id))


def compute_aggregated_mean(list_of_aggregated_data, debug=False):
    mean = 0.0
    value_count = 0
    for agg in list_of_aggregated_data:
        mean += agg.mean * agg.value_count
        value_count += agg.value_count
    if debug:
        print len(list_of_aggregated_data), value_count, mean / value_count
    return mean / value_count


def lookup_rounding_runtimes(rr_result):
    rr_settings_list = get_list_of_rr_settings()

    result = []
    for rr_settings in rr_settings_list:
        if rr_settings[0] == treewidth_model.LPRecomputationMode.RECOMPUTATION_WITHOUT_SEPARATION:
            # only then consider results
            result.append(rr_result.rounding_runtimes[rr_settings].mean)
    return result


lp_runtime_metric = dict(
    name="$\mathsf{LP}_{\mathsf{DynVMP}}$ Runtime",
    y_axis_title="Runtime [s]",
    lookup_function=lambda rr_result: rr_result.lp_time_optimization + rr_result.lp_time_preprocess,
    result_num="single",
    filename="lp_optimization_time",
)

lp_rounding_time_metric = dict(
    name="Recomp. Heuristic Runtime",
    y_axis_title="Runtime [s]",
    lookup_function=lambda rr_result: lookup_rounding_runtimes(rr_result),
    result_num="many",
    filename="solution_rounding_time",
)

lp_dynvmp_time_metric = dict(
    name="LP DynVMP Runtime (Total)",
    y_axis_title="Runtime [s]",
    lookup_function=lambda rr_result: rr_result.lp_time_optimization +
                                      rr_result.lp_time_preprocess -
                                      rr_result.lp_time_tree_decomposition.mean * rr_result.lp_time_tree_decomposition.value_count -
                                      rr_result.lp_time_gurobi_optimization.mean * rr_result.lp_time_gurobi_optimization.value_count,
    result_num="single",
    filename="lp_dynvmp_time_total",
)

lp_dynvmp_time_metric_percentage = dict(
    name="Sep. Runtime Share    ",
    y_axis_title="Fraction of Total [%]",
    lookup_function=lambda rr_result: 100.0*(rr_result.lp_time_optimization +
                                      rr_result.lp_time_preprocess -
                                      rr_result.lp_time_tree_decomposition.mean * rr_result.lp_time_tree_decomposition.value_count -
                                      rr_result.lp_time_gurobi_optimization.mean * rr_result.lp_time_gurobi_optimization.value_count)/(rr_result.lp_time_optimization +
                                      rr_result.lp_time_preprocess),
    result_num="single",
    non_log=True,
    filename="lp_dynvmp_time_total_percentage",
)

lp_dynvmp_time_metric_average_single_runtime = dict(
    name="Avg. DynVMP Runtime",
    y_axis_title="Runtime [s]",
    lookup_function=lambda rr_result: [res.mean for res in rr_result.lp_time_dynvmp_computation],
    result_num="many",
    filename="lp_dynvmp_time_computation_per_request",
)

lp_dynvmp_time_metric_average_separation_runtime = dict(
    name="Separation Runtime     ",
    y_axis_title="Runtime [s]",
    lookup_function=lambda rr_result: sum([res.mean for res in rr_result.lp_time_dynvmp_computation]),
    result_num="single",
    filename="lp_dynvmp_time_per_separation",
)



lp_dynvmp_init_time_metric_sum = dict(
    name="LP DynVMP Initialization (Sum)",
    y_axis_title="Runtime [s]",
    lookup_function=lambda rr_result: rr_result.lp_time_dynvmp_initialization.mean * rr_result.lp_time_dynvmp_initialization.value_count,
    result_num="single",
    filename="lp_dynvmp_init_sum",
)

lp_dynvmp_init_time_metric_mean = dict(
    name="LP DynVMP Initialization (Mean)",
    y_axis_title="Runtime [s]",
    lookup_function=lambda rr_result: rr_result.lp_time_dynvmp_initialization.mean,
    result_num="single",
    filename="lp_dynvmp_init_mean",
)

global_metric_specifications = (
    lp_runtime_metric,
    lp_rounding_time_metric,
    lp_dynvmp_time_metric,
    lp_dynvmp_time_metric_percentage,
    lp_dynvmp_init_time_metric_sum,
    lp_dynvmp_init_time_metric_mean,
    lp_dynvmp_time_metric_average_single_runtime,
    lp_dynvmp_time_metric_average_separation_runtime
)

"""
Axes specifications used for the heatmap plots.
Each specification contains the following elements:
- x_axis_parameter: the parameter name on the x-axis
- y_axis_parameter: the parameter name on the y-axis
- x_axis_title:     the legend of the x-axis
- y_axis_title:     the legend of the y-axis
- foldername:       the folder to store the respective plots in
"""
boxplot_axes_specification_resources = dict(
    x_axis_parameter="node_resource_factor",
    x_axis_title="Node Resource Factor",
    x_axis_title_short="NRF",
)

boxplot_axes_specification_requests_treewidth = dict(
    x_axis_parameter="treewidth",
    x_axis_title="Treewidth",
    x_axis_title_short="TW",
)

boxplot_axes_specification_requests_num_req = dict(
    x_axis_parameter="number_of_requests",
    x_axis_title="Number of Requests",
    x_axis_title_short="#req.",
)

boxplot_outer_axes_specifications = (
    boxplot_axes_specification_requests_treewidth,
    # boxplot_axes_specification_resources,
    # boxplot_axes_specification_requests_num_req,
)

boxplot_inner_axes_specifications = (
    # boxplot_axes_specification_requests_treewidth,
    # boxplot_axes_specification_resources,
    boxplot_axes_specification_requests_num_req,
)


def get_title_for_filter_specifications(filter_specifications):
    result = "\n".join(
        [filter_specification['parameter'] + "=" + str(filter_specification['value']) + "; " for filter_specification in
         filter_specifications])
    return result[:-2]


def extract_parameter_range(scenario_parameter_space, key):
    # if the scenario parameter container was merged with another, the parameter space is a list of dicts
    # we iterate over all of these parameter subspaces and collect all values matching the parameter
    if not isinstance(scenario_parameter_space, list):
        scenario_parameter_space = [scenario_parameter_space]
    path = None
    values = set()
    for sps in scenario_parameter_space:
        new_path, new_values = _extract_parameter_range(
            sps, key, min_recursion_depth=2
        )
        if path is None:
            path = new_path
        else:
            assert path == new_path  # this should usually not happen unless we merged incompatible parameter containers
        values = values.union(new_values)
    return path, sorted(values)


def _extract_parameter_range(scenario_parameter_space_dict, key, min_recursion_depth=0):
    if not isinstance(scenario_parameter_space_dict, dict):
        return None
    for generator_name, value in scenario_parameter_space_dict.iteritems():
        if generator_name == key and min_recursion_depth <= 0:
            return [key], value
        if isinstance(value, list):
            if len(value) != 1:
                continue
            value = value[0]
            result = _extract_parameter_range(value, key, min_recursion_depth=min_recursion_depth - 1)
            if result is not None:
                path, values = result
                return [generator_name, 0] + path, values
        elif isinstance(value, dict):
            result = _extract_parameter_range(value, key, min_recursion_depth=min_recursion_depth - 1)
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


def _get_lp_str(lp_mode):
    lp_str = None
    if lp_mode == treewidth_model.LPRecomputationMode.NONE:
        lp_str = "no_recomp"
    elif lp_mode == treewidth_model.LPRecomputationMode.RECOMPUTATION_WITHOUT_SEPARATION:
        lp_str = "recomp_no_sep"
    elif lp_mode == treewidth_model.LPRecomputationMode.RECOMPUTATION_WITH_SINGLE_SEPARATION:
        lp_str = "recomp_single_sep"
    else:
        raise ValueError()
    return lp_str


def _get_rounding_str(rounding_mode):
    rounding_str = None
    if rounding_mode == treewidth_model.RoundingOrder.RANDOM:
        rounding_str = "round_rand"
    elif rounding_mode == treewidth_model.RoundingOrder.STATIC_REQ_PROFIT:
        rounding_str = "round_static_profit"
    elif rounding_mode == treewidth_model.RoundingOrder.ACHIEVED_REQ_PROFIT:
        rounding_str = "round_achieved_profit"
    else:
        raise ValueError()
    return rounding_str


def get_specific_rr_name(rr_settings):
    return "rr_seplp_{}__{}".format(
        _get_lp_str(rr_settings[0]),
        _get_rounding_str(rr_settings[1]),
    )


def get_all_rr_settings_list_with_names():
    result = []

    rr_settings_list = get_list_of_rr_settings()
    result.append((rr_settings_list, "rr_seplp_ALL"))  # first off: every vine combination

    # second: each specific one
    for rr_settings in rr_settings_list:
        result.append(([rr_settings], get_specific_rr_name(rr_settings)))

    # third: each aggregation level, when applicable, i.e. there is more than one setting for that
    for lp_mode in treewidth_model.LPRecomputationMode:
        matching_settings = []
        for rr_settings in rr_settings_list:
            if rr_settings[0] == lp_mode:
                matching_settings.append(rr_settings)
        if len(matching_settings) > 0 and len(matching_settings) != len(rr_settings_list):
            result.append((matching_settings, "rr_seplp_{}".format(
                _get_lp_str(lp_mode).upper())))

    for rounding_mode in treewidth_model.RoundingOrder:
        matching_settings = []
        for rr_settings in rr_settings_list:
            if rr_settings[1] == rounding_mode:
                matching_settings.append(rr_settings)
        if len(matching_settings) > 0 and len(matching_settings) != len(rr_settings_list):
            result.append((matching_settings, "rr_seplp_{}".format(
                _get_rounding_str(rounding_mode).upper()
            )))

    return result


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
        filter_filename = "no_filter.{}".format(self.output_filetype)
        if filter_specifications:
            filter_spec_path, filter_filename = self._construct_path_and_filename_for_filter_spec(filter_specifications)
        base = os.path.normpath(OUTPUT_PATH)
        date = strftime("%Y-%m-%d", gmtime())
        output_path = os.path.join(base, date, self.output_filetype, "general_plots", filter_spec_path)
        filename = os.path.join(output_path, title + "_" + filter_filename)
        return output_path, filename

    def _construct_path_and_filename_for_filter_spec(self, filter_specifications):
        filter_path = ""
        filter_filename = ""
        for spec in filter_specifications:
            filter_path = os.path.join(filter_path, (spec['parameter'] + "_" + str(spec['value'])))
            filter_filename += spec['parameter'] + "_" + str(spec['value']) + "_"
        filter_filename = filter_filename[:-1] + "." + self.output_filetype
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


class RuntimeBoxplotPlotter(AbstractPlotter):
    def __init__(self,
                 output_path,
                 output_filetype,
                 scenario_solution_storage,
                 algorithm_id,
                 execution_id,
                 metric_specifications=global_metric_specifications,
                 list_of_outer_axes_specifications=boxplot_outer_axes_specifications,
                 list_of_inner_axes_specifications=boxplot_inner_axes_specifications,
                 algorithm_variant_to_be_considered="rr_seplp_ALL",
                 show_plot=False,
                 save_plot=True,
                 overwrite_existing_files=False,
                 forbidden_scenario_ids=None,
                 paper_mode=True
                 ):
        super(RuntimeBoxplotPlotter, self).__init__(output_path, output_filetype, scenario_solution_storage,
                                                    algorithm_id, execution_id, show_plot, save_plot,
                                                    overwrite_existing_files, forbidden_scenario_ids, paper_mode)
        if not metric_specifications:
            raise ValueError("Requires metric specifications")
        self.metric_specifications = metric_specifications
        if not list_of_outer_axes_specifications or not list_of_inner_axes_specifications:
            raise RuntimeError("Axes need to be provided.")
        self.boxplot_outer_axes_specifications = list_of_outer_axes_specifications
        self.boxplot_inner_axes_specifications = list_of_inner_axes_specifications

        self.algorithm_variant_to_be_considered = algorithm_variant_to_be_considered
        self.algorithm_variant_parameters_list = next((
            (params, name) for (params, name) in get_all_rr_settings_list_with_names()
            if name == self.algorithm_variant_to_be_considered
        ))

    def _construct_output_path_and_filename(self, metric_specification,
                                            inner_axis, outer_axis,
                                            filter_specifications=None):
        filter_spec_path = ""
        filter_filename = "no_filter.{}".format(self.output_filetype)
        if filter_specifications:
            filter_spec_path, filter_filename = self._construct_path_and_filename_for_filter_spec(filter_specifications)

        base = os.path.normpath(OUTPUT_PATH)
        date = strftime("%Y-%m-%d", gmtime())
        axes_foldername = "{}__{}".format(outer_axis["x_axis_title_short"], inner_axis["x_axis_title_short"])
        sub_param_string = self.algorithm_variant_to_be_considered

        if sub_param_string is not None:
            output_path = os.path.join(base, date, self.output_filetype, axes_foldername, sub_param_string, filter_spec_path)
        else:
            output_path = os.path.join(base, date, self.output_filetype, axes_foldername, filter_spec_path)

        fname = "{}__{}".format(metric_specification["filename"], filter_filename)
        filename = os.path.join(output_path, fname)
        return output_path, filename

    def plot_figure(self, filter_specifications):
        for outer_axis in self.boxplot_outer_axes_specifications:
            for inner_axis in self.boxplot_inner_axes_specifications:
                for ms in self.metric_specifications:
                    self.plot_single_boxplot_general(ms, outer_axis, inner_axis, filter_specifications)

    def _lookup_solutions(self, scenario_ids):
        solution_dicts = [
            self.scenario_solution_storage.get_solutions_by_scenario_index(x)
            for x in scenario_ids
        ]
        result = [x[self.algorithm_id][self.execution_id] for x in solution_dicts]
        return result

    def plot_single_boxplot_general(self, metric_specification, outer_axis,
                                    inner_axis,
                                    filter_specifications=None):
        # data extraction

        sps = self.scenarioparameter_room
        spd = self.scenario_parameter_dict

        output_path, filename = self._construct_output_path_and_filename(metric_specification,
                                                                         inner_axis, outer_axis,
                                                                         filter_specifications)

        logger.debug("output_path is {};\t filename is {}".format(output_path, filename))
        if not self.overwrite_existing_files and os.path.exists(filename):
            logger.info("Skipping generation of {} as this file already exists".format(filename))
            return
        # check if filter specification conflicts with axes specification
        if filter_specifications is not None:
            for filter_specification in filter_specifications:
                if (inner_axis['x_axis_parameter'] == filter_specification['parameter'] or
                        outer_axis['x_axis_parameter'] == filter_specification['parameter']):
                    logger.debug("Skipping generation of {} as the filter specification conflicts with the axes specification.")
                    return

        path_outer_axis, outer_axis_parameters = extract_parameter_range(
            sps,
            outer_axis['x_axis_parameter'],
        )
        path_inner_axis, inner_axis_parameters = extract_parameter_range(
            sps,
            inner_axis['x_axis_parameter'],
        )

        # for heatmap plot
        outer_axis_parameters.sort()
        inner_axis_parameters.sort()

        # all heatmap values will be stored in X
        data = {
            outer_val: {
                inner_val: [] for inner_val in inner_axis_parameters
            } for outer_val in outer_axis_parameters
        }

        min_number_of_observed_values = 10000000000000
        max_number_of_observed_values = 0
        observed_values = np.empty(0)

        for outer_index, outer_val in enumerate(outer_axis_parameters):
            # all scenario indices which has x_val as xaxis parameter (e.g. node_resource_factor = 0.5
            scenario_ids_matching_x_axis = lookup_scenarios_having_specific_values(spd, path_outer_axis, outer_val)
            for inner_index, inner_val in enumerate(inner_axis_parameters):
                scenario_ids_matching_y_axis = lookup_scenarios_having_specific_values(spd, path_inner_axis, inner_val)

                filter_indices = self._obtain_scenarios_based_on_filters(filter_specifications)
                scenario_ids_to_consider = (scenario_ids_matching_x_axis &
                                            scenario_ids_matching_y_axis &
                                            filter_indices) - self.forbidden_scenario_ids

                solutions = self._lookup_solutions(scenario_ids_to_consider)

                # for solution in solutions:
                #     print solution

                if metric_specification["result_num"] == "single":
                    values = [metric_specification["lookup_function"](solution) for solution in solutions]
                else:
                    values = [value for solution in solutions for value in metric_specification["lookup_function"](solution)]

                observed_values = np.append(observed_values, values)

                if len(values) < min_number_of_observed_values:
                    min_number_of_observed_values = len(values)
                if len(values) > max_number_of_observed_values:
                    max_number_of_observed_values = len(values)

                logger.debug("values are {}".format(values))
                m = np.nanmean(values)
                logger.debug("mean is {}".format(m))

                data[outer_val][inner_val] = values

        if min_number_of_observed_values == max_number_of_observed_values:
            solution_count_string = "{} values per square".format(min_number_of_observed_values)
        else:
            solution_count_string = "between {} and {} values per square".format(min_number_of_observed_values,
                                                                                 max_number_of_observed_values)

        fig, ax = plt.subplots(figsize=(4.5, 3.5))
        if self.paper_mode:
            ax.set_title(metric_specification["name"], fontsize=PLOT_TITLE_FONTSIZE)
        else:
            title = metric_specification["name"] + "\n"
            title += self.algorithm_variant_to_be_considered + "\n"
            if filter_specifications:
                title += get_title_for_filter_specifications(filter_specifications) + "\n"
            title += solution_count_string + "\n"
            title += "min: {:.2f}; mean: {:.2f}; max: {:.2f}".format(np.nanmin(observed_values),
                                                                     np.nanmean(observed_values),
                                                                     np.nanmax(observed_values))

            ax.set_title(title)

        # group data
        bins = []
        positions = []
        labels = {}
        x_ticks = []
        colors = []
        pos = 0
        cmap = plt.get_cmap("inferno")
        for outer_index, outer_val in enumerate(outer_axis_parameters):
            group_start_pos = pos
            for inner_index, inner_val in enumerate(inner_axis_parameters):
                bins.append(data[outer_val][inner_val])
                positions.append(pos)
                pos += 0.6
                color = cmap(1.0 - (1 + float(inner_index)) / len(inner_axis_parameters))
                colors.append(color)
                labels[inner_val] = (str(inner_val), color)
            x_ticks.append(group_start_pos + 0.5 * (pos - group_start_pos) - 0.5 * BOX_SEPARATION_WITHIN_GROUP)
            pos += BOX_SEPARATION_BETWEEN_GROUPS

        bplots = []
        for _bin, pos in zip(bins, positions):
            bplots.append(ax.boxplot(
                x=_bin,
                positions=[pos],
                widths=[BOX_WIDTH],
                patch_artist=True,
                notch=True,
                bootstrap=1000,
            ))
        for bplot, color in zip(
                bplots,
                colors,
        ):
            for patch in itertools.chain(bplot['boxes']):
                patch.set_edgecolor(color)
                patch.set_facecolor(
                    matplotlib.colors.to_rgba(color, alpha=0.3)
                )
            for line in itertools.chain(
                    bplot['medians'],
                    bplot['fliers'],
                    bplot['whiskers'],
                    bplot['caps'],
            ):
                line.set_color(color)
            for marker in bplot['fliers']:
                marker.set(
                    marker='o',
                    markeredgecolor=matplotlib.colors.to_rgba(color, alpha=0.8),
                )

        legend_handles = [
            mpatches.Patch(color=matplotlib.colors.to_rgba(color, alpha=0.6), label=label)
            for (val, (label, color)) in sorted(labels.items())
        ]

        #fig.subplots_adjust(top=0.85)
        #fig.subplots_adjust(bottom=0.125)
        fig.subplots_adjust(right=0.8)
        #fig.subplots_adjust(hspace=0.3)
        #fig.subplots_adjust(left=0.15)

        legend = plt.legend(handles=legend_handles,
                            title=inner_axis["x_axis_title_short"],
                            loc='center right',
                            fontsize=LEGEND_LABEL_FONTSIZE,
                            handletextpad=0.35, bbox_to_anchor=(0.99, 0.5), bbox_transform=plt.gcf().transFigure,
                            borderaxespad=-0.175, borderpad=0.2,
                            handlelength=0.35)
        legend.get_frame().set_alpha(1.0)
        legend.get_frame().set_facecolor("#FFFFFF")
        plt.setp(legend.get_title(), fontsize=LEGEND_TITLE_FONTSIZE)
        plt.gca().add_artist(legend)

        ax.set_xlim(min(positions) - 0.5, max(positions) + 0.5)
        ax.set_xticks(x_ticks, minor=False)
        ax.set_xticklabels(outer_axis_parameters, minor=False, fontsize=AXIS_TICKLABEL_FONTSIZE)

        if not "non_log" in metric_specification.keys():
            ax.set_yscale("log", nonposy='clip')

        ax.yaxis.grid(True, which="major", linestyle="-")
        ax.yaxis.grid(True, which="minor", linestyle=":", linewidth=0.8, alpha=0.4)

        for label in ax.get_yticklabels():
            plt.setp(label, fontsize=AXIS_TICKLABEL_FONTSIZE)
        ax.set_xlabel(outer_axis['x_axis_title'], fontsize=AXIS_LABEL_FONTSIZE)
        ax.set_ylabel(metric_specification["y_axis_title"], fontsize=AXIS_LABEL_FONTSIZE)

        # plt.tight_layout()

        # if "additional_hlines_at" in metric_specification:
        #     for y in metric_specification["additional_hlines_at"]:
        #         ax.axhline(y, linestyle=':', color='gray', alpha=0.4, linewidth=0.8)

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


def evaluate_randround_runtimes(dc_randround,
                                randround_algorithm_id,
                                randround_execution_id,
                                exclude_generation_parameters=None,
                                parameter_filter_keys=None,
                                forbidden_scenario_ids=None,
                                show_plot=False,
                                save_plot=True,
                                overwrite_existing_files=True,
                                papermode=True,
                                maxdepthfilter=2,
                                output_path="./",
                                output_filetype="png"):
    if forbidden_scenario_ids is None:
        forbidden_scenario_ids = set()

    if exclude_generation_parameters is not None:
        for key, values_to_exclude in exclude_generation_parameters.iteritems():
            parameter_filter_path, parameter_values = extract_parameter_range(
                dc_randround.scenario_parameter_container.scenarioparameter_room, key)

            parameter_dicts_vine = lookup_scenario_parameter_room_dicts_on_path(
                dc_randround.scenario_parameter_container.scenarioparameter_room, parameter_filter_path)
            parameter_dicts_randround = lookup_scenario_parameter_room_dicts_on_path(
                dc_randround.scenario_parameter_container.scenarioparameter_room, parameter_filter_path)

            for value_to_exclude in values_to_exclude:

                if value_to_exclude not in parameter_values:
                    raise RuntimeError("The value {} is not contained in the list of parameter values {} for key {}".format(
                        value_to_exclude, parameter_values, key
                    ))

                # add respective scenario ids to the set of forbidden scenario ids
                forbidden_scenario_ids.update(set(lookup_scenarios_having_specific_values(
                    dc_randround.scenario_parameter_container.scenario_parameter_dict, parameter_filter_path, value_to_exclude)))

            # remove the respective values from the scenario parameter room such that these are not considered when
            # constructing e.g. axes
            parameter_dicts_vine[-1][key] = [value for value in parameter_dicts_vine[-1][key] if
                                             value not in values_to_exclude]
            parameter_dicts_randround[-1][key] = [value for value in parameter_dicts_randround[-1][key] if
                                                  value not in values_to_exclude]

    if parameter_filter_keys is not None:
        filter_specs = _construct_filter_specs(dc_randround.scenario_parameter_container.scenarioparameter_room,
                                               parameter_filter_keys,
                                               maxdepth=maxdepthfilter)
    else:
        filter_specs = [None]

    plotters = []

    boxplotter_plotter = RuntimeBoxplotPlotter(
        output_path=output_path,
        output_filetype=output_filetype,
        scenario_solution_storage=dc_randround,
        algorithm_id=randround_algorithm_id,
        execution_id=randround_execution_id,
        show_plot=show_plot,
        save_plot=save_plot,
        overwrite_existing_files=overwrite_existing_files,
        paper_mode=papermode,
    )

    plotters.append(boxplotter_plotter)

    for filter_spec in filter_specs:
        for plotter in plotters:
            plotter.plot_figure(filter_spec)
