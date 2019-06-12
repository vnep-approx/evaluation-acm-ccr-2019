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
import sys
from collections import namedtuple
from itertools import combinations, product
from time import gmtime, strftime
import copy

try:
    import cPickle as pickle
except ImportError:
    import pickle

import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
import matplotlib.patheffects as PathEffects
import matplotlib.patches as mpatches
from matplotlib import gridspec
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
FIGSIZE = (5,3.5)

logger = util.get_logger(__name__, make_file=False, propagate=True)


class HeatmapPlotType(object):
    ViNE = 0  # a plot only for OfflineViNEResult data
    RandRoundSepLPDynVMP = 1  # a plot only for RandRoundSepLPOptDynVMPCollectionResult data
    SeparationLP = 2  # a plot only for SeparationLPSolution data
    ComparisonVineRandRound = 3
    VALUE_RANGE = [0, 1, 2, 3]


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


def get_list_of_vine_settings():
    result = []
    for (edge_embedding_model, lp_objective, rounding_procedure) in itertools.product(
            vine.ViNEEdgeEmbeddingModel,
            vine.ViNELPObjective,
            vine.ViNERoundingProcedure,
    ):
        if lp_objective == vine.ViNELPObjective.ViNE_LB_INCL_SCENARIO_COSTS or lp_objective == vine.ViNELPObjective.ViNE_COSTS_INCL_SCENARIO_COSTS:
            continue
        if edge_embedding_model == vine.ViNEEdgeEmbeddingModel.SPLITTABLE:
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

class AbstractHeatmapSpecificationVineFactory(object):

    prototype = dict()

    @classmethod
    def get_hs(cls, vine_settings_list, name):
        result = copy.deepcopy(cls.prototype)
        result['lookup_function'] = lambda x: cls.prototype['lookup_function'](x, vine_settings_list)
        result['alg_variant'] = name
        return result

    @classmethod
    def get_specific_vine_name(cls, vine_settings):
        vine.ViNESettingsFactory.check_vine_settings(vine_settings)
        is_splittable = vine_settings.edge_embedding_model == vine.ViNEEdgeEmbeddingModel.SPLITTABLE
        is_load_balanced_objective = (
                vine_settings.lp_objective in
                [vine.ViNELPObjective.ViNE_LB_DEF, vine.ViNELPObjective.ViNE_LB_INCL_SCENARIO_COSTS]
        )
        is_scenario_cost_objective = (
                vine_settings.lp_objective in
                [vine.ViNELPObjective.ViNE_LB_INCL_SCENARIO_COSTS, vine.ViNELPObjective.ViNE_COSTS_INCL_SCENARIO_COSTS]
        )
        is_random_rounding_procedure = vine_settings.rounding_procedure == vine.ViNERoundingProcedure.RANDOMIZED
        return "vine_{}_{}_{}_{}".format(
            "mcf" if is_splittable else "sp",
            "lb" if is_load_balanced_objective else "cost",
            "scenario" if is_scenario_cost_objective else "def",
            "rand" if is_random_rounding_procedure else "det",
        )

    @classmethod
    def get_all_vine_settings_list_with_names(cls):
        result = []

        vine_settings_list = get_list_of_vine_settings()
        result.append((vine_settings_list, "vine_ALL")) #first off: every vine combination

        # second: each specific one
        for vine_settings in vine_settings_list:
            result.append(([vine_settings], cls.get_specific_vine_name(vine_settings)))

        #third: each aggregation level, when applicable, i.e. there is more than one setting for that
        for edge_embedding_model in vine.ViNEEdgeEmbeddingModel:
            matching_settings = []
            for vine_settings in vine_settings_list:
                if vine_settings.edge_embedding_model == edge_embedding_model:
                    matching_settings.append(vine_settings)
            if len(matching_settings) > 0 and len(matching_settings) != len(vine_settings_list):
                result.append((matching_settings, "vine_{}".format(
                    "MCF" if edge_embedding_model is vine.ViNEEdgeEmbeddingModel.SPLITTABLE else "SP")))

        for lp_objective in vine.ViNELPObjective:
            matching_settings = []
            for vine_settings in vine_settings_list:
                if vine_settings.lp_objective == lp_objective:
                    matching_settings.append(vine_settings)
            if len(matching_settings) > 0 and len(matching_settings) != len(vine_settings_list):
                is_load_balanced_objective = (
                        vine_settings.lp_objective in
                        [vine.ViNELPObjective.ViNE_LB_DEF, vine.ViNELPObjective.ViNE_LB_INCL_SCENARIO_COSTS]
                )
                is_scenario_cost_objective = (
                        vine_settings.lp_objective in
                        [vine.ViNELPObjective.ViNE_LB_INCL_SCENARIO_COSTS,
                         vine.ViNELPObjective.ViNE_COSTS_INCL_SCENARIO_COSTS]
                )
                result.append((matching_settings, "vine_{}_{}".format(
                    "LB" if is_load_balanced_objective else "COST",
                    "SCENARIO" if is_scenario_cost_objective else "DEF"
                )))

        for rounding_proc in vine.ViNERoundingProcedure:
            matching_settings = []
            for vine_settings in vine_settings_list:
                if vine_settings.rounding_procedure == rounding_proc:
                    matching_settings.append(vine_settings)
            if len(matching_settings) > 0 and len(matching_settings) != len(vine_settings_list):
                result.append((matching_settings, "vine_{}".format(
                    "RAND" if rounding_proc is vine.ViNERoundingProcedure.RANDOMIZED else "DET")))

        return result

    @classmethod
    def get_all_hs(cls):
        return [cls.get_hs(vine_settings_list, name) for vine_settings_list, name in cls.get_all_vine_settings_list_with_names()]


def compute_aggregated_mean(list_of_aggregated_data, debug=False):
    mean = 0.0
    value_count = 0
    for agg in list_of_aggregated_data:
        mean += agg.mean * agg.value_count
        value_count += agg.value_count
    if debug:
        print len(list_of_aggregated_data), value_count, mean/value_count
    return mean / value_count



class HSF_Vine_Runtime(AbstractHeatmapSpecificationVineFactory):

    prototype = dict(
        name="ViNE: Mean Runtime [s]",
        filename="vine_mean_runtime",
        vmin=0,
        vmax=20,
        alg_variant=None,
        colorbar_ticks=[x for x in range(0, 21, 4)],
        cmap="Greys",
        plot_type=HeatmapPlotType.ViNE,
        lookup_function=lambda vine_result_dict, vine_settings_list: compute_aggregated_mean([
            vine_result.total_runtime
            for vine_settings in vine_settings_list
            for vine_result in vine_result_dict[vine_settings]
        ]),
        rounding_function=lambda x: int(round(x)),
    )


# class HSF_Vine_MaxNodeLoad(AbstractHeatmapSpecificationVineFactory):
#     prototype = dict(
#         name="ViNE: Max. Node Load [%]",
#         filename="max_node_load",
#         vmin=0.0,
#         vmax=100,
#         colorbar_ticks=[x for x in range(0, 101, 20)],
#         cmap="Oranges",
#         plot_type=HeatmapPlotType.ViNE,
#         lookup_function=lambda vine_result_dict, vine_settings_list: max(
#             vine_result.max_node_load.max
#             for vine_settings in vine_settings_list
#             for vine_result in vine_result_dict[vine_settings]
#         )
#     )
#
# class HSF_Vine_MaxEdgeLoad(AbstractHeatmapSpecificationVineFactory):
#
#     prototype = dict(
#         name="ViNE: Max. Edge Load [%]",
#         filename="max_edge_load",
#         vmin=0.0,
#         vmax=100,
#         colorbar_ticks=[x for x in range(0, 101, 20)],
#         cmap="Purples",
#         plot_type=HeatmapPlotType.ViNE,
#         lookup_function=lambda vine_result_dict, vine_settings_list: max(
#             vine_result.max_edge_load.max
#             for vine_settings in vine_settings_list
#             for vine_result in vine_result_dict[vine_settings]
#         )
#     )
#
# class HSF_Vine_MaxLoad(AbstractHeatmapSpecificationVineFactory):
#
#     prototype = dict(
#         name="ViNE: MaxLoad (Edge and Node)",
#         filename="max_load",
#         vmin=0.0,
#         vmax=100,
#         colorbar_ticks=[x for x in range(0, 101, 20)],
#         cmap="Reds",
#         plot_type=HeatmapPlotType.ViNE,
#         lookup_function=lambda vine_result_dict, vine_settings_list: max(
#             max(vine_result.max_node_load.max, vine_result.max_edge_load.max)
#             for vine_settings in vine_settings_list
#             for vine_result in vine_result_dict[vine_settings]
#         )
#     )

class AbstractHeatmapSpecificationSepLPRRFactory(object):

    prototype = dict()

    @classmethod
    def get_hs(cls, rr_settings, name):
        result = copy.deepcopy(cls.prototype)
        result['lookup_function'] = lambda x: cls.prototype['lookup_function'](x, rr_settings)
        result['alg_variant'] = name
        return result

    @classmethod
    def _get_lp_str(cls, lp_mode):
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

    @classmethod
    def _get_rounding_str(cls, rounding_mode):
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


    @classmethod
    def get_specific_rr_name(cls, rr_settings):

        return "rr_seplp_{}__{}".format(
            cls._get_lp_str(rr_settings[0]),
            cls._get_rounding_str(rr_settings[1]),
        )

    @classmethod
    def get_all_rr_settings_list_with_names(cls):
        result = []

        rr_settings_list = get_list_of_rr_settings()
        result.append((rr_settings_list, "rr_seplp_ALL")) #first off: every vine combination

        # second: each specific one
        for rr_settings in rr_settings_list:
            result.append(([rr_settings], cls.get_specific_rr_name(rr_settings)))

        # third: each aggregation level, when applicable, i.e. there is more than one setting for that
        for lp_mode in treewidth_model.LPRecomputationMode:
            matching_settings = []
            for rr_settings in rr_settings_list:
                if rr_settings[0] == lp_mode:
                    matching_settings.append(rr_settings)
            if len(matching_settings) > 0 and len(matching_settings) != len(rr_settings_list):
                result.append((matching_settings, "rr_seplp_{}".format(
                    cls._get_lp_str(lp_mode).upper())))

        for rounding_mode in treewidth_model.RoundingOrder:
            matching_settings = []
            for rr_settings in rr_settings_list:
                if rr_settings[1] == rounding_mode:
                    matching_settings.append(rr_settings)
            if len(matching_settings) > 0 and len(matching_settings) != len(rr_settings_list):
                result.append((matching_settings, "rr_seplp_{}".format(
                    cls._get_rounding_str(rounding_mode).upper()
                )))

        return result

    @classmethod
    def get_all_hs(cls):
        return [cls.get_hs(rr_settings, name) for rr_settings, name in cls.get_all_rr_settings_list_with_names()]

# class HSF_RR_MaxNodeLoad(AbstractHeatmapSpecificationSepLPRRFactory):
#     prototype = dict(
#         name="RR: Max node load",
#         filename="randround_max_node_load",
#         vmin=0.0,
#         vmax=100,
#         colorbar_ticks=[x for x in range(0, 101, 20)],
#         cmap="Reds",
#         plot_type=HeatmapPlotType.RandRoundSepLPDynVMP,
#         lookup_function=lambda rr_seplp_result, rr_seplp_settings_list: 100.0 * np.mean([value for rr_seplp_settings in rr_seplp_settings_list for value in rr_seplp_result.max_node_loads[rr_seplp_settings]])
#     )
#
# class HSF_RR_MaxEdgeLoad(AbstractHeatmapSpecificationSepLPRRFactory):
#     prototype = dict(
#         name="RR: Max edge load",
#         filename="randround_max_edge_load",
#         vmin=0.0,
#         vmax=100,
#         colorbar_ticks=[x for x in range(0, 101, 20)],
#         cmap="Reds",
#         plot_type=HeatmapPlotType.RandRoundSepLPDynVMP,
#         lookup_function=lambda rr_seplp_result, rr_seplp_settings_list: 100.0 * np.mean([value for rr_seplp_settings in rr_seplp_settings_list for value in rr_seplp_result.max_edge_loads[rr_seplp_settings]])
#     )
#
# class HSF_RR_MeanProfit(AbstractHeatmapSpecificationSepLPRRFactory):
#     prototype = dict(
#         name="RR: Mean Profit",
#         filename="randround_mean_profit",
#         vmin=0.0,
#         vmax=100,
#         colorbar_ticks=[x for x in range(0, 101, 20)],
#         cmap="Reds",
#         plot_type=HeatmapPlotType.RandRoundSepLPDynVMP,
#         lookup_function=lambda rr_seplp_result, rr_seplp_settings_list: np.mean([value for rr_seplp_settings in rr_seplp_settings_list for value in rr_seplp_result.profits[rr_seplp_settings]])
#     )

class HSF_RR_MeanRoundingRuntime(AbstractHeatmapSpecificationSepLPRRFactory):
    prototype = dict(
        name="RR: Mean Rounding Runtime",
        filename="randround_mean_profit",
        vmin=0.0,
        vmax=200,
        colorbar_ticks=[x for x in range(0, 201, 40)],
        cmap="Reds",
        plot_type=HeatmapPlotType.RandRoundSepLPDynVMP,
        lookup_function=lambda rr_seplp_result, rr_seplp_settings_list: np.mean([rr_seplp_result.rounding_runtimes[rr_seplp_settings].mean for rr_seplp_settings in rr_seplp_settings_list])
    )

class HSF_RR_MeanDynVMPInitTimes(AbstractHeatmapSpecificationSepLPRRFactory):
    prototype = dict(
        name="RR: Mean DynVMP Initialization Runtimes",
        filename="randround_mean_dynvmp_initialization",
        vmin=0.0,
        vmax=50,
        colorbar_ticks=[x for x in range(0, 51, 10)],
        cmap="Reds",
        plot_type=HeatmapPlotType.RandRoundSepLPDynVMP,
        lookup_function=lambda rr_seplp_result, rr_seplp_settings_list: rr_seplp_result.lp_time_dynvmp_initialization.mean * rr_seplp_result.lp_time_dynvmp_initialization.value_count
    )

    @classmethod
    def get_all_rr_settings_list_with_names(cls):
        result = []

        rr_settings_list = get_list_of_vine_settings()
        result.append(([rr_settings_list[0]], "rr_seplp_ALL"))  # select arbitrary rr_settings to derive plots from

        return result

class HSF_RR_LP_Runtime(AbstractHeatmapSpecificationSepLPRRFactory):
    prototype = dict(
        name="RR: LP runtime",
        filename="randround_lp_runtime",
        vmin=0.0,
        vmax=100,
        colorbar_ticks=[x for x in range(0, 101, 20)],
        cmap="Blues",
        plot_type=HeatmapPlotType.RandRoundSepLPDynVMP,
        lookup_function=lambda rr_seplp_result, rr_seplp_settings_list: rr_seplp_result.lp_time_optimization + rr_seplp_result.lp_time_preprocess
    )

    @classmethod
    def get_all_rr_settings_list_with_names(cls):
        result = []

        rr_settings_list = get_list_of_vine_settings()
        result.append(([rr_settings_list[0]], "rr_seplp_ALL"))  # select arbitrary rr_settings to derive plots from

        return result


class HSF_RR_Runtime(AbstractHeatmapSpecificationSepLPRRFactory):
    prototype = dict(
        name="RR: Rounding Runtime",
        filename="randround_rounding_runtime",
        vmin=0.0,
        vmax=100,
        colorbar_ticks=[x for x in range(0, 101, 20)],
        cmap="Blues",
        plot_type=HeatmapPlotType.RandRoundSepLPDynVMP,
        lookup_function=lambda rr_seplp_result, rr_seplp_settings_list: np.mean([rr_seplp_result.rounding_runtimes[rr_settings].mean for rr_settings in rr_seplp_settings_list])
    )

class HSF_RR_GeneratedMappings(AbstractHeatmapSpecificationSepLPRRFactory):
    prototype = dict(
        name="Generated Mappings [k]",
        filename="lp_generated_mappings",
        vmin=0.0,
        vmax=25,
        colorbar_ticks=[x for x in range(0, 26, 5)],
        cmap="Greens",
        plot_type=HeatmapPlotType.RandRoundSepLPDynVMP,
        lookup_function=lambda rr_seplp_result, rr_seplp_settings_list: rr_seplp_result.lp_generated_columns / 1000.0
    )

    @classmethod
    def get_all_rr_settings_list_with_names(cls):
        result = []

        rr_settings_list = get_list_of_vine_settings()
        result.append(([rr_settings_list[0]], "rr_seplp_ALL"))  # select arbitrary rr_settings to derive plots from

        return result


class AbstractHeatmapSpecificationVineVsRandRoundFactory(object):

    prototype = dict()

    @classmethod
    def get_hs(cls, vine_settings_list, randround_settings_list, name):
        result = copy.deepcopy(cls.prototype)
        result['lookup_function'] = lambda x: cls.prototype['lookup_function'](x[0], x[1], vine_settings_list, randround_settings_list)
        result['alg_variant'] = name
        return result

    # @classmethod
    # def get_specific_vine_name(cls, vine_settings):
    #     vine.ViNESettingsFactory.check_vine_settings(vine_settings)
    #     is_splittable = vine_settings.edge_embedding_model == vine.ViNEEdgeEmbeddingModel.SPLITTABLE
    #     is_load_balanced_objective = (
    #             vine_settings.lp_objective in
    #             [vine.ViNELPObjective.ViNE_LB_DEF, vine.ViNELPObjective.ViNE_LB_INCL_SCENARIO_COSTS]
    #     )
    #     is_scenario_cost_objective = (
    #             vine_settings.lp_objective in
    #             [vine.ViNELPObjective.ViNE_LB_INCL_SCENARIO_COSTS, vine.ViNELPObjective.ViNE_COSTS_INCL_SCENARIO_COSTS]
    #     )
    #     is_random_rounding_procedure = vine_settings.rounding_procedure == vine.ViNERoundingProcedure.RANDOMIZED
    #     return "vine_{}_{}_{}_{}".format(
    #         "mcf" if is_splittable else "sp",
    #         "lb" if is_load_balanced_objective else "cost",
    #         "scenario" if is_scenario_cost_objective else "def",
    #         "rand" if is_random_rounding_procedure else "det",
    #     )

    @classmethod
    def get_specific_comparison_settings_list_with_names(cls):
        result = []

        vine_settings_list = get_list_of_vine_settings()

        rr_settings_list = get_list_of_rr_settings()

        result.append((vine_settings_list, rr_settings_list, "vine_ALL_vs_randround_ALL"))

        vine_settings_list_mcf = []
        vine_settings_list_sp = []

        for vine_settings in vine_settings_list:
            if vine_settings.edge_embedding_model == vine.ViNEEdgeEmbeddingModel.SPLITTABLE:
                vine_settings_list_mcf.append(vine_settings)
            else:
                vine_settings_list_sp.append(vine_settings)

        result.append((vine_settings_list_sp, rr_settings_list, "vine_SP_vs_randround_ALL"))
        #result.append((vine_settings_list_mcf, rr_settings_list, "vine_MCF_vs_randround_ALL"))

        return result

    @classmethod
    def get_all_hs(cls):
        return [cls.get_hs(vine_settings_list, rr_settings_list, name) for vine_settings_list, rr_settings_list, name in cls.get_specific_comparison_settings_list_with_names()]


def _comparison_profit_best_relative(vine_result, rr_result, vine_settings_list, rr_settings_list):
    # print vine_result
    # print rr_result
    # print vine_settings_list
    # print rr_settings_list
    best_vine = max([vine_result[vine_settings][0].profit.max for vine_settings in vine_settings_list])
    best_rr = max([rr_result.profits[rr_settings].max for rr_settings in rr_settings_list])
    return 100*(best_rr - best_vine) / best_vine


def _comparison_profit_absolute(vine_result, rr_result, vine_settings_list, rr_settings_list):
    best_vine = max([vine_result[vine_settings][0].profit.max for vine_settings in vine_settings_list])
    best_rr = max([rr_result.profits[rr_settings].max for rr_settings in rr_settings_list])
    return best_rr - best_vine

def _comparison_profit_qualitative_randround_5perc(vine_result, rr_result, vine_settings_list, rr_settings_list):
    best_vine = max([vine_result[vine_settings][0].profit.max for vine_settings in vine_settings_list])
    best_rr = max([rr_result.profits[rr_settings].max for rr_settings in rr_settings_list])
    if (best_rr - best_vine)/ best_vine >= 0.05:
        return 100
    else:
        return 0

def _comparison_profit_qualitative_vine_5perc(vine_result, rr_result, vine_settings_list, rr_settings_list):
    best_vine = max([vine_result[vine_settings][0].profit.max for vine_settings in vine_settings_list])
    best_rr = max([rr_result.profits[rr_settings].max for rr_settings in rr_settings_list])
    if (best_vine - best_rr)/ best_rr >= 0.05:
        return 100
    else:
        return 0

def _profit_relative_to_lp_bound_rr(rr_result, rr_settings_list):
    best_rr = max([rr_result.profits[rr_settings].max for rr_settings in rr_settings_list])
    lp_bound = rr_result.lp_profit
    return 100.0*(best_rr / lp_bound)

def _profit_relative_to_lp_bound_vine(vine_result, rr_result, vine_settings_list, rr_settings_list):
    best_vine = max([vine_result[vine_settings][0].profit.max for vine_settings in vine_settings_list])
    lp_bound = rr_result.lp_profit
    return 100.0*(best_vine / lp_bound)


def _relative_profit_difference_to_lp_bound(vine_result, rr_result, vine_settings_list, rr_settings_list):
    best_rr = max([rr_result.profits[rr_settings].max for rr_settings in rr_settings_list])
    best_vine = max([vine_result[vine_settings][0].profit.max for vine_settings in vine_settings_list])
    lp_bound = rr_result.lp_profit
    return 100.0*(best_rr / lp_bound) - 100.0*(best_vine / lp_bound)


class HSF_Comp_BestProfit(AbstractHeatmapSpecificationVineVsRandRoundFactory):

    prototype = dict(
        name="Relative Profit: rand round vs ViNE",
        filename="comparison_vine_rand_round",
        vmin=-100,
        vmax=+100,
        colorbar_ticks=[x for x in range(-100, 101, 33)],
        cmap="Reds",
        plot_type=HeatmapPlotType.ComparisonVineRandRound,
        lookup_function=lambda vine_result, rr_result, vine_settings_list, rr_settings_list : _comparison_profit_best_relative(vine_result,
                                                                                                                               rr_result,
                                                                                                                               vine_settings_list,
                                                                                                                               rr_settings_list)
    )

class HSF_Comp_QualProfitDiff_RR(AbstractHeatmapSpecificationVineVsRandRoundFactory):

    prototype = dict(
        name="Qualitative Difference > 5%: Rand Round",
        filename="qual_diff_5perc_rand_round",
        vmin=0,
        vmax=+100,
        colorbar_ticks=[x for x in range(0, 101, 20)],
        cmap="Reds",
        plot_type=HeatmapPlotType.ComparisonVineRandRound,
        lookup_function=lambda vine_result, rr_result, vine_settings_list, rr_settings_list : _comparison_profit_qualitative_randround_5perc(vine_result,
                                                                                                                               rr_result,
                                                                                                                               vine_settings_list,
                                                                                                                               rr_settings_list)
    )

class HSF_Comp_QualProfitDiff_Vine(AbstractHeatmapSpecificationVineVsRandRoundFactory):

    prototype = dict(
        name="Qualitative Difference > 5%: ViNE",
        filename="qual_diff_5perc_vine",
        vmin=0,
        vmax=+100,
        colorbar_ticks=[x for x in range(0, 101, 20)],
        cmap="Reds",
        plot_type=HeatmapPlotType.ComparisonVineRandRound,
        lookup_function=lambda vine_result, rr_result, vine_settings_list, rr_settings_list : _comparison_profit_qualitative_vine_5perc(vine_result,
                                                                                                                                        rr_result,
                                                                                                                                        vine_settings_list,
                                                                                                                                        rr_settings_list)
    )


class HSF_Comp_RelProfitToLPBound_RR(AbstractHeatmapSpecificationVineVsRandRoundFactory):

    prototype = dict(
        name="Rel. Profit: Rand Round",
        filename="rel_profit_lpbound_rr",
        vmin=0,
        vmax=+100,
        colorbar_ticks=[x for x in range(0, 101, 20)],
        cmap="Reds",
        plot_type=HeatmapPlotType.ComparisonVineRandRound,
        lookup_function=lambda vine_result, rr_result, vine_settings_list, rr_settings_list : _profit_relative_to_lp_bound_rr(rr_result,
                                                                                                                              rr_settings_list)
    )

class HSF_Comp_RelProfitToLPBound_Vine(AbstractHeatmapSpecificationVineVsRandRoundFactory):

    prototype = dict(
        name="Rel. Profit: WiNE",
        filename="rel_profit_lpbound_vine",
        vmin=0,
        vmax=+100,
        colorbar_ticks=[x for x in range(0, 101, 20)],
        cmap="Reds",
        plot_type=HeatmapPlotType.ComparisonVineRandRound,
        lookup_function=lambda vine_result, rr_result, vine_settings_list, rr_settings_list : _profit_relative_to_lp_bound_vine(vine_result,
                                                                                                                                rr_result,
                                                                                                                                vine_settings_list,
                                                                                                                                rr_settings_list)
    )

class HSF_Comp_RelProfitToLPBound_RR_minus_Vine(AbstractHeatmapSpecificationVineVsRandRoundFactory):

    prototype = dict(
        name="Rel. Improv.: ($\mathsf{RR}_{\mathsf{best}}$ - $\mathsf{WiNE}_{\mathsf{best}}$)/$\mathsf{LP}_{\mathsf{UB}}$ [%]",
        filename="rel_profit_difference_lpbound",
        vmin=-25,
        vmax=+25,
        colorbar_ticks=[x for x in range(-24, 25, 6)],
        cmap="RdBu_r",
        plot_type=HeatmapPlotType.ComparisonVineRandRound,
        lookup_function=lambda vine_result, rr_result, vine_settings_list, rr_settings_list : _relative_profit_difference_to_lp_bound(vine_result,
                                                                                                                                      rr_result,
                                                                                                                                      vine_settings_list,
                                                                                                                                      rr_settings_list)
    )





global_heatmap_specfications = HSF_Vine_Runtime.get_all_hs() + \
                               HSF_RR_MeanRoundingRuntime.get_all_hs() + \
                               HSF_RR_MeanDynVMPInitTimes.get_all_hs() + \
                               HSF_RR_GeneratedMappings.get_all_hs() + \
                               HSF_RR_Runtime.get_all_hs() + \
                               HSF_RR_LP_Runtime.get_all_hs() + \
                               HSF_Comp_BestProfit.get_all_hs() + \
                               HSF_Comp_QualProfitDiff_RR.get_all_hs() + \
                               HSF_Comp_QualProfitDiff_Vine.get_all_hs() + \
                               HSF_Comp_RelProfitToLPBound_RR.get_all_hs() + \
                               HSF_Comp_RelProfitToLPBound_Vine.get_all_hs() + \
                               HSF_Comp_RelProfitToLPBound_RR_minus_Vine.get_all_hs()

heatmap_specifications_per_type = {
    plot_type_item: [
        heatmap_specification for heatmap_specification in global_heatmap_specfications
        if heatmap_specification['plot_type'] == plot_type_item
    ]
    for plot_type_item in [HeatmapPlotType.ViNE,
                           HeatmapPlotType.RandRoundSepLPDynVMP,
                           HeatmapPlotType.ComparisonVineRandRound]
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

heatmap_axes_specification_treewidth_edge_rf = dict(
    x_axis_parameter="treewidth",
    y_axis_parameter="edge_resource_factor",
    x_axis_title="Treewidth",
    y_axis_title="Ede Resource Factor",
    foldername="AXES_TREEWIDTH_vs_EDGE_RF"
)


global_heatmap_axes_specifications = (
    heatmap_axes_specification_requests_edge_load,
    heatmap_axes_specification_requests_treewidth,
    heatmap_axes_specification_resources,
    heatmap_axes_specification_requests_node_load,
    heatmap_axes_specification_treewidth_edge_rf,
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
        filter_filename = "no_filter.{}".format(self.output_filetype)
        if filter_specifications:
            filter_spec_path, filter_filename = self._construct_path_and_filename_for_filter_spec(filter_specifications)
        base = os.path.normpath(self.output_path)
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

    def _show_and_or_save_plots(self, output_path, filename, perform_tight_layout=True):
        if perform_tight_layout:
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


    def _construct_output_path_and_filename(self, metric_specification,
                                            heatmap_axes_specification,
                                            filter_specifications=None):
        filter_spec_path = ""
        filter_filename = "no_filter.{}".format(self.output_filetype)
        if filter_specifications:
            filter_spec_path, filter_filename = self._construct_path_and_filename_for_filter_spec(filter_specifications)

        base = os.path.normpath(self.output_path)
        date = strftime("%Y-%m-%d", gmtime())
        axes_foldername = heatmap_axes_specification['foldername']
        sub_param_string = metric_specification['alg_variant']

        if sub_param_string is not None:
            output_path = os.path.join(base, date, self.output_filetype, axes_foldername, sub_param_string, filter_spec_path)
        else:
            output_path = os.path.join(base, date, self.output_filetype, axes_foldername, filter_spec_path)

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
        #todo check whether this is okay...
        # if self.heatmap_plot_type == HeatmapPlotType.ViNE:
        #     # result should be a list of dicts mapping vine_settings to lists of ReducedOfflineViNEResultCollection instances
        #     if result and self.algorithm_sub_parameter not in result[0]:
        #         return None
        # elif self.heatmap_plot_type == HeatmapPlotType.RandRoundSepLPDynVMP:
        #     # result should be a list of ReducedRandRoundSepLPOptDynVMPCollectionResult instances
        #     if result and self.algorithm_sub_parameter not in result[0].profits:
        #         return None
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
        )
        path_y_axis, yaxis_parameters = extract_parameter_range(
            sps,
            heatmap_axes_specification['y_axis_parameter'],
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

                # for solution in solutions:
                #     print solution

                values = [heatmap_metric_specification['lookup_function'](solution)
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

        fig, ax = plt.subplots(figsize=FIGSIZE)
        if self.paper_mode:
            ax.set_title(heatmap_metric_specification['name'], fontsize=17)
        else:
            title = heatmap_metric_specification['name'] + "\n"
            title += heatmap_metric_specification['alg_variant'] + "\n"
            if filter_specifications:
                title += get_title_for_filter_specifications(filter_specifications) + "\n"
            title += solution_count_string + "\n"
            title += "min: {:.4f}; mean: {:.4f}; max: {:.4f}".format(np.nanmin(observed_values),
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



class ComparisonHeatmapPlotter(SingleHeatmapPlotter):

    def __init__(self,
                 output_path,
                 output_filetype,
                 vine_solution_storage,
                 vine_algorithm_id,
                 vine_execution_id,
                 randround_scenario_solution_storage,
                 randround_algorithm_id,
                 randround_execution_id,
                 heatmap_plot_type,
                 list_of_axes_specifications = global_heatmap_axes_specifications,
                 list_of_metric_specifications = None,
                 show_plot=False,
                 save_plot=True,
                 overwrite_existing_files=False,
                 forbidden_scenario_ids=None,
                 paper_mode=True
                 ):
        super(ComparisonHeatmapPlotter, self).__init__(output_path,
                                                       output_filetype,
                                                       vine_solution_storage,
                                                       vine_algorithm_id,
                                                       vine_execution_id,
                                                       heatmap_plot_type,
                                                       list_of_axes_specifications,
                                                       list_of_metric_specifications,
                                                       show_plot,
                                                       save_plot,
                                                       overwrite_existing_files,
                                                       forbidden_scenario_ids,
                                                       paper_mode)
        self.randround_scenario_solution_storage = randround_scenario_solution_storage
        self.randround_algorithm_id = randround_algorithm_id
        self.randround_execution_id = randround_execution_id

        if heatmap_plot_type != HeatmapPlotType.ComparisonVineRandRound:
            raise RuntimeError("Only comparison heatmap plots are allowed")

    def _lookup_solutions(self, scenario_ids):
        return [(self.scenario_solution_storage.get_solutions_by_scenario_index(x)[self.algorithm_id][self.execution_id],
                 self.randround_scenario_solution_storage.get_solutions_by_scenario_index(x)[self.randround_algorithm_id][self.randround_execution_id])
                for x in scenario_ids]


class ComparisonPlotter_ECDF_BoxPlot(AbstractPlotter):

    def __init__(self,
                 output_path,
                 output_filetype,
                 vine_solution_storage,
                 vine_algorithm_id,
                 vine_execution_id,
                 randround_solution_storage,
                 randround_algorithm_id,
                 randround_execution_id,
                 show_plot=False,
                 save_plot=True,
                 overwrite_existing_files=False,
                 forbidden_scenario_ids=None,
                 paper_mode=True,
                 vine_settings_to_consider=None,
                 rr_settings_to_consider=None,
                 request_sets=None
                 ):
        super(ComparisonPlotter_ECDF_BoxPlot, self).__init__(output_path, output_filetype, vine_solution_storage,
                                                             vine_algorithm_id, vine_execution_id, show_plot, save_plot,
                                                             overwrite_existing_files, forbidden_scenario_ids, paper_mode)
        self.randround_solution_storage = randround_solution_storage
        self.randround_algorithm_id = randround_algorithm_id
        self.randround_execution_id = randround_execution_id

        filter_path_number_of_requests, list_number_of_requests = extract_parameter_range(self.scenarioparameter_room,
                                                                                          "number_of_requests")

        self._number_of_requests_list = list_number_of_requests
        self._filter_path_number_of_requests = filter_path_number_of_requests

        filter_path_edge_rf, list_edge_rfs = extract_parameter_range(self.scenarioparameter_room,
                                                                                          "edge_resource_factor")

        self._edge_rfs_list = list_edge_rfs
        self._filter_path_edge_rf = filter_path_edge_rf

        self.vine_settings_to_consider = vine_settings_to_consider
        self.rr_settings_to_consider = rr_settings_to_consider

        if self.vine_settings_to_consider is None:
            self.vine_settings_to_consider = get_list_of_vine_settings()

        if self.rr_settings_to_consider is None:
            self.rr_settings_to_consider = get_list_of_rr_settings()

        if request_sets is None:
            self.request_sets = [[40,60], [80,100]]
        else:
            self.request_sets = request_sets


    def _lookup_vine_solution(self, scenario_id):
        return self.scenario_solution_storage.get_solutions_by_scenario_index(scenario_id)[self.algorithm_id][self.execution_id]

    def _lookup_randround_solution(self, scenario_id):
        return self.randround_solution_storage.get_solutions_by_scenario_index(scenario_id)[self.randround_algorithm_id][self.randround_execution_id]

    def _compute_profit_best_rr_div_best_vine(self, vine_result, rr_result):
        best_rr = max([rr_result.profits[rr_settings].max for rr_settings in self.rr_settings_to_consider])
        best_vine = max([vine_result[vine_settings][0].profit.max for vine_settings in self.vine_settings_to_consider])
        return best_rr / best_vine

    def compute_relative_profits_arrays(self, list_of_scenarios):

        result = {edge_rf :
                      {number_of_requests: None
                       for number_of_requests in self._number_of_requests_list}
                  for edge_rf in self._edge_rfs_list
                  }

        for edge_rf in self._edge_rfs_list:
            for number_of_requests in self._number_of_requests_list:
                scenario_ids_with_right_edge_rf = self._obtain_scenarios_based_on_filters([{"parameter": "edge_resource_factor", "value": edge_rf}])
                scenario_ids_with_right_number_requests = self._obtain_scenarios_based_on_filters([{"parameter": "number_of_requests", "value": number_of_requests}])
                scenario_ids_to_consider = set(list_of_scenarios)
                scenario_ids_to_consider &= scenario_ids_with_right_edge_rf
                scenario_ids_to_consider &= scenario_ids_with_right_number_requests
                result[edge_rf][number_of_requests] = np.full(len(scenario_ids_to_consider), np.NaN)
                for i, scenario_id in enumerate(scenario_ids_to_consider):
                    vine_result = self._lookup_vine_solution(scenario_id)
                    rr_result = self._lookup_randround_solution(scenario_id)
                    result[edge_rf][number_of_requests][i] = self._compute_profit_best_rr_div_best_vine(vine_result, rr_result)

        return result



    def plot_figure(self, filter_specifications):
        self.plot_profit_ecdf(filter_specifications)
        self.plot_relative_performance_Vine_and_RandRound(filter_specifications)

    def plot_profit_ecdf(self, filter_specifications):

        output_filename = "ECDF_profit"

        output_path, filename = self._construct_output_path_and_filename(output_filename,
                                                                         filter_specifications)

        logger.debug("output_path is {};\t filename is {}".format(output_path, filename))

        if not self.overwrite_existing_files and os.path.exists(filename):
            logger.info("Skipping generation of {} as this file already exists".format(filename))
            return

        if filter_specifications:
            for filter_specification in filter_specifications:
                if filter_specification["parameter"] == "number_of_requests":
                    logger.info("Skipping generation of {} as this conflicts with the filter specification {}".format(
                        output_filename, filter_specification))
                    return

        scenario_ids = self._obtain_scenarios_based_on_filters(filter_specifications)

        if self.forbidden_scenario_ids:
            scenario_ids = scenario_ids - self.forbidden_scenario_ids

        result = self.compute_relative_profits_arrays(scenario_ids)
        print result

        fig, axs = plt.subplots(nrows=2, figsize=FIGSIZE, sharex="col", sharey="row")
        # ax.set_xscale("log", basex=10)

        #colors_erf = ['k', 'g', 'b', 'r', 'y']
        colors_erf = [plt.cm.inferno(val) for val in [0.8,0.6,0.4,0.2,0.0]]
        max_observed_value = 0

        linestyles = [":", "-.", "--", "-"]

        number_requests_legend_handlers = []
        erf_legend_handlers = []

        for j, number_of_requests_list in enumerate(self.request_sets):

            for i, erf in enumerate(self._edge_rfs_list):

                result_slice = np.zeros(0)

                for number_of_requests in number_of_requests_list:
                    result_slice = np.concatenate((result_slice, result[erf][number_of_requests]))

                ratio_rr_better = (len(np.where(result_slice > 1.29999)[0]))/(float(len(result_slice)))
                print "{:0.2f} {:^12s} {:0.10f}".format(erf, number_of_requests_list, ratio_rr_better)

                sorted_data = np.sort(result_slice[~np.isnan(result_slice)])
                max_observed_value = np.maximum(max_observed_value, sorted_data[-1])
                yvals = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))
                yvals *= 100
                sorted_data *= 100
                axs[j].plot(sorted_data, yvals, color=colors_erf[i], alpha=0.8, linestyle="-",
                        label="{} {}".format(erf, number_of_requests_list), linewidth=2.8)

                # if j == 0:
                #     number_requests_legend_handlers.append(
                #         matplotlib.lines.Line2D([], [], color='gray', linestyle=linestyles[j+2],
                #                                 label='{}'.format(number_of_requests_list)))

                if j == 0:
                    erf_legend_handlers.append(matplotlib.lines.Line2D([], [], color=colors_erf[i], linestyle="-", linewidth=2.4,
                                                               label='{}'.format(erf)))

                ax = axs[j]

                #ax.set_title("#Requests: {} & {}".format(number_of_requests_list[0],number_of_requests_list[1]), fontsize=15)
                props = dict(boxstyle='round', facecolor='white', alpha=0.5)
                ax.text(25, 95, "#req.:\n{} & {}".format(number_of_requests_list[0],number_of_requests_list[1]), fontsize=13, bbox=props, verticalalignment="top")
                #ax.set_ylabel("ECDF [%]", fontsize=14)
                ax.grid(True, which="both", linestyle=":")
                ax.set_xlim(20,200)

                major_x = [40, 70, 100, 130, 160, 190]
                minor_x = [25, 55, 85, 115, 145, 175]
                ax.set_xticks(major_x, minor=False)
                ax.set_xticks(minor_x, minor=True)
                for x in major_x:
                    if x == 100:
                        ax.axvline(x, linestyle=':', color='red', alpha=0.6, linewidth=0.8)
                    else:
                        ax.axvline(x, linestyle=':', color='gray', alpha=0.4, linewidth=0.8)

                major_y = [0, 25, 50, 75, 100]

                ax.set_yticks(major_y, minor=False)

                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(15)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(14.5)

                if j == 1:
                    ax.set_xlabel("profit($\mathsf{RR}_{\mathsf{best}}$) / profit($\mathsf{WiNE}_{\mathsf{best}}$) [%]", fontsize=15)

        fig.text(0.01, 0.54, 'ECDF [%]', va='center', rotation='vertical', fontsize=15)
        fig.subplots_adjust(top=0.9)
        fig.subplots_adjust(bottom=0.18)
        fig.subplots_adjust(right=0.78)
        fig.subplots_adjust(hspace=0.1)
        fig.subplots_adjust(left=0.16)

        first_legend = plt.legend(handles=erf_legend_handlers, title="ERF", loc=4, fontsize=14,
                                  handletextpad=0.35, bbox_to_anchor=(1,0.25), bbox_transform = plt.gcf().transFigure,
             borderaxespad=0.175, borderpad=0.2)
        plt.setp(first_legend.get_title(), fontsize='15')
        plt.gca().add_artist(first_legend)


        plt.setp(axs[0].get_xticklabels(), visible=True)

        # o_leg = plt.legend(handles=number_requests_legend_handlers, loc=2, title="#Requests", fontsize=14,
        #                    handletextpad=.35, borderaxespad=0.175, borderpad=0.2)
        # plt.setp(o_leg.get_title(), fontsize='15')

        plt.suptitle("Profit Comparison: $\mathsf{RR}_{\mathsf{best}}$ / $\mathsf{WiNE}_{\mathsf{best}}$", fontsize=17)
        #ax.set_xlabel("rel profit$)", fontsize=16)


        # for tick in ax.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(15.5)
        # for tick in ax.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(15.5)

        # ax.set_xticks([ 1, 1.5, 2, 2.5, 3, 3.5], minor=False)
        # ax.set_xticks([0.75, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75, 3.25, 3.5], minor=True)
        # ax.set_yticks([x*0.1 for x in range(1,10)], minor=True)
        # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # ax.set_xticklabels([], minor=True)



        # gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        # for line in gridlines:
        #     line.set_linestyle(':')

        self._show_and_or_save_plots(output_path, filename, perform_tight_layout=False)

    def plot_profit_ecdf_pre_box(self, filter_specifications):

        output_filename = "ECDF_profit"

        output_path, filename = self._construct_output_path_and_filename(output_filename,
                                                                         filter_specifications)

        logger.debug("output_path is {};\t filename is {}".format(output_path, filename))

        if not self.overwrite_existing_files and os.path.exists(filename):
            logger.info("Skipping generation of {} as this file already exists".format(filename))
            return

        if filter_specifications:
            for filter_specification in filter_specifications:
                if filter_specification["parameter"] == "number_of_requests":
                    logger.info("Skipping generation of {} as this conflicts with the filter specification {}".format(
                        output_filename, filter_specification))
                    return

        scenario_ids = self._obtain_scenarios_based_on_filters(filter_specifications)

        if self.forbidden_scenario_ids:
            scenario_ids = scenario_ids - self.forbidden_scenario_ids

        result = self.compute_relative_profits_arrays(scenario_ids)
        print result

        fig, axs = plt.subplots(nrows=2, figsize=FIGSIZE, sharex="col")
        # ax.set_xscale("log", basex=10)

        #colors_erf = ['k', 'g', 'b', 'r', 'y']
        colors_erf = [plt.cm.inferno(val) for val in [0.8,0.6,0.4,0.2,0.0]]
        max_observed_value = 0

        linestyles = [":", "-.", "--", "-"]

        number_requests_legend_handlers = []
        erf_legend_handlers = []

        for j, number_of_requests_list in enumerate([[40, 60], [80, 100]]):

            for i, erf in enumerate(self._edge_rfs_list):

                result_slice = np.zeros(0)

                for number_of_requests in number_of_requests_list:
                    result_slice = np.concatenate((result_slice, result[erf][number_of_requests]))

                sorted_data = np.sort(result_slice[~np.isnan(result_slice)])
                max_observed_value = np.maximum(max_observed_value, sorted_data[-1])
                yvals = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))
                axs[j].plot(sorted_data, yvals, color=colors_erf[i], alpha=0.8, linestyle="-",
                        label="{} {}".format(erf, number_of_requests_list), linewidth=2.8)

                # if j == 0:
                #     number_requests_legend_handlers.append(
                #         matplotlib.lines.Line2D([], [], color='gray', linestyle=linestyles[j+2],
                #                                 label='{}'.format(number_of_requests_list)))

                if j == 0:
                    erf_legend_handlers.append(matplotlib.lines.Line2D([], [], color=colors_erf[i], linestyle="-", linewidth=2.4,
                                                               label='{}'.format(erf)))

                ax = axs[j]

                ax.set_title("#Requests: {} & {}".format(number_of_requests_list[0],number_of_requests_list[1]), fontsize=15)
                ax.set_ylabel("ECDF", fontsize=14)
                ax.grid(True, which="both", linestyle=":")
                ax.set_xlim(0.2,2)

                major_x = [0.4, 0.7, 1.0, 1.3, 1.6,1.9]
                minor_x = [0.25, 0.55, 0.85, 1.15, 1.45, 1.75]
                ax.set_xticks(major_x, minor=False)
                ax.set_xticks(minor_x, minor=True)
                for x in major_x:
                    ax.axvline(x, linestyle=':', color='gray', alpha=0.4, linewidth=0.8)

                major_y = [0, 0.25, 0.5, 0.75, 1.0]

                ax.set_yticks(major_y, minor=False)

                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_fontsize(15)
                for tick in ax.yaxis.get_major_ticks():
                    tick.label.set_fontsize(14)

                if j == 1:
                    ax.set_xlabel("profit($\mathsf{RR}_{\mathsf{best}}$) / profit($\mathsf{WiNE}_{\mathsf{best}}$)", fontsize=15)

        fig.subplots_adjust(top=0.825)
        fig.subplots_adjust(bottom=0.15)
        fig.subplots_adjust(right=0.78)
        fig.subplots_adjust(hspace=0.3)
        fig.subplots_adjust(left=0.18)

        first_legend = plt.legend(handles=erf_legend_handlers, title="ERF", loc=4, fontsize=14,
                                  handletextpad=0.35, bbox_to_anchor=(1,0.25), bbox_transform = plt.gcf().transFigure,
             borderaxespad=0.175, borderpad=0.2)
        plt.setp(first_legend.get_title(), fontsize='15')
        plt.gca().add_artist(first_legend)


        plt.setp(axs[0].get_xticklabels(), visible=True)

        # o_leg = plt.legend(handles=number_requests_legend_handlers, loc=2, title="#Requests", fontsize=14,
        #                    handletextpad=.35, borderaxespad=0.175, borderpad=0.2)
        # plt.setp(o_leg.get_title(), fontsize='15')

        plt.suptitle("Relative Profit", fontsize=17)
        #ax.set_xlabel("rel profit$)", fontsize=16)


        # for tick in ax.xaxis.get_major_ticks():
        #     tick.label.set_fontsize(15.5)
        # for tick in ax.yaxis.get_major_ticks():
        #     tick.label.set_fontsize(15.5)

        # ax.set_xticks([ 1, 1.5, 2, 2.5, 3, 3.5], minor=False)
        # ax.set_xticks([0.75, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75, 3.25, 3.5], minor=True)
        # ax.set_yticks([x*0.1 for x in range(1,10)], minor=True)
        # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # ax.set_xticklabels([], minor=True)



        # gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        # for line in gridlines:
        #     line.set_linestyle(':')

        self._show_and_or_save_plots(output_path, filename, perform_tight_layout=False)

    def plot_relative_performance_Vine_and_RandRound(self, filter_specifications):

        output_filename = "boxplot_relative_performance"

        output_path, filename = self._construct_output_path_and_filename(output_filename,
                                                                         filter_specifications)

        logger.debug("output_path is {};\t filename is {}".format(output_path, filename))

        if not self.overwrite_existing_files and os.path.exists(filename):
            logger.info("Skipping generation of {} as this file already exists".format(filename))
            return

        if filter_specifications:
            for filter_specification in filter_specifications:
                if filter_specification["parameter"] == "number_of_requests":
                    logger.info("Skipping generation of {} as this conflicts with the filter specification {}".format(
                        output_filename, filter_specification))
                    return

        scenario_ids = self._obtain_scenarios_based_on_filters(filter_specifications)

        if self.forbidden_scenario_ids:
            scenario_ids = scenario_ids - self.forbidden_scenario_ids

        vine_settings_list = get_list_of_vine_settings()
        rr_settings_list = get_list_of_rr_settings()

        plot_data_raw = {vine_settings: {scenario_id: None for scenario_id in scenario_ids} for vine_settings in
                         vine_settings_list}
        plot_data_raw.update(
            {rr_settings: {scenario_id: None for scenario_id in scenario_ids} for rr_settings in rr_settings_list})

        for scenario_id in scenario_ids:
            best_vine = max([self._lookup_vine_solution(scenario_id)[vine_settings][0].profit.max for vine_settings in
                             vine_settings_list])
            best_rr = max([self._lookup_randround_solution(scenario_id).profits[rr_settings].max for rr_settings in
                           rr_settings_list])
            best_bound = self._lookup_randround_solution(scenario_id).lp_profit
            best_vine = best_bound
            best_rr = best_bound

            for vine_settings in vine_settings_list:
                plot_data_raw[vine_settings][scenario_id] = (
                    100.0 * self._lookup_vine_solution(scenario_id)[vine_settings][0].profit.max / best_vine,
                    100.0 * self._lookup_vine_solution(scenario_id)[vine_settings][0].profit.mean / best_vine
                )
            for rr_settings in rr_settings_list:
                plot_data_raw[rr_settings][scenario_id] = (
                    100.0 * self._lookup_randround_solution(scenario_id).profits[rr_settings].max / best_rr,
                    100.0 * self._lookup_randround_solution(scenario_id).profits[rr_settings].mean / best_rr
                )

        y_min = -5
        y_max = 105

        fig, axs = plt.subplots(ncols=2, nrows=1, figsize=FIGSIZE, gridspec_kw={'width_ratios': [13, 20]}, sharey="row")
        ax = axs[0]

        vine_det = []
        vine_rand = []

        for vine_settings in vine_settings_list:
            if vine_settings.edge_embedding_model == vine.ViNEEdgeEmbeddingModel.SPLITTABLE:
                continue
            if vine_settings.rounding_procedure == vine.ViNERoundingProcedure.DETERMINISTIC:
                vine_det.append(vine_settings)
            else:
                vine_rand.append(vine_settings)

        ordered_vine_settings = [vine_det, vine_rand]

        positions = []
        values = []

        minor_labels = []
        minor_label_locations = []

        major_labels = []
        major_label_locations = []
        current_pos = 0.5

        cmap = plt.get_cmap("inferno")

        color_best = cmap(0.6)
        color_mean = cmap(0)
        color_def = cmap(0.6)

        colors = []

        # vine!
        for i in range(2):
            # i == 0: det
            # i == 1: rand
            for vine_settings in ordered_vine_settings[i]:
                if i == 0:
                    current_values = [plot_data_raw[vine_settings][scenario_id][0] for scenario_id in scenario_ids]
                    values.append(current_values)
                    positions.append(current_pos)
                    if vine_settings.lp_objective == vine.ViNELPObjective.ViNE_LB_DEF:
                        minor_labels.append("L")
                    else:
                        minor_labels.append("C")
                    minor_label_locations.append(current_pos)
                    current_pos += 1.75
                    colors.append(color_def)
                else:
                    for j in range(2):
                        current_values = [plot_data_raw[vine_settings][scenario_id][j] for scenario_id in scenario_ids]
                        values.append(current_values)
                        positions.append(current_pos)
                        current_pos += 0.75
                        if j == 0:
                            colors.append(color_best)
                        else:
                            colors.append(color_mean)

                    if vine_settings.lp_objective == vine.ViNELPObjective.ViNE_LB_DEF:
                        minor_labels.append("L")
                    else:
                        minor_labels.append("C")
                    minor_label_locations.append((positions[-1] + positions[-2]) / 2.0)
                    current_pos += 0.5
            if i == 0:
                major_label_locations.append(np.mean(positions))
                major_labels.append("Det.")
                current_pos += 0.75
            else:
                major_label_locations.append((positions[2] + positions[-1]) / 2.0)
                major_labels.append("Rand.")

        # bplots = []
        #
        # for _bin, pos in zip(values, positions):
        #     print "plot...", pos
        #     bplots.append(ax.boxplot(x=_bin,
        #                              positions=[pos],
        #                              widths=[0.5],
        #                              patch_artist=True))

        bplots = ax.boxplot(x=values,
                            positions=positions,
                            widths=[0.5] * len(positions),
                            patch_artist=True,
                            notch=True,
                            bootstrap=10000)

        for i in range(len(bplots)):
            color = colors[i]
            bplots['boxes'][i].set_edgecolor(color)
            bplots['boxes'][i].set_facecolor(
                matplotlib.colors.to_rgba(color, alpha=0.3)
            )

            for keyword in ["medians", "fliers", "whiskers", "caps"]:
                if keyword == "whiskers" or keyword == "caps":
                    bplots[keyword][i * 2].set_color(color)
                    bplots[keyword][i * 2 + 1].set_color(color)
                else:
                    bplots[keyword][i].set_color(color)
                if keyword == "fliers":
                    bplots[keyword][i].set(
                        marker='o',
                        markeredgecolor=matplotlib.colors.to_rgba(color, alpha=0.15),
                    )

        ax.set_ylim(y_min, y_max)

        for k in range(len(minor_label_locations)):
            ax.text(x=minor_label_locations[k], y=y_min - 11, s=minor_labels[k], horizontalalignment='center',
                    fontdict={'fontsize': 14})

        for k in range(len(major_label_locations)):
            ax.text(x=major_label_locations[k], y=y_min - 21, s=major_labels[k], horizontalalignment='center',
                    fontdict={'fontsize': 14})

        ax.set_xticks([])

        ax.set_yticks([x * 10 for x in range(1, 10, 2)], minor=True)

        ax.grid(True, which="major", linestyle="-")
        ax.grid(True, which="minor", linestyle=":")

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15)

        ax.set_title("WiNE(ViNE)", fontsize=16)

        ax.set_ylabel("Profit / $\mathsf{LP}_{\mathsf{UB}}$ [%]", fontsize=16)

        # RAND ROUND!

        ax = axs[1]

        rr_no_recomp = [(treewidth_model.LPRecomputationMode.NONE, treewidth_model.RoundingOrder.RANDOM),
                        (treewidth_model.LPRecomputationMode.NONE, treewidth_model.RoundingOrder.STATIC_REQ_PROFIT),
                        (treewidth_model.LPRecomputationMode.NONE, treewidth_model.RoundingOrder.ACHIEVED_REQ_PROFIT)]
        rr_recomp = [(treewidth_model.LPRecomputationMode.RECOMPUTATION_WITHOUT_SEPARATION,
                      treewidth_model.RoundingOrder.RANDOM),
                     (treewidth_model.LPRecomputationMode.RECOMPUTATION_WITHOUT_SEPARATION,
                      treewidth_model.RoundingOrder.STATIC_REQ_PROFIT),
                     (treewidth_model.LPRecomputationMode.RECOMPUTATION_WITHOUT_SEPARATION,
                      treewidth_model.RoundingOrder.ACHIEVED_REQ_PROFIT)]

        ordered_rr_settings = [rr_no_recomp, rr_recomp]

        positions = []
        values = []

        minor_labels = []
        minor_label_locations = []

        major_labels = []
        major_label_locations = []
        current_pos = 0.5

        colors = []

        fig.subplots_adjust(bottom=0.18, top=0.84, right=0.83, wspace=0.12, left=0.14)

        # rand round
        for i in range(2):
            # i == 0: no_recomp
            # i == 1: recomp!
            for rr_settings in ordered_rr_settings[i]:
                for j in range(2):
                    current_values = [plot_data_raw[rr_settings][scenario_id][j] for scenario_id in scenario_ids]
                    values.append(current_values)
                    positions.append(current_pos)
                    current_pos += 0.75
                    if j == 0:
                        colors.append(color_best)
                    else:
                        colors.append(color_mean)

                if rr_settings[1] == treewidth_model.RoundingOrder.RANDOM:
                    minor_labels.append("R")
                elif rr_settings[1] == treewidth_model.RoundingOrder.ACHIEVED_REQ_PROFIT:
                    minor_labels.append("A")
                elif rr_settings[1] == treewidth_model.RoundingOrder.STATIC_REQ_PROFIT:
                    minor_labels.append("S")
                else:
                    raise ValueError()
                minor_label_locations.append((positions[-1] + positions[-2]) / 2.0)
                current_pos += 0.5

            if i == 0:
                major_label_locations.append(np.mean(positions))
                major_labels.append("No Recomp.")
                current_pos += 1
            else:
                major_label_locations.append((positions[6] + positions[-1]) / 2.0)
                major_labels.append("Recomp.")

        bplots = ax.boxplot(x=values,
                            positions=positions,
                            widths=[0.5] * len(positions),
                            patch_artist=True,
                            notch=True,
                            bootstrap=1000)

        print bplots
        print colors

        for i in range(len(positions)):
            print "Setting color of boxplot ", i
            color = colors[i]
            bplots['boxes'][i].set_edgecolor(color)
            bplots['boxes'][i].set_facecolor(
                matplotlib.colors.to_rgba(color, alpha=0.3)
            )

            for keyword in ["medians", "fliers", "whiskers", "caps"]:
                if keyword == "whiskers" or keyword == "caps":
                    bplots[keyword][i * 2].set_color(color)
                    bplots[keyword][i * 2 + 1].set_color(color)
                else:
                    bplots[keyword][i].set_color(color)
                if keyword == "fliers":
                    bplots[keyword][i].set(
                        marker='o',
                        markeredgecolor=matplotlib.colors.to_rgba(color, alpha=0.15),
                    )

        ax.set_ylim(y_min, y_max)

        for k in range(len(minor_label_locations)):
            ax.text(x=minor_label_locations[k], y=y_min - 11, s=minor_labels[k], horizontalalignment='center',
                    fontdict={'fontsize': 14})

        for k in range(len(major_label_locations)):
            ax.text(x=major_label_locations[k], y=y_min - 21, s=major_labels[k], horizontalalignment='center',
                    fontdict={'fontsize': 14})

        ax.set_xticks([])

        ax.set_title("RR Heuristics", fontsize=16)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15)

        ax.set_yticks([x * 10 for x in range(1, 10, 2)], minor=True)

        ax.grid(True, which="major", linestyle="-")
        ax.grid(True, which="minor", linestyle=":")

        # LEGEND!

        best_patch = mpatches.Patch(color=matplotlib.colors.to_rgba(color_best, alpha=0.6), label='best')
        mean_patch = mpatches.Patch(color=matplotlib.colors.to_rgba(color_mean, alpha=0.6), label='mean')

        plt.legend(handles=[best_patch, mean_patch], loc=4, fontsize=14, handlelength=0.5,
                   handletextpad=0.35, bbox_to_anchor=(1, 0.5), bbox_transform=plt.gcf().transFigure,
                   borderaxespad=0.175, borderpad=0.2)

        plt.suptitle("Performance of Algorithm Variants", fontsize=17)

        self._show_and_or_save_plots(output_path, filename, perform_tight_layout=False)

    # def plot_relative_performance_Vine_and_RandRound(self, filter_specifications):
    #
    #     output_filename = "boxplot_relative_performance"
    #
    #     output_path, filename = self._construct_output_path_and_filename(output_filename,
    #                                                                      filter_specifications)
    #
    #     logger.debug("output_path is {};\t filename is {}".format(output_path, filename))
    #
    #     if not self.overwrite_existing_files and os.path.exists(filename):
    #         logger.info("Skipping generation of {} as this file already exists".format(filename))
    #         return
    #
    #     if filter_specifications:
    #         for filter_specification in filter_specifications:
    #             if filter_specification["parameter"] == "number_of_requests":
    #                 logger.info("Skipping generation of {} as this conflicts with the filter specification {}".format(
    #                     output_filename, filter_specification))
    #                 return
    #
    #     scenario_ids = self._obtain_scenarios_based_on_filters(filter_specifications)
    #
    #     if self.forbidden_scenario_ids:
    #         scenario_ids = scenario_ids - self.forbidden_scenario_ids
    #
    #     vine_settings_list = get_list_of_vine_settings()
    #     rr_settings_list = get_list_of_rr_settings()
    #
    #     plot_data_raw = {vine_settings: {scenario_id: None for scenario_id in scenario_ids} for vine_settings in
    #                      vine_settings_list}
    #     plot_data_raw.update(
    #         {rr_settings: {scenario_id: None for scenario_id in scenario_ids} for rr_settings in rr_settings_list})
    #
    #     for scenario_id in scenario_ids:
    #         best_vine = max([self._lookup_vine_solution(scenario_id)[vine_settings][0].profit.max for vine_settings in
    #                          vine_settings_list])
    #         best_rr = max([self._lookup_randround_solution(scenario_id).profits[rr_settings].max for rr_settings in
    #                        rr_settings_list])
    #         best_bound = self._lookup_randround_solution(scenario_id).lp_profit
    #         best_vine = best_bound
    #         best_rr = best_bound
    #
    #         for vine_settings in vine_settings_list:
    #             plot_data_raw[vine_settings][scenario_id] = (
    #                 100.0 * self._lookup_vine_solution(scenario_id)[vine_settings][0].profit.max / best_vine,
    #                 100.0 * self._lookup_vine_solution(scenario_id)[vine_settings][0].profit.mean / best_vine
    #             )
    #         for rr_settings in rr_settings_list:
    #             plot_data_raw[rr_settings][scenario_id] = (
    #                 100.0 * self._lookup_randround_solution(scenario_id).profits[rr_settings].max / best_rr,
    #                 100.0 * self._lookup_randround_solution(scenario_id).profits[rr_settings].mean / best_rr
    #             )
    #
    #     y_min = -5
    #     y_max = 105
    #
    #     fig, axs = plt.subplots(ncols=2, nrows=1, figsize=FIGSIZE, gridspec_kw={'width_ratios': [13, 20]}, sharey="row")
    #     ax = axs[0]
    #
    #     vine_det = []
    #     vine_rand = []
    #
    #     for vine_settings in vine_settings_list:
    #         if vine_settings.edge_embedding_model == vine.ViNEEdgeEmbeddingModel.SPLITTABLE:
    #             continue
    #         if vine_settings.rounding_procedure == vine.ViNERoundingProcedure.DETERMINISTIC:
    #             vine_det.append(vine_settings)
    #         else:
    #             vine_rand.append(vine_settings)
    #
    #     ordered_vine_settings = [vine_det, vine_rand]
    #
    #     positions = []
    #     values = []
    #
    #     minor_labels = []
    #     minor_label_locations = []
    #
    #     major_labels = []
    #     major_label_locations = []
    #     current_pos = 0.5
    #
    #     cmap = plt.get_cmap("inferno")
    #
    #     color_best = cmap(0.6)
    #     color_mean = cmap(0)
    #     color_def = cmap(0.6)
    #
    #     colors = []
    #
    #     # vine!
    #     for i in range(2):
    #         # i == 0: det
    #         # i == 1: rand
    #         for vine_settings in ordered_vine_settings[i]:
    #             if i == 0:
    #                 current_values = [plot_data_raw[vine_settings][scenario_id][0] for scenario_id in scenario_ids]
    #                 values.append(current_values)
    #                 positions.append(current_pos)
    #                 if vine_settings.lp_objective == vine.ViNELPObjective.ViNE_LB_DEF:
    #                     minor_labels.append("L")
    #                 else:
    #                     minor_labels.append("C")
    #                 minor_label_locations.append(current_pos)
    #                 current_pos += 1.75
    #                 colors.append(color_def)
    #             else:
    #                 for j in range(2):
    #                     current_values = [plot_data_raw[vine_settings][scenario_id][j] for scenario_id in scenario_ids]
    #                     values.append(current_values)
    #                     positions.append(current_pos)
    #                     current_pos += 0.75
    #                     if j == 0:
    #                         colors.append(color_best)
    #                     else:
    #                         colors.append(color_mean)
    #
    #                 if vine_settings.lp_objective == vine.ViNELPObjective.ViNE_LB_DEF:
    #                     minor_labels.append("L")
    #                 else:
    #                     minor_labels.append("C")
    #                 minor_label_locations.append((positions[-1] + positions[-2]) / 2.0)
    #                 current_pos += 0.5
    #         if i == 0:
    #             major_label_locations.append(np.mean(positions))
    #             major_labels.append("Det.")
    #             current_pos += 0.75
    #         else:
    #             major_label_locations.append((positions[2] + positions[-1]) / 2.0)
    #             major_labels.append("Rand.")
    #
    #     # bplots = []
    #     #
    #     # for _bin, pos in zip(values, positions):
    #     #     print "plot...", pos
    #     #     bplots.append(ax.boxplot(x=_bin,
    #     #                              positions=[pos],
    #     #                              widths=[0.5],
    #     #                              patch_artist=True))
    #
    #     bplots = ax.boxplot(x=values,
    #                         positions=positions,
    #                         widths=[0.5] * len(positions),
    #                         patch_artist=True,
    #                         notch=True,
    #                         bootstrap=10000)
    #
    #     for i in range(len(bplots)):
    #         color = colors[i]
    #         bplots['boxes'][i].set_edgecolor(color)
    #         bplots['boxes'][i].set_facecolor(
    #             matplotlib.colors.to_rgba(color, alpha=0.3)
    #         )
    #
    #         for keyword in ["medians", "fliers", "whiskers", "caps"]:
    #             if keyword == "whiskers" or keyword == "caps":
    #                 bplots[keyword][i * 2].set_color(color)
    #                 bplots[keyword][i * 2 + 1].set_color(color)
    #             else:
    #                 bplots[keyword][i].set_color(color)
    #             if keyword == "fliers":
    #                 bplots[keyword][i].set(
    #                     marker='o',
    #                     markeredgecolor=matplotlib.colors.to_rgba(color, alpha=0.15),
    #                 )
    #
    #     ax.set_ylim(y_min, y_max)
    #
    #     for k in range(len(minor_label_locations)):
    #         ax.text(x=minor_label_locations[k], y=y_min - 11, s=minor_labels[k], horizontalalignment='center',
    #                 fontdict={'fontsize': 14})
    #
    #     for k in range(len(major_label_locations)):
    #         ax.text(x=major_label_locations[k], y=y_min - 21, s=major_labels[k], horizontalalignment='center',
    #                 fontdict={'fontsize': 14})
    #
    #     ax.set_xticks([])
    #
    #     ax.set_yticks([x * 10 for x in range(1, 10, 2)], minor=True)
    #
    #     ax.grid(True, which="major", linestyle="-")
    #     ax.grid(True, which="minor", linestyle=":")
    #
    #     ax.set_title("ViNE", fontsize=15.5)
    #
    #     ax.set_ylabel("Relative Performance [%]", fontsize=14)
    #
    #     # RAND ROUND!
    #
    #     ax = axs[1]
    #
    #     rr_no_recomp = [(treewidth_model.LPRecomputationMode.NONE, treewidth_model.RoundingOrder.RANDOM),
    #                     (treewidth_model.LPRecomputationMode.NONE, treewidth_model.RoundingOrder.STATIC_REQ_PROFIT),
    #                     (treewidth_model.LPRecomputationMode.NONE, treewidth_model.RoundingOrder.ACHIEVED_REQ_PROFIT)]
    #     rr_recomp = [(treewidth_model.LPRecomputationMode.RECOMPUTATION_WITHOUT_SEPARATION,
    #                   treewidth_model.RoundingOrder.RANDOM),
    #                  (treewidth_model.LPRecomputationMode.RECOMPUTATION_WITHOUT_SEPARATION,
    #                   treewidth_model.RoundingOrder.STATIC_REQ_PROFIT),
    #                  (treewidth_model.LPRecomputationMode.RECOMPUTATION_WITHOUT_SEPARATION,
    #                   treewidth_model.RoundingOrder.ACHIEVED_REQ_PROFIT)]
    #
    #     ordered_rr_settings = [rr_no_recomp, rr_recomp]
    #
    #     positions = []
    #     values = []
    #
    #     minor_labels = []
    #     minor_label_locations = []
    #
    #     major_labels = []
    #     major_label_locations = []
    #     current_pos = 0.5
    #
    #     colors = []
    #
    #     fig.subplots_adjust(bottom=0.18, top=0.84, right=0.83, wspace=0.12)
    #
    #     # rand round
    #     for i in range(2):
    #         # i == 0: no_recomp
    #         # i == 1: recomp!
    #         for rr_settings in ordered_rr_settings[i]:
    #             for j in range(2):
    #                 current_values = [plot_data_raw[rr_settings][scenario_id][j] for scenario_id in scenario_ids]
    #                 values.append(current_values)
    #                 positions.append(current_pos)
    #                 current_pos += 0.75
    #                 if j == 0:
    #                     colors.append(color_best)
    #                 else:
    #                     colors.append(color_mean)
    #
    #             if rr_settings[1] == treewidth_model.RoundingOrder.RANDOM:
    #                 minor_labels.append("R")
    #             elif rr_settings[1] == treewidth_model.RoundingOrder.ACHIEVED_REQ_PROFIT:
    #                 minor_labels.append("A")
    #             elif rr_settings[1] == treewidth_model.RoundingOrder.STATIC_REQ_PROFIT:
    #                 minor_labels.append("S")
    #             else:
    #                 raise ValueError()
    #             minor_label_locations.append((positions[-1] + positions[-2]) / 2.0)
    #             current_pos += 0.5
    #
    #         if i == 0:
    #             major_label_locations.append(np.mean(positions))
    #             major_labels.append("No Recomp.")
    #             current_pos += 1
    #         else:
    #             major_label_locations.append((positions[6] + positions[-1]) / 2.0)
    #             major_labels.append("Recomp.")
    #
    #     bplots = ax.boxplot(x=values,
    #                         positions=positions,
    #                         widths=[0.5] * len(positions),
    #                         patch_artist=True,
    #                         notch=True,
    #                         bootstrap=1000)
    #
    #     print bplots
    #     print colors
    #
    #     for i in range(len(positions)):
    #         print "Setting color of boxplot ", i
    #         color = colors[i]
    #         bplots['boxes'][i].set_edgecolor(color)
    #         bplots['boxes'][i].set_facecolor(
    #             matplotlib.colors.to_rgba(color, alpha=0.3)
    #         )
    #
    #         for keyword in ["medians", "fliers", "whiskers", "caps"]:
    #             if keyword == "whiskers" or keyword == "caps":
    #                 bplots[keyword][i * 2].set_color(color)
    #                 bplots[keyword][i * 2 + 1].set_color(color)
    #             else:
    #                 bplots[keyword][i].set_color(color)
    #             if keyword == "fliers":
    #                 bplots[keyword][i].set(
    #                     marker='o',
    #                     markeredgecolor=matplotlib.colors.to_rgba(color, alpha=0.15),
    #                 )
    #
    #     ax.set_ylim(y_min, y_max)
    #
    #     for k in range(len(minor_label_locations)):
    #         ax.text(x=minor_label_locations[k], y=y_min - 11, s=minor_labels[k], horizontalalignment='center',
    #                 fontdict={'fontsize': 14})
    #
    #     for k in range(len(major_label_locations)):
    #         ax.text(x=major_label_locations[k], y=y_min - 21, s=major_labels[k], horizontalalignment='center',
    #                 fontdict={'fontsize': 14})
    #
    #     ax.set_xticks([])
    #
    #     ax.set_title("RR Heuristics", fontsize=15.5)
    #
    #     ax.set_yticks([x * 10 for x in range(1, 10, 2)], minor=True)
    #
    #     ax.grid(True, which="major", linestyle="-")
    #     ax.grid(True, which="minor", linestyle=":")
    #
    #     # LEGEND!
    #
    #     best_patch = mpatches.Patch(color=matplotlib.colors.to_rgba(color_best, alpha=0.6), label='best')
    #     mean_patch = mpatches.Patch(color=matplotlib.colors.to_rgba(color_mean, alpha=0.6), label='mean')
    #
    #     plt.legend(handles=[best_patch, mean_patch], loc=4, fontsize=14, handlelength=0.5,
    #                handletextpad=0.35, bbox_to_anchor=(1, 0.5), bbox_transform=plt.gcf().transFigure,
    #                borderaxespad=0.175, borderpad=0.2)
    #
    #     plt.suptitle("Performance of Algorithm Variants", fontsize=17)
    #
    #     self._show_and_or_save_plots(output_path, filename, perform_tight_layout=False)

    def plot_profit_ecdf_old(self, filter_specifications):

        output_filename = "ECDF_profit"

        output_path, filename = self._construct_output_path_and_filename(output_filename,
                                                                         filter_specifications)

        logger.debug("output_path is {};\t filename is {}".format(output_path, filename))

        if not self.overwrite_existing_files and os.path.exists(filename):
            logger.info("Skipping generation of {} as this file already exists".format(filename))
            return

        if filter_specifications:
            for filter_specification in filter_specifications:
                if filter_specification["parameter"] == "number_of_requests":
                    logger.info("Skipping generation of {} as this conflicts with the filter specification {}".format(
                        output_filename, filter_specification))
                    return

        scenario_ids = self._obtain_scenarios_based_on_filters(filter_specifications)

        if self.forbidden_scenario_ids:
            scenario_ids = scenario_ids - self.forbidden_scenario_ids

        result = self.compute_relative_profits_arrays(scenario_ids)
        print result

        fix, ax = plt.subplots(figsize=FIGSIZE)
        # ax.set_xscale("log", basex=10)

        colors_erf = ['k', 'g', 'b', 'r', 'y']
        max_observed_value = 0

        linestyles = [":", "-.", "--", "-"]

        number_requests_legend_handlers = []
        erf_legend_handlers = []

        for i, erf in enumerate(self._edge_rfs_list):

            previous_slice = None

            for j, number_of_requests in enumerate(self._number_of_requests_list):

                result_slice = result[erf][number_of_requests]
                sorted_data = np.sort(result_slice[~np.isnan(result_slice)])
                max_observed_value = np.maximum(max_observed_value, sorted_data[-1])
                yvals = np.arange(1, len(sorted_data) + 1) / float(len(sorted_data))
                ax.plot(sorted_data, yvals, color=colors_erf[i], linestyle=linestyles[j],
                        label="{} {}".format(erf, number_of_requests), linewidth=1.8)

                if i == 0:
                    number_requests_legend_handlers.append(
                        matplotlib.lines.Line2D([], [], color='gray', linestyle=linestyles[j],
                                                label='|req|: {}'.format(number_of_requests)))

            erf_legend_handlers.append(matplotlib.lines.Line2D([], [], color=colors_erf[i], linestyle="-",
                                                               label='ERF: {}'.format(erf)))

        first_legend = plt.legend(title="", handles=erf_legend_handlers, loc=(0.225, 0.0125), fontsize=14,
                                  handletextpad=0.35, borderaxespad=0.175, borderpad=0.2)
        plt.setp(first_legend.get_title(), fontsize='15')
        plt.gca().add_artist(first_legend)
        o_leg = plt.legend(handles=number_requests_legend_handlers, loc=4, title="#Requests", fontsize=14,
                           handletextpad=.35, borderaxespad=0.175, borderpad=0.2)
        plt.setp(o_leg.get_title(), fontsize='15')

        ax.set_title("FOO", fontsize=17)
        ax.set_xlabel("rel profit$)", fontsize=16)
        ax.set_ylabel("ECDF", fontsize=16)

        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(15.5)
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(15.5)

        # ax.set_xticks([ 1, 1.5, 2, 2.5, 3, 3.5], minor=False)
        # ax.set_xticks([0.75, 1.25, 1.5, 1.75, 2.25, 2.5, 2.75, 3.25, 3.5], minor=True)
        # ax.set_yticks([x*0.1 for x in range(1,10)], minor=True)
        # ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

        # ax.set_xticklabels([], minor=True)

        ax.grid(True, which="both", linestyle=":")

        # gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        # for line in gridlines:
        #     line.set_linestyle(':')

        self._show_and_or_save_plots(output_path, filename)


def evaluate_vine_and_randround(dc_vine,
                                vine_algorithm_id,
                                vine_execution_id,
                                dc_randround_seplp_dynvmp,
                                randround_seplp_algorithm_id,
                                randround_seplp_execution_id,
                                exclude_generation_parameters=None,
                                parameter_filter_keys=None,
                                show_plot=False,
                                save_plot=True,
                                overwrite_existing_files=True,
                                forbidden_scenario_ids=None,
                                papermode=True,
                                maxdepthfilter=2,
                                output_path="./",
                                output_filetype="png",
                                request_sets=None):
    """ Main function for evaluation, creating plots and saving them in a specific directory hierarchy.
    A large variety of plots is created. For heatmaps, a generic plotter is used while for general
    comparison plots (ECDF and scatter) an own class is used. The plots that shall be generated cannot
    be controlled at the moment but the respective plotters can be easily adjusted.

    :param heatmap_plot_type:
    :param dc_vine: unpickled datacontainer of vine experiments
    :param vine_algorithm_id: algorithm id of the vine algorithm
    :param vine_execution_id: execution config (numeric) of the vine algorithm execution
    :param dc_randround_seplp_dynvmp: unpickled datacontainer of randomized rounding experiments
    :param randround_seplp_algorithm_id: algorithm id of the randround algorithm
    :param randround_seplp_execution_id: execution config (numeric) of the randround algorithm execution
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
                dc_vine.scenario_parameter_container.scenarioparameter_room, key)

            parameter_dicts_vine = lookup_scenario_parameter_room_dicts_on_path(
                dc_vine.scenario_parameter_container.scenarioparameter_room, parameter_filter_path)
            parameter_dicts_randround = lookup_scenario_parameter_room_dicts_on_path(
                dc_randround_seplp_dynvmp.scenario_parameter_container.scenarioparameter_room, parameter_filter_path)

            for value_to_exclude in values_to_exclude:

                if value_to_exclude not in parameter_values:
                    raise RuntimeError("The value {} is not contained in the list of parameter values {} for key {}".format(
                        value_to_exclude, parameter_values, key
                    ))

                # add respective scenario ids to the set of forbidden scenario ids
                forbidden_scenario_ids.update(set(lookup_scenarios_having_specific_values(
                    dc_vine.scenario_parameter_container.scenario_parameter_dict, parameter_filter_path, value_to_exclude)))

            # remove the respective values from the scenario parameter room such that these are not considered when
            # constructing e.g. axes
            parameter_dicts_vine[-1][key] = [value for value in parameter_dicts_vine[-1][key] if
                                                 value not in values_to_exclude]
            parameter_dicts_randround[-1][key] = [value for value in parameter_dicts_randround[-1][key] if
                                                  value not in values_to_exclude]

    if parameter_filter_keys is not None:
        filter_specs = _construct_filter_specs(dc_vine.scenario_parameter_container.scenarioparameter_room,
                                               parameter_filter_keys,
                                               maxdepth=maxdepthfilter)
    else:
        filter_specs = [None]

    plotters = []
    # initialize plotters for each valid vine setting...

    vine_plotter = SingleHeatmapPlotter(output_path=output_path,
                                        output_filetype=output_filetype,
                                        scenario_solution_storage=dc_vine,
                                        algorithm_id=vine_algorithm_id,
                                        execution_id=vine_execution_id,
                                        heatmap_plot_type=HeatmapPlotType.ViNE,
                                        show_plot=show_plot,
                                        save_plot=save_plot,
                                        overwrite_existing_files=overwrite_existing_files,
                                        forbidden_scenario_ids=forbidden_scenario_ids,
                                        paper_mode=papermode)

    plotters.append(vine_plotter)

    randround_plotter = SingleHeatmapPlotter(output_path=output_path,
                                             output_filetype=output_filetype,
                                             scenario_solution_storage=dc_randround_seplp_dynvmp,
                                             algorithm_id=randround_seplp_algorithm_id,
                                             execution_id=randround_seplp_execution_id,
                                             heatmap_plot_type=HeatmapPlotType.RandRoundSepLPDynVMP,
                                             show_plot=show_plot,
                                             save_plot=save_plot,
                                             overwrite_existing_files=overwrite_existing_files,
                                             forbidden_scenario_ids=forbidden_scenario_ids,
                                             paper_mode=papermode)

    plotters.append(randround_plotter)

    comparison_plotter = ComparisonHeatmapPlotter(output_path=output_path,
                                                  output_filetype=output_filetype,
                                                  vine_solution_storage=dc_vine,
                                                  vine_algorithm_id=vine_algorithm_id,
                                                  vine_execution_id=vine_execution_id,
                                                  randround_scenario_solution_storage=dc_randround_seplp_dynvmp,
                                                  randround_algorithm_id=randround_seplp_algorithm_id,
                                                  randround_execution_id=randround_seplp_execution_id,
                                                  heatmap_plot_type=HeatmapPlotType.ComparisonVineRandRound,
                                                  show_plot=show_plot,
                                                  save_plot=save_plot,
                                                  overwrite_existing_files=overwrite_existing_files,
                                                  forbidden_scenario_ids=forbidden_scenario_ids,
                                                  paper_mode=papermode)

    plotters.append(comparison_plotter)

    ecdf_plotter = ComparisonPlotter_ECDF_BoxPlot(output_path=output_path,
                                                  output_filetype=output_filetype,
                                                  vine_solution_storage=dc_vine,
                                                  vine_algorithm_id=vine_algorithm_id,
                                                  vine_execution_id=vine_execution_id,
                                                  randround_solution_storage=dc_randround_seplp_dynvmp,
                                                  randround_algorithm_id=randround_seplp_algorithm_id,
                                                  randround_execution_id=randround_seplp_execution_id,
                                                  show_plot=show_plot,
                                                  save_plot=save_plot,
                                                  overwrite_existing_files=overwrite_existing_files,
                                                  forbidden_scenario_ids=forbidden_scenario_ids,
                                                  paper_mode=papermode,
                                                  request_sets=request_sets)

    plotters.append(ecdf_plotter)


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
