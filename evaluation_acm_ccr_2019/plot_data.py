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

import os
from collections import namedtuple
import numpy as np
from alib import solutions, util

try:
    import cPickle as pickle
except ImportError:
    import pickle

from vnep_approx import vine, treewidth_model

REQUIRED_FOR_PICKLE = solutions  # this prevents pycharm from removing this import, which is required for unpickling solutions

ReducedOfflineViNEResultCollection = namedtuple(
    "ReducedOfflineViNEResultCollection",
    [
        "total_runtime",  # AggregatedData
        "profit",  # AggregatedData
        "runtime_per_request",  # AggregatedData
        "num_initial_lp_failed",  # sum across repetitions
        "num_node_mapping_failed",  # sum across repetitions
        "num_edge_mapping_failed",  # sum across repetitions
        "original_number_requests",
        "num_req_with_profit",
        "max_node_load",  # AggregatedData
        "max_edge_load",  # AggregatedData
    ],
)

ReducedRandRoundSepLPOptDynVMPCollectionResult = namedtuple(
    "ReducedRandRoundSepLPOptDynVMPCollectionResult",
    [
        "lp_time_preprocess",
        "lp_time_tree_decomposition",
        "lp_time_dynvmp_initialization",
        "lp_time_gurobi_optimization",
        "lp_time_optimization",
        "lp_status",
        "lp_profit",
        "lp_generated_columns",
        "max_node_loads",
        "max_edge_loads",
        "rounding_runtimes",
        "profits"
    ],
)

AggregatedData = namedtuple(
    "AggregatedData",
    [
        "min",
        "mean",
        "max",
        "std_dev",
        "value_count"
    ]
)


def get_aggregated_data(list_of_values):
    _min = np.min(list_of_values)
    _mean = np.mean(list_of_values)
    _max = np.max(list_of_values)
    _std_dev = np.std(list_of_values)
    _value_count = len(list_of_values)
    return AggregatedData(min=_min,
                          max=_max,
                          mean=_mean,
                          std_dev=_std_dev,
                          value_count=_value_count)


logger = util.get_logger(__name__, make_file=False, propagate=True)


class OfflineViNEResultCollectionReducer(object):

    def __init__(self):
        pass

    def reduce_vine_result_collection(self, baseline_solutions_input_pickle_name,
                                      reduced_baseline_solutions_output_pickle_name=None):

        baseline_solutions_input_pickle_path = os.path.join(
            util.ExperimentPathHandler.INPUT_DIR,
            baseline_solutions_input_pickle_name
        )

        if reduced_baseline_solutions_output_pickle_name is None:
            file_basename = os.path.basename(baseline_solutions_input_pickle_path).split(".")[0]
            reduced_baseline_solutions_output_pickle_path = os.path.join(util.ExperimentPathHandler.OUTPUT_DIR,
                                                                         file_basename + "_reduced.pickle")
        else:
            reduced_baseline_solutions_output_pickle_path = os.path.join(util.ExperimentPathHandler.OUTPUT_DIR,
                                                                         baseline_solutions_input_pickle_name)

        logger.info("\nWill read from ..\n\t{} \n\t\tand store reduced data into\n\t{}\n".format(baseline_solutions_input_pickle_path, reduced_baseline_solutions_output_pickle_path))

        logger.info("Reading pickle file at {}".format(baseline_solutions_input_pickle_path))
        with open(baseline_solutions_input_pickle_path, "rb") as input_file:
            scenario_solution_storage = pickle.load(input_file)

        ssd = scenario_solution_storage.algorithm_scenario_solution_dictionary
        ssd_reduced = {}
        for algorithm in ssd.keys():
            logger.info(".. Reducing results of algorithm {}".format(algorithm))
            ssd_reduced[algorithm] = {}
            for scenario_id in ssd[algorithm].keys():
                logger.info("   .. handling scenario {}".format(scenario_id))
                ssd_reduced[algorithm][scenario_id] = {}
                for exec_id in ssd[algorithm][scenario_id].keys():
                    ssd_reduced[algorithm][scenario_id][exec_id] = {}
                    params, scenario = scenario_solution_storage.scenario_parameter_container.scenario_triple[scenario_id]
                    solution_collection = ssd[algorithm][scenario_id][exec_id].get_solution()
                    for vine_settings, result_list in solution_collection.iteritems():
                        ssd_reduced[algorithm][scenario_id][exec_id][vine_settings] = []
                        number_of_req_profit = 0
                        for req in scenario.requests:
                            if req.profit > 0.001:
                                number_of_req_profit += 1
                        number_of_requests = len(scenario.requests)

                        max_node_load_vals = np.zeros(len(result_list))
                        max_edge_load_vals = np.zeros(len(result_list))
                        total_runtime_vals = np.zeros(len(result_list))
                        profit_vals = np.zeros(len(result_list))

                        num_edge_mapping_failed = 0
                        num_initial_lp_failed = 0
                        num_node_mapping_failed = 0

                        runtimes_per_request_vals = []
                        for (result_index, result) in result_list:
                            assert isinstance(result, vine.OfflineViNEResult)
                            solution_object = result.get_solution()
                            mappings = solution_object.request_mapping

                            load = _initialize_load_dict(scenario)
                            for req in scenario.requests:
                                runtimes_per_request_vals.append(
                                    result.runtime_per_request[req]
                                )
                                req_mapping = mappings[req]
                                if req_mapping is not None and req_mapping.is_embedded:
                                    profit_vals[result_index] += req.profit
                                    _compute_mapping_load(load, req, req_mapping)

                            edge_mapping_failed, lp_failed, is_embedded, node_mapping_failed = self._count_mapping_status(result)
                            num_edge_mapping_failed += edge_mapping_failed
                            num_initial_lp_failed += lp_failed
                            num_node_mapping_failed += node_mapping_failed

                            max_edge_load, max_node_load = get_max_node_and_edge_load(load, scenario.substrate)
                            max_node_load_vals[result_index] = max_node_load
                            max_edge_load_vals[result_index] = max_edge_load
                            total_runtime_vals[result_index] = result.total_runtime

                        reduced = ReducedOfflineViNEResultCollection(
                            max_node_load=get_aggregated_data(max_node_load_vals),
                            max_edge_load=get_aggregated_data(max_edge_load_vals),
                            total_runtime=get_aggregated_data(total_runtime_vals),
                            profit=get_aggregated_data(profit_vals),
                            runtime_per_request=get_aggregated_data(runtimes_per_request_vals),
                            num_initial_lp_failed=num_initial_lp_failed,
                            num_node_mapping_failed=num_node_mapping_failed,
                            num_edge_mapping_failed=num_edge_mapping_failed,
                            num_req_with_profit=number_of_req_profit,
                            original_number_requests=number_of_requests
                        )
                        ssd_reduced[algorithm][scenario_id][exec_id][vine_settings].append(reduced)
        del scenario_solution_storage.scenario_parameter_container.scenario_list
        del scenario_solution_storage.scenario_parameter_container.scenario_triple
        scenario_solution_storage.algorithm_scenario_solution_dictionary = ssd_reduced

        logger.info("Writing result pickle to {}".format(reduced_baseline_solutions_output_pickle_path))
        with open(reduced_baseline_solutions_output_pickle_path, "wb") as f:
            pickle.dump(scenario_solution_storage, f)
        logger.info("All done.")
        return scenario_solution_storage

    def _count_mapping_status(self, vine_result):
        assert isinstance(vine_result, vine.OfflineViNEResult)
        num_is_embedded = 0
        num_initial_lp_failed = 0
        num_node_mapping_failed = 0
        num_edge_mapping_failed = 0
        for status in vine_result.mapping_status_per_request.values():
            if status == vine.ViNEMappingStatus.is_embedded:
                num_is_embedded += 1
            elif status == vine.ViNEMappingStatus.initial_lp_failed:
                num_initial_lp_failed += 1
            elif status == vine.ViNEMappingStatus.node_mapping_failed:
                num_node_mapping_failed += 1
            elif status == vine.ViNEMappingStatus.edge_mapping_failed:
                num_edge_mapping_failed += 1
            else:
                raise ValueError("Unexpected mapping status!")
        return num_edge_mapping_failed, num_initial_lp_failed, num_is_embedded, num_node_mapping_failed


class RandRoundSepLPOptDynVMPCollectionResultReducer(object):

    def __init__(self):
        pass

    def reduce_randround_result_collection(self,
                                           randround_solutions_input_pickle_name,
                                           reduced_randround_solutions_output_pickle_name=None):

        randround_solutions_input_pickle_path = os.path.join(util.ExperimentPathHandler.INPUT_DIR,
                                                             randround_solutions_input_pickle_name)

        if reduced_randround_solutions_output_pickle_name is None:
            file_basename = os.path.basename(randround_solutions_input_pickle_path).split(".")[0]
            reduced_randround_solutions_output_pickle_path = os.path.join(util.ExperimentPathHandler.OUTPUT_DIR,
                                                                          file_basename + "_reduced.pickle")
        else:
            reduced_randround_solutions_output_pickle_path = os.path.join(util.ExperimentPathHandler.OUTPUT_DIR,
                                                                          randround_solutions_input_pickle_name)

        logger.info("\nWill read from ..\n\t{} \n\t\tand store reduced data into\n\t{}\n".format(
            randround_solutions_input_pickle_path, reduced_randround_solutions_output_pickle_path))

        logger.info("Reading pickle file at {}".format(randround_solutions_input_pickle_path))
        with open(randround_solutions_input_pickle_path, "rb") as f:
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

        logger.info("Writing result pickle to {}".format(reduced_randround_solutions_output_pickle_path))
        with open(reduced_randround_solutions_output_pickle_path, "w") as f:
            pickle.dump(sss, f)
        logger.info("All done.")
        return sss

    def reduce_single_solution(self, solution):
        if solution is None:
            return None
        assert isinstance(solution, treewidth_model.RandRoundSepLPOptDynVMPCollectionResult)

        max_node_loads = {}
        max_edge_loads = {}
        rounding_runtimes = {}
        profits = {}

        for algorithm_sub_parameters, rounding_result_list in solution.solutions.items():
            max_node_loads[algorithm_sub_parameters] = []
            max_edge_loads[algorithm_sub_parameters] = []
            rounding_runtimes[algorithm_sub_parameters] = []
            profits[algorithm_sub_parameters] = []

            for rounding_result in rounding_result_list:
                max_node_loads[algorithm_sub_parameters].append(rounding_result.max_node_load)
                max_edge_loads[algorithm_sub_parameters].append(rounding_result.max_edge_load)
                rounding_runtimes[algorithm_sub_parameters].append(rounding_result.time_to_round_solution)
                profits[algorithm_sub_parameters].append(rounding_result.profit)

        for algorithm_sub_parameters in solution.solutions.keys():
            max_node_loads[algorithm_sub_parameters] = get_aggregated_data(max_node_loads[algorithm_sub_parameters])
            max_edge_loads[algorithm_sub_parameters] = get_aggregated_data(max_edge_loads[algorithm_sub_parameters])
            rounding_runtimes[algorithm_sub_parameters] = get_aggregated_data(rounding_runtimes[algorithm_sub_parameters])
            profits[algorithm_sub_parameters] = get_aggregated_data(profits[algorithm_sub_parameters])

        assert isinstance(solution.lp_computation_information, treewidth_model.SeparationLPSolution)
        # TODO Check which information is actually of interest
        # TODO Some of the data can be reduced further (store only mean and std. dev.)

        solution = ReducedRandRoundSepLPOptDynVMPCollectionResult(
            lp_time_preprocess=solution.lp_computation_information.time_preprocessing,
            lp_time_tree_decomposition=get_aggregated_data(solution.lp_computation_information.tree_decomp_runtimes),
            lp_time_dynvmp_initialization=get_aggregated_data(solution.lp_computation_information.dynvmp_init_runtimes),
            lp_time_gurobi_optimization=get_aggregated_data(solution.lp_computation_information.gurobi_runtimes),
            lp_time_optimization=solution.lp_computation_information.time_optimization,
            lp_status=solution.lp_computation_information.status,
            lp_profit=solution.lp_computation_information.profit,
            lp_generated_columns=solution.lp_computation_information.number_of_generated_mappings,
            max_node_loads=max_node_loads,
            max_edge_loads=max_edge_loads,
            rounding_runtimes=rounding_runtimes,
            profits=profits,
        )
        return solution


def _initialize_load_dict(scenario):
    load = dict([((u, v), 0.0) for (u, v) in scenario.substrate.edges])
    for u in scenario.substrate.nodes:
        for t in scenario.substrate.node[u]['supported_types']:
            load[(t, u)] = 0.0
    return load


def _compute_mapping_load(load, req, req_mapping):
    for i, u in req_mapping.mapping_nodes.iteritems():
        node_demand = req.get_node_demand(i)
        load[(req.get_type(i), u)] += node_demand

    if isinstance(req_mapping, solutions.Mapping):
        _compute_mapping_edge_load_unsplittable(load, req, req_mapping)
    elif isinstance(req_mapping, vine.SplittableMapping):
        _compute_mapping_edge_load_splittable(load, req, req_mapping)
    return load


def _compute_mapping_edge_load_unsplittable(load, req, req_mapping):
    for ij, sedge_list in req_mapping.mapping_edges.iteritems():
        edge_demand = req.get_edge_demand(ij)
        for uv in sedge_list:
            load[uv] += edge_demand


def _compute_mapping_edge_load_splittable(load, req, req_mapping):
    for ij, edge_vars_dict in req_mapping.mapping_edges.iteritems():
        edge_demand = req.get_edge_demand(ij)
        for uv, x in edge_vars_dict.items():
            load[uv] += edge_demand * x


def get_max_node_and_edge_load(load_dict, substrate):
    max_node_load = 0
    max_edge_load = 0
    for resource, value in load_dict.iteritems():
        x, y = resource
        if resource in substrate.edges:
            max_edge_load = max(max_edge_load, value)
        elif x in substrate.get_types() and y in substrate.nodes:
            max_node_load = max(max_node_load, value)
        else:
            raise ValueError("Invalid resource {}".format(resource))
    return max_edge_load, max_node_load
