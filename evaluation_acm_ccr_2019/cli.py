# MIT License
#
# Copyright (c) 2016-2018 Matthias Rost, Elias Doehne
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
import sys
import logging

import click
from . import treewidth_computation_experiments
from . import treewidth_computation_plots
from . import runtime_comparison_separation_dynvmp_vs_lp as sep_dynvmp_vs_lp
from . import plot_data, algorithm_heatmap_plots, runtime_evaluation
from alib import util
from alib import datamodel

try:
    import cPickle as pickle
except ImportError:
    import pickle


def initialize_logger(filename, log_level_print, log_level_file, allow_override=False):
    log_level_print = logging._levelNames[log_level_print.upper()]
    log_level_file = logging._levelNames[log_level_file.upper()]
    util.initialize_root_logger(filename, log_level_print, log_level_file, allow_override=allow_override)

@click.group()
def cli():
    """
    This command-line interface allows you to access major parts of the VNEP-Approx framework
    developed by Matthias Rost, Elias Döhne, Alexander Elvers, and Tom Koch.
    In particular, it allows to reproduce the results presented in the paper:

    "Parametrized Complexity of Virtual Network Embeddings: Dynamic & Linear Programming Approximations": Matthias Rost, Elias Döhne, Stefan Schmid. ACM CCR January 2019

    Note that each commands provides a help page. To access the help, simply type the commmand and --help.

    """
    pass


@cli.command(short_help="Generate random graphs and compute the treewidth using Tamaki's algorithm.")
@click.argument('yaml_parameter_file', type=click.File('r'))
@click.option('--threads', default=1)
@click.option('--timeout', type=click.INT, default=-1)
@click.option('--remove_intermediate_solutions/--keep_intermediate_solutions', is_flag=True, default=False, help="shall intermediate solutions be removed after execution?")
def execute_treewidth_computation_experiment(yaml_parameter_file, threads, timeout, remove_intermediate_solutions):
    click.echo('Generate Scenarios for evaluation of the treewidth model')

    util.ExperimentPathHandler.initialize()

    if timeout <= 0:
        timeout = None

    file_basename = os.path.basename(yaml_parameter_file.name).split(".")[0].lower()
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR, "{}_parent.log".format(file_basename))
    output_file = os.path.join(util.ExperimentPathHandler.OUTPUT_DIR,
                               "{}_results_{{process_index}}.pickle".format(file_basename))
    util.initialize_root_logger(log_file)
    treewidth_computation_experiments.run_experiment_from_yaml(yaml_parameter_file,
                                                               output_file,
                                                               threads,
                                                               timeout,
                                                               remove_intermediate_solutions)

@cli.command(short_help="Extracts undirected graphs from treewidth experiments")
@click.argument('input_pickle_file', type=click.Path())
@click.argument('output_pickle_file', type=click.Path())
@click.argument('min_tw', type=click.INT)
@click.argument('max_tw', type=click.INT)
@click.option('--min_nodes', type=click.INT, default=0)
@click.option('--max_nodes', type=click.INT, default=sys.maxint)
@click.option('--min_conn_prob', type=click.FLOAT, default=0)
@click.option('--max_conn_prob', type=click.FLOAT, default=1.0)
def create_undirected_graph_storage_from_treewidth_experiments(input_pickle_file,
                                                               output_pickle_file,
                                                               min_tw,
                                                               max_tw,
                                                               min_nodes,
                                                               max_nodes,
                                                               min_conn_prob,
                                                               max_conn_prob):
    util.ExperimentPathHandler.initialize()
    file_basename = os.path.basename(input_pickle_file).split(".")[0].lower()
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR, "creation_undirected_graph_storage_from_treewidth_{}.log".format(file_basename))
    util.initialize_root_logger(log_file)

    # get root logger
    logger = logging.getLogger()

    graph_storage = datamodel.UndirectedGraphStorage(parameter_name="treewidth")

    input_contents = None
    logger.info("Reading file {}".format(input_pickle_file))
    with open(input_pickle_file, "r") as f:
        input_contents = pickle.load(f)

    for number_of_nodes in input_contents.keys():
        logger.info("Handling graphs stored for number of nodes {}".format(number_of_nodes))
        data_for_nodes = input_contents[number_of_nodes]
        for connection_probability in data_for_nodes.keys():
            if connection_probability < min_conn_prob or connection_probability > max_conn_prob:
                continue
            list_of_results = data_for_nodes[connection_probability]
            for treewidth_computation_result in list_of_results:
                result_tw = treewidth_computation_result.treewidth
                if result_tw is None:
                    continue
                if result_tw < min_tw or result_tw > max_tw:
                    continue
                undirected_edge_representation = treewidth_computation_result.undirected_graph_edge_representation
                if undirected_edge_representation is None:
                    continue
                graph_storage.add_graph_as_edge_representation(result_tw, undirected_edge_representation)

    logger.info("Writing file {}".format(output_pickle_file))
    with open(output_pickle_file, "w") as f:
        pickle.dump(graph_storage, f)


@cli.command(short_help="Generate plots for treewidth computation by Tamaki's algorithm")
@click.argument('parameters_file', type=click.File('r'))
@click.argument('results_pickle_file', type=click.File('r'))
@click.argument('output_path', type=click.Path())
@click.option('--output_filetype', type=click.Choice(['png', 'pdf', 'eps']), default="png", help="the filetype which shall be created")
def treewidth_plot_computation_results(parameters_file, results_pickle_file, output_path, output_filetype):
    treewidth_computation_plots.make_plots(parameters_file, results_pickle_file, output_path, output_filetype)


@cli.command(short_help="Extracts data to be plotted for the randomized rounding algorithms (using the separation LP and DynVMP)")
@click.argument('input_pickle_file', type=click.Path())
@click.option('--output_pickle_file', type=click.Path(), default=None, help="file to write to")
@click.option('--log_level_print', type=click.STRING, default="info", help="log level for stdout")
@click.option('--log_level_file', type=click.STRING, default="debug", help="log level for log file")
def reduce_to_plotdata_rr_seplp_optdynvmp(input_pickle_file, output_pickle_file, log_level_print, log_level_file):
    """ Given a scenario solution pickle (input_pickle_file) this function extracts data
        to be plotted and writes it to --output_pickle_file. If --output_pickle_file is not
        given, a default name (derived from the input's basename) is derived.

        The input_file must be contained in ALIB_EXPERIMENT_HOME/input and the output
        will be written to ALIB_EXPERIMENT_HOME/output while the log is saved in
        ALIB_EXPERIMENT_HOME/log.
    """
    util.ExperimentPathHandler.initialize(check_emptiness_log=False, check_emptiness_output=False)
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR,
                            "reduce_{}.log".format(os.path.basename(input_pickle_file)))
    initialize_logger(log_file, log_level_print, log_level_file)
    reducer = plot_data.RandRoundSepLPOptDynVMPCollectionResultReducer()
    reducer.reduce_randround_result_collection(input_pickle_file, output_pickle_file)


@cli.command(short_help="Extracts data to be plotted the vine executions")
@click.argument('input_pickle_file', type=click.Path())
@click.option('--output_pickle_file', type=click.Path(), default=None, help="file to write to")
@click.option('--log_level_print', type=click.STRING, default="info", help="log level for stdout")
@click.option('--log_level_file', type=click.STRING, default="debug", help="log level for log file")
def reduce_to_plotdata_vine(input_pickle_file, output_pickle_file, log_level_print, log_level_file):
    """ Given a scenario solution pickle (input_pickle_file) this function extracts data
        to be plotted and writes it to --output_pickle_file. If --output_pickle_file is not
        given, a default name (derived from the input's basename) is derived.

        The input_file must be contained in ALIB_EXPERIMENT_HOME/input and the output
        will be written to ALIB_EXPERIMENT_HOME/output while the log is saved in
        ALIB_EXPERIMENT_HOME/log.
    """
    util.ExperimentPathHandler.initialize(check_emptiness_log=False, check_emptiness_output=False)
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR,
                            "reduce_{}.log".format(os.path.basename(input_pickle_file)))
    initialize_logger(log_file, log_level_print, log_level_file)
    reducer = plot_data.OfflineViNEResultCollectionReducer()
    reducer.reduce_vine_result_collection(input_pickle_file, output_pickle_file)


def collect_existing_alg_ids(execution_parameter_container):
    list_of_alg_ids = []
    for alg_dict in execution_parameter_container.algorithm_parameter_list:
        if alg_dict['ALG_ID'] not in list_of_alg_ids:
            list_of_alg_ids.append(alg_dict['ALG_ID'])
    return list_of_alg_ids


def query_algorithm_id_and_execution_id(logger,
                                        pickle_name,
                                        execution_parameter_container,
                                        algorithm_id,
                                        execution_config_id,
                                        query_even_when_only_one_option=False):

    list_of_alg_ids = collect_existing_alg_ids(execution_parameter_container)
    if algorithm_id is not None and algorithm_id not in list_of_alg_ids:
        logger.error("The provided algorithm id {} for the pickle {} is not contained in the contained list of algorithm ids: {}.".format(algorithm_id, pickle_name, list_of_alg_ids))
        algorithm_id=None
    if algorithm_id is None:
        if len(list_of_alg_ids) == 0:
            raise RuntimeError("It seems that the pickle {} does not contain any algorithm information. Abort.".format(pickle_name))
        if not query_even_when_only_one_option and len(list_of_alg_ids) == 1:
            algorithm_id = list_of_alg_ids[0]
            logger.info(
                " .. selected algorithm id '{}' for the pickle {}".format(algorithm_id, pickle_name))
        else:
            logger.info("\nAvailable algorithm ids for the pickle {} are: {}".format(pickle_name, list_of_alg_ids))
            algorithm_id = click.prompt("Please select an algorithm id for the pickle {}:".format(pickle_name), type=click.Choice(list_of_alg_ids))

    list_of_suitable_execution_ids = [x for x in execution_parameter_container.get_execution_ids(ALG_ID=algorithm_id)]
    if execution_config_id is not None and execution_config_id not in list_of_suitable_execution_ids:
        logger.error(
            "The provided execution id {} for the algorithm id {} for the pickle {} is not contained in the contained list of algorithm ids: {}.".format(
                execution_config_id, algorithm_id, pickle_name, list_of_alg_ids))
        execution_config_id=None
    if execution_config_id is None:
        if len(list_of_suitable_execution_ids) == 0:
            raise RuntimeError(
                "It seems that the pickle {} does not contain any suitable execution ids for algorithm id {}. Abort.".format(
                    pickle_name, algorithm_id))
        if not query_even_when_only_one_option and len(list_of_suitable_execution_ids) == 1:
            execution_config_id = list_of_suitable_execution_ids[0]
            logger.info(
                " .. selected execution id '{}' for the pickle {} as it is the only one for algorithm id {}".format(execution_config_id, pickle_name, algorithm_id))
        else:
            logger.info("\nAvailable execution ids for the algorithm id {} of the pickle {} are...".format(algorithm_id,
                                                                                                           pickle_name,
                                                                                                           list_of_alg_ids))
            for execution_id in list_of_suitable_execution_ids:
                logger.info(
                    "\nExecution id {} corresponds to {}".format(execution_id,execution_parameter_container.algorithm_parameter_list[execution_id]))


            execution_config_id = click.prompt("Please select an execution id for the algorithm id {} for the pickle {}:".format(algorithm_id,
                                                                                                                                 pickle_name),
                                               type=click.Choice([str(x) for x in list_of_suitable_execution_ids]))
            execution_config_id = int(execution_config_id)

    return algorithm_id, execution_config_id

@cli.command(short_help="Create plot comparing runtime of cactus lp and the separation lp with dynvmp")
@click.argument('sep_lp_dynvmp_reduced_pickle', type=click.Path())     #pickle in ALIB_EXPERIMENT_HOME/input storing randround results
@click.argument('cactus_lp_reduced_pickle', type=click.Path())      #pickle in ALIB_EXPERIMENT_HOME/input storing baseline results
@click.argument('output_directory', type=click.Path())          #path to which the result will be written
@click.option('--sep_lp_dynvmp_algorithm_id', type=click.STRING, default=None, help="algorithm id of sep_lp_dynvmp algorithm; if not given it will be asked for.")
@click.option('--sep_lp_dynvmp_execution_config', type=click.INT, default=None, help="execution (configuration) id of sep_lp_dynvmp alg; if not given it will be asked for.")
@click.option('--cactus_lp_algorithm_id', type=click.STRING, default=None, help="algorithm id of cactus lp algorithm; if not given it will be asked for.")
@click.option('--cactus_lp_execution_config', type=click.INT, default=None, help="execution (configuration) id of cactus lp alg; if not given it will be asked for.")
@click.option('--exclude_generation_parameters', type=click.STRING, default=None, help="generation parameters that shall be excluded. "
                                                                                       "Must ge given as python evaluable list of dicts. "
                                                                                       "Example format: \"{'number_of_requests': [20]}\"")
@click.option('--output_filetype', type=click.Choice(['png', 'pdf', 'eps']), default="png", help="the filetype which shall be created")
@click.option('--log_level_print', type=click.STRING, default="info", help="log level for stdout")
@click.option('--log_level_file', type=click.STRING, default="debug", help="log level for stdout")
@click.option('--request_sets', type=click.STRING, default="[[40,60],[80,100]]", help="list of request lists to aggregate")
def evaluate_separation_vs_cactus_lp(sep_lp_dynvmp_reduced_pickle,
                                     cactus_lp_reduced_pickle,
                                     output_directory,
                                     sep_lp_dynvmp_algorithm_id,
                                     sep_lp_dynvmp_execution_config,
                                     cactus_lp_algorithm_id,
                                     cactus_lp_execution_config,
                                     exclude_generation_parameters,
                                     output_filetype,
                                     log_level_print,
                                     log_level_file,
                                     request_sets):

    util.ExperimentPathHandler.initialize(check_emptiness_log=False, check_emptiness_output=False)
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR,
                            "evaluate_pickles_{}_{}.log".format(os.path.basename(sep_lp_dynvmp_reduced_pickle),
                                                                os.path.basename(cactus_lp_reduced_pickle)))
    initialize_logger(log_file, log_level_print, log_level_file, allow_override=True)

    lp_sep_pickle_path = os.path.join(util.ExperimentPathHandler.INPUT_DIR, sep_lp_dynvmp_reduced_pickle)
    randround_pickle_path = os.path.join(util.ExperimentPathHandler.INPUT_DIR, cactus_lp_reduced_pickle)

    #get root logger
    logger = logging.getLogger()

    logger.info("Reading reduced lp_sep_pickle pickle at {}".format(lp_sep_pickle_path))
    sep_lp_dynvmp_results = None
    with open(lp_sep_pickle_path, "rb") as f:
        sep_lp_dynvmp_results = pickle.load(f)

    logger.info("Reading reduced randround pickle at {}".format(randround_pickle_path))
    randround_results = None
    with open(randround_pickle_path, "rb") as f:
        randround_results = pickle.load(f)

    logger.info("Loading algorithm identifiers and execution ids..")

    sep_lp_dynvmp_algorithm_id, sep_lp_dynvmp_execution_config = query_algorithm_id_and_execution_id(logger,
                                                                                                     sep_lp_dynvmp_reduced_pickle,
                                                                                                     sep_lp_dynvmp_results.execution_parameter_container,
                                                                                                     sep_lp_dynvmp_algorithm_id,
                                                                                                     sep_lp_dynvmp_execution_config)

    cactus_lp_algorithm_id, cactus_lp_execution_config = query_algorithm_id_and_execution_id(logger,
                                                                                             cactus_lp_reduced_pickle,
                                                                                             randround_results.execution_parameter_container,
                                                                                             cactus_lp_algorithm_id,
                                                                                             cactus_lp_execution_config)

    output_directory = os.path.normpath(output_directory)

    logger.info("Setting output path to {}".format(output_directory))

    if exclude_generation_parameters is not None:
        exclude_generation_parameters = eval(exclude_generation_parameters)

    logger.info("Starting evaluation...")

    logger.info("Trying to parse request sets (expecting list of lists)")
    request_sets_parsed = eval(request_sets)

    sep_dynvmp_vs_lp.evaluate_baseline_and_randround(sep_lp_dynvmp_results,
                                                     sep_lp_dynvmp_algorithm_id,
                                                     sep_lp_dynvmp_execution_config,
                                                     randround_results,
                                                     cactus_lp_algorithm_id,
                                                     cactus_lp_execution_config,
                                                     exclude_generation_parameters=exclude_generation_parameters,
                                                     output_path=output_directory,
                                                     output_filetype=output_filetype,
                                                     request_sets=request_sets_parsed)


@cli.command(short_help="Create plots comparing randomized rounding solutions (using the separation LP) with ViNE solutions")
@click.argument('sep_lp_dynvmp_reduced_pickle', type=click.Path())     #pickle in ALIB_EXPERIMENT_HOME/input storing randround results
@click.argument('vine_reduced_pickle', type=click.Path())      #pickle in ALIB_EXPERIMENT_HOME/input storing baseline results
@click.argument('output_directory', type=click.Path())          #path to which the result will be written
@click.option('--sep_lp_dynvmp_algorithm_id', type=click.STRING, default=None, help="algorithm id of sep_lp_dynvmp algorithm; if not given it will be asked for.")
@click.option('--sep_lp_dynvmp_execution_config', type=click.INT, default=None, help="execution (configuration) id of sep_lp_dynvmp alg; if not given it will be asked for.")
@click.option('--vine_algorithm_id', type=click.STRING, default=None, help="algorithm id of randround algorithm; if not given it will be asked for.")
@click.option('--vine_execution_config', type=click.INT, default=None, help="execution (configuration) id of randround alg; if not given it will be asked for.")
@click.option('--exclude_generation_parameters', type=click.STRING, default=None, help="generation parameters that shall be excluded. "
                                                                                       "Must ge given as python evaluable list of dicts. "
                                                                                       "Example format: \"{'number_of_requests': [20]}\"")
@click.option('--overwrite/--no_overwrite', default=True, help="overwrite existing files?")
@click.option('--papermode/--non-papermode', default=True, help="output 'paper-ready' figures or figures containing additional statistical data?")
@click.option('--output_filetype', type=click.Choice(['png', 'pdf', 'eps']), default="png", help="the filetype which shall be created")
@click.option('--log_level_print', type=click.STRING, default="info", help="log level for stdout")
@click.option('--log_level_file', type=click.STRING, default="debug", help="log level for stdout")
@click.option('--request_sets', type=click.STRING, default="[[40,60],[80,100]]", help="list of request lists to aggregate")
def evaluate_separation_randround_vs_vine(sep_lp_dynvmp_reduced_pickle,
                                          vine_reduced_pickle,
                                          output_directory,
                                          sep_lp_dynvmp_algorithm_id,
                                          sep_lp_dynvmp_execution_config,
                                          vine_algorithm_id,
                                          vine_execution_config,
                                          exclude_generation_parameters,
                                          overwrite,
                                          papermode,
                                          output_filetype,
                                          log_level_print,
                                          log_level_file,
                                          request_sets):

    util.ExperimentPathHandler.initialize(check_emptiness_log=False, check_emptiness_output=False)
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR,
                            "evaluate_pickles_{}_{}.log".format(os.path.basename(sep_lp_dynvmp_reduced_pickle),
                                                                os.path.basename(vine_reduced_pickle)))
    initialize_logger(log_file, log_level_print, log_level_file, allow_override=True)

    lp_sep_pickle_path = os.path.join(util.ExperimentPathHandler.INPUT_DIR, sep_lp_dynvmp_reduced_pickle)
    vine_pickle_path = os.path.join(util.ExperimentPathHandler.INPUT_DIR, vine_reduced_pickle)

    #get root logger
    logger = logging.getLogger()

    logger.info("Reading reduced lp_sep_pickle pickle at {}".format(lp_sep_pickle_path))
    sep_lp_dynvmp_results = None
    with open(lp_sep_pickle_path, "rb") as f:
        sep_lp_dynvmp_results = pickle.load(f)

    logger.info("Reading reduced vine pickle at {}".format(vine_pickle_path))
    vine_results = None
    with open(vine_pickle_path, "rb") as f:
        vine_results = pickle.load(f)

    logger.info("Loading algorithm identifiers and execution ids..")

    sep_lp_dynvmp_algorithm_id, sep_lp_dynvmp_execution_config = query_algorithm_id_and_execution_id(logger,
                                                                                                     sep_lp_dynvmp_reduced_pickle,
                                                                                                     sep_lp_dynvmp_results.execution_parameter_container,
                                                                                                     sep_lp_dynvmp_algorithm_id,
                                                                                                     sep_lp_dynvmp_execution_config)

    vine_algorithm_id, vine_execution_config = query_algorithm_id_and_execution_id(logger,
                                                                                   vine_reduced_pickle,
                                                                                   vine_results.execution_parameter_container,
                                                                                   vine_algorithm_id,
                                                                                   vine_execution_config)

    output_directory = os.path.normpath(output_directory)

    logger.info("Setting output path to {}".format(output_directory))

    if exclude_generation_parameters is not None:
        exclude_generation_parameters = eval(exclude_generation_parameters)

    logger.info("Starting evaluation...")

    logger.info("Trying to parse request sets (expecting list of lists)")
    request_sets_parsed = eval(request_sets)

    if exclude_generation_parameters is not None:
        exclude_generation_parameters = eval(exclude_generation_parameters)

    algorithm_heatmap_plots.evaluate_vine_and_randround(
        dc_vine=vine_results,
        vine_algorithm_id=vine_algorithm_id,
        vine_execution_id=vine_execution_config,
        dc_randround_seplp_dynvmp=sep_lp_dynvmp_results,
        randround_seplp_algorithm_id=sep_lp_dynvmp_algorithm_id,
        randround_seplp_execution_id=sep_lp_dynvmp_execution_config,
        exclude_generation_parameters=exclude_generation_parameters,
        parameter_filter_keys=None,
        show_plot=False,
        save_plot=True,
        overwrite_existing_files=overwrite,
        forbidden_scenario_ids=None,
        papermode=papermode,
        maxdepthfilter=2,
        output_path=output_directory,
        output_filetype=output_filetype,
        request_sets=request_sets_parsed
    )

    runtime_evaluation.evaluate_randround_runtimes(
        dc_randround_seplp_dynvmp=sep_lp_dynvmp_results,
        randround_seplp_algorithm_id=sep_lp_dynvmp_algorithm_id,
        randround_seplp_execution_id=sep_lp_dynvmp_execution_config,
        exclude_generation_parameters=exclude_generation_parameters,
        parameter_filter_keys=None,
        show_plot=False,
        save_plot=True,
        overwrite_existing_files=overwrite,
        forbidden_scenario_ids=None,
        papermode=papermode,
        maxdepthfilter=2,
        output_path=output_directory,
        output_filetype=output_filetype
    )





# --------------------------------------- LATENCY EVALUATION ------------------------------------


@cli.command(short_help="Extracts data to be plotted for the latency evaluation of the randomized rounding algorithms")
@click.argument('input_pickle_file', type=click.Path())
@click.option('--output_pickle_file', type=click.Path(), default=None, help="file to write to")
@click.option('--log_level_print', type=click.STRING, default="info", help="log level for stdout")
@click.option('--log_level_file', type=click.STRING, default="debug", help="log level for log file")
def reduce_to_plotdata_rr_seplp_optdynvmp(input_pickle_file, output_pickle_file, log_level_print, log_level_file):
    """ Given a scenario solution pickle (input_pickle_file) this function extracts data
        to be plotted and writes it to --output_pickle_file. If --output_pickle_file is not
        given, a default name (derived from the input's basename) is derived.

        The input_file must be contained in ALIB_EXPERIMENT_HOME/input and the output
        will be written to ALIB_EXPERIMENT_HOME/output while the log is saved in
        ALIB_EXPERIMENT_HOME/log.
    """
    util.ExperimentPathHandler.initialize(check_emptiness_log=False, check_emptiness_output=False)
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR,
                            "reduce_{}.log".format(os.path.basename(input_pickle_file)))
    initialize_logger(log_file, log_level_print, log_level_file)
    reducer = plot_data.RandRoundSepLPOptDynVMPCollectionResultReducer()
    reducer.reduce_randround_result_collection(input_pickle_file, output_pickle_file)




@cli.command(short_help="Create plots comparing the DynVMP runtime with and without considering latencies")
@click.argument('sep_lp_dynvmp_reduced_pickle', type=click.Path())     #pickle in ALIB_EXPERIMENT_HOME/input storing randround results
@click.argument('vine_reduced_pickle', type=click.Path())      #pickle in ALIB_EXPERIMENT_HOME/input storing baseline results
@click.argument('output_directory', type=click.Path())          #path to which the result will be written
@click.option('--sep_lp_dynvmp_algorithm_id', type=click.STRING, default=None, help="algorithm id of sep_lp_dynvmp algorithm; if not given it will be asked for.")
@click.option('--sep_lp_dynvmp_execution_config', type=click.INT, default=None, help="execution (configuration) id of sep_lp_dynvmp alg; if not given it will be asked for.")
@click.option('--vine_algorithm_id', type=click.STRING, default=None, help="algorithm id of randround algorithm; if not given it will be asked for.")
@click.option('--vine_execution_config', type=click.INT, default=None, help="execution (configuration) id of randround alg; if not given it will be asked for.")
@click.option('--exclude_generation_parameters', type=click.STRING, default=None, help="generation parameters that shall be excluded. "
                                                                                       "Must ge given as python evaluable list of dicts. "
                                                                                       "Example format: \"{'number_of_requests': [20]}\"")
@click.option('--overwrite/--no_overwrite', default=True, help="overwrite existing files?")
@click.option('--papermode/--non-papermode', default=True, help="output 'paper-ready' figures or figures containing additional statistical data?")
@click.option('--output_filetype', type=click.Choice(['png', 'pdf', 'eps']), default="png", help="the filetype which shall be created")
@click.option('--log_level_print', type=click.STRING, default="info", help="log level for stdout")
@click.option('--log_level_file', type=click.STRING, default="debug", help="log level for stdout")
@click.option('--request_sets', type=click.STRING, default="[[40,60],[80,100]]", help="list of request lists to aggregate")
def evaluate_separation_with_latencies(no_latencies_reduced_pickle,
                                          with_latencies_reduced_pickle,
                                          output_directory,
                                          sep_lp_dynvmp_algorithm_id,
                                          sep_lp_dynvmp_execution_config,
                                          vine_algorithm_id,
                                          vine_execution_config,
                                          exclude_generation_parameters,
                                          overwrite,
                                          papermode,
                                          output_filetype,
                                          log_level_print,
                                          log_level_file,
                                          request_sets):

    util.ExperimentPathHandler.initialize(check_emptiness_log=False, check_emptiness_output=False)
    log_file = os.path.join(util.ExperimentPathHandler.LOG_DIR,
                            "evaluate_pickles_{}_{}.log".format(os.path.basename(no_latencies_reduced_pickle),
                                                                os.path.basename(with_latencies_reduced_pickle)))
    initialize_logger(log_file, log_level_print, log_level_file, allow_override=True)

    no_lat_pickle_path = os.path.join(util.ExperimentPathHandler.INPUT_DIR, no_latencies_reduced_pickle)
    with_lat_pickle_path = os.path.join(util.ExperimentPathHandler.INPUT_DIR, with_latencies_reduced_pickle)

    #get root logger
    logger = logging.getLogger()

    logger.info("Reading reduced no_latencies pickle at {}".format(no_latencies_reduced_pickle))
    no_latencies_results = None
    with open(no_lat_pickle_path, "rb") as f:
        no_latencies_results = pickle.load(f)

    logger.info("Reading reduced with_latencies pickle at {}".format(with_latencies_reduced_pickle))
    with_latencies_results = None
    with open(with_lat_pickle_path, "rb") as f:
        with_latencies_results = pickle.load(f)

    logger.info("Loading algorithm identifiers and execution ids..")

    no_latencies_algorithm_id, no_latencies_execution_config = query_algorithm_id_and_execution_id(logger,
                                                                                                     no_latencies_reduced_pickle,
                                                                                                     no_latencies_results.execution_parameter_container,
                                                                                                     sep_lp_dynvmp_algorithm_id,
                                                                                                     sep_lp_dynvmp_execution_config)

    with_latencies_algorithm_id, with_latencies_execution_config = query_algorithm_id_and_execution_id(logger,
                                                                                   with_latencies_reduced_pickle,
                                                                                   with_latencies_results.execution_parameter_container,
                                                                                   vine_algorithm_id,
                                                                                   vine_execution_config)

    output_directory = os.path.normpath(output_directory)

    logger.info("Setting output path to {}".format(output_directory))

    if exclude_generation_parameters is not None:
        exclude_generation_parameters = eval(exclude_generation_parameters)

    logger.info("Starting evaluation...")

    logger.info("Trying to parse request sets (expecting list of lists)")
    request_sets_parsed = eval(request_sets)

    if exclude_generation_parameters is not None:
        exclude_generation_parameters = eval(exclude_generation_parameters)

    algorithm_heatmap_plots.evaluate_vine_and_randround(
        dc_vine=with_latencies_results,
        vine_algorithm_id=vine_algorithm_id,
        vine_execution_id=vine_execution_config,
        dc_randround_seplp_dynvmp=no_latencies_results,
        randround_seplp_algorithm_id=sep_lp_dynvmp_algorithm_id,
        randround_seplp_execution_id=sep_lp_dynvmp_execution_config,
        exclude_generation_parameters=exclude_generation_parameters,
        parameter_filter_keys=None,
        show_plot=False,
        save_plot=True,
        overwrite_existing_files=overwrite,
        forbidden_scenario_ids=None,
        papermode=papermode,
        maxdepthfilter=2,
        output_path=output_directory,
        output_filetype=output_filetype,
        request_sets=request_sets_parsed
    )

    runtime_evaluation.evaluate_randround_runtimes(
        dc_randround_seplp_dynvmp=no_latencies_results,
        randround_seplp_algorithm_id=sep_lp_dynvmp_algorithm_id,
        randround_seplp_execution_id=sep_lp_dynvmp_execution_config,
        exclude_generation_parameters=exclude_generation_parameters,
        parameter_filter_keys=None,
        show_plot=False,
        save_plot=True,
        overwrite_existing_files=overwrite,
        forbidden_scenario_ids=None,
        papermode=papermode,
        maxdepthfilter=2,
        output_path=output_directory,
        output_filetype=output_filetype
    )

# --------------------------------------------- END ---------------------------------------------


if __name__ == '__main__':
    cli()
