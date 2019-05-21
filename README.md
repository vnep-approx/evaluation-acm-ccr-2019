
# Overview

This repository contains the evaluation code as well as the raw results presented in our published in the ACM Computer Communication Review Journal [1].

The implementation of the respective algorithms can be found in our separate python packages: 
- **[alib](https://github.com/vnep-approx/alib)**, providing for example the data model and the Mixed-Integer Program for the classic multi-commodity formulation, as well as
- **[vnep_approx](https://github.com/vnep-approx/vnep_approx)**, providing novel Linear Programming formulations, specifically the one based on the Dyn-VMP algorithm, as well as our proposed Randomized Rounding algorithms.
- **[evaluation_ifip_networking_2018](https://github.com/vnep-approx/evaluation_ifip_networking_2018)**, providing the base line LP solutions for our runtime comparison.

## Contents

- The folder **[evaluation_acm_ccr_2019](evaluation_acm_ccr_2019)** contains the actual python package, which can be easily installed using the provided setup.py. A more detailed explanation of the provided functionality can be found below.
- The folder **[sample](sample)** contains minimal samples for the three different evaluations presented in [1]:
  - **[sample/runtime_comparison_cactus_lp_vs_separation_dynvmp](sample/runtime_comparison_cactus_lp_vs_separation_dynvmp)** provides all configuration files and executable bash-scripts to generate scenarios (using cactus requests), run the cactus LP and the separation LP, and compare the runtimes.
  - **[sample/runtime_comparison_cactus_lp_vs_separation_dynvmp](sample/treewidth_study)** provides all configuration files and executable bash-scripts to generate random graph, run Tamaki's algorithm to compute the treewidth (exactly), and to evaluate the runtimes and the treewidth.
  - **[sample/vine_vs_randround](sample/vine_vs_randround)** provides all configuration files and executable bash-scripts to generate requests of varying treewidth, run vine and the randomized rounding heuristics, and compare the performance of the found solutions. 
Due to the size of the respective pickle files of the **actual data**, these files are not contained in the repository directly,  but are contained in the releases of this repository.


## Papers

**[1]** Matthias Rost, Elias Döhne, Stefan Schmid: Parametrized Complexity of Virtual Network Embeddings: Dynamic \& Linear Programming Approximations. [ACM CCR January 2019](https://ccronline.sigcomm.org/wp-content/uploads/2019/02/sigcomm-ccr-final255.pdf)

# Dependencies and Requirements

The **vnep_approx** library requires Python 2.7. Required python libraries: gurobipy, numpy, cPickle, networkx , matplotlib, **[alib](https://github.com/vnep-approx/alib)**, **[vnep-approx](https://github.com/vnep-approx/vnep-approx)**, and **[evaluation-ifip-networking-2018](https://github.com/vnep-approx/evaluation-ifip-networking-2018)**.  

Gurobi must be installed and the .../gurobi64/lib directory added to the environment variable LD_LIBRARY_PATH.

Furthermore, we use Tamaki's algorithm presented in his [paper at ESA 2017](http://drops.dagstuhl.de/opus/volltexte/2017/7880/pdf/LIPIcs-ESA-2017-68.pdf) to compute tree decompositions (efficiently). The corresponding GitHub repository [TCS-Meiji/PACE2017-TrackA](https://github.com/TCS-Meiji/PACE2017-TrackA) must be cloned locally and the environment variable **PACE_TD_ALGORITHM_PATH** must be set to point the location of the repository: PACE_TD_ALGORITHM_PATH="$PATH_TO_PACE/PACE2017-TrackA".

For generating and executing (etc.) experiments, the environment variable ALIB_EXPERIMENT_HOME must be set to a path, such that the subfolders input/ output/ and log/ exist.

**Note**: Our source was only tested on Linux (specifically Ubuntu 14/16).  

# Installation

To install the package, we provide a setup script. Simply execute from within evaluation_acm_ccr_2019's root directory: 

```
pip install .
```

Furthermore, if the code base will be edited by you, we propose to install it as editable:
```
pip install -e .
```
When choosing this option, sources are not copied during the installation but the local sources are used: changes to
the sources are directly reflected in the installed package.

We generally propose to install our libraries (i.e. **alib**, **vnep_approx**, **evaluation_ifip_networking_2018**) into a virtual environment.

# Usage

You may either use our code via our API by importing the library or via our command line interface:

```
python -m evaluation_acm_ccr_2019.cli --help
Usage: cli.py [OPTIONS] COMMAND [ARGS]...

  This command-line interface allows you to access major parts of the VNEP-
  Approx framework developed by Matthias Rost, Elias Döhne, Alexander
  Elvers, and Tom Koch. In particular, it allows to reproduce the results
  presented in the paper:

  "Parametrized Complexity of Virtual Network Embeddings: Dynamic & Linear
  Programming Approximations": Matthias Rost, Elias Döhne, Stefan Schmid.
  ACM CCR January 2019

  Note that each commands provides a help page. To access the help, simply
  type the commmand and --help.

Options:
  --help  Show this message and exit.

Commands:
  create_undirected_graph_storage_from_treewidth_experiments
                                  Extracts undirected graphs from treewidth
                                  experiments
  evaluate_separation_randround_vs_vine
                                  Create plots comparing randomized rounding
                                  solutions (using the separation LP) with
                                  ViNE solutions
  evaluate_separation_vs_cactus_lp
                                  Create plot comparing runtime of cactus lp
                                  and the separation lp with dynvmp
  execute_treewidth_computation_experiment
                                  Generate random graphs and compute the
                                  treewidth using Tamaki's algorithm.
  reduce_to_plotdata_rr_seplp_optdynvmp
                                  Extracts data to be plotted for the
                                  randomized rounding algorithms (using the
                                  separation LP and DynVMP)
  reduce_to_plotdata_vine         Extracts data to be plotted the vine
                                  executions
  treewidth_plot_computation_results
                                  Generate plots for treewidth computation by
                                  Tamaki's algorithm
```

# Step-by-Step Manual to Reproduce Results

The following worked on Ubuntu 16.04, but depending on the operating system or Linux variant,
some minor changes might be necessary. In the following, we outline the general idea of our framework
based on the examples provided in the **[sample](sample)** folder. In fact, the steps discussed below
can all be found in the respective bash-scripts **run_sample.sh**, which can be executed after having created
the virtual environment for the project and having installed all required dependencies.


## Creating a Virtual Environment and Installing Packages

First, create and activate a novel virtual environment for python2.7. 

```
virtualenv --python=python2.7 venv  #create new virtual environment in folder venv 
source venv/bin/activate            #activate the virtual environment
```

With the virtual environment still active, install the python extensions of [Gurobi](http://www.gurobi.com/) within the
virtual environment. Note that you need to first download and install a license of Gurobi (which is free for academic use). 
```
cd ~/programs/gurobi811/linux64/    #change to the directory of gurobi
python setup.py install             #install gurobipy within (!) the virtual environment
```

Then, assuming that all packages, i.e. **alib, vnep_approx**, **evaluation_ifip_networking_2018**, and **evaluation_acm_ccr_2019** are downloaded / cloned to the same directory, simply execute the following within each of the packages' root directories:

```
pip install -e .
```

## Setting up TCS-Meiji/PACE2017-TrackA

After cloning Tamaki's algorithm [TCS-Meiji/PACE2017-TrackA](https://github.com/TCS-Meiji/PACE2017-TrackA) 
and setting the environment variable with its location, the Java code needs to be compiled. 
JRE and JDK can be installed on Linux systems with (default packages should work fine):
```
sudo apt install default-jre
sudo apt install default-jdk
```
Running the `make` command in the project root should be successful.
To enable usage of the `tw-exact` shell script, it must be set to executable.
By default the `tw-extract` script comes with a setting of 30 Gb heap size allocation. In case of running our algorithm on a PC, the java
 command's input values might be set to some appropriately lower value (e.g. `-Xmx7g -Xms7g`). 

## Runtime Comparison Cactus LP vs. Separation LP
First, to use our framework, make sure that you set the environment variable **ALIB_EXPERIMENT_HOME** to a directory
containing (initially empty) folders **input/**, **output/**, and **log/**. Having said that, and activated the 
virtual environment created above, you can execute the following command to generate scenarios according to the parameters
specified in the file **[sample/runtime_comparison_cactus_lp_vs_separation_dynvmp/run_sample.sh](sample/runtime_comparison_cactus_lp_vs_separation_dynvmp/run_sample.sh)**. Most of the parameters of the file
**[sample_scenarios.yml](sample/sample_scenarios.yml)** should be quite self-explanatory and you might read-up on the
meaning of the parameters in the respective command-line interface helps.

While we have used the scenarios generated for the IFIP Networking 2018 evaluation, our example shows how to generate appropriate instances using:
```
#generate scenarios
python -m vnep_approx.cli generate_scenarios sample_scenarios.pickle sample_scenarios.yml
```
Above, the file sample_scenarios.yml details the (quite many) parameters for the instance generation.
Having generated the scenarios, both algorithms can be executed using the following commands:

```
#run randomized rounding algorithm using cactus formulation
python -m vnep_approx.cli start_experiment sample_randround_execution.yml 0 10000 --concurrent 2 --overwrite_existing_intermediate_solutions --remove_intermediate_solutions 

#run randomized rounding algorithm using separation lp with dynvmp
python -m vnep_approx.cli start_experiment sample_rr_seplp_optdynvmp.yml 0 10000 --concurrent 2 --overwrite_existing_intermediate_solutions --remove_temporary_scenarios --remove_intermediate_solutions
```

Above, the parameters 0 and 10000 specify which scenarios -- identified by their numeric id -- shall be executed. Furthermore, the number of processes to be used can be specified using the --concurrent option. As the execution process writes files for each scenario and each intermediate computed solution, several flags exist to control the behavior (see below for the details). Importantly, the algorithm to be executed are specified by the yaml-configuration files.

The complete options for experiment executions are:
```
python -m vnep_approx.cli start_experiment  --help
Usage: cli.py start_experiment [OPTIONS] EXPERIMENT_YAML MIN_SCENARIO_INDEX
                               MAX_SCENARIO_INDEX

Options:
  --concurrent INTEGER            number of processes to be used in parallel
  --log_level_print TEXT          log level for stdout
  --log_level_file TEXT           log level for log file
  --shuffle_instances / --original_order
                                  shall instances be shuffled or ordered
                                  according to their ids (ascendingly)
  --overwrite_existing_temporary_scenarios / --use_existing_temporary_scenarios
                                  shall existing temporary scenario files be
                                  overwritten or used?
  --overwrite_existing_intermediate_solutions / --use_existing_intermediate_solutions
                                  shall existing intermediate solution files
                                  be overwritten or used?
  --remove_temporary_scenarios / --keep_temporary_scenarios
                                  shall temporary scenario files be removed
                                  after execution?
  --remove_intermediate_solutions / --keep_intermediate_solutions
                                  shall intermediate solutions be removed
                                  after execution?
  --help                          Show this message and exit.
```

After having computed the solutions, the results are processed to extract only the data needed for plotting.
The respective commands are:
```
#extract data to be plotted
python -m evaluation_ifip_networking_2018.cli reduce_to_plotdata_randround_pickle sample_scenarios_results_cactus.pickle
move_logs_and_output log_cactus_reduction_to_plotdata

python -m evaluation_acm_ccr_2019.cli reduce_to_plotdata_rr_seplp_optdynvmp sample_scenarios_results_seplp_dynvmp.pickle
move_logs_and_output log_seplp_dynvmp_reduction_to_plotdata
```

Given the respective reduced plot data pickles, the runtime comparison plots can be created using:

```
python -m evaluation_acm_ccr_2019.cli evaluate_separation_vs_cactus_lp sample_scenarios_results_seplp_dynvmp_reduced.pickle sample_scenarios_results_cactus_reduced.pickle  ./plots/ --output_filetype png --request_sets "[[20,30],[40,50]]"

python -m evaluation_acm_ccr_2019.cli evaluate_separation_vs_cactus_lp sample_scenarios_results_seplp_dynvmp_reduced.pickle sample_scenarios_results_cactus_reduced.pickle  ./plots/ --output_filetype pdf --request_sets "[[20,30],[40,50]]"
```

## Study Treewidth using Tamaki's Algorithm

To study the treewidth of random graphs (and extract graphs of a specific treewidth), our evaluation framework offers the function  **execute_treewidth_computation_experiment**:

```
python -m evaluation_acm_ccr_2019.cli execute_treewidth_computation_experiment  --help
Usage: cli.py execute_treewidth_computation_experiment [OPTIONS]
                                                       YAML_PARAMETER_FILE

Options:
  --threads INTEGER
  --timeout INTEGER
  --remove_intermediate_solutions / --keep_intermediate_solutions
                                  shall intermediate solutions be removed
                                  after execution?
  --help                          Show this message and exit.
```
Again, to specify the properties and the count of the random graphs to be created, a yaml file is used. In our example, this yaml file has the following structure:

```
number_of_nodes: [5, 10, 15, 20, 25, 30, 35, 40, 45]
probability: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
scenario_repetition: 1
random_seed_base: 0
store_graphs_of_treewidth: []      #for only plotting the graphs, we do not need to keep them..
store_only_connected_graphs: False #as we do not store any graphs, this flag does not matter
```

Importantly, according to the above specification no graphs -- but only the treewidth etc. -- will be stored.
If you want to keep graphs of a specific treewidth, set the **store_graphs_of_treewidth** parameter accordingly, e.g., [2,3,4] to keep all graphs of treewidth 2, 3, or 4.

Given the results, the plots to analyze the treewidth and the runtime of Tamaki's algorithm, the following command can be called, which will readily generate the plots:

```
python -m evaluation_acm_ccr_2019.cli treewidth_plot_computation_results sample_treewidth_computation.yml input/sample_treewidth_computation_results_aggregated_results.pickle ./plots/ --output_filetype pdf
```

## Compare ViNE and Randomized Rounding Heuristics

To compare the performance of the ViNE offline variant WiNE with our randomized rounding heuristics using the Separation LP detailed in [1], we at first have to generate scenarios with requests of a specific treewidth. To this end, we again first generate random graphs (of course we actually used the previously generated graphs) and extract the so called **undirected graph storage**:  

```
#compute treewidths of random graphs according to parameters of the yml file
python -m evaluation_acm_ccr_2019.cli execute_treewidth_computation_experiment --threads 4 sample_treewidth_computation.yml --timeout 5400 --remove_intermediate_solutions

#extract undirected graph storage
python -m evaluation_acm_ccr_2019.cli create_undirected_graph_storage_from_treewidth_experiments input/sample_treewidth_computation_results_aggregated_results.pickle input/sample_undirected_graph_storage.pickle 2 3 
```
The undirected graph storage pickle will contain (memory-efficient) representations of the generated graphs, which are classified using the treewidth. To use these graphs using the generation, the respective undirected graph storage contained has to be specified in the yaml file when generating scenarios (see [sample/vine_vs_randround/sample_scenario_generation.yml](sample/vine_vs_randround/sample_scenario_generation.yml)). The actual scenario generation is then again performed by the base library:

```
#generate scenarios
python -m vnep_approx.cli generate_scenarios sample_scenarios.pickle sample_scenario_generation.yml --threads 4
```
 

Afterwards, the respective algorithms are executed using the **python -m vnep_approx.cli start_experiment** command shortly discussed above. As the actual specification of the algorithms is contained in the yaml files, we only discuss these shortly.

### ViNE Yaml File 
```
SCENARIO_INPUT_PICKLE: "sample_scenarios.pickle"
RESULT_OUTPUT_PICKLE:  "sample_scenarios_ViNE_results.pickle"

RUN_PARAMETERS:
    - ALGORITHM:
        ID: "OfflineViNEAlgorithmCollection"

        GUROBI_PARAMETERS:
            threads: [1] #use a single thread
            logtoconsole: [0]

        ALGORITHM_PARAMETERS:
            edge_embedding_model_list: [ !!python/tuple ['SP', 'MCF']]
            lp_objective_list: [ !!python/tuple ['ViNE_COSTS_DEF', 'ViNE_LB_DEF']]
            rounding_procedure_list: [ !!python/tuple ['DET', 'RAND']]
            repetitions_for_randomized_experiments: [2]
```

Above, the identifier of the algorithm is used to determine the actual algorithm, while the respective parameters detail the algorithm's parameters. Our ViNE implementation allows for the execution of several combinations of **ViNE variants**, which are controlled via the above shown parameters which are specified as tuples (for implementation specific reasons).

### Randround Yaml File 
```
SCENARIO_INPUT_PICKLE: "sample_scenarios.pickle"
RESULT_OUTPUT_PICKLE:  "sample_scenarios_results_seplp_dynvmp.pickle"

RUN_PARAMETERS:
    - ALGORITHM:
        ID: RandRoundSepLPOptDynVMPCollection
        
        GUROBI_PARAMETERS:
          threads: [1]
       
        ALGORITHM_PARAMETERS:
          rounding_order_list : [ !!python/tuple ["RAND", "STATIC_REQ_PROFIT", "ACHIEVED_REQ_PROFIT"]] # 
          lp_recomputation_mode_list : [ !!python/tuple ["NONE", "RECOMPUTATION_WITHOUT_SEPARATION"]] #"RECOMPUTATION_WITH_SINGLE_SEPARATION"
          lp_relative_quality : [0.001]
          rounding_samples_per_lp_recomputation_mode : [ !!python/tuple [ !!python/tuple ["NONE", 50], !!python/tuple ["RECOMPUTATION_WITHOUT_SEPARATION", 2] ] ]
          number_initial_mappings_to_compute : [50]
          number_further_mappings_to_add : [10]
```

For the randomized rounding heuristics, again several variants are specified via tuple specifications. According to the above specification 6 different randomized rounding heuristics are executed: 3 without LP recomputations and 3 with LP recomputations, which differ in the order in which requests are processed. Furthermore, the number of computed solutions can be specified in dependence of whether LPs shall be recomputed or not: here, the python tuple specification actually is meant to model a dictionary.


### Plotting

Having executed both ViNE and the randomized rounding heuristics based on the separation LP, the plot data have again to be extracted: 

```
#extract data to be plotted
python -m evaluation_acm_ccr_2019.cli reduce_to_plotdata_rr_seplp_optdynvmp sample_scenarios_results_seplp_dynvmp.pickle

python -m evaluation_acm_ccr_2019.cli reduce_to_plotdata_vine sample_scenarios_ViNE_results.pickle
```

Lastly, using the command **python -m evaluation_acm_ccr_2019.cli evaluate_separation_randround_vs_vine ** several different types of plots are executed: 

```

python -m evaluation_acm_ccr_2019.cli evaluate_separation_randround_vs_vine sample_scenarios_results_seplp_dynvmp_reduced.pickle sample_scenarios_ViNE_results_reduced.pickle ./plots/ --request_sets "[[20,30], [40,50]]" --output_filetype pdf --papermode
move_logs_and_output log_plot_pdf
```

The most important plots are contained in this package at [results/vine_vs_randround/plots](results/vine_vs_randround/plots).


# Contact

If you have any questions, simply write a mail to mrost(AT)inet.tu-berlin(DOT)de.
