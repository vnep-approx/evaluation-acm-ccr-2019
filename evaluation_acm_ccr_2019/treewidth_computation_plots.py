import os
import pickle
from time import gmtime, strftime, time

import matplotlib.colors
import matplotlib.patches as mpatches
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import yaml
from alib import util
from matplotlib.colors import LogNorm
from vnep_approx import treewidth_model

REQUIRED_FOR_PICKLE = treewidth_model  # this prevents pycharm from removing this import, which is required for unpickling solutions

OUTPUT_PATH = "./output"
PARAMETERS_FILE = "data/parameters.yml"
DATA_FILE = "data/5_50_0_results_aggregated.pickle"
OUTPUT_FILETYPE = "pdf"
FIGSIZE = (5, 3.5)

PLOT_TITLE_FONT_SIZE = 17  # for axis titles
X_AXIS_LABEL_FONT_SIZE = 16
Y_AXIS_LABEL_FONT_SIZE = 16
LEGEND_LABEL_FONT_SIZE = 9

# TICK PARAMETERS:
TICK_LABEL_FONT_SIZE = 12
COLORBAR_TICK_FONT_SIZE = TICK_LABEL_FONT_SIZE

# list of possible parameters: https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.tick_params.html
DEFAULT_MAJOR_TICK_PARAMS = dict(
    which="major",
    length=8.0,
    width=1.5,
    grid_linewidth=0.33,
    grid_color="k",
    grid_alpha=0.7,
    grid_linestyle=":",
    labelsize=TICK_LABEL_FONT_SIZE,
)

DEFAULT_MINOR_TICK_PARAMS = dict(
    which="minor",
    length=4.0,
    width=1.0,
)
HEATMAP_MAJOR_TICK_PARAMS = DEFAULT_MAJOR_TICK_PARAMS
HEATMAP_COLORBAR_TICK_PARAMS = dict(
    which="major",
    length=4.0,
    width=1.,
    labelsize=COLORBAR_TICK_FONT_SIZE,
)

logger = util.get_logger(__name__, make_file=False, propagate=True)


class HeatmapPlotType(object):
    Simple_Treewidth_Evaluation_Average = 0
    Simple_Treewidth_Evaluation_Max = 1
    VALUE_RANGE = range(Simple_Treewidth_Evaluation_Average, Simple_Treewidth_Evaluation_Max + 1)


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
heatmap_specification_avg_treewidth = dict(
    name="Treewidth (avg.)",
    filename="treewidth_avg",
    vmin=1.0,
    vmax=40.0,
    colorbar_ticks=[1, 5, 10, 20, 40],
    cmap="inferno",
    plot_type=HeatmapPlotType.Simple_Treewidth_Evaluation_Average,
    lookup_function=lambda tw_result: tw_result.treewidth,
    metric_filter=lambda obj: (obj >= -0.00001)
)
heatmap_specification_avg_runtime = dict(
    name="Avg. Decomposition Runtime",
    filename="runtime_avg",
    vmin=0.1,
    vmax=5.0,
    # colorbar_ticks=[x for x in range(0, 5, 2)],
    cmap="inferno",
    plot_type=HeatmapPlotType.Simple_Treewidth_Evaluation_Average,
    lookup_function=lambda tw_result: tw_result.runtime_algorithm,
    metric_filter=lambda obj: (obj >= -0.00001)
)
heatmap_specification_max_treewidth = dict(
    name="Max. Treewidth",
    filename="treewidth_max",
    vmin=1.0,
    vmax=40.0,
    # colorbar_ticks=[x for x in range(0, 41, 2)],
    cmap="inferno",
    plot_type=HeatmapPlotType.Simple_Treewidth_Evaluation_Max,
    lookup_function=lambda tw_result: tw_result.treewidth,
    metric_filter=lambda obj: (obj >= -0.00001)
)
heatmap_specification_max_runtime = dict(
    name="Max. Decomposition Runtime",
    filename="runtime_max",
    vmin=0.1,
    vmax=20.0,
    # colorbar_ticks=[x for x in range(0, 21, 2)],
    cmap="inferno",
    plot_type=HeatmapPlotType.Simple_Treewidth_Evaluation_Max,
    lookup_function=lambda tw_result: tw_result.runtime_algorithm,
    metric_filter=lambda obj: (obj >= -0.00001)
)

global_heatmap_specfications = [
    heatmap_specification_avg_treewidth,
    heatmap_specification_avg_runtime,
    heatmap_specification_max_treewidth,
    heatmap_specification_max_runtime,
]

heatmap_specifications_per_type = {
    plot_type_item: [heatmap_specification for heatmap_specification in global_heatmap_specfications if heatmap_specification['plot_type'] == plot_type_item]
    for plot_type_item in [HeatmapPlotType.Simple_Treewidth_Evaluation_Average, HeatmapPlotType.Simple_Treewidth_Evaluation_Max]
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
heatmap_axes_specification_basic = dict(
    x_axis_parameter="number_of_nodes",
    y_axis_parameter="probability",
    x_axis_title="Number of Nodes",
    y_axis_title="Edge Probability",
    x_axis_ticks=[5, 10, 20, 30, 40, 50],
    y_axis_ticks=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5],
    x_axis_tick_formatting=str,
    y_axis_tick_formatting=lambda val: "{} %".format(int(100 * val)),
    foldername="heatmap"
)

global_heatmap_axes_specifications = [heatmap_axes_specification_basic]

"""
Boxplots: define plot types, metric specifications and axes specifications analogously. Key differences:
  - y-axis parameters are tied directly to metric specifications
  - axis_specifications only define the x-axis
"""


class BoxplotPlotType(object):
    Simple_Treewidth_Evaluation_Boxplot = 0


boxplot_metric_specification_runtime = dict(
    name="Runtime",
    filename="runtime_boxplot",
    y_axis_title="Runtime",
    use_log_scale=True,
    plot_type=BoxplotPlotType.Simple_Treewidth_Evaluation_Boxplot,
    lookup_function=lambda tw_result: tw_result.runtime_algorithm,
)

global_boxplot_specfications = [
    boxplot_metric_specification_runtime,
]

boxplot_specifications_per_type = {
    BoxplotPlotType.Simple_Treewidth_Evaluation_Boxplot: [boxplot_metric_specification_runtime]
}

boxplot_axis_specification_treewidth = dict(
    x_axis_title="Treewidth",
    x_axis_ticks=[0, 10, 20, 30, 38],
    filename="treewidth",
    x_axis_function=lambda tw_result: tw_result.treewidth,
    plot_title="Decomposition Runtime"
)

boxplot_axis_specification_num_nodes = dict(
    x_axis_title="Number of Nodes",
    x_axis_ticks=[5, 10, 20, 30, 40, 50],
    filename="num_nodes",
    x_axis_function=lambda tw_result: tw_result.num_nodes,
    plot_title="undefined"
)

boxplot_axis_specification_probability = dict(
    x_axis_title="Edge Connection Probability (%)",
    x_axis_ticks=[1, 10, 20, 30, 40, 50],
    filename="probability",
    x_axis_function=lambda tw_result: tw_result.probability,
    box_position_function=lambda x: 100 * x,
    plot_title="undefined"
)

global_boxplot_axes_specfications = [
    boxplot_axis_specification_treewidth,
    boxplot_axis_specification_num_nodes,
    boxplot_axis_specification_probability,
]

"""  
Decomposition Runtime Plots: define plot types, metric specifications and axes specifications analogously to Boxplots.
"""


class DecompositionRuntimePlotType(object):
    Simple_Treewidth_Evaluation_DecompositionRuntimePlot = 0


decomposition_runtime_plot_metric_specification_runtime = dict(
    name="Runtime",
    filename="decomposition_runtime",
    y_axis_title="Runtime",
    use_log_scale=True,
    plot_type=DecompositionRuntimePlotType.Simple_Treewidth_Evaluation_DecompositionRuntimePlot,
    lookup_function=lambda tw_result: tw_result.runtime_algorithm,
    percentiles=[0, 1, 25, 50, 75, 99, 100],
    linewidths=[0.25, 0.5, 0.5, 2, 0.5, 0.5, 0.25],
    color_values=[0.9, 0.6, 0.3, 0.3, 0.6, 0.9],
)

global_decomposition_runtime_plot_specfications = [
    decomposition_runtime_plot_metric_specification_runtime,
]

decomposition_runtime_plot_axis_specification_treewidth = dict(
    x_axis_title="Treewidth",
    x_axis_ticks=range(0, 50, 5),
    filename="treewidth",
    x_axis_function=lambda tw_result: tw_result.treewidth,
    plot_title="Decomposition Runtime"
)

decomposition_runtime_plot_axis_specification_num_nodes = dict(
    x_axis_title="Number of Nodes",
    x_axis_ticks=[5, 10, 20, 30, 40, 50],
    filename="num_nodes",
    x_axis_function=lambda tw_result: tw_result.num_nodes,
    plot_title="undefined"
)

decomposition_runtime_plot_axis_specification_probability = dict(
    x_axis_title="Edge Connection Probability (%)",
    x_axis_ticks=[1, 10, 20, 30, 40, 50],
    filename="probability",
    x_axis_function=lambda tw_result: tw_result.probability,
    plot_title="undefined"
)

global_decomposition_runtime_plot_axes_specfications = [
    decomposition_runtime_plot_axis_specification_treewidth,
    decomposition_runtime_plot_axis_specification_num_nodes,
    decomposition_runtime_plot_axis_specification_probability,
]


class AbstractPlotter(object):
    ''' Abstract Plotter interface providing functionality used by the majority of plotting classes of this module.
    '''

    def __init__(self,
                 output_path,
                 output_filetype,
                 experiment_parameters,
                 data_dict,
                 show_plot=False,
                 save_plot=True,
                 overwrite_existing_files=False,
                 paper_mode=True
                 ):
        self.output_path = output_path
        self.output_filetype = output_filetype

        self.experiment_parameters = experiment_parameters
        self.data_dict = data_dict

        self.show_plot = show_plot
        self.save_plot = save_plot
        self.overwrite_existing_files = overwrite_existing_files

        self.paper_mode = paper_mode

    def _construct_output_path_and_filename(self, title, filter_specifications=None):
        filter_spec_path = ""
        filter_filename = "no_filter.{}".format(OUTPUT_FILETYPE)
        if filter_specifications:
            filter_spec_path, filter_filename = self._construct_path_and_filename_for_filter_spec(filter_specifications)
        base = os.path.normpath(OUTPUT_PATH)
        date = strftime("%Y-%m-%d", gmtime())
        output_path = os.path.join(base, date, OUTPUT_FILETYPE, "general", filter_spec_path)
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

    def plot_figure(self):
        raise RuntimeError("This is an abstract method")


class SingleBoxplotPlotter(AbstractPlotter):
    def __init__(self,
                 output_path,
                 output_filetype,
                 experiment_parameters,
                 data_dict,
                 boxplot_plot_type,
                 list_of_axes_specifications=tuple(global_boxplot_axes_specfications),
                 list_of_metric_specifications=None,
                 show_plot=False,
                 save_plot=True,
                 overwrite_existing_files=False,
                 paper_mode=True,
                 sampling_rate=None,
                 read_pickle=True,
                 write_pickle=True,
                 ):
        super(SingleBoxplotPlotter, self).__init__(output_path, output_filetype,
                                                   experiment_parameters, data_dict,
                                                   show_plot, save_plot,
                                                   overwrite_existing_files, paper_mode)
        self.plot_type = boxplot_plot_type

        if not list_of_axes_specifications:
            raise RuntimeError("Axes need to be provided.")
        self.list_of_axes_specifications = list_of_axes_specifications

        if not list_of_metric_specifications:
            self.list_of_metric_specifications = [boxplot_metric_specification_runtime]
        else:
            self.list_of_metric_specifications = list_of_metric_specifications
        if sampling_rate is None:
            sampling_rate = 1
        self.sampling_rate = sampling_rate
        print "Sampling every {} values per parameter combination (at least one)".format(self.sampling_rate)
        self.read_pickle = read_pickle
        self.write_pickle = write_pickle

    def plot_figure(self):
        for axes_specification in self.list_of_axes_specifications:
            for metric_specfication in self.list_of_metric_specifications:
                self.plot_single_boxplot_general(metric_specfication, axes_specification)

    def plot_single_boxplot_general(self,
                                    boxplot_metric_specification,
                                    boxplot_axes_specification):

        base_filename = "{}__by__{}".format(boxplot_metric_specification["filename"], boxplot_axes_specification["filename"])
        output_path, filename = self._construct_output_path_and_filename(base_filename)

        logger.debug("output_path is {};\t filename is {}".format(output_path, filename))

        if not self.overwrite_existing_files and os.path.exists(filename):
            logger.info("Skipping generation of {} as this file already exists".format(filename))
            return

        fig, ax = plt.subplots(figsize=FIGSIZE)

        values_dict = self._read_data_from_pickle_if_allowed(base_filename)
        if values_dict is None:
            values_dict = self._process_data(boxplot_axes_specification, boxplot_metric_specification)
            data_was_read_from_pickle = False
        else:
            data_was_read_from_pickle = True

        if self.write_pickle and not data_was_read_from_pickle:  # no need to write same data back to pickle file
            self._write_data_to_pickle(base_filename, values_dict)

        if self.paper_mode:
            print boxplot_axes_specification
            ax.set_title(boxplot_axes_specification['plot_title'], fontsize=PLOT_TITLE_FONT_SIZE)
        else:
            title = boxplot_metric_specification['name'] + "\n"
            title += self._get_sol_count_string(values_dict)
            ax.set_title(title, fontsize=PLOT_TITLE_FONT_SIZE)

        sorted_x_values = sorted(values_dict.keys())

        t_start = time()
        boxes = ax.boxplot(
            [values_dict[key] for key in sorted_x_values],  # convert to list of lists
            positions=map(
                boxplot_axes_specification.get("box_position_function", lambda x: x),
                sorted_x_values
            ),
            flierprops=dict(
                marker='o',
                markersize=4,
                # markerfacecolor="k",
                linestyle='none',
                alpha=0.25,
            ),
            # boxprops=dict(linestyle='-', linewidth=3, color='black'),
            medianprops=dict(
                color="k",
            ),
        )
        print "Plotting:", time() - t_start, "seconds"

        if "x_axis_ticks" in boxplot_axes_specification:
            ax.set_xticks(boxplot_axes_specification["x_axis_ticks"])
            ax.set_xticklabels(map(str, boxplot_axes_specification["x_axis_ticks"]))

        if boxplot_metric_specification.get("use_log_scale", False):
            plt.yscale('log')
            plt.autoscale(True)
        ax.tick_params(axis="x", **DEFAULT_MAJOR_TICK_PARAMS)
        ax.tick_params(axis="y", **DEFAULT_MAJOR_TICK_PARAMS)
        ax.tick_params(axis="x", **DEFAULT_MINOR_TICK_PARAMS)
        ax.tick_params(axis="y", **DEFAULT_MINOR_TICK_PARAMS)

        ax.grid()  # to change grid style parameters, modify the BOXPLOT_..._TICK_PARAMS dicts defined at the top of the file

        ax.set_xlabel(boxplot_axes_specification['x_axis_title'], fontsize=X_AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel(boxplot_metric_specification['y_axis_title'], fontsize=Y_AXIS_LABEL_FONT_SIZE)

        self._show_and_or_save_plots(output_path, filename)

    def _read_data_from_pickle_if_allowed(self, base_filename):
        values_dict = None
        if self.read_pickle:
            plot_data_pickle_file = self._get_path_to_pickle_file(base_filename)
            print("Trying to read file: {}".format(plot_data_pickle_file))
            if os.path.exists(plot_data_pickle_file):
                print "Reading plot data pickle from {}".format(plot_data_pickle_file)
                with open(plot_data_pickle_file, "r") as f:
                    values_dict = pickle.load(f)
        return values_dict

    def _write_data_to_pickle(self, base_filename, values_dict):
        plot_data_pickle_file = self._get_path_to_pickle_file(base_filename)
        parent_dir = os.path.dirname(plot_data_pickle_file)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        with open(plot_data_pickle_file, "w") as f:
            print "Writing pickle to {}".format(plot_data_pickle_file)
            pickle.dump(values_dict, f)

    def _get_path_to_pickle_file(self, base_filename):
        pickle_filename = "{}__sample_rate_{}.pickle".format(base_filename, self.sampling_rate)
        return os.path.join(os.path.normpath(OUTPUT_PATH), "data_pickles", pickle_filename)

    def _process_data(self, boxplot_axes_specification, boxplot_metric_specification):
        t_start = time()
        values_dict = {}
        lookup_function = boxplot_metric_specification["lookup_function"]
        x_axis_function = boxplot_axes_specification["x_axis_function"]
        for num_nodes, prob_results_dict in self.data_dict.iteritems():
            for prob, results in prob_results_dict.iteritems():
                # assert len(results) == self.experiment_parameters["scenario_repetition"]  # sanity check for now
                logger.debug("values are {}".format(values_dict))
                for result in results[::self.sampling_rate]:
                    x_val = x_axis_function(result)
                    if x_val not in values_dict:
                        values_dict[x_val] = []
                    values_dict[x_val].append(lookup_function(result))
        print "Boxplot data processing:", time() - t_start, "seconds"
        return values_dict

    def _get_sol_count_string(self, values_dict):
        lens = map(len, values_dict.values())
        min_number_of_observed_values = min(lens)
        max_number_of_observed_values = max(lens)
        if not self.paper_mode:
            if min_number_of_observed_values == max_number_of_observed_values:
                solution_count_string = "{} values per box".format(min_number_of_observed_values)
            else:
                solution_count_string = "between {} and {} values per box".format(min_number_of_observed_values,
                                                                                  max_number_of_observed_values)
        return solution_count_string


class DecompositionRuntimePlotter(AbstractPlotter):
    def __init__(self,
                 output_path,
                 output_filetype,
                 experiment_parameters,
                 data_dict,
                 decomposition_runtime_plot_type,
                 list_of_axes_specifications=tuple(global_decomposition_runtime_plot_axes_specfications),
                 list_of_metric_specifications=None,
                 show_plot=False,
                 save_plot=True,
                 overwrite_existing_files=False,
                 paper_mode=True,
                 sampling_rate=None,
                 read_pickle=True,
                 write_pickle=True,
                 ):
        super(DecompositionRuntimePlotter, self).__init__(output_path, output_filetype,
                                                          experiment_parameters, data_dict,
                                                          show_plot, save_plot,
                                                          overwrite_existing_files, paper_mode)
        self.plot_type = decomposition_runtime_plot_type

        if not list_of_axes_specifications:
            raise RuntimeError("Axes need to be provided.")
        self.list_of_axes_specifications = list_of_axes_specifications

        if not list_of_metric_specifications:
            self.list_of_metric_specifications = [
                decomposition_runtime_plot_metric_specification_runtime,
            ]
        else:
            self.list_of_metric_specifications = list_of_metric_specifications
        if sampling_rate is None:
            sampling_rate = 1
        self.sampling_rate = sampling_rate
        print "Sampling every {} values per parameter combination (at least one)".format(self.sampling_rate)
        self.read_pickle = read_pickle
        self.write_pickle = write_pickle

    def plot_figure(self):
        for axes_specification in self.list_of_axes_specifications:
            for metric_specfication in self.list_of_metric_specifications:
                self.plot_single_decomposition_runtime(metric_specfication, axes_specification)

    def plot_single_decomposition_runtime(self,
                                          decomposition_runtime_metric_specification,
                                          decomposition_runtime_axes_specification):

        base_filename = "{}__by__{}".format(decomposition_runtime_metric_specification["filename"],
                                            decomposition_runtime_axes_specification["filename"])
        output_path, filename = self._construct_output_path_and_filename(base_filename)

        logger.debug("output_path is {};\t filename is {}".format(output_path, filename))

        if not self.overwrite_existing_files and os.path.exists(filename):
            logger.info("Skipping generation of {} as this file already exists".format(filename))
            return

        values_dict = self._get_values_dict(base_filename,
                                            decomposition_runtime_axes_specification,
                                            decomposition_runtime_metric_specification)

        fig, ax = plt.subplots(figsize=FIGSIZE)
        if self.paper_mode:
            print decomposition_runtime_axes_specification
            ax.set_title(decomposition_runtime_axes_specification['plot_title'], fontsize=PLOT_TITLE_FONT_SIZE)
        else:
            title = decomposition_runtime_metric_specification['name'] + "\n"
            title += self._get_sol_count_string(values_dict)
            ax.set_title(title, fontsize=PLOT_TITLE_FONT_SIZE)

        sorted_keys = sorted(values_dict.keys())
        percentiles = decomposition_runtime_metric_specification["percentiles"]
        linewidths = decomposition_runtime_metric_specification.get("linewidths",
                                                                    [0.5] * len(percentiles))
        color_values = decomposition_runtime_metric_specification.get("color_values")

        lines = np.zeros((len(percentiles), len(sorted_keys)))
        for i, key in enumerate(sorted_keys):
            lines[:, i] = np.percentile(values_dict[key], percentiles)

        t_start = time()
        handles = []
        median_legend_handle = None
        for i, p in enumerate(percentiles):
            line = ax.plot(
                sorted_keys,
                lines[i][:],
                linewidth=linewidths[i],
                color="k",
                label="median" if p == 50 else None
            )

            line[0].set_path_effects([path_effects.Stroke(linewidth=linewidths[i] + 0.5, foreground='w'),
                                      path_effects.Normal()])

            if i < len(percentiles) - 1:
                if color_values:
                    c = plt.cm.inferno(color_values[i])
                else:
                    c = plt.cm.inferno(0.01 * p)
                if p == 50:
                    median_legend_handle = line[0]
                    handles[-1].set_label("{}% - {}%".format(percentiles[i - 1], percentiles[i + 1]))
                else:
                    handles.append(mpatches.Patch(color=c, label="{}% - {}%".format(p, percentiles[i + 1])))
                ax.fill_between(
                    sorted_keys,
                    lines[i][:],
                    lines[i + 1][:],
                    facecolor=matplotlib.colors.to_rgba(c, 0.8),  # apply alpha only to facecolor
                )
        handles.reverse()
        if median_legend_handle:
            handles.append(median_legend_handle)
        ax.legend(handles=handles, fontsize=LEGEND_LABEL_FONT_SIZE, loc=2, title="percentiles", handletextpad=.35,
                  borderaxespad=0.175, borderpad=0.2, handlelength=1.75)

        print "Plotting:", time() - t_start, "seconds"

        if "x_axis_ticks" in decomposition_runtime_axes_specification:
            ax.set_xticks(decomposition_runtime_axes_specification["x_axis_ticks"])
            ax.set_xticklabels(map(str, decomposition_runtime_axes_specification["x_axis_ticks"]))

        if decomposition_runtime_metric_specification.get("use_log_scale", False):
            plt.yscale('log')
            plt.autoscale(True)
        ax.tick_params(axis="x", **DEFAULT_MAJOR_TICK_PARAMS)
        ax.tick_params(axis="y", **DEFAULT_MAJOR_TICK_PARAMS)
        ax.tick_params(axis="x", **DEFAULT_MINOR_TICK_PARAMS)
        ax.tick_params(axis="y", **DEFAULT_MINOR_TICK_PARAMS)

        ax.grid()  # to change grid style parameters, modify the BOXPLOT_..._TICK_PARAMS dicts defined at the top of the file

        ax.set_xlabel(decomposition_runtime_axes_specification['x_axis_title'], fontsize=X_AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel(decomposition_runtime_metric_specification['y_axis_title'], fontsize=Y_AXIS_LABEL_FONT_SIZE)

        self._show_and_or_save_plots(output_path, filename)

    def _get_values_dict(self, base_filename,
                         decomposition_runtime_axes_specification,
                         decomposition_runtime_metric_specification):
        values_dict = self._read_data_from_pickle_if_allowed(base_filename)
        if values_dict is None:
            values_dict = self._process_data(decomposition_runtime_axes_specification,
                                             decomposition_runtime_metric_specification)
            data_was_read_from_pickle = False
        else:
            data_was_read_from_pickle = True
        if self.write_pickle and not data_was_read_from_pickle:  # no need to write same data back to pickle file
            self._write_data_to_pickle(base_filename, values_dict)
        return values_dict

    def _read_data_from_pickle_if_allowed(self, base_filename):
        values_dict = None
        if self.read_pickle:
            plot_data_pickle_file = self._get_path_to_pickle_file(base_filename)
            print("Trying to read file: {}".format(plot_data_pickle_file))
            if os.path.exists(plot_data_pickle_file):
                print "Reading plot data pickle from {}".format(plot_data_pickle_file)
                with open(plot_data_pickle_file, "r") as f:
                    values_dict = pickle.load(f)
        return values_dict

    def _write_data_to_pickle(self, base_filename, values_dict):
        plot_data_pickle_file = self._get_path_to_pickle_file(base_filename)
        parent_dir = os.path.dirname(plot_data_pickle_file)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        with open(plot_data_pickle_file, "w") as f:
            print "Writing pickle to {}".format(plot_data_pickle_file)
            pickle.dump(values_dict, f)

    def _get_path_to_pickle_file(self, base_filename):
        pickle_filename = "{}__sample_rate_{}.pickle".format(base_filename, self.sampling_rate)
        return os.path.join(os.path.normpath(OUTPUT_PATH), "data_pickles", pickle_filename)

    def _process_data(self, decomposition_runtime_axes_specification, boxplot_metric_specification):
        t_start = time()
        values_dict = {}
        lookup_function = boxplot_metric_specification["lookup_function"]
        x_axis_function = decomposition_runtime_axes_specification["x_axis_function"]
        for num_nodes, prob_results_dict in self.data_dict.iteritems():
            for prob, results in prob_results_dict.iteritems():
                # assert len(results) == self.experiment_parameters["scenario_repetition"]  # sanity check for now
                logger.debug("values are {}".format(values_dict))
                for result in results[::self.sampling_rate]:
                    x_val = x_axis_function(result)
                    if x_val not in values_dict:
                        values_dict[x_val] = []
                    values_dict[x_val].append(lookup_function(result))
        print "DecompositionRuntime data processing:", time() - t_start, "seconds"
        return values_dict

    def _get_sol_count_string(self, values_dict):
        lens = map(len, values_dict.values())
        min_number_of_observed_values = min(lens)
        max_number_of_observed_values = max(lens)
        solution_count_string = ""
        if not self.paper_mode:
            if min_number_of_observed_values == max_number_of_observed_values:
                solution_count_string = "{} values per position".format(min_number_of_observed_values)
            else:
                solution_count_string = "between {} and {} values per position".format(min_number_of_observed_values,
                                                                                       max_number_of_observed_values)
        return solution_count_string


class SingleHeatmapPlotter(AbstractPlotter):
    def __init__(self,
                 output_path,
                 output_filetype,
                 experiment_parameters,
                 data_dict,
                 heatmap_plot_type,
                 list_of_axes_specifications=tuple(global_heatmap_axes_specifications),
                 list_of_metric_specifications=None,
                 show_plot=False,
                 save_plot=True,
                 overwrite_existing_files=False,
                 paper_mode=True,
                 ):

        """
        output_path,
                 output_filetype,
                 experiment_parameters,
                 data_dict,
                 show_plot=False,
                 save_plot=True,
                 overwrite_existing_files=False,
                 paper_mode=True
        """

        super(SingleHeatmapPlotter, self).__init__(output_path, output_filetype,
                                                   experiment_parameters, data_dict,
                                                   show_plot, save_plot,
                                                   overwrite_existing_files, paper_mode)
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

    def _construct_output_path_and_filename(self, metric_specification, heatmap_axes_specification, filter_specifications=None):
        filter_spec_path = ""
        filter_filename = "no_filter.{}".format(OUTPUT_FILETYPE)
        if filter_specifications:
            filter_spec_path, filter_filename = self._construct_path_and_filename_for_filter_spec(filter_specifications)
        base = os.path.normpath(OUTPUT_PATH)
        date = strftime("%Y-%m-%d", gmtime())
        axes_foldername = heatmap_axes_specification['foldername']
        output_path = os.path.join(base, date, OUTPUT_FILETYPE, axes_foldername, filter_spec_path)
        filename = os.path.join(output_path, metric_specification['filename'] + "_" + filter_filename)
        return output_path, filename

    def plot_figure(self):
        for axes_specification in self.list_of_axes_specifications:
            for metric_specfication in self.list_of_metric_specifications:
                self.plot_single_heatmap_general(metric_specfication, axes_specification)

    def extract_parameter_values(self, axis_spec):
        return sorted(self.experiment_parameters[axis_spec])

    def plot_single_heatmap_general(self,
                                    heatmap_metric_specification,
                                    heatmap_axes_specification):
        # data extraction

        output_path, filename = self._construct_output_path_and_filename(heatmap_metric_specification,
                                                                         heatmap_axes_specification)

        logger.debug("output_path is {};\t filename is {}".format(output_path, filename))

        if not self.overwrite_existing_files and os.path.exists(filename):
            logger.info("Skipping generation of {} as this file already exists".format(filename))
            return

        xaxis_parameters = self.extract_parameter_values(heatmap_axes_specification['x_axis_parameter'])
        yaxis_parameters = self.extract_parameter_values(heatmap_axes_specification['y_axis_parameter'])

        # all heatmap values will be stored in X
        X = np.zeros((len(yaxis_parameters), len(xaxis_parameters)))

        fig, ax = plt.subplots(figsize=FIGSIZE)

        min_number_of_observed_values = 10000000000000
        max_number_of_observed_values = 0
        observed_values = np.empty(0)

        for x_index, x_val in enumerate(xaxis_parameters):
            # all scenario indices which has x_val as xaxis parameter (e.g. node_resource_factor = 0.5
            sub_dict = self.data_dict[x_val]
            for y_index, y_val in enumerate(yaxis_parameters):
                results = self.data_dict[x_val][y_val]
                values = [heatmap_metric_specification['lookup_function'](result) for result in results]

                if 'metric_filter' in heatmap_metric_specification:
                    values = [value for value in values if heatmap_metric_specification['metric_filter'](value)]

                observed_values = np.append(observed_values, values)

                if len(values) < min_number_of_observed_values:
                    min_number_of_observed_values = len(values)
                if len(values) > max_number_of_observed_values:
                    max_number_of_observed_values = len(values)

                logger.debug("values are {}".format(values))
                if heatmap_metric_specification["plot_type"] == HeatmapPlotType.Simple_Treewidth_Evaluation_Max:
                    m = np.max(values)
                    logger.debug("max is {}".format(m))
                elif heatmap_metric_specification["plot_type"] == HeatmapPlotType.Simple_Treewidth_Evaluation_Average:
                    m = np.mean(values)
                    logger.debug("mean is {}".format(m))
                else:
                    raise ValueError("Invalid HeatmapPlotType")
                if 'rounding_function' in heatmap_metric_specification:
                    rounded_m = heatmap_metric_specification['rounding_function'](m)
                else:
                    rounded_m = float("{0:.1f}".format(round(m, 2)))
                X[y_index, x_index] = rounded_m

        if not self.paper_mode:
            if min_number_of_observed_values == max_number_of_observed_values:
                solution_count_string = "{} values per square".format(min_number_of_observed_values)
            else:
                solution_count_string = "between {} and {} values per square".format(min_number_of_observed_values,
                                                                                     max_number_of_observed_values)

        ax.tick_params(axis="both", **HEATMAP_MAJOR_TICK_PARAMS)

        if self.paper_mode:
            ax.set_title(heatmap_metric_specification['name'], fontsize=PLOT_TITLE_FONT_SIZE)
        else:
            title = heatmap_metric_specification['name'] + "\n"
            title += solution_count_string + "\n"
            title += "min: {:.2f}; mean: {:.2f}; max: {:.2f}".format(np.nanmin(observed_values),
                                                                     np.nanmean(observed_values),
                                                                     np.nanmax(observed_values))
            ax.set_title(title)

        norm = LogNorm(vmin=X[X > 0].min(), vmax=X.max())
        heatmap = ax.pcolor(
            X,
            cmap=heatmap_metric_specification['cmap'],
            vmin=heatmap_metric_specification['vmin'],
            vmax=heatmap_metric_specification['vmax'],
            norm=norm,
        )

        if not self.paper_mode:
            fig.colorbar(heatmap, label=heatmap_metric_specification['name'] + ' - mean in blue')
        else:
            cbar = fig.colorbar(heatmap)
            if 'colorbar_ticks' in heatmap_metric_specification:
                ticks = heatmap_metric_specification['colorbar_ticks']
                tick_labels = [str(tick).ljust(3) for tick in ticks]
                cbar.set_ticks(ticks)
                cbar.set_ticklabels(tick_labels)
            else:
                print "No colorbar tick labels were specified for {}".format(heatmap_metric_specification["name"])
            # for label in cbar.ax.get_yticklabels():
            #    label.set_fontproperties(font_manager.FontProperties(family="Courier New",weight='bold'))
            cbar.ax.tick_params(**HEATMAP_COLORBAR_TICK_PARAMS)

        if "x_axis_ticks" in heatmap_axes_specification:
            tick_locations = [xaxis_parameters.index(x) for x in heatmap_axes_specification["x_axis_ticks"]]
            tick_locations = np.array(tick_locations) + 0.5
            ax.set_xticks(tick_locations, minor=False)
            x_labels = map(heatmap_axes_specification["x_axis_tick_formatting"], heatmap_axes_specification["x_axis_ticks"])
            ax.set_xticklabels(x_labels, minor=False, fontsize=TICK_LABEL_FONT_SIZE)
        else:
            ax.set_xticks(np.arange(X.shape[1]) + 0.5, minor=False)

        if "y_axis_ticks" in heatmap_axes_specification:
            tick_locations = [yaxis_parameters.index(x) for x in heatmap_axes_specification["y_axis_ticks"]]
            tick_locations = np.array(tick_locations) + 0.5
            ax.set_yticks(tick_locations, minor=False)
            y_labels = map(heatmap_axes_specification["y_axis_tick_formatting"], heatmap_axes_specification["y_axis_ticks"])
            ax.set_yticklabels(y_labels, minor=False)
        else:
            ax.set_yticks(np.arange(X.shape[0]) + 0.5, minor=False)

        # column_labels = yaxis_parameters
        ax.set_xlabel(heatmap_axes_specification['x_axis_title'], fontsize=X_AXIS_LABEL_FONT_SIZE)
        ax.set_ylabel(heatmap_axes_specification['y_axis_title'], fontsize=Y_AXIS_LABEL_FONT_SIZE)
        # ax.set_yticklabels(column_labels, minor=False, fontsize=Y_AXIS_TICK_LABEL_FONT_SIZE)

        self._show_and_or_save_plots(output_path, filename)


def plot_heatmaps(parameters, data):
    baseline_plotter = SingleHeatmapPlotter(
        output_path=OUTPUT_PATH,
        output_filetype=OUTPUT_FILETYPE,
        experiment_parameters=parameters,
        heatmap_plot_type=HeatmapPlotType.Simple_Treewidth_Evaluation_Average,
        data_dict=data,
        show_plot=False,
        save_plot=True,
        overwrite_existing_files=True,
        paper_mode=True,
    )
    baseline_plotter.plot_figure()
    baseline_plotter = SingleHeatmapPlotter(
        output_path=OUTPUT_PATH,
        output_filetype=OUTPUT_FILETYPE,
        experiment_parameters=parameters,
        heatmap_plot_type=HeatmapPlotType.Simple_Treewidth_Evaluation_Max,
        data_dict=data,
        show_plot=False,
        save_plot=True,
        overwrite_existing_files=True,
        paper_mode=True,
    )
    baseline_plotter.plot_figure()


def plot_boxplots(parameters, data):
    baseline_plotter = SingleBoxplotPlotter(
        output_path=OUTPUT_PATH,
        output_filetype=OUTPUT_FILETYPE,
        experiment_parameters=parameters,
        boxplot_plot_type=BoxplotPlotType.Simple_Treewidth_Evaluation_Boxplot,
        data_dict=data,
        show_plot=False,
        save_plot=True,
        overwrite_existing_files=True,
        paper_mode=True,
        sampling_rate=1,
    )
    baseline_plotter.plot_figure()


def plot_decomposition_runtime_plots(parameters, data):
    baseline_plotter = DecompositionRuntimePlotter(
        output_path=OUTPUT_PATH,
        output_filetype=OUTPUT_FILETYPE,
        experiment_parameters=parameters,
        decomposition_runtime_plot_type=DecompositionRuntimePlotType.Simple_Treewidth_Evaluation_DecompositionRuntimePlot,
        data_dict=data,
        show_plot=False,
        save_plot=True,
        overwrite_existing_files=True,
        paper_mode=True,
        sampling_rate=1,
        read_pickle=True,
        write_pickle=True
    )
    baseline_plotter.plot_figure()


def make_plots(parameters_file, results_pickle):
    parameters = yaml.load(parameters_file)
    results = pickle.load(results_pickle)

    plot_heatmaps(parameters, results)
    plot_decomposition_runtime_plots(parameters, results)
    plot_boxplots(parameters, results)
