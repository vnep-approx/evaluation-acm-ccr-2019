
# Overview

This repository contains the evaluation code as well as the raw results presented in our published in the ACM Computer Communication Review Journal [1].

The implementation of the respective algorithms can be found in our separate python packages: 
- **[alib](https://github.com/vnep-approx/alib)**, providing for example the data model and the Mixed-Integer Program for the classic multi-commodity formulation), as well as
- **[vnep_approx](https://github.com/vnep-approx/vnep_approx)**, providing novel Linear Programming formulations, specifically the one based on the Dyn-VMP algorithm, as well as our proposed Randomized Rounding algorithms.
- **[evaluation_ifip_networking_2018](https://github.com/vnep-approx/evaluation_ifip_networking_2018)**, providing the base line LP solutions for our runtime comparison.


Due to the size of the respective pickle-files, the generated scenarios and the full results for the algorithms is not contained in the repository but can be made accessible 
to anyone interested (see contact at the end of the page). The data for plotting -- containing the most essential information -- together with the plots
are stored within the **[results](results)** folder. 

## Papers

**[1]** Matthias Rost, Elias Döhne, Stefan Schmid: Parametrized Complexity of Virtual Network Embeddings: \break Dynamic \& Linear Programming Approximations. ACM CCR 2019 (to appear).

# Dependencies and Requirements

The **vnep_approx** library requires Python 2.7. Required python libraries: gurobipy, numpy, cPickle, networkx 1.9, matplotlib, and **[alib](https://github.com/vnep-approx/alib)**. 

Gurobi must be installed and the .../gurobi64/lib directory added to the environment variable LD_LIBRARY_PATH.

TODO: Note on the usage of https://github.com/TCS-Meiji/PACE2017-TrackA.

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

# Contact

If you have any questions, simply write a mail to mrost(AT)inet.tu-berlin(DOT)de.
