Implementation
==============

Overview
--------

Central
-----------------
The central part is responsible for the orchestration and aggregation of the algorithm.

``central``
~~~~~~~~~~~
Central function to aggregate descriptive statistics from multiple organisations.

This function collects descriptive statistics from multiple organisations,
aggregates the results, and returns the combined statistics. It handles
both numerical and categorical data, and sends tasks for both regular descriptive statistics as aggregate adjusted quantile computation.
For all computations, the algorithm leverages the [STRONG AYA v6-tools-general library](https://github.com/STRONGAYA/v6-tools-general).

Partials
--------
Partials are the computations that are executed on each node. The partials have access
to the data that is stored on the node. The partials are executed in parallel on each
node.

``partial_general_statistics``
~~~~~~~~~~~
Partial function to collect general statistics from a DataFrame.

This function processes a DataFrame containing data from a single node,
aggregates descriptive statistics for both numerical and categorical variables,
and returns the combined statistics. It allows data stratification and it handles cases where the sample size
is below a specified threshold and excludes such variables from the analysis.
For all computations, the algorithm leverages the [STRONG AYA v6-tools-general library](https://github.com/STRONGAYA/v6-tools-general).

``partial_aggregate_adjusted_deviation``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Partial function to collect aggregate adjusted deviation statistics from a DataFrame.

This function processes a DataFrame containing data from a single node,
and calculates aggregate adjusted deviation statistics for both numerical variables,
and returns the adjusted deviations and general statistics statistics. It allows data stratification and it handles cases where the sample size
is below a specified threshold and excludes such variables from the analysis.
For all computations, the algorithm leverages the [STRONG AYA v6-tools-general library](https://github.com/STRONGAYA/v6-tools-general).