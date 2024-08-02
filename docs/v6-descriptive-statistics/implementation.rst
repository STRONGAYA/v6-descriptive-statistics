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
both numerical and categorical data, and ensures that organisations not
meeting the sample size threshold are excluded from the analysis.

``aggregate_numerical_statistics``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Aggregates numerical statistics from a DataFrame.

This function processes a DataFrame containing numerical statistics,
aggregates specific statistics (sum, count, outliers, nan, sq_dev_sum),
calculates the minimum and maximum values, and computes the mean and
standard deviation for each variable.


Partials
--------
Partials are the computations that are executed on each node. The partials have access
to the data that is stored on the node. The partials are executed in parallel on each
node.

``partial``
~~~~~~~~~~~
Partial function to aggregate descriptive statistics from a DataFrame.

This function processes a DataFrame containing data from a single node,
aggregates descriptive statistics for both numerical and categorical variables,
and returns the combined statistics. It handles cases where the sample size
is below a specified threshold and excludes such variables from the analysis.

``retrieve_categorical_descriptives``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Retrieve descriptive statistics for categorical variables in a DataFrame.

This function processes categorical columns in the provided DataFrame,
removes outliers based on the provided inliers list, and returns a DataFrame
with the value counts and outliers for each categorical variable.

``retrieve_numerical_descriptives``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Retrieve descriptive statistics for numerical variables in a DataFrame.

This function processes numerical columns in the provided DataFrame,
removes outliers based on the provided range tuple,
and returns a DataFrame with the statistics for each numerical variable.