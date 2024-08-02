"""
This file contains all central algorithm functions. It is important to note
that the central method is executed on a node, just like any other method.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled).
"""
import numpy as np
import pandas as pd

from typing import Any
from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.client import AlgorithmClient

from io import StringIO as stringIO


@algorithm_client
def central(client: AlgorithmClient, variables_to_describe: dict, variables_to_stratify: dict = None,
            organization_ids: list = None, return_partials: bool = False) -> Any:
    """
    Central function to aggregate descriptive statistics from multiple organisations.

    This function collects descriptive statistics from multiple organisations,
    aggregates the results, and returns the combined statistics. It handles
    both numerical and categorical data, and ensures that organisations not
    meeting the sample size threshold are excluded from the analysis.

    Parameters:
    client (AlgorithmClient): The client to communicate with the vantage6 server.
    variables_to_describe (dict): Dictionary of variables to describe.
    variables_to_stratify (dict, optional): Dictionary of variables to stratify. Defaults to None.
    organization_ids (list, optional): List of organisation IDs to include. Defaults to None.
    return_partials (bool, optional): Whether to return partial results. Defaults to False.

    Returns:
    Any: A dictionary containing the aggregated descriptive statistics and
    the list of included and excluded organisations.
    """

    # Collect all organizations that participate in this collaboration unless specified
    if isinstance(organization_ids, list) is False:
        organisations = client.organization.list()
        ids = [organisation.get("id") for organisation in organisations]
    else:
        ids = organization_ids

    # Create a list to store the IDs of organizations that do not meet privacy guards
    excluded_ids = []

    # Define input parameters for the subtasks
    info("Defining input parameters")
    input_ = {
        "method": "partial",
        "kwargs": {
            "variables_to_describe": variables_to_describe,
            "variables_to_stratify": variables_to_stratify,

        }
    }

    # Create a subtask for all organisations in the collaboration.
    info("Creating subtask for all organizations in the collaboration")
    task = client.task.create(
        input_=input_,
        organizations=ids,
        name="Descriptive statistics retrieval",
        description="Retrieve descriptive statistics.",
    )

    n_loops = 0
    n_threshold_met = False
    while not n_threshold_met:
        # This list represents the organisations that will be excluded in the following loop
        _excluded_ids = []
        if n_loops > 2:
            error("Sample size violations should be eliminated yet criteria are not met. Exiting")
            raise ValueError("Sample size violations should be eliminated yet criteria are not. Exiting")

        n_loops += 1

        # Create a subtask for all organisations in the collaboration.
        info("Creating subtask for all organizations in the collaboration")
        task = client.task.create(
            input_=input_,
            organizations=ids,
            name="Descriptive statistics retrieval",
            description="Retrieve descriptive statistics.",
        )

        # Wait for the node to return results of the subtask.
        info("Waiting for results")
        results = client.wait_for_results(task_id=task.get("id"))
        info("Results obtained!")

        unique_time_events = []
        for output in results:

            # Exclude organizations that do not meet the N-threshold
            if "N-Threshold not met" in output:
                warn(f"Insufficient samples for organization {output['N-Threshold not met']}. "
                     f"Excluding organization from analysis.")
                ids.remove(output["N-Threshold not met"])
                excluded_ids.append(output["N-Threshold not met"])
                _excluded_ids.append(output["N-Threshold not met"])
                continue

        if len(_excluded_ids) == 0:
            n_threshold_met = True
        elif len(ids) == 0:
            warn("No organizations meet the minimal sample size threshold, returning NaN.")
            return {"excluded_organizations": excluded_ids, "table": np.nan}

    # Wait for the node to return results of the subtask.
    info("Waiting for results")
    results = client.wait_for_results(task_id=task.get("id"))
    info("Results obtained!")

    # Aggregate results
    aggregate_categorical_df = pd.DataFrame(columns=["variable", "value", "count"])
    aggregate_numerical_df = pd.DataFrame(columns=["variable", "statistic", "value"])

    for result in results:
        for datatype, variables in result.items():
            if datatype == "categorical":
                categorical_df = pd.read_json(stringIO(result["categorical"]))
                aggregate_categorical_df = pd.concat([aggregate_categorical_df, categorical_df])

            elif datatype == "numerical":
                numerical_df = pd.read_json(stringIO(result["numerical"]))
                numerical_df = numerical_df[~numerical_df["statistic"].isin(["q1", "median", "q3", "mean", "std"])]
                aggregate_numerical_df = pd.concat([aggregate_numerical_df, numerical_df], ignore_index=True)

    # TODO federated variance computation

    # Aggregate the numerical results
    aggregate_numerical_df = aggregate_numerical_statistics(aggregate_numerical_df)

    # Aggregate the categorical results
    aggregate_categorical_df = aggregate_categorical_df.groupby(["variable", "value"], as_index=False).sum()

    if return_partials:
        warn("Returning partial descriptive statistics")
        return {"categorical_descriptives": aggregate_categorical_df.to_json(),
                "numerical_descriptives": aggregate_numerical_df.to_json(),
                "included_organizations": ids, "excluded_organizations": excluded_ids,
                "partial_results": results}
    else:
        return {"categorical_descriptives": aggregate_categorical_df.to_json(),
                "numerical_descriptives": aggregate_numerical_df.to_json(),
                "included_organizations": ids, "excluded_organizations": excluded_ids}


def aggregate_numerical_statistics(aggregate_numerical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregates numerical statistics from a DataFrame.

    This function processes a DataFrame containing numerical statistics, 
    aggregates specific statistics (sum, count, outliers, nan, sq_dev_sum), 
    calculates the minimum and maximum values, and computes the mean and 
    standard deviation for each variable.

    Parameters:
    aggregate_numerical_df (pd.DataFrame): DataFrame containing numerical statistics 
                                           with columns "variable", "statistic", and "value".

    Returns:
    pd.DataFrame: Aggregated DataFrame with combined statistics including mean and std.
    """
    for variable in aggregate_numerical_df["variable"].unique():
        # Filter the DataFrame for the current variable
        variable_df = aggregate_numerical_df[aggregate_numerical_df["variable"] == variable]

        # Remove the aggregated data for the current variable from the main DataFrame
        aggregate_numerical_df = aggregate_numerical_df[~aggregate_numerical_df["variable"].isin([variable])]

        # Sum specific statistics for the current variable
        summed_stats = variable_df[
            variable_df["statistic"].isin(
                ["sum", "count", "outliers", "nan", "sq_dev_sum"])].groupby(
            ["variable", "statistic"]).sum().reset_index()

        # Calculate the minimum value for the current variable
        min_val = variable_df[variable_df["statistic"] == "min"]["value"].min()

        # Calculate the maximum value for the current variable
        max_val = variable_df[variable_df["statistic"] == "max"]["value"].max()

        # Create a new DataFrame for min and max values
        min_max_stats = pd.DataFrame({
            "variable": ["Age", "Age"],
            "statistic": ["min", "max"],
            "value": [min_val, max_val]
        })

        # Combine the summed statistics and min/max values
        stats_df = pd.concat([min_max_stats, summed_stats], ignore_index=True)

        # Compute mean and standard deviation for the current variable
        mean_std_df = stats_df.pivot_table(index="variable", columns="statistic", values="value",
                                           aggfunc="sum").reset_index()
        mean_std_df["mean"] = mean_std_df["sum"] / (mean_std_df["count"] - mean_std_df["nan"])
        mean_std_df["std"] = (mean_std_df["sq_dev_sum"] / (
                mean_std_df["count"] - mean_std_df["nan"])) ** 0.5

        # Convert the pivot table back to long format
        mean_std_data = mean_std_df.melt(id_vars=["variable"], var_name="statistic", value_name="value")

        # Combine the mean and std with the rest of the DataFrame
        complete_stats = pd.concat([stats_df, mean_std_data[mean_std_data["statistic"].isin(["mean", "std"])]],
                                   ignore_index=True)

        # Place the aggregated data back into the main DataFrame
        aggregate_numerical_df = pd.concat([aggregate_numerical_df, complete_stats], ignore_index=True)

    return aggregate_numerical_df
