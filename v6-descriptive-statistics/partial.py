"""
This file contains all partial algorithm functions, that are normally executed
on all nodes for which the algorithm is executed.

The results in a return statement are sent to the vantage6 server (after
encryption if that is enabled). From there, they are sent to the partial task
or directly to the user (if they requested partial results).
"""
import os

import pandas as pd

from .post_query import post_sparql_query
from typing import Any
from vantage6.algorithm.tools.util import info, warn, error
from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.decorators import data
from vantage6.algorithm.client import AlgorithmClient

sample_size_threshold = os.environ.get("SAMPLE_SIZE_THRESHOLD")
try:
    sample_size_threshold = int(sample_size_threshold)
except TypeError:
    sample_size_threshold = 10


@data(1)
@algorithm_client
def partial(client: AlgorithmClient, df: pd.DataFrame, variables_to_describe: dict,
            variables_to_stratify: dict = None) -> Any:
    """
    Partial function to aggregate descriptive statistics from a DataFrame.

    This function processes a DataFrame containing data from a single node,
    aggregates descriptive statistics for both numerical and categorical variables,
    and returns the combined statistics. It handles cases where the sample size
    is below a specified threshold and excludes such variables from the analysis.

    Parameters:
    client (AlgorithmClient): The client to communicate with the vantage6 server.
    df (pd.DataFrame): The input DataFrame containing the data.
    variables_to_describe (dict): Dictionary of variables to describe.
    variables_to_stratify (dict, optional): Dictionary of variables to stratify. Defaults to None.

    Returns:
    Any: A dictionary containing the aggregated descriptive statistics and the list of excluded variables.
    """
    # Suboptimal SPARQL solution, to be improved
    if "endpoint" in df.columns:
        try:
            # The "r" argument means the file will be opened in read mode
            query = open(
                f"{os.path.sep}app{os.path.sep}v6-descriptive-statistics{os.path.sep}retrieve_columns.rq",
                "r").read()
        except Exception as e:
            # If there is an error reading the file, log an error
            error(f"Unexpected error occurred whilst reading the SPARQL query file, error: {e}")

        # TODO place the classes in the query; replace PLACERHOLDER_CLASS with the class in variables_to_dict.keys()

        try:
            # Post the SPARQL query to the endpoint specified in df
            info(f"Posting SPARQL query to {df['endpoint'].iloc[0]}.")
            result = post_sparql_query(endpoint=df["endpoint"].iloc[0], query=query)
        except Exception as e:
            error(f"Unexpected error occurred whilst posting SPARQL query, error: {e}")

        # If the result is empty (i.e. the query returned no results) it is likely that the data is not available
        if not result:
            return {
                "Variable": [],
                "Value": [],
                "count": []
            }

        # If the result is not empty, create a DataFrame from the result
        df = pd.DataFrame(result)

    if len(df) <= sample_size_threshold:
        warn(f"Sub-task was not executed because the number of samples is too small (n <= {sample_size_threshold})")
        return {"N-Threshold not met": client.organization_id}

    # Create a list to store the names of excluded variables (if any)
    excluded_variables = []

    # Convert specified columns to categorical or numerical types
    for variable, variable_info in variables_to_describe.items():
        if (df[variable].notnull().sum() <= sample_size_threshold or (
                df[variable] != "ncit:C54031").sum() <= sample_size_threshold):
            warn(
                f"Descriptive statistics for {variable} were not computed because "
                f"the number of samples is too small (n <= {sample_size_threshold})")
            df = df.drop(columns=variable)
            continue

        if "categorical" in variable_info["datatype"]:
            df[variable] = df[variable].astype("category")
        if "numerical" in variable_info["datatype"]:
            df[variable] = pd.to_numeric(df[variable], errors="coerce")

    # Drop all unnecessary columns
    df = df.drop(columns=[col for col in df.columns if col not in variables_to_describe.keys()])

    # TODO handle subsets of data through stratification

    # Handle categorical columns
    categorical_df = retrieve_categorical_descriptives(df, variables_to_describe)

    # Handle numerical columns
    numerical_df = retrieve_numerical_descriptives(df, variables_to_describe)

    return {"categorical": categorical_df.to_json(), "numerical": numerical_df.to_json(),
            "excluded_variables": excluded_variables}


def retrieve_categorical_descriptives(df: pd.DataFrame, variables_to_describe: dict) -> pd.DataFrame:
    """
    Retrieve descriptive statistics for categorical variables in a DataFrame.

    This function processes categorical columns in the provided DataFrame,
    removes outliers based on the provided inliers list, and returns a DataFrame
    with the value counts and outliers for each categorical variable.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    variables_to_describe (dict): A dictionary where keys are column names and
                                  values are dictionaries containing inliers.

    Returns:
    pd.DataFrame: A DataFrame with columns "Variable", "Value", and "count"
                  representing the value counts and outliers for each categorical variable.
    """
    # Select categorical columns from the DataFrame
    categorical_columns = df.select_dtypes(include=["category"]).columns
    # Initialize a dictionary to store the descriptive statistics
    categorical_descriptives = {}

    # Iterate over each categorical column
    for column_name in categorical_columns:
        info(f"Categorical column {column_name} is being described")
        # Get the value counts for the column
        value_counts = df[column_name].value_counts().to_dict()
        # Get the inliers for the column from the provided dictionary
        inliers = variables_to_describe[column_name].get("inliers", [])

        # Identify outliers by excluding inliers from value counts
        outliers = {value: count for value, count in value_counts.items() if value not in inliers and len(inliers) > 0}
        # Remove outliers from the value counts
        for outlier in outliers:
            del value_counts[outlier]

        # Store the value counts and the sum of outliers in the dictionary
        categorical_descriptives[column_name] = {
            "value_counts": value_counts,
            "outliers": sum(outliers.values())
        }

    # Prepare the data for the final DataFrame
    categorical_data = [
                           (var, val, cnt) for var, vals in categorical_descriptives.items()
                           for val, cnt in vals["value_counts"].items()
                       ] + [
                           (var, "outliers", vals["outliers"]) for var, vals in categorical_descriptives.items()
                       ]

    # Return the final DataFrame
    return pd.DataFrame(categorical_data, columns=["variable", "value", "count"])


def retrieve_numerical_descriptives(df: pd.DataFrame, variables_to_describe: dict) -> pd.DataFrame:
    """
    Retrieve descriptive statistics for numerical variables in a DataFrame.

    This function processes numerical columns in the provided DataFrame, 
    removes outliers based on the provided range tuple,
    and returns a DataFrame with the statistics for each numerical variable.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    variables_to_describe (dict): A dictionary where keys are column names and 
    values are dictionaries containing inliers.

    Returns:
    pd.DataFrame: A DataFrame with columns "variable", "statistic", 
    and "value" representing the statistics for each numerical variable.
    """
    numerical_columns = df.select_dtypes(include=["number"]).columns

    # Initialize a list to store the numerical descriptives
    numerical_data = []

    # Compute the descriptive statistics for the numerical columns
    for column_name in numerical_columns:
        info(f"Numerical column {column_name} is being described")
        column_values = df[column_name]
        inliers_range = variables_to_describe[column_name].get("inliers", (float("-inf"), float("inf")))

        # Check if inliers is a tuple of two values
        if not (isinstance(inliers_range, tuple) and len(inliers_range) == 2):
            warn(f"Inliers for {column_name} are not a tuple of two values. Proceeding without determining outliers.")
            inliers_range = (float("-inf"), float("inf"))

        # Identify outliers by excluding values outside the inliers range
        outliers = column_values[(column_values < inliers_range[0]) | (column_values > inliers_range[1])]
        inlier_values = column_values[(column_values >= inliers_range[0]) & (column_values <= inliers_range[1])]

        q1, median, q3 = inlier_values.quantile([0.25, 0.5, 0.75]).values

        # Append the statistics to the list
        numerical_data.extend([
            (column_name, "min", float(inlier_values.min())),
            (column_name, "q1", float(q1)),
            (column_name, "median", float(median)),
            (column_name, "mean", float(inlier_values.mean())),
            (column_name, "q3", float(q3)),
            (column_name, "max", float(inlier_values.max())),
            (column_name, "nan", int(inlier_values.isna().sum())),
            (column_name, "sum", float(inlier_values.sum())),
            (column_name, "count", int(inlier_values.count())),
            (column_name, "sq_dev_sum", float((inlier_values - inlier_values.mean()).pow(2).sum())),
            (column_name, "std", float(inlier_values.std())),
            (column_name, "outliers", int(len(outliers)))
        ])

    # Convert the list to a DataFrame
    numerical_df = pd.DataFrame(numerical_data, columns=["variable", "statistic", "value"])

    return numerical_df
