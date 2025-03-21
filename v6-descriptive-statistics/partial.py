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
        df = collect_sparql_data(df, variables_to_describe)

    if len(df) <= sample_size_threshold:
        warn(f"Sub-task was not executed because the number of samples is too small (n <= {sample_size_threshold})")
        return {"N-Threshold not met": client.organization_id}

    # Create a list to store the names of excluded variables (if any)
    excluded_variables = []

    # Convert specified columns to categorical or numerical types
    for variable, variable_info in variables_to_describe.items():
        if variable in df.columns:
            if (df[variable].notnull().sum() <= sample_size_threshold or (
                    df[variable] !=
                    "http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C54031").sum() <= sample_size_threshold):
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

    return {"organisation": client.organization.get(client.organization_id).get("name"),
            "categorical": categorical_df.to_json(), "numerical": numerical_df.to_json(),
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

        # Count the occurrences of 'true_na'
        if value_counts.get('_true_missing_'):
            true_na_count = value_counts.get('_true_missing_')
            # Delete it from the value counts for cleanliness
            del value_counts['_true_missing_']
        else:
            true_na_count = 0

        # Store the value counts and the sum of outliers in the dictionary
        categorical_descriptives[column_name] = {
            "value_counts": value_counts,
            "outliers": sum(outliers.values()),
            "nan": true_na_count
        }

    # Prepare the data for the final DataFrame
    categorical_data = [
                           (var, val, cnt) for var, vals in categorical_descriptives.items()
                           for val, cnt in vals["value_counts"].items()
                       ] + [
                           (var, "outliers", vals["outliers"]) for var, vals in categorical_descriptives.items()
                       ] + [
                           (var, "nan", vals["nan"]) for var, vals in categorical_descriptives.items()
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

        # Count the occurrences of the numerical placeholder
        true_na_count = (column_values == -9999999999999999999999999999).sum()

        # Replace the numerical placeholder with pd.NA
        column_values = column_values.replace(-9999999999999999999999999999, pd.NA)

        inliers_range = variables_to_describe[column_name].get("inliers", (float("-inf"), float("inf")))

        # Check if inliers is a tuple of two values
        if not (isinstance(inliers_range, list) and len(inliers_range) == 2):
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
            (column_name, "nan", true_na_count),
            (column_name, "sum", float(inlier_values.sum())),
            (column_name, "count", int(inlier_values.count())),
            (column_name, "sq_dev_sum", float((inlier_values - inlier_values.mean()).pow(2).sum())),
            (column_name, "std", float(inlier_values.std())),
            (column_name, "outliers", int(len(outliers)))
        ])

    # Convert the list to a DataFrame
    numerical_df = pd.DataFrame(numerical_data, columns=["variable", "statistic", "value"])

    return numerical_df


def collect_sparql_data(df: pd.DataFrame, variables_to_describe: dict) -> pd.DataFrame:
    """
    Collect data from SPARQL endpoints for categorical and numerical variables.

    This function reads SPARQL query templates, executes them for each variable based on its datatype,
    processes the results, and combines them into a single DataFrame. It handles both categorical and
    numerical data differently, with special processing for categorical variables that may have
    subclasses.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing at least an 'endpoint' column with SPARQL endpoint URLs.
    variables_to_describe (dict): Dictionary mapping variable names to their properties, including 'datatype'.
                                Each variable must have a 'datatype' key with value 'categorical' or 'numerical'.

    Returns:
    pd.DataFrame: A combined DataFrame containing all retrieved data, with 'patient_id' as the index column
                and each variable as a separate column. Returns the input DataFrame if no data is retrieved.
    """
    try:
        # Load SPARQL query templates for both categorical and numerical data
        _query_categories = open(
            f'{os.path.sep}app{os.path.sep}v6-descriptive-statistics{os.path.sep}retrieve_categorical_columns.rq',
            'r').read()
        _query_continuous = open(
            f'{os.path.sep}app{os.path.sep}v6-descriptive-statistics{os.path.sep}retrieve_continuous_columns.rq',
            'r').read()

    except Exception as e:
        # Return original DataFrame if query templates cannot be loaded
        error(f"Error reading SPARQL query file: {e}")
        return df

    # Initialize result storage
    intermediate_df = pd.DataFrame()
    endpoint = df["endpoint"].iloc[0]  # Get SPARQL endpoint URL

    # Process each variable according to its type
    for variable, variable_info in variables_to_describe.items():
        try:
            result_df = pd.DataFrame()

            if variable_info["datatype"] == "categorical":
                # Extract ontology prefix for categorical variables (e.g., "ncit:")
                ontology_part = variable.split(":")[0] + ":"

                # Prepare and execute categorical query
                query = _query_categories.replace("PLACEHOLDER_CLASS", variable).replace("PLACEHOLDER_ONTOLOGY",
                                                                                         ontology_part)
                info(f"Posting categorical SPARQL query for {variable}")
                result = post_sparql_query(endpoint=endpoint, query=query)

                if result:
                    # Process categorical query results
                    result_df = pd.DataFrame(result)
                    result_df.drop(columns=['patient'], inplace=True)
                    result_df['patient_id'] = result_df.index

                    # Handle hierarchical categorical data with subclasses
                    if 'sub_class' in result_df.columns:
                        # Use subclass values where available, fall back to direct values
                        result_df[variable] = result_df.apply(
                            lambda row: row['value'] if pd.isna(row['sub_class']) or row['sub_class'] == '' else row[
                                'sub_class'],
                            axis=1
                        )
                        result_df = result_df.drop(columns=['sub_class', 'value'])
                    else:
                        # Direct value mapping for non-hierarchical categories
                        result_df = result_df.rename(columns={'value': variable})

                    # Replace specific ontology URI with NA
                    result_df = result_df.replace("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C54031",
                                                  "_true_missing_")

            elif variable_info["datatype"] == "numerical":
                # Prepare and execute numerical query
                query = _query_continuous.replace("PLACEHOLDER_CLASS", variable)
                info(f"Posting numerical SPARQL query for {variable}")
                result = post_sparql_query(endpoint=endpoint, query=query)

                if result:
                    # Process numerical query results
                    result_df = pd.DataFrame(result)
                    result_df['patient_id'] = result_df.index
                    result_df = result_df.rename(columns={'value': variable})
                    # Replace specific ontology URI with NA
                    result_df = result_df.replace("http://ncicb.nci.nih.gov/xml/owl/EVS/Thesaurus.owl#C54031",
                                                  -9999999999999999999999999999)
                    result_df.fillna(-9999999999999999999999999999, inplace=True)

            # Combine results using outer join to preserve all patient data
            if not result_df.empty:
                if intermediate_df.empty:
                    intermediate_df = result_df
                else:
                    intermediate_df = pd.merge(intermediate_df, result_df, on="patient_id", how="outer")

        except Exception as e:
            error(f"Error processing {variable}: {e}")
            continue

    # Return combined results or original DataFrame if no data was retrieved
    return intermediate_df if not intermediate_df.empty else df
