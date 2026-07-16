import pandas as pd

from io import StringIO
from typing import Dict, Optional
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.exceptions import UserInputError

# General federated algorithm functions
from vantage6_strongaya_general.general_statistics import (
    compute_local_general_statistics,
    _compute_local_inliers_and_outliers,
    _compute_local_aggregated_adjusted_deviation,
)
from vantage6_strongaya_general.miscellaneous import (
    apply_data_stratification,
    set_datatypes,
    safe_log,
    safe_calculate,
    VariableDetails,
    StratificationDetails,
)
from vantage6_strongaya_general.privacy_measures import (
    apply_sample_size_threshold,
    mask_unnecessary_variables,
)

from vantage6_strongaya_rdf.collect_sparql_data import collect_sparql_data

from .miscellaneous import (
    check_and_enforce_sample_size_threshold,
    remove_min_max_from_results,
)


def _normalize_inliers(
    variables_to_describe: Dict[str, VariableDetails],
) -> Dict[str, VariableDetails]:
    """
    Normalize inliers in variable descriptions from tuples to lists.

    The upstream general library requires inliers to be lists, not tuples.
    This function converts any tuple inliers to lists to ensure compatibility.

    Args:
        variables_to_describe: Dictionary of variables with their details.

    Returns:
        Dictionary with inliers converted to lists where applicable.
    """
    normalized = {}
    for variable_name, details in variables_to_describe.items():
        details_copy = dict(details)
        if "inliers" in details_copy and isinstance(details_copy["inliers"], tuple):
            details_copy["inliers"] = list(details_copy["inliers"])
        normalized[variable_name] = details_copy
    return normalized


def compute_local_adjusted_deviation(
    df: pd.DataFrame, numerical_aggregated_results: Optional[str] = None
) -> Dict[str, str]:
    """
    Compute local adjusted deviation for the given DataFrame.

    This is a patched version of the upstream function that fixes a kwarg name
    mismatch (aggregated_mean vs aggregate_mean) in the call to
    _compute_local_aggregated_adjusted_deviation.

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        numerical_aggregated_results (Optional[str]): JSON string with the general numerical statistics.

    Returns:
        Dict[str, str]: A dictionary with the local adjusted deviation in JSON format.
    """
    # Collect the general numerical aggregates (safely)
    numerical_results = (
        "{}" if numerical_aggregated_results is None else numerical_aggregated_results
    )
    numerical_df = pd.read_json(StringIO(numerical_results))

    # Collect the variable(s) for which an adjusted deviation can actually be calculated
    variables_to_analyse = [
        column_name
        for column_name in df.columns
        if not numerical_df.empty and column_name in numerical_df["variable"].unique()
    ]

    if not variables_to_analyse:
        adjusted_deviation = pd.DataFrame(columns=["variable", "statistic", "value"])
        safe_log(
            "warn",
            "No variables to analyse for adjusted deviation due to lacking aggregate numerical statistics",
        )
    else:
        adjusted_deviation = safe_calculate(
            _orchestrate_local_adjusted_deviation,
            pd.DataFrame(columns=["variable", "statistic", "value"]),
            df=df,
            numerical_aggregated_results=numerical_df,
        )

    return {"adjusted_deviation": adjusted_deviation.to_json()}


def _orchestrate_local_adjusted_deviation(
    df: pd.DataFrame,
    numerical_aggregated_results: pd.DataFrame,
    variable_details: Optional[Dict] = None,
) -> pd.DataFrame:
    """
    Compute local adjusted deviation for the given DataFrame.

    This is a patched version that fixes the kwarg name mismatch in the upstream library
    (aggregated_mean vs aggregate_mean).

    Args:
        df (pd.DataFrame): The input DataFrame containing the data.
        numerical_aggregated_results (pd.DataFrame): DataFrame with general numerical statistics.
        variable_details (Optional[Dict]): A dictionary where keys are column names
                                 and values are dictionaries containing inliers.

    Returns:
        pd.DataFrame: DataFrame with adjusted deviation for each variable.
    """
    adjusted_deviations = []

    for column_name in df.columns:
        safe_log(
            "info", f"Adjusted deviation for variable {column_name} is being computed"
        )

        column_values = df[column_name]

        # Get the aggregated mean safely
        try:
            aggregated_mean = numerical_aggregated_results.loc[
                (numerical_aggregated_results["variable"] == column_name)
                & (numerical_aggregated_results["statistic"] == "mean"),
                "value",
            ].values[0]
        except IndexError:
            safe_log("warn", f"No aggregated mean found for variable {column_name}")
            continue

        # Get the inliers for the column from the provided dictionary
        if variable_details is not None and column_name in variable_details:
            inliers_range = variable_details[column_name].get(
                "inliers", [float("-inf"), float("inf")]
            )
            datatype = variable_details[column_name].get("datatype", "numerical")
        else:
            inliers_range = [float("-inf"), float("inf")]
            datatype = "numerical"

        # Identify outliers by excluding values outside the inliers range safely
        inliers_series, outliers_series = safe_calculate(
            _compute_local_inliers_and_outliers,
            (pd.Series(dtype="float64"), pd.Series(dtype="float64")),
            column_values=column_values,
            inliers=inliers_range,
            datatype=datatype,
        )

        # Compute the adjusted sum of squared errors safely (fixed kwarg name)
        adjusted_sum_of_squared_errors, number_of_rows = safe_calculate(
            _compute_local_aggregated_adjusted_deviation,
            (0.0, 0),
            inliers_series=inliers_series,
            aggregate_mean=aggregated_mean,
        )

        adjusted_deviations.append(
            (
                column_name,
                "adjusted_sum_of_squared_errors",
                adjusted_sum_of_squared_errors,
            )
        )
        adjusted_deviations.append((column_name, "count", number_of_rows))

    # Convert the list to a DataFrame
    adjusted_deviation_df = pd.DataFrame(
        adjusted_deviations, columns=["variable", "statistic", "value"]
    )

    return adjusted_deviation_df


@data(1)
@algorithm_client
def partial_general_statistics(
    client: AlgorithmClient,
    df: pd.DataFrame,
    variables_to_describe: Dict[str, VariableDetails],
    variables_to_stratify: StratificationDetails = None,
) -> Dict[str, str]:
    """
    Execute the partial algorithm for general statistics computation.

    Args:
        client (AlgorithmClient): The client to communicate with the vantage6 server.
        df (pd.DataFrame): The DataFrame containing the data to be processed.
        variables_to_describe (Dict[str, VariableDetails]): Dictionary of variables to describe.
                                                                Example:
                                                                 {"Gender": {"datatype": "categorical",
                                                                             "inliers": ("M", "F", "X")},
                                                                  "Age": {"datatype": "numerical",
                                                                          "inliers": (15, 39)}},
        variables_to_stratify (StratificationDetails, optional): Dictionary of variables to stratify. Defaults to None.
                                                                Example:
                                                                    {'Age':
                                                                            {
                                                                            'end': 39,
                                                                            'datatype': 'int'
                                                                            }
                                                                    }

    Returns:
        dict: A dictionary containing the computed general statistics.
    """
    safe_log("info", "Executing partial algorithm for general statistics computation.")

    # Normalize inliers from tuples to lists for compatibility with upstream library
    variables_to_describe = _normalize_inliers(variables_to_describe)

    # Add the variables to stratify to the variables to analyse
    if variables_to_stratify is not None:
        variables_to_analyse = list(variables_to_describe.keys()) + [
            variable_to_stratify
            for variable_to_stratify in variables_to_stratify.keys()
        ]
    else:
        variables_to_analyse = list(variables_to_describe.keys())

    # Retrieve RDF/SPARQL data if its use is indicated in the data - suboptimal solution, to be improved in the future
    if "endpoint" in df.columns:
        df = collect_sparql_data(variables_to_analyse, endpoint=df["endpoint"].iloc[0])

    # Mask unnecessary variables by removal - relevant, for example, with csv data
    df = mask_unnecessary_variables(df, variables_to_analyse)

    # Add the variables to stratify details to the variables to analyse details
    if variables_to_stratify is not None:
        # Set datatypes for each variable
        df = set_datatypes(df, variables_to_describe | variables_to_stratify)
    else:
        # Set datatypes for the variables to be described
        df = set_datatypes(df, variables_to_describe)

    # Reformat the variables_to_stratify to the expected format after datatypes have been set
    if variables_to_stratify is not None:
        variables_to_stratify = {
            variable: (
                details["values"]
                if details.get("datatype") == "categorical" and "values" in details
                else details
            )
            for variable, details in variables_to_stratify.items()
        }

    # Apply stratification if necessary
    df = apply_data_stratification(df, variables_to_stratify)

    # Ensure that any of the variables to describe are present
    if not any(variable in df.columns for variable in variables_to_describe):
        raise UserInputError(
            "None of the variables to describe are present in the data."
        )

    # Ensure that the sample size threshold is met
    df = apply_sample_size_threshold(client, df, variables_to_analyse)

    # Compute general statistics
    result = compute_local_general_statistics(df, variables_to_describe)

    # Remove minimum and maximum from the results
    result = remove_min_max_from_results(result)

    # Check the output to make sure all counts are above the sample size threshold or mask them otherwise
    result = check_and_enforce_sample_size_threshold(result)

    return result


@data(1)
@algorithm_client
def partial_aggregate_adjusted_deviation(
    client: AlgorithmClient,
    df: pd.DataFrame,
    variables_to_describe: Dict[str, VariableDetails],
    numerical_aggregated_results: Dict[str, str],
    variables_to_stratify: StratificationDetails = None,
) -> dict[str, str]:
    """
    Execute the partial algorithm for aggregate-adjusted deviation computation.

    Args:
        client (AlgorithmClient): The client to communicate with the vantage6 server.
        variables_to_describe (Dict[str, VariableDetails]): Dictionary of variables to describe.
                                                                Example:
                                                                 {"Gender": {"datatype": "categorical",
                                                                             "inliers": ("M", "F", "X")},
                                                                  "Age": {"datatype": "numerical",
                                                                          "inliers": (15, 39)}},
        numerical_aggregated_results (dict): Dictionary of numerical aggregated results.
        variables_to_stratify (StratificationDetails, optional): Dictionary of variables to stratify. Defaults to None.
                                                                Example:
                                                                    {'Age':
                                                                            {
                                                                            'end': 39,
                                                                            'datatype': 'int'
                                                                            }
                                                                    }

    Returns:
        dict: A dictionary containing the computed aggregate adjusted deviation.
    """
    safe_log(
        "info",
        "Executing partial algorithm to compute the aggregate adjusted deviation.",
    )

    # Normalize inliers from tuples to lists for compatibility with upstream library
    variables_to_describe = _normalize_inliers(variables_to_describe)

    # Add the variables to stratify to the variables to analyse
    if variables_to_stratify is not None:
        variables_to_analyse = list(variables_to_describe.keys()) + [
            variable_to_stratify
            for variable_to_stratify in variables_to_stratify.keys()
        ]
    else:
        variables_to_analyse = list(variables_to_describe.keys())

    # Retrieve RDF/SPARQL data if its use is indicated in the data - suboptimal solution, to be improved in the future
    if "endpoint" in df.columns:
        df = collect_sparql_data(variables_to_analyse, endpoint=df["endpoint"].iloc[0])

    # Mask unnecessary variables by removal - relevant, for example, with csv data
    df = mask_unnecessary_variables(df, variables_to_analyse)

    # Add the variables to stratify details to the variables to analyse details
    if variables_to_stratify is not None:
        # Set datatypes for each variable
        df = set_datatypes(df, variables_to_describe | variables_to_stratify)
    else:
        # Set datatypes for the variables to be described
        df = set_datatypes(df, variables_to_describe)

    # Reformat the variables_to_stratify to the expected format after datatypes have been set
    if variables_to_stratify is not None:
        variables_to_stratify = {
            variable: (
                details["values"]
                if details.get("datatype") == "categorical" and "values" in details
                else details
            )
            for variable, details in variables_to_stratify.items()
        }

    # Apply stratification if necessary
    df = apply_data_stratification(df, variables_to_stratify)

    # Ensure that any of the variables to describe are present
    if not any(variable in df.columns for variable in variables_to_describe):
        raise UserInputError(
            "None of the variables to describe are present in the data."
        )

    # Ensure that the sample size threshold is met
    df = apply_sample_size_threshold(client, df, variables_to_analyse)

    # Compute aggregate-adjusted deviation
    result = compute_local_adjusted_deviation(df, numerical_aggregated_results)

    return result
