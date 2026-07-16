import pandas as pd

from typing import Dict
from vantage6.algorithm.tools.decorators import algorithm_client, data
from vantage6.algorithm.client import AlgorithmClient
from vantage6.algorithm.tools.exceptions import UserInputError

# General federated algorithm functions
from vantage6_strongaya_general.general_statistics import (
    compute_local_general_statistics,
    compute_local_adjusted_deviation,
)
from vantage6_strongaya_general.miscellaneous import (
    apply_data_stratification,
    set_datatypes,
    safe_log,
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
