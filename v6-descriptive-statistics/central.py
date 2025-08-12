from typing import Any, Dict, List

from vantage6.algorithm.tools.decorators import algorithm_client
from vantage6.algorithm.tools.exceptions import UserInputError
from vantage6.algorithm.client import AlgorithmClient

# General federated algorithm functions
from vantage6_strongaya_general.miscellaneous import (collect_organisation_ids, safe_log,
                                                      VariableDetails, StratificationDetails)
from vantage6_strongaya_general.general_statistics import compute_aggregate_general_statistics, \
    compute_aggregate_adjusted_deviation

from miscellaneous import check_input_structure


@algorithm_client
def central(client: AlgorithmClient, variables_to_describe: Dict[str, VariableDetails],
            variables_to_stratify: StratificationDetails = None,
            organisation_ids: List[int] = None) -> Dict[str, Any]:
    """
    Central function to aggregate descriptive statistics from multiple organisations.

    Args:
        client (AlgorithmClient): The client to communicate with the vantage6 server.
        variables_to_describe (VariablesToDescribe): Dictionary of variables to describe.
                                                                Example:
                                                                 {"Gender": {"datatype": "categorical",
                                                                             "inliers": ("M", "F", "X")},
                                                                  "Age": {"datatype": "numerical",
                                                                          "inliers": (15, 39)}},
        variables_to_stratify (StratificationDetails, optional): Dictionary of variables to stratify.
                                                                 Defaults to None.
                                                                Example:
                                                                    {'Age':
                                                                            {
                                                                            'end': 39,
                                                                            'datatype': 'int'
                                                                            }
                                                                    }
        organisation_ids (list[int], optional): List of organisation IDs to include.
                                                Defaults to None - therewith including all organisations.

    Returns:
        Any: A dictionary containing the aggregated descriptive statistics and
        the list of included and excluded organisations.
    """
    # Check if the users' input structure is correct
    if not check_input_structure(variables_to_describe):
        raise UserInputError("Algorithm input is incorrect. Please check the algorithm input.")

    # Collect all organisations that participate in this collaboration unless specified
    organisation_ids = collect_organisation_ids(organisation_ids, client)

    # Create the subtask for general statistics
    safe_log("info", "Creating subtask to calculate general statistics.")

    input_ = {"method": "partial_general_statistics",
              "kwargs": {
                  "variables_to_describe": variables_to_describe,
                  "variables_to_stratify": variables_to_stratify}
              }

    task_general_statistics = client.task.create(input_, organisation_ids,
                                                 "Descriptive Statistics - General",
                                                 "This subtask determines the general statistics of "
                                                 "the variables to describe.")

    # Wait for the node(s) to return the results of the subtask
    safe_log("info", f"Waiting for results of task {task_general_statistics.get('id')}")
    results_general_statistics = client.wait_for_results(task_general_statistics.get("id"))
    safe_log("info", f"Results of task {task_general_statistics.get('id')} obtained")

    # Aggregate the general statistics
    results_general_statistics = compute_aggregate_general_statistics(results_general_statistics)

    # Create a subtask to calculate aggregate-adjusted deviation; using the aggregated numerical general statistics
    safe_log("info", "Creating subtask to calculate aggregate-adjusted deviation using general statistics.")

    input_ = {"method": "partial_aggregate_adjusted_deviation",
              "kwargs": {
                  "numerical_aggregated_results": results_general_statistics["numerical_general_statistics"],
                  "variables_to_describe": variables_to_describe,
                  "variables_to_stratify": variables_to_stratify}
              }

    task_adjusted_deviation = client.task.create(input_, organisation_ids,
                                                 "Descriptive Statistics - Aggregate Adjusted Deviation",
                                                 "This subtask determines the aggregate-adjusted deviation of "
                                                 "the variables to describe.")

    # Wait for the node(s) to return the results of the subtask
    safe_log("info", f"Waiting for results of task {task_adjusted_deviation.get('id')}")
    results_deviation = client.wait_for_results(task_adjusted_deviation.get("id"))
    safe_log("info", f"Results of task {task_adjusted_deviation.get('id')} obtained")

    # Compute the aggregate of the aggregate-adjusted deviation and include it in the general statistics
    results = compute_aggregate_adjusted_deviation(results_deviation, results_general_statistics)

    # Return the final results of the algorithm
    return results
