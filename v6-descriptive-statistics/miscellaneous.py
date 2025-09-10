import pandas as pd

from io import StringIO
from typing import Dict, Union, List, TypedDict
from vantage6.algorithm.tools.exceptions import PrivacyThresholdViolation
from vantage6.algorithm.tools.util import get_env_var

from vantage6_strongaya_general.miscellaneous import safe_log


# Define VariableDetails type locally to avoid external dependency during testing
class CategoricalDetails(TypedDict):
    datatype: str
    inliers: List[str]


class NonCategoricalDetails(TypedDict):
    datatype: str
    inliers: List[Union[int, float]]


VariableDetails = Union[CategoricalDetails, NonCategoricalDetails]

# Try to import from vantage6_strongaya_general, fall back to local definition if not available
try:
    from vantage6_strongaya_general.miscellaneous import VariableDetails
except ImportError:
    # Use local definition defined above
    pass


def check_input_structure(variables_to_describe: Dict[str, VariableDetails]) -> bool:
    """
    Check if the input structure for the algorithm is correct.

    Args:
        variables_to_describe (Dict[str, VariableDetails]): The dictionary containing the variables to describe.

    Returns:
        bool: True if the input structure is correct, False otherwise.
    """
    if not isinstance(variables_to_describe, dict):
        return False

    if not variables_to_describe:
        return False

    for variable_name, variable_details in variables_to_describe.items():
        if not isinstance(variable_name, str):
            return False

        if not isinstance(variable_details, dict):
            return False

        # Check required fields
        if "datatype" not in variable_details:
            return False

        datatype = variable_details["datatype"]
        if datatype not in ["numerical", "categorical"]:
            return False

        # If inliers are specified, validate their structure
        if "inliers" in variable_details:
            inliers = variable_details["inliers"]
            if datatype == "numerical":
                # Should be a tuple of (min, max)
                if not isinstance(inliers, (tuple, list)) or len(inliers) != 2:
                    return False
                if not all(isinstance(x, (int, float)) for x in inliers):
                    return False
                if inliers[0] >= inliers[1]:
                    return False
            elif datatype == "categorical":
                # Should be a list/tuple of valid categories
                if not isinstance(inliers, (tuple, list)):
                    return False
                if not all(isinstance(x, str) for x in inliers):
                    return False

    return True


def check_and_enforce_sample_size_threshold(result: Dict[str, str]) -> Dict[str, str]:
    """
    Check if all counts in the statistics result meet the sample size threshold.
    Remove statistics that don't meet the threshold and raise privacy violations.

    Args:
        result (Dict[str, str]): Dictionary containing statistical results with JSON strings

    Returns:
        Dict[str, str]: Filtered result with only statistics meeting the threshold

    Raises:
        PrivacyThresholdViolation: If privacy threshold violations are detected
    """
    # Retrieve the sample size threshold
    sample_size_threshold = get_env_var("SAMPLE_SIZE_THRESHOLD")
    try:
        sample_size_threshold = int(sample_size_threshold)
    except TypeError:
        sample_size_threshold = 10

    safe_log(
        "info",
        f"Applying sample size threshold of '{str(sample_size_threshold)}' to result and any sub-results.",
    )

    filtered_result = {}
    privacy_violations = []

    # Process categorical statistics
    if "categorical_general_partial_statistics" in result:
        categorical_json = result["categorical_general_partial_statistics"]
        categorical_df = pd.read_json(StringIO(categorical_json))

        if not categorical_df.empty:
            valid_categorical_rows = []

            # Group by variable to check each variable separately
            for variable_name in categorical_df["variable"].unique():
                variable_rows = categorical_df[
                    categorical_df["variable"] == variable_name
                ]

                # Separate data rows from metadata rows (na, outliers)
                data_rows = variable_rows[
                    ~variable_rows["value"].isin(["na", "outliers"])
                ]
                metadata_rows = variable_rows[
                    variable_rows["value"].isin(["na", "outliers"])
                ]

                # Check if any data categories meet the threshold
                valid_data_rows = data_rows[data_rows["count"] >= sample_size_threshold]

                if valid_data_rows.empty and not data_rows.empty:
                    # No valid categories left for this variable (excluding na/outliers)
                    privacy_violations.append("categorical_violation")
                    continue  # Skip this variable entirely

                # Keep valid data rows and all metadata rows (na/outliers are always allowed)
                valid_rows = pd.concat(
                    [valid_data_rows, metadata_rows], ignore_index=True
                )
                valid_categorical_rows.append(valid_rows)

            # Combine all valid categorical rows
            if valid_categorical_rows:
                combined_categorical_df = pd.concat(
                    valid_categorical_rows, ignore_index=True
                )
                filtered_result["categorical_general_partial_statistics"] = (
                    combined_categorical_df.to_json()
                )

    # Process numerical statistics
    if "numerical_general_partial_statistics" in result:
        numerical_json = result["numerical_general_partial_statistics"]
        numerical_df = pd.read_json(StringIO(numerical_json))

        if not numerical_df.empty:
            valid_numerical_rows = []

            # Group by variable to check each variable separately
            for variable_name in numerical_df["variable"].unique():
                variable_rows = numerical_df[numerical_df["variable"] == variable_name]

                # Find the count row for this variable
                count_row = variable_rows[variable_rows["statistic"] == "count"]

                if not count_row.empty:
                    count_value = count_row.iloc[0]["value"]

                    if count_value < sample_size_threshold:
                        # Count doesn't meet threshold - remove all statistics for this variable
                        privacy_violations.append("numerical_violation")
                        continue  # Skip this variable entirely

                # Variable meets threshold - keep all its statistics
                valid_numerical_rows.append(variable_rows)

            # Combine all valid numerical rows
            if valid_numerical_rows:
                combined_numerical_df = pd.concat(
                    valid_numerical_rows, ignore_index=True
                )
                filtered_result["numerical_general_partial_statistics"] = (
                    combined_numerical_df.to_json()
                )

    # Copy any other results that don't need threshold checking
    for key, value in result.items():
        if key not in [
            "categorical_general_partial_statistics",
            "numerical_general_partial_statistics",
        ]:
            filtered_result[key] = value

    # Raise privacy violation if any were detected
    if privacy_violations:
        raise PrivacyThresholdViolation(
            "Privacy threshold violation detected in statistical results."
        )

    return filtered_result
