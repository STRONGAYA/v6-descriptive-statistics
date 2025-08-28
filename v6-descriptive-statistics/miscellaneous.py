from typing import Dict

from vantage6_strongaya_general.miscellaneous import VariableDetails


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
