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
    # TODO implement checks for the input structure that is to be called in central.py 'central'
    #  function, not as test, but as callable function.
    return True
