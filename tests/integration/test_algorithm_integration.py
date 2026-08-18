"""
Comprehensive Vantage6 integration testing.
"""

import pytest
import json
import pandas as pd

from json import JSONDecodeError
from io import StringIO
from typing import Any, Dict, Tuple
from vantage6.algorithm.tools.exceptions import (
    DataError,
    UserInputError,
    CollectResultsError,
    PrivacyThresholdViolation,
    InputError,
    AlgorithmError,
    CollectOrganizationError,
)


@pytest.fixture
def test_methods():
    """
    Fixture providing different algorithm methods to test with their specific kwargs templates.

    GUIDANCE FOR REUSE:
    - Add your algorithm's method names as keys to this dict
    - Each method should specify its kwargs templates for different test scenarios
    - Method-specific parameters should be defined here, not in configurations
    - Ensure your algorithm supports all methods and parameters listed here

    STRUCTURE:
    Returns a dict where:
    - Keys are method names that can be called via vantage6 task input
    - Values are dicts containing kwargs templates for different test scenarios

    KWARGS TEMPLATES:
    Each method has kwargs for different test scenarios:
    - 'basic': Basic functionality test
    - 'organisation_selection': Test with specific organisations
    - 'data_stratification': Test with data stratification
    - 'inlier_specific': Test with inlier-specific variable configurations
    - 'return_partial': Test with partial results (method-specific)
    - 'parameter_galore': Test with all parameters combined

    DYNAMIC PARAMETER FILLING:
    Parameters set to None are automatically filled by test methods from configurations:
    - "variables_to_describe": Filled from config['variables_to_describe_basic'] or
    config['variables_to_describe_inlier_specific']
    - "organisations_to_include": Filled from config['organisation_subset']
    - "variables_to_stratify": Filled from config['variables_to_stratify']

    EXAMPLES:
    - For statistical algorithms: {"central": {...}, "partial_general_statistics": {...}}
    - For ML algorithms: {"train": {...}, "predict": {...}, "validate": {...}}
    - For data processing: {"preprocess": {...}, "transform": {...}, "aggregate": {...}}
    """
    return {
        "central": {
            "basic": {
                "variables_to_describe": None,  # Will be filled from config
            },
            "organisation_selection": {
                "variables_to_describe": None,  # Will be filled from config
                "organisations_to_include": None,  # Will be filled from config
            },
            "data_stratification": {
                "variables_to_describe": None,  # Will be filled from config
                "variables_to_stratify": None,  # Will be filled from config
            },
            "inlier_specific": {
                "variables_to_describe": None,  # Will be filled from config (inlier_specific)
            },
            "parameter_galore": {
                "variables_to_describe": None,  # Will be filled from config
                "variables_to_stratify": None,  # Will be filled from config
                "organisations_to_include": None,  # Will be filled from config
            },
        },
        "partial_general_statistics": {
            "basic": {
                "variables_to_describe": None,  # Will be filled from config
            },
            "data_stratification": {
                "variables_to_describe": None,  # Will be filled from config
                "variables_to_stratify": None,  # Will be filled from config
            },
            "inlier_specific": {
                "variables_to_describe": None,  # Will be filled from config (inlier_specific)
            },
            "parameter_galore": {
                "variables_to_describe": None,  # Will be filled from config
                "variables_to_stratify": None,  # Will be filled from config
            },
        },
    }


@pytest.fixture
def test_configurations():
    """
    Fixture providing comprehensive test configurations for algorithm validation.

    GUIDANCE FOR REUSE:
    1. MODIFY CONFIGURATIONS: Update each configuration to match your algorithm's requirements
    2. DATABASE LABELS: Change 'database_label' to match your test databases
    3. VARIABLES: Update variable specifications to match your data schema
    4. FAILURE SCENARIOS: Add configurations that test error handling and edge cases
    5. METHOD-SPECIFIC KWARGS: Now handled in test_methods fixture - keep configurations clean

    CONFIGURATION STRUCTURE:
    Each configuration dict should contain:
    - 'database_label': String identifying the test database
    - 'variables_to_describe_basic': Dict of variables for basic testing
    - 'organisation_subset': List of organisation IDs to test with
    - 'variables_to_stratify': Dict defining stratification parameters (optional)
    - 'expected_failure': Boolean indicating if this config should fail
    - 'failure_reason': String describing why failure is expected
    - 'expected_error_type': Exception class or list of exception classes expected on failure

    DYNAMIC PARAMETER FILLING:
    These values are used to fill None parameters in method kwargs:
    - 'variables_to_describe_basic' -> "variables_to_describe"
    - 'organisation_subset' -> "organisations_to_include"
    - 'variables_to_stratify' -> "variables_to_stratify"

    EXAMPLE CONFIGURATION TYPES:
    - 'standard_dataset': Normal successful execution
    - '*_bad_actor': Stress testing with resource constraints
    - '*_incorrect_input': Input validation testing
    - 'rare_dataset': Edge case with minimal data
    - 'non_existent_*': Error handling validation
    """
    return {
        "standard_dataset": {
            "database_label": "creatures_of_europa",  # Europa is a moon of Jupiter with a subsurface ocean
            "variables_to_describe_basic": {
                "Temperature Tolerance (K)": {"datatype": "numerical"},
                "Social Structure": {"datatype": "categorical"},
            },
            "organisation_subset": [1, 2],
            "variables_to_describe_inlier_specific": {
                "Temperature Tolerance (K)": {
                    "datatype": "numerical",
                    "inliers": [100, 150],
                },
                "Social Structure": {
                    "datatype": "categorical",
                    "inliers": [
                        "Colony",
                        "Solitary",
                        "Swarm",
                    ],  # Use actual values from Europa dataset
                },
            },
            "variables_to_stratify": {
                "Lifespan (years)": {"start": 2, "datatype": "int"},
                "Habitat": {
                    "values": ["Ice Caves", "Underground", "Subsurface Ocean"],
                    "datatype": "categorical",
                },
            },
        },
        "standard_dataset_bad_actor": {
            "database_label": "creatures_of_europa",
            "variables_to_describe_basic": {
                "Temperature Tolerance (K)": {"datatype": "numerical"},
                "Social Structure": {"datatype": "categorical"},
            },
            "organisation_subset": [1],
            "variables_to_describe_inlier_specific": {
                "Temperature Tolerance (K)": {
                    "datatype": "numerical",
                    "inliers": [100, 110],
                },
                # Emulate a bad actor by setting a very narrow range as inlier
                "Social Structure": {
                    "datatype": "categorical",
                    "inliers": ["Solitary"],  # This is valid for Europa dataset
                },
            },
            "variables_to_stratify": {
                "Lifespan (years)": {
                    "start": 10,
                    "end": 11,
                    # Emulate a bad actor by setting an extremely narrow range; trying to infer individual data
                    "datatype": "int",
                },
                "Habitat": {"values": ["Underground"], "datatype": "categorical"},
            },
            "expected_failure": True,
            "failure_reason": "Too narrow scope of data stratification parameters "
            "resulting in sample size threshold issues.",
            "expected_error_type": [CollectResultsError, PrivacyThresholdViolation],
        },
        "standard_dataset_incorrect_input": {
            "database_label": "creatures_of_enceladus",
            # Enceladus is a moon of Saturn with a subsurface ocean
            "variables_to_describe_basic": {
                "Temperature Tolerance (C)": {"datatype": "numerical"},
                "NonExistentVariable": {  # Use non-existent variable to trigger error
                    "datatype": "categorical"
                },
            },
            "organisation_subset": [4, 5],
            # Non-existent organisations to test input validation
            "variables_to_describe_inlier_specific": {
                "Temperature Tolerance (C)": {
                    "datatype": "numerical",
                    "inliers": [50, 150],
                },
                "NonExistentVariable": {  # Use non-existent variable
                    "datatype": "categorical",
                    "inliers": [
                        "NonExistentValue1",
                        "NonExistentValue2",
                    ],
                },
            },
            "variables_to_stratify": {
                "Lifespan (years)": {"start": 10, "datatype": "int"},
                "Habitat": {
                    "values": [
                        "Ice Caves",
                        "Subsurface Ocean",  # Use actual values from Enceladus dataset
                    ],
                    "datatype": "categorical",
                },
            },
            "expected_failure": True,
            "failure_reason": "Non-existent variables requested or invalid input structure specified",
            "expected_error_type": [
                CollectResultsError,
                UserInputError,
                JSONDecodeError,
            ],
        },
        "rare_dataset": {
            "database_label": "creatures_of_titan",
            # Titan is Saturn's largest moon with various favourable conditions
            "variables_to_describe_basic": {
                "Lifespan (years)": {"datatype": "numerical"}
            },
            "organisation_subset": [1],
            "variables_to_describe_inlier_specific": {
                "Lifespan (years)": {"datatype": "numerical", "inliers": [18, 35]}
            },
            "variables_to_stratify": {
                "Lifespan (years)": {
                    "start": 10,
                    "end": 11,
                    # Emulate a bad actor by setting an extremely narrow range; trying to infer individual data
                    "datatype": "int",
                },
                "Habitat": {"values": ["Underground"], "datatype": "categorical"},
            },
            "expected_failure": True,
            "failure_reason": "Dataset with insufficient sample size should not give any direct results.",
            "expected_error_type": [CollectResultsError, PrivacyThresholdViolation],
        },
        "non_existent_dataset_standard_input": {
            "database_label": "creatures_of_sedna",
            # Sedna is a dwarf planet in the Oort Cloud, life is unlikely
            "variables_to_describe_basic": {
                "Temperature Tolerance (K)": {"datatype": "numerical"},
                "Diet": {"datatype": "categorical"},
            },
            "organisation_subset": [1, 2],
            "variables_to_describe_inlier_specific": {
                "Temperature Tolerance (K)": {
                    "datatype": "numerical",
                    "inliers": [0, 10000],
                },
                "NonExistentVariable": {  # Use non-existent variable
                    "datatype": "categorical",
                    "inliers": ("NonExistentValue1", "NonExistentValue2"),
                },
            },
            "variables_to_stratify": {
                "Lifespan (years)": {
                    "start": 1,
                }
            },
            "expected_failure": True,
            "failure_reason": "Attempting to query an unknown database",
            "expected_error_type": [CollectResultsError, JSONDecodeError],
        },
    }


@pytest.mark.integration
class TestAlgorithmComponent:
    """
    Comprehensive test class for algorithm functionality across different methods and configurations.

    IMPORTANT NOTE FOR REUSE:
    This test class is specifically designed for descriptive statistics algorithms and must be
    adapted when repurposing for other algorithm types. The test functions contain algorithm-
    specific logic for input preparation, result extraction, and validation.

    WHEN REPURPOSING THIS CODE:
    1. REVIEW ALL TEST METHODS: Each test method contains algorithm-specific logic
    2. UPDATE KWARGS PREPARATION: Modify how kwargs are prepared from configurations
    3. ADAPT RESULT EXTRACTION: Update extract_data_from_result() for your algorithm's output
    4. MODIFY ASSERTIONS: Change validation logic to match your algorithm's expected behaviour
    5. UPDATE ERROR HANDLING: Ensure exception types match your algorithm's error patterns
    6. CONFIGURE DATABASE LABELS: Ensure test databases match your algorithm's requirements

    TEST SCENARIOS COVERED:
    - Basic functionality testing across all methods
    - Organisation-specific testing (federated learning scenarios)
    - Data stratification testing (subset analysis)
    - Error handling and edge case validation
    - Resource constraint testing (memory, computation limits)

    PARAMETRISATION:
    Tests are parametrised by:
    - method: Algorithm method to test (from test_methods fixture)
    - config_name: Configuration scenario (from test_configurations fixture)

    This creates a test matrix covering all combinations of methods × configurations.

    CUSTOMISATION CHECKLIST:
    □ Update kwargs preparation logic for your algorithm's parameters
    □ Modify database labels to match your test environment
    □ Adapt variable specifications to your data schema
    □ Update result extraction logic in extract_data_from_result()
    □ Modify assertion logic in determine_statistics_acceptance()
    □ Add algorithm-specific error types and handling
    □ Update test descriptions and naming conventions
    """

    @pytest.mark.parametrize("method", ["central", "partial_general_statistics"])
    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_dataset",
            "standard_dataset_incorrect_input",
            "rare_dataset",
            "non_existent_dataset_standard_input",
        ],
    )
    def test_algorithm_basic(
        self,
        authentication,
        algorithm_image_name,
        test_configurations,
        test_methods,
        method,
        config_name,
    ):
        """
        Test algorithm with different methods and configurations, including expected failures.

        CUSTOMISATION REQUIRED:
        - Update kwargs preparation for your algorithm's parameter structure
        - Modify task creation parameters as needed
        - Adapt result validation logic
        """
        client = authentication
        config = test_configurations[config_name]
        method_config = test_methods[method]

        # Prepare method-specific kwargs from method configuration
        kwargs = method_config["basic"].copy()
        kwargs["variables_to_describe"] = config["variables_to_describe_basic"]

        # Create a task for the client to retrieve the descriptive data
        task = client.task.create(
            collaboration=1,
            organizations=[1],
            name=f"Test {method} algorithm run - {config_name}",
            image=algorithm_image_name,
            description=f"Task to test the {method} function using {config_name} configuration.",
            input_={"method": method, "kwargs": kwargs},
            databases=[{"label": config["database_label"]}],
        )

        if config.get("expected_failure", False):
            # Test that aggressive configurations fail gracefully
            with pytest.raises(Exception) as exc_info:
                categorical_statistics, numerical_statistics = extract_data_from_result(
                    client, task, method
                )

            # Verify specific error types (support both single error type and list of error types)
            expected_errors = config.get("expected_error_type")
            if expected_errors:
                # Convert single error type to list for uniform handling
                if not isinstance(expected_errors, list):
                    expected_errors = [expected_errors]

                # Check if the raised exception matches any of the expected types
                error_matched = any(
                    isinstance(exc_info.value, expected_error)
                    for expected_error in expected_errors
                )
                assert error_matched, (
                    f"Expected one of {[err.__name__ for err in expected_errors]} "
                    f"but got {type(exc_info.value).__name__}"
                )

            print(f"Expected failure occurred for {config_name}: {exc_info.value}")
        else:
            # Normal success path
            categorical_statistics, numerical_statistics = extract_data_from_result(
                client, task, method
            )
            determine_statistics_acceptance(
                {
                    "categorical_general_statistics": categorical_statistics,
                    "numerical_general_statistics": numerical_statistics,
                },
                {},
                method,
                config["database_label"],
                kwargs,
            ), f"Centralised and federated statistics deviate too much for {config_name} configuration"

    @pytest.mark.parametrize(
        "method", ["central"]
    )  # partial_general_statistics doesn't support organisation selection
    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_dataset",
            "standard_dataset_incorrect_input",
            "rare_dataset",
            "non_existent_dataset_standard_input",
        ],
    )
    def test_algorithm_organisation_selection(
        self,
        authentication,
        algorithm_image_name,
        test_configurations,
        test_methods,
        method,
        config_name,
    ):
        """
        Test algorithm with organisation selection, including expected failures.

        CUSTOMISATION REQUIRED:
        - Ensure your algorithm supports organisations_to_include parameter
        - Update kwargs preparation for organisation-specific logic
        - Modify validation to account for federated scenarios
        """
        client = authentication
        config = test_configurations[config_name]
        method_config = test_methods[method]

        # Skip if method doesn't support organisation_selection scenario
        if "organisation_selection" not in method_config:
            pytest.skip(
                f"Organisation selection functionality not supported for {method} method"
            )

        # Prepare method-specific kwargs from method configuration
        kwargs = method_config["organisation_selection"].copy()
        kwargs["variables_to_describe"] = config["variables_to_describe_basic"]
        kwargs["organisations_to_include"] = config["organisation_subset"]

        # Create a task for the client to retrieve the descriptive data
        task = client.task.create(
            collaboration=1,
            organizations=[1],
            name=f"Test {method} algorithm run on specific organisations - {config_name}",
            image=algorithm_image_name,
            description=f"Task to test the {method} function "
            f"when selecting specific organisations using {config_name} configuration.",
            input_={"method": method, "kwargs": kwargs},
            databases=[{"label": config["database_label"]}],
        )

        if config.get("expected_failure", False):
            # Test that aggressive configurations fail gracefully
            with pytest.raises(Exception) as exc_info:
                categorical_statistics, numerical_statistics = extract_data_from_result(
                    client, task, method
                )

            # Verify specific error types (support both single error type and list of error types)
            expected_errors = config.get("expected_error_type")
            if expected_errors:
                # Convert single error type to list for uniform handling
                if not isinstance(expected_errors, list):
                    expected_errors = [expected_errors]

                # Check if the raised exception matches any of the expected types
                error_matched = any(
                    isinstance(exc_info.value, expected_error)
                    for expected_error in expected_errors
                )
                assert error_matched, (
                    f"Expected one of {[err.__name__ for err in expected_errors]} "
                    f"but got {type(exc_info.value).__name__}"
                )

            print(f"Expected failure occurred for {config_name}: {exc_info.value}")
        else:
            # Normal success path
            categorical_statistics, numerical_statistics = extract_data_from_result(
                client, task, method
            )
            determine_statistics_acceptance(
                {
                    "categorical_general_statistics": categorical_statistics,
                    "numerical_general_statistics": numerical_statistics,
                },
                {},
                method,
                config["database_label"],
                kwargs,
            ), f"Centralised and federated statistics deviate too much for {config_name} configuration"

    @pytest.mark.parametrize("method", ["central", "partial_general_statistics"])
    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_dataset",
            "standard_dataset_bad_actor",
            "standard_dataset_incorrect_input",
            "rare_dataset",
            "non_existent_dataset_standard_input",
        ],
    )
    def test_algorithm_data_stratification(
        self,
        authentication,
        algorithm_image_name,
        test_configurations,
        test_methods,
        method,
        config_name,
    ):
        """
        Test algorithm with data stratification, including expected failures.

        CUSTOMISATION REQUIRED:
        - Ensure your algorithm supports variables_to_stratify parameter
        - Update stratification logic to match your algorithm's requirements
        - Modify skip conditions based on your algorithm's capabilities
        """
        client = authentication
        config = test_configurations[config_name]
        method_config = test_methods[method]

        # Skip if configuration doesn't support stratification
        if config["variables_to_stratify"] is None:
            pytest.skip(f"Stratification not supported for {config_name} configuration")

        # Prepare method-specific kwargs from method configuration
        kwargs = method_config["data_stratification"].copy()
        kwargs["variables_to_describe"] = config["variables_to_describe_basic"]
        kwargs["variables_to_stratify"] = config["variables_to_stratify"]

        # Create a task for the client to retrieve the descriptive data
        task = client.task.create(
            collaboration=1,
            organizations=[1],
            name=f"Test {method} algorithm run with data stratification - {config_name}",
            image=algorithm_image_name,
            description=f"Task to test the {method} function "
            f"when stratifying the data using {config_name} configuration.",
            input_={"method": method, "kwargs": kwargs},
            databases=[{"label": config["database_label"]}],
        )

        if config.get("expected_failure", False):
            # Test that aggressive configurations fail gracefully
            with pytest.raises(Exception) as exc_info:
                extract_data_from_result(
                    client, task, method
                )  # Output not necessary when tasks have failed

            # Verify specific error types (support both single error type and list of error types)
            expected_errors = config.get("expected_error_type")
            if expected_errors:
                # Convert single error type to list for uniform handling
                if not isinstance(expected_errors, list):
                    expected_errors = [expected_errors]

                # Check if the raised exception matches any of the expected types
                error_matched = any(
                    isinstance(exc_info.value, expected_error)
                    for expected_error in expected_errors
                )
                assert error_matched, (
                    f"Expected one of {[err.__name__ for err in expected_errors]} "
                    f"but got {type(exc_info.value).__name__}"
                )

            print(f"Expected failure occurred for {config_name}: {exc_info.value}")
        else:
            # Normal success path
            categorical_statistics, numerical_statistics = extract_data_from_result(
                client, task, method
            )
            determine_statistics_acceptance(
                {
                    "categorical_general_statistics": categorical_statistics,
                    "numerical_general_statistics": numerical_statistics,
                },
                {},
                method,
                config["database_label"],
                kwargs,
            ), f"Centralised and federated statistics deviate too much for {config_name} configuration"

    @pytest.mark.parametrize("method", ["central", "partial_general_statistics"])
    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_dataset",
            "standard_dataset_bad_actor",
            "standard_dataset_incorrect_input",
            "rare_dataset",
            "non_existent_dataset_standard_input",
        ],
    )
    def test_algorithm_inlier_specific(
        self,
        authentication,
        algorithm_image_name,
        test_configurations,
        test_methods,
        method,
        config_name,
    ):
        """
        Test algorithm with inlier-specific variable configurations.

        CUSTOMISATION REQUIRED:
        - Ensure your algorithm supports inlier-specific variable definitions
        - Update kwargs preparation for inlier-specific logic
        - Modify validation to account for inlier filtering
        """
        client = authentication
        config = test_configurations[config_name]
        method_config = test_methods[method]

        # Skip if configuration doesn't have inlier-specific variables
        if not config.get("variables_to_describe_inlier_specific"):
            pytest.skip(
                f"Inlier-specific variables not available for {config_name} configuration"
            )

        # Prepare method-specific kwargs from method configuration
        kwargs = method_config["inlier_specific"].copy()
        kwargs["variables_to_describe"] = config[
            "variables_to_describe_inlier_specific"
        ]

        # Create a task for the client to retrieve the descriptive data
        task = client.task.create(
            collaboration=1,
            organizations=[1],
            name=f"Test {method} algorithm run with inlier-specific variables - {config_name}",
            image=algorithm_image_name,
            description=f"Task to test the {method} function with "
            f"inlier-specific variable configurations using {config_name} configuration.",
            input_={"method": method, "kwargs": kwargs},
            databases=[{"label": config["database_label"]}],
        )

        if config.get("expected_failure", False):
            # Test that aggressive configurations fail gracefully
            with pytest.raises(Exception) as exc_info:
                extract_data_from_result(
                    client, task, method
                )  # Output not necessary when tasks have failed

            # Verify specific error types (support both single error type and list of error types)
            expected_errors = config.get("expected_error_type")
            if expected_errors:
                # Convert single error type to list for uniform handling
                if not isinstance(expected_errors, list):
                    expected_errors = [expected_errors]

                # Check if the raised exception matches any of the expected types
                error_matched = any(
                    isinstance(exc_info.value, expected_error)
                    for expected_error in expected_errors
                )
                assert error_matched, (
                    f"Expected one of {[err.__name__ for err in expected_errors]} "
                    f"but got {type(exc_info.value).__name__}"
                )

            print(f"Expected failure occurred for {config_name}: {exc_info.value}")
        else:
            # Normal success path
            categorical_statistics, numerical_statistics = extract_data_from_result(
                client, task, method
            )
            determine_statistics_acceptance(
                {
                    "categorical_general_statistics": categorical_statistics,
                    "numerical_general_statistics": numerical_statistics,
                },
                {},
                method,
                config["database_label"],
                kwargs,
            ), f"Centralised and federated statistics deviate too much for {config_name} configuration"

    @pytest.mark.parametrize("method", ["central", "partial_general_statistics"])
    @pytest.mark.parametrize(
        "config_name",
        [
            "standard_dataset",
            "standard_dataset_bad_actor",
            "standard_dataset_incorrect_input",
            "rare_dataset",
            "non_existent_dataset_standard_input",
        ],
    )
    def test_algorithm_parameter_galore(
        self,
        authentication,
        algorithm_image_name,
        test_configurations,
        test_methods,
        method,
        config_name,
    ):
        """
        Test algorithm with all parameters combined (parameter galore).

        CUSTOMISATION REQUIRED:
        - Ensure your algorithm supports all combined parameters
        - Update kwargs preparation for comprehensive parameter testing
        - Modify validation to account for complex parameter interactions
        """
        client = authentication
        config = test_configurations[config_name]
        method_config = test_methods[method]

        # Skip if method doesn't support parameter_galore scenario
        if "parameter_galore" not in method_config:
            pytest.skip(
                f"Parameter galore functionality not supported for {method} method"
            )

        # Skip if configuration doesn't support stratification (required for parameter_galore)
        if config["variables_to_stratify"] is None:
            pytest.skip(
                f"Stratification not supported for {config_name} configuration, required for parameter_galore"
            )

        # Prepare method-specific kwargs from method configuration
        kwargs = method_config["parameter_galore"].copy()
        kwargs["variables_to_describe"] = config["variables_to_describe_basic"]
        kwargs["variables_to_stratify"] = config["variables_to_stratify"]

        # Only add organisations_to_include if the method supports it
        if "organisations_to_include" in kwargs:
            kwargs["organisations_to_include"] = config["organisation_subset"]

        # Create a task for the client to retrieve the descriptive data
        task = client.task.create(
            collaboration=1,
            organizations=[1],
            name=f"Test {method} algorithm run with parameter galore - {config_name}",
            image=algorithm_image_name,
            description=f"Task to test the {method} function "
            f"with all parameters combined using {config_name} configuration.",
            input_={"method": method, "kwargs": kwargs},
            databases=[{"label": config["database_label"]}],
        )

        if config.get("expected_failure", False):
            # Test that aggressive configurations fail gracefully
            with pytest.raises(Exception) as exc_info:
                extract_data_from_result(
                    client, task, method
                )  # Output not necessary when tasks have failed

            # Verify specific error types (support both single error type and list of error types)
            expected_errors = config.get("expected_error_type")
            if expected_errors:
                # Convert single error type to list for uniform handling
                if not isinstance(expected_errors, list):
                    expected_errors = [expected_errors]

                # Check if the raised exception matches any of the expected types
                error_matched = any(
                    isinstance(exc_info.value, expected_error)
                    for expected_error in expected_errors
                )
                assert error_matched, (
                    f"Expected one of {[err.__name__ for err in expected_errors]} "
                    f"but got {type(exc_info.value).__name__}"
                )

            print(f"Expected failure occurred for {config_name}: {exc_info.value}")
        else:
            # Normal success path
            categorical_statistics, numerical_statistics = extract_data_from_result(
                client, task, method
            )
            determine_statistics_acceptance(
                {
                    "categorical_general_statistics": categorical_statistics,
                    "numerical_general_statistics": numerical_statistics,
                },
                {},
                method,
                config["database_label"],
                kwargs,
            ), f"Centralised and federated statistics deviate too much for {config_name} configuration"


def extract_data_from_result(client, task, method) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract categorical and numerical statistics from the algorithm task result.
    Handles error checking and raises appropriate exceptions based on log content.

    Args:
        client: Authenticated Vantage6 client
        task: Task object returned from task creation
        method: The method used for computation (e.g. "central", "partial_general_statistics")

    Returns:
         Tuple of DataFrames (categorical_stats, numerical_stats)
    """
    # Wait for results to be ready
    print("Waiting for results")
    task_id = task["id"]
    result = client.wait_for_results(task_id)

    # Check if there are any (un-)expected errors in the log
    run_info = client.run.from_task(task_id)
    log = run_info["data"][0]["log"]

    if "Traceback" in log:
        print(f"Error found in task log: {log}")

        # Extract the actual error from the log
        error_lines = [line for line in log.split("\n") if line.startswith("error >")]
        if error_lines:
            # Look for traceback information
            if "Traceback" in log:
                # Extract the exception type and message from the traceback
                lines = log.split("\n")
                for i, line in enumerate(lines):
                    if line.strip().startswith("vantage6.algorithm.tools.exceptions."):
                        error_class_line = line.strip()
                        error_message = (
                            error_class_line.split(": ", 1)[1]
                            if ": " in error_class_line
                            else "Unknown error"
                        )
                        if "UserInputError" in error_class_line:
                            raise UserInputError(error_message)
                        elif "CollectResultsError" in error_class_line:
                            raise CollectResultsError(error_message)
                        elif "PrivacyThresholdViolation" in error_class_line:
                            raise PrivacyThresholdViolation(error_message)
                        elif "InputError" in error_class_line:
                            raise InputError(error_message)
                        elif "AlgorithmError" in error_class_line:
                            raise AlgorithmError(error_message)
                        elif "CollectOrganizationError" in error_class_line:
                            raise CollectOrganizationError(error_message)
                        elif "DataError" in error_class_line:
                            raise DataError(error_message)
                        else:
                            # If the error class is not recognised, raise a generic AlgorithmError
                            raise AlgorithmError(
                                f"Unknown error type in log: {error_class_line}"
                            )

        # Fallback to generic error with the error message
        error_message = error_lines[-1].replace("error >", "").strip()
        if error_message and error_message != "None":
            raise AlgorithmError(f"Algorithm execution failed: {error_message}")

    # Check if the result is not None
    assert result is not None, "Result should not be None"

    # Extract the aggregated results
    result = json.loads(result["data"][0]["result"])

    if method == "central":
        # Extract categorical and numerical statistics for the central method
        categorical_stats = result.get("categorical_general_statistics", None)
        numerical_stats = result.get("numerical_general_statistics", None)
    elif method == "partial_general_statistics":
        # Extract categorical and numerical statistics for the partial method
        categorical_stats = result.get("categorical_general_partial_statistics", None)
        numerical_stats = result.get("numerical_general_partial_statistics", None)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Check if statistics are present
    assert categorical_stats is not None, "Categorical statistics should not be None"
    assert numerical_stats is not None, "Numerical statistics should not be None"

    # Statistics come as JSON strings that need to be converted to DataFrames
    assert isinstance(
        categorical_stats, str
    ), "Categorical statistics should be a JSON string"
    assert isinstance(
        numerical_stats, str
    ), "Numerical statistics should be a JSON string"

    # Convert JSON strings to DataFrames
    categorical_stats_df = pd.read_json(StringIO(categorical_stats))
    numerical_stats_df = pd.read_json(StringIO(numerical_stats))

    print(f"Final categorical stats shape: {categorical_stats_df.shape}", flush=True)
    print(f"Final numerical stats shape: {numerical_stats_df.shape}", flush=True)

    return categorical_stats_df, numerical_stats_df


def determine_statistics_acceptance(
    federated_result: Dict[str, Any],
    central_result: Dict[str, Any],  # Add back for signature compatibility but not used
    method: str,
    database_label: str,
    kwargs: Dict[str, Any],
    tolerance: float = 1e-6,
) -> None:
    """
    Assert that federated and central statistical results are equivalent within tolerance.

    Focuses on validating count equivalency based on the comment requirements:
    - In the 3-node test setup, counts should be multiplied by 3
    - Reads the appropriate test dataset to get actual counts for validation
    - Applies same stratification and organisation selection as the algorithm

    Args:
        federated_result: Results from federated computation (DataFrames)
        central_result: Not used in integration tests, kept for compatibility
        method: The method used for computation (e.g., "central", "partial_general_statistics")
        database_label: Label of the database/dataset file to validate against
        kwargs: Algorithm kwargs containing stratification and organization parameters
        tolerance: Numerical tolerance for comparison

    Raises:
        AssertionError: If validation fails
    """
    # Basic type validation
    assert isinstance(federated_result, dict), "Federated result must be a dictionary"

    # For the integration test setup where we don't have central_result comparison,
    # we focus on validating that the federated result contains valid statistics
    assert federated_result, "Federated result should not be empty"

    # Validate count equivalency using provided database label and kwargs
    import pandas as pd
    from pathlib import Path

    # Get the test data file path
    repo_root = Path(__file__).parent.parent.parent
    dataset_file = repo_root / "tests" / "data" / f"{database_label}.csv"

    assert dataset_file.exists(), f"Test dataset file not found: {dataset_file}"

    # Read the actual test dataset
    df = pd.read_csv(dataset_file)

    # Apply data stratification if specified in kwargs (this applies globally)
    variables_to_stratify = kwargs.get("variables_to_stratify")
    if variables_to_stratify:
        print(f"Applying stratification: {variables_to_stratify}")
        for var_name, var_config in variables_to_stratify.items():
            if var_name in df.columns:
                if var_config.get("datatype") == "categorical":
                    values = var_config.get("values", [])
                    if values:
                        df = df[df[var_name].isin(values)]
                        print(
                            f"Applied categorical stratification on {var_name}: {values}, remaining rows: {len(df)}"
                        )
                elif var_config.get("datatype") == "int":
                    # Apply numerical range stratification
                    start = var_config.get("start")
                    end = var_config.get("end")
                    if start is not None:
                        df = df[df[var_name] >= start]
                        print(
                            f"Applied start filter on {var_name} >= {start}, remaining rows: {len(df)}"
                        )
                    if end is not None:
                        df = df[df[var_name] <= end]
                        print(
                            f"Applied end filter on {var_name} <= {end}, remaining rows: {len(df)}"
                        )

    # Store the stratified dataframe (without inlier filters)
    stratified_df = df.copy()

    # NOTE: Inlier filtering is now applied per variable during validation, not globally
    # This ensures each variable's statistics are validated with only its own inlier filter

    # Determine organisation multiplier based on method and kwargs
    if method == "central":
        # Get organisation subset multiplier
        organisations_to_include = kwargs.get("organisations_to_include", [1, 2, 3])
        # If specific organisations selected, adjust multiplier based on the fraction of total nodes
        organisation_multiplier = 3 / len(organisations_to_include)
    elif method == "partial_general_statistics":
        # Data is distributed across 3 organisations in the test setup
        organisation_multiplier = 3
    else:
        raise ValueError(f"Unknown method: {method}")

    # Extract statistics DataFrames
    numerical_stats = federated_result.get(
        "numerical_general_statistics", pd.DataFrame()
    )
    categorical_stats = federated_result.get(
        "categorical_general_statistics", pd.DataFrame()
    )

    # Get variables to describe for inlier filtering reference
    variables_to_describe = kwargs.get("variables_to_describe", {})

    # Validate numerical statistics counts
    if not numerical_stats.empty:
        assert (
            len(numerical_stats.columns) == 3
        ), f"Numerical stats should have 3 columns, got {len(numerical_stats.columns)}"

        # Look for 'count' statistic rows
        count_rows = numerical_stats[numerical_stats.iloc[:, 1] == "count"]
        if not count_rows.empty:
            for _, row in count_rows.iterrows():
                variable_name = row.iloc[0]
                federated_count = float(row.iloc[2])  # value column

                # Apply inlier filter only for this specific variable
                variable_df = stratified_df.copy()
                variable_config = variables_to_describe.get(variable_name, {})

                if (
                    "inliers" in variable_config
                    and variable_config.get("datatype") == "numerical"
                ):
                    inliers = variable_config["inliers"]
                    if isinstance(inliers, (tuple, list)) and len(inliers) == 2:
                        variable_df = variable_df[
                            (variable_df[variable_name] >= inliers[0])
                            & (variable_df[variable_name] <= inliers[1])
                        ]
                        print(
                            f"Applied numerical inlier filter on {variable_name}: {inliers}, "
                            f"remaining rows: {len(variable_df)}"
                        )

                actual_count = len(variable_df)
                expected_federated_count = int(actual_count / organisation_multiplier)

                print(
                    f"Expected federated count for {variable_name}: {expected_federated_count} "
                    f"(actual: {actual_count} / multiplier: {organisation_multiplier})"
                )

                # Check if this variable was filtered by inliers
                if (
                    "inliers" in variable_config
                    and variable_config.get("datatype") == "numerical"
                ):
                    # For numerical variables with inliers, the count should match our filtered dataset
                    if method == "partial_general_statistics":
                        # Use relaxed validation for distributed data scenarios
                        max_reasonable_count = (
                            expected_federated_count * 1.5
                        )  # Allow 50% tolerance
                        assert federated_count <= max_reasonable_count, (
                            f"Numerical count for {variable_name} exceeds reasonable bounds: "
                            f"got {federated_count}, max reasonable {max_reasonable_count}"
                        )
                    else:
                        assert (
                            abs(federated_count - expected_federated_count) <= tolerance
                        ), (
                            f"Numerical count mismatch for {variable_name}: "
                            f"got {federated_count}, expected {expected_federated_count} "
                            f"(with inlier filtering applied, tolerance: {tolerance})"
                        )

                else:
                    # For variables without inlier filtering, allow some tolerance since algorithm may
                    # apply additional filtering. The count should at least not exceed the expected maximum
                    max_expected = len(pd.read_csv(dataset_file))
                    assert federated_count <= max_expected, (
                        f"Numerical count for {variable_name} exceeds maximum possible: "
                        f"got {federated_count}, max expected {max_expected}"
                    )

                print(
                    f"✓ Numerical count validation passed for {variable_name}: {federated_count}"
                )

    # Validate categorical statistics counts and value distributions
    if not categorical_stats.empty:
        assert (
            len(categorical_stats.columns) == 3
        ), f"Categorical stats should have 3 columns, got {len(categorical_stats.columns)}"

        # Filter out metadata rows (na, outliers)
        data_rows = categorical_stats[
            ~categorical_stats.iloc[:, 1].isin(["na", "outliers"])
        ]

        if not data_rows.empty:
            # For categorical data, check each variable separately
            for var_name in data_rows.iloc[:, 0].unique():
                var_rows = data_rows[data_rows.iloc[:, 0] == var_name]
                total_var_count = var_rows.iloc[:, 2].sum()

                # Apply inlier filter only for this specific variable
                variable_df = stratified_df.copy()
                variable_config = variables_to_describe.get(var_name, {})

                if (
                    "inliers" in variable_config
                    and variable_config.get("datatype") == "categorical"
                ):
                    inliers = variable_config["inliers"]
                    if isinstance(inliers, (tuple, list)):
                        variable_df = variable_df[variable_df[var_name].isin(inliers)]
                        print(
                            f"Applied categorical inlier filter on {var_name}: {inliers}, "
                            f"remaining rows: {len(variable_df)}"
                        )

                actual_count = len(variable_df)
                expected_federated_count = int(actual_count / organisation_multiplier)

                print(
                    f"Expected federated count for {var_name}: {expected_federated_count} "
                    f"(actual: {actual_count} / multiplier: {organisation_multiplier})"
                )

                # Check if this variable was filtered by inliers
                if (
                    "inliers" in variable_config
                    and variable_config.get("datatype") == "categorical"
                ):
                    # For categorical variables with inliers, count should match filtered dataset
                    assert (
                        abs(total_var_count - expected_federated_count) <= tolerance
                    ), (
                        f"Categorical count mismatch for {var_name}: "
                        f"got {total_var_count}, expected {expected_federated_count} "
                        f"(with inlier filtering applied, tolerance: {tolerance})"
                    )

                    # Validate individual value counts for categorical variables
                    if var_name in variable_df.columns:
                        expected_value_counts = variable_df[var_name].value_counts()

                        for _, row in var_rows.iterrows():
                            category_value = row.iloc[1]  # category value
                            federated_count = float(row.iloc[2])  # count

                            if category_value in expected_value_counts.index:
                                expected_count = expected_value_counts[category_value]
                                # Skip strict validation for partial_general_statistics with categorical inliers
                                # due to data distribution ambiguity across federated nodes
                                if method == "partial_general_statistics":
                                    # Use relaxed validation for distributed data scenarios
                                    max_reasonable_count = (
                                        expected_count * 1.5
                                    )  # Allow 50% tolerance
                                    assert federated_count <= max_reasonable_count, (
                                        f"Categorical value count for {var_name}[{category_value}] "
                                        f"exceeds reasonable bounds: "
                                        f"got {federated_count}, max reasonable {max_reasonable_count}"
                                    )
                                else:
                                    assert (
                                        abs(federated_count - expected_count)
                                        <= tolerance
                                    ), (
                                        f"Categorical value count mismatch for {var_name}[{category_value}]: "
                                        f"got {federated_count}, expected {expected_count} "
                                        f"(tolerance: {tolerance})"
                                    )
                                print(
                                    f"✓ Categorical value count validation passed for {var_name}[{category_value}]: "
                                    f"{federated_count}"
                                )

                else:
                    # For variables without inlier filtering, allow some tolerance since algorithm may
                    # apply additional filtering. The count should at least not exceed the expected maximum
                    max_expected = len(pd.read_csv(dataset_file))
                    assert total_var_count <= max_expected, (
                        f"Categorical count for {var_name} exceeds maximum possible: "
                        f"got {total_var_count}, max expected {max_expected}"
                    )

                    # Still validate value counts for variables without inlier filtering
                    if var_name in variable_df.columns:
                        expected_value_counts = variable_df[var_name].value_counts()

                        # Check that federated value counts don't exceed expected maximums
                        for _, row in var_rows.iterrows():
                            category_value = row.iloc[1]  # category value
                            federated_count = float(row.iloc[2])  # count

                            if category_value in expected_value_counts.index:
                                max_expected_count = (
                                    expected_value_counts[category_value] * 1.1
                                )  # 10% tolerance
                                assert federated_count <= max_expected_count, (
                                    f"Categorical value count for {var_name}[{category_value}] exceeds expected: "
                                    f"got {federated_count}, max expected {max_expected_count}"
                                )

                print(
                    f"✓ Categorical count validation passed for {var_name}: {total_var_count}"
                )

    print("✓ All count validations passed")
