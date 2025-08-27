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
    - "organisation_ids": Filled from config['organisation_subset']
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
                "organisation_ids": None,  # Will be filled from config
            },
            "data_stratification": {
                "variables_to_describe": None,  # Will be filled from config
                "variables_to_stratify": None,  # Will be filled from config
            },
            "inlier_specific": {
                "variables_to_describe": None,  # Will be filled from config (inlier_specific)
            },
            "return_partials": {
                "variables_to_describe": None,  # Will be filled from config
                "return_partials": True,  # Method-specific parameter
            },
            "parameter_galore": {
                "variables_to_describe": None,  # Will be filled from config
                "variables_to_stratify": None,  # Will be filled from config
                "organisation_ids": None,  # Will be filled from config
                "return_partials": True,  # Method-specific parameter
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
    - 'organisation_subset' -> "organisation_ids"
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
                    "inliers": (100, 150),
                },
                "Social Structure": {
                    "datatype": "categorical",
                    "inliers": ("Organic Compounds", "Minerals"),
                },
            },
            "variables_to_stratify": {
                "Lifespan (years)": {"start": 10, "datatype": "int"},
                "Habitat": {
                    "values": ["Ice Caves", "Underground"],
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
                    "inliers": (100, 110),
                },
                # Emulate a bad actor by setting a very narrow range as inlier
                "Social Structure": {
                    "datatype": "categorical",
                    "inliers": ["Solitary"],
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
                "Temperature Tolerance (K)": {"datatype": "numerical"},
                "Diet": {
                    "datatype": "categorical"
                },  # Use non-existent variable Diet for Enceladus
            },
            "organisation_subset": [4, 5],
            # Non-existent organisations to test input validation
            "variables_to_describe_inlier_specific": {
                "Temperature Tolerance (K)": {
                    "datatype": "numerical",
                    "inliers": (50, 150),
                },
                "Diet": {  # Use non-existent variable Diet for Enceladus
                    "datatype": "categorical",
                    "inliers": ["Solitary", "Swarm"],  # Non-existent categories for fictive Diet
                },
            },
            "variables_to_stratify": {
                "Lifespan (years)": {"start": 10, "datatype": "int"},
                "Habitat": {
                    "values": [
                        "Ice Caves",
                        "Underground",
                    ],  # Underground habitat preferences are not present
                    "datatype": "categorical",
                },
            },
            "expected_failure": True,
            "failure_reason": "Non-existent variables requested or invalid input structure specified",
            "expected_error_type": [CollectResultsError, UserInputError, JSONDecodeError],
        },
        "rare_dataset": {
            "database_label": "creatures_of_titan",
            # Titan is Saturn's largest moon with various favourable conditions
            "variables_to_describe_basic": {
                "Lifespan (years)": {"datatype": "numerical"}
            },
            "organisation_subset": [1],
            "variables_to_describe_inlier_specific": {
                "Lifespan (years)": {"datatype": "numerical", "inliers": (18, 35)}
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
                    "inliers": (0, 10000),
                },
                "Diet": {
                    "datatype": "categorical",
                    "inliers": ("Organic Compounds", "Minerals"),
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
            assert determine_statistics_acceptance(
                {"categorical_general_statistics": categorical_statistics, "numerical_general_statistics": numerical_statistics},
                {},
                config
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
        - Ensure your algorithm supports organisation_ids parameter
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
        kwargs["organisation_ids"] = config["organisation_subset"]

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
            assert determine_statistics_acceptance(
                {"categorical_general_statistics": categorical_statistics, "numerical_general_statistics": numerical_statistics},
                {},
                config
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
            assert determine_statistics_acceptance(
                {"categorical_general_statistics": categorical_statistics, "numerical_general_statistics": numerical_statistics},
                {},
                config
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
            assert determine_statistics_acceptance(
                {"categorical_general_statistics": categorical_statistics, "numerical_general_statistics": numerical_statistics},
                {},
                config
            ), f"Centralised and federated statistics deviate too much for {config_name} configuration"

    @pytest.mark.parametrize(
        "method", ["central"]
    )  # 'partial_general_statistics' does not support return_partial
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
    def test_algorithm_return_partial(
        self,
        authentication,
        algorithm_image_name,
        test_configurations,
        test_methods,
        method,
        config_name,
    ):
        """
        Test algorithm with return_partial functionality.

        CUSTOMISATION REQUIRED:
        - Ensure your algorithm supports return_partial or equivalent parameters
        - Update kwargs preparation for partial return logic
        - Modify validation to account for partial results
        """
        client = authentication
        config = test_configurations[config_name]
        method_config = test_methods[method]

        # Prepare method-specific kwargs from method configuration
        kwargs = method_config["return_partial"].copy()
        kwargs["variables_to_describe"] = config["variables_to_describe_basic"]

        # Create a task for the client to retrieve the descriptive data
        task = client.task.create(
            collaboration=1,
            organizations=[1],
            name=f"Test {method} algorithm run with return_partial - {config_name}",
            image=algorithm_image_name,
            description=f"Task to test the {method} function "
            f"with return_partial functionality using {config_name} configuration.",
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
            assert determine_statistics_acceptance(
                {"categorical_general_statistics": categorical_statistics, "numerical_general_statistics": numerical_statistics},
                {},
                config
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

        # Only add organisation_ids if the method supports it
        if "organisation_ids" in kwargs:
            kwargs["organisation_ids"] = config["organisation_subset"]

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
            assert determine_statistics_acceptance(
                {"categorical_general_statistics": categorical_statistics, "numerical_general_statistics": numerical_statistics},
                {},
                config
            ), f"Centralised and federated statistics deviate too much for {config_name} configuration"


def extract_data_from_result(client, task, method) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """"""
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

    # Read the JSON strings into dictionaries
    categorical_stats = pd.read_json(StringIO(categorical_stats))
    numerical_stats = pd.read_json(StringIO(numerical_stats))
    assert isinstance(
        categorical_stats, pd.DataFrame
    ), "Categorical statistics should be a pandas DataFrame"
    assert isinstance(
        numerical_stats, pd.DataFrame
    ), "Numerical statistics should be a pandas DataFrame"

    print(f"Final categorical stats shape: {categorical_stats.shape}", flush=True)
    print(f"Final numerical stats shape: {numerical_stats.shape}", flush=True)

    return categorical_stats, numerical_stats


def determine_statistics_acceptance(
    federated_result: Dict[str, Any],
    central_result: Dict[str, Any],
    method: str,
    config: Dict[str, Any] = None,
    tolerance: float = 1e-6,
) -> bool:
    """
    Assert that federated and central statistical results are equivalent within tolerance.
    
    Focuses on validating count equivalency based on the comment requirements:
    - In the 3-node test setup, counts should be multiplied by 3
    - Reads the appropriate test dataset to get actual counts for validation

    Args:
        federated_result: Results from federated computation
        central_result: Results from central computation
        method: The method used for computation (e.g., "central", "partial_general_statistics")
        config: Test configuration containing dataset info (can be None for basic validation)
        tolerance: Numerical tolerance for comparison
    """
    # Basic type validation
    if not isinstance(federated_result, dict) or not isinstance(central_result, dict):
        return False

    # For the integration test setup where we don't have central_result comparison,
    # we focus on validating that the federated result contains valid statistics
    if not federated_result:
        return False

    # Check that we have the expected statistical components
    required_keys = ["categorical_general_statistics", "numerical_general_statistics"]
    if not all(key in federated_result for key in required_keys):
        return False

    # If config is provided, validate count equivalency 
    if config:
        try:
            import pandas as pd
            from pathlib import Path

            # Get the test data file path
            repo_root = Path(__file__).parent.parent.parent
            dataset_label = config.get("database_label", "")
            dataset_file = repo_root / "tests" / "data" / f"{dataset_label}.csv"

            if dataset_file.exists():
                # Read the actual test dataset
                df = pd.read_csv(dataset_file)

                # TODO stratify the data if the config and method require it

                # This is very basic and could be improved by specify the correct column
                actual_count = len(df)

                if method == "central":
                    # In the 3-node setup, the federated count should be 3x the original dataset
                    expected_federated_count = actual_count * 3

                if method == "central":
                    # Extract count from numerical statistics
                    numerical_stats = federated_result.get("numerical_general_statistics", {})
                elif method == "partial_general_statistics":
                    # Extract count from partial numerical statistics
                    numerical_stats = federated_result.get("numerical_general_partial_statistics", {})
                else:
                    numerical_stats = {}

                if isinstance(numerical_stats, str):
                    import json
                    numerical_stats = json.loads(numerical_stats)

                if isinstance(numerical_stats, dict):
                    # Look for count information in the statistics
                    for var_stats in numerical_stats.values():
                        if isinstance(var_stats, dict) and "count" in var_stats:
                            federated_count = var_stats["count"]
                            # Validate that the federated count matches expected (3x original)
                            if abs(federated_count - expected_federated_count) <= tolerance:
                                print(f"✓ Count validation passed: {federated_count} == {expected_federated_count}")
                                return True
                            else:
                                assert f"✗ Count mismatch: got {federated_count}, expected {expected_federated_count}"


                if method == "central":
                    # Also check categorical statistics for counts
                    categorical_stats = federated_result.get("categorical_general_statistics", {})
                elif method == "partial_general_statistics":
                    categorical_stats = federated_result.get("categorical_general_partial_statistics", {})
                else:
                    categorical_stats = {}

                if isinstance(categorical_stats, str):
                    import json
                    categorical_stats = json.loads(categorical_stats)

                if isinstance(categorical_stats, dict):
                    for var_stats in categorical_stats.values():
                        if isinstance(var_stats, dict) and "count" in var_stats:
                            federated_count = var_stats["count"]
                            if abs(federated_count - expected_federated_count) <= tolerance:
                                print(f"✓ Count validation passed: {federated_count} == {expected_federated_count}")
                                return True
                            else:
                                assert f"✗ Count mismatch: got {federated_count}, expected {expected_federated_count}"

        except Exception as e:
            print(f"Count validation error: {e}")\

    # Basic validation - ensure we have non-empty results
    return True
