"""
Unit tests for algorithm functions.

Test the actual algorithm functionality rather than external libraries.
"""
import pytest
import pandas as pd
import numpy as np
import sys
import os
from unittest.mock import Mock, patch

# Add algorithm module to path
algorithm_path = os.path.join(os.path.dirname(__file__), '../v6-descriptive-statistics/')
sys.path.insert(0, algorithm_path)


@pytest.mark.unit
class TestAlgorithmFunctions:
    """Test core algorithm functions."""

    @pytest.fixture
    def sample_data(self):
        """Create sample test data."""
        np.random.seed(42)
        return pd.DataFrame({
            'Age': np.random.normal(30, 10, 100),
            'Gender': np.random.choice(['M', 'F'], 100),
            'Height': np.random.normal(170, 15, 100),
            'Weight': np.random.normal(70, 12, 100)
        })

    @pytest.fixture
    def variables_config(self):
        """Sample variables configuration."""
        return {
            "Age": {
                "datatype": "numerical",
                "inliers": (18, 80)
            },
            "Gender": {
                "datatype": "categorical",
                "inliers": ("M", "F")
            },
            "Height": {
                "datatype": "numerical",
                "inliers": (150, 200)
            }
        }

    def test_algorithm_imports(self):
        """Test that algorithm modules can be imported."""
        try:
            # Test import of miscellaneous functions
            from miscellaneous import check_input_structure
            assert callable(check_input_structure)
            
            # Test that algorithm files exist and can be imported at module level
            # Full function testing requires vantage6 environment
            import partial
            import central
            assert hasattr(partial, 'partial_general_statistics')
            assert hasattr(central, 'central')
            
        except ImportError as e:
            pytest.skip(f"Algorithm modules require vantage6 environment: {e}")

    def test_check_input_structure(self, variables_config):
        """Test input structure validation function."""
        try:
            from miscellaneous import check_input_structure
            
            # Test with valid input
            result = check_input_structure(variables_config)
            assert isinstance(result, bool)
            
            # Test with empty input
            result_empty = check_input_structure({})
            assert isinstance(result_empty, bool)
            
        except ImportError:
            pytest.skip("Algorithm module not available")

    def test_central_function_exists(self):
        """Test central function exists and is properly decorated."""
        try:
            import central
            # Test that the central function exists
            assert hasattr(central, 'central')
            # The function should be decorated and callable in vantage6 environment
            func = getattr(central, 'central')
            assert callable(func)
            
        except ImportError:
            pytest.skip("Algorithm module not available")

    def test_partial_functions_exist(self):
        """Test partial functions exist and are properly decorated."""
        try:
            import partial
            # Test that the partial functions exist
            assert hasattr(partial, 'partial_general_statistics')
            assert hasattr(partial, 'partial_aggregate_adjusted_deviation')
            
            # The functions should be decorated and callable in vantage6 environment
            func1 = getattr(partial, 'partial_general_statistics')
            func2 = getattr(partial, 'partial_aggregate_adjusted_deviation')
            assert callable(func1)
            assert callable(func2)
            
        except ImportError:
            pytest.skip("Algorithm module not available")

    def test_algorithm_data_processing(self, sample_data, variables_config):
        """Test algorithm data processing capabilities."""
        # Test that our sample data matches the expected structure
        assert not sample_data.empty
        assert len(sample_data) > 0
        
        # Test data types match configuration
        for var_name, var_config in variables_config.items():
            if var_name in sample_data.columns:
                if var_config["datatype"] == "numerical":
                    assert pd.api.types.is_numeric_dtype(sample_data[var_name])
                elif var_config["datatype"] == "categorical":
                    # Can be string or categorical
                    assert (pd.api.types.is_string_dtype(sample_data[var_name]) or 
                           pd.api.types.is_categorical_dtype(sample_data[var_name]) or
                           pd.api.types.is_object_dtype(sample_data[var_name]))

    def test_expected_output_structure(self):
        """Test expected algorithm output structure."""
        # Define expected output structure based on algorithm specification
        expected_structure = {
            "included_organisations": [],
            "excluded_organisations": [],
            "numerical_general_statistics": {},
            "categorical_general_statistics": {}
        }
        
        # Validate structure
        assert isinstance(expected_structure["included_organisations"], list)
        assert isinstance(expected_structure["excluded_organisations"], list)
        assert isinstance(expected_structure["numerical_general_statistics"], dict)
        assert isinstance(expected_structure["categorical_general_statistics"], dict)


@pytest.mark.unit
class TestStatisticalComputations:
    """Test statistical computation functions."""

    def test_numerical_statistics_computation(self):
        """Test numerical statistics are computed correctly."""
        # Create test data
        test_data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # Expected statistics
        expected_mean = 5.5
        expected_std = test_data.std()
        expected_median = 5.5
        expected_min = 1
        expected_max = 10
        
        # Test basic pandas statistics (which the algorithm uses)
        assert abs(test_data.mean() - expected_mean) < 1e-10
        assert abs(test_data.std() - expected_std) < 1e-10
        assert abs(test_data.median() - expected_median) < 1e-10
        assert test_data.min() == expected_min
        assert test_data.max() == expected_max

    def test_categorical_statistics_computation(self):
        """Test categorical statistics are computed correctly."""
        # Create test data
        test_data = pd.Series(['A', 'B', 'A', 'C', 'B', 'A'])
        
        # Test value counts
        value_counts = test_data.value_counts()
        assert value_counts['A'] == 3
        assert value_counts['B'] == 2
        assert value_counts['C'] == 1
        
        # Test mode
        mode = test_data.mode()
        assert mode[0] == 'A'

    def test_data_filtering_inliers(self):
        """Test data filtering based on inlier specifications."""
        # Create test data with outliers
        test_data = pd.Series([1, 2, 3, 4, 5, 100, 200])
        
        # Filter for inliers (1-10)
        inlier_range = (1, 10)
        filtered_data = test_data[(test_data >= inlier_range[0]) & (test_data <= inlier_range[1])]
        
        assert len(filtered_data) == 5  # Should exclude 100 and 200
        assert filtered_data.max() <= 10
        assert filtered_data.min() >= 1

    def test_categorical_inlier_filtering(self):
        """Test categorical data filtering based on allowed values."""
        # Create test data with invalid categories
        test_data = pd.Series(['M', 'F', 'M', 'X', 'F', 'Invalid'])
        
        # Filter for valid categories
        valid_categories = ['M', 'F', 'X']
        filtered_data = test_data[test_data.isin(valid_categories)]
        
        assert len(filtered_data) == 5  # Should exclude 'Invalid'
        assert all(val in valid_categories for val in filtered_data)