"""
Test configuration and fixtures for v6-descriptive-statistics testing framework.
"""
import pytest
import pandas as pd
from pathlib import Path
import os
import tempfile
from unittest.mock import Mock, MagicMock
from vantage6.algorithm.tools.mock_client import MockAlgorithmClient

# Test data location
TEST_DATA_DIR = Path(__file__).parent.parent / "test"

@pytest.fixture
def test_data_one():
    """Load test data from first organization."""
    return pd.read_csv(TEST_DATA_DIR / "test_data_one.csv")

@pytest.fixture
def test_data_two():
    """Load test data from second organization."""
    return pd.read_csv(TEST_DATA_DIR / "test_data_two.csv")

@pytest.fixture
def mock_algorithm_client():
    """Create mock algorithm client for testing."""
    client = MockAlgorithmClient(
        datasets=[
            [{
                "database": TEST_DATA_DIR / "test_data_one.csv",
                "db_type": "csv",
                "input_data": {}
            }],
            [{
                "database": TEST_DATA_DIR / "test_data_two.csv",
                "db_type": "csv",
                "input_data": {}
            }]
        ],
        module="v6-descriptive-statistics"
    )
    return client

@pytest.fixture
def sample_variables_to_describe():
    """Standard test variables configuration."""
    return {
        "Gender": {
            "datatype": "categorical",
            "inliers": ("M", "F", "X")
        },
        "Age": {
            "datatype": "numerical",
            "inliers": (15, 39)
        }
    }

@pytest.fixture
def edge_case_variables():
    """Edge case test variables configuration."""
    return {
        "Gender": {
            "datatype": "categorical",
            "inliers": ("M", "F")
        },
        "Age": {
            "datatype": "numerical", 
            "inliers": (20, 35)
        },
        "Height(in)": {
            "datatype": "numerical",
            "inliers": (60, 80)
        }
    }

@pytest.fixture
def mock_client():
    """Create a mock Vantage6 AlgorithmClient."""
    client = Mock()
    client.organization.list.return_value = [
        {"id": 1, "name": "org1"},
        {"id": 2, "name": "org2"}
    ]
    client.task.create.return_value = {"id": "test_task_id"}
    client.wait_for_results.return_value = [
        {"organization_id": 1, "result": {"test": "data1"}},
        {"organization_id": 2, "result": {"test": "data2"}}
    ]
    return client

@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)

@pytest.fixture(scope="session")
def docker_test_setup():
    """Setup for Docker-related tests."""
    # Check if Docker is available
    docker_available = os.system("docker --version > /dev/null 2>&1") == 0
    if not docker_available:
        pytest.skip("Docker not available for testing")
    return {"docker_available": True}

@pytest.fixture
def small_dataset():
    """Create a very small dataset for edge case testing."""
    return pd.DataFrame({
        "Name": ["Alice", "Bob"],
        "Gender": ["F", "M"], 
        "Age": [25, 30],
        "Height(in)": [65, 70],
        "Weight(lbs)": [120, 150]
    })

@pytest.fixture
def heterogeneous_datasets():
    """Create datasets with very different distributions for testing."""
    dataset1 = pd.DataFrame({
        "Name": [f"Person{i}" for i in range(20)],
        "Gender": ["F"] * 15 + ["M"] * 5,
        "Age": [25 + i for i in range(20)],
        "Height(in)": [65 + i*0.5 for i in range(20)],
        "Weight(lbs)": [120 + i*2 for i in range(20)]
    })
    
    dataset2 = pd.DataFrame({
        "Name": [f"Individual{i}" for i in range(15)],
        "Gender": ["M"] * 12 + ["F"] * 3,
        "Age": [60 + i for i in range(15)],
        "Height(in)": [70 + i*0.3 for i in range(15)],
        "Weight(lbs)": [180 + i*3 for i in range(15)]
    })
    
    return [dataset1, dataset2]

@pytest.fixture
def missing_data_dataset():
    """Create dataset with missing values for testing."""
    data = pd.DataFrame({
        "Name": ["Alice", "Bob", "Charlie", "Diana"],
        "Gender": ["F", None, "M", "F"],
        "Age": [25, 30, None, 35],
        "Height(in)": [65, None, 70, 68],
        "Weight(lbs)": [120, 150, 165, None]
    })
    return data

# Test environment setup
def pytest_configure(config):
    """Configure pytest environment."""
    # Set environment variables for testing
    os.environ["TESTING"] = "true"
    os.environ["V6_ALGORITHM_ENV"] = "test"

def pytest_unconfigure(config):
    """Clean up after testing."""
    # Clean up environment variables
    if "TESTING" in os.environ:
        del os.environ["TESTING"]
    if "V6_ALGORITHM_ENV" in os.environ:
        del os.environ["V6_ALGORITHM_ENV"]

# Helper functions for test validation
def assert_federated_equals_centralised(federated_result, centralised_result, tolerance=0.001):
    """Assert that federated and centralised results are equivalent within tolerance."""
    if isinstance(federated_result, dict) and isinstance(centralised_result, dict):
        assert set(federated_result.keys()) == set(centralised_result.keys())
        for key in federated_result.keys():
            assert_federated_equals_centralised(
                federated_result[key], 
                centralised_result[key], 
                tolerance
            )
    elif isinstance(federated_result, (int, float)) and isinstance(centralised_result, (int, float)):
        assert abs(federated_result - centralised_result) <= tolerance
    else:
        assert federated_result == centralised_result

def validate_algorithm_output_structure(result):
    """Validate that algorithm output has the expected structure."""
    assert isinstance(result, dict), "Result should be a dictionary"
    assert "included_organisations" in result
    assert "excluded_organisations" in result
    assert isinstance(result["included_organisations"], list)
    assert isinstance(result["excluded_organisations"], list)