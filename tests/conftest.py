"""
Pytest configuration and fixtures for v6-descriptive-statistics tests.
"""
import pytest
import pandas as pd
import numpy as np
import docker
import os
from pathlib import Path


@pytest.fixture(scope="session")
def docker_test_setup():
    """Check if Docker is available for testing."""
    try:
        client = docker.from_env()
        client.ping()
        return {"docker_available": True, "client": client}
    except docker.errors.DockerException:
        return {"docker_available": False, "client": None}


@pytest.fixture
def sample_test_data():
    """Create sample test data for algorithm validation."""
    np.random.seed(42)
    
    data = pd.DataFrame({
        'Name': [f'Person_{i}' for i in range(100)],
        'Gender': np.random.choice(['M', 'F'], size=100),
        'Age': np.random.normal(30, 10, 100),
        'Height(in)': np.random.normal(68, 4, 100),
        'Weight(lbs)': np.random.normal(150, 20, 100)
    })
    
    return data


@pytest.fixture
def algorithm_variables_config():
    """Standard variables configuration for algorithm testing."""
    return {
        "Gender": {
            "datatype": "categorical",
            "inliers": ("M", "F", "X")
        },
        "Age": {
            "datatype": "numerical",
            "inliers": (15, 80)
        },
        "Height(in)": {
            "datatype": "numerical",
            "inliers": (60, 80)
        },
        "Weight(lbs)": {
            "datatype": "numerical",
            "inliers": (100, 300)
        }
    }


# Pytest markers for organizing tests
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line("markers", "unit: Unit tests for algorithm functions")
    config.addinivalue_line("markers", "integration: Integration tests with Vantage6")
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")
    config.addinivalue_line("markers", "vantage6: Tests that require Vantage6 infrastructure")
    config.addinivalue_line("markers", "docker: Tests that require Docker")


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers based on test names/paths."""
    for item in items:
        # Add slow marker to integration tests
        if "integration" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Add vantage6 marker to tests that use vantage6
        if "vantage6" in item.nodeid or "vantage6" in item.name:
            item.add_marker(pytest.mark.vantage6)
        
        # Add docker marker to tests that use docker
        if "docker" in item.nodeid or "docker" in item.name:
            item.add_marker(pytest.mark.docker)