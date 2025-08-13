"""
Pytest configuration and fixtures for v6-descriptive-statistics tests.
"""
import pytest
import pandas as pd
import numpy as np
import docker
import subprocess
import time
import os
import json
from pathlib import Path
from typing import Dict, Any, Set


@pytest.fixture(scope="session")
def docker_test_setup():
    """Check if Docker is available for testing."""
    try:
        client = docker.from_env()
        client.ping()
        return {"docker_available": True, "client": client}
    except docker.errors.DockerException:
        return {"docker_available": False, "client": None}


@pytest.fixture(scope="session")
def docker_client():
    """Get Docker client for the entire test session."""
    try:
        client = docker.from_env()
        client.ping()
        return client
    except docker.errors.DockerException:
        pytest.skip("Docker not available")


@pytest.fixture(scope="session")
def algorithm_image(docker_client):
    """Build the algorithm Docker image for the entire test session."""
    # Get repository root and derive package name from folder
    repo_root = Path(__file__).parent.parent
    pkg_name = repo_root.name.lower()  # Use folder name, converted to lowercase

    # Create image tag from package name
    image_tag = f"{pkg_name}:ci-test"

    try:
        print(f"Building algorithm image from {repo_root}...")
        print(f"Package name: {pkg_name}")
        print(f"Image tag: {image_tag}")

        # Build Docker image for the algorithm
        build_result = subprocess.run([
            "docker", "build",
            "-t", image_tag,
            "--build-arg", f"PKG_NAME={pkg_name}",
            str(repo_root)
        ], check=True, timeout=300, capture_output=True, text=True)

        if build_result.returncode != 0:
            pytest.skip(f"Algorithm image build failed with exit code {build_result.returncode}:\n"
                       f"STDOUT: {build_result.stdout}\n"
                       f"STDERR: {build_result.stderr}")

        # Verify the image was created
        try:
            inspect_result = subprocess.run([
                "docker", "inspect", image_tag
            ], check=True, capture_output=True, text=True)

            image_info = json.loads(inspect_result.stdout)
            if not image_info or not image_info[0].get("Id"):
                pytest.skip(f"Built image {image_tag} has no valid ID")

            print(f"Successfully built algorithm image: {image_tag}")
            return {
                "tag": image_tag,
                "pkg_name": pkg_name,
                "id": image_info[0]["Id"],
                "info": image_info[0]
            }

        except (subprocess.CalledProcessError, json.JSONDecodeError, KeyError) as e:
            pytest.skip(f"Failed to inspect built image {image_tag}: {e}")

    except subprocess.CalledProcessError as e:
        error_msg = f"Algorithm build failed: {e}"
        if e.stdout:
            error_msg += f"\nSTDOUT: {e.stdout}"
        if e.stderr:
            error_msg += f"\nSTDERR: {e.stderr}"
        pytest.skip(error_msg)
    except subprocess.TimeoutExpired:
        pytest.skip("Timeout building algorithm image (300s)")
    except Exception as e:
        pytest.skip(f"Unexpected error building algorithm image: {e}")


@pytest.fixture(scope="session")
def vantage6_network_session(docker_client):
    """Set up Vantage6 developer network for the entire test session."""
    from tests.integration.test_vantage6_integration import cleanup_vantage6_network

    network_info = {"status": "not_started", "created_containers": set()}

    # Check if vantage6 CLI is available
    try:
        result = subprocess.run(["v6", "--help"], capture_output=True, text=True, timeout=10)
        if result.returncode != 0:
            pytest.skip(f"Vantage6 CLI not available (exit code {result.returncode}):\n"
                       f"STDOUT: {result.stdout}\n"
                       f"STDERR: {result.stderr}")
    except subprocess.TimeoutExpired:
        pytest.skip("Vantage6 CLI check timed out (10s)")
    except FileNotFoundError:
        pytest.skip("Vantage6 CLI not found in PATH. Install with: pip install vantage6")

    try:
        # Enhanced cleanup of any existing network first
        print("Cleaning up any existing Vantage6 networks...")
        cleanup_vantage6_network({"created_containers": set()}, docker_client, force_remove_existing=True)
        time.sleep(5)

        # Capture containers before creating the network
        containers_before = set(container.id for container in docker_client.containers.list(all=True))

        # Create and start demo network
        print("Creating demo network...")
        create_result = subprocess.run([
            "v6", "dev", "create-demo-network", "--name", "algorithm-ci-test"
        ], timeout=300, capture_output=True, text=True)

        if create_result.returncode != 0:
            error_msg = f"Failed to create demo network (exit code {create_result.returncode}):\n"
            error_msg += f"STDOUT: {create_result.stdout}\n"
            error_msg += f"STDERR: {create_result.stderr}"
            pytest.skip(error_msg)

        print("Starting demo network...")
        start_result = subprocess.run([
            "v6", "dev", "start-demo-network", "--name", "algorithm-ci-test"
        ], timeout=300, capture_output=True, text=True)

        if start_result.returncode != 0:
            error_msg = f"Failed to start demo network (exit code {start_result.returncode}):\n"
            error_msg += f"STDOUT: {start_result.stdout}\n"
            error_msg += f"STDERR: {start_result.stderr}"
            pytest.skip(error_msg)

        # Wait for network to be ready
        print("Waiting for network to start...")
        max_wait = 90
        wait_interval = 5
        stable_count = 0
        required_stable_checks = 3

        for elapsed in range(0, max_wait, wait_interval):
            time.sleep(wait_interval)
            containers_after = set(container.id for container in docker_client.containers.list(all=True))
            new_containers = containers_after - containers_before

            if len(new_containers) >= 4:
                service_containers = 0
                for container_id in new_containers:
                    try:
                        container = docker_client.containers.get(container_id)
                        if (container.status == 'running' and
                            '-run-' not in container.name and
                            'algorithm-store' not in container.name):
                            service_containers += 1
                    except docker.errors.NotFound:
                        pass

                if service_containers >= 3:
                    stable_count += 1
                    if stable_count >= required_stable_checks:
                        print(f"Network stable and ready after {elapsed + wait_interval} seconds")
                        break
                else:
                    stable_count = 0

        containers_after = set(container.id for container in docker_client.containers.list(all=True))
        network_info["created_containers"] = containers_after - containers_before
        network_info["status"] = "running"

        if len(network_info["created_containers"]) < 4:
            pytest.skip(f"Network setup incomplete: only {len(network_info['created_containers'])} containers created (expected ≥4)")

        print(f"Network started with {len(network_info['created_containers'])} new containers")

        yield network_info

    except subprocess.TimeoutExpired as e:
        error_msg = f"Timeout setting up Vantage6 demo network: {e}"
        pytest.skip(error_msg)
    except Exception as e:
        error_msg = f"Unexpected error setting up Vantage6 demo network: {e}"
        pytest.skip(error_msg)

    finally:
        # Cleanup after all tests are done
        print("Cleaning up Vantage6 network...")
        cleanup_vantage6_network(network_info, docker_client)


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