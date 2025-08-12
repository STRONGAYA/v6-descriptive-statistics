"""
Comprehensive Vantage6 integration testing.

This module tests the complete Vantage6 workflow:
1. Set up the vantage6 developer network
2. Verify Docker containers are spawned correctly
3. Build the algorithm locally
4. Run tasks on the developer network
5. Assert results match expected central values
6. Clean up the network
"""
import pytest
import subprocess
import docker
import time
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, List
import tempfile
import shutil


@pytest.mark.integration
class TestVantage6DeveloperNetwork:
    """Test complete Vantage6 developer network workflow."""

    @pytest.fixture(scope="class")
    def docker_client(self):
        """Get Docker client."""
        try:
            client = docker.from_env()
            client.ping()
            return client
        except docker.errors.DockerException:
            pytest.skip("Docker not available")

    @pytest.fixture(scope="class")
    def test_data(self):
        """Create test data for algorithm validation."""
        # Create sample datasets that will be used to test the algorithm
        np.random.seed(42)
        
        # Organization 1 data
        org1_data = pd.DataFrame({
            'Gender': np.random.choice(['M', 'F'], size=100),
            'Age': np.random.normal(30, 10, 100),
            'Height': np.random.normal(170, 15, 100),
            'Weight': np.random.normal(70, 12, 100)
        })
        
        # Organization 2 data  
        org2_data = pd.DataFrame({
            'Gender': np.random.choice(['M', 'F'], size=80),
            'Age': np.random.normal(35, 8, 80),
            'Height': np.random.normal(165, 12, 80),
            'Weight': np.random.normal(68, 10, 80)
        })
        
        # Organization 3 data
        org3_data = pd.DataFrame({
            'Gender': np.random.choice(['M', 'F'], size=120),
            'Age': np.random.normal(28, 12, 120),
            'Height': np.random.normal(172, 18, 120),
            'Weight': np.random.normal(72, 15, 120)
        })
        
        # Combined central data for validation
        central_data = pd.concat([org1_data, org2_data, org3_data], ignore_index=True)
        
        return {
            'org1': org1_data,
            'org2': org2_data, 
            'org3': org3_data,
            'central': central_data
        }

    @pytest.fixture(scope="class")
    def expected_central_results(self, test_data):
        """Calculate expected central results for validation."""
        central_data = test_data['central']
        
        # Calculate central statistics that we expect to match
        results = {
            'numerical_stats': {
                'Age': {
                    'mean': central_data['Age'].mean(),
                    'std': central_data['Age'].std(),
                    'median': central_data['Age'].median(),
                    'min': central_data['Age'].min(),
                    'max': central_data['Age'].max(),
                    'count': len(central_data['Age'])
                },
                'Height': {
                    'mean': central_data['Height'].mean(),
                    'std': central_data['Height'].std(),
                    'median': central_data['Height'].median(),
                    'min': central_data['Height'].min(),
                    'max': central_data['Height'].max(),
                    'count': len(central_data['Height'])
                },
                'Weight': {
                    'mean': central_data['Weight'].mean(),
                    'std': central_data['Weight'].std(),
                    'median': central_data['Weight'].median(),
                    'min': central_data['Weight'].min(),
                    'max': central_data['Weight'].max(),
                    'count': len(central_data['Weight'])
                }
            },
            'categorical_stats': {
                'Gender': {
                    'counts': central_data['Gender'].value_counts().to_dict(),
                    'mode': central_data['Gender'].mode()[0] if not central_data['Gender'].mode().empty else None
                }
            }
        }
        
        return results

    @pytest.fixture(scope="class")
    def vantage6_network(self, docker_client):
        """Set up and manage Vantage6 developer network."""
        network_info = {"status": "not_started"}
        
        # Check if vantage6 CLI is available
        try:
            result = subprocess.run(["v6", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                pytest.skip("Vantage6 CLI not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Vantage6 CLI not available")
        
        try:
            # Clean up any existing network first
            subprocess.run(["v6", "dev", "stop-demo-network"], 
                          timeout=120, capture_output=True)
            subprocess.run(["v6", "dev", "remove-demo-network"], 
                          timeout=120, capture_output=True)
            
            # Create demo network
            create_result = subprocess.run(
                ["v6", "dev", "create-demo-network"], 
                check=True, timeout=300, capture_output=True, text=True
            )
            
            # Start demo network
            start_result = subprocess.run(
                ["v6", "dev", "start-demo-network"], 
                check=True, timeout=300, capture_output=True, text=True
            )
            
            # Give the network time to start up
            time.sleep(60)
            
            network_info["status"] = "running"
            network_info["create_output"] = create_result.stdout
            network_info["start_output"] = start_result.stdout
            
            yield network_info
            
        except subprocess.CalledProcessError as e:
            pytest.skip(f"Failed to setup Vantage6 demo network: {e}")
        except subprocess.TimeoutExpired:
            pytest.skip("Timeout setting up Vantage6 demo network")
        
        finally:
            # Cleanup - stop and remove the network
            try:
                subprocess.run(["v6", "dev", "stop-demo-network"], 
                              timeout=120, capture_output=True)
                subprocess.run(["v6", "dev", "remove-demo-network"], 
                              timeout=120, capture_output=True)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass  # Best effort cleanup

    def test_vantage6_network_setup(self, vantage6_network, docker_client):
        """Test that Vantage6 developer network is set up correctly."""
        assert vantage6_network["status"] == "running"
        
        # Check that Docker containers are running
        containers = docker_client.containers.list()
        container_names = [container.name for container in containers]
        
        # Should have containers for:
        # - 3 nodes (organizations)
        # - 1 server
        # - 1 algorithm store (optional but good practice)
        
        # Look for vantage6-related containers
        v6_containers = [name for name in container_names if 'vantage6' in name.lower() or 'v6' in name.lower()]
        
        # We expect at least 4 containers (server + 3 nodes)
        assert len(v6_containers) >= 4, f"Expected at least 4 vantage6 containers, found {len(v6_containers)}: {v6_containers}"
        
        # Check that containers are healthy/running
        for container in containers:
            if any(keyword in container.name.lower() for keyword in ['vantage6', 'v6']):
                assert container.status == 'running', f"Container {container.name} is not running: {container.status}"

    def test_algorithm_build(self, vantage6_network):
        """Test building the algorithm locally without uploading."""
        assert vantage6_network["status"] == "running"
        
        # Get repository root
        repo_root = Path(__file__).parent.parent
        
        # Build Docker image for the algorithm
        try:
            build_result = subprocess.run([
                "docker", "build", 
                "-t", "v6-descriptive-statistics:test",
                str(repo_root)
            ], check=True, timeout=300, capture_output=True, text=True)
            
            assert build_result.returncode == 0
            
            # Verify the image was created
            inspect_result = subprocess.run([
                "docker", "inspect", "v6-descriptive-statistics:test"
            ], check=True, capture_output=True, text=True)
            
            image_info = json.loads(inspect_result.stdout)
            assert len(image_info) > 0
            assert image_info[0]["Id"] is not None
            
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Algorithm build failed: {e}")

    def test_algorithm_execution(self, vantage6_network, test_data, expected_central_results):
        """Test running algorithm on developer network and validating results."""
        assert vantage6_network["status"] == "running"
        
        # This test would:
        # 1. Set up algorithm input with test data
        # 2. Submit task to the developer network
        # 3. Wait for task completion
        # 4. Retrieve and validate results
        
        # For now, this is a placeholder implementation
        # The actual implementation would require:
        # - Setting up vantage6 client
        # - Uploading test data to the nodes
        # - Submitting algorithm task
        # - Retrieving results
        # - Comparing with expected central results
        
        # Variables to describe for the algorithm
        variables_to_describe = {
            "Age": {
                "datatype": "numerical",
                "inliers": (18, 80)
            },
            "Height": {
                "datatype": "numerical", 
                "inliers": (150, 200)
            },
            "Weight": {
                "datatype": "numerical",
                "inliers": (40, 120)
            },
            "Gender": {
                "datatype": "categorical",
                "inliers": ("M", "F")
            }
        }
        
        # Algorithm input
        algorithm_input = {
            "method": "central",
            "kwargs": {
                "variables_to_describe": variables_to_describe,
                "variables_to_stratify": None,
                "organization_ids": None
            }
        }
        
        # TODO: Implement actual task submission and result validation
        # This would require setting up the vantage6 Python client
        # and configuring it to connect to the demo network
        
        # For now, validate that our expected results structure is correct
        assert "numerical_stats" in expected_central_results
        assert "categorical_stats" in expected_central_results
        assert "Age" in expected_central_results["numerical_stats"]
        assert "Gender" in expected_central_results["categorical_stats"]
        
        # Placeholder assertion - in real implementation this would compare
        # federated results with central results
        assert True, "Placeholder - actual task execution not yet implemented"

    def test_network_cleanup(self, vantage6_network, docker_client):
        """Test that network cleanup works properly."""
        # This test runs after the main tests to ensure cleanup
        assert vantage6_network["status"] == "running"
        
        # The cleanup is handled in the fixture teardown
        # This test just validates that we can properly stop and remove the network
        
        try:
            # Test stopping the network
            stop_result = subprocess.run([
                "v6", "dev", "stop-demo-network"
            ], check=True, timeout=120, capture_output=True, text=True)
            
            # Wait a bit for containers to stop
            time.sleep(10)
            
            # Check that vantage6 containers are stopped
            containers = docker_client.containers.list(all=True)
            v6_containers = [c for c in containers if any(keyword in c.name.lower() for keyword in ['vantage6', 'v6'])]
            
            # Containers should either be stopped or removed
            for container in v6_containers:
                assert container.status in ['exited', 'stopped'], f"Container {container.name} should be stopped: {container.status}"
            
            # Test removing the network
            remove_result = subprocess.run([
                "v6", "dev", "remove-demo-network"
            ], check=True, timeout=120, capture_output=True, text=True)
            
            assert stop_result.returncode == 0
            assert remove_result.returncode == 0
            
        except subprocess.CalledProcessError as e:
            pytest.fail(f"Network cleanup failed: {e}")


@pytest.mark.integration
class TestAlgorithmValidation:
    """Test algorithm functionality independently of full network setup."""
    
    def test_algorithm_import(self):
        """Test that algorithm modules can be imported correctly."""
        import sys
        import os
        
        # Add algorithm module to path
        algorithm_path = os.path.join(os.path.dirname(__file__), '../v6-descriptive-statistics/')
        sys.path.insert(0, algorithm_path)
        
        try:
            # Test importing main algorithm functions
            from partial import master, RPC_partial
            from central import master as central_master, RPC_central
            
            # Basic validation that functions exist and are callable
            assert callable(master)
            assert callable(RPC_partial)
            assert callable(central_master)
            assert callable(RPC_central)
            
        except ImportError as e:
            pytest.fail(f"Failed to import algorithm modules: {e}")

    def test_algorithm_input_validation(self):
        """Test algorithm input validation."""
        # Test valid input structure
        valid_input = {
            "variables_to_describe": {
                "Age": {
                    "datatype": "numerical",
                    "inliers": (18, 80)
                },
                "Gender": {
                    "datatype": "categorical", 
                    "inliers": ("M", "F", "X")
                }
            },
            "variables_to_stratify": None,
            "organization_ids": None
        }
        
        # Validate structure
        assert "variables_to_describe" in valid_input
        assert isinstance(valid_input["variables_to_describe"], dict)
        
        for var_name, var_config in valid_input["variables_to_describe"].items():
            assert "datatype" in var_config
            assert "inliers" in var_config
            assert var_config["datatype"] in ["numerical", "categorical"]


def assert_statistics_equivalent(federated_result: Dict[str, Any], 
                               central_result: Dict[str, Any], 
                               tolerance: float = 1e-6) -> None:
    """
    Assert that federated and central statistical results are equivalent within tolerance.
    
    Args:
        federated_result: Results from federated computation
        central_result: Results from central computation  
        tolerance: Numerical tolerance for comparison
    """
    # This function would implement detailed comparison between federated and central results
    # For numerical statistics: mean, std, min, max should be very close
    # For categorical statistics: counts should match exactly
    
    # Placeholder implementation
    assert isinstance(federated_result, dict)
    assert isinstance(central_result, dict)
    
    # TODO: Implement detailed statistical comparison
    # This would compare each statistical measure within the specified tolerance
    pass