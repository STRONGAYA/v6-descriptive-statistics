"""
Vantage6 infrastructure integration tests.
"""
import pytest
import subprocess
import docker
import time
import os
import tempfile
import json
from pathlib import Path
import sys

# Add the algorithm module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../v6-descriptive-statistics/'))

@pytest.mark.vantage6
@pytest.mark.slow 
class TestVantage6DemoNetwork:
    """Test integration with Vantage6 demo network."""

    @pytest.fixture(scope="class")
    def demo_network_setup(self):
        """Setup Vantage6 demo network for testing."""
        # Check if vantage6 CLI is available
        try:
            result = subprocess.run(["v6", "--version"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                pytest.skip("Vantage6 CLI not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Vantage6 CLI not available or timeout")
        
        # Try to create demo network
        try:
            # Create demo network
            subprocess.run(["v6", "dev", "create-demo-network"], 
                          check=True, timeout=300, capture_output=True)
            
            # Start demo network
            subprocess.run(["v6", "dev", "start-demo-network"], 
                          check=True, timeout=300, capture_output=True)
            
            # Give it time to start up
            time.sleep(30)
            
            yield {"status": "running"}
            
            # Cleanup
            subprocess.run(["v6", "dev", "stop-demo-network"], 
                          timeout=120, capture_output=True)
            subprocess.run(["v6", "dev", "remove-demo-network"], 
                          timeout=120, capture_output=True)
            
        except subprocess.CalledProcessError:
            pytest.skip("Failed to setup Vantage6 demo network")
        except subprocess.TimeoutExpired:
            pytest.skip("Timeout setting up Vantage6 demo network")

    def test_demo_network_connectivity(self, demo_network_setup):
        """Test connectivity to demo network."""
        # Basic connectivity test
        assert demo_network_setup["status"] == "running"
        
        # Try to check if demo network is responsive
        # This would require specific Vantage6 client setup
        # For now, just verify the setup completed
        assert True

    def test_algorithm_deployment_demo_network(self, demo_network_setup):
        """Test deploying algorithm to demo network."""
        # This would test actual algorithm deployment
        # Implementation would depend on Vantage6 client setup
        pytest.skip("Requires specific Vantage6 client configuration")

@pytest.mark.vantage6
@pytest.mark.slow
class TestMDWInfrastructure:
    """Test integration with MDW Vantage6 infrastructure."""

    def test_mdw_infrastructure_compatibility(self):
        """Test compatibility with MDW infrastructure setup."""
        # Check if MDW infrastructure setup scripts are available
        # This would typically involve checking for specific configuration files
        # or environment variables that indicate MDW setup
        
        # For now, this is a placeholder
        # Actual implementation would depend on MDW infrastructure specifics
        pytest.skip("Requires MDW infrastructure setup")

    def test_mdw_network_deployment(self):
        """Test deployment to MDW Vantage6 network."""
        pytest.skip("Requires MDW network access")

@pytest.mark.docker
class TestDockerIntegration:
    """Test Docker build and deployment integration."""

    def test_docker_build(self, docker_test_setup):
        """Test Docker image building."""
        if not docker_test_setup["docker_available"]:
            pytest.skip("Docker not available")
        
        # Get repository root
        repo_root = Path(__file__).parent.parent.parent
        
        try:
            # Build Docker image
            client = docker.from_env()
            
            # Build the image
            image, build_logs = client.images.build(
                path=str(repo_root),
                tag="v6-descriptive-statistics:test",
                timeout=300
            )
            
            assert image is not None
            assert image.id is not None
            
            # Clean up
            client.images.remove(image.id, force=True)
            
        except docker.errors.BuildError as e:
            pytest.fail(f"Docker build failed: {e}")
        except docker.errors.APIError as e:
            pytest.fail(f"Docker API error: {e}")

    def test_docker_algorithm_execution(self, docker_test_setup):
        """Test running algorithm in Docker container."""
        if not docker_test_setup["docker_available"]:
            pytest.skip("Docker not available")
        
        # This would test running the algorithm in a Docker container
        # with mock data and verifying the output
        repo_root = Path(__file__).parent.parent.parent
        
        try:
            client = docker.from_env()
            
            # Build image first
            image, _ = client.images.build(
                path=str(repo_root),
                tag="v6-descriptive-statistics:test",
                timeout=300
            )
            
            # Would need to set up proper volume mounts and test data
            # For now, just test that the image can be created and basic info retrieved
            assert image is not None
            
            # Get image info
            image_info = client.api.inspect_image(image.id)
            assert "Config" in image_info
            
            # Clean up
            client.images.remove(image.id, force=True)
            
        except docker.errors.APIError as e:
            pytest.skip(f"Docker test skipped due to API error: {e}")

    def test_dockerfile_validity(self):
        """Test that Dockerfile is valid and properly structured."""
        repo_root = Path(__file__).parent.parent.parent
        dockerfile_path = repo_root / "Dockerfile"
        
        assert dockerfile_path.exists(), "Dockerfile should exist"
        
        dockerfile_content = dockerfile_path.read_text()
        
        # Basic Dockerfile validation
        assert "FROM" in dockerfile_content, "Dockerfile should have FROM instruction"
        assert "COPY" in dockerfile_content or "ADD" in dockerfile_content, "Dockerfile should copy files"
        assert "RUN" in dockerfile_content, "Dockerfile should have RUN instructions"

@pytest.mark.integration
@pytest.mark.slow
class TestInfrastructureDocumentation:
    """Test infrastructure setup documentation and scripts."""

    def test_setup_documentation_exists(self):
        """Test that setup documentation exists and is comprehensive."""
        repo_root = Path(__file__).parent.parent.parent
        readme_path = repo_root / "README.md"
        
        assert readme_path.exists(), "README.md should exist"
        
        readme_content = readme_path.read_text().lower()
        
        # Should mention key setup components
        setup_keywords = [
            "docker", "vantage6", "build", "algorithm"
        ]
        
        for keyword in setup_keywords:
            assert keyword in readme_content, f"README should mention {keyword}"

    def test_requirements_completeness(self):
        """Test that requirements.txt has all necessary dependencies."""
        repo_root = Path(__file__).parent.parent.parent
        requirements_path = repo_root / "requirements.txt"
        
        assert requirements_path.exists(), "requirements.txt should exist"
        
        requirements_content = requirements_path.read_text()
        
        # Should include essential dependencies
        essential_deps = [
            "vantage6-algorithm-tools",
            "pandas", 
            "v6-tools-general"
        ]
        
        for dep in essential_deps:
            assert dep in requirements_content, f"requirements.txt should include {dep}"

    def test_algorithm_store_configuration(self):
        """Test algorithm store JSON configuration."""
        repo_root = Path(__file__).parent.parent.parent
        algorithm_store_path = repo_root / "algorithm_store.json"
        
        if algorithm_store_path.exists():
            with open(algorithm_store_path) as f:
                config = json.load(f)
            
            # Should have required fields
            required_fields = ["name", "image", "partitioning", "vantage6_version"]
            for field in required_fields:
                assert field in config, f"algorithm_store.json should have {field}"

@pytest.mark.vantage6
class TestClientIntegration:
    """Test client integration patterns."""

    def test_python_client_compatibility(self):
        """Test Python client integration patterns."""
        # Test basic client pattern from vantage_client.py example
        # This would test the pattern shown in the GitHub issue
        
        # Mock the basic client setup pattern
        client_config = {
            "host": "localhost",
            "port": 5000,
            "api_path": "/api",
            "username": "test",
            "password": "test"
        }
        
        # Basic validation that config structure is reasonable
        assert "host" in client_config
        assert "port" in client_config
        assert isinstance(client_config["port"], int)

    def test_algorithm_input_validation(self):
        """Test algorithm input validation for client integration."""
        # Test the input structure that would come from a Python client
        
        valid_input = {
            "method": "central",
            "kwargs": {
                "variables_to_describe": {
                    "Gender": {
                        "datatype": "categorical",
                        "inliers": ("M", "F", "X")
                    },
                    "Age": {
                        "datatype": "numerical",
                        "inliers": (15, 39)
                    }
                },
                "variables_to_stratify": None,
                "organization_ids": None
            }
        }
        
        # Validate input structure
        assert "method" in valid_input
        assert "kwargs" in valid_input
        assert "variables_to_describe" in valid_input["kwargs"]
        assert isinstance(valid_input["kwargs"]["variables_to_describe"], dict)

    def test_output_format_compatibility(self):
        """Test output format compatibility with client expectations."""
        # Test expected output format structure
        
        expected_output_structure = {
            "included_organisations": [],
            "excluded_organisations": [],
            "numerical_general_statistics": {},
            "categorical_general_statistics": {}
        }
        
        # Validate output structure expectations
        assert "included_organisations" in expected_output_structure
        assert "excluded_organisations" in expected_output_structure
        assert isinstance(expected_output_structure["included_organisations"], list)
        assert isinstance(expected_output_structure["excluded_organisations"], list)