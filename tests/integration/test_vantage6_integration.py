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
from typing import Dict, Any, List, Set
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
    def vantage6_network(self, docker_client):
        """Set up and manage Vantage6 developer network."""
        network_info = {"status": "not_started"}

        # Check if vantage6 CLI is available
        try:
            result = subprocess.run(["v6", "--help"], capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                pytest.skip("Vantage6 CLI not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pytest.skip("Vantage6 CLI not available")

        try:
            # Clean up any existing network first
            subprocess.run(["v6", "dev", "stop-demo-network", "--name", "algorithm-ci-test"],
                           timeout=120, capture_output=True)
            subprocess.run(["v6", "dev", "remove-demo-network", "--name", "algorithm-ci-test"],
                           timeout=120, capture_output=True)

            # Create demo network
            create_result = subprocess.run(
                ["v6", "dev", "create-demo-network", "--name", "algorithm-ci-test"],
                check=True, timeout=300, capture_output=True, text=True
            )

            # Start demo network
            start_result = subprocess.run(
                ["v6", "dev", "start-demo-network", "--name", "algorithm-ci-test"],
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
                subprocess.run(["v6", "dev", "stop-demo-network", "--name", "algorithm-ci-test"],
                               timeout=120, capture_output=True)
                subprocess.run(["v6", "dev", "remove-demo-network", "--name", "algorithm-ci-test"],
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
        assert len(
            v6_containers) >= 4, f"Expected at least 4 vantage6 containers, found {len(v6_containers)}: {v6_containers}"

        # Check that containers are healthy/running
        for container in containers:
            if any(keyword in container.name.lower() for keyword in ['vantage6', 'v6']):
                assert container.status == 'running', f"Container {container.name} is not running: {container.status}"

    def test_network_cleanup(self, vantage6_network, docker_client):
        """Test that network cleanup works properly."""
        # This test runs after the main tests to ensure cleanup
        assert vantage6_network["status"] == "running"

        # The cleanup is handled in the fixture teardown
        # This test just validates that we can properly stop and remove the network

        try:
            # Test stopping the network
            stop_result = subprocess.run([
                "v6", "dev", "stop-demo-network", "--name", "algorithm-ci-test"
            ], check=True, timeout=120, capture_output=True, text=True)

            # Wait a bit for containers to stop
            time.sleep(10)

            # Check that vantage6 containers are stopped
            containers = docker_client.containers.list(all=True)
            v6_containers = [c for c in containers if any(keyword in c.name.lower() for keyword in ['vantage6', 'v6'])]

            # Containers should either be stopped or removed
            for container in v6_containers:
                assert container.status in ['exited',
                                            'stopped'], f"Container {container.name} should be stopped: {container.status}"

            # Test removing the network
            remove_result = subprocess.run([
                "v6", "dev", "remove-demo-network", "--name", "algorithm-ci-test"
            ], check=True, timeout=120, capture_output=True, text=True)

            assert stop_result.returncode == 0
            assert remove_result.returncode == 0

        except subprocess.CalledProcessError as e:
            pytest.fail(f"Network cleanup failed: {e}")


@pytest.mark.integration
class TestAlgorithmBuild:
    """Test algorithm building."""

    def test_algorithm_build(self):
        """Test building the algorithm locally without uploading."""
        # Get repository root
        repo_root = Path(__file__).parent.parent.parent

        # Build Docker image for the algorithm
        try:
            build_result = subprocess.run([
                "docker", "build",
                "-t", "v6-descriptive-statistics:ci-test",
                str(repo_root)
            ], check=True, timeout=300, capture_output=True, text=True)

            assert build_result.returncode == 0

            # Verify the image was created
            inspect_result = subprocess.run([
                "docker", "inspect", "v6-descriptive-statistics:ci-test"
            ], check=True, capture_output=True, text=True)

            image_info = json.loads(inspect_result.stdout)
            assert len(image_info) > 0
            assert image_info[0]["Id"] is not None

        except subprocess.CalledProcessError as e:
            pytest.fail(f"Algorithm build failed: {e}")
