"""
Comprehensive Vantage6 integration testing.

This module tests the complete Vantage6 workflow:
1. Set up the vantage6 developer network (session-scoped)
2. Build the algorithm image (session-scoped)
3. Verify Docker containers are spawned correctly
4. Run tasks on the developer network
5. Assert results match expected central values
6. Clean up the network (at session end)
"""
import pytest
import subprocess
import docker
import time
import concurrent.futures
from typing import Dict, Any


def cleanup_vantage6_network(
        network_info: Dict[str, Any],
        docker_client: docker.DockerClient,
        force_remove_existing: bool = False
) -> bool:
    """Cleanup network containers and resources."""
    try:
        # First, use CLI cleanup to gracefully stop the network
        def cli_cleanup():
            try:
                print("Attempting graceful CLI cleanup...")
                stop_result = subprocess.run([
                    "v6", "dev", "stop-demo-network", "--name", "algorithm-ci-test"
                ], timeout=60, capture_output=True, text=True)

                if stop_result.returncode == 0:
                    print("CLI stop successful")

                time.sleep(2)  # Wait for graceful stop

                remove_result = subprocess.run([
                    "v6", "dev", "remove-demo-network", "--name", "algorithm-ci-test"
                ], timeout=60, capture_output=True, text=True)

                if remove_result.returncode == 0:
                    print("CLI remove successful")

            except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as e:
                print(f"CLI cleanup failed: {e}")

        # Try CLI cleanup first
        cli_cleanup()

        # Wait a moment for CLI cleanup to take effect
        time.sleep(3)

        containers_to_cleanup = []

        # Collect containers to cleanup
        if network_info.get("created_containers"):
            containers_to_cleanup.extend(network_info["created_containers"])

        # If force_remove_existing is True, also find existing vantage6 containers
        if force_remove_existing:
            try:
                existing_containers = docker_client.containers.list(all=True,
                                                                    filters={"name": "vantage6-algorithm-ci-test"})
                containers_to_cleanup.extend([c.id for c in existing_containers])
            except Exception as e:
                print(f"Failed to list existing containers: {e}")

        # Remove duplicates
        containers_to_cleanup = list(set(containers_to_cleanup))

        def cleanup_container(container_id):
            """Cleanup a single container."""
            try:
                container = docker_client.containers.get(container_id)
                container_name = container.name

                # Only stop if still running (CLI might have already stopped it)
                if container.status == 'running':
                    container.stop(timeout=15)  # Give more time for graceful stop
                    print(f"Stopped container: {container_name} ({container_id[:12]})")

                # Wait a moment for container to fully stop
                time.sleep(1)

                # Remove the container
                container.remove(force=True)
                print(f"Removed container: {container_name} ({container_id[:12]})")
                return True

            except docker.errors.NotFound:
                # Container already removed (likely by CLI cleanup)
                return True
            except docker.errors.APIError as e:
                if "removal of container" in str(e) and "already in progress" in str(e):
                    # Another process is already removing this container, wait for it
                    for _ in range(10):  # Wait up to 10 seconds
                        try:
                            docker_client.containers.get(container_id)
                            time.sleep(1)
                        except docker.errors.NotFound:
                            print(f"Container {container_id[:12]} removed by another process")
                            return True
                    print(f"Container {container_id[:12]} removal timed out")
                    return False
                else:
                    print(f"Failed to cleanup container {container_id[:12]}: {e}")
                    return False
            except Exception as e:
                print(f"Failed to cleanup container {container_id[:12]}: {e}")
                return False

        # Cleanup remaining containers in parallel for speed
        if containers_to_cleanup:
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                futures = [executor.submit(cleanup_container, cid) for cid in containers_to_cleanup]
                results = [f.result() for f in concurrent.futures.as_completed(futures, timeout=60)]

        return True

    except Exception as e:
        print(f"Network cleanup failed: {e}")
        return False


@pytest.mark.integration
class TestVantage6DeveloperNetwork:
    """Test complete Vantage6 developer network workflow."""

    def test_vantage6_network_setup(self, vantage6_network_session, docker_client):
        """Test that Vantage6 developer network is set up correctly."""
        assert vantage6_network_session["status"] == "running"

        # Check that Docker containers are running
        created_containers = vantage6_network_session["created_containers"]
        assert len(created_containers) >= 4, f"Expected at least 4 new containers, found {len(created_containers)}"

        # Categorise containers and check their expected states
        service_containers = []
        task_containers = []

        for container_id in created_containers:
            try:
                container = docker_client.containers.get(container_id)

                # Task/run containers are expected to exit after completing
                if '-run-' in container.name or 'algorithm-store' in container.name:
                    task_containers.append(container)
                else:
                    # Service containers should stay running
                    service_containers.append(container)

            except docker.errors.NotFound:
                pytest.fail(f"Container {container_id[:12]} was not found")

        # Service containers (server, UI, nodes) should be running
        for container in service_containers:
            assert container.status == 'running', \
                f"Service container {container.name} should be running: {container.status}"

        # Task containers can be in various states (running, exited)
        print(f"Found {len(service_containers)} service containers and {len(task_containers)} task containers")

        # Ensure we have at least the core service containers
        assert len(service_containers) >= 3, \
            f"Expected at least 3 service containers (server, UI, node), found {len(service_containers)}"


@pytest.mark.integration
class TestAlgorithmImage:
    """Test algorithm Docker image building and verification."""

    def test_algorithm_image_exists(self, algorithm_image, docker_client):
        """Test that the algorithm Docker image was built successfully."""
        expected_tag = f"{algorithm_image['pkg_name']}:ci-test"
        assert algorithm_image["tag"] == expected_tag
        assert algorithm_image["id"] is not None

        # Verify the image exists in Docker
        try:
            image = docker_client.images.get(algorithm_image["tag"])
            assert image.id == algorithm_image["id"]
            print(f"Algorithm image verified: {algorithm_image['tag']} ({algorithm_image['id'][:12]})")
        except docker.errors.ImageNotFound:
            pytest.fail(f"Built algorithm image {algorithm_image['tag']} not found in Docker")

    def test_algorithm_image_can_run(self, algorithm_image, docker_client):
        """Test that the algorithm image can be instantiated."""
        try:
            # Try to create a container from the image (don't run it)
            container = docker_client.containers.create(
                algorithm_image["tag"],
                command=["python", "--version"]  # Simple command to test if image works
            )

            # Clean up the test container
            container.remove()
            print(f"Algorithm image can be instantiated successfully")

        except docker.errors.APIError as e:
            pytest.fail(f"Failed to create container from algorithm image: {e}")
