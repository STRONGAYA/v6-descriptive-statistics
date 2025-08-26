
# v6-descriptive-statistics

## Testing Status

<p align="center">
<a href="https://github.com/STRONGAYA/v6-descriptive-statistics/workflows/"><img alt="Test status" src="https://github.com/STRONGAYA/v6-descriptive-statistics/workflows/Test%20Suite/badge.svg)"></a>
<a><img alt="Coverage" src="https://raw.githubusercontent.com/STRONGAYA/v6-descriptive-statistics/main/tests/coverage-badge.svg"></a>
<a href="https://www.python.org/downloads/"><img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10+-blue.svg"></a>
<a href="https://opensource.org/licenses/Apache-2.0"><img alt="Licence: Apache 2.0" src="https://img.shields.io/badge/Licence-Apache%202.0-blue.svg"></a>
<br>
<a href="https://github.com/vantage6/vantage6/"><img alt="Vantage6 4.10, 4.11, 4.12" src="https://img.shields.io/badge/vantage6-4.10 | 4.11 | 4.12-blue.svg"></a>
<a href="https://strongaya.eu/wp-content/uploads/2025/07/algorithm_review_guidelines.pdf"><img alt="STRONG AYA Algorithm Guideline Conformity: v1.0.0 Pending" src="https://img.shields.io/badge/STRONG%20AYA%20Algorithm%20Guideline%20Conformity-v1.0.0%20pending-yellow"></a>
<br>
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
<a href="https://flake8.pycqa.org/"><img alt="Linting: flake8" src="https://img.shields.io/badge/linting-flake8-informational"></a>
<a href="http://mypy-lang.org/"><img alt="Type checking: mypy" src="https://img.shields.io/badge/type%20checking-mypy-informational"></a>
<a href="https://github.com/PyCQA/bandit"><img alt="Security: bandit" src="https://img.shields.io/badge/security-bandit-informational"></a>
<a href="https://github.com/pyupio/safety"><img alt="Security: safety" src="https://img.shields.io/badge/security-safety-informational"></a>
</p>

<!--
To show the approved badge instead, use:
<a href="https://strongaya.eu/wp-content/uploads/2025/07/algorithm_review_guidelines.pdf"><img alt="STRONG AYA Algorithm Guideline Conformity: v1.0.0 Approved" src="https://img.shields.io/badge/STRONG%20AYA%20Algorithm%20Guideline%20Conformity-v1.0.0%20approved-brightgreen">
-->


Vantage6 algorithm that retrieves descriptive statistics with comprehensive Vantage6 integration testing.

This algorithm is designed to be run with the [vantage6](https://vantage6.ai)
infrastructure for distributed analysis and learning.

[//]: # (TODO clean-up)

The base code for this algorithm has been created via the
[v6-algorithm-template](https://github.com/vantage6/v6-algorithm-template)
template generator.

## Tested Vantage6 Versions

This algorithm is continuously tested against the following Vantage6 versions:
- **Vantage6 4.12** - Current tested version

## Features

- **Vantage6 Integration Testing**: Complete developer network setup, Docker container validation, and algorithm execution testing
- **Algorithm Functionality Testing**: Unit tests for core algorithm functions and statistical computations
- **Quality Assurance**: Automated code quality checks with Black, Flake8, MyPy, Bandit, and Safety
- **Docker Integration**: Automated Docker build and deployment validation
- **Docker Support**: Validated Docker builds and deployment testing
- **Empirical Validation**: Federated vs centralised computation equivalence testing
- **Edge Case Coverage**: Comprehensive testing of edge cases and error conditions

## Testing Framework

### Test Structure

```
tests/
├── conftest.py                 # Common fixtures and test utilities
├── unit/                       # Unit tests for individual functions
│   └── test_algorithm_functions.py
├── integration/                # Integration tests for complete workflows
│   ├── test_algorithm_workflows.py
│   └── test_vantage6_integration.py
├── empirical/                  # Empirical validation tests
│   └── test_federated_equivalence.py
└── utils/                      # Test helper utilities
    └── test_helpers.py
```

### Running Tests

#### Prerequisites

Install test dependencies:

```bash
pip install pytest pytest-cov pytest-mock hypothesis faker psutil scipy docker
pip install -r requirements.txt
```

#### Basic Test Execution

```bash
# Run all tests
pytest

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only  
pytest -m empirical     # Empirical validation tests only
pytest -m vantage6      # Vantage6 integration tests only
pytest -m docker        # Docker-related tests only
pytest -m edge_case     # Edge case tests only

# Run all tests
pytest

# Run with coverage report
pytest --cov=v6_descriptive_statistics --cov-report=html

# Run specific test categories
pytest -m unit          # Unit tests only
pytest -m integration   # Integration tests only
pytest -m vantage6      # Vantage6 integration tests
pytest -m docker        # Docker tests

# Run with verbose output
pytest -v
```

### Test Categories

- **Unit Tests**: Test core algorithm functions and statistical computations
- **Integration Tests**: Test complete Vantage6 developer network workflow including:
  - Demo network setup and teardown
  - Docker container validation (3 nodes, server, algorithm store)
  - Algorithm build and deployment
  - Task execution and result validation
  - Network cleanup

### Vantage6 Integration Testing

The test suite implements a comprehensive Vantage6 integration workflow:

#### Vantage6 Developer Network Testing
```bash
# The integration tests automatically:
# 1. Set up vantage6 developer network
# 2. Verify Docker containers are spawned correctly
# 3. Build algorithm locally (without uploading)
# 4. Run tasks on the developer network
# 5. Assert results match expected central values
# 6. Clean up network after testing

# Run integration tests (requires Docker and vantage6 CLI)
pytest -m integration tests/test_vantage6_integration.py
```

#### Tested Workflow
1. **Network Setup**: `v6 dev create-demo-network` and `v6 dev start-demo-network`
2. **Container Validation**: Verify 3 nodes + server + algorithm store containers
3. **Local Build**: Build algorithm Docker image without repository upload
4. **Task Execution**: Run algorithm on developer network with test data
5. **Result Validation**: Compare federated results with expected central values
6. **Cleanup**: `v6 dev stop-demo-network` and `v6 dev remove-demo-network`

#### Docker Integration
- Algorithm Docker image build validation
- Container execution testing
- Multi-version compatibility testing

### Statistical Validation

Tests validate that federated computations produce mathematically equivalent results to centralised approaches (note: quantiles may differ due to federated nature):

```python
def test_federated_equals_centralised():
    # Split data across organizations
    federated_data = split_by_organisation(test_data)
    
    # Run federated algorithm via Vantage6
    federated_result = run_vantage6_task(federated_data)
    
    # Compute centralised result
    centralised_result = compute_centralised_stats(combined_data)
    
    # Validate equivalence within tolerance
    assert_statistics_equivalent(federated_result, centralised_result)
```

## Code Quality and Security

### Automated Quality Checks

All code changes are automatically validated through:

- **Black**: Code formatting consistency
- **Flake8**: PEP 8 compliance and code quality
- **MyPy**: Static type checking
- **Bandit**: Security vulnerability scanning
- **Safety**: Dependency vulnerability checking

### Coverage Requirements

- Minimum code coverage: 50%
- Coverage badge automatically updated on successful builds
- Detailed coverage reports available in CI artifacts

## Continuous Integration

### GitHub Actions Workflows

The repository includes comprehensive CI/CD workflows:

1. **Comprehensive Test Suite** (`test-suite.yml`): Runs all test categories
2. **Individual Quality Checks**: Separate workflows for Black, Flake8, MyPy, Bandit, Safety
3. **Docker Integration**: Validates Docker builds and container execution
4. **Vantage6 Integration**: Tests integration with Vantage6 infrastructure

### Branch Protection

- All checks must pass before merging
- Test failures block pull request merges
- Security vulnerabilities block deployments

### Dockerizing your algorithm

To finally run your algorithm on the vantage6 infrastructure, you need to
create a Docker image of your algorithm.

#### Automatically

The easiest way to create a Docker image is to use the GitHub Actions pipeline to
automatically build and push the Docker image. All that you need to do is push a
commit to the ``main`` branch.

#### Manually

A Docker image can be created by executing the following command in the root of your
algorithm directory:

```bash
docker build -t [my_docker_image_name] .
```

where you should provide a sensible value for the Docker image name. The
`docker build` command will create a Docker image that contains your algorithm.
You can create an additional tag for it by running

```bash
docker tag [my_docker_image_name] [another_image_name]
```

This way, you can e.g. do
`docker tag local_average_algorithm harbor2.vantage6.ai/algorithms/average` to
make the algorithm available on a remote Docker registry (in this case
`harbor2.vantage6.ai`).

Finally, you need to push the image to the Docker registry. This can be done
by running

```bash
docker push [my_docker_image_name]
```

Note that you need to be logged in to the Docker registry before you can push
the image. You can do this by running `docker login` and providing your
credentials. Check [this page](https://docs.docker.com/get-started/04_sharing_app/)
For more details on sharing images on Docker Hub. If you are using a different
Docker registry, check the documentation of that registry and be sure that you
have sufficient permissions.

## Development Setup

### Local Development

```bash
# Clone repository
git clone https://github.com/STRONGAYA/v6-descriptive-statistics.git
cd v6-descriptive-statistics

# Switch to development branch
git checkout revamped-version

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install pytest pytest-cov pytest-mock

# Run tests
pytest

# Run algorithm locally with mock client
python test/test.py
```

### Custom Infrastructure Setup

For integration testing with Vantage6 infrastructure:

1. **Demo Network Setup**:
   ```bash
   v6 dev create-demo-network
   v6 dev start-demo-network
   ```

2. **MDW Infrastructure**: See [mdw-nl/v6-infrastructure-sh](https://github.com/mdw-nl/v6-infrastructure-sh) for setup instructions

3. **Docker Testing**:
   ```bash
   docker build -t v6-descriptive-statistics:test .
   docker run --rm v6-descriptive-statistics:test
   ```

## Algorithm Usage

The algorithm provides descriptive statistics computation in a federated manner:

### Central Method

```python
# Run central aggregation task
central_task = client.task.create(
    input_={
        "method": "central",
        "kwargs": {
            "variables_to_describe": {
                "Gender": {"datatype": "categorical", "inliers": ("M", "F", "X")},
                "Age": {"datatype": "numerical", "inliers": (15, 39)}
            },
            "variables_to_stratify": None,
            "organization_ids": None,
        }
    },
    organizations=[org_id]
)
```

### Partial Method

```python
# Run partial computation task
partial_task = client.task.create(
    input_={
        "method": "partial",
        "kwargs": {
            "variables_to_describe": {
                "Gender": {"datatype": "categorical", "inliers": ("M", "F")},
                "Age": {"datatype": "numerical", "inliers": (15, 39)}
            },
            "variables_to_stratify": None,
        }
    },
    organizations=org_ids
)
```

## Contributing

When contributing new functionality:

1. **Add tests** for all new features (unit, integration, empirical as appropriate)
2. **Ensure all quality checks pass** (Black, Flake8, MyPy, Bandit, Safety)
3. **Maintain test coverage** above the minimum threshold
4. **Update documentation** for any new features or changes
5. **Test with both demo network and Docker** before submitting PR

### Testing Guidelines

- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- Test edge cases and error conditions
- Use realistic synthetic data
- Mock external dependencies appropriately
- Validate both structure and values of results

## Alignment with v6-tools-general

This testing framework is aligned with the patterns and standards established in [@STRONGAYA/v6-tools-general](https://github.com/STRONGAYA/v6-tools-general), focusing on:

- Vantage6 integration testing rather than library functionality testing
- Federated computation validation
- Production-grade CI/CD practices
- Comprehensive quality and security validation

The framework is designed to be modular and reusable for future Vantage6 algorithm projects.