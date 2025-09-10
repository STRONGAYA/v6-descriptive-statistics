# v6-descriptive-statistics

<p align="center">
<a href="https://github.com/STRONGAYA/v6-descriptive-statistics/workflows/"><img alt="Test status" src="https://github.com/STRONGAYA/v6-descriptive-statistics/workflows/Test%20Suite/badge.svg)"></a>
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

---

## What does this algorithm do?

Vantage6 algorithm that retrieves descriptive statistics.

The algorithm computes various descriptions in a federated manner whilst upholding basic privacy considerations.
The statistics that are computed include:

- Count
- Mean
- Minimum
- Maximum
- Percentiles/Quantiles (set to 25th, 50th, and 75th percentile by default)
- Frequency counts for categorical variables

A more detailed description of the algorithm and how to use it can be found in the algorithm's
[Wiki](https://github.com/STRONGAYA/v6-descriptive-statistics/wiki) and the `/docs` directory.

---
This algorithm is designed to be run with the [vantage6](https://vantage6.ai)
infrastructure for distributed analysis and learning.

The base code for this algorithm has been created via the
[v6-algorithm-template](https://github.com/vantage6/v6-algorithm-template)
template generator.

---

## Continuous Integration

### GitHub Actions Workflows

The repository includes comprehensive CI/CD workflows:

At every commit and pull request, the following workflows are executed:

1. **Comprehensive Test Suite** (`test-suite.yml`): Runs all test categories
2. **Individual Quality Checks**: Separate workflows for Black, Flake8, MyPy, Bandit, Safety
3. **Docker Integration**: Validates Docker builds and container execution
4. **Vantage6 Integration**: Tests integration with Vantage6 infrastructure

_This workflow verifies code quality, and security, and the workflow simultaneously performs integration tests for
various scenarios which the algorithm can be subjected to._

At releases and tags on the main branch, the following workflows are executed:

1. **Release Build** (`release.yml`): Builds and pushes Docker image to GitHub Container Registry
2. **Documentation Deployment** (`deploy-docs.yml`): Builds and deploys documentation to GitHub Pages

_This workflow ensures that new releases are properly built, documented, and made publicly available._

---

## Contributing

When contributing new functionality:

1. **Add tests** for all new features (unit, integration, empirical as appropriate)
2. **Ensure all quality checks pass** (Black, Flake8, MyPy, Bandit, Safety)
3. **Update documentation** for any new features or changes

### Testing Guidelines

- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- Test edge cases and error conditions
- Use realistic synthetic data and test-case scenarios
- Mock external dependencies appropriately
- Validate both structure and values of results

---

## Creating your own version?

If you want to create your own version of this algorithm,
you can do so by forking or cloning the repository and follow the following steps:

### Dockerising your algorithm

To finally run your algorithm on the vantage6 infrastructure, you need to
create a Docker image of your algorithm.

#### Automatically

The easiest way to create a Docker image is to use the GitHub Actions pipeline to
automatically build and push the Docker image.
All that you need to do is create a release or tag on the `main` branch.

#### Manually

A Docker image can be created by executing the following command in the root of your
algorithm directory:

```bash
docker build -t [my_docker_image_name] .
```

Here you should provide a sensible value for the Docker image name.
The `docker build` command will create a Docker image that contains your algorithm.
You can create an additional tag for it by running:

```bash
docker tag [my_docker_image_name] [another_image_name]
```

This way, you can e.g. do `docker tag local_average_algorithm harbor2.vantage6.ai/algorithms/average`
to make the algorithm available on a remote Docker registry (in this case `harbor2.vantage6.ai`).

Finally, you need to push the image to the Docker registry.
This can be done by running:

```bash
docker push [my_docker_image_name]
```

Note that you need to be logged in to the Docker registry before you can push
the image.
You can do this by running `docker login` and providing your
credentials. Check [this page](https://docs.docker.com/get-started/04_sharing_app/)
For more details on sharing images on Docker Hub. If you are using a different
Docker registry, check the documentation of that registry and be sure that you
have sufficient permissions.