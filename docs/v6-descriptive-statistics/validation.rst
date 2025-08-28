Validation
==========
Empirical testing
====================
The mathematics used in this algorithm originate from the [STRONG AYA v6-tools-general library](https://github.com/STRONGAYA/v6-tools-general).
Empirical validation and testing of these functions is performed in this library's repository.

Integration testing
====================
The integration testing in the Vantage6 infrastructure and is performed through continuous integration tests using GitHub Actions.
At every commit and pull request, the following workflow is executed for this:

1. **Comprehensive Test Suite** (`test-suite.yml`): Runs all test categories
2. **Individual Quality Checks**: Separate workflows for Black, Flake8, MyPy, Bandit, Safety
3. **Docker Integration**: Validates Docker builds and container execution
4. **Vantage6 Integration**: Tests integration with Vantage6 infrastructure

_This workflow therewith verifies code quality, and security, and the workflow simultaneously performs integration tests for
various scenarios which the algorithm can be subjected to._