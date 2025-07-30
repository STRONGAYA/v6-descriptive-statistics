#!/usr/bin/env python3
"""
Basic validation script for the testing framework.
This script validates the framework without requiring external dependencies.
"""
import os
import sys
from pathlib import Path

def validate_test_structure():
    """Validate that the test directory structure is correct."""
    repo_root = Path(__file__).parent
    
    # Check main test directories
    required_dirs = [
        "tests",
        "tests/unit", 
        "tests/integration",
        "tests/empirical",
        "tests/utils"
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        full_path = repo_root / dir_path
        if not full_path.exists():
            missing_dirs.append(dir_path)
    
    if missing_dirs:
        print(f"❌ Missing directories: {missing_dirs}")
        return False
    else:
        print("✅ All test directories exist")
        return True

def validate_test_files():
    """Validate that test files exist and have basic structure."""
    repo_root = Path(__file__).parent
    
    required_files = [
        "tests/conftest.py",
        "tests/unit/test_algorithm_functions.py",
        "tests/integration/test_algorithm_workflows.py", 
        "tests/integration/test_vantage6_integration.py",
        "tests/empirical/test_federated_equivalence.py",
        "tests/utils/test_helpers.py",
        "pytest.ini"
    ]
    
    missing_files = []
    for file_path in required_files:
        full_path = repo_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    else:
        print("✅ All test files exist")
        return True

def validate_ci_workflows():
    """Validate that CI workflow files exist."""
    repo_root = Path(__file__).parent
    
    required_workflows = [
        ".github/workflows/test-suite.yml",
        ".github/workflows/black.yml",
        ".github/workflows/flake8.yml", 
        ".github/workflows/mypy.yml",
        ".github/workflows/bandit.yml",
        ".github/workflows/safety.yml"
    ]
    
    missing_workflows = []
    for workflow_path in required_workflows:
        full_path = repo_root / workflow_path
        if not full_path.exists():
            missing_workflows.append(workflow_path)
    
    if missing_workflows:
        print(f"❌ Missing workflows: {missing_workflows}")
        return False
    else:
        print("✅ All CI workflows exist")
        return True

def validate_pytest_config():
    """Validate pytest configuration."""
    repo_root = Path(__file__).parent
    pytest_ini = repo_root / "pytest.ini"
    
    if not pytest_ini.exists():
        print("❌ pytest.ini not found")
        return False
    
    content = pytest_ini.read_text()
    
    # Check for essential configuration
    required_markers = ["unit:", "integration:", "empirical:", "docker:", "vantage6:", "edge_case:"]
    missing_markers = []
    
    for marker in required_markers:
        if marker not in content:
            missing_markers.append(marker)
    
    if missing_markers:
        print(f"❌ Missing pytest markers: {missing_markers}")
        return False
    else:
        print("✅ pytest.ini is properly configured")
        return True

def validate_algorithm_structure():
    """Validate algorithm module structure."""
    repo_root = Path(__file__).parent
    
    algorithm_files = [
        "v6-descriptive-statistics/__init__.py",
        "v6-descriptive-statistics/central.py",
        "v6-descriptive-statistics/partial.py",
        "v6-descriptive-statistics/miscellaneous.py"
    ]
    
    missing_files = []
    for file_path in algorithm_files:
        full_path = repo_root / file_path
        if not full_path.exists():
            missing_files.append(file_path)
    
    if missing_files:
        print(f"❌ Missing algorithm files: {missing_files}")
        return False
    else:
        print("✅ Algorithm module structure is correct")
        return True

def validate_requirements():
    """Validate requirements.txt structure."""
    repo_root = Path(__file__).parent
    requirements_file = repo_root / "requirements.txt"
    
    if not requirements_file.exists():
        print("❌ requirements.txt not found")
        return False
    
    content = requirements_file.read_text()
    
    # Check that v6-tools-rdf is NOT present (as requested in issue)
    if "v6-tools-rdf" in content:
        print("❌ v6-tools-rdf dependency should be removed")
        return False
    
    # Check for essential dependencies
    required_deps = ["vantage6-algorithm-tools", "pandas", "v6-tools-general"]
    missing_deps = []
    
    for dep in required_deps:
        if dep not in content:
            missing_deps.append(dep)
    
    if missing_deps:
        print(f"❌ Missing dependencies: {missing_deps}")
        return False
    else:
        print("✅ requirements.txt is properly configured")
        return True

def validate_gitignore():
    """Validate .gitignore includes test artifacts."""
    repo_root = Path(__file__).parent
    gitignore_file = repo_root / ".gitignore"
    
    if not gitignore_file.exists():
        print("❌ .gitignore not found")
        return False
    
    content = gitignore_file.read_text()
    
    # Check for test-related ignores
    test_ignores = ["htmlcov/", "coverage.json", ".pytest_cache/", "bandit-report.json", "safety-report.json"]
    missing_ignores = []
    
    for ignore in test_ignores:
        if ignore not in content:
            missing_ignores.append(ignore)
    
    if missing_ignores:
        print(f"❌ Missing .gitignore entries: {missing_ignores}")
        return False
    else:
        print("✅ .gitignore properly configured for testing")
        return True

def validate_readme():
    """Validate README has CI badges."""
    repo_root = Path(__file__).parent
    readme_file = repo_root / "README.md"
    
    if not readme_file.exists():
        print("❌ README.md not found")
        return False
    
    content = readme_file.read_text()
    
    # Check for CI badges
    required_badges = ["Test%20Suite", "Black%20Code", "Flake8", "MyPy", "Bandit", "Safety"]
    missing_badges = []
    
    for badge in required_badges:
        if badge not in content:
            missing_badges.append(badge)
    
    if missing_badges:
        print(f"❌ Missing README badges: {missing_badges}")
        return False
    else:
        print("✅ README has CI badges")
        return True

def main():
    """Run all validation checks."""
    print("🔍 Validating v6-descriptive-statistics testing framework...")
    print()
    
    validations = [
        validate_test_structure,
        validate_test_files,
        validate_ci_workflows,
        validate_pytest_config,
        validate_algorithm_structure,
        validate_requirements,
        validate_gitignore,
        validate_readme
    ]
    
    results = []
    for validation in validations:
        try:
            result = validation()
            results.append(result)
        except Exception as e:
            print(f"❌ Validation error: {e}")
            results.append(False)
    
    print()
    if all(results):
        print("🎉 All validations passed! Testing framework is properly set up.")
        print()
        print("Next steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Run tests: pytest")
        print("3. Check CI workflows trigger properly")
        print("4. Validate Vantage6 integration when network allows")
        return 0
    else:
        failed_count = sum(1 for r in results if not r)
        print(f"❌ {failed_count} validation(s) failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())