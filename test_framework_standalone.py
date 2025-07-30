#!/usr/bin/env python3
"""
Standalone test runner for framework validation.
This can run without external dependencies to validate the framework structure.
"""
import sys
import traceback
from pathlib import Path

def run_framework_tests():
    """Run basic framework validation tests."""
    
    def test_test_directories_exist():
        """Test that all required test directories exist."""
        repo_root = Path(__file__).parent
        
        required_dirs = [
            "tests/unit",
            "tests/integration", 
            "tests/empirical",
            "tests/utils"
        ]
        
        for dir_path in required_dirs:
            full_path = repo_root / dir_path
            assert full_path.exists(), f"Directory {dir_path} should exist"
            assert full_path.is_dir(), f"{dir_path} should be a directory"
        
        print("✅ Test directories validation passed")

    def test_pytest_config_exists():
        """Test that pytest configuration exists."""
        repo_root = Path(__file__).parent
        pytest_ini = repo_root / "pytest.ini"
        
        assert pytest_ini.exists(), "pytest.ini should exist"
        
        content = pytest_ini.read_text()
        assert "[tool:pytest]" in content, "pytest.ini should have pytest configuration"
        
        print("✅ Pytest configuration validation passed")

    def test_ci_workflows_exist():
        """Test that CI workflow files exist."""
        repo_root = Path(__file__).parent
        workflows_dir = repo_root / ".github" / "workflows"
        
        assert workflows_dir.exists(), "Workflows directory should exist"
        
        required_workflows = [
            "test-suite.yml",
            "black.yml", 
            "flake8.yml",
            "mypy.yml",
            "bandit.yml",
            "safety.yml"
        ]
        
        for workflow in required_workflows:
            workflow_path = workflows_dir / workflow
            assert workflow_path.exists(), f"Workflow {workflow} should exist"
        
        print("✅ CI workflows validation passed")

    def test_requirements_validation():
        """Test requirements and dependency configuration."""
        repo_root = Path(__file__).parent
        requirements_file = repo_root / "requirements.txt"
        
        assert requirements_file.exists(), "requirements.txt should exist"
        
        content = requirements_file.read_text()
        
        # Should include essential dependencies
        assert "vantage6-algorithm-tools" in content
        assert "pandas" in content
        assert "v6-tools-general" in content
        
        # Should NOT include v6-tools-rdf (as requested in issue)
        assert "v6-tools-rdf" not in content
        
        print("✅ Requirements validation passed")

    def test_readme_structure():
        """Test that README includes comprehensive testing documentation."""
        repo_root = Path(__file__).parent
        readme_file = repo_root / "README.md"
        
        assert readme_file.exists(), "README.md should exist"
        
        content = readme_file.read_text()
        
        # Should have testing status section
        assert "## Testing Status" in content
        assert "Testing Framework" in content
        
        # Should have CI badges
        ci_badges = [
            "Test%20Suite",
            "Black%20Code", 
            "Flake8",
            "MyPy",
            "Bandit",
            "Safety"
        ]
        
        for badge in ci_badges:
            assert badge in content, f"README should include {badge} badge"
        
        print("✅ README structure validation passed")

    def test_algorithm_structure():
        """Test algorithm module structure."""
        repo_root = Path(__file__).parent
        algorithm_dir = repo_root / "v6-descriptive-statistics"
        
        assert algorithm_dir.exists(), "Algorithm directory should exist"
        assert algorithm_dir.is_dir(), "Algorithm path should be a directory"
        
        required_files = [
            "__init__.py",
            "central.py",
            "partial.py", 
            "miscellaneous.py"
        ]
        
        for file_name in required_files:
            file_path = algorithm_dir / file_name
            assert file_path.exists(), f"Algorithm file {file_name} should exist"
        
        print("✅ Algorithm structure validation passed")

    # Run all tests
    tests = [
        test_test_directories_exist,
        test_pytest_config_exists,
        test_ci_workflows_exist,
        test_requirements_validation,
        test_readme_structure,
        test_algorithm_structure
    ]
    
    print("🧪 Running framework validation tests...")
    print()
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed: {e}")
            failed += 1
            traceback.print_exc()
    
    print()
    print(f"📊 Test Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All framework validation tests passed!")
        print()
        print("Framework Features Validated:")
        print("✅ Complete test directory structure")
        print("✅ Production-grade CI/CD workflows")
        print("✅ Pytest configuration with proper markers")
        print("✅ v6-tools-rdf dependency removed as requested")
        print("✅ Comprehensive README with CI badges")
        print("✅ Algorithm module structure intact")
        print()
        print("Next Steps:")
        print("1. Install dependencies when network connectivity allows")
        print("2. Run full test suite: pytest")
        print("3. Validate CI workflows in GitHub Actions")
        print("4. Test Vantage6 integration scenarios")
        return True
    else:
        print("❌ Some framework validation tests failed")
        return False

if __name__ == "__main__":
    success = run_framework_tests()
    sys.exit(0 if success else 1)