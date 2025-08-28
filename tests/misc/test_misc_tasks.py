"""
Miscellaneous tests for project setup and documentation.

These tests check project structure and documentation but only provide notifications
rather than failing the test suite.
"""

import pytest
from pathlib import Path
from datetime import datetime


class TestProjectMiscellaneous:
    """Test project setup and documentation with notifications only."""

    def test_algorithm_wiki_exists(self):
        """Check whether the algorithm documentation has been created."""
        repo_root = Path(__file__).parent.parent.parent

        # Check for common documentation files
        doc_files = [
            repo_root / "docs" / repo_root.__getattribute__("name") / "usage.rst",
            repo_root / "README.md",
        ]

        found_docs = [doc for doc in doc_files if doc.exists()]

        if not found_docs:
            pytest.fail(
                "No documentation files found. Consider adding usage.rst or comprehensive README.md"
            )

        # TODO add implementation.rst check and assess whether functions are documented there as well
        # Check if usage.rst contains function definitions
        usage_rst = (
            repo_root / "docs" / repo_root.__getattribute__("name") / "usage.rst"
        )
        if usage_rst.exists():
            content = usage_rst.read_text()
            # Look for function references
            function_indicators = [
                "central",
                "partial_general_statistics",
                "partial_aggregate_adjusted_deviation",
            ]
            found_functions = [func for func in function_indicators if func in content]

            if found_functions:
                print(
                    f"✓ Documentation found with function references: {found_functions}"
                )
            else:
                print(
                    "⚠ Warning: usage.rst exists but may not contain function definitions"
                )
        else:
            print(
                "⚠ Note: usage.rst not found - consider adding algorithm usage documentation"
            )

    def test_algorithm_store_json_exists(self):
        """Check whether the algorithm store JSON file has been created."""
        repo_root = Path(__file__).parent.parent.parent
        algorithm_store_file = repo_root / "algorithm_store.json"

        if not algorithm_store_file.exists():
            print(
                "⚠ Warning: algorithm_store.json not found - this may be needed for algorithm store registration"
            )
            return

        try:
            import json

            with open(algorithm_store_file, "r") as f:
                store_config = json.load(f)

            # Check for required fields
            required_fields = ["name", "description", "image"]
            missing_fields = [
                field for field in required_fields if field not in store_config
            ]

            if missing_fields:
                print(
                    f"⚠ Warning: algorithm_store.json missing required fields: {missing_fields}"
                )
            else:
                print("✓ algorithm_store.json found and appears correctly specified")

        except json.JSONDecodeError:
            print("⚠ Warning: algorithm_store.json exists but contains invalid JSON")
        except Exception as e:
            print(f"⚠ Warning: Could not validate algorithm_store.json: {e}")

    def test_license_file_exists(self):
        """Check whether the license file has been created and contains correct information."""
        repo_root = Path(__file__).parent.parent.parent

        # Common license file names
        license_files = [
            repo_root / "LICENSE",
            repo_root / "LICENSE.txt",
            repo_root / "LICENSE.md",
            repo_root / "LICENCE",
            repo_root / "LICENCE.txt",
        ]

        found_license = None
        for license_file in license_files:
            if license_file.exists():
                found_license = license_file
                break

        if not found_license:
            print("⚠ Warning: No LICENCE file found - consider adding a license")
            return

        try:
            content = found_license.read_text()
            current_year = datetime.now().year

            # Check if current year is in the license
            if str(current_year) in content:
                print(f"✓ LICENCE file found with current year ({current_year})")
            else:
                # Check for any year in the license
                import re

                years = re.findall(r"\b(19|20)\d{2}\b", content)
                if years:
                    latest_year = max(int(year) for year in years)
                    if current_year - latest_year > 1:
                        print(
                            f"⚠ Warning: LICENCE file may need year update (found {latest_year}, current {current_year})"
                        )
                    else:
                        print("✓ LICENCE file found with recent year")
                else:
                    print("✓ LICENCE file found (no year detected)")

            # Check for license holder placeholder
            placeholders = [
                "[fullname]",
                "[name of copyright owner]",
                "COPYRIGHT_HOLDER",
                "<OWNER>",
            ]
            has_placeholder = any(
                placeholder in content for placeholder in placeholders
            )

            if has_placeholder:
                print(
                    "⚠ Warning: LICENCE file contains placeholder text that should be replaced with actual copyright holder"
                )
            else:
                print(
                    "✓ LICENCE file appears to have proper copyright holder information"
                )

        except Exception as e:
            print(f"⚠ Warning: Could not validate LICENCE file: {e}")
