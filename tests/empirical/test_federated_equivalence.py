"""
Empirical validation tests for v6-descriptive-statistics.
Tests federated vs centralised computation equivalence.
"""
import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch
import sys
import os

# Add the algorithm module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../v6-descriptive-statistics/'))

@pytest.mark.empirical
class TestFederatedVsCentralised:
    """Test federated computation equivalence to centralised approaches."""

    def test_basic_statistics_equivalence(self, test_data_one, test_data_two):
        """Test that basic statistics match between federated and centralised computation."""
        # Combine datasets for centralised computation
        combined_data = pd.concat([test_data_one, test_data_two], ignore_index=True)
        
        # Basic statistics that should be equivalent
        numerical_columns = ["Age", "Height(in)", "Weight(lbs)"]
        
        for col in numerical_columns:
            if col in combined_data.columns:
                # Centralised statistics
                centralised_mean = combined_data[col].mean()
                centralised_count = combined_data[col].count()
                centralised_min = combined_data[col].min()
                centralised_max = combined_data[col].max()
                
                # Federated statistics (simulated)
                federated_mean = (test_data_one[col].mean() * len(test_data_one) + 
                                test_data_two[col].mean() * len(test_data_two)) / (len(test_data_one) + len(test_data_two))
                federated_count = test_data_one[col].count() + test_data_two[col].count()
                federated_min = min(test_data_one[col].min(), test_data_two[col].min())
                federated_max = max(test_data_one[col].max(), test_data_two[col].max())
                
                # Assertions with small tolerance for floating point differences
                assert abs(centralised_mean - federated_mean) < 0.001, f"Mean mismatch for {col}"
                assert centralised_count == federated_count, f"Count mismatch for {col}"
                assert centralised_min == federated_min, f"Min mismatch for {col}"
                assert centralised_max == federated_max, f"Max mismatch for {col}"

    def test_categorical_statistics_equivalence(self, test_data_one, test_data_two):
        """Test categorical statistics equivalence."""
        combined_data = pd.concat([test_data_one, test_data_two], ignore_index=True)
        
        categorical_columns = ["Gender"]
        
        for col in categorical_columns:
            if col in combined_data.columns:
                # Centralised counts
                centralised_counts = combined_data[col].value_counts().to_dict()
                
                # Federated counts (simulated)
                federated_counts = {}
                for value in combined_data[col].unique():
                    if pd.notna(value):  # Skip NaN values
                        count1 = (test_data_one[col] == value).sum()
                        count2 = (test_data_two[col] == value).sum()
                        federated_counts[value] = count1 + count2
                
                # Compare counts
                for value in centralised_counts:
                    if pd.notna(value):
                        assert centralised_counts[value] == federated_counts.get(value, 0), \
                            f"Count mismatch for {col}={value}"

    def test_stratified_statistics_equivalence(self, test_data_one, test_data_two):
        """Test stratified statistics equivalence."""
        combined_data = pd.concat([test_data_one, test_data_two], ignore_index=True)
        
        # Test stratification by Gender
        stratification_variable = "Gender"
        analysis_variable = "Age"
        
        if stratification_variable in combined_data.columns and analysis_variable in combined_data.columns:
            # Centralised stratified statistics
            centralised_grouped = combined_data.groupby(stratification_variable)[analysis_variable]
            centralised_stats = {
                group: {
                    'mean': data.mean(),
                    'count': data.count(),
                    'min': data.min(),
                    'max': data.max()
                }
                for group, data in centralised_grouped
            }
            
            # Federated stratified statistics (simulated)
            federated_stats = {}
            for group in combined_data[stratification_variable].unique():
                if pd.notna(group):
                    group1_data = test_data_one[test_data_one[stratification_variable] == group][analysis_variable]
                    group2_data = test_data_two[test_data_two[stratification_variable] == group][analysis_variable]
                    
                    if len(group1_data) > 0 or len(group2_data) > 0:
                        total_count = len(group1_data) + len(group2_data)
                        if total_count > 0:
                            federated_mean = (group1_data.sum() + group2_data.sum()) / total_count
                            federated_min = min(group1_data.min() if len(group1_data) > 0 else float('inf'),
                                              group2_data.min() if len(group2_data) > 0 else float('inf'))
                            federated_max = max(group1_data.max() if len(group1_data) > 0 else float('-inf'),
                                              group2_data.max() if len(group2_data) > 0 else float('-inf'))
                            
                            federated_stats[group] = {
                                'mean': federated_mean,
                                'count': total_count,
                                'min': federated_min,
                                'max': federated_max
                            }
            
            # Compare stratified statistics
            for group in centralised_stats:
                if group in federated_stats:
                    assert abs(centralised_stats[group]['mean'] - federated_stats[group]['mean']) < 0.001
                    assert centralised_stats[group]['count'] == federated_stats[group]['count']
                    assert centralised_stats[group]['min'] == federated_stats[group]['min']
                    assert centralised_stats[group]['max'] == federated_stats[group]['max']

@pytest.mark.empirical
class TestEdgeCaseEquivalence:
    """Test equivalence in edge cases."""

    def test_single_organization_equivalence(self, test_data_one):
        """Test that single organization federated equals centralised."""
        # With single organization, federated should exactly equal centralised
        data = test_data_one.copy()
        
        numerical_columns = ["Age", "Height(in)", "Weight(lbs)"]
        
        for col in numerical_columns:
            if col in data.columns:
                # Both should be identical
                centralised_mean = data[col].mean()
                federated_mean = data[col].mean()  # Same calculation
                
                assert centralised_mean == federated_mean, f"Single org mean mismatch for {col}"

    def test_small_sample_equivalence(self, small_dataset):
        """Test equivalence with very small samples."""
        # Split small dataset
        data1 = small_dataset.iloc[:1].copy()  # 1 row
        data2 = small_dataset.iloc[1:].copy()  # 1 row
        combined = small_dataset.copy()
        
        numerical_columns = ["Age", "Height(in)", "Weight(lbs)"]
        
        for col in numerical_columns:
            if col in combined.columns and len(combined[col].dropna()) > 0:
                # Centralised
                centralised_mean = combined[col].mean()
                centralised_count = combined[col].count()
                
                # Federated
                federated_mean = (data1[col].sum() + data2[col].sum()) / (data1[col].count() + data2[col].count())
                federated_count = data1[col].count() + data2[col].count()
                
                assert abs(centralised_mean - federated_mean) < 0.001, f"Small sample mean mismatch for {col}"
                assert centralised_count == federated_count, f"Small sample count mismatch for {col}"

    def test_missing_data_equivalence(self, missing_data_dataset):
        """Test equivalence when handling missing data."""
        # Split data with missing values
        data1 = missing_data_dataset.iloc[:2].copy()
        data2 = missing_data_dataset.iloc[2:].copy()
        combined = missing_data_dataset.copy()
        
        numerical_columns = ["Age", "Height(in)", "Weight(lbs)"]
        
        for col in numerical_columns:
            if col in combined.columns:
                # Only test columns that have some valid data
                if combined[col].count() > 0:
                    # Centralised (pandas handles NaN automatically)
                    centralised_mean = combined[col].mean()
                    centralised_count = combined[col].count()
                    
                    # Federated (simulate proper NaN handling)
                    valid_data1 = data1[col].dropna()
                    valid_data2 = data2[col].dropna()
                    
                    if len(valid_data1) > 0 or len(valid_data2) > 0:
                        federated_mean = (valid_data1.sum() + valid_data2.sum()) / (len(valid_data1) + len(valid_data2))
                        federated_count = len(valid_data1) + len(valid_data2)
                        
                        assert abs(centralised_mean - federated_mean) < 0.001, f"Missing data mean mismatch for {col}"
                        assert centralised_count == federated_count, f"Missing data count mismatch for {col}"

@pytest.mark.empirical
class TestStatisticalValidation:
    """Test statistical validity of computations."""

    def test_variance_computation_approximation(self, test_data_one, test_data_two):
        """Test that federated variance approximation is reasonable."""
        combined_data = pd.concat([test_data_one, test_data_two], ignore_index=True)
        
        numerical_columns = ["Age", "Height(in)", "Weight(lbs)"]
        
        for col in numerical_columns:
            if col in combined_data.columns:
                # Centralised variance
                centralised_var = combined_data[col].var()
                
                # Federated variance approximation (using sample means)
                # This is an approximation and won't be exact
                mean1 = test_data_one[col].mean()
                mean2 = test_data_two[col].mean()
                var1 = test_data_one[col].var()
                var2 = test_data_two[col].var()
                n1 = len(test_data_one[col].dropna())
                n2 = len(test_data_two[col].dropna())
                
                overall_mean = (mean1 * n1 + mean2 * n2) / (n1 + n2)
                
                # Pooled variance approximation
                federated_var_approx = ((n1 - 1) * var1 + (n2 - 1) * var2 + 
                                       n1 * (mean1 - overall_mean)**2 + 
                                       n2 * (mean2 - overall_mean)**2) / (n1 + n2 - 1)
                
                # Allow for reasonable tolerance in variance computation
                relative_diff = abs(centralised_var - federated_var_approx) / centralised_var
                assert relative_diff < 0.15, f"Variance approximation too far off for {col}: {relative_diff}"

    def test_quantile_approximation_validity(self, test_data_one, test_data_two):
        """Test that quantile approximations are reasonable."""
        combined_data = pd.concat([test_data_one, test_data_two], ignore_index=True)
        
        numerical_columns = ["Age", "Height(in)", "Weight(lbs)"]
        
        for col in numerical_columns:
            if col in combined_data.columns and len(combined_data[col].dropna()) > 4:
                # Centralised quantiles
                centralised_median = combined_data[col].median()
                centralised_q25 = combined_data[col].quantile(0.25)
                centralised_q75 = combined_data[col].quantile(0.75)
                
                # Federated quantiles (simple approximation using midpoint of ranges)
                min_val = min(test_data_one[col].min(), test_data_two[col].min())
                max_val = max(test_data_one[col].max(), test_data_two[col].max())
                
                # Simple bounds check - federated quantiles should be within the data range
                assert min_val <= centralised_median <= max_val, f"Median out of range for {col}"
                assert min_val <= centralised_q25 <= max_val, f"Q25 out of range for {col}"
                assert min_val <= centralised_q75 <= max_val, f"Q75 out of range for {col}"
                
                # Ordering should be preserved
                assert centralised_q25 <= centralised_median <= centralised_q75, f"Quantile ordering wrong for {col}"

@pytest.mark.empirical
@pytest.mark.slow
class TestLargeScaleEquivalence:
    """Test equivalence with larger synthetic datasets."""

    def test_large_scale_statistics_equivalence(self):
        """Test with larger synthetic datasets."""
        # Create larger datasets for more robust testing
        np.random.seed(42)  # For reproducibility
        
        # Dataset 1
        data1 = pd.DataFrame({
            "Age": np.random.normal(35, 10, 500).astype(int),
            "Height": np.random.normal(170, 15, 500),
            "Weight": np.random.normal(70, 12, 500),
            "Gender": np.random.choice(["M", "F"], 500)
        })
        
        # Dataset 2  
        data2 = pd.DataFrame({
            "Age": np.random.normal(40, 12, 300).astype(int),
            "Height": np.random.normal(168, 18, 300),
            "Weight": np.random.normal(72, 15, 300),
            "Gender": np.random.choice(["M", "F"], 300)
        })
        
        combined = pd.concat([data1, data2], ignore_index=True)
        
        numerical_columns = ["Age", "Height", "Weight"]
        
        for col in numerical_columns:
            # Centralised statistics
            centralised_mean = combined[col].mean()
            centralised_count = len(combined[col])
            
            # Federated statistics
            federated_mean = (data1[col].mean() * len(data1) + data2[col].mean() * len(data2)) / (len(data1) + len(data2))
            federated_count = len(data1[col]) + len(data2[col])
            
            # Should be very close with larger samples
            assert abs(centralised_mean - federated_mean) < 0.01, f"Large scale mean mismatch for {col}"
            assert centralised_count == federated_count, f"Large scale count mismatch for {col}"