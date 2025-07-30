"""
Test utilities and helper functions for v6-descriptive-statistics testing.
"""
import pandas as pd
import numpy as np
import json
from typing import Dict, Any, List, Union
from pathlib import Path

def create_synthetic_medical_data(n_samples: int = 100, seed: int = 42) -> pd.DataFrame:
    """
    Create synthetic medical research data for testing.
    
    Args:
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic medical data
    """
    np.random.seed(seed)
    
    # Generate synthetic data that mimics medical research datasets
    data = pd.DataFrame({
        "PatientID": [f"PAT_{i:06d}" for i in range(n_samples)],
        "Age": np.random.normal(45, 15, n_samples).astype(int).clip(18, 85),
        "Gender": np.random.choice(["M", "F"], n_samples, p=[0.48, 0.52]),
        "BMI": np.random.normal(25, 4, n_samples).clip(15, 40),
        "BloodPressure_Systolic": np.random.normal(125, 20, n_samples).astype(int).clip(90, 180),
        "BloodPressure_Diastolic": np.random.normal(80, 12, n_samples).astype(int).clip(60, 110),
        "Cholesterol": np.random.normal(200, 40, n_samples).clip(100, 400),
        "BloodSugar": np.random.normal(95, 15, n_samples).clip(70, 200),
        "Smoking_Status": np.random.choice(["Never", "Former", "Current"], n_samples, p=[0.5, 0.3, 0.2]),
        "Treatment_Group": np.random.choice(["Control", "Treatment_A", "Treatment_B"], n_samples, p=[0.4, 0.3, 0.3]),
        "Outcome_Score": np.random.normal(50, 10, n_samples).clip(0, 100)
    })
    
    return data

def create_edge_case_datasets() -> List[pd.DataFrame]:
    """
    Create various edge case datasets for testing.
    
    Returns:
        List of DataFrames with different edge cases
    """
    datasets = []
    
    # Very small dataset
    small_data = pd.DataFrame({
        "ID": ["P1", "P2"],
        "Age": [25, 30],
        "Gender": ["M", "F"],
        "Score": [10, 20]
    })
    datasets.append(small_data)
    
    # Dataset with missing values
    missing_data = pd.DataFrame({
        "ID": ["P1", "P2", "P3", "P4"],
        "Age": [25, None, 35, 40],
        "Gender": ["M", "F", None, "M"],
        "Score": [10, 20, 30, None]
    })
    datasets.append(missing_data)
    
    # Dataset with outliers
    outlier_data = pd.DataFrame({
        "ID": [f"P{i}" for i in range(10)],
        "Age": [25, 30, 35, 40, 45, 50, 55, 60, 999, 28],  # 999 is outlier
        "Gender": ["M", "F"] * 5,
        "Score": [10, 20, 30, 40, 50, 60, 70, 80, 90, -999]  # -999 is outlier
    })
    datasets.append(outlier_data)
    
    # Single-value dataset (no variance)
    constant_data = pd.DataFrame({
        "ID": [f"P{i}" for i in range(5)],
        "Age": [30] * 5,  # All same value
        "Gender": ["M"] * 5,  # All same value
        "Score": [50] * 5  # All same value
    })
    datasets.append(constant_data)
    
    return datasets

def split_data_by_organization(data: pd.DataFrame, n_orgs: int = 3, method: str = "random") -> List[pd.DataFrame]:
    """
    Split data across multiple organizations for federated testing.
    
    Args:
        data: DataFrame to split
        n_orgs: Number of organizations to split into
        method: Split method ("random", "sequential", "stratified")
        
    Returns:
        List of DataFrames, one per organization
    """
    if method == "random":
        # Random split
        indices = np.random.permutation(len(data))
        splits = np.array_split(indices, n_orgs)
        return [data.iloc[split].reset_index(drop=True) for split in splits]
    
    elif method == "sequential":
        # Sequential split
        splits = np.array_split(range(len(data)), n_orgs)
        return [data.iloc[split].reset_index(drop=True) for split in splits]
    
    elif method == "stratified":
        # Stratified split (if Gender column exists)
        if "Gender" in data.columns:
            org_data = []
            for i in range(n_orgs):
                org_subset = data[data.index % n_orgs == i].reset_index(drop=True)
                org_data.append(org_subset)
            return org_data
        else:
            # Fall back to random if no stratification variable
            return split_data_by_organization(data, n_orgs, "random")
    
    else:
        raise ValueError(f"Unknown split method: {method}")

def compute_federated_statistics(org_datasets: List[pd.DataFrame], variable: str) -> Dict[str, float]:
    """
    Compute federated statistics for a variable across organizations.
    
    Args:
        org_datasets: List of DataFrames from different organizations
        variable: Variable name to compute statistics for
        
    Returns:
        Dictionary of federated statistics
    """
    all_values = []
    org_stats = []
    
    for org_data in org_datasets:
        if variable in org_data.columns:
            org_values = org_data[variable].dropna()
            if len(org_values) > 0:
                all_values.extend(org_values.tolist())
                org_stats.append({
                    "count": len(org_values),
                    "sum": org_values.sum(),
                    "mean": org_values.mean(),
                    "min": org_values.min(),
                    "max": org_values.max(),
                    "var": org_values.var() if len(org_values) > 1 else 0
                })
    
    if not org_stats:
        return {}
    
    # Aggregate statistics
    total_count = sum(stat["count"] for stat in org_stats)
    total_sum = sum(stat["sum"] for stat in org_stats)
    federated_mean = total_sum / total_count if total_count > 0 else 0
    federated_min = min(stat["min"] for stat in org_stats)
    federated_max = max(stat["max"] for stat in org_stats)
    
    # Federated variance calculation
    if total_count > 1:
        weighted_var_sum = 0
        mean_diff_sum = 0
        for stat in org_stats:
            n = stat["count"]
            if n > 0:
                weighted_var_sum += (n - 1) * stat["var"]
                mean_diff_sum += n * (stat["mean"] - federated_mean) ** 2
        
        federated_var = (weighted_var_sum + mean_diff_sum) / (total_count - 1)
        federated_std = np.sqrt(federated_var)
    else:
        federated_var = 0
        federated_std = 0
    
    return {
        "count": total_count,
        "mean": federated_mean,
        "min": federated_min,
        "max": federated_max,
        "var": federated_var,
        "std": federated_std
    }

def compute_centralised_statistics(data: pd.DataFrame, variable: str) -> Dict[str, float]:
    """
    Compute centralised statistics for comparison with federated results.
    
    Args:
        data: Combined DataFrame
        variable: Variable name to compute statistics for
        
    Returns:
        Dictionary of centralised statistics
    """
    if variable not in data.columns:
        return {}
    
    values = data[variable].dropna()
    
    if len(values) == 0:
        return {}
    
    return {
        "count": len(values),
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "var": float(values.var()) if len(values) > 1 else 0.0,
        "std": float(values.std()) if len(values) > 1 else 0.0
    }

def assert_statistics_equivalent(federated_stats: Dict[str, float], 
                               centralised_stats: Dict[str, float], 
                               tolerance: float = 0.001,
                               relative_tolerance: float = 0.01) -> None:
    """
    Assert that federated and centralised statistics are equivalent within tolerance.
    
    Args:
        federated_stats: Federated statistics
        centralised_stats: Centralised statistics
        tolerance: Absolute tolerance for comparison
        relative_tolerance: Relative tolerance for comparison
    """
    assert set(federated_stats.keys()) == set(centralised_stats.keys()), \
        "Statistics should have same keys"
    
    for key in federated_stats:
        fed_val = federated_stats[key]
        cent_val = centralised_stats[key]
        
        # Use relative tolerance for larger values, absolute for smaller
        if abs(cent_val) > 1:
            assert abs(fed_val - cent_val) <= relative_tolerance * abs(cent_val), \
                f"Relative difference too large for {key}: {fed_val} vs {cent_val}"
        else:
            assert abs(fed_val - cent_val) <= tolerance, \
                f"Absolute difference too large for {key}: {fed_val} vs {cent_val}"

def validate_algorithm_result_structure(result: Any) -> None:
    """
    Validate that algorithm result has expected structure.
    
    Args:
        result: Algorithm result to validate
    """
    assert isinstance(result, dict), "Result should be a dictionary"
    
    # Check for expected top-level keys
    expected_keys = ["included_organisations", "excluded_organisations"]
    for key in expected_keys:
        assert key in result, f"Result should contain {key}"
        assert isinstance(result[key], list), f"{key} should be a list"

def create_test_configuration() -> Dict[str, Any]:
    """
    Create standard test configuration for algorithm testing.
    
    Returns:
        Dictionary with test configuration
    """
    return {
        "variables_to_describe": {
            "Age": {
                "datatype": "numerical",
                "inliers": (18, 85)
            },
            "Gender": {
                "datatype": "categorical", 
                "inliers": ("M", "F")
            },
            "BMI": {
                "datatype": "numerical",
                "inliers": (15, 40)
            }
        },
        "variables_to_stratify": None,
        "organization_ids": None
    }

def save_test_results(results: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """
    Save test results to file for analysis.
    
    Args:
        results: Test results to save
        filepath: Path to save results to
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2, default=str)

def load_test_results(filepath: Union[str, Path]) -> Dict[str, Any]:
    """
    Load test results from file.
    
    Args:
        filepath: Path to load results from
        
    Returns:
        Test results dictionary
    """
    with open(filepath, 'r') as f:
        return json.load(f)

class TestDataGenerator:
    """Generator for various test data scenarios."""
    
    @staticmethod
    def medical_trial_data(n_patients: int = 200, n_sites: int = 4) -> List[pd.DataFrame]:
        """Generate medical trial data across multiple sites."""
        site_datasets = []
        
        for site_id in range(n_sites):
            # Each site has different patient characteristics
            site_offset = site_id * 10  # Age offset per site
            gender_ratio = 0.5 + (site_id - n_sites/2) * 0.1  # Varying gender ratios
            
            n_site_patients = n_patients // n_sites + (1 if site_id < n_patients % n_sites else 0)
            
            site_data = pd.DataFrame({
                "PatientID": [f"Site{site_id}_P{i:04d}" for i in range(n_site_patients)],
                "SiteID": site_id,
                "Age": np.random.normal(45 + site_offset, 12, n_site_patients).astype(int).clip(18, 85),
                "Gender": np.random.choice(["M", "F"], n_site_patients, 
                                         p=[1-gender_ratio, gender_ratio]),
                "BaselineScore": np.random.normal(50 + site_id*2, 8, n_site_patients).clip(0, 100),
                "TreatmentArm": np.random.choice(["Placebo", "Treatment"], n_site_patients, p=[0.5, 0.5]),
                "OutcomeScore": np.random.normal(55 + site_id, 10, n_site_patients).clip(0, 100)
            })
            
            site_datasets.append(site_data)
        
        return site_datasets
    
    @staticmethod
    def epidemiological_study_data(n_regions: int = 3) -> List[pd.DataFrame]:
        """Generate epidemiological study data across regions."""
        regional_datasets = []
        
        region_names = ["North", "South", "Central"][:n_regions]
        
        for i, region in enumerate(region_names):
            # Different population characteristics per region
            n_participants = np.random.randint(150, 300)
            
            # Regional age and health patterns
            age_mean = 40 + i * 5
            disease_prevalence = 0.1 + i * 0.05
            
            regional_data = pd.DataFrame({
                "ParticipantID": [f"{region}_{j:05d}" for j in range(n_participants)],
                "Region": region,
                "Age": np.random.normal(age_mean, 15, n_participants).astype(int).clip(18, 90),
                "Gender": np.random.choice(["M", "F"], n_participants),
                "SocioeconomicStatus": np.random.choice(["Low", "Medium", "High"], n_participants, 
                                                      p=[0.3, 0.5, 0.2]),
                "HasDisease": np.random.choice([0, 1], n_participants, 
                                            p=[1-disease_prevalence, disease_prevalence]),
                "RiskScore": np.random.gamma(2, 2, n_participants).clip(0, 20)
            })
            
            regional_datasets.append(regional_data)
        
        return regional_datasets