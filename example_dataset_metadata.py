#!/usr/bin/env python3
"""
Example usage of the expanded DatasetMetadata interface.

This demonstrates how to create and use comprehensive dataset metadata
with the LCM system.
"""

import sys
import os

# Add the project root to path for local imports
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ciaf.lcm.dataset_manager import DatasetMetadata, create_dataset_metadata_from_dataframe, DatasetSplit

def example_manual_metadata():
    """Example of manually creating comprehensive dataset metadata."""
    print("Creating manual dataset metadata...")
    
    metadata = DatasetMetadata(
        name="customer_churn_dataset",
        owner="data_science_team",
        license="MIT",
        schema_digest="abc123...",
        sampling_rules={"method": "stratified", "ratio": 0.8},
        version="2.1.0",
        content_root="merkle_root_xyz...",
        
        # Dataset structure
        num_samples=10000,
        num_features=15,
        feature_names=[
            "customer_id", "age", "gender", "tenure", "monthly_charges",
            "total_charges", "contract_type", "payment_method", 
            "internet_service", "online_security", "device_protection",
            "tech_support", "streaming_tv", "streaming_movies", "churn"
        ],
        target_column="churn",
        target_type="classification",
        
        # Domain information
        domain="customer_analytics",
        task_type="supervised",
        description="Customer churn prediction dataset with demographics and service usage",
        
        # Feature types
        feature_types={
            "customer_id": "categorical",
            "age": "numerical",
            "gender": "categorical", 
            "tenure": "numerical",
            "monthly_charges": "numerical",
            "total_charges": "numerical",
            "contract_type": "categorical",
            "payment_method": "categorical",
            "internet_service": "categorical",
            "online_security": "categorical",
            "device_protection": "categorical",
            "tech_support": "categorical",
            "streaming_tv": "categorical", 
            "streaming_movies": "categorical",
            "churn": "categorical"
        },
        
        # Data quality
        data_quality_score=0.95,
        missing_values={"total_charges": 11},
        duplicate_rows=0,
        
        # Bias considerations
        protected_attributes=["age", "gender"],
        known_biases=["Gender imbalance in churn rates"],
        
        # Compliance
        contains_pii=True,
        privacy_level="confidential",
        compliance_frameworks=["GDPR", "CCPA"],
    )
    
    # Add feature statistics
    metadata.add_feature_statistics("age", {
        "mean": 46.8,
        "std": 12.3,
        "min": 18,
        "max": 80,
        "median": 45
    })
    
    metadata.add_categorical_mapping("gender", {
        "Male": 5174,
        "Female": 4826
    })
    
    # Validate metadata
    issues = metadata.validate_metadata()
    if issues:
        print(f"Metadata validation issues: {issues}")
    else:
        print("Metadata validation passed!")
    
    # Get feature summary
    summary = metadata.get_feature_summary()
    print(f"Feature summary: {summary}")
    
    return metadata

def example_from_mock_dataframe():
    """Example using the helper function with a mock DataFrame."""
    print("\nCreating metadata from mock dataframe...")
    
    # Mock DataFrame class for demonstration
    class MockDataFrame:
        def __init__(self):
            self.shape = (1000, 5)
            self.columns = ["feature1", "feature2", "category", "target", "id"]
            self._dtypes = {
                "feature1": "float64",
                "feature2": "int64", 
                "category": "object",
                "target": "int64",
                "id": "object"
            }
            
        @property
        def dtypes(self):
            return self._dtypes
            
        def __getitem__(self, col):
            # Return a mock series
            return MockSeries(col, self._dtypes.get(col, "object"))
    
    class MockSeries:
        def __init__(self, name, dtype):
            self.name = name
            self.dtype = dtype
            
        def describe(self):
            return {"mean": 0.5, "std": 0.2, "min": 0.0, "max": 1.0}
            
        def value_counts(self):
            mock_counts = MockValueCounts()
            mock_counts.index = ["A", "B", "C"]
            return mock_counts
            
        def isnull(self):
            return MockSeries("null", "bool")
            
        def sum(self):
            return 5  # Mock missing count
            
        def nunique(self):
            return 3 if self.dtype == "object" else 100
    
    class MockValueCounts:
        def __init__(self):
            self.index = ["A", "B", "C"]
            self._values = [400, 350, 250]
            
        def __getitem__(self, idx):
            return self._values[idx] if isinstance(idx, int) else 100
            
        def __len__(self):
            return 3
        
        @property    
        def iloc(self):
            return self._values
            
        def head(self, n):
            return {"A": 400, "B": 350, "C": 250}
    
    # Create metadata from mock dataframe
    mock_df = MockDataFrame()
    
    try:
        metadata = create_dataset_metadata_from_dataframe(
            mock_df,
            name="synthetic_test_data",
            owner="test_user",
            license="Apache-2.0",
            target_column="target",
            domain="testing",
            description="Mock dataset for testing metadata creation",
            version="1.0.0",
            tags=["test", "synthetic", "demo"]
        )
        
        print(f"Created metadata for dataset: {metadata.name}")
        print(f"Features: {metadata.num_features}")
        print(f"Samples: {metadata.num_samples}")
        print(f"Feature types: {metadata.feature_types}")
        print(f"Data quality score: {metadata.data_quality_score}")
        
        return metadata
        
    except Exception as e:
        print(f"Error creating metadata from dataframe: {e}")
        return None

def main():
    """Run metadata examples."""
    print("=" * 60)
    print("CIAF LCM Dataset Metadata Examples")
    print("=" * 60)
    
    # Example 1: Manual metadata creation
    manual_metadata = example_manual_metadata()
    
    # Example 2: From mock dataframe
    df_metadata = example_from_mock_dataframe()
    
    print(f"\nCompleted metadata examples!")
    print(f"Manual metadata validation issues: {len(manual_metadata.validate_metadata())}")
    if df_metadata:
        print(f"DataFrame metadata validation issues: {len(df_metadata.validate_metadata())}")

if __name__ == "__main__":
    main()