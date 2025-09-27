"""
CIAF Data Utilities
Standardized data handling for CIAF integration
Following naming improvement recommendations from CIAF_LCM_IMPROVEMENT_RECOMMENDATIONS.md
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import logging
import json

logger = logging.getLogger(__name__)


class CIAFDataUtils:
    """Simplified data utilities for CIAF integration"""
    
    @staticmethod
    def to_ciaf_format(X: pd.DataFrame, y: Optional[pd.Series] = None) -> List[Dict[str, Any]]:
        """
        Convert data to CIAF format (simplified naming)
        Previously: convert_to_ciaf_format_with_full_validation()
        """
        try:
            data_records = []
            
            for i in range(len(X)):
                # Format expected by CIAFModelWrapper:
                # Each item needs 'content' and 'metadata' fields
                record = {
                    'content': X.iloc[i].to_dict(),  # The actual feature data
                    'metadata': {
                        'id': str(i),  # Required: unique identifier
                        'index': i,
                        'feature_names': list(X.columns)
                    }
                }
                
                # Add target to metadata if provided
                if y is not None:
                    record['metadata']['target'] = float(y.iloc[i])
                
                data_records.append(record)
            
            logger.info(f"✅ Converted {len(data_records)} samples to CIAF format")
            return data_records
            
        except Exception as e:
            logger.error(f"❌ Data conversion failed: {e}")
            return []
    
    @staticmethod
    def validate(data: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """
        Validate CIAF data format (simplified naming)
        Previously: validate_ciaf_data_format_with_comprehensive_checks()
        """
        errors = []
        
        if not isinstance(data, list):
            errors.append("Data must be a list")
            return False, errors
        
        if not data:
            errors.append("Data cannot be empty")
            return False, errors
        
        for i, record in enumerate(data):
            if not isinstance(record, dict):
                errors.append(f"Record {i} must be a dictionary")
                continue
                
            if 'content' not in record:
                errors.append(f"Record {i} missing 'content' field")
                
            if 'metadata' not in record:
                errors.append(f"Record {i} missing 'metadata' field")
            elif not isinstance(record['metadata'], dict):
                errors.append(f"Record {i} 'metadata' must be a dictionary")
            elif 'id' not in record['metadata']:
                errors.append(f"Record {i} metadata missing 'id' field")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    @staticmethod
    def from_ciaf_format(data: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Convert from CIAF format back to pandas
        Previously: extract_dataframes_from_ciaf_provenance_capsules()
        """
        try:
            features_list = []
            targets_list = []
            
            for record in data:
                # Extract content (features)
                features_list.append(record['content'])
                
                # Extract target from metadata if available
                if 'metadata' in record and 'target' in record['metadata']:
                    targets_list.append(record['metadata']['target'])
            
            X = pd.DataFrame(features_list)
            y = pd.Series(targets_list) if targets_list else None
            
            return X, y
            
        except Exception as e:
            logger.error(f"❌ CIAF format conversion failed: {e}")
            return pd.DataFrame(), None
    
    @staticmethod
    def get_schema(data: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Get data schema (simplified naming)
        Previously: generate_comprehensive_ciaf_data_schema_definition()
        """
        if not data:
            return {}
        
        sample_record = data[0]
        schema = {}
        
        if 'content' in sample_record:
            for key, value in sample_record['content'].items():
                schema[f"feature_{key}"] = type(value).__name__
        
        if 'metadata' in sample_record and 'target' in sample_record['metadata']:
            schema['target'] = type(sample_record['metadata']['target']).__name__
        
        return schema