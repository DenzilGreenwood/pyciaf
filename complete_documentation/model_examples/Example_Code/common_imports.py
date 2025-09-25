"""
Common import module for all CIAF examples
Handles imports with proper fallbacks for different CIAF configurations
"""

import sys
import os

# Add CIAF package to Python path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
ciaf_path = os.path.join(project_root, 'ciaf')
if os.path.exists(ciaf_path):
    sys.path.insert(0, project_root)

def get_ciaf_imports():
    """Get CIAF imports with proper fallbacks."""
    imports = {}
    
    try:
        # Try importing from the actual CIAF framework
        from ciaf.api.framework import CIAFFramework
        imports['CIAFFramework'] = CIAFFramework
        
        from ciaf.wrappers.model_wrapper import CIAFModelWrapper
        imports['CIAFModelWrapper'] = CIAFModelWrapper
        
        # Try importing LCM components
        try:
            from ciaf.lcm.model_manager import ModelArchitecture, TrainingEnvironment
            imports['ModelArchitecture'] = ModelArchitecture
            imports['TrainingEnvironment'] = TrainingEnvironment
        except ImportError:
            imports['ModelArchitecture'] = None
            imports['TrainingEnvironment'] = None
        
        # Try importing optional components
        try:
            from ciaf.compliance import BiasValidator, ComplianceValidator
            imports['BiasValidator'] = BiasValidator
            imports['ComplianceValidator'] = ComplianceValidator
        except ImportError:
            imports['BiasValidator'] = None
            imports['ComplianceValidator'] = None
            
        try:
            from ciaf.metadata_tags import create_text_tag, create_classification_tag, AIModelType
            imports['create_text_tag'] = create_text_tag
            imports['create_classification_tag'] = create_classification_tag
            imports['AIModelType'] = AIModelType
        except ImportError:
            def mock_create_text_tag(*args, **kwargs):
                import random
                return type('MockTag', (), {'tag_id': f"tag_{random.randint(1000, 9999)}"})()
            def mock_create_classification_tag(*args, **kwargs):
                import random
                return type('MockTag', (), {'tag_id': f"tag_{random.randint(1000, 9999)}"})()
            imports['create_text_tag'] = mock_create_text_tag
            imports['create_classification_tag'] = mock_create_classification_tag
            imports['AIModelType'] = type('AIModelType', (), {'LLM': 'llm', 'CLASSIFIER': 'classifier'})()
            
        try:
            from ciaf.uncertainty import CIAFUncertaintyQuantifier
            imports['CIAFUncertaintyQuantifier'] = CIAFUncertaintyQuantifier
        except ImportError:
            imports['CIAFUncertaintyQuantifier'] = None
            
        try:
            from ciaf.explainability import CIAFExplainer
            imports['CIAFExplainer'] = CIAFExplainer
        except ImportError:
            imports['CIAFExplainer'] = None
        
        imports['CIAF_AVAILABLE'] = True
        return imports
        
    except ImportError as e:
        print(f"⚠️ CIAF not available: {e}")
        print("Running in demo mode with mock implementations")
        
        # Provide mock implementations
        imports.update(create_mock_implementations())
        imports['CIAF_AVAILABLE'] = False
        return imports

def create_mock_implementations():
    """Create mock implementations for when CIAF is not available."""
    import numpy as np
    
    class MockCIAFFramework:
        def __init__(self, name): 
            self.name = name
            print(f"🎭 Mock CIAF Framework initialized: {name}")
        
        def create_dataset_anchor(self, dataset_id, dataset_metadata, master_password):
            return type('Anchor', (), {'dataset_id': dataset_id})()
        
        def create_provenance_capsules(self, dataset_id, data_items):
            return [f"capsule_{i}" for i in range(len(data_items))]
        
        def create_model_anchor(self, model_name, model_parameters, model_architecture, authorized_datasets, master_password):
            return {
                'model_name': model_name,
                'parameters_fingerprint': 'mock_param_hash_' + 'a'*32,
                'architecture_fingerprint': 'mock_arch_hash_' + 'b'*32
            }
        
        def train_model_with_audit(self, model_name, capsules, training_params, model_version, user_id, training_metadata):
            return type('Snapshot', (), {'snapshot_id': f"snapshot_{model_name}_{model_version}"})()
        
        def validate_training_integrity(self, snapshot):
            return True
        
        def get_complete_audit_trail(self, model_name):
            return {
                'verification': {
                    'total_datasets': 1,
                    'total_audit_records': 10,
                    'integrity_verified': True
                },
                'inference_connections': {
                    'total_receipts': 3
                }
            }
        
        def get_performance_metrics(self, model_name):
            return {'total_inferences': 3, 'avg_response_time': 2.3}
    
    class MockCIAFModelWrapper:
        def __init__(self, model, model_name, **kwargs):
            self.model = model
            self.model_name = model_name
            print(f"🎭 Mock CIAF Model Wrapper created for {model_name}")
        
        def predict(self, query, model_version=None):
            if hasattr(self.model, 'predict'):
                response = self.model.predict(query)
            elif hasattr(self.model, 'generate'):
                response = self.model.generate(query)
            else:
                response = f"Mock response for: {query}"
                
            receipt = type('Receipt', (), {
                'receipt_hash': 'mock_receipt_' + 'c'*32,
                'receipt_integrity': True
            })()
            return response, receipt
        
        def verify(self, receipt):
            return {'receipt_integrity': True}
    
    def mock_create_text_tag(*args, **kwargs):
        import random
        return type('MockTag', (), {'tag_id': f"tag_{random.randint(1000, 9999)}"})()
    
    def mock_create_classification_tag(*args, **kwargs):
        import random
        return type('MockTag', (), {'tag_id': f"tag_{random.randint(1000, 9999)}"})()
    
    return {
        'CIAFFramework': MockCIAFFramework,
        'CIAFModelWrapper': MockCIAFModelWrapper,
        'ModelArchitecture': None,
        'TrainingEnvironment': None,
        'BiasValidator': None,
        'ComplianceValidator': None,
        'create_text_tag': mock_create_text_tag,
        'create_classification_tag': mock_create_classification_tag,
        'AIModelType': type('AIModelType', (), {'LLM': 'llm', 'CLASSIFIER': 'classifier'})(),
        'CIAFUncertaintyQuantifier': None,
        'CIAFExplainer': None
    }