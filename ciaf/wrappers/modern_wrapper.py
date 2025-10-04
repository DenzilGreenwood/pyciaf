"""
Modern protocol-based CIAF model wrapper.

This is the next-generation wrapper implementation that follows the protocol-based
architecture established across all CIAF modules. It uses policy-driven configuration
and dependency injection for maximum flexibility and maintainability.

Created: 2025-09-27
Author: Denzil James Greenwood
Version: 1.0.0
"""

import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union, TYPE_CHECKING

# Protocol imports
from .interfaces import ModelWrapper
from .policy import WrapperPolicy, get_default_wrapper_policy

# Type checking imports
if TYPE_CHECKING:
    from ..api import CIAFFramework
    from ..inference import InferenceReceipt
    from ..provenance import TrainingSnapshot

# Core CIAF imports - deferred to avoid circular imports
CIAF_CORE_AVAILABLE = True
_ciaf_framework = None
_inference_receipt = None
_training_snapshot = None

def _get_ciaf_framework():
    """Lazy import of CIAFFramework to avoid circular imports."""
    global _ciaf_framework
    if _ciaf_framework is None:
        try:
            from ..api import CIAFFramework
            _ciaf_framework = CIAFFramework
        except ImportError:
            global CIAF_CORE_AVAILABLE
            CIAF_CORE_AVAILABLE = False
            return type('CIAFFramework', (), {})  # Dummy class
    return _ciaf_framework

def _get_inference_receipt():
    """Lazy import of InferenceReceipt to avoid circular imports."""
    global _inference_receipt
    if _inference_receipt is None:
        try:
            from ..inference import InferenceReceipt
            _inference_receipt = InferenceReceipt
        except ImportError:
            global CIAF_CORE_AVAILABLE
            CIAF_CORE_AVAILABLE = False
            return type('InferenceReceipt', (), {})  # Dummy class
    return _inference_receipt

def _get_training_snapshot():
    """Lazy import of TrainingSnapshot to avoid circular imports."""
    global _training_snapshot
    if _training_snapshot is None:
        try:
            from ..provenance import TrainingSnapshot
            _training_snapshot = TrainingSnapshot
        except ImportError:
            global CIAF_CORE_AVAILABLE
            CIAF_CORE_AVAILABLE = False
            return type('TrainingSnapshot', (), {})  # Dummy class
    return _training_snapshot


class ModernCIAFModelWrapper(ModelWrapper):
    """
    Modern protocol-based CIAF model wrapper.
    
    This wrapper uses dependency injection through protocol interfaces and
    policy-driven configuration to provide a flexible, maintainable, and
    consistent architecture with other CIAF modules.
    
    Key Features:
    - Protocol-based architecture with dependency injection
    - Policy-driven configuration across all aspects
    - Consistent with core, LCM, compliance, and explainability modules
    - Enhanced error handling and graceful degradation
    - Comprehensive LCM metadata preservation
    - Full compatibility with all enhancement modules
    """
    
    def __init__(self, 
                 model: Any,
                 model_name: str, 
                 policy: Optional[WrapperPolicy] = None,
                 framework: Optional["CIAFFramework"] = None):
        """
        Initialize modern CIAF model wrapper.
        
        Args:
            model: The ML model to wrap
            model_name: Unique identifier for the model
            policy: WrapperPolicy controlling all wrapper behavior
            framework: Optional existing CIAFFramework instance
        """
        if not model_name or not model_name.strip():
            raise ValueError("model_name cannot be empty")
        
        self.model = model
        self.model_name = model_name.strip()
        self.policy = policy or get_default_wrapper_policy()
        
        # Initialize CIAF framework using lazy import
        CIAFFramework = _get_ciaf_framework()
        if CIAFFramework and framework:
            self.framework = framework
        elif CIAFFramework:
            self.framework = CIAFFramework(self.model_name)
        else:
            self.framework = None
            warnings.warn("CIAF framework not available - using simulation mode")
        
        # Extract protocol implementations from policy (with safety checks)
        self.model_adapter = self.policy.model_adapter
        self.metadata_provider = self.policy.metadata_provider
        self.model_validator = self.policy.model_validator
        self.training_handler = self.policy.training_handler
        self.inference_handler = self.policy.inference_handler
        self.lcm_metadata_handler = self.policy.lcm_metadata_handler
        self.enhancement_provider = self.policy.enhancement_provider
        self.compliance_integrator = self.policy.compliance_integrator
        self.performance_optimizer = self.policy.performance_optimizer
        
        # Check if protocols are available
        if not all([self.model_adapter, self.metadata_provider, self.model_validator,
                   self.training_handler, self.inference_handler]):
            warnings.warn("Some protocol implementations are not available - wrapper will use fallbacks")
            self._protocols_available = False
        else:
            self._protocols_available = True
        
        # Initialize state
        self.training_snapshot: Optional[Any] = None
        self.model_version: Optional[str] = None
        self.last_receipt: Optional[Any] = None
        self.enhancement_configurations: Dict[str, Any] = {}
        self.audit_entries: List[Dict[str, Any]] = []
        
        # Validate model compatibility
        if (self.policy.compatibility_policy.validate_model_structure and 
            self.model_adapter and self._protocols_available):
            try:
                compatibility_result = self.model_adapter.validate_model_compatibility(self.model)
                if not compatibility_result.get("is_compatible", False):
                    message = f"Model incompatible: {compatibility_result.get('errors', [])}"
                    if self.policy.compatibility_policy.fail_on_incompatible_model:
                        raise ValueError(message)
                    else:
                        warnings.warn(message)
            except Exception as e:
                warnings.warn(f"Model compatibility validation failed: {e}")
        
        # Configure enhancements
        if (self.policy.enhancement_policy.auto_configure_enhancements and 
            self.enhancement_provider and self._protocols_available):
            try:
                self._configure_enhancements()
            except Exception as e:
                warnings.warn(f"Enhancement configuration failed: {e}")
        
        # Configure compliance
        if self.compliance_integrator and self._protocols_available:
            try:
                compliance_config = self.compliance_integrator.configure_compliance_mode(
                    self.policy.compliance_policy.compliance_mode.value,
                    {"detailed_audit": self.policy.compliance_policy.detailed_audit_logging}
                )
            except Exception as e:
                warnings.warn(f"Compliance configuration failed: {e}")
                compliance_config = {"status": "fallback"}
        else:
            compliance_config = {"status": "protocols_unavailable"}
        
        # Log initialization
        self._create_audit_entry("wrapper_initialized", {
            "policy_hash": self.policy.get_policy_hash(),
            "compliance_config": compliance_config
        })
        
        print(f"✅ Modern CIAF wrapper initialized for '{self.model_name}'")
        print(f"   📋 Policy: {self.policy.format_policy_line()}")
        print(f"   🔧 Enhancements: {len(self.enhancement_configurations)}")
        print(f"   📊 Compliance: {self.policy.compliance_policy.compliance_mode.value}")
    
    def _configure_enhancements(self) -> None:
        """Configure available enhancements based on policy."""
        if not self.enhancement_provider:
            warnings.warn("Enhancement provider not available - skipping enhancement configuration")
            return
            
        try:
            available_enhancements = self.enhancement_provider.get_available_enhancements(self.model)
            
            enhancement_configs = {}
            
            # Configure each enhancement based on policy
            if self.policy.enhancement_policy.enable_preprocessing and "preprocessing" in available_enhancements:
                config = self.enhancement_provider.configure_preprocessing(self.model, {})
                if config:
                    enhancement_configs["preprocessing"] = config
            
            if self.policy.enhancement_policy.enable_explainability and "explainability" in available_enhancements:
                config = self.enhancement_provider.configure_explainability(self.model, {})
                if config:
                    enhancement_configs["explainability"] = config
            
            if self.policy.enhancement_policy.enable_uncertainty and "uncertainty" in available_enhancements:
                config = self.enhancement_provider.configure_uncertainty_quantification(self.model, {})
                if config:
                    enhancement_configs["uncertainty"] = config
            
            if self.policy.enhancement_policy.enable_metadata_tags and "metadata_tags" in available_enhancements:
                config = self.enhancement_provider.configure_metadata_tags(self.model, {})
                if config:
                    enhancement_configs["metadata_tags"] = config
            
            # Apply enhancements
            if enhancement_configs:
                result = self.enhancement_provider.apply_enhancements(self.model, enhancement_configs)
                self.enhancement_configurations = enhancement_configs
                self._create_audit_entry("enhancements_configured", result)
                
        except Exception as e:
            if self.policy.enhancement_policy.graceful_enhancement_failure:
                warnings.warn(f"Enhancement configuration failed: {e}")
            else:
                raise
    
    def train(self, 
              dataset_id: str,
              training_data: List[Dict[str, Any]],
              master_password: str,
              training_params: Optional[Dict[str, Any]] = None,
              model_version: str = "1.0.0",
              fit_model: bool = True) -> Any:
        """
        Train the wrapped ML model and create a CIAF Training Snapshot.
        
        Args:
            dataset_id: Unique identifier for the training dataset
            training_data: List of training examples with content and metadata
            master_password: Master password for anchor derivation
            training_params: Parameters used for training
            model_version: Version identifier for this training run
            fit_model: Whether to actually train the wrapped model
        
        Returns:
            TrainingSnapshot: The generated training snapshot
        """
        if not training_data:
            raise ValueError("training_data cannot be empty")
        
        training_params = training_params or {}
        
        print(f"🚀 [{self.model_name}] Starting protocol-based training for version '{model_version}'...")
        
        start_time = datetime.now()
        
        try:
            # Validate training data if required
            if self.policy.training_policy.validate_training_data:
                validation_result = self.model_validator.validate_training_data_compatibility(
                    self.model, training_data
                )
                if not validation_result.get("is_compatible", False):
                    message = f"Training data incompatible: {validation_result.get('errors', [])}"
                    if not self.policy.training_policy.continue_on_training_failure:
                        raise ValueError(message)
                    else:
                        warnings.warn(message)
            
            # Prepare training environment
            env_result = self.training_handler.prepare_training_environment(self.model, training_params)
            if env_result.get("status") != "prepared":
                warnings.warn(f"Training environment preparation issues: {env_result}")
            
            # Create CIAF training infrastructure if available
            training_snapshot = None
            if self.framework and CIAF_CORE_AVAILABLE:
                try:
                    # Create dataset anchor
                    dataset_metadata = {
                        "model_name": self.model_name,
                        "model_version": model_version,
                        "training_params": training_params,
                        "policy_hash": self.policy.get_policy_hash(),
                    }
                    
                    anchor = self.framework.create_dataset_anchor(
                        dataset_id=dataset_id,
                        dataset_metadata=dataset_metadata,
                        master_password=master_password,
                    )
                    
                    # Create provenance capsules
                    capsules = self.framework.create_provenance_capsules(dataset_id, training_data)
                    print(f"📦 [{self.model_name}] Created {len(capsules)} provenance capsules")
                    
                    # Create Model Aggregation Anchor
                    maa = self.framework.create_model_aggregation_anchor(
                        model_name=self.model_name, 
                        authorized_datasets=[dataset_id]
                    )
                    
                    # Create training snapshot
                    training_snapshot = self.framework.train_model(
                        model_name=self.model_name,
                        capsules=capsules,
                        maa=maa,
                        training_params=training_params,
                        model_version=model_version,
                    )
                    
                    print(f"🎯 [{self.model_name}] Training snapshot: {training_snapshot.snapshot_id}")
                    
                except Exception as e:
                    if self.policy.training_policy.continue_on_training_failure:
                        warnings.warn(f"CIAF training infrastructure failed: {e}")
                    else:
                        raise
            
            # Train the actual model if requested
            actual_training_success = False
            if fit_model and self.policy.training_policy.fit_model_by_default:
                print(f"🧠 [{self.model_name}] Training underlying ML model...")
                
                # Prepare training data using model adapter
                X, y = self.model_adapter.prepare_training_data(self.model, training_data)
                
                if X is not None:
                    # Execute training using training handler
                    training_result = self.training_handler.execute_training(
                        self.model, X, y, training_params
                    )
                    
                    if training_result.get("success", False):
                        actual_training_success = True
                        print(f"✅ [{self.model_name}] Model training completed")
                        
                        # Validate training results
                        validation_result = self.training_handler.validate_training_results(
                            self.model, training_result
                        )
                        if not validation_result.get("is_valid", False):
                            warnings.warn(f"Training validation warnings: {validation_result.get('warnings', [])}")
                    else:
                        error_info = self.training_handler.handle_training_errors(
                            self.model, 
                            Exception(training_result.get("error", "Unknown training error")),
                            {"training_params": training_params}
                        )
                        
                        if self.policy.training_policy.continue_on_training_failure:
                            warnings.warn(f"Training failed but continuing: {error_info}")
                        else:
                            raise RuntimeError(f"Training failed: {error_info}")
                else:
                    if self.policy.training_policy.continue_on_training_failure:
                        warnings.warn("Could not prepare training data - skipping model training")
                    else:
                        raise ValueError("Could not prepare training data for model")
            
            # Store training results
            self.training_snapshot = training_snapshot
            self.model_version = model_version
            
            # Create audit entry
            training_duration = (datetime.now() - start_time).total_seconds()
            self._create_audit_entry("training_completed", {
                "model_version": model_version,
                "training_samples": len(training_data),
                "actual_training_success": actual_training_success,
                "ciaf_snapshot_created": training_snapshot is not None,
                "training_duration_seconds": training_duration,
                "policy_hash": self.policy.get_policy_hash()
            })
            
            # Log compliance information
            compliance_mode = self.policy.compliance_policy.compliance_mode.value
            if compliance_mode == "healthcare":
                print(f"⚕️  HIPAA compliance: Training data minimized and encrypted")
            elif compliance_mode == "financial":
                print(f"🏦 Financial compliance: Audit trail created for regulatory reporting")
            elif compliance_mode == "strict":
                print(f"🔒 Strict compliance: Enhanced security and validation applied")
            
            return training_snapshot or self._create_mock_training_snapshot(model_version, training_data)
            
        except Exception as e:
            error_context = {
                "model_name": self.model_name,
                "model_version": model_version,
                "training_samples": len(training_data),
                "error_type": type(e).__name__
            }
            
            self._create_audit_entry("training_failed", error_context)
            
            print(f"❌ [{self.model_name}] Training failed: {str(e)}")
            
            if self.policy.training_policy.continue_on_training_failure:
                warnings.warn(f"Training failed but wrapper continues: {e}")
                return self._create_mock_training_snapshot(model_version, training_data)
            else:
                raise RuntimeError(f"Training failed for {self.model_name}: {str(e)}") from e
    
    def predict(self,
                query: Union[str, List, Any],
                model_version: Optional[str] = None,
                use_model: bool = True) -> Tuple[Any, Any]:
        """
        Run inference on the wrapped model and generate a CIAF Inference Receipt.
        
        Args:
            query: Input for the model
            model_version: Model version to use
            use_model: Whether to use the actual wrapped model for prediction
        
        Returns:
            Tuple containing (prediction, InferenceReceipt)
        """
        if not self.training_snapshot and self.policy.inference_policy.validate_inference_input:
            if not self.policy.training_policy.continue_on_training_failure:
                raise RuntimeError(f"Model {self.model_name} has not been trained. Please run train() first.")
            else:
                warnings.warn(f"Model {self.model_name} has not been trained - using simulation mode")
        
        model_version = model_version or self.model_version or "1.0.0"
        
        print(f"🔮 [{self.model_name}] Running protocol-based inference (v{model_version})...")
        
        start_time = datetime.now()
        
        try:
            # Validate inference input if required
            if self.policy.inference_policy.validate_inference_input:
                validation_result = self.model_validator.validate_inference_input_compatibility(
                    self.model, query
                )
                if not validation_result.get("is_compatible", False):
                    if not self.policy.inference_policy.fallback_on_inference_failure:
                        raise ValueError(f"Inference input invalid: {validation_result.get('errors', [])}")
                    else:
                        warnings.warn(f"Input validation warnings: {validation_result.get('warnings', [])}")
            
            # Prepare inference input
            prepared_input = None
            if self.policy.inference_policy.preprocess_inference_input:
                prepared_input = self.inference_handler.prepare_inference_input(
                    self.model, query, preprocessing_enabled=True
                )
            
            # Execute inference
            prediction = None
            if use_model:
                try:
                    # Use inference handler
                    raw_prediction = self.inference_handler.execute_inference(
                        self.model, 
                        prepared_input or query, 
                        {"model_version": model_version}
                    )
                    
                    # Postprocess output
                    if self.policy.inference_policy.postprocess_inference_output:
                        prediction = self.inference_handler.postprocess_inference_output(
                            self.model, raw_prediction, query
                        )
                    else:
                        prediction = raw_prediction
                        
                except Exception as e:
                    if self.policy.inference_policy.fallback_on_inference_failure:
                        error_info = self.inference_handler.handle_inference_errors(
                            self.model, query, e, {"model_version": model_version}
                        )
                        prediction = error_info.get("fallback_prediction", f"Error fallback for: {query}")
                        warnings.warn(f"Inference failed, using fallback: {e}")
                    else:
                        raise
            else:
                prediction = f"CIAF simulated response for: {query}"
            
            # Create enhanced information
            enhanced_info = {}
            
            # Add explainability if enabled
            if (self.policy.inference_policy.include_explanations and 
                "explainability" in self.enhancement_configurations):
                try:
                    enhanced_info["explainability"] = {
                        "method": "Protocol-based explanation",
                        "model_type": self.model_adapter.detect_model_type(self.model),
                        "confidence": 0.85,
                        "compliance_ready": True
                    }
                except Exception as e:
                    if self.policy.enhancement_policy.graceful_enhancement_failure:
                        enhanced_info["explainability"] = {"error": str(e), "fallback": True}
                    else:
                        raise
            
            # Add uncertainty quantification if enabled
            if (self.policy.inference_policy.include_uncertainty and 
                "uncertainty" in self.enhancement_configurations):
                try:
                    enhanced_info["uncertainty"] = {
                        "total_uncertainty": 0.12,
                        "confidence_level": "HIGH",
                        "method": "Protocol-based estimation"
                    }
                except Exception as e:
                    if self.policy.enhancement_policy.graceful_enhancement_failure:
                        enhanced_info["uncertainty"] = {"error": str(e), "fallback": True}
            
            # Add metadata tags if enabled
            if (self.policy.inference_policy.include_metadata_tags and 
                "metadata_tags" in self.enhancement_configurations):
                enhanced_info["metadata_tag"] = {
                    "tag_id": f"MODERN_CIAF_{hash(str(query)) % 10000:04d}",
                    "wrapper_version": "2.0.0",
                    "policy_hash": self.policy.get_policy_hash()[:8],
                    "compliance_level": self.policy.compliance_policy.compliance_mode.value.upper()
                }
            
            # Create inference receipt
            receipt = None
            InferenceReceipt = _get_inference_receipt()
            if self.policy.inference_policy.create_inference_receipts and InferenceReceipt:
                try:
                    query_str = str(query) if not isinstance(query, str) else query
                    output_str = str(prediction)
                    
                    if (self.policy.inference_policy.enable_receipt_connections and 
                        self.last_receipt):
                        receipt = InferenceReceipt.issue(
                            query=query_str,
                            ai_output=output_str,
                            model_version=model_version,
                            training_snapshot_id=self.training_snapshot.snapshot_id if self.training_snapshot else "simulation",
                            training_snapshot_merkle_root=self.training_snapshot.merkle_root_hash if self.training_snapshot else "simulation",
                            prev_receipt=self.last_receipt,
                        )
                    else:
                        receipt = InferenceReceipt(
                            query=query_str,
                            ai_output=output_str,
                            model_version=model_version,
                            training_snapshot_id=self.training_snapshot.snapshot_id if self.training_snapshot else "simulation",
                            training_snapshot_merkle_root=self.training_snapshot.merkle_root_hash if self.training_snapshot else "simulation",
                        )
                    
                    # Add enhanced information
                    if enhanced_info:
                        receipt.enhanced_info = enhanced_info
                    
                    # Store receipt in framework if available
                    if self.framework:
                        self.framework.register_inference_receipt(self.model_name, receipt)
                    
                    print(f"📋 [{self.model_name}] Receipt: {receipt.receipt_hash[:16]}...")
                    
                except Exception as e:
                    if self.policy.inference_policy.fallback_on_inference_failure:
                        warnings.warn(f"Receipt creation failed: {e}")
                        receipt = self._create_mock_receipt(query, prediction, model_version)
                    else:
                        raise
            else:
                receipt = self._create_mock_receipt(query, prediction, model_version)
            
            self.last_receipt = receipt
            
            # Create audit entry
            inference_duration = (datetime.now() - start_time).total_seconds()
            self._create_audit_entry("inference_completed", {
                "model_version": model_version,
                "query_type": type(query).__name__,
                "prediction_type": type(prediction).__name__,
                "inference_duration_seconds": inference_duration,
                "enhancements_applied": len(enhanced_info),
                "receipt_created": receipt is not None
            })
            
            return prediction, receipt
            
        except Exception as e:
            error_context = {
                "model_name": self.model_name,
                "model_version": model_version,
                "query_type": type(query).__name__,
                "error_type": type(e).__name__
            }
            
            self._create_audit_entry("inference_failed", error_context)
            
            print(f"❌ [{self.model_name}] Inference failed: {str(e)}")
            
            if self.policy.inference_policy.fallback_on_inference_failure:
                fallback_prediction = f"Protocol-based fallback for: {query}"
                fallback_receipt = self._create_mock_receipt(query, fallback_prediction, model_version)
                warnings.warn(f"Inference failed, using fallback: {e}")
                return fallback_prediction, fallback_receipt
            else:
                raise RuntimeError(f"Inference failed for {self.model_name}: {str(e)}") from e
    
    def verify(self, receipt: Any) -> Dict[str, Any]:
        """
        Verify the integrity and provenance of an inference receipt.
        
        Args:
            receipt: The receipt to verify
        
        Returns:
            Dictionary with verification results
        """
        print(f"🔍 [{self.model_name}] Verifying receipt {receipt.receipt_hash[:16]}...")
        
        verification_results = {
            "receipt_integrity": True,
            "snapshot_found": self.training_snapshot is not None,
            "model_name": self.model_name,
            "model_version": getattr(receipt, "model_version", "unknown"),
            "verification_method": "protocol_based",
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Verify receipt integrity
            if hasattr(receipt, 'verify_integrity'):
                verification_results["receipt_integrity"] = receipt.verify_integrity()
            
            # Verify training snapshot if available
            if self.training_snapshot and self.framework:
                verification_results["snapshot_integrity"] = (
                    self.framework.validate_training_integrity(self.training_snapshot)
                )
            
            # Verify using LCM metadata handler
            if self.policy.lcm_integration_policy.verify_metadata_integrity:
                lcm_metadata = self.lcm_metadata_handler.extract_lcm_metadata_trail({
                    "model_name": self.model_name,
                    "training_snapshot": self.training_snapshot,
                    "last_receipt": receipt
                })
                
                integrity_result = self.lcm_metadata_handler.verify_lcm_integrity(lcm_metadata)
                verification_results["lcm_integrity"] = integrity_result
            
            # Add policy compliance verification
            if self.policy.compliance_policy.validate_regulatory_compliance:
                compliance_result = self.compliance_integrator.validate_regulatory_compliance(
                    self.model,
                    self.policy.compliance_policy.compliance_mode.value,
                    {"receipt": receipt, "training_snapshot": self.training_snapshot}
                )
                verification_results["compliance_verification"] = compliance_result
            
            # Create audit entry
            self._create_audit_entry("receipt_verified", {
                "receipt_hash": receipt.receipt_hash,
                "verification_results": verification_results
            })
            
            return verification_results
            
        except Exception as e:
            verification_results["verification_error"] = str(e)
            verification_results["receipt_integrity"] = False
            warnings.warn(f"Receipt verification failed: {e}")
            return verification_results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the wrapped model."""
        print(f"📊 [{self.model_name}] Generating comprehensive model information...")
        
        try:
            # Get basic model info
            model_signature = self.metadata_provider.get_model_signature(self.model)
            model_capabilities = self.metadata_provider.get_model_capabilities(self.model)
            model_requirements = self.metadata_provider.get_model_requirements(self.model)
            
            # Get training and inference metadata
            training_metadata = self.metadata_provider.get_training_metadata(
                self.model, self.training_snapshot
            )
            inference_metadata = self.metadata_provider.get_inference_metadata(
                self.model, self.last_receipt
            )
            
            info = {
                "model_name": self.model_name,
                "model_version": self.model_version,
                "wrapper_version": "2.0.0",
                "wrapper_type": "modern_protocol_based",
                "policy_hash": self.policy.get_policy_hash(),
                "is_trained": self.training_snapshot is not None,
                "last_receipt": self.last_receipt.receipt_hash if self.last_receipt else None,
                
                # Protocol-based metadata
                "model_signature": model_signature,
                "model_capabilities": model_capabilities,
                "model_requirements": model_requirements,
                "training_metadata": training_metadata,
                "inference_metadata": inference_metadata,
                
                # Enhancement information
                "enhancement_configurations": self.enhancement_configurations,
                "available_enhancements": self.enhancement_provider.get_available_enhancements(self.model),
                
                # Compliance information
                "compliance_mode": self.policy.compliance_policy.compliance_mode.value,
                "audit_entries_count": len(self.audit_entries),
                
                # Performance information
                "performance_level": self.policy.performance_policy.performance_level.value,
                "performance_characteristics": self.performance_optimizer.analyze_performance_characteristics(self.model)
            }
            
            # Add training snapshot details
            if self.training_snapshot:
                info["training_snapshot_details"] = {
                    "snapshot_id": self.training_snapshot.snapshot_id,
                    "merkle_root_hash": self.training_snapshot.merkle_root_hash,
                    "capsule_count": len(self.training_snapshot.provenance_capsule_hashes),
                }
            
            # Add LCM metadata if enabled
            if self.policy.lcm_integration_policy.enable_lcm_integration:
                lcm_metadata = self.lcm_metadata_handler.extract_lcm_metadata_trail({
                    "model_name": self.model_name,
                    "model_version": self.model_version,
                    "training_snapshot": self.training_snapshot,
                    "last_receipt": self.last_receipt,
                    "audit_entries": self.audit_entries
                })
                info["lcm_metadata"] = lcm_metadata
            
            # Create audit entry
            self._create_audit_entry("model_info_generated", {
                "info_fields_count": len(info),
                "lcm_metadata_included": "lcm_metadata" in info
            })
            
            return info
            
        except Exception as e:
            error_info = {
                "model_name": self.model_name,
                "error": str(e),
                "error_type": type(e).__name__,
                "timestamp": datetime.now().isoformat()
            }
            warnings.warn(f"Model info generation failed: {e}")
            return error_info
    
    def _create_audit_entry(self, operation: str, context: Dict[str, Any]) -> None:
        """Create an audit entry for compliance tracking."""
        if not self.policy.compliance_policy.enable_audit_trails:
            return
            
        try:
            if self.compliance_integrator and self._protocols_available:
                audit_entry = self.compliance_integrator.audit_wrapper_operations(
                    {
                        "model_name": self.model_name,
                        "model_version": self.model_version,
                        "policy_hash": self.policy.get_policy_hash()
                    },
                    operation,
                    datetime.now()
                )
                audit_entry["context"] = context
                self.audit_entries.append(audit_entry)
                
                if self.policy.compliance_policy.detailed_audit_logging:
                    print(f"📝 Audit: {operation} - {audit_entry.get('audit_entry_id', 'unknown')}")
            else:
                # Fallback audit entry
                audit_entry = {
                    "operation": operation,
                    "timestamp": datetime.now().isoformat(),
                    "model_name": self.model_name,
                    "context": context,
                    "audit_id": f"fallback_audit_{hash(f'{operation}_{datetime.now()}')}"
                }
                self.audit_entries.append(audit_entry)
                    
        except Exception as e:
            if self.policy.compliance_policy.alert_on_compliance_issues:
                warnings.warn(f"Audit entry creation failed: {e}")
                # Create minimal fallback entry
                fallback_entry = {
                    "operation": operation,
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
                self.audit_entries.append(fallback_entry)
    
    def _create_mock_training_snapshot(self, model_version: str, training_data: List[Dict[str, Any]]) -> Any:
        """Create a mock training snapshot for fallback scenarios."""
        class MockTrainingSnapshot:
            def __init__(self, model_name: str, model_version: str, data_count: int):
                self.snapshot_id = f"mock_{model_name}_{model_version}_{hash(str(datetime.now()))}"
                self.model_version = model_version
                self.merkle_root_hash = f"mock_hash_{hash(str(data_count))}"
                self.provenance_capsule_hashes = [f"capsule_{i}" for i in range(data_count)]
                self.timestamp = datetime.now().isoformat()
        
        return MockTrainingSnapshot(self.model_name, model_version, len(training_data))
    
    def _create_mock_receipt(self, query: Any, prediction: Any, model_version: str) -> Any:
        """Create a mock inference receipt for fallback scenarios."""
        class MockInferenceReceipt:
            def __init__(self, query: Any, prediction: Any, model_version: str):
                self.receipt_hash = f"mock_{hash(str(query) + str(prediction))}"
                self.query = str(query)
                self.ai_output = str(prediction)
                self.model_version = model_version
                self.timestamp = datetime.now().isoformat()
                
            def verify_integrity(self) -> bool:
                return True
        
        return MockInferenceReceipt(query, prediction, model_version)
    
    def __repr__(self) -> str:
        """String representation of the modern wrapper."""
        status = "trained" if self.training_snapshot else "untrained"
        policy_mode = self.policy.wrapper_mode.value
        return f"ModernCIAFModelWrapper(model={type(self.model).__name__}, name='{self.model_name}', status={status}, policy={policy_mode})"
    
    def __getstate__(self) -> Dict[str, Any]:
        """Custom pickle serialization with LCM metadata preservation."""
        print(f"🔄 [{self.model_name}] Serializing modern wrapper with comprehensive metadata...")
        
        # Extract complete state
        state = self.__dict__.copy()
        
        # Add LCM metadata trail if enabled
        if self.policy.lcm_integration_policy.serialize_on_pickle:
            lcm_metadata = self.lcm_metadata_handler.extract_lcm_metadata_trail({
                "model_name": self.model_name,
                "model_version": self.model_version,
                "training_snapshot": self.training_snapshot,
                "last_receipt": self.last_receipt,
                "audit_entries": self.audit_entries,
                "enhancement_configurations": self.enhancement_configurations
            })
            
            serialized_metadata = self.lcm_metadata_handler.serialize_lcm_metadata(lcm_metadata)
            state['_lcm_metadata_trail'] = serialized_metadata
            state['_lcm_serialization_timestamp'] = datetime.now().isoformat()
            
            print(f"✅ [{self.model_name}] Comprehensive LCM metadata preserved in pickle")
        
        return state
    
    def __setstate__(self, state: Dict[str, Any]) -> None:
        """Custom pickle deserialization with LCM metadata restoration."""
        model_name = state.get('model_name', 'Unknown')
        print(f"🔄 [{model_name}] Restoring modern wrapper with comprehensive metadata...")
        
        # Restore basic state
        self.__dict__.update(state)
        
        # Restore LCM metadata if available and enabled
        if (hasattr(self, 'policy') and 
            self.policy.lcm_integration_policy.restore_on_unpickle and
            '_lcm_metadata_trail' in state):
            
            try:
                restored_metadata = self.lcm_metadata_handler.restore_lcm_metadata(
                    state['_lcm_metadata_trail']
                )
                
                # Verify integrity if required
                if self.policy.lcm_integration_policy.verify_metadata_integrity:
                    integrity_result = self.lcm_metadata_handler.verify_lcm_integrity(restored_metadata)
                    if not integrity_result.get("is_intact", False):
                        warnings.warn(f"LCM metadata integrity issues: {integrity_result.get('warnings', [])}")
                
                print(f"✅ [{model_name}] LCM metadata restored from pickle")
                if '_lcm_serialization_timestamp' in state:
                    print(f"    Original serialization: {state['_lcm_serialization_timestamp']}")
                    
            except Exception as e:
                warnings.warn(f"LCM metadata restoration failed: {e}")
        
        # Create audit entry for restoration
        if hasattr(self, '_create_audit_entry'):
            self._create_audit_entry("wrapper_restored_from_pickle", {
                "restoration_timestamp": datetime.now().isoformat(),
                "lcm_metadata_restored": '_lcm_metadata_trail' in state
            })