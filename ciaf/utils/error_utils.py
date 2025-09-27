"""
CIAF Error Utilities
User-friendly error handling and translation
Following naming improvement recommendations from CIAF_LCM_IMPROVEMENT_RECOMMENDATIONS.md
"""

import logging
from typing import Dict, Optional, Any
import re

logger = logging.getLogger(__name__)


class CIAFErrorUtils:
    """User-friendly error handling for CIAF operations"""
    
    # Error translation mappings
    ERROR_TRANSLATIONS = {
        # Provenance errors
        "ProvenanceCapsuleGenerationException": "CIAF training failed. Check your data format and try again.",
        "Hash mismatch": "Data integrity check failed. Verify your data hasn't been modified.",
        "Merkle tree validation": "CIAF verification failed. Try retraining the model.",
        
        # Training errors  
        "Training failed": "Model training failed. Check your data format and parameters.",
        "DataAnchorException": "CIAF data setup failed. Ensure your data is properly formatted.",
        "MAA initialization": "CIAF security setup failed. Try again or contact support.",
        
        # Prediction errors
        "InferenceReceiptException": "Prediction receipt generation failed. Model may need retraining.",
        "Model not trained": "Model not trained yet. Call train_with_ciaf() first.",
        "Invalid input": "Invalid data format. Use CIAFDataUtils.to_ciaf_format() to convert.",
        
        # Data format errors
        "cannot unpack": "Invalid data format. Check your input data structure.",
        "NoneType object": "Missing data. Ensure all required fields are provided.",
        "float() argument": "Invalid data type. Check that numeric fields contain valid numbers.",
        
        # Wrapper errors
        "Wrapper not initialized": "CIAF wrapper not set up. Initialize the model first.",
        "CIAFModelWrapper": "CIAF integration failed. Check your CIAF configuration."
    }
    
    @staticmethod
    def translate_error(error: Exception) -> str:
        """
        Translate technical error to user-friendly message (simplified naming)
        Previously: generate_comprehensive_user_friendly_error_message_with_detailed_diagnostics()
        """
        if error is None:
            return "Unknown error occurred"
        
        error_str = str(error)
        error_type = type(error).__name__
        
        # Check for exact matches first
        for pattern, message in CIAFErrorUtils.ERROR_TRANSLATIONS.items():
            if pattern.lower() in error_str.lower() or pattern.lower() in error_type.lower():
                return message
        
        # Fallback to generic messages based on error type
        if "training" in error_str.lower():
            return "Training failed. Check your data format and model parameters."
        elif "prediction" in error_str.lower():
            return "Prediction failed. Ensure model is trained and input data is valid."
        elif "data" in error_str.lower():
            return "Data processing failed. Check your data format and content."
        elif "ciaf" in error_str.lower():
            return "CIAF operation failed. Check your CIAF configuration and try again."
        
        # Last resort: return original error but make it more readable
        return f"Operation failed: {error_str}"
    
    @staticmethod
    def get_help_message(error_category: str) -> str:
        """
        Get helpful guidance for error category (simplified naming)
        Previously: generate_comprehensive_error_resolution_guidance_with_step_by_step_instructions()
        """
        help_messages = {
            "data_format": """
Data Format Help:
• Use CIAFDataUtils.to_ciaf_format(X, y) to convert your data
• Ensure X is a pandas DataFrame and y is a pandas Series
• Check for missing values or invalid data types
""",
            "training": """
Training Help:
• Verify your data format with CIAFDataUtils.validate()
• Ensure sufficient data for training (minimum 10 samples)
• Check that target values are numeric for regression
""",
            "prediction": """
Prediction Help:
• Make sure model is trained first with train_with_ciaf()
• Input data must have same features as training data
• Use same data preprocessing as during training
""",
            "ciaf_setup": """
CIAF Setup Help:
• Ensure CIAF components are properly installed
• Check that metadata storage is accessible
• Verify CIAF wrapper configuration
"""
        }
        
        return help_messages.get(error_category, "No specific help available for this error.")
    
    @staticmethod
    def categorize_error(error: Exception) -> str:
        """
        Categorize error for better handling (simplified naming)
        Previously: perform_comprehensive_error_classification_and_categorization_analysis()
        """
        if error is None:
            return "unknown"
        
        error_str = str(error).lower()
        
        if any(term in error_str for term in ["format", "convert", "data", "type"]):
            return "data_format"
        elif any(term in error_str for term in ["train", "fit", "learn"]):
            return "training"
        elif any(term in error_str for term in ["predict", "inference", "forecast"]):
            return "prediction"
        elif any(term in error_str for term in ["ciaf", "wrapper", "provenance"]):
            return "ciaf_setup"
        else:
            return "general"
    
    @staticmethod
    def log_error(
        error: Exception, 
        context: str = "", 
        model_name: Optional[str] = None
    ) -> str:
        """
        Log error with context (simplified naming)
        Previously: comprehensive_error_logging_with_detailed_context_and_stack_trace_analysis()
        """
        user_message = CIAFErrorUtils.translate_error(error)
        category = CIAFErrorUtils.categorize_error(error)
        
        log_message = f"❌ {user_message}"
        if context:
            log_message += f" (Context: {context})"
        if model_name:
            log_message += f" (Model: {model_name})"
        
        # Log technical details for developers
        logger.error(f"{log_message} (Original: {str(error)})")
        
        # Return user-friendly message
        return user_message
    
    @staticmethod
    def create_error_report(
        error: Exception,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Create comprehensive error report (simplified naming)
        Previously: generate_comprehensive_error_diagnostic_report_with_resolution_recommendations()
        """
        user_message = CIAFErrorUtils.translate_error(error)
        category = CIAFErrorUtils.categorize_error(error)
        help_message = CIAFErrorUtils.get_help_message(category)
        
        report = {
            "user_message": user_message,
            "category": category,
            "help": help_message,
            "technical_error": str(error),
            "error_type": type(error).__name__,
            "timestamp": "2024-01-01 00:00:00",  # Would use datetime.now() in production
        }
        
        if context:
            report["context"] = context
        
        return report