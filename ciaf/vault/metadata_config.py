"""
CIAF Metadata Storage Configuration

Configuration settings for the CIAF metadata storage system.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional


class MetadataConfig:
    """Configuration class for CIAF metadata storage."""

    # Default configuration
    DEFAULT_CONFIG = {
        # Storage settings
        "storage_backend": "json",  # Options: "json", "sqlite", "pickle"
        "storage_path": "ciaf_metadata",
        "enable_compression": False,
        "max_file_size_mb": 100,
        # Performance optimization settings
        "enable_lazy_materialization": True,  # Only create full audit trails when needed
        "batch_size": 100,  # Batch metadata operations
        "async_writes": True,  # Use async I/O for better performance
        "cache_size": 10000,  # Large in-memory cache
        "defer_validation": True,  # Skip expensive validations during training
        "fast_inference_mode": True,  # Optimize for inference speed
        "memory_buffer_size": 50,  # Buffer operations in memory
        # Deferred LCM settings
        "enable_deferred_lcm": True,  # Enable deferred LCM processing
        "default_lcm_mode": "adaptive",  # Options: "immediate", "deferred", "adaptive"
        "lcm_batch_size": 50,  # Number of receipts to process in each batch
        "lcm_processing_interval": 2.0,  # Seconds between batch processing
        "lcm_queue_max_size": 10000,  # Maximum queue size for lightweight receipts
        "lcm_immediate_threshold_ms": 50.0,  # Threshold for adaptive mode switching
        "lcm_cpu_threshold_percent": 80.0,  # CPU threshold for adaptive decisions
        "lcm_memory_threshold_mb": 500.0,  # Memory threshold for adaptive decisions
        "lcm_enable_persistence": True,  # Persist queue on shutdown
        "lcm_storage_dir": "deferred_lcm_storage",  # Directory for LCM storage
        "lcm_audit_batch_format": "json",  # Format for audit batch files
        "lcm_worker_threads": 2,  # Number of worker threads for background processing
        "lcm_overflow_strategy": "drop_oldest",  # Strategy when queue overflows: "drop_oldest", "drop_newest", "block"
        "lcm_enable_parallel_processing": True,  # Enable parallel processing of batches
        "lcm_retry_attempts": 3,  # Number of retry attempts for failed processing
        "lcm_retry_delay_seconds": 1.0,  # Delay between retry attempts
        "lcm_health_check_interval": 30.0,  # Seconds between health checks
        "lcm_metrics_collection": True,  # Enable LCM metrics collection
        "lcm_compression_enabled": False,  # Enable compression for LCM storage
        "deployment_anchor_implementation": "full",  # "full", "stub", or "disabled"
        # Database settings (for SQLite backend)
        "db_connection_timeout": 30,
        "db_journal_mode": "WAL",
        "db_cache_size": 10000,
        "db_connection_pool_size": 5,  # Connection pooling
        # Retention settings
        "metadata_retention_days": 365,
        "compliance_retention_days": 2555,  # 7 years for compliance
        "auto_cleanup_enabled": False,
        "cleanup_schedule": "weekly",
        # Export settings
        "default_export_format": "json",
        "export_batch_size": 1000,
        "include_metadata_hash": True,
        # Security settings
        "encrypt_sensitive_data": False,
        "encryption_key_path": None,
        "audit_all_access": True,
        # Performance settings (legacy - kept for compatibility)
        "cache_size": 10000,  # Duplicated above for compatibility
        "async_writes": True,  # Duplicated above for compatibility
        "batch_size": 100,    # Duplicated above for compatibility
        # Compliance settings
        "compliance_frameworks": ["GDPR", "FDA", "EEOC", "FCRA", "HIPAA", "ISO_13485"],
        "required_metadata_fields": ["model_name", "stage", "event_type", "timestamp"],
        "compliance_score_threshold": 0.8,
        # Monitoring settings
        "enable_metrics": True,
        "metrics_endpoint": None,
        "alert_on_compliance_failure": True,
        "alert_webhook_url": None,
    }

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize configuration.

        Args:
            config_path: Path to configuration file (JSON)
        """
        self.config = self.DEFAULT_CONFIG.copy()

        # Load from environment variables
        self._load_from_env()

        # Load from config file if provided
        if config_path:
            self._load_from_file(config_path)

    def _load_from_env(self):
        """Load configuration from environment variables."""
        env_mappings = {
            "CIAF_STORAGE_BACKEND": "storage_backend",
            "CIAF_STORAGE_PATH": "storage_path",
            "CIAF_DB_TIMEOUT": "db_connection_timeout",
            "CIAF_RETENTION_DAYS": "metadata_retention_days",
            "CIAF_COMPLIANCE_RETENTION": "compliance_retention_days",
            "CIAF_AUTO_CLEANUP": "auto_cleanup_enabled",
            "CIAF_ENCRYPT_DATA": "encrypt_sensitive_data",
            "CIAF_ENCRYPTION_KEY": "encryption_key_path",
            "CIAF_AUDIT_ACCESS": "audit_all_access",
            "CIAF_CACHE_SIZE": "cache_size",
            "CIAF_ASYNC_WRITES": "async_writes",
            "CIAF_BATCH_SIZE": "batch_size",
            "CIAF_ENABLE_METRICS": "enable_metrics",
            "CIAF_METRICS_ENDPOINT": "metrics_endpoint",
            "CIAF_ALERT_WEBHOOK": "alert_webhook_url",
        }

        for env_var, config_key in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Type conversion based on default value type
                default_type = type(self.config[config_key])
                if default_type == bool:
                    self.config[config_key] = value.lower() in (
                        "true",
                        "1",
                        "yes",
                        "on",
                    )
                elif default_type == int:
                    self.config[config_key] = int(value)
                elif default_type == float:
                    self.config[config_key] = float(value)
                else:
                    self.config[config_key] = value

    def _load_from_file(self, config_path: str):
        """Load configuration from JSON file."""
        import json

        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                file_config = json.load(f)
                self.config.update(file_config)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)

    def set(self, key: str, value: Any):
        """Set configuration value."""
        self.config[key] = value

    def save_to_file(self, config_path: str):
        """Save current configuration to file."""
        import json

        config_file = Path(config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w") as f:
            json.dump(self.config, f, indent=2)

    def validate(self) -> Dict[str, str]:
        """
        Validate configuration settings.

        Returns:
            Dictionary of validation errors (empty if valid)
        """
        errors = {}

        # Validate storage backend
        valid_backends = ["json", "sqlite", "pickle"]
        if self.config["storage_backend"] not in valid_backends:
            errors["storage_backend"] = f"Must be one of: {valid_backends}"

        # Validate retention days
        if self.config["metadata_retention_days"] < 1:
            errors["metadata_retention_days"] = "Must be at least 1 day"

        if (
            self.config["compliance_retention_days"]
            < self.config["metadata_retention_days"]
        ):
            errors["compliance_retention_days"] = "Should be >= metadata_retention_days"

        # Validate cache size
        if self.config["cache_size"] < 0:
            errors["cache_size"] = "Must be non-negative"

        # Validate batch size
        if self.config["batch_size"] < 1:
            errors["batch_size"] = "Must be at least 1"

        # Validate deferred LCM settings
        valid_lcm_modes = ["immediate", "deferred", "adaptive"]
        if self.config["default_lcm_mode"] not in valid_lcm_modes:
            errors["default_lcm_mode"] = f"Must be one of: {valid_lcm_modes}"
            
        if self.config["lcm_batch_size"] < 1:
            errors["lcm_batch_size"] = "Must be at least 1"
            
        if self.config["lcm_processing_interval"] < 0.1:
            errors["lcm_processing_interval"] = "Must be at least 0.1 seconds"
            
        if self.config["lcm_queue_max_size"] < 100:
            errors["lcm_queue_max_size"] = "Must be at least 100"
            
        if self.config["lcm_immediate_threshold_ms"] < 0:
            errors["lcm_immediate_threshold_ms"] = "Must be non-negative"
            
        if not (0 <= self.config["lcm_cpu_threshold_percent"] <= 100):
            errors["lcm_cpu_threshold_percent"] = "Must be between 0 and 100"
            
        if self.config["lcm_memory_threshold_mb"] < 0:
            errors["lcm_memory_threshold_mb"] = "Must be non-negative"

        # Validate compliance score threshold
        threshold = self.config["compliance_score_threshold"]
        if not (0.0 <= threshold <= 1.0):
            errors["compliance_score_threshold"] = "Must be between 0.0 and 1.0"

        # Validate storage path
        try:
            Path(self.config["storage_path"]).resolve()
        except Exception as e:
            errors["storage_path"] = f"Invalid path: {e}"

        return errors

    def get_storage_path(self) -> Path:
        """Get resolved storage path."""
        return Path(self.config["storage_path"]).resolve()

    def is_compliance_framework_enabled(self, framework: str) -> bool:
        """Check if a compliance framework is enabled."""
        return framework in self.config["compliance_frameworks"]

    def get_retention_days(self, data_type: str = "metadata") -> int:
        """Get retention days for specific data type."""
        if data_type == "compliance":
            return self.config["compliance_retention_days"]
        return self.config["metadata_retention_days"]


# Global configuration instance
_global_config = None


def get_metadata_config(config_path: Optional[str] = None) -> MetadataConfig:
    """Get global metadata configuration instance."""
    global _global_config
    if _global_config is None:
        _global_config = MetadataConfig(config_path)
    return _global_config


def load_config_from_file(config_path: str) -> MetadataConfig:
    """Load configuration from file and set as global."""
    global _global_config
    _global_config = MetadataConfig(config_path)
    return _global_config


# Example configuration templates
CONFIG_TEMPLATES = {
    "development": {
        "storage_backend": "json",
        "storage_path": "dev_metadata",
        "metadata_retention_days": 30,
        "auto_cleanup_enabled": True,
        "enable_metrics": False,
        "audit_all_access": False,
    },
    "production": {
        "storage_backend": "sqlite",
        "storage_path": "/var/lib/ciaf/metadata",
        "metadata_retention_days": 365,
        "compliance_retention_days": 2555,
        "auto_cleanup_enabled": True,
        "enable_metrics": True,
        "encrypt_sensitive_data": True,
        "audit_all_access": True,
        "cleanup_schedule": "daily",
    },
    "testing": {
        "storage_backend": "json",
        "storage_path": "test_metadata",
        "metadata_retention_days": 7,
        "auto_cleanup_enabled": True,
        "enable_metrics": False,
        "audit_all_access": False,
        "cache_size": 100,
    },
    "high_performance": {
        "storage_backend": "sqlite",
        "storage_path": "ciaf_metadata_hp",
        "async_writes": True,
        "batch_size": 500,
        "cache_size": 10000,
        "db_cache_size": 50000,
        "enable_compression": True,
        "enable_lazy_materialization": True,
        "defer_validation": True,
        "fast_inference_mode": True,
        "memory_buffer_size": 100,
        "db_connection_pool_size": 10,
    },
    "ultra_fast": {
        "storage_backend": "sqlite", 
        "storage_path": "ciaf_metadata_ultrafast",
        "async_writes": True,
        "batch_size": 1000,
        "cache_size": 50000,
        "db_cache_size": 100000,
        "enable_compression": False,  # Disable compression for speed
        "enable_lazy_materialization": True,
        "defer_validation": True,
        "fast_inference_mode": True,
        "memory_buffer_size": 200,
        "db_connection_pool_size": 20,
        "audit_all_access": False,  # Reduce audit overhead
        "enable_metrics": False,    # Disable metrics collection
        "include_metadata_hash": False,  # Skip hash calculations
    },
    "inference_optimized": {
        "storage_backend": "sqlite",
        "storage_path": "ciaf_metadata_inference",
        "async_writes": True,
        "batch_size": 50,  # Smaller batches for inference
        "cache_size": 20000,
        "db_cache_size": 100000,
        "enable_compression": False,
        "enable_lazy_materialization": True,
        "defer_validation": True,
        "fast_inference_mode": True,
        "memory_buffer_size": 300,  # Large buffer for inference
        "db_connection_pool_size": 15,
        "audit_all_access": False,
        "enable_metrics": False,
        "include_metadata_hash": False,
        "compliance_frameworks": [],  # Skip compliance checks during inference
    },
}


def create_config_template(template_name: str, output_path: str):
    """Create a configuration file from template."""
    if template_name not in CONFIG_TEMPLATES:
        raise ValueError(
            f"Unknown template: {template_name}. Available: {list(CONFIG_TEMPLATES.keys())}"
        )

    config = MetadataConfig()
    config.config.update(CONFIG_TEMPLATES[template_name])
    config.save_to_file(output_path)

    return output_path


def create_deferred_lcm_config(
    mode: str = "adaptive",
    batch_size: int = 50,
    processing_interval: float = 2.0,
    enable_fast_inference: bool = True
) -> Dict[str, Any]:
    """
    Create a configuration optimized for deferred LCM processing.
    
    Args:
        mode: LCM mode ("immediate", "deferred", "adaptive")
        batch_size: Number of receipts to process in each batch
        processing_interval: Seconds between batch processing
        enable_fast_inference: Enable optimizations for fast inference
        
    Returns:
        Configuration dictionary
    """
    config = {
        "enable_deferred_lcm": True,
        "default_lcm_mode": mode,
        "lcm_batch_size": batch_size,
        "lcm_processing_interval": processing_interval,
        "fast_inference_mode": enable_fast_inference,
        "enable_lazy_materialization": True,
        "async_writes": True,
        "cache_size": 20000 if enable_fast_inference else 10000,
        "memory_buffer_size": 100 if enable_fast_inference else 50,
    }
    
    if mode == "adaptive":
        config.update({
            "lcm_immediate_threshold_ms": 50.0,
            "lcm_cpu_threshold_percent": 80.0,
            "lcm_memory_threshold_mb": 500.0,
        })
    elif mode == "deferred":
        config.update({
            "lcm_queue_max_size": 20000,
            "lcm_enable_persistence": True,
        })
        
    return config


def create_high_performance_config() -> Dict[str, Any]:
    """Create configuration optimized for maximum performance."""
    return create_deferred_lcm_config(
        mode="deferred",
        batch_size=100,
        processing_interval=1.0,
        enable_fast_inference=True
    )


def create_compliance_first_config() -> Dict[str, Any]:
    """Create configuration prioritizing compliance over performance."""
    return create_deferred_lcm_config(
        mode="immediate",
        batch_size=10,
        processing_interval=5.0,
        enable_fast_inference=False
    )


def create_balanced_config() -> Dict[str, Any]:
    """Create balanced configuration between performance and compliance."""
    return create_deferred_lcm_config(
        mode="adaptive",
        batch_size=50,
        processing_interval=2.0,
        enable_fast_inference=True
    )


def create_enterprise_config() -> Dict[str, Any]:
    """Create enterprise-grade configuration with full features."""
    config = MetadataConfig.DEFAULT_CONFIG.copy()
    config.update({
        # Enterprise performance settings
        "lcm_worker_threads": 8,
        "lcm_queue_max_size": 50000,
        "lcm_enable_parallel_processing": True,
        "lcm_batch_size": 200,
        "lcm_processing_interval": 0.5,
        
        # Enterprise reliability settings
        "lcm_enable_persistence": True,
        "lcm_retry_attempts": 5,
        "lcm_health_check_interval": 10.0,
        "lcm_metrics_collection": True,
        "deployment_anchor_implementation": "full",
        
        # Enterprise security settings
        "encrypt_sensitive_data": True,
        "audit_all_access": True,
        "compliance_frameworks": [
            "GDPR", "HIPAA", "SOX", "ISO_27001", "EU_AI_ACT", "NIST_AI_RMF"
        ],
        
        # Enterprise storage settings
        "storage_backend": "sqlite",
        "enable_compression": True,
        "db_connection_pool_size": 10,
        "metadata_retention_days": 2555,  # 7 years
        "compliance_retention_days": 3650,  # 10 years
        
        # Enterprise monitoring
        "enable_metrics": True,
        "alert_on_compliance_failure": True,
        "lcm_overflow_strategy": "block",  # Don't drop data in enterprise
    })
    return config


def create_development_config() -> Dict[str, Any]:
    """Create development-friendly configuration."""
    config = MetadataConfig.DEFAULT_CONFIG.copy()
    config.update({
        # Development settings for fast iteration
        "lcm_worker_threads": 1,
        "lcm_queue_max_size": 1000,
        "lcm_batch_size": 10,
        "lcm_processing_interval": 5.0,
        "lcm_enable_parallel_processing": False,
        
        # Minimal retention for dev
        "metadata_retention_days": 30,
        "compliance_retention_days": 90,
        
        # Simple storage for dev
        "storage_backend": "json",
        "enable_compression": False,
        "deployment_anchor_implementation": "stub",
        
        # Debug-friendly settings
        "defer_validation": False,  # Enable validations in dev
        "lcm_metrics_collection": True,
        "enable_metrics": True,
    })
    return config


def create_testing_config() -> Dict[str, Any]:
    """Create configuration optimized for testing."""
    config = MetadataConfig.DEFAULT_CONFIG.copy()
    config.update({
        # Fast processing for tests
        "lcm_worker_threads": 1,
        "lcm_queue_max_size": 100,
        "lcm_batch_size": 5,
        "lcm_processing_interval": 0.1,
        "lcm_enable_parallel_processing": False,
        
        # Minimal storage for tests
        "storage_backend": "json",
        "storage_path": "test_ciaf_metadata",
        "lcm_storage_dir": "test_deferred_lcm_storage",
        "enable_compression": False,
        
        # No retention in tests
        "metadata_retention_days": 1,
        "compliance_retention_days": 1,
        "auto_cleanup_enabled": True,
        
        # Simplified settings
        "deployment_anchor_implementation": "stub",
        "encrypt_sensitive_data": False,
        "lcm_enable_persistence": False,
    })
    return config


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1] == "create-template":
            template = sys.argv[2] if len(sys.argv) > 2 else "development"
            output = (
                sys.argv[3] if len(sys.argv) > 3 else f"ciaf_config_{template}.json"
            )

            try:
                path = create_config_template(template, output)
                print(f"Created configuration template '{template}' at: {path}")
            except ValueError as e:
                print(f"Error: {e}")
                print(f"Available templates: {list(CONFIG_TEMPLATES.keys())}")
        elif sys.argv[1] == "validate":
            config_path = sys.argv[2] if len(sys.argv) > 2 else None
            config = MetadataConfig(config_path)

            errors = config.validate()
            if errors:
                print("Configuration validation errors:")
                for key, error in errors.items():
                    print(f"  {key}: {error}")
            else:
                print("Configuration is valid!")
    else:
        # Default: show current configuration
        config = get_metadata_config()
        print("Current CIAF Metadata Configuration:")
        import json

        print(json.dumps(config.config, indent=2))
