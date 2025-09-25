"""
Determinism Metadata Tracking for CIAF

Captures random seeds, library versions, hardware fingerprints, and environment
details required for reproducible ML operations.
"""

import os
import sys
import random
import hashlib
import platform
import subprocess
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, asdict
import json

@dataclass
class DeterminismMetadata:
    """Complete determinism metadata for reproducible operations."""
    random_seeds: Dict[str, int]
    library_versions: Dict[str, str] 
    hardware_fingerprint: str
    environment_info: Dict[str, Any]
    python_info: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DeterminismMetadata':
        """Create from dictionary."""
        return cls(**data)

class DeterminismCapture:
    """Captures determinism metadata for reproducible operations."""
    
    @staticmethod
    def capture_random_seeds() -> Dict[str, int]:
        """Capture current random seeds from various libraries."""
        seeds = {}
        
        # Python built-in random
        try:
            seeds["python"] = random.getstate()[1][0]
        except Exception:
            seeds["python"] = random.randint(0, 2**32-1)
        
        # NumPy
        try:
            import numpy as np
            seeds["numpy"] = np.random.get_state()[1][0]
        except ImportError:
            pass
        except Exception:
            try:
                import numpy as np
                seeds["numpy"] = np.random.randint(0, 2**32-1)
            except ImportError:
                pass
        
        # PyTorch
        try:
            import torch
            seeds["torch"] = torch.initial_seed()
        except ImportError:
            pass
        
        # TensorFlow
        try:
            import tensorflow as tf
            # TF doesn't expose current seed easily, so we record what we can
            seeds["tensorflow"] = "captured"
        except ImportError:
            pass
        
        return seeds
    
    @staticmethod
    def capture_library_versions() -> Dict[str, str]:
        """Capture versions of key ML libraries."""
        versions = {}
        
        # Core libraries
        libraries = [
            "numpy", "pandas", "scikit-learn", "scipy",
            "torch", "tensorflow", "keras", "xgboost",
            "lightgbm", "catboost", "joblib", "pickle"
        ]
        
        for lib in libraries:
            try:
                module = __import__(lib)
                if hasattr(module, "__version__"):
                    versions[lib] = module.__version__
                elif hasattr(module, "version"):
                    versions[lib] = module.version
            except ImportError:
                continue
        
        return versions
    
    @staticmethod
    def capture_hardware_fingerprint() -> str:
        """Generate hardware fingerprint for reproducibility tracking."""
        info = []
        
        # CPU info
        try:
            info.append(f"cpu:{platform.processor()}")
            info.append(f"machine:{platform.machine()}")
            info.append(f"arch:{platform.architecture()[0]}")
        except Exception:
            info.append("cpu:unknown")
        
        # Memory info (approximate)
        try:
            import psutil
            memory_gb = round(psutil.virtual_memory().total / (1024**3))
            info.append(f"memory:{memory_gb}GB")
        except ImportError:
            pass
        
        # GPU info (if available)
        try:
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "unknown"
                info.append(f"gpu:{gpu_count}x{gpu_name}")
            else:
                info.append("gpu:none")
        except ImportError:
            info.append("gpu:unknown")
        
        # Create fingerprint hash
        fingerprint_str = "|".join(info)
        return hashlib.sha256(fingerprint_str.encode()).hexdigest()[:16]
    
    @staticmethod
    def capture_environment_info() -> Dict[str, Any]:
        """Capture environment details affecting reproducibility."""
        env_info = {
            "platform": platform.system(),
            "platform_version": platform.version(),
            "python_version": sys.version,
            "python_executable": sys.executable,
            "working_directory": os.getcwd(),
            "environment_variables": {
                k: v for k, v in os.environ.items() 
                if k.startswith(('PYTHON', 'CUDA', 'MKL', 'OMP', 'OPENBLAS'))
            }
        }
        
        # Git commit if available
        try:
            git_commit = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                stderr=subprocess.DEVNULL, 
                cwd=os.getcwd()
            ).decode().strip()
            env_info["git_commit"] = git_commit
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return env_info
    
    @staticmethod
    def capture_python_info() -> Dict[str, str]:
        """Capture detailed Python environment info."""
        return {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
            "build": str(platform.python_build()),
            "executable": sys.executable
        }
    
    @classmethod
    def capture_full_metadata(self) -> DeterminismMetadata:
        """Capture complete determinism metadata."""
        return DeterminismMetadata(
            random_seeds=self.capture_random_seeds(),
            library_versions=self.capture_library_versions(),
            hardware_fingerprint=self.capture_hardware_fingerprint(),
            environment_info=self.capture_environment_info(),
            python_info=self.capture_python_info()
        )

def capture_determinism_metadata() -> DeterminismMetadata:
    """Capture determinism metadata for current environment."""
    return DeterminismCapture.capture_full_metadata()

def set_reproducible_seeds(base_seed: int = 42) -> Dict[str, int]:
    """Set reproducible seeds across all libraries."""
    seeds_set = {}
    
    # Python random
    random.seed(base_seed)
    seeds_set["python"] = base_seed
    
    # NumPy
    try:
        import numpy as np
        np.random.seed(base_seed)
        seeds_set["numpy"] = base_seed
    except ImportError:
        pass
    
    # PyTorch
    try:
        import torch
        torch.manual_seed(base_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(base_seed)
        seeds_set["torch"] = base_seed
    except ImportError:
        pass
    
    # TensorFlow
    try:
        import tensorflow as tf
        tf.random.set_seed(base_seed)
        seeds_set["tensorflow"] = base_seed
    except ImportError:
        pass
    
    return seeds_set