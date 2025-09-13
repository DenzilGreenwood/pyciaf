# CIAF Simulation Framework

The simulation package provides mock implementations and testing utilities for demonstrating and validating CIAF components without requiring actual ML frameworks or sensitive data.

## Overview

The simulation framework enables safe testing and demonstration of CIAF capabilities:

- **Mock LLM** — Simulated language model for testing inference and training flows
- **ML Framework Simulator** — Mock ML training and inference framework
- **Provenance Testing** — Safe testing of data lineage without sensitive data
- **Compliance Validation** — Test regulatory compliance workflows
- **Integration Testing** — Validate CIAF component interactions
- **Performance Benchmarking** — Test system performance with synthetic workloads

## Components

### MockLLM (`mock_llm.py`)

A simulated language model for testing CIAF inference and audit capabilities.

**Key Features:**
- **Deterministic Responses** — Consistent outputs for testing
- **Configurable Parameters** — Simulated model architecture and size
- **Training Simulation** — Mock training processes for provenance testing
- **Text Generation** — Simple text generation for inference testing
- **Performance Simulation** — Realistic timing delays for testing

**Usage Example:**
```python
from ciaf.simulation import MockLLM

# Create mock LLM
llm = MockLLM("TestModel-10M-Params")
print(f"Model: {llm.model_name}")
print(f"Parameters: {llm.parameter_count:,}")

# Simulate text generation
prompt = "What is machine learning?"
response = llm.generate_text(prompt)
print(f"Response: {response}")

# Simulate training
training_data_hashes = [
    "sha256:hash1...",
    "sha256:hash2...", 
    "sha256:hash3..."
]
training_params = {
    "learning_rate": 0.001,
    "batch_size": 32,
    "epochs": 10
}

llm.conceptual_train(training_data_hashes, training_params)
```

**Model Configuration:**
```python
# Configure different model sizes
small_model = MockLLM("SmallModel-1M")
medium_model = MockLLM("MediumModel-100M") 
large_model = MockLLM("LargeModel-1B")

# Access model parameters
model_specs = small_model.model_params
print(f"Architecture: {model_specs['architecture']}")
print(f"Layers: {model_specs['layers']}")
print(f"Attention heads: {model_specs['attention_heads']}")
```

### MLFrameworkSimulator (`ml_framework.py`)

Comprehensive simulation of ML framework integration with CIAF components.

**Key Features:**
- **Data Preparation** — Simulate creation of provenance capsules
- **Training Simulation** — Mock model training with audit trails
- **MAA Integration** — Test Model Aggregation Anchor workflows
- **Snapshot Creation** — Generate training snapshots for testing
- **Model Anchoring** — Simulate cryptographic model fingerprinting

**Usage Example:**
```python
from ciaf.simulation import MLFrameworkSimulator
from ciaf.provenance import ModelAggregationAnchor
from ciaf.core import secure_random_bytes

# Create ML framework simulator
simulator = MLFrameworkSimulator("DiagnosticModel")

# Prepare mock training data
raw_training_data = [
    {
        "id": "patient_001",
        "content": "Patient symptoms: fever, cough, fatigue",
        "metadata": {
            "source": "hospital_alpha",
            "consent_status": "obtained",
            "data_type": "clinical_notes"
        }
    },
    {
        "id": "patient_002", 
        "content": "Patient symptoms: chest pain, shortness of breath",
        "metadata": {
            "source": "clinic_beta",
            "consent_status": "obtained",
            "data_type": "clinical_notes"
        }
    }
]

# Create provenance capsules
training_capsules = simulator.prepare_data(raw_training_data)
print(f"Created {len(training_capsules)} provenance capsules")

# Create Model Aggregation Anchor
maa = ModelAggregationAnchor(
    key_id="medical_model_maa_v1",
    secret_material="secure_medical_model_secret"
)

# Define training parameters
training_params = {
    "algorithm": "neural_network",
    "learning_rate": 0.001,
    "batch_size": 16,
    "epochs": 50,
    "optimizer": "adam",
    "loss_function": "categorical_crossentropy"
}

# Simulate model training
training_snapshot = simulator.train_model(
    training_data_capsules=training_capsules,
    maa=maa,
    training_params=training_params,
    model_version="diagnostic_v1.0"
)

print(f"Training snapshot ID: {training_snapshot.snapshot_id}")
print(f"Merkle root: {training_snapshot.merkle_root_hash}")

# Get model information
master_anchor = secure_random_bytes(32)
model_info = simulator.get_model_info(
    parameters=training_params,
    master_anchor=master_anchor
)

print(f"Model anchor: {model_info['model_anchor']}")
print(f"Parameter hash: {model_info['model_param_hash']}")
```

## Advanced Simulation Scenarios

### End-to-End Training Simulation

Complete simulation of a training pipeline:

```python
from ciaf.simulation import MLFrameworkSimulator, MockLLM
from ciaf.provenance import ModelAggregationAnchor, ProvenanceCapsule
from ciaf.compliance import AuditTrailGenerator
from datetime import datetime

class ComprehensiveTrainingSimulation:
    def __init__(self, model_name):
        self.simulator = MLFrameworkSimulator(model_name)
        self.llm = MockLLM(f"{model_name}-LLM")
        self.audit_generator = AuditTrailGenerator("training_simulation")
        self.training_session_id = None
    
    def simulate_full_training_pipeline(self, dataset_config, training_config):
        # Start audit session
        self.training_session_id = self.audit_generator.start_training_session(
            model_name=self.simulator.model_name,
            user_id="simulation_system",
            purpose="testing_and_validation"
        )
        
        # Generate synthetic training data
        training_data = self._generate_synthetic_data(dataset_config)
        
        # Create provenance capsules
        capsules = self.simulator.prepare_data(training_data)
        
        # Log data preparation
        for capsule in capsules:
            self.audit_generator.log_data_inclusion(
                session_id=self.training_session_id,
                data_hash=capsule.hash_proof,
                consent_status="simulated_consent"
            )
        
        # Create MAA
        maa = ModelAggregationAnchor(
            key_id=f"{self.simulator.model_name}_maa",
            secret_material="simulation_secret_material"
        )
        
        # Simulate training
        snapshot = self.simulator.train_model(
            training_data_capsules=capsules,
            maa=maa,
            training_params=training_config["parameters"],
            model_version=training_config["version"]
        )
        
        # Complete audit session
        self.audit_generator.complete_training_session(
            session_id=self.training_session_id,
            snapshot_id=snapshot.snapshot_id,
            data_count=len(capsules)
        )
        
        return {
            "snapshot": snapshot,
            "capsules": capsules,
            "maa": maa,
            "audit_session": self.training_session_id,
            "model_info": self.simulator.get_model_info(training_config["parameters"])
        }
    
    def _generate_synthetic_data(self, config):
        # Generate synthetic training data based on configuration
        synthetic_data = []
        
        for i in range(config["sample_count"]):
            data_item = {
                "id": f"synthetic_{i:04d}",
                "content": self._generate_synthetic_content(config["data_type"], i),
                "metadata": {
                    "source": "synthetic_generator",
                    "data_type": config["data_type"],
                    "quality_score": 0.95,
                    "synthetic": True,
                    "generation_seed": config.get("seed", 42) + i
                }
            }
            synthetic_data.append(data_item)
        
        return synthetic_data
    
    def _generate_synthetic_content(self, data_type, index):
        # Generate different types of synthetic content
        templates = {
            "medical": [
                f"Patient {index}: Age 25-65, presenting with symptoms consistent with respiratory condition",
                f"Clinical case {index}: Cardiovascular assessment with normal parameters",
                f"Medical record {index}: Routine checkup with preventive care recommendations"
            ],
            "financial": [
                f"Transaction {index}: Standard banking operation with compliance verification",
                f"Account {index}: Regular activity pattern within normal parameters",
                f"Financial record {index}: Routine transaction processing completed"
            ],
            "text": [
                f"Document {index}: This is a sample text for training language models",
                f"Sample {index}: Natural language processing training data example",
                f"Text {index}: Demonstration content for ML model development"
            ]
        }
        
        template_list = templates.get(data_type, templates["text"])
        return template_list[index % len(template_list)]

# Example usage
simulation = ComprehensiveTrainingSimulation("MedicalDiagnosticModel")

dataset_config = {
    "sample_count": 100,
    "data_type": "medical",
    "seed": 12345
}

training_config = {
    "version": "v1.0_simulation",
    "parameters": {
        "algorithm": "transformer",
        "learning_rate": 0.0001,
        "batch_size": 8,
        "epochs": 20,
        "attention_heads": 12,
        "hidden_layers": 6
    }
}

results = simulation.simulate_full_training_pipeline(dataset_config, training_config)
print(f"Simulation completed. Snapshot: {results['snapshot'].snapshot_id}")
```

### Compliance Testing Simulation

Test compliance validation with simulated data:

```python
from ciaf.simulation import MLFrameworkSimulator
from ciaf.compliance import ComplianceValidator, BiasValidator

class ComplianceTestingSimulation:
    def __init__(self):
        self.simulator = MLFrameworkSimulator("ComplianceTestModel")
        self.compliance_validator = ComplianceValidator("simulation_system")
        self.bias_validator = BiasValidator("bias_testing")
    
    def simulate_compliance_validation(self, scenarios):
        results = {}
        
        for scenario_name, scenario_config in scenarios.items():
            print(f"Running compliance scenario: {scenario_name}")
            
            # Generate scenario-specific data
            training_data = self._generate_compliance_test_data(scenario_config)
            capsules = self.simulator.prepare_data(training_data)
            
            # Create training snapshot
            snapshot = TrainingSnapshot(
                model_version=f"compliance_test_{scenario_name}",
                training_parameters=scenario_config["training_params"],
                provenance_capsule_hashes=[c.hash_proof for c in capsules]
            )
            
            # Run compliance validation
            compliance_result = self.compliance_validator.validate_training_snapshot(
                snapshot=snapshot,
                frameworks=scenario_config["frameworks"]
            )
            
            # Run bias assessment
            bias_result = self.bias_validator.assess_training_data_bias(
                provenance_hashes=snapshot.provenance_capsule_hashes,
                protected_attributes=scenario_config.get("protected_attributes", [])
            )
            
            results[scenario_name] = {
                "snapshot": snapshot,
                "compliance": compliance_result,
                "bias_assessment": bias_result,
                "data_count": len(capsules)
            }
        
        return results
    
    def _generate_compliance_test_data(self, scenario_config):
        # Generate data with specific compliance characteristics
        test_data = []
        data_types = scenario_config.get("data_characteristics", ["standard"])
        
        for i in range(scenario_config["sample_count"]):
            characteristic = data_types[i % len(data_types)]
            
            data_item = {
                "id": f"compliance_test_{i}",
                "content": self._generate_characteristic_data(characteristic, i),
                "metadata": {
                    "source": scenario_config.get("source", "test_system"),
                    "consent_status": scenario_config.get("consent_status", "explicit"),
                    "data_classification": scenario_config.get("classification", "test_data"),
                    "compliance_characteristic": characteristic,
                    "regulatory_region": scenario_config.get("region", "US")
                }
            }
            test_data.append(data_item)
        
        return test_data
    
    def _generate_characteristic_data(self, characteristic, index):
        # Generate data with specific compliance-relevant characteristics
        data_templates = {
            "standard": f"Standard test case {index} with normal characteristics",
            "sensitive": f"Sensitive test case {index} requiring special handling",
            "biased": f"Test case {index} with potential bias indicators",
            "international": f"International test case {index} with cross-border considerations",
            "high_risk": f"High-risk test case {index} requiring enhanced oversight"
        }
        
        return data_templates.get(characteristic, data_templates["standard"])

# Example compliance testing
compliance_sim = ComplianceTestingSimulation()

test_scenarios = {
    "hipaa_medical": {
        "sample_count": 50,
        "frameworks": ["HIPAA"],
        "data_characteristics": ["sensitive", "standard"],
        "classification": "PHI",
        "consent_status": "explicit_written",
        "training_params": {"epochs": 10, "learning_rate": 0.01},
        "protected_attributes": ["age", "gender"]
    },
    "gdpr_european": {
        "sample_count": 75,
        "frameworks": ["GDPR"],
        "data_characteristics": ["international", "sensitive"],
        "classification": "personal_data",
        "region": "EU",
        "training_params": {"epochs": 15, "learning_rate": 0.005},
        "protected_attributes": ["nationality", "age"]
    },
    "ai_act_high_risk": {
        "sample_count": 100,
        "frameworks": ["EU_AI_ACT"],
        "data_characteristics": ["high_risk", "biased"],
        "classification": "high_risk_ai",
        "training_params": {"epochs": 20, "learning_rate": 0.001},
        "protected_attributes": ["ethnicity", "gender", "age"]
    }
}

compliance_results = compliance_sim.simulate_compliance_validation(test_scenarios)

for scenario, result in compliance_results.items():
    print(f"\nScenario: {scenario}")
    print(f"Compliance status: {result['compliance']['status']}")
    print(f"Bias detected: {result['bias_assessment']['bias_detected']}")
    print(f"Data samples: {result['data_count']}")
```

### Performance Testing Simulation

Simulate high-volume training scenarios:

```python
import time
from concurrent.futures import ThreadPoolExecutor

class PerformanceTestingSimulation:
    def __init__(self):
        self.simulator = MLFrameworkSimulator("PerformanceTestModel")
        self.metrics = {}
    
    def simulate_large_scale_training(self, scale_config):
        print(f"Starting large-scale simulation: {scale_config['scale_name']}")
        
        start_time = time.time()
        
        # Generate large dataset
        dataset_generation_start = time.time()
        large_dataset = self._generate_large_dataset(scale_config)
        dataset_generation_time = time.time() - dataset_generation_start
        
        # Create provenance capsules in batches
        capsule_creation_start = time.time()
        all_capsules = self._create_capsules_in_batches(
            large_dataset, 
            scale_config["batch_size"]
        )
        capsule_creation_time = time.time() - capsule_creation_start
        
        # Create large training snapshot
        snapshot_creation_start = time.time()
        snapshot = TrainingSnapshot(
            model_version=f"performance_test_{scale_config['scale_name']}",
            training_parameters=scale_config["training_params"],
            provenance_capsule_hashes=[c.hash_proof for c in all_capsules]
        )
        snapshot_creation_time = time.time() - snapshot_creation_start
        
        # Verify performance
        verification_start = time.time()
        verification_results = self._verify_large_snapshot(snapshot, scale_config)
        verification_time = time.time() - verification_start
        
        total_time = time.time() - start_time
        
        # Collect performance metrics
        self.metrics[scale_config['scale_name']] = {
            "total_time": total_time,
            "dataset_generation_time": dataset_generation_time,
            "capsule_creation_time": capsule_creation_time,
            "snapshot_creation_time": snapshot_creation_time,
            "verification_time": verification_time,
            "data_count": len(all_capsules),
            "throughput": len(all_capsules) / total_time,
            "verification_results": verification_results
        }
        
        return snapshot, all_capsules
    
    def _generate_large_dataset(self, config):
        # Generate large synthetic dataset efficiently
        dataset = []
        sample_count = config["sample_count"]
        
        print(f"Generating {sample_count:,} data samples...")
        
        for i in range(sample_count):
            if i % 10000 == 0:
                print(f"Generated {i:,} / {sample_count:,} samples")
            
            data_item = {
                "id": f"perf_test_{i:06d}",
                "content": f"Performance test sample {i} with synthetic content for scale testing",
                "metadata": {
                    "source": "performance_generator",
                    "batch_id": i // config["batch_size"],
                    "sample_index": i,
                    "generation_timestamp": time.time()
                }
            }
            dataset.append(data_item)
        
        return dataset
    
    def _create_capsules_in_batches(self, dataset, batch_size):
        # Create capsules in batches for memory efficiency
        all_capsules = []
        
        print(f"Creating provenance capsules in batches of {batch_size:,}...")
        
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            batch_capsules = self.simulator.prepare_data(batch)
            all_capsules.extend(batch_capsules)
            
            if (i // batch_size) % 10 == 0:
                print(f"Processed {i + len(batch):,} / {len(dataset):,} samples")
        
        return all_capsules
    
    def _verify_large_snapshot(self, snapshot, config):
        # Verify snapshot integrity and performance
        verification_start = time.time()
        
        # Sample verification (not all data for performance)
        sample_size = min(1000, len(snapshot.provenance_capsule_hashes))
        sample_hashes = snapshot.provenance_capsule_hashes[:sample_size]
        
        verified_count = 0
        for hash_proof in sample_hashes:
            if snapshot.verify_provenance(hash_proof):
                verified_count += 1
        
        verification_time = time.time() - verification_start
        
        return {
            "sample_size": sample_size,
            "verified_count": verified_count,
            "verification_rate": verified_count / sample_size,
            "verification_time": verification_time,
            "merkle_root_valid": bool(snapshot.merkle_root_hash)
        }
    
    def run_performance_benchmarks(self):
        # Run multiple performance scenarios
        benchmark_configs = {
            "small_scale": {
                "scale_name": "small",
                "sample_count": 1000,
                "batch_size": 100,
                "training_params": {"epochs": 5}
            },
            "medium_scale": {
                "scale_name": "medium", 
                "sample_count": 10000,
                "batch_size": 1000,
                "training_params": {"epochs": 10}
            },
            "large_scale": {
                "scale_name": "large",
                "sample_count": 100000,
                "batch_size": 5000,
                "training_params": {"epochs": 20}
            }
        }
        
        results = {}
        
        for config_name, config in benchmark_configs.items():
            print(f"\n{'='*50}")
            print(f"Running benchmark: {config_name}")
            print(f"{'='*50}")
            
            try:
                snapshot, capsules = self.simulate_large_scale_training(config)
                results[config_name] = {
                    "success": True,
                    "snapshot_id": snapshot.snapshot_id,
                    "metrics": self.metrics[config["scale_name"]]
                }
            except Exception as e:
                print(f"Benchmark {config_name} failed: {e}")
                results[config_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        return results

# Example performance testing
perf_sim = PerformanceTestingSimulation()
benchmark_results = perf_sim.run_performance_benchmarks()

# Display results
print("\n" + "="*60)
print("PERFORMANCE BENCHMARK RESULTS")
print("="*60)

for benchmark, result in benchmark_results.items():
    if result["success"]:
        metrics = result["metrics"]
        print(f"\n{benchmark.upper()}:")
        print(f"  Data count: {metrics['data_count']:,}")
        print(f"  Total time: {metrics['total_time']:.2f}s")
        print(f"  Throughput: {metrics['throughput']:.2f} samples/sec")
        print(f"  Verification rate: {metrics['verification_results']['verification_rate']:.2%}")
    else:
        print(f"\n{benchmark.upper()}: FAILED - {result['error']}")
```

## Integration Testing

### CIAF Component Integration

Test integration between CIAF components:

```python
from ciaf.api import CIAFFramework
from ciaf.simulation import MLFrameworkSimulator, MockLLM

class CIAFIntegrationTest:
    def __init__(self):
        self.framework = CIAFFramework("IntegrationTestSystem")
        self.simulator = MLFrameworkSimulator("TestModel")
        self.mock_llm = MockLLM("TestLLM")
    
    def test_end_to_end_integration(self):
        # Test complete CIAF workflow
        print("Testing end-to-end CIAF integration...")
        
        # 1. Data preparation with framework
        test_data = self._generate_test_data()
        dataset_anchor = self.framework.create_dataset_anchor(test_data)
        
        # 2. Model training with simulation
        training_capsules = self.simulator.prepare_data(test_data)
        training_snapshot = self.framework.train_model_with_provenance(
            model_name="integration_test_model",
            training_capsules=training_capsules,
            training_parameters={"epochs": 5, "lr": 0.01}
        )
        
        # 3. Inference with audit trail
        test_query = "What is the diagnostic recommendation?"
        inference_receipt = self.framework.perform_inference_with_audit(
            model_name="integration_test_model",
            query=test_query,
            ai_output=self.mock_llm.generate_text(test_query),
            training_snapshot=training_snapshot,
            user_id="integration_tester"
        )
        
        # 4. Compliance validation
        compliance_report = self.framework.validate_system_compliance(
            frameworks=["NIST_AI_RMF", "EU_AI_ACT"]
        )
        
        # 5. Generate audit report
        audit_report = self.framework.generate_comprehensive_audit_report()
        
        return {
            "dataset_anchor": dataset_anchor,
            "training_snapshot": training_snapshot,
            "inference_receipt": inference_receipt,
            "compliance_report": compliance_report,
            "audit_report": audit_report
        }
    
    def _generate_test_data(self):
        return [
            {
                "id": "integration_test_1",
                "content": "Test case for CIAF integration",
                "metadata": {
                    "source": "integration_test",
                    "purpose": "system_validation"
                }
            }
        ]

# Run integration test
integration_test = CIAFIntegrationTest()
integration_results = integration_test.test_end_to_end_integration()
print("Integration test completed successfully!")
```

## Best Practices

### 1. Simulation Configuration

```python
# Configure simulation parameters
SIMULATION_CONFIG = {
    "deterministic_mode": True,  # For reproducible results
    "timing_simulation": True,   # Simulate realistic delays
    "memory_limits": {
        "max_capsules_in_memory": 10000,
        "batch_processing_size": 1000
    },
    "logging_level": "INFO",
    "verification_sampling": 0.1  # Verify 10% of data for performance
}

def configure_simulation(config):
    # Apply configuration to all simulation components
    MockLLM.timing_enabled = config["timing_simulation"]
    MLFrameworkSimulator.deterministic_mode = config["deterministic_mode"]
    MLFrameworkSimulator.batch_size = config["memory_limits"]["batch_processing_size"]
```

### 2. Test Data Generation

```python
# Generate realistic test data
def generate_domain_specific_data(domain, sample_count):
    generators = {
        "healthcare": generate_healthcare_data,
        "finance": generate_financial_data,
        "general": generate_general_text_data
    }
    
    generator = generators.get(domain, generators["general"])
    return generator(sample_count)

def generate_healthcare_data(count):
    # Generate realistic but synthetic healthcare data
    conditions = ["hypertension", "diabetes", "asthma", "normal"]
    symptoms = ["chest pain", "fatigue", "headache", "normal"]
    
    data = []
    for i in range(count):
        condition = conditions[i % len(conditions)]
        symptom = symptoms[i % len(symptoms)]
        
        data.append({
            "id": f"patient_{i:04d}",
            "content": f"Patient presenting with {symptom}, history of {condition}",
            "metadata": {
                "domain": "healthcare",
                "synthetic": True,
                "condition_category": condition,
                "symptom_category": symptom
            }
        })
    
    return data
```

### 3. Performance Monitoring

```python
# Monitor simulation performance
class SimulationMonitor:
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_operation(self, operation_name):
        self.start_times[operation_name] = time.time()
    
    def end_operation(self, operation_name, metadata=None):
        if operation_name in self.start_times:
            duration = time.time() - self.start_times[operation_name]
            
            if operation_name not in self.metrics:
                self.metrics[operation_name] = []
            
            self.metrics[operation_name].append({
                "duration": duration,
                "timestamp": time.time(),
                "metadata": metadata or {}
            })
    
    def get_performance_summary(self):
        summary = {}
        for operation, measurements in self.metrics.items():
            durations = [m["duration"] for m in measurements]
            summary[operation] = {
                "count": len(durations),
                "avg_duration": sum(durations) / len(durations),
                "min_duration": min(durations),
                "max_duration": max(durations),
                "total_duration": sum(durations)
            }
        return summary

# Use monitor in simulations
monitor = SimulationMonitor()

monitor.start_operation("capsule_creation")
capsules = simulator.prepare_data(test_data)
monitor.end_operation("capsule_creation", {"data_count": len(test_data)})

performance_summary = monitor.get_performance_summary()
```

## Contributing

When extending the simulation package:

1. **Maintain Realism** — Simulations should reflect real-world behaviors
2. **Performance Considerations** — Support large-scale testing scenarios
3. **Deterministic Results** — Ensure reproducible test outcomes
4. **Comprehensive Coverage** — Test all CIAF component interactions
5. **Clear Documentation** — Document simulation parameters and limitations

## Dependencies

The simulation package depends on:
- `ciaf.provenance` — For creating mock provenance capsules and snapshots
- `ciaf.core` — For cryptographic utilities used in simulation
- `time` — For realistic timing simulation
- `hashlib` — For parameter hashing in ML framework simulation
- `typing` — For type hints and better code clarity

---

*For integration examples and testing patterns, see the [examples folder](../examples/) and [API documentation](../api/).*