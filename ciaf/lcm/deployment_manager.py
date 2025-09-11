"""
CIAF LCM Deployment Manager

Enhanced deployment management with pre-deployment and deployment stages,
including artifact digests, SBOM, approvals, and infrastructure tracking.

Created: 2025-09-09
Last Modified: 2025-09-11
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
from datetime import datetime
from enum import Enum
from typing import Dict, List, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

from ..core import sha256_hash, MerkleTree
from .policy import LCMPolicy, get_default_policy, CommitmentType, DomainType

if TYPE_CHECKING:
    from .model_manager import LCMModelAnchor


class DeploymentStatus(Enum):
    """Deployment status types."""
    PREPARED = "prepared"
    DEPLOYED = "deployed"
    FAILED = "failed"
    ROLLBACK = "rollback"


@dataclass
class BuildArtifact:
    """Build artifact metadata."""
    artifact_type: str  # e.g., "wheel", "docker", "binary"
    artifact_digest: str
    build_timestamp: str
    builder_info: str
    size_bytes: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "artifact_type": self.artifact_type,
            "artifact_digest": self.artifact_digest,
            "build_timestamp": self.build_timestamp,
            "builder_info": self.builder_info,
            "size_bytes": self.size_bytes
        }


@dataclass
class SBOM:
    """Software Bill of Materials."""
    dependencies: Dict[str, str]  # package_name -> version
    security_scan_digest: str
    vulnerability_count: int = 0
    compliance_status: str = "unknown"
    
    def compute_sbom_digest(self) -> str:
        """Compute SBOM digest."""
        sbom_data = {
            "dependencies": self.dependencies,
            "security_scan_digest": self.security_scan_digest,
            "vulnerability_count": self.vulnerability_count,
            "compliance_status": self.compliance_status
        }
        canonical_json = json.dumps(sbom_data, sort_keys=True, separators=(',', ':'))
        return sha256_hash(canonical_json.encode('utf-8'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dependencies": self.dependencies,
            "security_scan_digest": self.security_scan_digest,
            "vulnerability_count": self.vulnerability_count,
            "compliance_status": self.compliance_status,
            "sbom_digest": self.compute_sbom_digest()
        }


class LCMPreDeploymentAnchor:
    """Pre-deployment anchor for LCM."""
    
    def __init__(
        self,
        predeployment_id: str,
        build_artifact: BuildArtifact,
        sbom: SBOM,
        approval_ticket_id: str,
        intended_env: str,
        intended_region: str,
        rollout_plan_digest: str = None,
        policy: LCMPolicy = None
    ):
        """
        Initialize pre-deployment anchor.
        
        Args:
            predeployment_id: Unique pre-deployment identifier
            build_artifact: Build artifact metadata
            sbom: Software Bill of Materials
            approval_ticket_id: Approval ticket/workflow ID
            intended_env: Intended deployment environment
            intended_region: Intended deployment region
            rollout_plan_digest: Digest of rollout plan document
            policy: LCM policy
        """
        self.predeployment_id = predeployment_id
        self.build_artifact = build_artifact
        self.sbom = sbom
        self.approval_ticket_id = approval_ticket_id
        self.intended_env = intended_env
        self.intended_region = intended_region
        self.rollout_plan_digest = rollout_plan_digest or "default_rollout"
        self.policy = policy or get_default_policy()
        
        # Compute pre-deployment hash
        self.predeployment_hash = self._compute_predeployment_hash()
        self.anchor_id = f"pd_{self.predeployment_hash[:8]}..."
        self.status = DeploymentStatus.PREPARED
        
        print(f"🧰 Pre-deployment anchor '{self.predeployment_id}' created: {self.anchor_id}")
    
    def _compute_predeployment_hash(self) -> str:
        """Compute pre-deployment hash."""
        predeployment_data = {
            "predeployment_id": self.predeployment_id,
            "artifact_digest": self.build_artifact.artifact_digest,
            "sbom_digest": self.sbom.compute_sbom_digest(),
            "approval_ticket_id": self.approval_ticket_id,
            "intended_env": self.intended_env,
            "intended_region": self.intended_region,
            "rollout_plan_digest": self.rollout_plan_digest
        }
        canonical_json = json.dumps(predeployment_data, sort_keys=True, separators=(',', ':'))
        return sha256_hash(canonical_json.encode('utf-8'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "artifact_digest": self.build_artifact.artifact_digest[:12] + "...",
            "sbom_digest": self.sbom.compute_sbom_digest()[:8] + "...",
            "approval_ticket": self.approval_ticket_id,
            "intended_env": self.intended_env,
            "intended_region": self.intended_region,
            "anchor": self.anchor_id
        }


class LCMDeploymentAnchor:
    """Deployment anchor for LCM."""
    
    def __init__(
        self,
        deployment_id: str,
        predeployment_anchor: LCMPreDeploymentAnchor,
        deployment_time: str = None,
        deployment_env: str = None,
        location: str = None,
        infrastructure_hash: str = None,
        config_digest: str = None,
        policy: LCMPolicy = None
    ):
        """
        Initialize deployment anchor.
        
        Args:
            deployment_id: Unique deployment identifier
            predeployment_anchor: Associated pre-deployment anchor
            deployment_time: Actual deployment timestamp
            deployment_env: Actual deployment environment
            location: Actual deployment location
            infrastructure_hash: Infrastructure specification hash
            config_digest: Runtime configuration digest
            policy: LCM policy
        """
        self.deployment_id = deployment_id
        self.predeployment_anchor = predeployment_anchor
        self.deployment_time = deployment_time or datetime.now().isoformat()
        self.deployment_env = deployment_env or predeployment_anchor.intended_env
        self.location = location or predeployment_anchor.intended_region
        self.infrastructure_hash = infrastructure_hash or "default_infra"
        self.config_digest = config_digest or "default_config"
        self.policy = policy or get_default_policy()
        
        # Compute deployment hash
        self.deployment_hash = self._compute_deployment_hash()
        self.anchor_id = f"dp_{self.deployment_hash[:8]}..."
        self.status = DeploymentStatus.DEPLOYED
        
        # Compute intent-to-actual digest
        self.intent_to_actual_digest = self._compute_intent_to_actual_digest()
        
        print(f"🚀 Deployment anchor '{self.deployment_id}' created: {self.anchor_id}")
        print(f"   🔗 Intent→actual digest: {self.intent_to_actual_digest[:16]}...")
    
    def _compute_deployment_hash(self) -> str:
        """Compute deployment hash."""
        deployment_data = {
            "deployment_id": self.deployment_id,
            "deployment_time": self.deployment_time,
            "deployment_env": self.deployment_env,
            "location": self.location,
            "deployment_commit": self.predeployment_anchor.build_artifact.artifact_digest,
            "infrastructure_hash": self.infrastructure_hash,
            "config_digest": self.config_digest
        }
        canonical_json = json.dumps(deployment_data, sort_keys=True, separators=(',', ':'))
        return sha256_hash(canonical_json.encode('utf-8'))
    
    def _compute_intent_to_actual_digest(self) -> str:
        """Compute intent-to-actual binding digest."""
        binding_data = f"{self.predeployment_anchor.predeployment_hash}||{self.deployment_hash}"
        return sha256_hash(binding_data.encode('utf-8'))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "status": self.status.value,
            "time": self.deployment_time,
            "env": self.deployment_env,
            "location": self.location,
            "deployment_commit": self.predeployment_anchor.build_artifact.artifact_digest[:12] + "...",
            "infrastructure_hash": self.infrastructure_hash[:8] + "...",
            "intent_to_actual_digest": self.intent_to_actual_digest[:8] + "...",
            "anchor": self.anchor_id
        }


class LCMDeploymentManager:
    """Enhanced deployment manager for LCM."""
    
    def __init__(self, policy: LCMPolicy = None):
        """Initialize LCM deployment manager."""
        self.policy = policy or get_default_policy()
        self.predeployment_anchors: Dict[str, LCMPreDeploymentAnchor] = {}
        self.deployment_anchors: Dict[str, LCMDeploymentAnchor] = {}
    
    def create_predeployment_anchor(
        self,
        predeployment_id: str,
        artifact_digest: str,
        dependencies: Dict[str, str],
        approval_ticket_id: str,
        intended_env: str,
        intended_region: str,
        security_scan_results: Dict[str, Any] = None
    ) -> LCMPreDeploymentAnchor:
        """
        Create pre-deployment anchor.
        
        Args:
            predeployment_id: Unique pre-deployment identifier
            artifact_digest: Build artifact digest
            dependencies: Package dependencies
            approval_ticket_id: Approval ticket ID
            intended_env: Intended deployment environment
            intended_region: Intended deployment region
            security_scan_results: Security scan results
            
        Returns:
            LCMPreDeploymentAnchor instance
        """
        print(f"🧰 Creating pre-deployment anchor: {predeployment_id}")
        
        # Create build artifact
        build_artifact = BuildArtifact(
            artifact_type="docker",
            artifact_digest=artifact_digest,
            build_timestamp=datetime.now().isoformat(),
            builder_info="CIAF-Builder-v1.0"
        )
        
        # Create SBOM
        security_scan_results = security_scan_results or {"vulnerabilities": 0, "status": "clean"}
        sbom = SBOM(
            dependencies=dependencies,
            security_scan_digest=sha256_hash(json.dumps(security_scan_results, sort_keys=True).encode()),
            vulnerability_count=security_scan_results.get("vulnerabilities", 0),
            compliance_status=security_scan_results.get("status", "unknown")
        )
        
        # Create pre-deployment anchor
        anchor = LCMPreDeploymentAnchor(
            predeployment_id=predeployment_id,
            build_artifact=build_artifact,
            sbom=sbom,
            approval_ticket_id=approval_ticket_id,
            intended_env=intended_env,
            intended_region=intended_region,
            policy=self.policy
        )
        
        self.predeployment_anchors[predeployment_id] = anchor
        return anchor
    
    def create_deployment_anchor(
        self,
        deployment_id: str,
        predeployment_id: str,
        actual_env: str = None,
        actual_location: str = None,
        infrastructure_spec: Dict[str, Any] = None,
        runtime_config: Dict[str, Any] = None
    ) -> LCMDeploymentAnchor:
        """
        Create deployment anchor from pre-deployment.
        
        Args:
            deployment_id: Unique deployment identifier
            predeployment_id: Associated pre-deployment ID
            actual_env: Actual deployment environment
            actual_location: Actual deployment location
            infrastructure_spec: Infrastructure specification
            runtime_config: Runtime configuration
            
        Returns:
            LCMDeploymentAnchor instance
        """
        print(f"🚀 Creating deployment anchor: {deployment_id}")
        
        predeployment_anchor = self.predeployment_anchors.get(predeployment_id)
        if not predeployment_anchor:
            raise ValueError(f"Pre-deployment anchor not found: {predeployment_id}")
        
        # Compute infrastructure and config hashes
        infrastructure_hash = sha256_hash(
            json.dumps(infrastructure_spec or {}, sort_keys=True).encode()
        )
        config_digest = sha256_hash(
            json.dumps(runtime_config or {}, sort_keys=True).encode()
        )
        
        # Create deployment anchor
        anchor = LCMDeploymentAnchor(
            deployment_id=deployment_id,
            predeployment_anchor=predeployment_anchor,
            deployment_env=actual_env,
            location=actual_location,
            infrastructure_hash=infrastructure_hash,
            config_digest=config_digest,
            policy=self.policy
        )
        
        self.deployment_anchors[deployment_id] = anchor
        return anchor
    
    def get_predeployment_anchor(self, predeployment_id: str) -> Optional[LCMPreDeploymentAnchor]:
        """Get pre-deployment anchor by ID."""
        return self.predeployment_anchors.get(predeployment_id)
    
    def get_deployment_anchor(self, deployment_id: str) -> Optional[LCMDeploymentAnchor]:
        """Get deployment anchor by ID."""
        return self.deployment_anchors.get(deployment_id)
    
    def format_predeployment_summary(self, predeployment_id: str) -> str:
        """Format pre-deployment summary for pretty printing."""
        anchor = self.get_predeployment_anchor(predeployment_id)
        if not anchor:
            return f"Pre-deployment {predeployment_id} not found"
        
        lines = [
            f"  artifact_digest: {anchor.build_artifact.artifact_digest[:12]}...",
            f"  sbom_digest: {anchor.sbom.compute_sbom_digest()[:8]}...     approvals: {anchor.approval_ticket_id}",
            f"  intended_env: {anchor.intended_env}  intended_region: {anchor.intended_region}",
            f"  ✅ predeployment_anchor: {anchor.anchor_id} (status={anchor.status.value})"
        ]
        return "\n".join(lines)
    
    def format_deployment_summary(self, deployment_id: str) -> str:
        """Format deployment summary for pretty printing."""
        anchor = self.get_deployment_anchor(deployment_id)
        if not anchor:
            return f"Deployment {deployment_id} not found"
        
        lines = [
            f"  time: {anchor.deployment_time}",
            f"  env: {anchor.deployment_env}   location: {anchor.location}",
            f"  deployment_commit: {anchor.predeployment_anchor.build_artifact.artifact_digest[:12]}...  infra_hash: {anchor.infrastructure_hash[:8]}...",
            f"  ✅ deployment_anchor: {anchor.anchor_id} (status={anchor.status.value})",
            f"  🔗 intent→actual: H({anchor.predeployment_anchor.anchor_id} || {anchor.anchor_id}) => {anchor.intent_to_actual_digest[:8]}..."
        ]
        return "\n".join(lines)
    
    def simulate_predeployment_anchor(
        self,
        predeployment_id: str,
        model_anchor: 'LCMModelAnchor'
    ) -> LCMPreDeploymentAnchor:
        """
        Simulate pre-deployment anchor for demonstration.
        
        Args:
            predeployment_id: Pre-deployment identifier
            model_anchor: Model anchor
            
        Returns:
            LCMPreDeploymentAnchor instance
        """
        print(f"📋 Creating pre-deployment anchor: {predeployment_id}")
        
        # Create mock build artifact
        mock_artifact = BuildArtifact(
            artifact_type="docker",
            artifact_digest=sha256_hash(f"artifact_{model_anchor.model_name}".encode('utf-8')),
            build_timestamp=datetime.now().isoformat(),
            builder_info=f"builder-{model_anchor.model_name}",
            size_bytes=1024*1024*50  # 50MB
        )
        
        # Create mock SBOM
        mock_sbom = SBOM(
            dependencies={
                "pytorch": "1.9.0",
                "numpy": "1.21.0",
                model_anchor.model_name: model_anchor.version
            },
            security_scan_digest="scan_" + sha256_hash("mock_scan".encode('utf-8'))[:16],
            vulnerability_count=0,
            compliance_status="passed"
        )
        
        # Create pre-deployment anchor
        anchor = self.create_predeployment_anchor(
            predeployment_id=predeployment_id,
            artifact_digest=mock_artifact.artifact_digest,
            dependencies=mock_sbom.dependencies,
            approval_ticket_id="APR-001",
            intended_env="production",
            intended_region="us-east-1"
        )
        
        print(f"✅ Pre-deployment anchor created: {anchor.anchor_id}")
        return anchor
    
    def simulate_deployment_anchor(
        self,
        deployment_id: str,
        predeployment_anchor: LCMPreDeploymentAnchor
    ) -> LCMDeploymentAnchor:
        """
        Simulate deployment anchor for demonstration.
        
        Args:
            deployment_id: Deployment identifier
            predeployment_anchor: Pre-deployment anchor
            
        Returns:
            LCMDeploymentAnchor instance
        """
        print(f"🚀 Creating deployment anchor: {deployment_id}")
        
        # Create deployment anchor
        anchor = self.create_deployment_anchor(
            deployment_id=deployment_id,
            predeployment_id=predeployment_anchor.predeployment_id,
            actual_env="production",
            actual_location="us-east-1a",
            infrastructure_spec={
                "container_registry": "ecr.us-east-1.amazonaws.com",
                "kubernetes_cluster": "prod-cluster",
                "service_mesh": "istio"
            }
        )
        
        print(f"✅ Deployment anchor created: {anchor.anchor_id}")
        return anchor
