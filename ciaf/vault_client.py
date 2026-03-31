"""
CIAF Vault Client

Python client for connecting to the CIAF Vault web application.
Enables sending events, receipts, and audit data to the vault for
real-time monitoring and governance.

Created: 2026-03-31
Author: Denzil James Greenwood
Version: 1.0.0
"""

import json
import hashlib
import requests
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Union
from dataclasses import dataclass, asdict


@dataclass
class VaultConfig:
    """Configuration for CIAF Vault connection."""
    vault_url: str = "http://localhost:3000"
    timeout: int = 30
    verify_ssl: bool = True
    
    def __post_init__(self):
        # Remove trailing slash
        self.vault_url = self.vault_url.rstrip('/')


class VaultClient:
    """
    Client for sending events and data to CIAF Vault.
    
    Example:
        >>> client = VaultClient("https://vault.example.com")
        >>> receipt = client.send_inference_event(
        ...     model_name="credit-model-v1",
        ...     prediction={"score": 750, "approved": True}
        ... )
    """
    
    def __init__(self, vault_url: str = "http://localhost:3000", **config_kwargs):
        """
        Initialize vault client.
        
        Args:
            vault_url: Base URL of CIAF Vault (e.g., "http://localhost:3000")
            **config_kwargs: Additional configuration options
        """
        self.config = VaultConfig(vault_url=vault_url, **config_kwargs)
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'CIAF-Vault-Client/1.0.0'
        })
    
    def _make_request(self, method: str, endpoint: str, data: Optional[Dict] = None) -> Dict:
        """Make HTTP request to vault."""
        url = f"{self.config.vault_url}{endpoint}"
        
        try:
            if method == "GET":
                response = self.session.get(
                    url, 
                    params=data,
                    timeout=self.config.timeout,
                    verify=self.config.verify_ssl
                )
            else:
                response = self.session.request(
                    method,
                    url,
                    json=data,
                    timeout=self.config.timeout,
                    verify=self.config.verify_ssl
                )
            
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': f'Vault request failed: {str(e)}'
            }
    
    def send_core_event(
        self,
        model_name: str,
        event_type: str,
        stage: str,
        metadata: Dict[str, Any],
        model_version: Optional[str] = None,
        org_id: Optional[str] = None,
        user_id: Optional[str] = None,
        generate_receipt: bool = True
    ) -> Dict:
        """
        Send core AI lifecycle event to vault.
        
        Args:
            model_name: Name of the model
            event_type: Type of event (training, inference, deployment, monitoring)
            stage: LCM stage (A-H)
            metadata: Event-specific metadata
            model_version: Model version identifier
            org_id: Organization ID
            user_id: User ID
            generate_receipt: Generate cryptographic receipt
        
        Returns:
            API response with receipt if generated
        """
        event_data = {
            'model_name': model_name,
            'model_version': model_version or '1.0.0',
            'event_type': event_type,
            'stage': stage,
            'lcm_stage': event_type,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'metadata': metadata,
            'org_id': org_id or 'default',
            'user_id': user_id or 'system'
        }
        
        payload = {
            'event_type': 'core',
            'event_data': event_data,
            'generate_receipt': generate_receipt
        }
        
        return self._make_request('POST', '/api/events/ingest', payload)
    
    def send_inference_event(
        self,
        model_name: str,
        input_data: Any,
        prediction: Any,
        model_version: Optional[str] = None,
        confidence: Optional[float] = None,
        latency_ms: Optional[float] = None,
        **kwargs
    ) -> Dict:
        """
        Send inference event to vault.
        
        Args:
            model_name: Name of the model
            input_data: Input features/data (will be hashed)
            prediction: Model prediction output
            model_version: Model version
            confidence: Prediction confidence score
            latency_ms: Inference latency in milliseconds
            **kwargs: Additional metadata
        
        Returns:
            API response with receipt
        """
        # Hash input data
        input_hash = hashlib.sha256(
            json.dumps(input_data, sort_keys=True).encode()
        ).hexdigest()
        
        metadata = {
            'input_hash': input_hash,
            'prediction': prediction,
            'confidence': confidence,
            'latency_ms': latency_ms,
            **kwargs
        }
        
        return self.send_core_event(
            model_name=model_name,
            model_version=model_version,
            event_type='inference',
            stage='F',  # Production inference
            metadata=metadata
        )
    
    def send_training_event(
        self,
        model_name: str,
        epoch: int,
        metrics: Dict[str, float],
        model_version: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Send training event to vault.
        
        Args:
            model_name: Name of the model
            epoch: Training epoch number
            metrics: Training metrics (loss, accuracy, etc.)
            model_version: Model version
            **kwargs: Additional metadata
        
        Returns:
            API response with receipt
        """
        metadata = {
            'epoch': epoch,
            'metrics': metrics,
            **kwargs
        }
        
        return self.send_core_event(
            model_name=model_name,
            model_version=model_version,
            event_type='training',
            stage='B',  # Training stage
            metadata=metadata
        )
    
    def send_web_event(
        self,
        user_id: str,
        domain: str,
        tool_name: str,
        event_type: str,
        content: Optional[str] = None,
        policy_decision: str = "allow",
        sensitivity_score: float = 0.0,
        pii_detected: bool = False,
        **kwargs
    ) -> Dict:
        """
        Send web AI governance event to vault.
        
        Args:
            user_id: User identifier
            domain: Web domain (e.g., "chat.openai.com")
            tool_name: AI tool name (e.g., "ChatGPT")
            event_type: Event type (prompt_submit, response_received)
            content: Content text (will be hashed)
            policy_decision: Policy decision (allow, warn, redact, block)
            sensitivity_score: Content sensitivity score (0-100)
            pii_detected: Whether PII was detected
            **kwargs: Additional metadata
        
        Returns:
            API response
        """
        content_hash = None
        if content:
            content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        event_data = {
            'user_id': user_id,
            'domain': domain,
            'tool_name': tool_name,
            'event_type': event_type,
            'content_hash': content_hash,
            'policy_decision': policy_decision,
            'sensitivity_score': sensitivity_score,
            'pii_detected': pii_detected,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **kwargs
        }
        
        payload = {
            'event_type': 'web',
            'event_data': event_data,
            'generate_receipt': True
        }
        
        return self._make_request('POST', '/api/events/ingest', payload)
    
    def register_agent(
        self,
        principal_id: str,
        display_name: str,
        principal_type: str = "agent",
        roles: Optional[List[str]] = None,
        attributes: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None
    ) -> Dict:
        """
        Register an agent identity in the vault.
        
        Args:
            principal_id: Unique agent identifier
            display_name: Human-readable agent name
            principal_type: Type (agent, service, human, system)
            roles: List of role names
            attributes: Additional attributes (tenant_id, environment, etc.)
            created_by: Creator identifier
        
        Returns:
            API response with agent identity
        """
        payload = {
            'principal_id': principal_id,
            'display_name': display_name,
            'principal_type': principal_type,
            'roles': roles or [],
            'attributes': attributes or {},
            'created_by': created_by or 'system'
        }
        
        return self._make_request('POST', '/api/agents/register', payload)
    
    def execute_agent_action(
        self,
        principal_id: str,
        action: str,
        resource_id: str,
        resource_type: str,
        params: Optional[Dict] = None,
        justification: Optional[str] = None,
        correlation_id: Optional[str] = None,
        elevation_grant_id: Optional[str] = None
    ) -> Dict:
        """
        Execute an agent action with IAM/PAM enforcement.
        
        Args:
            principal_id: Agent identifier
            action: Action to perform (e.g., "read", "approve_payment")
            resource_id: Resource identifier
            resource_type: Resource type
            params: Action parameters
            justification: Reason for action
            correlation_id: Request correlation ID
            elevation_grant_id: Privilege elevation grant ID (if needed)
        
        Returns:
            API response with authorization decision and receipt
        """
        payload = {
            'principal_id': principal_id,
            'action': action,
            'resource': {
                'resource_id': resource_id,
                'resource_type': resource_type
            },
            'params': params or {},
            'justification': justification,
            'correlation_id': correlation_id,
            'elevation_grant_id': elevation_grant_id
        }
        
        return self._make_request('POST', '/api/agents/actions/execute', payload)
    
    def grant_elevation(
        self,
        principal_id: str,
        elevated_role: str,
        approved_by: str,
        purpose: str,
        valid_until: str,
        justification: Optional[str] = None,
        scope: Optional[Dict] = None,
        max_uses: Optional[int] = None
    ) -> Dict:
        """
        Grant temporary privilege elevation to an agent.
        
        Args:
            principal_id: Agent identifier
            elevated_role: Role to grant temporarily
            approved_by: Approver identifier
            purpose: Purpose of elevation
            valid_until: ISO timestamp for expiration
            justification: Detailed justification
            scope: Scope restrictions
            max_uses: Maximum uses allowed
        
        Returns:
            API response with elevation grant
        """
        payload = {
            'principal_id': principal_id,
            'elevated_role': elevated_role,
            'approved_by': approved_by,
            'purpose': purpose,
            'valid_until': valid_until,
            'justification': justification,
            'scope': scope or {},
            'max_uses': max_uses
        }
        
        return self._make_request('POST', '/api/agents/elevations/grant', payload)
    
    def get_agent(self, principal_id: str) -> Dict:
        """Get agent details and statistics."""
        return self._make_request('GET', f'/api/agents/{principal_id}')
    
    def list_agents(
        self,
        status: Optional[str] = None,
        principal_type: Optional[str] = None,
        limit: int = 50,
        offset: int = 0
    ) -> Dict:
        """
        List agents with optional filtering.
        
        Args:
            status: Filter by status (active, suspended, revoked)
            principal_type: Filter by type (agent, service, human, system)
            limit: Max results to return
            offset: Pagination offset
        
        Returns:
            API response with agent list
        """
        params = {'limit': limit, 'offset': offset}
        if status:
            params['status'] = status
        if principal_type:
            params['principal_type'] = principal_type
        
        return self._make_request('GET', '/api/agents', params)
    
    def get_dashboard_stats(self) -> Dict:
        """Get dashboard statistics."""
        return self._make_request('GET', '/api/stats')
    
    def health_check(self) -> bool:
        """Check if vault is accessible."""
        try:
            response = self._make_request('GET', '/api/stats')
            return response.get('success', False)
        except:
            return False


# Convenience function for quick usage
def create_client(vault_url: str = "http://localhost:3000") -> VaultClient:
    """
    Create a CIAF Vault client.
    
    Args:
        vault_url: Base URL of CIAF Vault
    
    Returns:
        Configured VaultClient instance
    """
    return VaultClient(vault_url=vault_url)
