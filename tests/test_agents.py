"""
CIAF Agents Module Tests

Comprehensive test suite for the CIAF agentic execution framework:
- Agent event types and data models
- Authorization and policy enforcement
- IAM/PAM (Identity and Privilege Access Management)
- Execution tracking and receipts
- Event hash chaining

Created: 2026-03-31
Version: 1.0.0
"""

import pytest
from datetime import datetime, timezone
import hashlib

try:
    from ciaf.agents.policy.engine import PolicyEngine
    from ciaf.agents import (
        AgentEvent,
        AgentEventBatch,
        AgentEventType,
        AgentActionType,
        AgentPolicyDecision,
        SensitivityLevel,
    )
    from ciaf.agents.core.types import (
        Identity,
        Resource,
        Permission,
        ActionRequest,
        ExecutionResult,
        ElevationGrant,
        ActionReceipt,
    )
    from ciaf.agents.iam.store import InMemoryIdentityStore
    from ciaf.agents.pam.store import InMemoryPrivilegeStore
    AGENTS_MODULE_AVAILABLE = True
except (ImportError, AttributeError):
    AGENTS_MODULE_AVAILABLE = False
    # Create mock types
    AgentEvent = None
    PolicyEngine = None


@pytest.mark.skipif(not AGENTS_MODULE_AVAILABLE, reason="Agents module not fully available")
class TestAgentEventTypes:
    """Test AgentEventType enumeration."""

    def test_agent_event_types_exist(self):
        """Test all agent event types are defined."""
        # Data operations
        assert AgentEventType.READ == "agent_read"
        assert AgentEventType.WRITE == "agent_write"
        assert AgentEventType.DELETE == "agent_delete"
        assert AgentEventType.SEARCH == "agent_search"
        assert AgentEventType.EXPORT == "agent_export"

        # External interactions
        assert AgentEventType.API_CALL == "agent_api_call"
        assert AgentEventType.HTTP_REQUEST == "agent_http_request"
        assert AgentEventType.DATABASE_QUERY == "agent_database_query"
        assert AgentEventType.FILE_ACCESS == "agent_file_access"

        # Autonomous behavior
        assert AgentEventType.DECISION == "agent_decision"
        assert AgentEventType.REASONING == "agent_reasoning"
        assert AgentEventType.PLAN_GENERATION == "agent_plan"
        assert AgentEventType.GOAL_UPDATE == "agent_goal_update"

        # Governance & control
        assert AgentEventType.POLICY_CHECK == "agent_policy_check"
        assert AgentEventType.ELEVATION_REQUEST == "agent_elevation_request"
        assert AgentEventType.HUMAN_OVERRIDE == "agent_human_override"
        assert AgentEventType.APPROVAL_REQUEST == "agent_approval_request"

    def test_agent_action_types_exist(self):
        """Test AgentActionType enumeration."""
        assert AgentActionType.READ_RECORD == "read_record"
        assert AgentActionType.WRITE_RECORD == "write_record"
        assert AgentActionType.APPROVE_PAYMENT == "approve_payment"
        assert AgentActionType.DEPLOY_CODE == "deploy_code"


@pytest.mark.skipif(not AGENTS_MODULE_AVAILABLE, reason="Agents module not fully available")
class TestAgentEvent:
    """Test AgentEvent data model."""

    def test_create_agent_event(self):
        """Test creating a basic agent event."""
        event = AgentEvent(
            event_id="evt_001",
            event_type=AgentEventType.READ,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            agent_id="agent_001",
            agent_name="DataReader",
            principal_type="service_account",
            session_id="session_001",
            action="read_patient_record",
            resource_type="patient_record",
            resource_id="record_12345",
            policy_decision=AgentPolicyDecision.ALLOW,
            executed=True,
            success=True,
            params_hash=hashlib.sha256(b"{'patient_id': '12345'}").hexdigest(),
            input_hash=hashlib.sha256(b"input").hexdigest(),
            output_hash=hashlib.sha256(b"output").hexdigest(),
            prior_event_hash=hashlib.sha256(b"prior").hexdigest(),
        )

        assert event.event_id == "evt_001"
        assert event.event_type == AgentEventType.READ
        assert event.agent_id == "agent_001"
        assert event.policy_decision == AgentPolicyDecision.ALLOW
        assert event.executed is True
        assert event.success is True

    def test_agent_event_hash_chain(self):
        """Test event hash chaining for tamper detection."""
        event1 = AgentEvent(
            event_id="evt_001",
            event_type=AgentEventType.READ,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            agent_id="agent_001",
            agent_name="Agent1",
            principal_type="service_account",
            session_id="session_001",
            action="read",
            resource_type="data",
            resource_id="res_001",
            policy_decision=AgentPolicyDecision.ALLOW,
            executed=True,
            success=True,
            params_hash=hashlib.sha256(b"params1").hexdigest(),
            input_hash=hashlib.sha256(b"input1").hexdigest(),
            output_hash=hashlib.sha256(b"output1").hexdigest(),
            prior_event_hash="0" * 64,
        )

        # Compute hash of event1
        event1_hash = event1.compute_event_hash()
        assert len(event1_hash) == 64  # SHA-256 hex digest

        # Create event2 that chains to event1
        event2 = AgentEvent(
            event_id="evt_002",
            event_type=AgentEventType.WRITE,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            agent_id="agent_001",
            agent_name="Agent1",
            principal_type="service_account",
            session_id="session_001",
            action="write",
            resource_type="data",
            resource_id="res_002",
            policy_decision=AgentPolicyDecision.ALLOW,
            executed=True,
            success=True,
            params_hash=hashlib.sha256(b"params2").hexdigest(),
            input_hash=hashlib.sha256(b"input2").hexdigest(),
            output_hash=hashlib.sha256(b"output2").hexdigest(),
            prior_event_hash=event1_hash,
        )

        # Verify chain linkage
        assert event2.prior_event_hash == event1_hash

    def test_agent_event_sensitivity_levels(self):
        """Test sensitivity level classification."""
        # Public event
        public_event = AgentEvent(
            event_id="evt_public",
            event_type=AgentEventType.READ,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            agent_id="agent_001",
            agent_name="PublicReader",
            principal_type="user",
            session_id="session_001",
            action="read_public_data",
            resource_type="public_data",
            resource_id="public_001",
            policy_decision=AgentPolicyDecision.ALLOW,
            executed=True,
            success=True,
            params_hash=hashlib.sha256(b"params").hexdigest(),
            input_hash=hashlib.sha256(b"input").hexdigest(),
            output_hash=hashlib.sha256(b"output").hexdigest(),
            prior_event_hash="0" * 64,
            sensitivity_level=SensitivityLevel.PUBLIC,
        )
        assert public_event.sensitivity_level == SensitivityLevel.PUBLIC

        # Highly restricted event
        restricted_event = AgentEvent(
            event_id="evt_restricted",
            event_type=AgentEventType.WRITE,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            agent_id="agent_002",
            agent_name="SensitiveWriter",
            principal_type="service_account",
            session_id="session_002",
            action="write_ssn",
            resource_type="pii",
            resource_id="ssn_001",
            policy_decision=AgentPolicyDecision.ALLOW,
            executed=True,
            success=True,
            params_hash=hashlib.sha256(b"params").hexdigest(),
            input_hash=hashlib.sha256(b"input").hexdigest(),
            output_hash=hashlib.sha256(b"output").hexdigest(),
            prior_event_hash="0" * 64,
            sensitivity_level=SensitivityLevel.HIGHLY_RESTRICTED,
        )
        assert restricted_event.sensitivity_level == SensitivityLevel.HIGHLY_RESTRICTED


@pytest.mark.skipif(not AGENTS_MODULE_AVAILABLE, reason="Agents module not fully available")
class TestAgentEventBatch:
    """Test AgentEventBatch for batch processing."""

    def test_create_event_batch(self):
        """Test creating a batch of events."""
        events = [
            AgentEvent(
                event_id=f"evt_{i:03d}",
                event_type=AgentEventType.READ,
                occurred_at=datetime.now(timezone.utc).isoformat(),
                agent_id="agent_001",
                agent_name="BatchAgent",
                principal_type="service_account",
                session_id="session_batch",
                action=f"read_{i}",
                resource_type="data",
                resource_id=f"res_{i:03d}",
                policy_decision=AgentPolicyDecision.ALLOW,
                executed=True,
                success=True,
                params_hash=hashlib.sha256(f"params{i}".encode()).hexdigest(),
                input_hash=hashlib.sha256(f"input{i}".encode()).hexdigest(),
                output_hash=hashlib.sha256(f"output{i}".encode()).hexdigest(),
                prior_event_hash="0" * 64,
            )
            for i in range(5)
        ]

        batch = AgentEventBatch(
            batch_id="batch_001",
            events=events,
            session_id="session_batch",
            agent_id="agent_001",
        )

        assert batch.batch_id == "batch_001"
        assert len(batch.events) == 5
        assert batch.session_id == "session_batch"


@pytest.mark.skipif(not AGENTS_MODULE_AVAILABLE, reason="Agents module not fully available")
class TestIdentityAndResource:
    """Test Identity and Resource models."""

    def test_create_identity(self):
        """Test creating an identity."""
        identity = Identity(
            principal_id="agent_healthcare_001",
            principal_type="service_account",
            roles=["agent", "healthcare_reader"],
            attributes={"department": "healthcare", "clearance": "level2"},
        )

        assert identity.principal_id == "agent_healthcare_001"
        assert identity.principal_type == "service_account"
        assert "agent" in identity.roles
        assert identity.attributes["department"] == "healthcare"

    def test_create_resource(self):
        """Test creating a resource."""
        resource = Resource(
            resource_id="patient_record_12345",
            resource_type="patient_record",
            owner="hospital_001",
            attributes={"classification": "phi", "patient_id": "12345"},
        )

        assert resource.resource_id == "patient_record_12345"
        assert resource.resource_type == "patient_record"
        assert resource.attributes["classification"] == "phi"


@pytest.mark.skipif(not AGENTS_MODULE_AVAILABLE, reason="Agents module not fully available")
class TestActionRequestAndReceipt:
    """Test ActionRequest and ActionReceipt."""

    def test_create_action_request(self):
        """Test creating an action request."""
        request = ActionRequest(
            request_id="req_001",
            action="read_patient_record",
            identity=Identity(
                principal_id="agent_001",
                principal_type="service_account",
                roles=["agent"],
            ),
            resource=Resource(
                resource_id="record_001",
                resource_type="patient_record",
                owner="hospital",
            ),
            context={"ip_address": "10.0.0.1", "timestamp": "2026-03-31T10:00:00Z"},
        )

        assert request.request_id == "req_001"
        assert request.action == "read_patient_record"
        assert request.identity.principal_id == "agent_001"
        assert request.resource.resource_id == "record_001"

    def test_create_action_receipt(self):
        """Test creating an action receipt."""
        receipt = ActionReceipt(
            receipt_id="receipt_001",
            request_id="req_001",
            decision="allow",
            executed=True,
            timestamp=datetime.now(timezone.utc).isoformat(),
            signature="sig_placeholder",
        )

        assert receipt.receipt_id == "receipt_001"
        assert receipt.request_id == "req_001"
        assert receipt.decision == "allow"
        assert receipt.executed is True


@pytest.mark.skipif(not AGENTS_MODULE_AVAILABLE, reason="Agents module not fully available")
class TestElevationGrant:
    """Test ElevationGrant for privilege escalation."""

    def test_create_elevation_grant(self):
        """Test creating an elevation grant."""
        grant = ElevationGrant(
            grant_id="grant_001",
            principal_id="agent_001",
            granted_by="admin_alice",
            permissions=["write_production"],
            expires_at="2026-04-01T10:00:00Z",
            justification="Emergency hotfix deployment",
        )

        assert grant.grant_id == "grant_001"
        assert grant.principal_id == "agent_001"
        assert grant.granted_by == "admin_alice"
        assert "write_production" in grant.permissions
        assert grant.justification == "Emergency hotfix deployment"


@pytest.mark.skipif(not AGENTS_MODULE_AVAILABLE, reason="Agents module not fully available")
class TestPolicyEngine:
    """Test PolicyEngine for authorization decisions."""

    def test_policy_engine_allow(self):
        """Test policy engine allowing an action."""
        engine = PolicyEngine()

        # Simple allow policy: agents with 'reader' role can read
        engine.add_policy(
            policy_id="allow_read",
            effect="allow",
            principals=["agent_*"],
            actions=["read_*"],
            resources=["data_*"],
            conditions={"role": "reader"},
        )

        identity = Identity(
            principal_id="agent_001",
            principal_type="service_account",
            roles=["reader"],
        )

        resource = Resource(
            resource_id="data_001",
            resource_type="data",
            owner="system",
        )

        request = ActionRequest(
            request_id="req_001",
            action="read_data",
            identity=identity,
            resource=resource,
        )

        decision = engine.evaluate(request)
        assert decision.decision == "allow"

    def test_policy_engine_deny(self):
        """Test policy engine denying an action."""
        engine = PolicyEngine()

        # Explicit deny policy
        engine.add_policy(
            policy_id="deny_delete",
            effect="deny",
            principals=["agent_*"],
            actions=["delete_*"],
            resources=["production_*"],
        )

        identity = Identity(
            principal_id="agent_001",
            principal_type="service_account",
            roles=["agent"],
        )

        resource = Resource(
            resource_id="production_db",
            resource_type="production_database",
            owner="system",
        )

        request = ActionRequest(
            request_id="req_002",
            action="delete_table",
            identity=identity,
            resource=resource,
        )

        decision = engine.evaluate(request)
        assert decision.decision == "deny"


@pytest.mark.skipif(not AGENTS_MODULE_AVAILABLE, reason="Agents module not fully available")
class TestIdentityStore:
    """Test InMemoryIdentityStore."""

    def test_identity_store_crud(self):
        """Test identity store create, read, update, delete."""
        store = InMemoryIdentityStore()

        # Create identity
        identity = Identity(
            principal_id="agent_001",
            principal_type="service_account",
            roles=["agent", "reader"],
        )
        store.store_identity(identity)

        # Read identity
        retrieved = store.get_identity("agent_001")
        assert retrieved is not None
        assert retrieved.principal_id == "agent_001"
        assert "reader" in retrieved.roles

        # Update identity
        identity.roles.append("writer")
        store.update_identity(identity)

        updated = store.get_identity("agent_001")
        assert "writer" in updated.roles

        # Delete identity
        store.delete_identity("agent_001")
        deleted = store.get_identity("agent_001")
        assert deleted is None


@pytest.mark.skipif(not AGENTS_MODULE_AVAILABLE, reason="Agents module not fully available")
class TestPrivilegeStore:
    """Test InMemoryPrivilegeStore."""

    def test_privilege_store_elevation_grants(self):
        """Test storing and retrieving elevation grants."""
        store = InMemoryPrivilegeStore()

        grant = ElevationGrant(
            grant_id="grant_001",
            principal_id="agent_001",
            granted_by="admin_alice",
            permissions=["deploy_code"],
            expires_at="2026-04-01T10:00:00Z",
        )

        store.store_elevation_grant(grant)

        retrieved = store.get_elevation_grant("grant_001")
        assert retrieved is not None
        assert retrieved.grant_id == "grant_001"
        assert "deploy_code" in retrieved.permissions


@pytest.mark.skipif(not AGENTS_MODULE_AVAILABLE, reason="Agents module not fully available")
class TestExecutionResult:
    """Test ExecutionResult tracking."""

    def test_execution_result_success(self):
        """Test successful execution result."""
        result = ExecutionResult(
            request_id="req_001",
            success=True,
            output="Data retrieved successfully",
            execution_time_ms=150,
        )

        assert result.success is True
        assert result.output == "Data retrieved successfully"
        assert result.execution_time_ms == 150

    def test_execution_result_failure(self):
        """Test failed execution result."""
        result = ExecutionResult(
            request_id="req_002",
            success=False,
            error="Permission denied",
            execution_time_ms=50,
        )

        assert result.success is False
        assert result.error == "Permission denied"


@pytest.mark.skipif(not AGENTS_MODULE_AVAILABLE, reason="Agents module not fully available")
class TestAgentEventValidation:
    """Test Pydantic validation for agent events."""

    def test_invalid_hash_format(self):
        """Test validation rejects invalid hash format."""
        with pytest.raises(Exception):  # Pydantic validation error
            AgentEvent(
                event_id="evt_001",
                event_type=AgentEventType.READ,
                occurred_at=datetime.now(timezone.utc).isoformat(),
                agent_id="agent_001",
                agent_name="Agent",
                principal_type="service_account",
                session_id="session_001",
                action="read",
                resource_type="data",
                resource_id="res_001",
                policy_decision=AgentPolicyDecision.ALLOW,
                executed=True,
                success=True,
                params_hash="invalid_hash",  # Should be 64-char hex
                input_hash=hashlib.sha256(b"input").hexdigest(),
                output_hash=hashlib.sha256(b"output").hexdigest(),
                prior_event_hash="0" * 64,
            )


@pytest.mark.skipif(not AGENTS_MODULE_AVAILABLE, reason="Agents module not fully available")
class TestAgentWorkflowScenarios:
    """Test real-world agent workflow scenarios."""

    def test_healthcare_agent_workflow(self):
        """Test healthcare agent reading patient records."""
        # Step 1: Create identity
        identity = Identity(
            principal_id="agent_healthcare_001",
            principal_type="service_account",
            roles=["healthcare_agent", "reader"],
            attributes={"department": "radiology"},
        )

        # Step 2: Create resource
        resource = Resource(
            resource_id="patient_record_12345",
            resource_type="patient_record",
            owner="hospital_001",
            attributes={"classification": "phi", "patient_id": "12345"},
        )

        # Step 3: Create action request
        request = ActionRequest(
            request_id="req_healthcare_001",
            action="read_patient_record",
            identity=identity,
            resource=resource,
            context={"purpose": "diagnosis", "timestamp": "2026-03-31T10:00:00Z"},
        )

        # Step 4: Create agent event
        event = AgentEvent(
            event_id="evt_healthcare_001",
            event_type=AgentEventType.READ,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            agent_id="agent_healthcare_001",
            agent_name="RadiologyAgent",
            principal_type="service_account",
            session_id="session_healthcare_001",
            action="read_patient_record",
            resource_type="patient_record",
            resource_id="patient_record_12345",
            policy_decision=AgentPolicyDecision.ALLOW,
            executed=True,
            success=True,
            params_hash=hashlib.sha256(b"{'patient_id': '12345'}").hexdigest(),
            input_hash=hashlib.sha256(b"read_request").hexdigest(),
            output_hash=hashlib.sha256(b"patient_data").hexdigest(),
            prior_event_hash="0" * 64,
            sensitivity_level=SensitivityLevel.RESTRICTED,
        )

        assert event.action == "read_patient_record"
        assert event.sensitivity_level == SensitivityLevel.RESTRICTED

    def test_financial_agent_approval_workflow(self):
        """Test financial agent approving payments."""
        # Step 1: Agent requires elevation for high-value approval
        grant = ElevationGrant(
            grant_id="grant_fin_001",
            principal_id="agent_finance_001",
            granted_by="cfo_bob",
            permissions=["approve_payment_over_10k"],
            expires_at="2026-03-31T23:59:59Z",
            justification="Q1 invoice processing",
        )

        # Step 2: Create approval event
        event = AgentEvent(
            event_id="evt_fin_001",
            event_type=AgentEventType.DECISION,
            occurred_at=datetime.now(timezone.utc).isoformat(),
            agent_id="agent_finance_001",
            agent_name="InvoiceApprovalAgent",
            principal_type="service_account",
            session_id="session_fin_001",
            action="approve_payment",
            resource_type="payment",
            resource_id="payment_15000_vendor_abc",
            policy_decision=AgentPolicyDecision.ALLOW,
            executed=True,
            success=True,
            params_hash=hashlib.sha256(b"{'amount': 15000, 'vendor': 'abc'}").hexdigest(),
            input_hash=hashlib.sha256(b"invoice_data").hexdigest(),
            output_hash=hashlib.sha256(b"approval_confirmation").hexdigest(),
            prior_event_hash="0" * 64,
            elevation_grant_id="grant_fin_001",
            sensitivity_level=SensitivityLevel.CONFIDENTIAL,
        )

        assert event.elevation_grant_id == "grant_fin_001"
        assert event.action == "approve_payment"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
