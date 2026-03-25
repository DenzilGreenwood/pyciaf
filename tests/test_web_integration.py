"""
CIAF Web - Basic Integration Tests

Tests for core CIAF Web functionality.

Created: 2026-03-24
Version: 1.0.0
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_event_creation():
    """Test 1: Create basic WebAIEvent."""
    print("[TEST 1] Event creation... ", end="", flush=True)

    try:
        from ciaf.web import WebAIEvent, EventType

        event = WebAIEvent.create(
            event_type=EventType.PROMPT_SUBMIT,
            org_id="test-org",
            user_id="test-user",
            session_id="test-session",
            tool_name="ChatGPT",
        )

        assert event.event_id is not None
        assert event.event_type == EventType.PROMPT_SUBMIT
        assert event.org_id == "test-org"
        assert event.occurred_at is not None

        print("[OK] PASSED")
        return True
    except Exception as e:
        print(f"[FAIL] {str(e)}")
        return False


def test_ai_tool_detection():
    """Test 2: Detect AI tools from URLs."""
    print("[TEST 2] AI tool detection... ", end="", flush=True)

    try:
        from ciaf.web import detect_ai_tool

        # Test ChatGPT detection
        result = detect_ai_tool(
            "https://chat.openai.com/c/abc123",
            approved_tools={"ChatGPT Enterprise"}
        )

        assert result is not None
        assert result.tool_name == "ChatGPT"
        assert result.is_shadow_ai()  # Not approved (different name)

        # Test Claude detection
        result2 = detect_ai_tool("https://claude.ai/chat/abc", approved_tools={"Claude"})
        assert result2 is not None
        assert result2.tool_name == "Claude"
        assert not result2.is_shadow_ai()  # Approved

        print("[OK] PASSED")
        return True
    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_content_classification():
    """Test 3: Classify sensitive content."""
    print("[TEST 3] Content classification... ", end="", flush=True)

    try:
        from ciaf.web import classify_content, DataClassification

        # Test PII detection
        text = "My SSN is 123-45-6789 and email is john@example.com"
        result = classify_content(text)

        assert result.classification in [
            DataClassification.INTERNAL,
            DataClassification.RESTRICTED
        ]
        assert len(result.matched_rules) > 0
        assert result.sensitivity_score > 0.3

        # Test safe content
        safe_text = "Hello, how are you today?"
        result2 = classify_content(safe_text)
        assert result2.classification == DataClassification.PUBLIC
        assert result2.sensitivity_score < 0.3

        print("[OK] PASSED")
        return True
    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_evaluation():
    """Test 4: Evaluate policy rules."""
    print("[TEST 4] Policy evaluation... ", end="", flush=True)

    try:
        from ciaf.web import PolicyEngine, WebAIEvent, EventType, PolicyDecision

        engine = PolicyEngine()

        # Test shadow AI detection - should warn
        event = WebAIEvent.create(
            event_type=EventType.PROMPT_SUBMIT,
            org_id="test-org",
            user_id="test-user",
            session_id="test-session",
            tool_name="ChatGPT",
            tool_approved=False,  # Shadow AI
        )

        result = engine.evaluate(event)
        assert result.decision == PolicyDecision.WARN
        assert result.matched_rule is not None

        # Test approved tool - should allow
        event2 = WebAIEvent.create(
            event_type=EventType.PROMPT_SUBMIT,
            org_id="test-org",
            user_id="test-user",
            session_id="test-session",
            tool_name="ChatGPT Enterprise",
            tool_approved=True,
        )

        result2 = engine.evaluate(event2)
        assert result2.decision == PolicyDecision.ALLOW

        print("[OK] PASSED")
        return True
    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_receipt_generation():
    """Test 5: Generate cryptographic receipts."""
    print("[TEST 5] Receipt generation... ", end="", flush=True)

    try:
        from ciaf.web import generate_receipt, WebAIEvent, EventType

        event = WebAIEvent.create(
            event_type=EventType.PROMPT_SUBMIT,
            org_id="test-org",
            user_id="test-user",
            session_id="test-session",
            tool_name="ChatGPT",
        )

        receipt = generate_receipt(event, signer_id="test-signer")

        assert receipt.receipt_id is not None
        assert receipt.event_id == event.event_id
        assert receipt.event_hash is not None
        assert receipt.receipt_hash is not None
        assert receipt.verify_hash()

        print("[OK] PASSED")
        return True
    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_content_redaction():
    """Test 6: Redact sensitive content."""
    print("[TEST 6] Content redaction... ", end="", flush=True)

    try:
        from ciaf.web import redact_pii, redact_credentials

        # Test PII redaction
        text = "My SSN is 123-45-6789 and email is john@example.com"
        redacted = redact_pii(text)

        assert "123-45-6789" not in redacted
        assert "john@example.com" not in redacted
        assert "[SSN-REDACTED]" in redacted
        assert "[EMAIL-REDACTED]" in redacted

        # Test credential redaction
        code = "api_key='sk-abc123456789' and password='secretpass123'"
        safe_code = redact_credentials(code)

        assert "sk-abc123456789" not in safe_code or "secretpass123" not in safe_code
        assert "REDACTED" in safe_code

        print("[OK] PASSED")
        return True
    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_event_collection():
    """Test 7: Full event collection workflow."""
    print("[TEST 7] Event collection workflow... ", end="", flush=True)

    try:
        from ciaf.web import collect_ai_event, EventType

        result = collect_ai_event(
            event_type=EventType.PROMPT_SUBMIT,
            org_id="test-org",
            user_id="test-user",
            session_id="test-session",
            url="https://chat.openai.com",
            content="Draft an email about Q4 results",
            approved_tools={"ChatGPT Enterprise"},
        )

        assert result.event is not None
        assert result.detection is not None
        assert result.detection.tool_name == "ChatGPT"
        assert result.is_shadow_ai()  # Not in approved list

        print("[OK] PASSED")
        return True
    except Exception as e:
        print(f"[FAIL] {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("CIAF Web - Basic Integration Tests")
    print("=" * 60)

    results = []

    results.append(test_event_creation())
    results.append(test_ai_tool_detection())
    results.append(test_content_classification())
    results.append(test_policy_evaluation())
    results.append(test_receipt_generation())
    results.append(test_content_redaction())
    results.append(test_event_collection())

    # Summary
    passed = sum(1 for r in results if r is True)
    failed = sum(1 for r in results if r is False)
    total = len(results)

    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed (of {total})")
    print("=" * 60)

    if failed == 0:
        print("\n[OK] All tests passed!")
        return 0
    else:
        print(f"\n[FAIL] {failed} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())
