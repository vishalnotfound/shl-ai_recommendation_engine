"""
Schema compliance tests — verify the exact response shape on every code path.

These tests use FastAPI's TestClient (sync) so they don't require a running server.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _assert_valid_schema(data: dict):
    """Assert the response matches the exact required schema."""
    assert "reply" in data, "Missing 'reply' field"
    assert isinstance(data["reply"], str), "'reply' must be a string"
    assert len(data["reply"]) > 0, "'reply' must be non-empty"

    assert "recommendations" in data, "Missing 'recommendations' field"
    assert isinstance(data["recommendations"], list), "'recommendations' must be a list"

    for rec in data["recommendations"]:
        assert "name" in rec, "Recommendation missing 'name'"
        assert "url" in rec, "Recommendation missing 'url'"
        assert "test_type" in rec, "Recommendation missing 'test_type'"
        assert isinstance(rec["name"], str)
        assert isinstance(rec["url"], str)
        assert isinstance(rec["test_type"], str)
        # URL must start with https://
        assert rec["url"].startswith("https://"), f"URL must be HTTPS: {rec['url']}"

    assert "end_of_conversation" in data, "Missing 'end_of_conversation' field"
    assert isinstance(data["end_of_conversation"], bool), "'end_of_conversation' must be boolean"

    # No extra top-level fields
    allowed_fields = {"reply", "recommendations", "end_of_conversation"}
    extra = set(data.keys()) - allowed_fields
    assert not extra, f"Extra top-level fields not allowed: {extra}"


class TestHealthEndpoint:
    def test_health_returns_ok(self):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data == {"status": "ok"}


class TestSchemaCompliance:
    def test_empty_messages(self):
        """Empty messages → clarifying question, valid schema."""
        resp = client.post("/chat", json={"messages": []})
        assert resp.status_code == 200
        data = resp.json()
        _assert_valid_schema(data)
        assert data["recommendations"] == []
        assert data["end_of_conversation"] is False

    def test_single_vague_message(self):
        """Vague first message → should clarify, not recommend."""
        resp = client.post("/chat", json={
            "messages": [
                {"role": "user", "content": "I need an assessment"}
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        _assert_valid_schema(data)
        # Should clarify, not recommend
        assert data["end_of_conversation"] is False

    def test_specific_message(self):
        """Specific request → should return recommendations."""
        resp = client.post("/chat", json={
            "messages": [
                {"role": "user", "content": "I need a Java programming test and a Python test for mid-level developers"}
            ]
        })
        assert resp.status_code == 200
        data = resp.json()
        _assert_valid_schema(data)
        # Should have recommendations
        assert len(data["recommendations"]) > 0
        assert len(data["recommendations"]) <= 10

    def test_malformed_role(self):
        """Invalid role → 422 validation error."""
        resp = client.post("/chat", json={
            "messages": [
                {"role": "system", "content": "hello"}
            ]
        })
        assert resp.status_code == 422

    def test_empty_content(self):
        """Empty content → 422 validation error."""
        resp = client.post("/chat", json={
            "messages": [
                {"role": "user", "content": ""}
            ]
        })
        assert resp.status_code == 422

    def test_missing_messages_field(self):
        """Missing messages field → should still work (defaults to [])."""
        resp = client.post("/chat", json={})
        assert resp.status_code == 200
        data = resp.json()
        _assert_valid_schema(data)

    def test_recommendations_never_null(self):
        """Recommendations is always a list, never null."""
        # Test with multiple scenarios
        scenarios = [
            {"messages": []},
            {"messages": [{"role": "user", "content": "hello"}]},
            {"messages": [{"role": "user", "content": "I need a Python test"}]},
        ]
        for scenario in scenarios:
            resp = client.post("/chat", json=scenario)
            data = resp.json()
            assert data["recommendations"] is not None
            assert isinstance(data["recommendations"], list)

    def test_end_of_conversation_is_boolean(self):
        """end_of_conversation must be a JSON boolean, not a string."""
        resp = client.post("/chat", json={
            "messages": [{"role": "user", "content": "test"}]
        })
        data = resp.json()
        assert isinstance(data["end_of_conversation"], bool)
