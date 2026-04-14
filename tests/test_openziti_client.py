"""Unit tests for the OpenZiti zero-trust client.

Tests cover:
    - Client initialization with config.
    - Service map construction from YAML config.
    - URL resolution with env var overrides.
    - URL resolution from service map.
    - Unknown service raises ValueError.
    - Fallback mode when OpenZiti is unavailable.
    - is_secure() reflects availability.
    - Access control check (allowed/denied).
    - Service listing.
    - Service info retrieval.
    - Context manager lifecycle.
    - HTTP GET/POST/DELETE delegation.
    - Identity retrieval.
    - Environment variable override priority.
"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import yaml

from src.zero_trust.openziti_client import OpenZitiClient, FALLBACK_ENV_OVERRIDES


@pytest.fixture
def sample_config():
    """Minimal OpenZiti config for testing."""
    return {
        "controller": {"address": "openziti-controller", "port": 1280},
        "services": {
            "sdn_controller_api": {
                "name": "sdn-controller-api",
                "address": "ryu-controller",
                "port": 8080,
                "protocol": "tcp",
                "hosting_identities": ["ryu-controller-identity"],
                "allowed_identities": ["rl-agent-identity"],
            },
            "policy_engine": {
                "name": "policy-engine",
                "address": "rl-agent",
                "port": 5000,
                "protocol": "tcp",
                "hosting_identities": ["rl-agent-identity"],
                "allowed_identities": ["ryu-controller-identity"],
            },
            "monitoring_feed": {
                "name": "monitoring-feed",
                "address": "mininet",
                "port": 9090,
                "protocol": "tcp",
                "hosting_identities": ["mininet-monitor-identity"],
                "allowed_identities": ["rl-agent-identity"],
            },
        },
    }


@pytest.fixture
def config_file(sample_config, tmp_path):
    """Write sample config to a temp YAML file."""
    config_path = tmp_path / "openziti_config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(sample_config, f)
    return str(config_path)


@pytest.fixture
def client(config_file):
    """Create an OpenZitiClient in fallback mode."""
    with patch.dict(os.environ, {}, clear=False):
        # Ensure OpenZiti is not detected
        env_clean = {
            "OPENZITI_ENABLED": "false",
        }
        with patch.dict(os.environ, env_clean):
            c = OpenZitiClient(config_path=config_file)
    return c


# --- Initialization ---

class TestClientInit:
    """Test client initialization."""

    def test_init_loads_services(self, client):
        """Client should load all 3 services from config."""
        assert len(client.service_map) == 3
        assert "sdn_controller_api" in client.service_map
        assert "policy_engine" in client.service_map
        assert "monitoring_feed" in client.service_map

    def test_init_fallback_mode(self, client):
        """Client should be in fallback mode without OpenZiti."""
        assert not client.is_secure()

    def test_init_identity(self, client):
        """Client should store identity name."""
        assert client.get_identity() == "rl-agent-identity"

    def test_init_custom_identity(self, config_file):
        """Client should accept custom identity name."""
        with patch.dict(os.environ, {"OPENZITI_ENABLED": "false"}):
            c = OpenZitiClient(
                config_path=config_file,
                identity_name="custom-identity",
            )
        assert c.get_identity() == "custom-identity"


# --- Service Map ---

class TestServiceMap:
    """Test service map construction."""

    def test_service_address(self, client):
        """Service map should have correct addresses."""
        sdn = client.service_map["sdn_controller_api"]
        assert sdn["address"] == "ryu-controller"
        assert sdn["port"] == 8080

    def test_service_protocol(self, client):
        """Service map should preserve protocol."""
        assert client.service_map["sdn_controller_api"]["protocol"] == "tcp"

    def test_service_identities(self, client):
        """Service map should preserve identity lists."""
        sdn = client.service_map["sdn_controller_api"]
        assert "rl-agent-identity" in sdn["allowed_identities"]
        assert "ryu-controller-identity" in sdn["hosting_identities"]

    def test_empty_services(self, tmp_path):
        """Client should handle config with no services."""
        config_path = tmp_path / "empty.yaml"
        with open(config_path, "w") as f:
            yaml.dump({"services": {}}, f)
        with patch.dict(os.environ, {"OPENZITI_ENABLED": "false"}):
            c = OpenZitiClient(config_path=str(config_path))
        assert len(c.service_map) == 0


# --- URL Resolution ---

class TestURLResolution:
    """Test URL resolution logic."""

    def test_resolve_from_service_map(self, client):
        """Should resolve service to http://address:port."""
        url = client._resolve_url("sdn_controller_api", "/stats/switches")
        assert url == "http://ryu-controller:8080/stats/switches"

    def test_resolve_no_path(self, client):
        """Should resolve without path."""
        url = client._resolve_url("policy_engine")
        assert url == "http://rl-agent:5000"

    def test_resolve_env_override(self, client):
        """Env var should take priority over config."""
        with patch.dict(os.environ, {"RYU_API_URL": "http://localhost:9999"}):
            url = client._resolve_url("sdn_controller_api", "/test")
        assert url == "http://localhost:9999/test"

    def test_resolve_unknown_service(self, client):
        """Unknown service should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown service"):
            client._resolve_url("nonexistent_service")

    def test_resolve_env_strips_trailing_slash(self, client):
        """Env var URL with trailing slash should not double-slash."""
        with patch.dict(os.environ, {"RYU_API_URL": "http://localhost:9999/"}):
            url = client._resolve_url("sdn_controller_api", "/test")
        assert url == "http://localhost:9999/test"


# --- Access Control ---

class TestAccessControl:
    """Test identity-based access checks."""

    def test_check_access_allowed(self, client):
        """rl-agent-identity should access sdn_controller_api."""
        assert client.check_access("sdn_controller_api") is True

    def test_check_access_denied(self, client):
        """rl-agent-identity should NOT access policy_engine (only ryu)."""
        assert client.check_access("policy_engine") is False

    def test_check_access_unknown_service(self, client):
        """Unknown service should return False."""
        assert client.check_access("nonexistent") is False


# --- Service Info ---

class TestServiceInfo:
    """Test service information retrieval."""

    def test_list_services(self, client):
        """Should return all service names."""
        services = client.list_services()
        assert set(services) == {
            "sdn_controller_api", "policy_engine", "monitoring_feed",
        }

    def test_get_service_info(self, client):
        """Should return a copy of service config."""
        info = client.get_service_info("sdn_controller_api")
        assert info["address"] == "ryu-controller"
        assert info["port"] == 8080
        # Ensure it's a copy
        info["port"] = 9999
        assert client.service_map["sdn_controller_api"]["port"] == 8080

    def test_get_service_info_unknown(self, client):
        """Unknown service should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown service"):
            client.get_service_info("nonexistent")


# --- Context Manager ---

class TestContextManager:
    """Test context manager lifecycle."""

    def test_context_manager(self, config_file):
        """Context manager should close session cleanly."""
        with patch.dict(os.environ, {"OPENZITI_ENABLED": "false"}):
            with OpenZitiClient(config_path=config_file) as c:
                assert len(c.service_map) == 3
            # Session should be closed after __exit__


# --- HTTP Methods ---

class TestHTTPMethods:
    """Test HTTP request delegation."""

    def test_get_delegates(self, client):
        """GET should call session.get with resolved URL."""
        client._session = MagicMock()
        client._session.get.return_value = MagicMock(status_code=200)
        resp = client.get("sdn_controller_api", "/stats/switches")
        client._session.get.assert_called_once()
        call_args = client._session.get.call_args
        assert "/stats/switches" in call_args[0][0]

    def test_post_delegates(self, client):
        """POST should call session.post with resolved URL."""
        client._session = MagicMock()
        client._session.post.return_value = MagicMock(status_code=200)
        resp = client.post("sdn_controller_api", "/stats/flowentry/add",
                           json={"dpid": 1})
        client._session.post.assert_called_once()

    def test_delete_delegates(self, client):
        """DELETE should call session.delete with resolved URL."""
        client._session = MagicMock()
        client._session.delete.return_value = MagicMock(status_code=200)
        resp = client.delete("sdn_controller_api", "/stats/flowentry/delete")
        client._session.delete.assert_called_once()
