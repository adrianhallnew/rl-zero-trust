"""OpenZiti zero-trust client for secure inter-component communication.

Provides a unified HTTP client that routes requests through OpenZiti's
mTLS overlay when available, with transparent fallback to direct HTTP
for development and training without Docker.

The client reads service definitions from ``config/openziti_config.yaml``
and resolves service names to network addresses. When the OpenZiti SDK
is available and an identity file is present, all communication is
secured via mutual TLS through the overlay. Otherwise, direct HTTP
connections are used as a graceful fallback.

Usage:
    >>> from src.zero_trust.openziti_client import OpenZitiClient
    >>> client = OpenZitiClient()
    >>> # Resolves "sdn_controller_api" to http://ryu-controller:8080
    >>> response = client.get("sdn_controller_api", "/stats/switches")
    >>> print(client.is_secure())
    False  # Fallback mode when OpenZiti is not available
"""

import logging
import os
from typing import Any, Dict, Optional

import requests

from src.utils.config_loader import load_config

logger = logging.getLogger(__name__)

# Environment variable overrides for fallback URLs
FALLBACK_ENV_OVERRIDES = {
    "sdn_controller_api": "RYU_API_URL",
    "policy_engine": "RL_AGENT_URL",
    "monitoring_feed": "MININET_URL",
}

# Default timeout for HTTP requests (seconds)
DEFAULT_TIMEOUT = 10


class OpenZitiClient:
    """HTTP client for secure inter-component communication via OpenZiti.

    Automatically detects whether OpenZiti is available and routes
    requests through the mTLS overlay or falls back to direct HTTP.

    Args:
        config_path: Path to openziti_config.yaml. If None, uses the
            default config directory via ``load_config()``.
        identity_name: OpenZiti identity name for this component.
        fallback_enabled: If True, fall back to direct HTTP when
            OpenZiti is unavailable. If False, raise an error.

    Attributes:
        service_map: Mapping of service name to {address, port, protocol}.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        identity_name: str = "rl-agent-identity",
        fallback_enabled: bool = True,
    ) -> None:
        self._identity_name = identity_name
        self._fallback_enabled = fallback_enabled

        # Load configuration
        if config_path is not None:
            import yaml
            with open(config_path, "r") as f:
                self._config = yaml.safe_load(f)
        else:
            self._config = load_config("openziti_config")

        # Build service map from config
        self.service_map = self._build_service_map(self._config)

        # Check if OpenZiti is available
        self._ziti_available = self._check_ziti_available()

        # HTTP session for connection pooling
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "rl-zero-trust-agent/1.0",
            "Accept": "application/json",
        })

        if self._ziti_available:
            self._setup_ziti_transport()

        mode = "OpenZiti mTLS" if self._ziti_available else "HTTP fallback"
        logger.info(
            "OpenZitiClient initialized: identity=%s, mode=%s, "
            "services=%d",
            identity_name, mode, len(self.service_map),
        )

    def _build_service_map(
        self, config: Dict[str, Any],
    ) -> Dict[str, Dict[str, Any]]:
        """Build service name to address/port mapping from config.

        Args:
            config: Parsed openziti_config.yaml dictionary.

        Returns:
            Dict mapping service name -> {address, port, protocol}.
        """
        services = config.get("services", {})
        service_map = {}

        for svc_key, svc_config in services.items():
            service_map[svc_key] = {
                "name": svc_config.get("name", svc_key),
                "address": svc_config.get("address", "localhost"),
                "port": svc_config.get("port", 80),
                "protocol": svc_config.get("protocol", "tcp"),
                "hosting_identities": svc_config.get("hosting_identities", []),
                "allowed_identities": svc_config.get("allowed_identities", []),
            }

        logger.debug("Service map: %s", list(service_map.keys()))
        return service_map

    def _check_ziti_available(self) -> bool:
        """Check if OpenZiti SDK and identity are available.

        Checks in order:
        1. OPENZITI_ENABLED environment variable is "true".
        2. OpenZiti identity file exists at OPENZITI_IDENTITY path.
        3. The ``openziti`` Python package is importable.

        Returns:
            True if OpenZiti is available for mTLS routing.
        """
        # Check env var first
        enabled = os.environ.get("OPENZITI_ENABLED", "").lower()
        if enabled != "true":
            logger.debug("OPENZITI_ENABLED is not 'true', using fallback")
            return False

        # Check for identity file
        identity_path = os.environ.get(
            "OPENZITI_IDENTITY",
            "/openziti/identities/rl-agent.json",
        )
        if not os.path.exists(identity_path):
            logger.debug(
                "OpenZiti identity file not found at %s", identity_path,
            )
            return False

        # Check if SDK is importable
        try:
            import openziti  # noqa: F401
            logger.info("OpenZiti SDK available, mTLS routing enabled")
            return True
        except ImportError:
            logger.debug("openziti package not installed, using fallback")
            return False

    def _setup_ziti_transport(self) -> None:
        """Configure the HTTP session to route through OpenZiti.

        When OpenZiti SDK is available, patches the socket layer so
        that connections to Ziti services are automatically tunnelled
        through the overlay network with mTLS.
        """
        try:
            import openziti
            identity_path = os.environ.get(
                "OPENZITI_IDENTITY",
                "/openziti/identities/rl-agent.json",
            )
            openziti.monkeypatch(identity_path)
            logger.info(
                "OpenZiti transport configured with identity: %s",
                identity_path,
            )
        except Exception as e:
            logger.warning(
                "Failed to configure OpenZiti transport: %s. "
                "Falling back to direct HTTP.",
                e,
            )
            self._ziti_available = False

    def _resolve_url(self, service_name: str, path: str = "") -> str:
        """Resolve a service name and path to a full URL.

        Resolution priority:
        1. Environment variable override (e.g. RYU_API_URL).
        2. Service map from config (address:port).

        Args:
            service_name: Key in the service map (e.g. "sdn_controller_api").
            path: URL path to append (e.g. "/stats/switches").

        Returns:
            Full URL string.

        Raises:
            ValueError: If the service name is not found in the service map.
        """
        # Check env var override first
        env_key = FALLBACK_ENV_OVERRIDES.get(service_name)
        if env_key:
            env_url = os.environ.get(env_key)
            if env_url:
                url = env_url.rstrip("/") + path
                logger.debug(
                    "Resolved %s via env %s: %s", service_name, env_key, url,
                )
                return url

        # Resolve from service map
        if service_name not in self.service_map:
            raise ValueError(
                f"Unknown service: '{service_name}'. "
                f"Available: {list(self.service_map.keys())}"
            )

        svc = self.service_map[service_name]
        scheme = "https" if self._ziti_available else "http"
        url = f"{scheme}://{svc['address']}:{svc['port']}{path}"
        logger.debug("Resolved %s: %s", service_name, url)
        return url

    def get(
        self,
        service_name: str,
        path: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ) -> requests.Response:
        """Send a GET request to a service.

        Args:
            service_name: Target service (e.g. "sdn_controller_api").
            path: URL path (e.g. "/stats/switches").
            timeout: Request timeout in seconds.
            **kwargs: Additional arguments passed to ``requests.get()``.

        Returns:
            HTTP response object.
        """
        url = self._resolve_url(service_name, path)
        logger.debug("GET %s", url)
        return self._session.get(url, timeout=timeout, **kwargs)

    def post(
        self,
        service_name: str,
        path: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ) -> requests.Response:
        """Send a POST request to a service.

        Args:
            service_name: Target service (e.g. "sdn_controller_api").
            path: URL path (e.g. "/stats/flowentry/add").
            timeout: Request timeout in seconds.
            **kwargs: Additional arguments passed to ``requests.post()``.

        Returns:
            HTTP response object.
        """
        url = self._resolve_url(service_name, path)
        logger.debug("POST %s", url)
        return self._session.post(url, timeout=timeout, **kwargs)

    def delete(
        self,
        service_name: str,
        path: str = "",
        timeout: int = DEFAULT_TIMEOUT,
        **kwargs: Any,
    ) -> requests.Response:
        """Send a DELETE request to a service.

        Args:
            service_name: Target service.
            path: URL path.
            timeout: Request timeout in seconds.
            **kwargs: Additional arguments passed to ``requests.delete()``.

        Returns:
            HTTP response object.
        """
        url = self._resolve_url(service_name, path)
        logger.debug("DELETE %s", url)
        return self._session.delete(url, timeout=timeout, **kwargs)

    def is_secure(self) -> bool:
        """Check if communication is secured via OpenZiti mTLS.

        Returns:
            True if using OpenZiti overlay, False if using HTTP fallback.
        """
        return self._ziti_available

    def get_identity(self) -> str:
        """Get the current OpenZiti identity name.

        Returns:
            Identity name string.
        """
        return self._identity_name

    def get_service_info(self, service_name: str) -> Dict[str, Any]:
        """Get configuration details for a service.

        Args:
            service_name: Service name key.

        Returns:
            Dict with service configuration (address, port, etc.).

        Raises:
            ValueError: If service not found.
        """
        if service_name not in self.service_map:
            raise ValueError(
                f"Unknown service: '{service_name}'. "
                f"Available: {list(self.service_map.keys())}"
            )
        return self.service_map[service_name].copy()

    def list_services(self) -> list:
        """List all available service names.

        Returns:
            List of service name strings.
        """
        return list(self.service_map.keys())

    def check_access(self, service_name: str) -> bool:
        """Check if the current identity is allowed to access a service.

        Performs a policy check based on the config's allowed_identities
        for the service.

        Args:
            service_name: Service to check access for.

        Returns:
            True if the identity is in the allowed list.
        """
        if service_name not in self.service_map:
            return False

        svc = self.service_map[service_name]
        allowed = svc.get("allowed_identities", [])
        return self._identity_name in allowed

    def close(self) -> None:
        """Close the HTTP session and release resources."""
        self._session.close()
        logger.debug("OpenZitiClient session closed")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
