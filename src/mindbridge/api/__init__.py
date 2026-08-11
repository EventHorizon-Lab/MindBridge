"""REST and MCP entry points for MindBridge."""

from mindbridge.api.app import create_app
from mindbridge.api.auth import TenantApiKeyAuthenticator, TenantPrincipal
from mindbridge.api.mcp import create_mcp_server
from mindbridge.api.runtime import (
    RuntimeSettings,
    create_production_app,
    create_production_mcp_server,
)

__all__ = [
    "RuntimeSettings",
    "TenantApiKeyAuthenticator",
    "TenantPrincipal",
    "create_app",
    "create_mcp_server",
    "create_production_app",
    "create_production_mcp_server",
]
