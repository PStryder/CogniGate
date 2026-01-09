"""Configuration and bootstrap system for CogniGate."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from pydantic_settings import BaseSettings, SettingsConfigDict


class AsyncGateConfig(BaseModel):
    """AsyncGate connection configuration."""
    endpoint: str = Field(description="AsyncGate API endpoint")
    auth_token: str = Field(description="Authentication token for AsyncGate")


class AIProviderConfig(BaseModel):
    """AI provider configuration (e.g., OpenRouter)."""
    endpoint: str = Field(default="https://openrouter.ai/api/v1")
    api_key: str = Field(description="API key for AI provider")
    model: str = Field(default="anthropic/claude-3-opus")
    max_tokens: int = Field(default=4096)


class MCPEndpoint(BaseModel):
    """Configuration for an MCP upstream endpoint."""
    name: str = Field(description="MCP server name")
    endpoint: str = Field(description="MCP server endpoint URL")
    auth_token: str | None = Field(default=None, description="Optional auth token")
    read_only: bool = Field(default=True, description="Whether to restrict to read-only operations")
    enabled: bool = Field(default=True)


class WorkerConfig(BaseModel):
    """Worker behavior configuration."""
    polling_interval_seconds: float = Field(default=5.0, description="Polling interval for AsyncGate")
    max_concurrent_jobs: int = Field(default=1, description="Maximum concurrent job executions")
    job_timeout_seconds: int = Field(default=300, description="Maximum time for a single job")
    max_retries: int = Field(default=3, description="Maximum retries for failed tool calls")


class Settings(BaseSettings):
    """Main application settings loaded from environment."""

    # Standalone mode (disables AsyncGate polling, enables receipt storage)
    standalone_mode: bool = Field(default=True, description="Run in standalone mode without AsyncGate")
    receipt_storage_dir: Path = Field(
        default=Path("./receipts"),
        description="Directory for receipt storage (standalone mode)"
    )

    # AsyncGate settings (optional in standalone mode)
    asyncgate_endpoint: str = Field(default="http://localhost:8080")
    asyncgate_auth_token: str = Field(default="")

    # AI provider settings
    ai_endpoint: str = Field(default="https://openrouter.ai/api/v1")
    ai_api_key: str = Field(default="")
    ai_model: str = Field(default="anthropic/claude-3-opus")
    ai_max_tokens: int = Field(default=4096)

    # Worker settings
    polling_interval: float = Field(default=5.0)
    max_concurrent_jobs: int = Field(default=1)
    job_timeout: int = Field(default=300)
    max_retries: int = Field(default=3)

    # Paths
    config_dir: Path = Field(default=Path("/etc/cognigate"))
    plugins_dir: Path = Field(default=Path("/etc/cognigate/plugins"))
    profiles_dir: Path = Field(default=Path("/etc/cognigate/profiles"))

    # Server settings
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    worker_id: str = Field(default="cognigate-worker-1")

    # Authentication settings
    api_key: str = Field(default="", description="API key for REST endpoint authentication")
    require_auth: bool = Field(default=True, description="Require authentication for REST endpoints")
    allow_insecure_dev: bool = Field(default=False, description="Allow unauthenticated access (dev only)")

    # Rate limiting
    rate_limit_enabled: bool = Field(default=True, description="Enable rate limiting")
    rate_limit_requests_per_minute: int = Field(default=50, description="Rate limit per minute")

    # CORS configuration (explicit allowlist for security)
    cors_allowed_origins: list[str] = Field(
        default=["http://localhost:3000", "http://localhost:8080"],
        description="Allowed CORS origins (explicit allowlist for security)"
    )
    cors_allow_credentials: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )
    cors_allowed_methods: list[str] = Field(
        default=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        description="Allowed HTTP methods"
    )
    cors_allowed_headers: list[str] = Field(
        default=["Authorization", "Content-Type", "X-Tenant-ID"],
        description="Allowed request headers"
    )

    model_config = SettingsConfigDict(
        env_prefix="COGNIGATE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Validators
    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate port number range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"Port must be between 1 and 65535, got {v}")
        return v

    @field_validator("asyncgate_endpoint")
    @classmethod
    def validate_asyncgate_endpoint(cls, v: str, info: ValidationInfo) -> str:
        """Validate AsyncGate endpoint URL (optional in standalone mode)."""
        standalone = info.data.get("standalone_mode", True)
        if not standalone and v and not v.startswith(("http://", "https://")):
            raise ValueError(f"Endpoint URL must start with http:// or https://, got {v}")
        return v

    @field_validator("asyncgate_auth_token")
    @classmethod
    def validate_asyncgate_auth_token(cls, v: str, info: ValidationInfo) -> str:
        """Allow empty AsyncGate auth token in standalone mode."""
        standalone = info.data.get("standalone_mode", True)
        if not standalone and not v:
            raise ValueError("asyncgate_auth_token required unless standalone_mode=true")
        return v

    @field_validator("ai_endpoint")
    @classmethod
    def validate_ai_endpoint(cls, v: str) -> str:
        """Validate AI endpoint URL."""
        if not v.startswith(("http://", "https://")):
            raise ValueError(f"Endpoint URL must start with http:// or https://, got {v}")
        return v

    @field_validator("ai_api_key")
    @classmethod
    def validate_ai_api_key(cls, v: str) -> str:
        """Validate AI API key is set."""
        if not v:
            raise ValueError("ai_api_key is required for CogniGate to function")
        return v

    @field_validator("api_key")
    @classmethod
    def validate_api_key(cls, v: str, info: ValidationInfo) -> str:
        """Validate API key is set when auth is required."""
        require_auth = info.data.get("require_auth", True)
        allow_insecure = info.data.get("allow_insecure_dev", False)
        if require_auth and not v and not allow_insecure:
            raise ValueError("api_key is required when require_auth=True and allow_insecure_dev=False")
        return v

    def get_asyncgate_config(self) -> AsyncGateConfig:
        return AsyncGateConfig(
            endpoint=self.asyncgate_endpoint,
            auth_token=self.asyncgate_auth_token
        )

    def get_ai_config(self) -> AIProviderConfig:
        return AIProviderConfig(
            endpoint=self.ai_endpoint,
            api_key=self.ai_api_key,
            model=self.ai_model,
            max_tokens=self.ai_max_tokens
        )

    def get_worker_config(self) -> WorkerConfig:
        return WorkerConfig(
            polling_interval_seconds=self.polling_interval,
            max_concurrent_jobs=self.max_concurrent_jobs,
            job_timeout_seconds=self.job_timeout,
            max_retries=self.max_retries
        )


class InstructionProfile(BaseModel):
    """An instruction profile loaded from filesystem."""
    name: str = Field(description="Profile identifier")
    system_instructions: str = Field(description="System prompt instructions")
    formatting_constraints: str = Field(default="", description="Output formatting rules")
    planning_schema: dict[str, Any] = Field(default_factory=dict, description="Planning output schema")
    tool_usage_rules: str = Field(default="", description="Rules for tool usage")


def load_instruction_profile(profile_path: Path) -> InstructionProfile:
    """Load an instruction profile from a YAML file."""
    with open(profile_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return InstructionProfile(**data)


def load_mcp_endpoints(config_path: Path) -> list[MCPEndpoint]:
    """Load MCP endpoint configurations from a YAML file."""
    if not config_path.exists():
        return []
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return [MCPEndpoint(**ep) for ep in data.get("mcp_endpoints", [])]


class Bootstrap:
    """Bootstrap manager that loads all configuration at startup."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.profiles: dict[str, InstructionProfile] = {}
        self.mcp_endpoints: list[MCPEndpoint] = []
        self._loaded = False

    def load(self) -> None:
        """Load all bootstrap configuration. Called once at startup."""
        if self._loaded:
            raise RuntimeError("Bootstrap already loaded; no runtime mutation allowed")

        self._load_profiles()
        self._load_mcp_config()
        self._loaded = True

    def _load_profiles(self) -> None:
        """Load all instruction profiles from profiles directory."""
        profiles_dir = self.settings.profiles_dir
        if not profiles_dir.exists():
            return

        for profile_file in profiles_dir.glob("*.yaml"):
            profile = load_instruction_profile(profile_file)
            self.profiles[profile.name] = profile

        for profile_file in profiles_dir.glob("*.yml"):
            profile = load_instruction_profile(profile_file)
            self.profiles[profile.name] = profile

    def _load_mcp_config(self) -> None:
        """Load MCP endpoint configuration."""
        mcp_config_path = self.settings.config_dir / "mcp.yaml"
        self.mcp_endpoints = load_mcp_endpoints(mcp_config_path)

    def get_profile(self, name: str) -> InstructionProfile | None:
        """Get a loaded instruction profile by name."""
        return self.profiles.get(name)

    def get_default_profile(self) -> InstructionProfile | None:
        """Get the default instruction profile."""
        return self.profiles.get("default")

    def get_mcp_endpoint(self, name: str) -> MCPEndpoint | None:
        """Get an MCP endpoint configuration by name."""
        for ep in self.mcp_endpoints:
            if ep.name == name and ep.enabled:
                return ep
        return None
