from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMSettings
from .mcp import McpServerConfig, parse_mcp_servers_json


def _load_dotenv() -> None:
    """仅填充尚未出现在进程环境中的键；pytest 下跳过，避免测试吃到本机密钥。"""

    if os.getenv("PYTEST_CURRENT_TEST"):
        return
    if os.getenv("MAS_SKIP_DOTENV", "").strip().lower() in {"1", "true", "yes"}:
        return
    path = Path(__file__).resolve().parents[2] / ".env"
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value

@dataclass(frozen=True)
class AppConfig:
    output_dir: Path
    upload_dir: Path
    db_path: Path
    database_url: str = field(repr=False)
    market_data_provider: str
    alphavantage_api_key: str | None = field(repr=False)
    host: str
    port: int
    api_key: str | None = field(repr=False)
    llm: LLMSettings
    local_tenant_id: str = "local"
    local_user_id: str = "owner"
    job_lease_seconds: int = 300
    job_max_attempts: int = 3
    job_retry_delay_seconds: int = 30
    operational_retention_days: int = 90
    completed_job_retention_days: int = 30
    model_input_token_budget: int = 300_000
    model_output_token_budget: int = 32_768
    max_upload_bytes: int = 25 * 1024 * 1024
    max_upload_files: int = 8
    max_pdf_pages: int = 500
    max_pdf_text_characters: int = 5_000_000
    allow_network: bool = False
    sec_user_agent: str | None = field(default=None, repr=False)
    fred_api_key: str | None = field(default=None, repr=False)
    fred_base_url: str = "https://api.stlouisfed.org"
    conversation_memory_enabled: bool = True
    conversation_context_tokens: int = 300_000
    conversation_recent_tokens: int = 20_000
    session_document_ttl_seconds: int = 60 * 60
    max_session_document_sessions: int = 100
    paddleocr_access_token: str | None = field(default=None, repr=False)
    paddleocr_job_url: str = "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs"
    paddleocr_model: str = "PaddleOCR-VL-1.6"
    brave_search_api_key: str | None = field(default=None, repr=False)
    bocha_search_api_key: str | None = field(default=None, repr=False)
    embedding_endpoint: str | None = None
    embedding_model: str | None = None
    embedding_api_key: str | None = field(default=None, repr=False)
    embedding_timeout_seconds: float = 30.0
    personal_memory_enabled: bool = True
    automatic_memory_consolidation_enabled: bool = True
    automatic_skill_learning_enabled: bool = True
    user_profile_path: Path | None = None
    personal_knowledge_enabled: bool = True
    max_personal_knowledge_documents: int = 100
    planning_evidence_characters: int = 24_000
    synthesis_evidence_characters: int = 48_000
    synthesis_output_tokens: int = 4_096
    mcp_servers: tuple[McpServerConfig, ...] = ()
    alltick_token: str | None = field(default=None, repr=False)
    biying_licence: str | None = field(default=None, repr=False)
    enable_yfinance_fallback: bool = False
    enable_akshare_fallback: bool = False
    fred_max_calls_per_minute: int = 8
    bocha_max_calls_per_minute: int = 6
    brave_max_calls_per_minute: int = 6
    market_max_calls_per_minute: int = 6

    def __post_init__(self) -> None:
        for principal_name, principal_value in (
            ("local_tenant_id", self.local_tenant_id),
            ("local_user_id", self.local_user_id),
        ):
            if (
                not principal_value.strip()
                or len(principal_value) > 200
                or any(ord(item) < 32 or ord(item) == 127 for item in principal_value)
            ):
                raise ValueError(f"{principal_name} is invalid")
        if self.market_data_provider not in {"yahoo", "alphavantage", "offline", "disabled", "none"}:
            raise ValueError("unsupported market data provider")
        if not 1 <= self.port <= 65_535:
            raise ValueError("port must be between 1 and 65535")
        if not 10 <= self.job_lease_seconds <= 3_600:
            raise ValueError("job lease must be between 10 and 3600 seconds")
        if not 1 <= self.job_max_attempts <= 10:
            raise ValueError("job max attempts must be between 1 and 10")
        if not 0 <= self.job_retry_delay_seconds <= 3_600:
            raise ValueError("job retry delay must be between 0 and 3600 seconds")
        if not 1 <= self.operational_retention_days <= 3_650:
            raise ValueError("operational retention must be between 1 and 3650 days")
        if not 1 <= self.completed_job_retention_days <= 3_650:
            raise ValueError("completed job retention must be between 1 and 3650 days")
        if not 16_000 <= self.model_input_token_budget <= 1_000_000:
            raise ValueError("model input-token budget must be between 16000 and 1000000")
        if not 1_024 <= self.model_output_token_budget <= 200_000:
            raise ValueError("model output-token budget must be between 1024 and 200000")
        if (
            min(
                self.max_upload_bytes,
                self.max_upload_files,
                self.max_pdf_pages,
                self.max_pdf_text_characters,
            )
            < 1
        ):
            raise ValueError("upload and PDF limits must be positive")
        if not 16_000 <= self.conversation_context_tokens <= 300_000:
            raise ValueError("conversation context budget must be between 16000 and 300000 tokens")
        if not 4_000 <= self.conversation_recent_tokens <= 100_000:
            raise ValueError("recent conversation budget must be between 4000 and 100000 tokens")
        if self.session_document_ttl_seconds < 60:
            raise ValueError("session document TTL must be at least 60 seconds")
        if self.max_session_document_sessions < 1:
            raise ValueError("session document session limit must be positive")
        if not 1 <= self.max_personal_knowledge_documents <= 10_000:
            raise ValueError("personal knowledge document limit is invalid")
        if not 4_000 <= self.planning_evidence_characters <= 200_000:
            raise ValueError("planning evidence budget must be between 4000 and 200000 characters")
        if not 4_000 <= self.synthesis_evidence_characters <= 200_000:
            raise ValueError("synthesis evidence budget must be between 4000 and 200000 characters")
        if not 256 <= self.synthesis_output_tokens <= 4_096:
            raise ValueError("synthesis output tokens must be between 256 and 4096")
        if len(self.mcp_servers) > 4:
            raise ValueError("at most four MCP servers are supported")
        if len({item.name for item in self.mcp_servers}) != len(self.mcp_servers):
            raise ValueError("MCP server names must be unique")
        if bool(self.embedding_endpoint) != bool(self.embedding_model):
            raise ValueError("embedding endpoint and model must be configured together")
        if self.embedding_api_key and not self.embedding_endpoint:
            raise ValueError("embedding API key requires an embedding endpoint")
        if not 0.1 <= self.embedding_timeout_seconds <= 120:
            raise ValueError("embedding timeout must be between 0.1 and 120 seconds")
        for name, value in (
            ("fred_max_calls_per_minute", self.fred_max_calls_per_minute),
            ("bocha_max_calls_per_minute", self.bocha_max_calls_per_minute),
            ("brave_max_calls_per_minute", self.brave_max_calls_per_minute),
            ("market_max_calls_per_minute", self.market_max_calls_per_minute),
        ):
            if not 1 <= value <= 60:
                raise ValueError(f"{name} must be between 1 and 60")

    @classmethod
    def from_env(cls) -> AppConfig:
        _load_dotenv()
        raw_output_dir = os.getenv("MAS_OUTPUT_DIR", "outputs")
        raw_db_path = os.getenv("MAS_DB_PATH", "data/mas_finance.db")
        return cls(
            output_dir=Path(raw_output_dir),
            upload_dir=Path(os.getenv("MAS_UPLOAD_DIR", "uploads")),
            db_path=Path(raw_db_path),
            database_url=os.getenv("MAS_DATABASE_URL", f"sqlite:///{raw_db_path.replace(os.sep, '/')}"),
            job_lease_seconds=int(os.getenv("MAS_JOB_LEASE_SECONDS", "300")),
            job_max_attempts=int(os.getenv("MAS_JOB_MAX_ATTEMPTS", "3")),
            job_retry_delay_seconds=int(os.getenv("MAS_JOB_RETRY_DELAY_SECONDS", "30")),
            operational_retention_days=int(os.getenv("MAS_OPERATIONAL_RETENTION_DAYS", "90")),
            completed_job_retention_days=int(os.getenv("MAS_COMPLETED_JOB_RETENTION_DAYS", "30")),
            model_input_token_budget=int(os.getenv("MAS_MODEL_INPUT_TOKEN_BUDGET", "300000")),
            model_output_token_budget=int(os.getenv("MAS_MODEL_OUTPUT_TOKEN_BUDGET", "32768")),
            # External market data is an explicit deployment choice.  The
            # default must not silently depend on an undocumented endpoint.
            market_data_provider=os.getenv("MAS_MARKET_DATA_PROVIDER", "offline").strip().lower(),
            alphavantage_api_key=os.getenv("ALPHAVANTAGE_API_KEY"),
            host=os.getenv("MAS_HOST", "127.0.0.1"),
            port=int(os.getenv("MAS_PORT", "8000")),
            api_key=os.getenv("MAS_API_KEY"),
            llm=LLMSettings.from_env(),
            local_tenant_id=os.getenv("MAS_LOCAL_TENANT_ID", "local"),
            local_user_id=os.getenv("MAS_LOCAL_USER_ID", "owner"),
            max_upload_bytes=int(os.getenv("MAS_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024))),
            max_upload_files=int(os.getenv("MAS_MAX_UPLOAD_FILES", "8")),
            max_pdf_pages=int(os.getenv("MAS_MAX_PDF_PAGES", "500")),
            max_pdf_text_characters=int(os.getenv("MAS_MAX_PDF_TEXT_CHARACTERS", "5000000")),
            allow_network=os.getenv("MAS_ALLOW_NETWORK", "false").strip().lower() in {"1", "true", "yes"},
            sec_user_agent=os.getenv("MAS_SEC_USER_AGENT"),
            fred_api_key=os.getenv("FRED_API_KEY") or None,
            fred_base_url=os.getenv("FRED_BASE_URL", "https://api.stlouisfed.org"),
            conversation_memory_enabled=os.getenv("MAS_CONVERSATION_MEMORY_ENABLED", "true").strip().lower()
            in {"1", "true", "yes"},
            conversation_context_tokens=int(os.getenv("MAS_CONVERSATION_CONTEXT_TOKENS", "300000")),
            conversation_recent_tokens=int(os.getenv("MAS_CONVERSATION_RECENT_TOKENS", "20000")),
            session_document_ttl_seconds=int(os.getenv("MAS_SESSION_DOCUMENT_TTL_SECONDS", str(60 * 60))),
            max_session_document_sessions=int(os.getenv("MAS_MAX_SESSION_DOCUMENT_SESSIONS", "100")),
            paddleocr_access_token=os.getenv("PADDLEOCR_ACCESS_TOKEN") or None,
            paddleocr_job_url=os.getenv(
                "PADDLEOCR_JOB_URL",
                "https://paddleocr.aistudio-app.com/api/v2/ocr/jobs",
            ),
            paddleocr_model=os.getenv("PADDLEOCR_MODEL", "PaddleOCR-VL-1.6"),
            brave_search_api_key=os.getenv("BRAVE_SEARCH_API_KEY") or None,
            bocha_search_api_key=os.getenv("BOCHA_SEARCH_API_KEY") or None,
            embedding_endpoint=os.getenv("MAS_EMBEDDING_ENDPOINT") or None,
            embedding_model=os.getenv("MAS_EMBEDDING_MODEL") or None,
            embedding_api_key=os.getenv("MAS_EMBEDDING_API_KEY") or None,
            embedding_timeout_seconds=float(os.getenv("MAS_EMBEDDING_TIMEOUT_SECONDS", "30")),
            personal_memory_enabled=os.getenv("MAS_PERSONAL_MEMORY_ENABLED", "true").strip().lower()
            in {"1", "true", "yes"},
            automatic_memory_consolidation_enabled=os.getenv(
                "MAS_AUTOMATIC_MEMORY_CONSOLIDATION_ENABLED", "true"
            ).strip().lower()
            in {"1", "true", "yes"},
            automatic_skill_learning_enabled=os.getenv(
                "MAS_AUTOMATIC_SKILL_LEARNING_ENABLED", "true"
            ).strip().lower()
            in {"1", "true", "yes"},
            user_profile_path=(Path(value) if (value := os.getenv("MAS_USER_PROFILE_PATH")) else None),
            personal_knowledge_enabled=os.getenv("MAS_PERSONAL_KNOWLEDGE_ENABLED", "true").strip().lower()
            in {"1", "true", "yes"},
            max_personal_knowledge_documents=int(os.getenv("MAS_MAX_PERSONAL_KNOWLEDGE_DOCUMENTS", "100")),
            planning_evidence_characters=int(os.getenv("MAS_PLANNING_EVIDENCE_CHARACTERS", "24000")),
            synthesis_evidence_characters=int(os.getenv("MAS_SYNTHESIS_EVIDENCE_CHARACTERS", "48000")),
            synthesis_output_tokens=int(os.getenv("MAS_SYNTHESIS_OUTPUT_TOKENS", "4096")),
            mcp_servers=parse_mcp_servers_json(os.getenv("MAS_MCP_SERVERS")),
            alltick_token=os.getenv("ALLTICK_TOKEN") or None,
            biying_licence=os.getenv("BIYING_LICENCE") or None,
            enable_yfinance_fallback=os.getenv("MAS_ENABLE_YFINANCE", "false").strip().lower()
            in {"1", "true", "yes"},
            enable_akshare_fallback=os.getenv("MAS_ENABLE_AKSHARE", "false").strip().lower()
            in {"1", "true", "yes"},
            fred_max_calls_per_minute=int(os.getenv("FRED_MAX_CALLS_PER_MINUTE", "8")),
            bocha_max_calls_per_minute=int(os.getenv("BOCHA_MAX_CALLS_PER_MINUTE", "6")),
            brave_max_calls_per_minute=int(os.getenv("BRAVE_MAX_CALLS_PER_MINUTE", "6")),
            market_max_calls_per_minute=int(os.getenv("MAS_MARKET_MAX_CALLS_PER_MINUTE", "6")),
        )
