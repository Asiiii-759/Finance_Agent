from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from .llm import LLMSettings


@dataclass(frozen=True)
class AppConfig:
    output_dir: Path
    upload_dir: Path
    db_path: Path
    database_url: str = field(repr=False)
    redis_url: str | None = field(repr=False)
    redis_queue_name: str
    market_data_provider: str
    alphavantage_api_key: str | None = field(repr=False)
    host: str
    port: int
    api_key: str | None = field(repr=False)
    llm: LLMSettings
    max_upload_bytes: int = 25 * 1024 * 1024
    max_upload_files: int = 8
    max_pdf_pages: int = 500
    max_pdf_text_characters: int = 5_000_000
    allow_network: bool = False
    sec_user_agent: str | None = field(default=None, repr=False)
    fred_api_key: str | None = field(default=None, repr=False)
    fred_base_url: str = "https://api.stlouisfed.org"
    conversation_memory_enabled: bool = True
    conversation_context_characters: int = 16_000
    conversation_recent_events: int = 12
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
    personal_knowledge_enabled: bool = True
    max_personal_knowledge_documents: int = 100
    planning_evidence_characters: int = 24_000
    synthesis_evidence_characters: int = 48_000
    synthesis_output_tokens: int = 4_096

    def __post_init__(self) -> None:
        if self.market_data_provider not in {"yahoo", "alphavantage", "offline", "disabled", "none"}:
            raise ValueError("unsupported market data provider")
        if not 1 <= self.port <= 65_535:
            raise ValueError("port must be between 1 and 65535")
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
        if not 4_000 <= self.conversation_context_characters <= 100_000:
            raise ValueError("conversation context budget must be between 4000 and 100000 characters")
        if not 4 <= self.conversation_recent_events <= 50:
            raise ValueError("recent conversation event count must be between 4 and 50")
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
        if bool(self.embedding_endpoint) != bool(self.embedding_model):
            raise ValueError("embedding endpoint and model must be configured together")
        if self.embedding_api_key and not self.embedding_endpoint:
            raise ValueError("embedding API key requires an embedding endpoint")
        if not 0.1 <= self.embedding_timeout_seconds <= 120:
            raise ValueError("embedding timeout must be between 0.1 and 120 seconds")

    @classmethod
    def from_env(cls) -> AppConfig:
        raw_output_dir = os.getenv("MAS_OUTPUT_DIR", "outputs")
        raw_db_path = os.getenv("MAS_DB_PATH", "data/mas_finance.db")
        return cls(
            output_dir=Path(raw_output_dir),
            upload_dir=Path(os.getenv("MAS_UPLOAD_DIR", "uploads")),
            db_path=Path(raw_db_path),
            database_url=os.getenv("MAS_DATABASE_URL", f"sqlite:///{raw_db_path.replace(os.sep, '/')}"),
            redis_url=os.getenv("MAS_REDIS_URL"),
            redis_queue_name=os.getenv("MAS_REDIS_QUEUE_NAME", "finance-analysis"),
            # External market data is an explicit deployment choice.  The
            # default must not silently depend on an undocumented endpoint.
            market_data_provider=os.getenv("MAS_MARKET_DATA_PROVIDER", "offline").strip().lower(),
            alphavantage_api_key=os.getenv("ALPHAVANTAGE_API_KEY"),
            host=os.getenv("MAS_HOST", "127.0.0.1"),
            port=int(os.getenv("MAS_PORT", "8000")),
            api_key=os.getenv("MAS_API_KEY"),
            llm=LLMSettings.from_env(),
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
            conversation_context_characters=int(os.getenv("MAS_CONVERSATION_CONTEXT_CHARACTERS", "16000")),
            conversation_recent_events=int(os.getenv("MAS_CONVERSATION_RECENT_EVENTS", "12")),
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
            personal_knowledge_enabled=os.getenv("MAS_PERSONAL_KNOWLEDGE_ENABLED", "true").strip().lower()
            in {"1", "true", "yes"},
            max_personal_knowledge_documents=int(os.getenv("MAS_MAX_PERSONAL_KNOWLEDGE_DOCUMENTS", "100")),
            planning_evidence_characters=int(os.getenv("MAS_PLANNING_EVIDENCE_CHARACTERS", "24000")),
            synthesis_evidence_characters=int(os.getenv("MAS_SYNTHESIS_EVIDENCE_CHARACTERS", "48000")),
            synthesis_output_tokens=int(os.getenv("MAS_SYNTHESIS_OUTPUT_TOKENS", "4096")),
        )
