from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).parent.parent

load_dotenv()


def _list(key: str, default: str = "") -> list[str]:
    raw = os.getenv(key, default)
    return [v.strip() for v in raw.split(",") if v.strip()]


@dataclass
class Settings:
    google_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_API_KEY", ""))
    agent_mode: str = field(default_factory=lambda: os.getenv("AGENT_MODE", "graph"))
    llm_provider: str = field(default_factory=lambda: os.getenv("LLM_PROVIDER", "gemini"))
    gemini_model: str = field(default_factory=lambda: os.getenv("GEMINI_MODEL", "gemini-2.5-flash"))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    workspace_id: str = field(default_factory=lambda: os.getenv("WORKSPACE_ID", "demo"))
    checkpoint_path: str = field(default_factory=lambda: os.getenv("CHECKPOINT_PATH", "data/checkpoints.sqlite"))
    feedback_db_path: str = field(default_factory=lambda: os.getenv("FEEDBACK_DB_PATH", "data/feedback.sqlite"))
    max_steps: int = field(default_factory=lambda: int(os.getenv("MAX_STEPS", "4")))
    history_token_limit: int = field(default_factory=lambda: int(os.getenv("HISTORY_TOKEN_LIMIT", "3000")))
    max_evidence_chars: int = field(default_factory=lambda: int(os.getenv("MAX_EVIDENCE_CHARS", "6000")))
    sql_max_attempts: int = field(default_factory=lambda: int(os.getenv("SQL_MAX_ATTEMPTS", "3")))
    sql_timeout_s: int = field(default_factory=lambda: int(os.getenv("SQL_TIMEOUT_S", "5")))
    sql_allowed_tables: list[str] = field(default_factory=lambda: _list("SQL_ALLOWED_TABLES", "employees,projects"))
    sql_forbidden_columns: list[str] = field(default_factory=lambda: _list("SQL_FORBIDDEN_COLUMNS", ""))
    email_retrieval: str = field(default_factory=lambda: os.getenv("EMAIL_RETRIEVAL", "dense"))
    database_path: str = field(default_factory=lambda: os.getenv("DATABASE_PATH", "data/demo.db"))
    company_site_urls: list[str] = field(default_factory=lambda: _list("COMPANY_SITE_URLS", ""))


settings = Settings()
