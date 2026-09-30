"""Runtime configuration loaded from environment variables."""
from __future__ import annotations

import math
import os


class Settings:
    def __init__(self) -> None:
        # SQLAlchemy URL, e.g. postgresql+psycopg2://radio:radio@db:5432/radio
        self.database_url: str = os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg2://radio:radio@localhost:5432/radio",
        )
        # Initial non-negative guard band for a *brand-new* database, in the
        # same unit as the frequency numbers (the API itself is unit-agnostic;
        # use MHz, kHz, ... consistently).  Once a database has been
        # initialised this value is never re-applied: online changes made via
        # /api/guard-band/confirm survive restarts and are stored per
        # revision.
        guard_raw = os.getenv("GUARD_BAND", "0")
        try:
            guard_band = float(guard_raw)
        except (TypeError, ValueError):
            raise ValueError(f"GUARD_BAND must be a finite number, got {guard_raw!r}")
        if not math.isfinite(guard_band):
            raise ValueError(f"GUARD_BAND must be a finite number, got {guard_raw!r}")
        if guard_band < 0:
            raise ValueError(f"GUARD_BAND must be non-negative, got {guard_raw!r}")
        self.guard_band: float = guard_band

        self.app_host: str = os.getenv("APP_HOST", "0.0.0.0")
        self.app_port: int = int(os.getenv("APP_PORT", "8000"))


settings = Settings()
