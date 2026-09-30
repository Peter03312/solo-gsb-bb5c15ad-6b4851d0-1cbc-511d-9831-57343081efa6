"""Database engine, startup readiness wait and schema initialisation."""
from __future__ import annotations

import logging
import time

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker

from .config import settings

log = logging.getLogger("radio.db")

engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, future=True)


def wait_for_database(timeout_seconds: float = 60.0) -> None:
    """Block until the database accepts connections (Compose startup race)."""
    deadline = time.monotonic() + timeout_seconds
    attempt = 0
    while True:
        attempt += 1
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            log.info("database is ready after %d attempt(s)", attempt)
            return
        except OperationalError:
            if time.monotonic() >= deadline:
                raise
            log.info("waiting for database (attempt %d) ...", attempt)
            time.sleep(1.0)


def _upgrade_revision_meta(bind) -> None:
    """Add the ``guard_band`` column to singleton tables created before
    online guard-band management existed, then backfill it.

    The ALTERs are backend-neutral (plain ADD COLUMN without a NOT NULL
    server default, which SQLite rejects on non-empty tables).
    """
    columns = {col["name"] for col in inspect(bind).get_columns("revision_meta")}
    if "guard_band" not in columns:
        with bind.begin() as conn:
            conn.execute(
                text("ALTER TABLE revision_meta ADD COLUMN guard_band FLOAT")
            )
    with bind.begin() as conn:
        conn.execute(
            text("UPDATE revision_meta SET guard_band = :gb WHERE guard_band IS NULL"),
            {"gb": settings.guard_band},
        )


def init_db() -> None:
    """Create tables, upgrade old schemas and seed revision 0.

    A brand-new database gets the singleton row and a revision-0 guard-band
    event seeded from ``GUARD_BAND``.  An existing database keeps whatever
    guard band is already in force: neither the column backfill nor the seed
    event are applied once a revision-0 event exists, so restarting the
    service never overrides a confirmed online setting.
    """
    # Imported here so models register on Base.metadata before create_all().
    from . import models  # noqa: F401
    from .models import Base, GuardBandEvent, RevisionMeta

    Base.metadata.create_all(bind=engine)
    _upgrade_revision_meta(engine)
    with SessionLocal() as session, session.begin():
        if session.get(RevisionMeta, 1) is None:
            session.add(
                RevisionMeta(id=1, revision=0, guard_band=settings.guard_band)
            )
            log.info("seeded revision 0 with guard band %s", settings.guard_band)
        if session.get(GuardBandEvent, 1) is None:
            # Revision 0 carries the initial value on both new databases and
            # upgraded ones; any later online change simply appends a row.
            meta = session.get(RevisionMeta, 1)
            initial_band = (
                meta.guard_band if meta is not None else settings.guard_band
            )
            session.add(
                GuardBandEvent(revision=0, guard_band=initial_band)
            )
