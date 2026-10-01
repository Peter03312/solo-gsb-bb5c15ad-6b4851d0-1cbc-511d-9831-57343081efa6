"""SQLAlchemy ORM models.

History is event-sourced:

* ``applications``      - every submitted request, rejected submissions are
                          never persisted (see whole-request validation);
* ``approval_events``   - append-only approval record, one per revision;
* ``revocation_events`` - append-only revocation record, one per revision;
* ``guard_band_events`` - append-only guard-band record, one per revision
                          (revision 0 is the initial value seeded at database
                          creation, every later row is one confirmed online
                          change);
* ``revision_meta``     - singleton row holding the current revision number
                          and the guard band currently in force.

A revocation never deletes history: it only removes an application from the
effective set used by later adjudications.  Snapshots for any past revision
are reconstructed from the event tables, including the guard band in force
at that revision.  Approval events keep their own immutable copy of the
band they were decided under and are never rewritten.
"""
from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Application(Base):
    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(200), nullable=False)

    # Coverage circle in planar coordinates.
    cx: Mapped[float] = mapped_column(Float, nullable=False)
    cy: Mapped[float] = mapped_column(Float, nullable=False)
    radius: Mapped[float] = mapped_column(Float, nullable=False)

    # Closed frequency interval [freq_low, freq_high].
    freq_low: Mapped[float] = mapped_column(Float, nullable=False)
    freq_high: Mapped[float] = mapped_column(Float, nullable=False)

    # UTC half-open interval [start_time, end_time).
    start_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    end_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )

    # pending -> approved -> revoked (irreversible transitions)
    status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="pending", server_default="pending"
    )
    approval_revision: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class ApprovalEvent(Base):
    __tablename__ = "approval_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("applications.id"), nullable=False
    )
    # Snapshot of the guard band in force when the decision was made.
    guard_band: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (Index("ix_approval_events_application", "application_id"),)


class RevocationEvent(Base):
    __tablename__ = "revocation_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    application_id: Mapped[int] = mapped_column(
        ForeignKey("applications.id"), nullable=False
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )

    __table_args__ = (Index("ix_revocation_events_application", "application_id"),)


class GuardBandEvent(Base):
    """Append-only guard-band record, one per revision.

    The row at revision 0 is the initial value seeded when the database was
    created (``previous_guard_band`` is NULL there); every later row is one
    confirmed online change and consumes exactly one revision, sharing the
    revision sequence with approvals and revocations.
    """

    __tablename__ = "guard_band_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, unique=True)
    previous_guard_band: Mapped[float | None] = mapped_column(
        Float, nullable=True
    )
    guard_band: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, default=_utcnow
    )


class RevisionMeta(Base):
    """Singleton row (id == 1): current revision + live guard band."""

    __tablename__ = "revision_meta"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # Live guard band used by every new adjudication.  Seeded once from the
    # GUARD_BAND environment variable; afterwards only changed through the
    # online config endpoint and never overwritten at startup.
    guard_band: Mapped[float] = mapped_column(Float, nullable=False)
