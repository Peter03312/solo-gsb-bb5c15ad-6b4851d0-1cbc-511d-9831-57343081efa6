"""FastAPI application: temporary-event radio spectrum requests.

Endpoints
---------
POST   /api/applications                 submit (whole-request validation)
GET    /api/applications                 list submissions
GET    /api/applications/{id}            fetch one submission
POST   /api/applications/{id}/preview    conflicts + current revision
POST   /api/applications/{id}/approve    conditional approval (expected rev.)
POST   /api/applications/batch-approve   atomic group approval (expected rev.)
POST   /api/applications/{id}/revoke     revoke (changes future effective set)
GET    /api/revision/current             current revision + effective set
GET    /api/revision/{revision}          as-of snapshot (approvals/revocations)
GET    /api/config/guard-band            live guard band + current revision
POST   /api/config/guard-band/preview    dry-run a proposed band (non-mutating)
POST   /api/config/guard-band            confirm a new band (expected rev.)
GET    /health                           liveness probe

Concurrency
-----------
Approval locks the singleton ``revision_meta`` row (``SELECT ... FOR UPDATE``)
for the duration of the transaction.  Concurrent single approvals, batch
approvals, revocations and guard-band changes therefore serialise on the
same revision arbitration: losers re-adjudicate against the winner's commit
and are refused with ``stale_revision`` or ``interference``.  A batch either
consumes one consecutive revision per member in input order, or nothing at
all; a guard-band change consumes exactly one revision.

Guard band
----------
The live guard band is stored on the ``revision_meta`` row and seeded once
from the ``GUARD_BAND`` environment variable (restarts never overwrite it).
Every adjudication (preview, single/batch approval) uses the live value;
each confirmed change is appended as an immutable ``guard_band_events``
record so historical revision snapshots restore the band in force at the
time.  Approval events keep their own copy of the band they were decided
under and are never rewritten.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError

from . import adjudication
from .adjudication import Circle, Conflict, License, PairConflict
from .config import settings
from .db import SessionLocal, engine, init_db, wait_for_database
from .errors import APIError, api_error_handler
from .models import (
    Application,
    ApprovalEvent,
    GuardBandEvent,
    RevisionMeta,
    RevocationEvent,
)
from .schemas import (
    ApplicationCreate,
    ApprovalRequest,
    BatchApprovalRequest,
    GuardBandPreviewRequest,
    GuardBandUpdateRequest,
    RevocationRequest,
)
from .semantic_validation import validate_application, validate_guard_band

log = logging.getLogger("radio.api")

# Error codes
E_VALIDATION = "validation_failed"
E_NOT_FOUND = "not_found"
E_NOT_PENDING = "not_pending"
E_ALREADY_REVOKED = "already_revoked"
E_REVISION_OUT_OF_RANGE = "revision_out_of_range"
E_STALE_REVISION = "stale_revision"
E_INTERFERENCE = "interference"
E_DUPLICATE_APPLICATION = "duplicate_application_id"


# --------------------------------------------------------------------------- #
# Serialisation helpers
# --------------------------------------------------------------------------- #
def _utc(value: datetime) -> datetime:
    """Coerce a possibly naive timestamp (e.g. under SQLite) to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def iso(value: datetime) -> str:
    return _utc(value).isoformat().replace("+00:00", "Z")


def serialize_application(app: Application) -> dict[str, Any]:
    return {
        "id": app.id,
        "name": app.name,
        "circle": {"cx": app.cx, "cy": app.cy, "radius": app.radius},
        "frequency": {"low": app.freq_low, "high": app.freq_high},
        "time_window": {"start": iso(app.start_time), "end": iso(app.end_time)},
        "status": app.status,
        "approval_revision": app.approval_revision,
        "created_at": iso(app.created_at),
    }


def _license_view(lic: License) -> dict[str, Any]:
    """The public shape of a license as a conflict/snapshot party."""
    return {
        "id": lic.application_id,
        "circle": {
            "cx": lic.circle.cx,
            "cy": lic.circle.cy,
            "radius": lic.circle.radius,
        },
        "frequency": {"low": lic.freq_low, "high": lic.freq_high},
        "time_window": {
            "start": iso(lic.start_time),
            "end": iso(lic.end_time),
        },
    }


def serialize_conflict(conflict: Conflict, guard_band: float) -> dict[str, Any]:
    """One conflicting effective grant with the trigger conditions."""
    return {
        "application": _license_view(conflict.other),
        "trigger": {
            "time_intersect": conflict.time_overlap,
            "area_intersect": conflict.area_overlap,
            "frequency_intersect": conflict.frequency_intersects,
            "minimum_frequency_gap": conflict.minimum_frequency_gap,
            "within_guard_band": (
                conflict.minimum_frequency_gap < guard_band
            ),
        },
        "reason": conflict.reason,
    }


def serialize_pair_conflict(
    pair: PairConflict, guard_band: float
) -> dict[str, Any]:
    """One interfering pair of effective licenses with trigger conditions.

    Same trigger shape as a preview conflict so existing clients can reuse
    their rendering.
    """
    return {
        "first": _license_view(pair.first),
        "second": _license_view(pair.second),
        "trigger": {
            "time_intersect": pair.time_overlap,
            "area_intersect": pair.area_overlap,
            "frequency_intersect": pair.frequency_intersects,
            "minimum_frequency_gap": pair.minimum_frequency_gap,
            "within_guard_band": (
                pair.minimum_frequency_gap < guard_band
            ),
        },
        "reason": pair.reason,
    }


def to_license(app: Application) -> License:
    return License(
        application_id=app.id,
        circle=Circle(app.cx, app.cy, app.radius),
        freq_low=app.freq_low,
        freq_high=app.freq_high,
        start_time=_utc(app.start_time),
        end_time=_utc(app.end_time),
    )


def serialize_group_conflict(
    conflict: Conflict, scope: str, guard_band: float
) -> dict[str, Any]:
    """A conflict inside a group decision, tagged with where it came from.

    ``scope`` is ``"effective"`` when the other side is a currently valid
    license, or ``"group"`` when it is another application in the same batch.
    The body is the same shape as a preview conflict so existing clients can
    reuse their rendering.
    """
    body = serialize_conflict(conflict, guard_band)
    body["scope"] = scope
    return body


def load_effective_licenses(session) -> list[License]:
    """Approved and not (yet) revoked = the effective set for adjudication."""
    rows = (
        session.query(Application)
        .filter(Application.status == "approved")
        .order_by(Application.id)
        .all()
    )
    return [to_license(row) for row in rows]


# --------------------------------------------------------------------------- #
# Revision snapshots
# --------------------------------------------------------------------------- #
def guard_band_at(session, revision: int) -> float:
    """Guard band in force at ``revision`` (latest change at or before it)."""
    event = (
        session.query(GuardBandEvent)
        .filter(GuardBandEvent.revision <= revision)
        .order_by(GuardBandEvent.revision.desc())
        .first()
    )
    if event is not None:
        return event.guard_band
    # Defensive: the revision-0 seed guarantees an event; fall back to the
    # live setting if it is somehow missing.
    meta = session.get(RevisionMeta, 1)
    return meta.guard_band if meta is not None else settings.guard_band


def build_snapshot(session, revision: int) -> dict[str, Any]:
    """Reconstruct the state of the world *at* ``revision``.

    Returns every approval event with revision <= the target and the
    application snapshot, flags those later revoked by the target revision,
    and includes all revocation events of the period, plus the guard band
    in force at that revision and the guard-band records behind it.
    """
    approvals = (
        session.query(ApprovalEvent)
        .filter(ApprovalEvent.revision <= revision)
        .order_by(ApprovalEvent.revision)
        .all()
    )
    revocations = (
        session.query(RevocationEvent)
        .filter(RevocationEvent.revision <= revision)
        .order_by(RevocationEvent.revision)
        .all()
    )
    guard_events = (
        session.query(GuardBandEvent)
        .filter(GuardBandEvent.revision <= revision)
        .order_by(GuardBandEvent.revision)
        .all()
    )
    revoked_at_or_before = {ev.application_id: ev for ev in revocations}

    approval_records: list[dict[str, Any]] = []
    effective_ids: list[int] = []
    for ev in approvals:
        app = session.get(Application, ev.application_id)
        rev = revoked_at_or_before.get(ev.application_id)
        was_revoked_by_then = rev is not None and rev.revision <= revision
        record = {
            "revision": ev.revision,
            "application_id": ev.application_id,
            "guard_band": ev.guard_band,
            "created_at": iso(ev.created_at),
            # Status of this grant *as of the queried revision*.
            "status_as_of": "revoked" if was_revoked_by_then else "approved",
            "application": serialize_application(app) if app else None,
        }
        if was_revoked_by_then:
            record["revoked_at_revision"] = rev.revision
        else:
            effective_ids.append(ev.application_id)
        approval_records.append(record)

    revocation_records = [
        {
            "revision": ev.revision,
            "application_id": ev.application_id,
            "reason": ev.reason,
            "created_at": iso(ev.created_at),
        }
        for ev in revocations
    ]

    return {
        "revision": revision,
        "guard_band": guard_band_at(session, revision),
        "guard_band_changes": [
            {
                "revision": ev.revision,
                "previous_guard_band": ev.previous_guard_band,
                "guard_band": ev.guard_band,
                "created_at": iso(ev.created_at),
            }
            for ev in guard_events
        ],
        "effective_application_ids": effective_ids,
        "approvals": approval_records,
        "revocations": revocation_records,
    }


# --------------------------------------------------------------------------- #
# Application factory / lifespan
# --------------------------------------------------------------------------- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    wait_for_database()
    init_db()
    with SessionLocal() as session:
        meta = session.get(RevisionMeta, 1)
        live_band = meta.guard_band if meta is not None else settings.guard_band
    log.info(
        "startup complete; guard_band=%s (GUARD_BAND seed=%s) listening on %s:%s",
        live_band,
        settings.guard_band,
        settings.app_host,
        settings.app_port,
    )
    yield
    engine.dispose()


app = FastAPI(
    title="Temporary Event Radio Spectrum API",
    version="1.0.0",
    description=(
        "Submit, preview, approve and revoke temporary radio applications. "
        "Frequencies are closed intervals, UTC times are half-open intervals, "
        "coverage is a planar circle."
    ),
    lifespan=lifespan,
)
app.add_exception_handler(APIError, api_error_handler)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """Normalise FastAPI structural errors into the standard error envelope."""
    details = [
        {
            "field": ".".join(str(p) for p in err["loc"] if p != "body"),
            "code": err["type"],
            "message": err["msg"],
        }
        for err in exc.errors()
    ]
    return JSONResponse(
        status_code=422,
        content={
            "error": {
                "code": E_VALIDATION,
                "message": "request validation failed; the whole request was rejected",
                "details": details,
            }
        },
    )


# --------------------------------------------------------------------------- #
# Health / revisions
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["meta"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/revision/current", tags=["revision"])
def current_revision() -> dict[str, Any]:
    with SessionLocal() as session:
        meta = session.get(RevisionMeta, 1)
        revision = meta.revision if meta else 0
        guard_band = meta.guard_band if meta else settings.guard_band
        snapshot = build_snapshot(session, revision)
        effective_applications = [
            serialize_application(app)
            for app_id in snapshot["effective_application_ids"]
            if (app := session.get(Application, app_id)) is not None
        ]
    return {
        "current_revision": revision,
        "guard_band": guard_band,
        "effective_application_ids": snapshot["effective_application_ids"],
        "effective_applications": effective_applications,
        "approvals": snapshot["approvals"],
        "revocations": snapshot["revocations"],
        "guard_band_changes": snapshot["guard_band_changes"],
    }


@app.get("/api/revision/{revision}", tags=["revision"])
def revision_snapshot(revision: int) -> dict[str, Any]:
    if revision < 0:
        raise APIError(
            E_REVISION_OUT_OF_RANGE,
            "revision must be >= 0",
            status_code=422,
        )
    with SessionLocal() as session:
        current = session.get(RevisionMeta, 1).revision
        if revision > current:
            raise APIError(
                E_REVISION_OUT_OF_RANGE,
                f"revision {revision} is in the future; current revision is {current}",
                status_code=404,
                details={"current_revision": current},
            )
        return build_snapshot(session, revision)


# --------------------------------------------------------------------------- #
# Applications: create / list / get
# --------------------------------------------------------------------------- #
@app.post("/api/applications", status_code=201, tags=["applications"])
def create_application(payload: ApplicationCreate) -> dict[str, Any]:
    errors = validate_application(payload)
    if errors:
        raise APIError(
            E_VALIDATION,
            "the whole application was rejected due to invalid fields",
            status_code=422,
            details=errors,
        )

    app_row = Application(
        name=payload.name,
        cx=payload.circle.cx,
        cy=payload.circle.cy,
        radius=payload.circle.radius,
        freq_low=payload.frequency.low,
        freq_high=payload.frequency.high,
        start_time=payload.time_window.start.astimezone(timezone.utc),
        end_time=payload.time_window.end.astimezone(timezone.utc),
        status="pending",
    )
    with SessionLocal() as session, session.begin():
        session.add(app_row)
        session.flush()
        result = serialize_application(app_row)
    return result


@app.get("/api/applications", tags=["applications"])
def list_applications(status: str | None = None) -> dict[str, Any]:
    with SessionLocal() as session:
        query = session.query(Application).order_by(Application.id)
        if status is not None:
            if status not in {"pending", "approved", "revoked"}:
                raise APIError(
                    E_VALIDATION,
                    "status filter must be one of pending, approved, revoked",
                    status_code=422,
                )
            query = query.filter(Application.status == status)
        rows = query.all()
        return {"applications": [serialize_application(row) for row in rows]}


@app.get("/api/applications/{application_id}", tags=["applications"])
def get_application(application_id: int) -> dict[str, Any]:
    with SessionLocal() as session:
        row = session.get(Application, application_id)
        if row is None:
            raise APIError(
                E_NOT_FOUND,
                f"application {application_id} does not exist",
                status_code=404,
            )
        return serialize_application(row)


# --------------------------------------------------------------------------- #
# Preview (non-mutating)
# --------------------------------------------------------------------------- #
@app.post("/api/applications/{application_id}/preview", tags=["applications"])
def preview_application(application_id: int) -> dict[str, Any]:
    with SessionLocal() as session:
        row = session.get(Application, application_id)
        if row is None:
            raise APIError(
                E_NOT_FOUND,
                f"application {application_id} does not exist",
                status_code=404,
            )
        if row.status != "pending":
            raise APIError(
                E_NOT_PENDING,
                f"application {application_id} is {row.status}, only pending "
                "applications can be previewed",
                status_code=409,
                details={"status": row.status},
            )

        meta = session.get(RevisionMeta, 1)
        current = meta.revision
        guard_band = meta.guard_band
        effective = load_effective_licenses(session)
        conflicts = adjudication.evaluate(to_license(row), effective, guard_band)

    return {
        "application_id": application_id,
        "current_revision": current,
        "guard_band": guard_band,
        "has_conflicts": bool(conflicts),
        "conflicts": [serialize_conflict(c, guard_band) for c in conflicts],
    }


# --------------------------------------------------------------------------- #
# Approve (conditional on expected revision, row-lock serialised)
# --------------------------------------------------------------------------- #
@app.post("/api/applications/{application_id}/approve", tags=["applications"])
def approve_application(
    application_id: int, payload: ApprovalRequest
) -> dict[str, Any]:
    with SessionLocal() as session:
        try:
            with session.begin():
                # Serialise against every other approve/revoke/guard-band
                # change on the singleton row (a real row lock on
                # PostgreSQL).
                meta = session.get(RevisionMeta, 1, with_for_update=True)
                current = meta.revision
                guard_band = meta.guard_band

                row = session.get(Application, application_id)
                if row is None:
                    raise APIError(
                        E_NOT_FOUND,
                        f"application {application_id} does not exist",
                        status_code=404,
                    )
                if row.status == "revoked":
                    raise APIError(
                        E_ALREADY_REVOKED,
                        f"application {application_id} has been revoked",
                        status_code=409,
                    )
                if row.status == "approved":
                    raise APIError(
                        E_NOT_PENDING,
                        f"application {application_id} is already approved at "
                        f"revision {row.approval_revision}",
                        status_code=409,
                        details={"approval_revision": row.approval_revision},
                    )

                # Optimistic revision check against the locked, latest revision.
                if payload.expected_revision != current:
                    raise APIError(
                        E_STALE_REVISION,
                        f"expected revision {payload.expected_revision} but "
                        f"current revision is {current}; re-preview and re-decide",
                        status_code=409,
                        details={
                            "expected_revision": payload.expected_revision,
                            "current_revision": current,
                        },
                    )

                # Re-adjudicate under the lock against the latest effective
                # set and the live guard band.
                effective = load_effective_licenses(session)
                conflicts = adjudication.evaluate(
                    to_license(row), effective, guard_band
                )
                if conflicts:
                    raise APIError(
                        E_INTERFERENCE,
                        f"application {application_id} interferes with "
                        f"{len(conflicts)} effective grant(s); frequency not granted",
                        status_code=409,
                        details={
                            "current_revision": current,
                            "conflicts": [
                                serialize_conflict(c, guard_band)
                                for c in conflicts
                            ],
                        },
                    )

                meta.revision = current + 1
                session.add(
                    ApprovalEvent(
                        revision=meta.revision,
                        application_id=application_id,
                        guard_band=guard_band,
                    )
                )
                row.status = "approved"
                row.approval_revision = meta.revision
                session.flush()
                response = {
                    "approved": True,
                    "application_id": application_id,
                    "revision": meta.revision,
                    "guard_band": guard_band,
                    "application": serialize_application(row),
                }
        except IntegrityError:
            # Last line of defence (e.g. backends without row locks such as
            # SQLite's tests): the revision slot was taken concurrently.
            session.rollback()
            latest = session.get(RevisionMeta, 1).revision
            raise APIError(
                E_STALE_REVISION,
                f"the revision advanced concurrently (now {latest}); "
                "re-preview and re-decide",
                status_code=409,
                details={"current_revision": latest},
            )
    return response


# --------------------------------------------------------------------------- #
# Batch approve (all-or-nothing group approval under one revision lock)
# --------------------------------------------------------------------------- #
@app.post("/api/applications/batch-approve", tags=["applications"])
def batch_approve_applications(payload: BatchApprovalRequest) -> dict[str, Any]:
    # Structural shape (non-empty list, non-negative revision) is enforced by
    # Pydantic; the uniqueness rule goes through the standard error envelope.
    seen: set[int] = set()
    duplicates: list[int] = []
    for app_id in payload.application_ids:
        if app_id in seen:
            duplicates.append(app_id)
        seen.add(app_id)
    if duplicates:
        raise APIError(
            E_VALIDATION,
            "the whole batch was rejected: application_ids must be distinct",
            status_code=422,
            details=[
                {
                    "field": "application_ids",
                    "code": E_DUPLICATE_APPLICATION,
                    "message": (
                        "application_ids must not contain duplicates; "
                        f"repeated: {duplicates}"
                    ),
                    "duplicates": duplicates,
                }
            ],
        )

    with SessionLocal() as session:
        try:
            with session.begin():
                # Same singleton lock as single approve/revoke/guard-band
                # change: all writers share one revision arbitration and
                # serialise here.
                meta = session.get(RevisionMeta, 1, with_for_update=True)
                current = meta.revision
                guard_band = meta.guard_band

                rows_by_id = {
                    row.id: row
                    for row in (
                        session.query(Application)
                        .filter(Application.id.in_(payload.application_ids))
                        .all()
                    )
                }

                # Failure type 1: unknown IDs (batch is not approved at all).
                missing = [
                    app_id
                    for app_id in payload.application_ids
                    if app_id not in rows_by_id
                ]
                if missing:
                    raise APIError(
                        E_NOT_FOUND,
                        f"{len(missing)} application(s) do not exist; the "
                        "whole batch was rejected and no revision was consumed",
                        status_code=404,
                        details={
                            "expected_revision": payload.expected_revision,
                            "current_revision": current,
                            "missing_application_ids": missing,
                        },
                    )

                # Failure type 2: any member no longer pending (same precedence
                # as single approval: revoked outranks already-approved).
                revoked = [
                    app_id
                    for app_id in payload.application_ids
                    if rows_by_id[app_id].status == "revoked"
                ]
                if revoked:
                    raise APIError(
                        E_ALREADY_REVOKED,
                        f"application(s) {revoked} have been revoked; the "
                        "whole batch was rejected and no revision was consumed",
                        status_code=409,
                        details={
                            "expected_revision": payload.expected_revision,
                            "current_revision": current,
                            "applications": [
                                {
                                    "application_id": app_id,
                                    "status": rows_by_id[app_id].status,
                                    "approval_revision": rows_by_id[
                                        app_id
                                    ].approval_revision,
                                }
                                for app_id in payload.application_ids
                                if rows_by_id[app_id].status != "pending"
                            ],
                        },
                    )
                not_pending = [
                    app_id
                    for app_id in payload.application_ids
                    if rows_by_id[app_id].status != "pending"
                ]
                if not_pending:
                    raise APIError(
                        E_NOT_PENDING,
                        f"application(s) {not_pending} are not pending; the "
                        "whole batch was rejected and no revision was consumed",
                        status_code=409,
                        details={
                            "expected_revision": payload.expected_revision,
                            "current_revision": current,
                            "applications": [
                                {
                                    "application_id": app_id,
                                    "status": rows_by_id[app_id].status,
                                    "approval_revision": rows_by_id[
                                        app_id
                                    ].approval_revision,
                                }
                                for app_id in not_pending
                            ],
                        },
                    )

                # Failure type 3: stale revision (a concurrent decision won).
                if payload.expected_revision != current:
                    raise APIError(
                        E_STALE_REVISION,
                        f"expected revision {payload.expected_revision} but "
                        f"current revision is {current}; re-preview and "
                        "re-decide; the whole batch was rejected",
                        status_code=409,
                        details={
                            "expected_revision": payload.expected_revision,
                            "current_revision": current,
                        },
                    )

                # Failure type 4: interference, re-adjudicated under the lock.
                # Every member is checked both against the current effective
                # set and against every other member of the group.
                ordered_rows = [rows_by_id[i] for i in payload.application_ids]
                candidates = [to_license(row) for row in ordered_rows]
                effective = load_effective_licenses(session)
                against_effective, within_group = adjudication.evaluate_group(
                    candidates, effective, guard_band
                )
                interference_details = []
                for candidate in candidates:
                    eff_conflicts = [
                        serialize_group_conflict(c, "effective", guard_band)
                        for c in against_effective[candidate.application_id]
                    ]
                    group_conflicts = [
                        serialize_group_conflict(c, "group", guard_band)
                        for c in within_group[candidate.application_id]
                    ]
                    if eff_conflicts or group_conflicts:
                        interference_details.append(
                            {
                                "application_id": candidate.application_id,
                                "conflicts": eff_conflicts + group_conflicts,
                            }
                        )
                if interference_details:
                    raise APIError(
                        E_INTERFERENCE,
                        f"{len(interference_details)} application(s) in the "
                        "batch interfere with effective grants or with each "
                        "other; the whole batch was rejected and no "
                        "frequency was granted",
                        status_code=409,
                        details={
                            "expected_revision": payload.expected_revision,
                            "current_revision": current,
                            "applications": interference_details,
                        },
                    )

                # All clear: one ApprovalEvent per member, consecutive
                # revisions assigned in the submitted order, one commit.
                start_revision = current
                results = []
                for row in ordered_rows:
                    meta.revision += 1
                    session.add(
                        ApprovalEvent(
                            revision=meta.revision,
                            application_id=row.id,
                            guard_band=guard_band,
                        )
                    )
                    row.status = "approved"
                    row.approval_revision = meta.revision
                    results.append(
                        {
                            "application_id": row.id,
                            "revision": meta.revision,
                            "application": serialize_application(row),
                        }
                    )
                session.flush()
                response = {
                    "approved": True,
                    "expected_revision": payload.expected_revision,
                    "from_revision": start_revision,
                    "to_revision": meta.revision,
                    # Headline final revision, same field name as single
                    # approve/revoke responses.
                    "revision": meta.revision,
                    "guard_band": guard_band,
                    "application_ids": payload.application_ids,
                    "results": results,
                }
        except IntegrityError:
            # Backends without row locks (SQLite in tests) still have the
            # unique revision constraint: a concurrent writer won, so the
            # whole batch is rejected and nothing was partially approved.
            session.rollback()
            latest = session.get(RevisionMeta, 1).revision
            raise APIError(
                E_STALE_REVISION,
                f"the revision advanced concurrently (now {latest}); "
                "re-preview and re-decide; the whole batch was rejected",
                status_code=409,
                details={"current_revision": latest},
            )
    return response


# --------------------------------------------------------------------------- #
# Revoke (only affects the effective set used by later adjudications)
# --------------------------------------------------------------------------- #
@app.post("/api/applications/{application_id}/revoke", tags=["applications"])
def revoke_application(
    application_id: int, payload: RevocationRequest
) -> dict[str, Any]:
    with SessionLocal() as session:
        try:
            with session.begin():
                meta = session.get(RevisionMeta, 1, with_for_update=True)

                row = session.get(Application, application_id)
                if row is None:
                    raise APIError(
                        E_NOT_FOUND,
                        f"application {application_id} does not exist",
                        status_code=404,
                    )
                if row.status == "revoked":
                    raise APIError(
                        E_ALREADY_REVOKED,
                        f"application {application_id} is already revoked",
                        status_code=409,
                    )
                if row.status != "approved":
                    raise APIError(
                        E_NOT_PENDING,
                        f"application {application_id} is pending and cannot "
                        "be revoked",
                        status_code=409,
                        details={"status": row.status},
                    )

                meta.revision += 1
                session.add(
                    RevocationEvent(
                        revision=meta.revision,
                        application_id=application_id,
                        reason=payload.reason,
                    )
                )
                row.status = "revoked"
                session.flush()
                response = {
                    "revoked": True,
                    "application_id": application_id,
                    "revision": meta.revision,
                    "previous_approval_revision": row.approval_revision,
                    "reason": payload.reason,
                }
        except IntegrityError:
            session.rollback()
            latest = session.get(RevisionMeta, 1).revision
            raise APIError(
                E_STALE_REVISION,
                f"the revision advanced concurrently (now {latest}); retry",
                status_code=409,
                details={"current_revision": latest},
            )
    return response


# --------------------------------------------------------------------------- #
# Guard-band configuration (online, revision-arbitrated)
# --------------------------------------------------------------------------- #
@app.get("/api/config/guard-band", tags=["config"])
def get_guard_band() -> dict[str, Any]:
    """Live guard band, the initial seeded value and the current revision."""
    with SessionLocal() as session:
        meta = session.get(RevisionMeta, 1)
        seed = (
            session.query(GuardBandEvent)
            .filter(GuardBandEvent.revision == 0)
            .first()
        )
        guard_band = meta.guard_band if meta else settings.guard_band
        return {
            "current_revision": meta.revision if meta else 0,
            "guard_band": guard_band,
            "initial_guard_band": seed.guard_band if seed else guard_band,
        }


@app.post("/api/config/guard-band/preview", tags=["config"])
def preview_guard_band(payload: GuardBandPreviewRequest) -> dict[str, Any]:
    """Dry-run a proposed guard band against the current effective set.

    Returns the current revision (to be carried into the confirm call as
    ``expected_revision``) and every pair of currently effective licenses
    that would interfere under the proposed band.  Never mutates state.
    """
    errors = validate_guard_band(payload.guard_band)
    if errors:
        raise APIError(
            E_VALIDATION,
            "the guard band preview was rejected due to an invalid value",
            status_code=422,
            details=errors,
        )
    with SessionLocal() as session:
        meta = session.get(RevisionMeta, 1)
        effective = load_effective_licenses(session)
        pairs = adjudication.find_interfering_pairs(
            effective, payload.guard_band
        )
        return {
            "current_revision": meta.revision,
            "current_guard_band": meta.guard_band,
            "proposed_guard_band": payload.guard_band,
            "effective_application_ids": [
                lic.application_id for lic in effective
            ],
            "has_conflicts": bool(pairs),
            "interfering_pair_count": len(pairs),
            "interfering_pairs": [
                serialize_pair_conflict(p, payload.guard_band) for p in pairs
            ],
        }


@app.post("/api/config/guard-band", tags=["config"])
def update_guard_band(payload: GuardBandUpdateRequest) -> dict[str, Any]:
    """Confirm a new guard band (conditional on the expected revision).

    Shares the singleton revision lock with single/batch approvals and
    revocations.  The change is refused when the revision is stale or when
    the proposed band would make any pair of currently effective licenses
    interfere; in both cases the band, the effective set and the revision
    are left untouched.  On success exactly one revision is consumed and
    every later adjudication uses the new band.
    """
    errors = validate_guard_band(payload.guard_band)
    if errors:
        raise APIError(
            E_VALIDATION,
            "the guard band change was rejected due to an invalid value",
            status_code=422,
            details=errors,
        )
    with SessionLocal() as session:
        try:
            with session.begin():
                # Same singleton lock as approve/batch/revoke: one revision
                # arbitration for every writer.
                meta = session.get(RevisionMeta, 1, with_for_update=True)
                current = meta.revision

                if payload.expected_revision != current:
                    raise APIError(
                        E_STALE_REVISION,
                        f"expected revision {payload.expected_revision} but "
                        f"current revision is {current}; re-preview and "
                        "re-decide; the guard band was not changed",
                        status_code=409,
                        details={
                            "expected_revision": payload.expected_revision,
                            "current_revision": current,
                        },
                    )

                # Re-check under the lock: the new band must not make any
                # pair of currently effective licenses interfere.
                effective = load_effective_licenses(session)
                pairs = adjudication.find_interfering_pairs(
                    effective, payload.guard_band
                )
                if pairs:
                    raise APIError(
                        E_INTERFERENCE,
                        f"proposed guard band {payload.guard_band} would make "
                        f"{len(pairs)} effective license pair(s) interfere; "
                        "the guard band was not changed",
                        status_code=409,
                        details={
                            "current_revision": current,
                            "current_guard_band": meta.guard_band,
                            "proposed_guard_band": payload.guard_band,
                            "interfering_pairs": [
                                serialize_pair_conflict(p, payload.guard_band)
                                for p in pairs
                            ],
                        },
                    )

                previous = meta.guard_band
                meta.revision = current + 1
                meta.guard_band = payload.guard_band
                session.add(
                    GuardBandEvent(
                        revision=meta.revision,
                        previous_guard_band=previous,
                        guard_band=payload.guard_band,
                    )
                )
                session.flush()
                response = {
                    "updated": True,
                    "revision": meta.revision,
                    "previous_guard_band": previous,
                    "guard_band": payload.guard_band,
                }
        except IntegrityError:
            # Backends without row locks (SQLite in tests): a concurrent
            # writer took the revision slot; nothing was changed.
            session.rollback()
            latest = session.get(RevisionMeta, 1).revision
            raise APIError(
                E_STALE_REVISION,
                f"the revision advanced concurrently (now {latest}); "
                "re-preview and re-decide; the guard band was not changed",
                status_code=409,
                details={"current_revision": latest},
            )
    return response
