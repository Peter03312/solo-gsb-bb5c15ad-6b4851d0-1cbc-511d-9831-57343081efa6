"""End-to-end tests for online guard-band management.

Covers the non-mutating preview, optimistic-revision confirmation,
interference refusal against the effective set, per-revision history,
restart persistence (GUARD_BAND only seeds a new database), the old-schema
upgrade path and the shared revision arbitration with approvals.
"""
import json

from sqlalchemy import text

from app import config
from app.db import SessionLocal, engine, init_db
from app.models import Base, GuardBandEvent, RevisionMeta
from factories import make_payload


def body(
    circle=(0, 0, 10),
    freq=(100, 110),
    start="2026-10-01T08:00:00Z",
    end="2026-10-01T10:00:00Z",
    name="Event",
):
    return make_payload(circle, freq, start, end, name)


PREVIEW = "/api/guard-band/preview"
CONFIRM = "/api/guard-band/confirm"


def post_raw(client, path, obj):
    """POST whose body keeps Python's NaN/Infinity JSON literals."""
    return client.post(
        path,
        content=json.dumps(obj),
        headers={"Content-Type": "application/json"},
    )


def submit_and_approve(client, name, freq, expected_revision, **kw):
    app_id = client.post(
        "/api/applications", json=body(name=name, freq=freq, **kw)
    ).json()["id"]
    r = client.post(
        f"/api/applications/{app_id}/approve",
        json={"expected_revision": expected_revision},
    )
    assert r.status_code == 200, r.text
    return app_id, r.json()["revision"]


class TestGuardBandPreview:
    def test_empty_effective_set_preview_is_clean_and_non_mutating(self, client):
        r = client.post(PREVIEW, json={"proposed_guard_band": 5.0})
        assert r.status_code == 200, r.text
        data = r.json()
        assert data == {
            "current_revision": 0,
            "current_guard_band": 2.0,
            "proposed_guard_band": 5.0,
            "has_conflicts": False,
            "effective_application_ids": [],
            "conflicts": [],
        }
        # preview changes nothing
        assert client.get("/api/revision/current").json()["current_revision"] == 0

    def test_single_effective_license_has_no_pair(self, client):
        submit_and_approve(client, "A", (100, 110), expected_revision=0)
        r = client.post(PREVIEW, json={"proposed_guard_band": 1000.0})
        assert r.status_code == 200
        data = r.json()
        assert data["current_revision"] == 1
        assert data["has_conflicts"] is False
        assert data["conflicts"] == []

    def test_pair_reported_under_proposed_band_only(self, client):
        # gap exactly 2.0: allowed at band 2.0, clash at 3.0
        a, _ = submit_and_approve(client, "A", (100, 110), expected_revision=0)
        b, rev = submit_and_approve(client, "B", (112, 120), expected_revision=1)
        assert rev == 2

        clean = client.post(PREVIEW, json={"proposed_guard_band": 2.0}).json()
        assert clean["has_conflicts"] is False
        assert clean["effective_application_ids"] == [a, b]

        clash = client.post(PREVIEW, json={"proposed_guard_band": 3.0}).json()
        assert clash["has_conflicts"] is True
        (pair,) = clash["conflicts"]
        assert [x["id"] for x in pair["applications"]] == [a, b]
        assert pair["reason"] == "within_guard_band"
        assert pair["trigger"] == {
            "time_intersect": True,
            "area_intersect": True,
            "frequency_intersect": False,
            "minimum_frequency_gap": 2.0,
            "within_guard_band": True,
        }
        # preview must not move the revision or the band
        current = client.get("/api/revision/current").json()
        assert current["current_revision"] == 2
        assert current["guard_band"] == 2.0

    def test_only_effective_licenses_take_part_in_pairs(self, client):
        a, _ = submit_and_approve(client, "A", (100, 110), expected_revision=0)
        # disjoint bands (gap 2 == band 2) so both can be effective
        b, _ = submit_and_approve(client, "B", (112, 120), expected_revision=1)
        # a third overlapping application stays pending and must not appear
        pending = client.post(
            "/api/applications", json=body(name="pending", freq=(100, 110))
        ).json()["id"]
        # revoke A: only B remains effective, so no pair even at a huge band
        client.post(f"/api/applications/{a}/revoke", json={"reason": "moved"})

        r = post_raw(client, PREVIEW, {"proposed_guard_band": 1000.0}).json()
        assert r["current_revision"] == 3
        assert r["has_conflicts"] is False
        assert r["effective_application_ids"] == [b]
        assert pending  # just to keep the id used

    def test_invalid_values_rejected(self, client):
        for raw in ("-0.001", "NaN", "Infinity", "-Infinity"):
            r = post_raw(client, PREVIEW, json.loads(
                f'{{"proposed_guard_band": {raw}}}'
            ))
            assert r.status_code == 422, raw
            err = r.json()["error"]
            assert err["code"] == "validation_failed"
            assert err["details"][0]["field"] == "proposed_guard_band"

        # missing field / wrong type use the same envelope
        assert client.post(PREVIEW, json={}).status_code == 422
        assert (
            client.post(PREVIEW, json={"proposed_guard_band": "wide"}).status_code
            == 422
        )
        # zero is legal
        assert (
            client.post(PREVIEW, json={"proposed_guard_band": 0}).status_code == 200
        )


class TestGuardBandConfirm:
    def test_confirm_consumes_one_revision_and_takes_effect(self, client):
        # A [100,110] approved at revision 1 under band 2.0
        a, _ = submit_and_approve(client, "A", (100, 110), expected_revision=0)

        # C [110.6,111]: gap 0.6 to A -> blocked at 2.0, allowed at 0.5
        c = client.post(
            "/api/applications", json=body(name="C", freq=(110.6, 111))
        ).json()["id"]
        assert (
            client.post(f"/api/applications/{c}/preview").json()["has_conflicts"]
            is True
        )

        # lower the band: revision 2
        r = client.post(
            CONFIRM,
            json={"expected_revision": 1, "proposed_guard_band": 0.5},
        )
        assert r.status_code == 200, r.text
        assert r.json() == {
            "changed": True,
            "revision": 2,
            "previous_guard_band": 2.0,
            "guard_band": 0.5,
        }

        # later preview / approval adjudicate under the new band
        preview = client.post(f"/api/applications/{c}/preview").json()
        assert preview["guard_band"] == 0.5
        assert preview["current_revision"] == 2
        assert preview["has_conflicts"] is False
        approved = client.post(
            f"/api/applications/{c}/approve", json={"expected_revision": 2}
        )
        assert approved.status_code == 200, approved.text
        assert approved.json()["revision"] == 3
        assert approved.json()["guard_band"] == 0.5

        # current view reports the new band
        current = client.get("/api/revision/current").json()
        assert current["guard_band"] == 0.5
        assert current["current_revision"] == 3

    def test_stale_revision_rejected_without_change(self, client):
        submit_and_approve(client, "A", (100, 110), expected_revision=0)
        r = client.post(
            CONFIRM,
            json={"expected_revision": 0, "proposed_guard_band": 1.0},
        )
        assert r.status_code == 409
        err = r.json()["error"]
        assert err["code"] == "stale_revision"
        assert err["details"] == {"expected_revision": 0, "current_revision": 1}
        current = client.get("/api/revision/current").json()
        assert current["current_revision"] == 1
        assert current["guard_band"] == 2.0

        # retrying on the fresh revision succeeds and continues the sequence
        ok = client.post(
            CONFIRM,
            json={"expected_revision": 1, "proposed_guard_band": 1.0},
        )
        assert ok.status_code == 200
        assert ok.json()["revision"] == 2

    def test_conflicting_proposal_rejected_everything_unchanged(self, client):
        # A [100,110] and B [112,120] coexist at band 2.0 (gap 2 == 2)
        submit_and_approve(client, "A", (100, 110), expected_revision=0)
        submit_and_approve(client, "B", (112, 120), expected_revision=1)

        r = client.post(
            CONFIRM,
            json={"expected_revision": 2, "proposed_guard_band": 3.0},
        )
        assert r.status_code == 409
        err = r.json()["error"]
        assert err["code"] == "interference"
        assert err["details"]["current_guard_band"] == 2.0
        assert err["details"]["proposed_guard_band"] == 3.0
        (pair,) = err["details"]["conflicts"]
        assert [x["id"] for x in pair["applications"]] == [
            client.get("/api/applications?status=approved")
            .json()["applications"][0]["id"],
            client.get("/api/applications?status=approved")
            .json()["applications"][1]["id"],
        ]

        # revision, band and effective set all unchanged: same value retried
        # after a re-preview is still rejected for the same reason
        current = client.get("/api/revision/current").json()
        assert current["current_revision"] == 2
        assert current["guard_band"] == 2.0
        assert current["effective_application_ids"] == [1, 2]

    def test_invalid_confirm_values_rejected_without_lock_side_effects(self, client):
        for raw in ("-1.0", "NaN", "Infinity"):
            r = post_raw(
                client,
                CONFIRM,
                json.loads(
                    f'{{"expected_revision": 0, "proposed_guard_band": {raw}}}'
                ),
            )
            assert r.status_code == 422, raw
            assert r.json()["error"]["details"][0]["field"] == (
                "proposed_guard_band"
            )
        # revision untouched
        assert client.get("/api/revision/current").json()["current_revision"] == 0

    def test_batch_approval_uses_the_confirmed_band(self, client):
        client.post(
            CONFIRM,
            json={"expected_revision": 0, "proposed_guard_band": 0.5},
        ).raise_for_status()
        # gaps 1.0 and 0.6: a group under 0.5 that would clash under 2.0
        ids = []
        for name, freq in (("X", (100, 100.4)), ("Y", (110, 110.4))):
            ids.append(
                client.post(
                    "/api/applications", json=body(name=name, freq=freq)
                ).json()["id"]
            )
        r = client.post(
            "/api/applications/batch-approve",
            json={"expected_revision": 1, "application_ids": ids},
        )
        assert r.status_code == 200, r.text
        assert [x["revision"] for x in r.json()["results"]] == [2, 3]
        assert r.json()["guard_band"] == 0.5


class TestGuardBandHistory:
    def test_snapshots_restore_band_and_changes(self, client):
        # revision 1: A approved under 2.0
        a, _ = submit_and_approve(client, "A", (100, 110), expected_revision=0)
        # revision 2: band lowered to 0.5 (single effective license -> safe)
        client.post(
            CONFIRM,
            json={"expected_revision": 1, "proposed_guard_band": 0.5},
        ).raise_for_status()

        snap0 = client.get("/api/revision/0").json()
        assert snap0["guard_band"] == 2.0
        assert snap0["guard_band_changes"] == [
            {"revision": 0, "guard_band": 2.0, "created_at": snap0[
                "guard_band_changes"
            ][0]["created_at"]}
        ]

        snap1 = client.get("/api/revision/1").json()
        assert snap1["guard_band"] == 2.0
        # the stored decision band must never be retroactively rewritten
        (approval,) = snap1["approvals"]
        assert approval["application_id"] == a
        assert approval["guard_band"] == 2.0

        snap2 = client.get("/api/revision/2").json()
        assert snap2["guard_band"] == 0.5
        assert [c["guard_band"] for c in snap2["guard_band_changes"]] == [2.0, 0.5]
        assert [c["revision"] for c in snap2["guard_band_changes"]] == [0, 2]
        # historical approval record still carries its original band
        assert snap2["approvals"][0]["guard_band"] == 2.0

    def test_future_revision_still_out_of_range(self, client):
        r = client.get("/api/revision/1")
        assert r.status_code == 404
        assert r.json()["error"]["code"] == "revision_out_of_range"


class TestPersistenceAndUpgrade:
    def test_restart_does_not_override_confirmed_band(self, client, monkeypatch):
        # change the band online at revision 1
        client.post(
            CONFIRM,
            json={"expected_revision": 0, "proposed_guard_band": 5.0},
        ).raise_for_status()

        # simulate a restart with a *different* GUARD_BAND environment value
        monkeypatch.setattr(config.settings, "guard_band", 9.0)
        init_db()

        current = client.get("/api/revision/current").json()
        assert current["current_revision"] == 1
        assert current["guard_band"] == 5.0
        with SessionLocal() as session:
            bands = {
                ev.revision: ev.guard_band
                for ev in session.query(GuardBandEvent).order_by(GuardBandEvent.revision)
            }
        assert bands == {0: 2.0, 1: 5.0}

    def test_upgrade_from_pre_guard_band_schema(self, monkeypatch):
        # Rebuild the database exactly as a deployment before this feature:
        # revision_meta has no guard_band column and there are no events.
        Base.metadata.drop_all(bind=engine)
        with engine.begin() as conn:
            conn.execute(
                text(
                    "CREATE TABLE revision_meta ("
                    "id INTEGER PRIMARY KEY, revision INTEGER NOT NULL)"
                )
            )
            conn.execute(text("INSERT INTO revision_meta (id, revision) VALUES (1, 0)"))

        monkeypatch.setattr(config.settings, "guard_band", 3.5)
        init_db()  # must ADD COLUMN, backfill and seed the revision-0 event

        with SessionLocal() as session:
            meta = session.get(RevisionMeta, 1)
            assert meta.revision == 0
            assert meta.guard_band == 3.5
            events = session.query(GuardBandEvent).order_by(
                GuardBandEvent.revision
            ).all()
            assert [(ev.revision, ev.guard_band) for ev in events] == [(0, 3.5)]

        # a second init (restart) neither duplicates the seed nor changes it
        monkeypatch.setattr(config.settings, "guard_band", 8.0)
        init_db()
        with SessionLocal() as session:
            assert session.get(RevisionMeta, 1).guard_band == 3.5
            assert session.query(GuardBandEvent).count() == 1


def test_invalid_env_guard_band_fails_fast(monkeypatch):
    for raw in ("-1", "nan", "inf", "wide"):
        monkeypatch.setenv("GUARD_BAND", raw)
        try:
            config.Settings()
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"GUARD_BAND={raw} should be rejected")
