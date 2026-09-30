#!/usr/bin/env bash
# End-to-end curl walkthrough for the Temporary Event Radio Spectrum API.
# Usage: ./examples/walkthrough.sh [BASE_URL]
set -euo pipefail

BASE="${1:-http://localhost:8000}"
J() { python3 -m json.tool; }

post() { curl -s -X POST "$BASE$1" -H 'Content-Type: application/json' -d "${2:-}"; }

echo "== health =="
curl -s "$BASE/health"; echo

echo "== 1. invalid submission (422, whole request rejected) =="
post /api/applications '{
  "name":"bad",
  "circle":{"cx":0,"cy":0,"radius":-5},
  "frequency":{"low":200,"high":100},
  "time_window":{"start":"2026-10-01T10:00:00Z","end":"2026-10-01T08:00:00Z"}
}' | J

echo "== 2. submit A =="
A=$(post /api/applications '{
  "name":"Festival A",
  "circle":{"cx":0,"cy":0,"radius":1000},
  "frequency":{"low":100,"high":110},
  "time_window":{"start":"2026-10-01T08:00:00Z","end":"2026-10-01T12:00:00Z"}
}')
echo "$A" | J
A_ID=$(echo "$A" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')

echo "== 3. preview A =="
post "/api/applications/$A_ID/preview" | J

echo "== 4. approve A at revision 0 =="
post "/api/applications/$A_ID/approve" '{"expected_revision":0}' | J

echo "== 5. submit B (1 MHz gap < 2 MHz guard band) =="
B=$(post /api/applications '{
  "name":"Marathon B",
  "circle":{"cx":100,"cy":0,"radius":1000},
  "frequency":{"low":111,"high":112},
  "time_window":{"start":"2026-10-01T09:00:00Z","end":"2026-10-01T11:00:00Z"}
}')
B_ID=$(echo "$B" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
post "/api/applications/$B_ID/preview" | J

echo "== 6. stale approval (expected 0, current 1) =="
post "/api/applications/$B_ID/approve" '{"expected_revision":0}' | J || true

echo "== 7. current revision but interference =="
post "/api/applications/$B_ID/approve" '{"expected_revision":1}' | J || true

echo "== 8. revoke A =="
post "/api/applications/$A_ID/revoke" '{"reason":"postponed"}' | J

echo "== 9. approve B at revision 2 =="
post "/api/applications/$B_ID/approve" '{"expected_revision":2}' | J

echo "== 10. history snapshots =="
curl -s "$BASE/api/revision/1" | J
curl -s "$BASE/api/revision/2" | J
curl -s "$BASE/api/revision/current" | J

echo "== 11. submit three more devices for one event (disjoint bands) =="
mk() { post /api/applications "{
  \"name\":\"$1\",
  \"circle\":{\"cx\":0,\"cy\":0,\"radius\":1000},
  \"frequency\":{\"low\":$2,\"high\":$3},
  \"time_window\":{\"start\":\"2026-10-01T13:00:00Z\",\"end\":\"2026-10-01T18:00:00Z\"}
}"; }
C=$(mk "Device C" 200 210); C_ID=$(echo "$C" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
D=$(mk "Device D" 300 310); D_ID=$(echo "$D" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
E=$(mk "Device E" 400 410); E_ID=$(echo "$E" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
echo "ids: $C_ID $D_ID $E_ID"

echo "== 12. batch approve at revision 3, input order D,C,E -> revisions 4,5,6 =="
post /api/applications/batch-approve \
  "{\"expected_revision\":3,\"application_ids\":[$D_ID,$C_ID,$E_ID]}" | J

echo "== 13. snapshot as of the second item (revision 5) has D and C effective =="
curl -s "$BASE/api/revision/5" | J

echo "== 14. failing batch: two members clash with each other -> whole group rejected =="
F=$(mk "Device F" 500 510); F_ID=$(echo "$F" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
G=$(mk "Device G" 505 515); G_ID=$(echo "$G" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
H=$(mk "Device H" 600 610); H_ID=$(echo "$H" | python3 -c 'import json,sys; print(json.load(sys.stdin)["id"])')
post /api/applications/batch-approve \
  "{\"expected_revision\":6,\"application_ids\":[$F_ID,$G_ID,$H_ID]}" | J || true
echo "revision is still 6 and all three are pending:"
curl -s "$BASE/api/revision/current" | python3 -c 'import json,sys; print("current_revision =", json.load(sys.stdin)["current_revision"])'

echo "== 15. malformed batches (duplicate ids / missing id) =="
post /api/applications/batch-approve \
  "{\"expected_revision\":6,\"application_ids\":[$F_ID,$F_ID]}" | J || true
post /api/applications/batch-approve \
  "{\"expected_revision\":6,\"application_ids\":[999999]}" | J || true

echo "== 16. preview guard band 100 MHz (state unchanged; two effective pairs clash) =="
post /api/guard-band/preview '{"proposed_guard_band":100}' | J

echo "== 17. confirm 100 MHz at revision 6 -> interference, band/revision/set unchanged =="
post /api/guard-band/confirm \
  '{"expected_revision":6,"proposed_guard_band":100}' | J || true
curl -s "$BASE/api/revision/current" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("current_revision =", d["current_revision"], "guard_band =", d["guard_band"])'

echo "== 18. invalid proposed values are rejected explicitly (422 envelope) =="
post /api/guard-band/preview '{"proposed_guard_band":-1}' | J || true
post /api/guard-band/confirm '{"expected_revision":6,"proposed_guard_band":-1}' | J || true

echo "== 19. stale confirmation -> stale_revision =="
post /api/guard-band/confirm '{"expected_revision":0,"proposed_guard_band":50}' | J || true

echo "== 20. confirm 50 MHz at revision 6 -> exactly one new revision (7) =="
post /api/guard-band/confirm \
  '{"expected_revision":6,"proposed_guard_band":50}' | J

echo "== 21. history restores the band per revision; past decision bands are untouched =="
curl -s "$BASE/api/revision/6" | python3 -c '
import json,sys
d=json.load(sys.stdin)
print("revision 6 guard_band =", d["guard_band"])
print("approval decision bands =", [a["guard_band"] for a in d["approvals"]])'
curl -s "$BASE/api/revision/7" | python3 -c '
import json,sys
d=json.load(sys.stdin)
print("revision 7 guard_band =", d["guard_band"])
print("guard_band_changes =", [(c["revision"], c["guard_band"]) for c in d["guard_band_changes"]])
print("approval decision bands still =", [a["guard_band"] for a in d["approvals"]])'
curl -s "$BASE/api/revision/current" | python3 -c 'import json,sys; d=json.load(sys.stdin); print("current_revision =", d["current_revision"], "guard_band =", d["guard_band"])'
