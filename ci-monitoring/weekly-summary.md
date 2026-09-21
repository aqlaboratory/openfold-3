# Weekly CI Summary

## Week of 2026-09-15 – 2026-09-21

**Scope:** 7 primary nightly runs (`17 3 * * *`) + 7 secondary nightly runs (`17 4 * * *`, all expected-skip post-PR #405) across 7 nights, against the three tracked jobs: `test-pixi-cuda (openfold3-cuda12)`, `test-pixi-cuda (openfold3-cuda13)`, `test-pixi-amd (openfold3-rocm7)`. One `workflow_dispatch` validation run (run #272, 09-18, `feature/rcsb-template-call-fix`) excluded from coverage stats per this log's established convention.

### Days with coverage < 3/3

3 of 7 days — 09-16, 09-17, and 09-19.

| Date | Primary (03:17 UTC cron) | Secondary (04:17 UTC cron) |
|---|---|---|
| 2026-09-15 | 3/3 | SKIP (expected, post-PR #405 guard) |
| 2026-09-16 | 1/3 | SKIP (expected) |
| 2026-09-17 | 0/3 | SKIP (expected) |
| 2026-09-18 | 3/3 | SKIP (expected) |
| 2026-09-19 | 0/3 | SKIP (expected) |
| 2026-09-20 | 3/3 | SKIP (expected) |
| 2026-09-21 | 3/3 | SKIP (expected) |

### Count per failure/non-pass class

| Class | Count | Where |
|---|---|---|
| `code` | 8 | 09-16 primary: cuda12 FAILED (`RuntimeError: Failed to fetch chain ID mappings from RCSB for 218 entries` — `data.rcsb.org` read-timeout) + cuda13 fail-fast CANCELLED, run #266 ([35052050368](https://github.com/aqlaboratory/openfold-3/actions/runs/35052050368)); 09-17 primary: amd FAILED + cuda13 FAILED (same RCSB chain-ID-mapping timeout signature) + cuda12 fail-fast CANCELLED, run #268 ([35178424759](https://github.com/aqlaboratory/openfold-3/actions/runs/35178424759)); 09-19 primary: amd FAILED + cuda13 FAILED (`requests.exceptions.ConnectionError` — `api.colabfold.com` read-timeout) + cuda12 fail-fast CANCELLED, run #273 ([35418661454](https://github.com/aqlaboratory/openfold-3/actions/runs/35418661454)) |

No `aws-capacity`, `gpu-unavailable`, `runner-offline`, `msa-hang`, `parameter-cache`, `build-push`, or `unclassified` non-passes this week — every non-pass job this week traces to a pytest FAILED line with an exception from test/library code (an external RCSB or ColabFold API read-timeout surfaced through `openfold3/core/data/tools/rscb.py` or `colabfold_msa_server.py`), or a matrix `fail-fast` CANCELLED cascade attributed to that same failing sibling.

**Most frequent class: `code` (8 occurrences) — the only class observed this week.**

SKIPPED results from the secondary-slot `if:` guard (every night this week, expected and by design since PR #405) are tracked in the coverage table above and are not counted in this class table, per the monitor's classification rules (SKIPPED is explained via the `if:` guard, not a log signature).

### Consecutive non-pass streak per job (as of 2026-09-21)

| Job | Current streak | Detail |
|---|---|---|
| `test-pixi-amd (openfold3-rocm7)` | **0 — currently passing** | Last non-pass: FAILED/`code` (ColabFold `api.colabfold.com` timeout), 2026-09-19 (run #273). 2 consecutive PASSED since (09-20, 09-21). Also non-pass 09-17 (RCSB timeout) with a clean 09-18 in between. |
| `test-pixi-cuda (openfold3-cuda12)` | **0 — currently passing** | Last non-pass: fail-fast CANCELLED (tied to `code`), 2026-09-19. 2 consecutive PASSED since (09-20, 09-21). |
| `test-pixi-cuda (openfold3-cuda13)` | **0 — currently passing** | Last non-pass: FAILED/`code`, 2026-09-19. 2 consecutive PASSED since (09-20, 09-21). |

All three tracked jobs are currently on a green streak, with tonight (09-21) the second consecutive 3/3 night. No job has an active non-pass streak entering the next week.

### Data-source notes

- Built directly from `daily-status.md` entries for 2026-09-15 through 2026-09-21 (runs #264–#278); no independent API re-fetch was needed since the coverage table and per-job notes for this window were already recorded contemporaneously by this monitor.
- `GET /repos/aqlaboratory/openfold-3/actions/variables` and `/repos/aqlaboratory/openfold-3/actions/runners` remain blocked in this execution environment (`403 Access to this GitHub Actions path is not permitted through this proxy`), so the live `RUN_NIGHTLY`/`NIGHTLY_CRON` variable values still could not be confirmed directly this week either — not required this week since no anomalous skip occurred on the primary slot.

---

## Week of 2026-09-08 – 2026-09-14

**Scope:** 13 scheduled nightly runs (primary `17 3 * * *` + secondary `17 4 * * *`, across 7 nights; one `workflow_dispatch` validation run on 2026-09-11 excluded from coverage stats) against the three tracked jobs: `test-pixi-cuda (openfold3-cuda12)`, `test-pixi-cuda (openfold3-cuda13)`, `test-pixi-amd (openfold3-rocm7)`.

### Days with coverage < 3/3

All 7 days — **no day this week reached full 3/3 coverage** on either nightly slot.

| Date | Primary (03:17 UTC cron) | Secondary (04:17 UTC cron) |
|---|---|---|
| 2026-09-08 | 0/3 | 0/3 |
| 2026-09-09 | 0/3 | 0/3 |
| 2026-09-10 | 0/3 | 0/3 |
| 2026-09-11 | 1/3 | 1/3 |
| 2026-09-12 | 1/3 | 1/3 |
| 2026-09-13 | 1/3 | 1/3 |
| 2026-09-14 | 1/3 | 1/3 |

### Count per failure/non-pass class

| Class | Count | Where |
|---|---|---|
| `aws-capacity` | 4 | cuda12 & cuda13 `start-aws-runner`, 09-09 primary ([34307353805](https://github.com/aqlaboratory/openfold-3/actions/runs/34307353805)); cuda12 `start-aws-runner` + cuda13 cascade-cancel, 09-10 primary ([34433550459](https://github.com/aqlaboratory/openfold-3/actions/runs/34433550459)) |
| `unclassified (log unavailable)` | 3 | amd `Build and push test image`, 09-09 primary+secondary, 09-10 primary — job logs returned HTTP 404 each time |
| `code` | 2 | amd `requests.exceptions.ConnectionError` (ColabFold API timeout) inside `test_inference_writes_outputs[msa-no_templates-ubiquitin]`, 09-08 primary+secondary — matches the `code` signature (pytest FAILED line + exception from test/library code) even though the underlying trigger is an external-service flake |
| `parameter-cache` | 2 | cuda13 FAILED + cuda12 fail-fast CANCELLED, 09-11 primary ([34558633289](https://github.com/aqlaboratory/openfold-3/actions/runs/34558633289)) |
| `runner-offline` (candidate, unconfirmed) | 1 | amd QUEUED ~9h13m before a runner picked it up, 09-10 secondary ([34437358055](https://github.com/aqlaboratory/openfold-3/actions/runs/34437358055)), missing label `amd-gpu` — could not confirm via `/actions/runners` (blocked by proxy) whether the self-hosted runner was actually offline at that time |

**Most frequent class: `aws-capacity` (4 occurrences).**

SKIPPED results from the `test-pixi-cuda` `if:` guard (09-08, 09-09 secondary, 09-10 secondary as expected non-`NIGHTLY_CRON` slots; 09-12/09-13/09-14 primary as the still-unresolved anomaly — see `daily-status.md`) are tracked in the coverage table above and are not double-counted in this class table, per the monitor's classification rules (SKIPPED is explained via the `if:` guard, not a log signature).

### Consecutive non-pass streak per job (as of 2026-09-14)

| Job | Current streak | Detail |
|---|---|---|
| `test-pixi-amd (openfold3-rocm7)` | **0 — currently passing** | Last non-pass: QUEUED/runner-offline candidate, 2026-09-10 secondary. 8 consecutive PASSED runs since (both slots, 09-11 through 09-14). |
| `test-pixi-cuda (openfold3-cuda12)` | **7/7 nights without a single PASSED result** | SKIPPED (09-08) → FAILED/`aws-capacity` (09-09) → FAILED/`aws-capacity` (09-10) → CANCELLED/fail-fast tied to `parameter-cache` (09-11) → SKIPPED (09-12, 09-13, 09-14 — `if:` guard anomaly, 3rd consecutive night as of tonight). |
| `test-pixi-cuda (openfold3-cuda13)` | **7/7 nights without a single PASSED result** | SKIPPED (09-08) → FAILED/`aws-capacity` (09-09) → CANCELLED/`aws-capacity` cascade (09-10) → FAILED/`parameter-cache` (09-11) → SKIPPED (09-12, 09-13, 09-14). |

**Headline: neither CUDA matrix leg has produced a single PASSED nightly result in the last 7 days.** Three nights of AWS-capacity/parameter-cache failures (09-08 through 09-11) were followed immediately by three nights of the `RUN_NIGHTLY`/`NIGHTLY_CRON` guard skipping CUDA outright (09-12 through 09-14, still unresolved as of tonight). PR #404's parameter-cache fix (merged 2026-09-11) has never once been exercised by nightly CI.

### Data-source notes

- The structured 3-job coverage table in `daily-status.md` was only established starting 2026-09-11; job-level states for 09-08 through 09-10 were reconstructed for this summary directly from the GitHub Actions API (`list_workflow_jobs` per run) and cross-checked against the existing `aws-outage-failures.md` / `test-failures.md` entries for those dates.
- `GET /repos/aqlaboratory/openfold-3/actions/variables` and `/repos/aqlaboratory/openfold-3/actions/runners` are both blocked in this execution environment (`403 Access to this GitHub Actions path is not permitted through this proxy`), so the live `RUN_NIGHTLY`/`NIGHTLY_CRON` variable values and the historical `amd-gpu` runner-offline status could not be confirmed directly. Both remain flagged for a human to check under Settings → Actions.

---
