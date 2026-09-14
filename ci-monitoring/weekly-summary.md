# Weekly CI Summary

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
