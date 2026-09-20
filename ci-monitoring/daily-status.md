# Integration Test — Daily Coverage

Coverage is tracked against the three jobs that must run every night on the primary schedule (`17 3 * * *`, NIGHTLY_CRON):
- `test-pixi-cuda (openfold3-cuda12)`
- `test-pixi-cuda (openfold3-cuda13)`
- `test-pixi-amd (openfold3-rocm7)`

States: **PASSED** / **FAILED** / **CANCELLED** (reason) / **SKIPPED** (reason) / **QUEUED**

---

## 2026-09-11

### Run #254 — primary nightly (schedule `17 3 * * *`, main, [34558633289](https://github.com/aqlaboratory/openfold-3/actions/runs/34558633289))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 50 min | |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 20 min | parameter-cache — `Default checkpoint openbind-2025-06-30-174k not found in /root/.openfold3` |
| test-pixi-cuda (openfold3-cuda12) | **CANCELLED** | 25 min | fail-fast — matrix sibling (cuda13) failed; job cancelled at "Run integration test" step |

**2026-09-11: 1/3 passed · 0 skipped · 0 queued · 2 need attention**

Root cause: `actions/cache` restored the AMD runner's cache entry (stored with `HOME=/home/jan`) onto EC2 runners where `HOME=/home/ubuntu`. The tar archive unpacked to `/home/jan/.openfold3`, not into the container-visible path, leaving `/root/.openfold3` empty while the cache-hit gate skipped the S3 download. PR #404 (`feature/split-parameter-cache-by-runner`) fixes this. See `test-failures.md` for full detail.

### Run #255 — secondary nightly (schedule `17 4 * * *`, main, [34562393228](https://github.com/aqlaboratory/openfold-3/actions/runs/34562393228))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 32 min | |
| test-pixi-cuda (openfold3-cuda12) | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` was false — this slot (`17 4 * * *`) is not the repo's NIGHTLY_CRON (`17 3 * * *`); CUDA was NOT tested this run |
| test-pixi-cuda (openfold3-cuda13) | **SKIPPED** | — | same |

Note: The secondary `17 4 * * *` schedule is intended for other repos; it does not run CUDA jobs on `aqlaboratory/openfold-3`. This run's overall `success` conclusion is misleading — CUDA tests were not executed.

### Run #256 — workflow_dispatch (feature/split-parameter-cache-by-runner, [34566940737](https://github.com/aqlaboratory/openfold-3/actions/runs/34566940737))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 33 min | |
| test-pixi-cuda (openfold3-cuda12) | **CANCELLED** | 87 min | run-level manual cancellation by jnwei |
| test-pixi-cuda (openfold3-cuda13) | **CANCELLED** | 87 min | run-level manual cancellation by jnwei |

Not a scheduled nightly; PR #404 validation run on feature branch.

---

## 2026-09-12

### Run #257 — primary nightly (schedule `17 3 * * *`, main @ `d102d5c` (post-PR #404), [34670453312](https://github.com/aqlaboratory/openfold-3/actions/runs/34670453312))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 47 min | |
| test-pixi-cuda (openfold3-cuda12) | **SKIPPED** | — | see anomaly note below |
| test-pixi-cuda (openfold3-cuda13) | **SKIPPED** | — | see anomaly note below |

**2026-09-12: 1/3 passed · 2 skipped · 0 queued · 0 need attention**

**Anomaly — CUDA skipped on the primary schedule itself.** The `if:` guard is `vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON`. This run's `github.event.schedule` is `17 3 * * *` — the exact value `vars.NIGHTLY_CRON` has matched on every previous night in this log, including yesterday's run #254 (where CUDA executed, just hit the parameter-cache bug). Tonight the guard evaluated **false** on this slot for the first time recorded here, which can only mean `vars.RUN_NIGHTLY` and/or `vars.NIGHTLY_CRON` changed between run #255 (2026-09-11 04:29 UTC) and this run. I could not read the actual variable values to say which: `GET /repos/aqlaboratory/openfold-3/actions/variables` and `/actions/runners` both return "Access to this GitHub Actions path is not permitted through this proxy" in this execution environment. **Flagging for a human to check Settings → Actions → Variables (`RUN_NIGHTLY`, `NIGHTLY_CRON`).** Practical impact: PR #404's parameter-cache fix (merged 2026-09-11, see yesterday's entry) has now gone a full night without CUDA coverage to confirm it — the fix remains unverified by nightly CI.

### Run #258 — secondary nightly (schedule `17 4 * * *`, main @ `d102d5c`, [34673044066](https://github.com/aqlaboratory/openfold-3/actions/runs/34673044066))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 28 min | runs again on the secondary slot — `test-pixi-amd` carries no per-repo `if:` guard (unlike `test-pixi-cuda`), so it fires on both cron slots every night regardless of `NIGHTLY_CRON` |
| test-pixi-cuda (openfold3-cuda12) | **SKIPPED** | — | `17 4 * * *` is not `vars.NIGHTLY_CRON` — CUDA is not intended to run on this slot (expected, same as run #255) |
| test-pixi-cuda (openfold3-cuda13) | **SKIPPED** | — | same |

Expected skip on its own (secondary schedule), but combined with run #257 above, CUDA was not tested at all tonight — 0/2 CUDA matrix jobs ran across both nightly triggers for the first time in this log.

---

## 2026-09-13

### Run #259 — primary nightly (schedule `17 3 * * *`, main @ `d102d5c` (unchanged since 09-12), [34735625159](https://github.com/aqlaboratory/openfold-3/actions/runs/34735625159))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 31 min | |
| test-pixi-cuda (openfold3-cuda12) | **SKIPPED** | — | `if:` guard: `vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 3 * * *`) — same anomaly as run #257 (2026-09-12); see note below |
| test-pixi-cuda (openfold3-cuda13) | **SKIPPED** | — | same |

**2026-09-13: 1/3 passed · 2 skipped · 0 queued · 0 need attention**

**Anomaly persists — second consecutive night CUDA skipped on the primary schedule.** Run #257 (2026-09-12) was the first time the `17 3 * * *` slot failed to match `vars.NIGHTLY_CRON` after matching on every prior night logged here. Tonight's run #259, on the identical schedule slot and the identical commit (`d102d5c`, no workflow changes since yesterday), shows the same guard evaluating false again. A guard that depends only on repo variables producing a different result on the same commit and the same cron slot two nights running means the variables (`RUN_NIGHTLY` and/or `NIGHTLY_CRON`) are not what they were before run #257, not a one-off fluke. I still cannot read the values directly: `GET /repos/aqlaboratory/openfold-3/actions/variables` returns `403 Access to this GitHub Actions path is not permitted through this proxy` in this execution environment (same restriction noted 2026-09-12). **Flagging again for a human to check Settings → Actions → Variables (`RUN_NIGHTLY`, `NIGHTLY_CRON`).** Practical impact: PR #404's parameter-cache fix (merged 2026-09-11) has now gone **two full nights** without CUDA coverage on the primary schedule — still unverified by nightly CI.

### Run #260 — secondary nightly (schedule `17 4 * * *`, main @ `d102d5c`, [34737958334](https://github.com/aqlaboratory/openfold-3/actions/runs/34737958334))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 30 min | runs again on the secondary slot (no per-repo `if:` guard on `test-pixi-amd`) |
| test-pixi-cuda (openfold3-cuda12) | **SKIPPED** | — | `17 4 * * *` is not `vars.NIGHTLY_CRON` — CUDA is not intended to run on this slot (expected) |
| test-pixi-cuda (openfold3-cuda13) | **SKIPPED** | — | same |

Expected skip on its own (secondary schedule), but combined with run #259 above, CUDA was not tested at all tonight — 0/2 CUDA matrix jobs ran across both nightly triggers for the second night running.

---

## 2026-09-14

### Run #261 — primary nightly (schedule `17 3 * * *`, main @ `d102d5c` (unchanged since 09-12), [34802985987](https://github.com/aqlaboratory/openfold-3/actions/runs/34802985987))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 29 min | |
| test-pixi-cuda (openfold3-cuda12) | **SKIPPED** | — | `if:` guard evaluated false on this slot (`17 3 * * *`) — third consecutive night; see note below |
| test-pixi-cuda (openfold3-cuda13) | **SKIPPED** | — | same |

**2026-09-14: 1/3 passed · 2 skipped · 0 queued · 0 need attention**

**Anomaly persists — third consecutive night CUDA skipped on the primary schedule.** Same commit (`d102d5c`) and same cron slot as run #257 (2026-09-12) and run #259 (2026-09-13); no workflow file changes in between (`.github/workflows/integration-test.yml` still reads `if: github.event_name != 'schedule' || (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` for `test-pixi-cuda` only — `test-pixi-amd` carries no such guard). Still cannot read the actual variable values: `GET /repos/aqlaboratory/openfold-3/actions/variables` returns `403 Access to this GitHub Actions path is not permitted through this proxy` in this execution environment, and no available GitHub MCP tool exposes repository Actions variables either. **Flagging again for a human to check Settings → Actions → Variables (`RUN_NIGHTLY`, `NIGHTLY_CRON`).** Practical impact: PR #404's parameter-cache fix (merged 2026-09-11) has now gone **three full nights** without CUDA coverage on the primary schedule — still unverified by nightly CI.

### Run #262 — secondary nightly (schedule `17 4 * * *`, main @ `d102d5c`, [34806397472](https://github.com/aqlaboratory/openfold-3/actions/runs/34806397472))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 31 min | runs again on the secondary slot (no per-repo `if:` guard on `test-pixi-amd`) |
| test-pixi-cuda (openfold3-cuda12) | **SKIPPED** | — | `17 4 * * *` is not `vars.NIGHTLY_CRON` — CUDA is not intended to run on this slot (expected) |
| test-pixi-cuda (openfold3-cuda13) | **SKIPPED** | — | same |

Expected skip on its own (secondary schedule), but combined with run #261 above, CUDA was not tested at all tonight — 0/2 CUDA matrix jobs ran across both nightly triggers for the third night running.

### Run #263 — workflow_dispatch (`jandom/2026-09/ci/fix-broken-integration-tests`, [34843543097](https://github.com/aqlaboratory/openfold-3/actions/runs/34843543097))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 28 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 31 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 33 min | |

Validation run for PR #405 (adds the `test-pixi-cuda` `if:` guard to `test-pixi-amd` as well), triggered manually ~15h before the fix merged to main. Not a scheduled nightly; excluded from any day's coverage line.

---

## 2026-09-15

### Run #264 — primary nightly (schedule `17 3 * * *`, main @ `6569fcc` (post-PR #405 CI fix), [34925272269](https://github.com/aqlaboratory/openfold-3/actions/runs/34925272269))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 31 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 42 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 38 min | |

**2026-09-15: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

First full 3/3 night in this log. PR #405 (`jandom/2026-09/ci/fix-broken-integration-tests`, validated by run #263, merged to main between run #262 and this run) adds the same `if:` guard already used by `test-pixi-cuda` to `test-pixi-amd`:
`github.event_name != 'schedule' || (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)`
This resolves the anomaly flagged on 2026-09-12 through 2026-09-14 (guard evaluating false for `test-pixi-cuda` on the primary schedule slot for three consecutive nights — see those entries). PR #404's parameter-cache fix (merged 2026-09-11) is also confirmed working here: `Cache download of parameters` hit and both CUDA legs completed with no checkpoint error, ending a run of nights where that fix went unverified by nightly CI.

### Run #265 — secondary nightly (schedule `17 4 * * *`, main @ `6569fcc`, [34929043712](https://github.com/aqlaboratory/openfold-3/actions/runs/34929043712))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-amd`, no `(openfold3-rocm7)` matrix suffix, confirming it never expanded) |
| test-pixi-cuda (openfold3-cuda12) | **SKIPPED** | — | same |
| test-pixi-cuda (openfold3-cuda13) | **SKIPPED** | — | same |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`). Unlike runs #255/#258/#260/#262, `test-pixi-amd` is now guarded identically to `test-pixi-cuda` (post-PR #405) and correctly skips here too — previously it had no per-repo guard and ran unconditionally on both nightly slots every night. Does not affect the day's coverage line above (based on the primary slot, run #264, per this log's established convention).

---

## 2026-09-16

### Run #266 — primary nightly (schedule `17 3 * * *`, main @ `6569fcc` (unchanged since 09-15), [35052050368](https://github.com/aqlaboratory/openfold-3/actions/runs/35052050368))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 33 min | |
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 13 min | code — `FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin] - RuntimeError: Failed to fetch chain ID mappings from RCSB for 218 entries.` (root cause: `requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='data.rcsb.org', port=443): Read timed out. (read timeout=30)`) |
| test-pixi-cuda (openfold3-cuda13) | **CANCELLED** | 18 min | matrix fail-fast — sibling (cuda12) failed at 03:45:54Z; this leg was cancelled mid-test (`##[error]The operation was canceled.`) at 03:50:58Z while submitting `test_pocket_constraint_localizes_ligand` to the ColabFold MSA server — same fail-fast pattern as run #254 (2026-09-11), just the opposite matrix leg |

**2026-09-16: 1/3 passed · 0 skipped · 0 queued · 2 need attention**

Same commit (`6569fcc`) that passed 3/3 last night (run #264). Not a regression from a code or workflow change: `test-pixi-cuda (openfold3-cuda12)` failed because `data.rcsb.org` (RCSB PDB GraphQL API, used by `fetch_label_to_author_chain_ids` in `openfold3/core/data/tools/rscb.py` to remap ColabFold template chain IDs) read-timed-out after 30s. This is a new external-service dependency not seen failing in this log before — distinct from the previously-recurring `api.colabfold.com` MSA-submission timeouts (2026-08-09, 2026-08-11, 2026-08-12, 2026-09-08). Per this log's classification rules the failure still bins as `code` (a pytest FAILED line naming a test, with an exception raised from library code — `openfold3/core/data/tools/rscb.py:70`), the same way the recurring ColabFold timeouts have been classified, even though the trigger is an external API being slow/unavailable rather than an OF3 logic bug. See `test-failures.md` for full detail. `test-pixi-cuda (openfold3-cuda13)` was cancelled by GitHub's default matrix fail-fast (not overridden with `fail-fast: false` in `test-pixi-cuda`'s `strategy:` block) once its cuda12 sibling failed — not a timeout-minutes cap, not a supersede-by-newer-run, not a manual cancellation.

### Run #267 — secondary nightly (schedule `17 4 * * *`, main @ `6569fcc`, [35055823641](https://github.com/aqlaboratory/openfold-3/actions/runs/35055823641))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda (openfold3-cuda13) | **SKIPPED** | — | same |
| test-pixi-amd (openfold3-rocm7) | **SKIPPED** | — | same guard (post-PR #405, `test-pixi-amd` carries the identical guard); job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as run #265 (2026-09-15). Does not affect the day's coverage line above (based on the primary slot, run #266, per this log's established convention).

---

## 2026-09-17

### Run #268 — primary nightly (schedule `17 3 * * *`, main @ `6569fcc` (unchanged since 09-15), [35178424759](https://github.com/aqlaboratory/openfold-3/actions/runs/35178424759))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **FAILED** | 7 min | code — `FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin] - RuntimeError: Failed to fetch chain ID mappings from RCSB for 218 entries. Cannot proceed without chain ID re-mapping.` |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 9 min | code — same signature: `FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin] - RuntimeError: Failed to fetch chain ID mappings from RCSB for 218 entries. Cannot proceed without chain ID re-mapping.` |
| test-pixi-cuda (openfold3-cuda12) | **CANCELLED** | 15 min | matrix fail-fast — sibling (cuda13) failed at 03:42:48Z; this leg was cancelled mid-test (`##[error]The operation was canceled.`) at 03:48:10Z while submitting `test_pocket_constraint_localizes_ligand` to the ColabFold MSA server — same fail-fast mechanism as runs #254 (2026-09-11) and #266 (2026-09-16) |

**2026-09-17: 0/3 passed · 0 skipped · 0 queued · 3 need attention**

Same commit (`6569fcc7edd4afd5f887bf924ae0d2f613977763`) as the previous three nights (09-14 through 09-16). The RCSB `data.rcsb.org` chain-ID-mapping timeout first seen on one leg on 2026-09-16 (run #266, `test-pixi-cuda (openfold3-cuda12)` only) has now hit **both** directly-run legs tonight (`test-pixi-amd` and `test-pixi-cuda (openfold3-cuda13)`), with the third leg (`test-pixi-cuda (openfold3-cuda12)`) taken out by matrix fail-fast once its cuda13 sibling failed — the first night in this log where all three tracked jobs are simultaneously non-passing. Per this log's classification rules this still bins as `code` (pytest FAILED line + exception raised from library code, `openfold3/core/data/tools/rscb.py`), even though the trigger is the same external RCSB API dependency flagged on 09-16. See `test-failures.md` for full detail.

### Run #269 — secondary nightly (schedule `17 4 * * *`, main @ `6569fcc`, [35182141166](https://github.com/aqlaboratory/openfold-3/actions/runs/35182141166))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda (openfold3-cuda13) | **SKIPPED** | — | same |
| test-pixi-amd (openfold3-rocm7) | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as runs #265 and #267. Does not affect the day's coverage line above (based on the primary slot, run #268, per this log's established convention).

---

## 2026-09-18

### Run #270 — primary nightly (schedule `17 3 * * *`, main @ `6569fcc` (unchanged since 09-15), [35303406395](https://github.com/aqlaboratory/openfold-3/actions/runs/35303406395))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 28 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 34 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 31 min | |

**2026-09-18: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Same commit (`6569fcc7edd4afd5f887bf924ae0d2f613977763`) that failed all three tracked jobs on 09-17 (run #268) with the RCSB `data.rcsb.org` chain-ID-mapping timeout — tonight it passed clean with no code or workflow change in between, confirming that failure was an external-service flake rather than a regression, consistent with the `code` classification note on 09-16/09-17. Separately, run #272 below shows a `workflow_dispatch` validation of an actual fix for that RCSB call in flight on `feature/rcsb-template-call-fix`.

### Run #271 — secondary nightly (schedule `17 4 * * *`, main @ `6569fcc`, [35307132379](https://github.com/aqlaboratory/openfold-3/actions/runs/35307132379))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as runs #265, #267, and #269. Does not affect the day's coverage line above (based on the primary slot, run #270, per this log's established convention).

### Run #272 — workflow_dispatch (`feature/rcsb-template-call-fix` @ `3de86cb`, [35307541953](https://github.com/aqlaboratory/openfold-3/actions/runs/35307541953))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 36 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 40 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 38 min | |

Not a scheduled nightly; excluded from the day's coverage line per this log's established convention (same treatment as run #263 on 09-14). Validation run for a branch that changes the RCSB template chain-ID-mapping call implicated in the 09-16/09-17 failures — all three legs passed here too, but this run alone doesn't confirm the fix since tonight's own unmodified-`main` run (#270) also passed clean, i.e. the RCSB timeout wasn't reproduced on either branch tonight.

---

## 2026-09-19

### Run #273 — primary nightly (schedule `17 3 * * *`, main @ `7de748b` — first nightly run on this commit, merging PR #417 `feature/rcsb-template-call-fix` (merged 2026-09-18T08:09:10Z, after last night's run #270), [35418661454](https://github.com/aqlaboratory/openfold-3/actions/runs/35418661454))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **FAILED** | 10 min | code — `FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin] - requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.` (6 retries exhausted) |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 14 min | code — same signature: `FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin] - requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.` |
| test-pixi-cuda (openfold3-cuda12) | **CANCELLED** | 19 min | matrix fail-fast, tied to `code` — sibling (cuda13) failed at 03:45:13Z; this leg was cancelled mid-test (`##[error]The operation was canceled.`) at 03:50:34Z while submitting `test_inference_writes_outputs[msa-templates-ubiquitin]` to the ColabFold MSA server, ~5 min after the sibling failure and only 19 of the 60 allotted minutes elapsed — not a timeout-minutes cap, not superseded by a newer run (no other run in this concurrency group tonight), not manual. Same fail-fast mechanism as runs #254 (09-11), #266 (09-16), and #268 (09-17) |

**2026-09-19: 0/3 passed · 0 skipped · 0 queued · 3 need attention**

First nightly run on `7de748b7bc93adb5af0a4032d5e9f208b6e1325f` (PR #417, the RCSB template-fetch fix validated via `workflow_dispatch` run #272 last night). The RCSB fix itself wasn't implicated tonight — instead, the recurring `api.colabfold.com` MSA-submission read-timeout (same external-service signature as 09-08, last seen six nights ago) hit both directly-run legs (`test-pixi-amd`, `test-pixi-cuda (openfold3-cuda13)`), with the third leg (`test-pixi-cuda (openfold3-cuda12)`) taken out by matrix fail-fast once its cuda13 sibling failed — the same all-three-non-passing pattern as 09-17. Classifies as `code` per this log's rules (pytest FAILED line + exception from library code), even though the trigger is an external ColabFold API outage/flake, not a regression from the PR #417 merge. See `test-failures.md` for full detail.

### Run #274 — secondary nightly (schedule `17 4 * * *`, main @ `7de748b`, [35421349254](https://github.com/aqlaboratory/openfold-3/actions/runs/35421349254))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as runs #265, #267, #269, and #271. Does not affect the day's coverage line above (based on the primary slot, run #273, per this log's established convention).

---

## 2026-09-20

### Run #275 — primary nightly (schedule `17 3 * * *`, main @ `7de748b` (unchanged since 09-19), [35486753962](https://github.com/aqlaboratory/openfold-3/actions/runs/35486753962))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 29 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 35 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 32 min | |

**2026-09-20: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Same commit (`7de748b7bc93adb5af0a4032d5e9f208b6e1325f`) as last night (run #273, 09-19). Tonight all three tracked jobs passed clean with no code or workflow change in between, confirming last night's `api.colabfold.com` MSA-submission read-timeout (see `test-failures.md`, 2026-09-19) was an external-service flake rather than a regression from PR #417.

### Run #276 — secondary nightly (schedule `17 4 * * *`, main @ `7de748b`, [35489235292](https://github.com/aqlaboratory/openfold-3/actions/runs/35489235292))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as runs #265, #267, #269, #271, and #274. Does not affect the day's coverage line above (based on the primary slot, run #275, per this log's established convention).

---
