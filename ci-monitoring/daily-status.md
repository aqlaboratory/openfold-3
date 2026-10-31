# Integration Test — Daily Coverage

Coverage is tracked against the jobs that must run every night on the primary schedule (`17 3 * * *`, NIGHTLY_CRON). Scope was three jobs through 2026-10-13; expanded to five effective 2026-10-14 (see that date's addendum) once the Brev comparison jobs were added in PR #448:
- `test-pixi-cuda (openfold3-cuda12)`
- `test-pixi-cuda (openfold3-cuda13)`
- `test-pixi-cuda-brev (openfold3-cuda12)` (since 2026-10-14)
- `test-pixi-cuda-brev (openfold3-cuda13)` (since 2026-10-14)
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

## 2026-09-21

### Run #277 — primary nightly (schedule `17 3 * * *`, main @ `7de748b` (unchanged since 09-19), [35557906410](https://github.com/aqlaboratory/openfold-3/actions/runs/35557906410))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 28 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 33 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 31 min | |

**2026-09-21: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Second consecutive clean night on the unchanged `7de748b` commit (following run #275, 09-20).

### Run #278 — secondary nightly (schedule `17 4 * * *`, main @ `7de748b`, [35561359308](https://github.com/aqlaboratory/openfold-3/actions/runs/35561359308))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as runs #265, #267, #269, #271, #274, and #276. Does not affect the day's coverage line above (based on the primary slot, run #277, per this log's established convention).

---

## 2026-09-22

### Run #279 — primary nightly (schedule `17 3 * * *`, main @ `7de748b` (unchanged since 09-19), [35683449269](https://github.com/aqlaboratory/openfold-3/actions/runs/35683449269))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 28 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 33 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 31 min | |

**2026-09-22: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Third consecutive clean night on the unchanged `7de748b7bc93adb5af0a4032d5e9f208b6e1325f` commit (following run #275 on 09-20 and run #277 on 09-21).

### Run #280 — secondary nightly (schedule `17 4 * * *`, main @ `7de748b`, [35687077344](https://github.com/aqlaboratory/openfold-3/actions/runs/35687077344))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as runs #265, #267, #269, #271, #274, #276, and #278. Does not affect the day's coverage line above (based on the primary slot, run #279, per this log's established convention).

---

## 2026-09-23

### Run #281 — primary nightly (schedule `17 3 * * *`, main @ `68b9c5e` — first nightly run on this commit, merging PR #424 dependabot bump of `docker/build-push-action` 7.3.0→7.4.0 (unchanged `7de748b` for the prior 3 nights), [35814661788](https://github.com/aqlaboratory/openfold-3/actions/runs/35814661788))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 32 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 37 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 35 min | |

**2026-09-23: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Fourth consecutive clean night (following runs #275 on 09-20, #277 on 09-21, #279 on 09-22). First nightly run since main advanced from `7de748b` to `68b9c5e` (PR #424, a dependency-bot bump touching only the `docker/build-push-action` version pin in the reusable workflows) — no regression observed from the bump.

### Run #283 — secondary nightly (schedule `17 4 * * *`, main @ `68b9c5e`, [35818592857](https://github.com/aqlaboratory/openfold-3/actions/runs/35818592857))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as runs #265, #267, #269, #271, #274, #276, #278, and #280. Does not affect the day's coverage line above (based on the primary slot, run #281, per this log's established convention).

### Run #282 — workflow_dispatch (`feature/ci-upstream-az-fallback` @ `5a8f3d8`, [35818035055](https://github.com/aqlaboratory/openfold-3/actions/runs/35818035055))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **FAILED** | 10 min | `code` — `FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin] - requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.` (6 retries exhausted), independent of the branch under test |
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 0 min | `aws-capacity` at `start-aws-runner` / "Report launch outcome" — `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 0 min | same `aws-capacity` signature — `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |

Not a scheduled nightly; excluded from the day's coverage line per this log's established convention (same treatment as runs #263 on 09-14 and #272 on 09-18). Validation run for PR #426, which swaps the hand-rolled multi-pool AWS capacity fallback (added in #400) for the upstream `start-aws-gha-runner` v1.4.0 per-region AZ fallback. Both CUDA legs still hit `g5.4xlarge` capacity exhaustion in both regions under the new action — inconclusive on whether v1.4.0's fallback is effective, since a hard double-region outage isn't distinguishable from a fallback that isn't walking zones; not logged to `aws-outage-failures.md` since it's off the tracked nightly schedule. The AMD leg's ColabFold timeout is the same `code` signature already tracked for the nightly job and is unrelated to the AWS-runner change under test in this run.

---

## 2026-09-24

### Run #284 — primary nightly (schedule `17 3 * * *`, main @ `3a9ff90` — first nightly run on this commit (PR #337, docs-only MSA chain-level fix; prior 68b9c5e unchanged for 1 night), [35951749406](https://github.com/aqlaboratory/openfold-3/actions/runs/35951749406))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 28 min | |
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 3 min | `aws-capacity` at `start-aws-runner` / "Report launch outcome" — `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 3 min | same `aws-capacity` signature — `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |

**2026-09-24: 1/3 passed · 0 skipped · 0 queued · 2 need attention**

AWS `g5.4xlarge` capacity exhaustion across all six capacity pools (3x us-east-2, 3x us-west-2) on both CUDA legs' `start-aws-runner` step, cascading to skipped test jobs and failed `stop-aws-runner` (no instance to stop) — same signature as runs #252 (09-10) and #282 (09-22 `workflow_dispatch`, off-schedule). AMD leg (self-hosted runner, unaffected by AWS capacity) passed. Recorded to `aws-outage-failures.md`.

### Run #285 — secondary nightly (schedule `17 4 * * *`, main @ `3a9ff90`, [35955919993](https://github.com/aqlaboratory/openfold-3/actions/runs/35955919993))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #284, per this log's established convention).

---

## 2026-09-25

### Run #286 — primary nightly (schedule `17 3 * * *`, main @ `3a9ff90` — unchanged since run #284 on 09-24, [36090588818](https://github.com/aqlaboratory/openfold-3/actions/runs/36090588818))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 28 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 38 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 45 min | |

**2026-09-25: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Clean recovery the night after the 09-24 AWS `g5.4xlarge` capacity exhaustion (run #284) — both CUDA legs launched on pool 1 in `us-east-2`/`us-west-2` without needing the pool 2–6 fallback. AMD leg unaffected as usual.

### Run #287 — secondary nightly (schedule `17 4 * * *`, main @ `3a9ff90`, [36094745386](https://github.com/aqlaboratory/openfold-3/actions/runs/36094745386))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #286, per this log's established convention).

Note: run #288 ([36102270668](https://github.com/aqlaboratory/openfold-3/actions/runs/36102270668)), a `workflow_dispatch` validation run on `ci/pr414-integration`, was still `in_progress` at scan time and is excluded from tonight's coverage (not a scheduled nightly, and not yet completed).

---

## 2026-09-26

### Run #289 — primary nightly (schedule `17 3 * * *`, main @ `f299981` — first nightly run on this commit, merging PR #425 (`ahmedtaha100/issue-420-cached-template-reuse`, fixes cached template reuse in inference query sets; prior `3a9ff90` unchanged for 2 nights), [36214957390](https://github.com/aqlaboratory/openfold-3/actions/runs/36214957390))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 33 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 37 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 35 min | |

**2026-09-26: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Clean night on the first nightly run since main advanced from `3a9ff90` to `f299981` (PR #425) — no regression observed from the merge.

### Run #290 — secondary nightly (schedule `17 4 * * *`, main @ `f299981`, [36217952280](https://github.com/aqlaboratory/openfold-3/actions/runs/36217952280))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #289, per this log's established convention).

**Closing run #288** ([36102270668](https://github.com/aqlaboratory/openfold-3/actions/runs/36102270668), `workflow_dispatch` on `ci/pr414-integration`), flagged `in_progress` at yesterday's (09-25) scan time: it has since completed. `test-pixi-amd (openfold3-rocm7)` **PASSED** (33 min); both `test-pixi-cuda` legs **FAILED** (`code`) — `FAILED openfold3/tests/test_cuda_allocator.py::test_matches_snapshot - AssertionError: Values are not sufficiently close.` on cuda12 and the identical signature on cuda13, at the "Run integration test" step. The test's own warning names the root cause as a stored-snapshot/runtime environment mismatch, not a regression: `Snapshot environment mismatch in nvidia/: torch_version: stored=2.12.1, current=2.10.0, gpu_name: stored=NVIDIA GB10, current=NVIDIA A10G`. Not a scheduled nightly; excluded from any day's coverage line per this log's established convention (same treatment as runs #263, #272, #282).

---

## 2026-09-27

### Run #291 — primary nightly (schedule `17 3 * * *`, main @ `f299981` (unchanged since run #289 on 09-26), [36291703877](https://github.com/aqlaboratory/openfold-3/actions/runs/36291703877))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 30 min | |
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 2 min | `aws-capacity` at `start-aws-runner` / "Report launch outcome" — `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 2 min | same `aws-capacity` signature — `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |

**2026-09-27: 1/3 passed · 0 skipped · 0 queued · 2 need attention**

AWS `g5.4xlarge` capacity exhaustion across all six capacity pools (3x us-east-2, 3x us-west-2) on both CUDA legs' `start-aws-runner` step — same signature as runs #252 (09-10), #282 (09-22 `workflow_dispatch`, off-schedule), and #284 (09-24). Underlying `botocore.exceptions.ClientError: ... (InsufficientInstanceCapacity) ...` present on both legs before the pool-exhaustion "No capacity" error. AMD leg (self-hosted runner, unaffected by AWS capacity) passed. Recorded to `aws-outage-failures.md`.

### Run #292 — secondary nightly (schedule `17 4 * * *`, main @ `f299981`, [36294497332](https://github.com/aqlaboratory/openfold-3/actions/runs/36294497332))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #291, per this log's established convention).

---

## 2026-09-28

### Run #293 — primary nightly (schedule `17 3 * * *`, main @ `f299981` (unchanged since run #289 on 09-26), [36374432516](https://github.com/aqlaboratory/openfold-3/actions/runs/36374432516))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 32 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 37 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 34 min | |

**2026-09-28: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Clean recovery the night after the 09-27 AWS `g5.4xlarge` capacity exhaustion (run #291, attempt 1) — both CUDA legs launched on pool 1 in `us-east-2`/`us-west-2` without needing the pool 2–6 fallback. AMD leg unaffected as usual.

### Run #294 — secondary nightly (schedule `17 4 * * *`, main @ `f299981`, [36378258603](https://github.com/aqlaboratory/openfold-3/actions/runs/36378258603))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #293, per this log's established convention).

**Closing run #291** ([36291703877](https://github.com/aqlaboratory/openfold-3/actions/runs/36291703877), primary nightly 09-27), flagged in yesterday's entry as 1/3 passed with AWS `g5.4xlarge` capacity exhaustion on both CUDA legs (attempt 1, 03:32–03:35 UTC): it was manually re-run (attempt 2, triggering actor `jandom`) at 2026-09-27T14:38:27Z–15:20:01Z, and this time all three jobs passed — `test-pixi-amd (openfold3-rocm7)` (28 min), `test-pixi-cuda (openfold3-cuda13)` (31 min), `test-pixi-cuda (openfold3-cuda12)` (33 min) — with both CUDA legs launching on pool 1 in `us-east-2`/`us-west-2` on the retry, no fallback pools needed. Not counted toward any coverage line (09-27's line already reflects attempt 1 and this log never overwrites a past entry); recorded here purely as a same-day resolution note for that outage.

---

## 2026-09-29

### Run #295 — primary nightly (schedule `17 3 * * *`, main @ `8605a8c` (`#426`, merged since run #293/09-28), [36517655108](https://github.com/aqlaboratory/openfold-3/actions/runs/36517655108))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 30 min | |
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 2 min | `aws-capacity` at `start-aws-runner` / "Report launch outcome" — `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 2 min | same `aws-capacity` signature — `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |

**2026-09-29: 1/3 passed · 0 skipped · 0 queued · 2 need attention**

AWS `g5.4xlarge` capacity exhaustion across all six capacity pools (3x us-east-2, 3x us-west-2) on both CUDA legs' `start-aws-runner` step — same signature as runs #252 (09-10), #282 (09-22 `workflow_dispatch`, off-schedule), #284 (09-24), and #291 attempt 1 (09-27). Both attempts landed on `Unsupported`/`InsufficientInstanceCapacity` across every zone tried before the wrapper step raised `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2`. AMD leg (self-hosted runner, unaffected by AWS capacity) passed. Recorded to `aws-outage-failures.md`.

### Run #296 — secondary nightly (schedule `17 4 * * *`, main @ `8605a8c`, [36522054758](https://github.com/aqlaboratory/openfold-3/actions/runs/36522054758))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #295, per this log's established convention).

---

## 2026-09-30

### Run #297 — primary nightly (schedule `17 3 * * *`, main @ `8605a8c` (unchanged since run #295 on 09-29), [36664988247](https://github.com/aqlaboratory/openfold-3/actions/runs/36664988247))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 31 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 40 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 37 min | |

**2026-09-30: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Clean run the night after the 09-29 AWS `g5.4xlarge` capacity exhaustion (run #295) — both CUDA legs launched without a `start-aws-runner` retry needed. No re-run of run #295 was triggered (`run_attempt` stayed at 1, conclusion `failure`), so 09-29's coverage line stands unrevised.

### Run #298 — secondary nightly (schedule `17 4 * * *`, main @ `8605a8c`, [36669325658](https://github.com/aqlaboratory/openfold-3/actions/runs/36669325658))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #297, per this log's established convention).

---

## 2026-10-01

### Run #299 — primary nightly (schedule `17 3 * * *`, main @ `8bfebfcc` (PR #433 "Don't skip GPU tests on ROCm", merged 2026-10-01T03:07:13Z — new since run #297/09-30), [36811446760](https://github.com/aqlaboratory/openfold-3/actions/runs/36811446760))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 52 min | |
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 2 min | `aws-capacity` at `start-aws-runner` / "Report launch outcome" — `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 2 min | same `aws-capacity` signature — `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |

**2026-10-01: 1/3 passed · 0 skipped · 0 queued · 2 need attention**

AWS `g5.4xlarge` capacity exhaustion across all six capacity pools (3x us-east-2, 3x us-west-2, `us-west-2d` additionally `Unsupported` for this instance type) on both CUDA legs' `start-aws-runner` step — same signature as runs #252 (09-10), #284 (09-24), #291 attempt 1 (09-27), and #295 (09-29). AMD leg (self-hosted runner, unaffected by AWS capacity) passed — first nightly run under PR #433 ("Don't skip GPU tests on ROCm", merged 2026-10-01T03:07:13Z, ~31 min before this run started), no regression observed. Recorded to `aws-outage-failures.md`.

### Run #300 — secondary nightly (schedule `17 4 * * *`, main @ `8bfebfcc`, [36815628951](https://github.com/aqlaboratory/openfold-3/actions/runs/36815628951))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #299, per this log's established convention).

---

## 2026-10-02

### Run #301 — primary nightly (schedule `17 3 * * *`, main @ `270425a` — PR #434 "fix: make register_dataset_config return the decorated class", merged since run #299/10-01, [36960646988](https://github.com/aqlaboratory/openfold-3/actions/runs/36960646988))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 41 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 35 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 33 min | |

**2026-10-02: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Clean recovery the night after the 10-01 AWS `g5.4xlarge` capacity exhaustion (run #299) — both CUDA legs launched on pool 1 in `us-east-2` without needing the pool 2–6 fallback (`Create cloud runner (us-west-2)` skipped on both legs, meaning the first region tried succeeded). First nightly run since main advanced from `8bfebfcc` to `270425a` (PR #434) — no regression observed from the merge.

### Run #302 — secondary nightly (schedule `17 4 * * *`, main @ `270425a`, [36964949436](https://github.com/aqlaboratory/openfold-3/actions/runs/36964949436))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #301, per this log's established convention).

---

## 2026-10-03

### Run #303 — primary nightly (schedule `17 3 * * *`, main @ `1d48c84` — PR #282 "Do boolean mask sums directly in fp32" (unchanged `270425a` for 1 night; merged since run #301/10-02), [37094018501](https://github.com/aqlaboratory/openfold-3/actions/runs/37094018501))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 53 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 33 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 30 min | |

**2026-10-03: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Clean night on the first nightly run since main advanced from `270425a` to `1d48c84` (PR #282, fp32 boolean-mask-sum change in the loss/diffusion masking code) — no regression observed from the merge. Both CUDA legs launched on pool 1 in `us-east-2` without needing fallback (`Create cloud runner (us-west-2)` skipped on both legs); `Cache download of parameters` hit cleanly on both CUDA legs.

### Run #304 — secondary nightly (schedule `17 4 * * *`, main @ `1d48c84`, [37096930930](https://github.com/aqlaboratory/openfold-3/actions/runs/37096930930))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #303, per this log's established convention).

---

## 2026-10-04

### Run #305 — primary nightly (schedule `17 3 * * *`, main @ `1d48c84` — unchanged for 2 nights since run #303/10-03, [37177368403](https://github.com/aqlaboratory/openfold-3/actions/runs/37177368403))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 45 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 53 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 45 min | |

**2026-10-04: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Clean night; both CUDA legs launched on pool 1 in `us-east-2` without needing the pool 2–6 fallback (`Create cloud runner (us-west-2)` skipped on both legs). Note: this run's `created_at` (04:33:51Z) was ~76 min after the 03:17 UTC cron fire time — markedly longer than the typical 14–23 min scheduler delay seen on prior nights in this log; the secondary slot run (#306 below) was delayed ~100 min similarly. Likely GitHub Actions scheduler contention; no impact on job outcomes.

### Run #306 — secondary nightly (schedule `17 4 * * *`, main @ `1d48c84`, [37181355554](https://github.com/aqlaboratory/openfold-3/actions/runs/37181355554))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #305, per this log's established convention).

---

## 2026-10-05

### Run #307 — primary nightly (schedule `17 3 * * *`, main @ `1d48c84` — unchanged for 3 nights since run #303/10-03, [37260402169](https://github.com/aqlaboratory/openfold-3/actions/runs/37260402169))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **FAILED** | 12 min | failed at step "Run integration test" (03:41:17–03:52:51); `unclassified (log unavailable)` — see note below |
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 20 min | failed at step "Run integration test" (03:45:11–04:04:28); `unclassified (log unavailable)` — see note below |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 20 min | failed at step "Run integration test" (03:45:01–04:04:33); `unclassified (log unavailable)` — see note below |

**2026-10-05: 0/3 passed · 0 skipped · 0 queued · 3 need attention**

First all-three-failure night since run #273 (09-19). All three legs' `start-aws-runner`/checkout/build/parameter-cache steps succeeded; each failed specifically at "Run integration test" within the same 11–20 minute window, which is consistent with a shared cause rather than three independent failures, but **this could not be confirmed from the job logs**: every attempt this session to fetch job logs (`gh api .../actions/jobs/<id>/logs`, `gh run view --log`, and a direct follow of the Azure Blob redirect) hit a `403 Forbidden`. The session's egress proxy status (`recentRelayFailures`) confirms this is an organization network-policy denial at the CONNECT level (`connect_rejected`, "gateway answered 403 to CONNECT") against both `productionresultssa1.blob.core.windows.net:443` and `results-receiver.actions.githubusercontent.com:443` — the two hosts GitHub Actions log downloads redirect to — not a GitHub-side error or an expired signature. Per this session's operating constraints, policy denials are not to be retried or routed around, so all three jobs are recorded `unclassified (log unavailable)` per this log's fallback rule and flagged for human follow-up, both on the underlying test failure itself and on restoring log-fetch access for future nightly scans. Recorded to `test-failures.md`.

### Run #308 — secondary nightly (schedule `17 4 * * *`, main @ `1d48c84`, [37264771144](https://github.com/aqlaboratory/openfold-3/actions/runs/37264771144))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #307, per this log's established convention).

---

## 2026-10-06

### Run #309 — primary nightly (schedule `17 3 * * *`, main @ `939ebd4` — PR #436 "Start --num_model_seeds from the configured seed instead of a fixed 42" (merged since run #307–308/10-05, previously on `1d48c84`), [37409527801](https://github.com/aqlaboratory/openfold-3/actions/runs/37409527801))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 47 min | |
| test-pixi-cuda (openfold3-cuda12) | **CANCELLED** | 60 min | hit timeout-minutes cap (60 min); `unclassified` — see note below |
| test-pixi-cuda (openfold3-cuda13) | **CANCELLED** | 60 min | hit timeout-minutes cap (60 min); `unclassified` — see note below |

**2026-10-06: 1/3 passed · 0 skipped · 0 queued · 2 need attention**

Both CUDA legs ran the full `test-openfold-docker-pixi` budget and were cancelled by the job-level `timeout-minutes: 60` cap (job wall time 60m28s and 60m29s respectively, matching the cap in `ci-integration-test-pixi-cuda-reusable.yml`) — not a fail-fast cascade (no sibling failure triggered it) and not superseded by a newer run (no other run shares either concurrency group tonight). Neither matches this log's `msa-hang` signature precisely enough to classify as such: cuda12's "Run integration test" step was mid-`openfold3/tests/inference/test_templates.py::test_template_lowers_rmsd[1y57]` (not a `test_inference_writes_outputs[msa-...]` case) with no `PASSED`/`FAILED` line following, and no `TimeoutError: MSA server` string anywhere in the log; cuda13's log shows `PASSED` on the prior test at 04:32:37Z and then **zero output for ~3.5 minutes** before the cancellation at 04:36:01Z — no next-test node ID was ever printed, unlike every other test transition in both jobs' logs (which print the node ID within single-digit seconds of the prior `PASSED`). That silent gap doesn't literally match a listed signature but is worth a closer look. Recorded `unclassified` in `test-failures.md`, flagged for human review. Note: earlier in the cuda12 job, on a *passing* test (`test_pocket_constraint_localizes_ligand`), the ColabFold MSA server rate-limited the run for ~10m44s (`Sleeping for Ns. Reason: RATELIMIT`, repeated ~80 times) — not itself the cause of the cancellation, but it shows the 60-minute budget is running tight on nights when ColabFold is slow, which may help explain why the suite ran out of time mid-suite rather than failing outright.

### Run #310 — secondary nightly (schedule `17 4 * * *`, main @ `939ebd4`, [37414182897](https://github.com/aqlaboratory/openfold-3/actions/runs/37414182897))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #309, per this log's established convention).

---

## 2026-10-07

### Run #311 — primary nightly (schedule `17 3 * * *`, main @ `332d3a1` — PR #414 "Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default" (merged 2026-10-06T12:23:40Z, new since run #309/10-06's `939ebd4`), [37567531088](https://github.com/aqlaboratory/openfold-3/actions/runs/37567531088))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 42 min | |
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 23 min | `code` — `FAILED openfold3/tests/test_cuda_allocator.py::test_matches_snapshot - AssertionError: Values are not sufficiently close.` at the "Run integration test" step |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 22 min | same signature — `FAILED openfold3/tests/test_cuda_allocator.py::test_matches_snapshot - AssertionError: Values are not sufficiently close.` at the "Run integration test" step |

**2026-10-07: 1/3 passed · 0 skipped · 0 queued · 2 need attention**

`start-aws-runner` succeeded on both CUDA legs (ruling out `aws-capacity`/`gpu-unavailable`); each failed specifically inside pytest at "Run integration test". Both legs' pytest run emits the same `UserWarning` seen on 2026-09-25/26 (run #288, a `workflow_dispatch` off-schedule run): `Snapshot environment mismatch in nvidia/: torch_version: stored=2.12.1, current=2.10.0, gpu_name: stored=NVIDIA GB10, current=NVIDIA A10G` (`openfold3/tests/conftest.py:261`), i.e. the stored allocator snapshot was captured on different hardware/torch version than the CI runner uses — the same stored-snapshot/runtime mismatch flagged as the likely root cause back then. New this time: main advanced from `939ebd4` (run #309/10-06, which had its own `unclassified` CUDA-leg cancellation) to `332d3a1`, which includes PR #414 — a change to `PYTORCH_CUDA_ALLOC_CONF` (`expandable_segments:True`) that directly touches CUDA allocator behavior, the exact subsystem `test_matches_snapshot` exercises. This failure is therefore classified `code` per this log's rules (pytest FAILED line + exception from test/library code), but given the direct overlap between PR #414's change and the failing test's subsystem, it should not be waved off purely as the pre-existing environment-mismatch issue without a human confirming whether PR #414 shifted the allocator's actual byte counts (not just the stored-snapshot env staleness). First time this specific `test_matches_snapshot` failure has hit the regular nightly schedule — it was previously only seen on the off-schedule run #288. Recorded to `test-failures.md`.

### Run #312 — secondary nightly (schedule `17 4 * * *`, main @ `332d3a1`, [37572037787](https://github.com/aqlaboratory/openfold-3/actions/runs/37572037787))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #311, per this log's established convention).

### Run #313 — workflow_dispatch (`jandom/2026-10/ci/fix-daily-tests-after-expandable-segments` @ `6dcb629`, [37603411922](https://github.com/aqlaboratory/openfold-3/actions/runs/37603411922))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 38 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 39 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 35 min | |

Not a scheduled nightly; excluded from the day's coverage line per this log's established convention (same treatment as runs #256, #263, #272, and #282). Validation run for commit `6dcb629` ("ci: fix daily tests after expandable_segments", Jan Domanski, 2026-10-07T09:38:35Z) — almost certainly the fix for run #311's (10-07) `test_matches_snapshot` allocator-snapshot failure. All three legs passed on this commit. `6dcb629` is also `main`'s head for tonight's primary run (#314) below, so this run is read as pre-confirmation that the fix holds before it hit the nightly schedule.

---

## 2026-10-08

### Run #314 — primary nightly (schedule `17 3 * * *`, main @ `6dcb629` — first nightly run on this commit ("ci: fix daily tests after expandable_segments", new since run #311–312/10-07's `332d3a1`), [37723512781](https://github.com/aqlaboratory/openfold-3/actions/runs/37723512781))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 34 min | |
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 2 min | `aws-capacity` at `start-aws-runner` / "Report launch outcome" — `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 2 min | same `aws-capacity` signature — `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |

**2026-10-08: 1/3 passed · 0 skipped · 0 queued · 2 need attention**

AWS `g5.4xlarge` capacity exhaustion across all six capacity pools (3x us-east-2 zones InsufficientInstanceCapacity, 3x us-west-2 zones — 2a/2b/2c InsufficientInstanceCapacity, 2d Unsupported for this instance type) on both CUDA legs' `start-aws-runner` step, failing at "Report launch outcome"; corresponding `stop-aws-runner` failed as a cascading consequence (no instance to stop) and `test-openfold-docker-pixi` never ran (cuda13: `skipped`; cuda12: `cancelled` — matrix sibling raced to the same empty-outputs state slightly differently, same root cause). Same signature as runs #252 (09-10), #282 (09-22, `workflow_dispatch`), #284 (09-24), #291 (09-27), #295 (09-29), and #299 (10-01). AMD leg (self-hosted runner, unaffected by AWS capacity) passed. Not related to the `6dcb629` allocator fix validated by run #313 just above — that fix held (AMD ran the full suite clean); tonight's non-pass is purely AWS infrastructure. Recorded to `aws-outage-failures.md`.

### Run #315 — secondary nightly (schedule `17 4 * * *`, main @ `6dcb629`, [37728067887](https://github.com/aqlaboratory/openfold-3/actions/runs/37728067887))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #314, per this log's established convention).

---

## 2026-10-09

### Run #316 — primary nightly (schedule `17 3 * * *`, main @ `1d85185` — merge "Merge branch 'public-main' into internal-main" (Jan Domanski, 1774+/180- across 29 files, merged 2026-10-08T14:54:31Z, new since run #314/315's `6dcb629`), [37880128785](https://github.com/aqlaboratory/openfold-3/actions/runs/37880128785))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 58 min | |
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 2 min | `aws-capacity` at `start-aws-runner` / "Report launch outcome" — `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 2 min | same `aws-capacity` signature — `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |

**2026-10-09: 1/3 passed · 0 skipped · 0 queued · 2 need attention**

AWS `g5.4xlarge` capacity exhaustion across all six capacity pools (3x us-east-2 zones InsufficientInstanceCapacity, 3x us-west-2 zones — 2a/2b/2c InsufficientInstanceCapacity, 2d Unsupported for this instance type) on both CUDA legs' `start-aws-runner` step, failing at "Report launch outcome"; corresponding `stop-aws-runner` failed as a cascading consequence (no instance to stop) and `test-openfold-docker-pixi` never ran (both legs: `skipped`). Same signature as runs #252 (09-10), #282 (09-22, `workflow_dispatch`), #284 (09-24), #291 (09-27), #295 (09-29), #299 (10-01), and #314 (10-08) — third such night in the last nine days. AMD leg (self-hosted runner, unaffected by AWS capacity) passed cleanly against the large `public-main` → `internal-main` merge now on `main`; the CUDA legs still haven't exercised that merge due to AWS capacity. Recorded to `aws-outage-failures.md`.

### Run #317 — secondary nightly (schedule `17 4 * * *`, main @ `1d85185`, [37884546220](https://github.com/aqlaboratory/openfold-3/actions/runs/37884546220))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #316, per this log's established convention).

---

## 2026-10-10

### Run #318 — primary nightly (schedule `17 3 * * *`, main @ `3b9ec5d` — merge PR #440 "Template preprocessing entry point" (new since run #316/317/10-09's `1d85185`), [38021091242](https://github.com/aqlaboratory/openfold-3/actions/runs/38021091242))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 39 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 35 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 32 min | |

**2026-10-10: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Clean recovery after three AWS `g5.4xlarge` capacity-exhaustion nights in the preceding nine days (10-01 run #299, 10-08 run #314, 10-09 run #316) — both CUDA legs' `start-aws-runner` launched on pool 1 in `us-east-2` without needing the `us-west-2` fallback (`Create cloud runner (us-west-2)` skipped on both legs), and `Cache download of parameters` hit cleanly on both. First nightly run since main advanced from `1d85185` to `3b9ec5d` (PR #440). This is also the first CUDA-leg exercise of the large `public-main` → `internal-main` merge that landed on `1d85185` (run #316/317, 10-09) — AWS capacity exhaustion had kept the CUDA legs from testing that merge until tonight; AMD passed against it cleanly both nights. No regression observed from either the merge or PR #440.

### Run #319 — secondary nightly (schedule `17 4 * * *`, main @ `3b9ec5d`, [38024412493](https://github.com/aqlaboratory/openfold-3/actions/runs/38024412493))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #318, per this log's established convention).

---

## 2026-10-11

### Run #320 — primary nightly (schedule `17 3 * * *`, main @ `3b9ec5d`, [38109056932](https://github.com/aqlaboratory/openfold-3/actions/runs/38109056932))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 33 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 39 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 35 min | |

**2026-10-11: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Second consecutive clean night on the same `main` commit (`3b9ec5d`, PR #440) as run #318/#319 (10-10) — no new merges landed since yesterday. Both CUDA legs' `start-aws-runner` launched on pool 1 in `us-east-2` without needing the `us-west-2` fallback again (`Create cloud runner (us-west-2)` skipped on both legs), and `Cache download of parameters` hit cleanly on both — continuing the recovery from the AWS `g5.4xlarge` capacity-exhaustion nights earlier in the month.

### Run #321 — secondary nightly (schedule `17 4 * * *`, main @ `3b9ec5d`, [38111960976](https://github.com/aqlaboratory/openfold-3/actions/runs/38111960976))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #320, per this log's established convention).

---

## 2026-10-12

### Run #322 — primary nightly (schedule `17 3 * * *`, main @ `2496c32` — "Remove unnecessary script __init__ files and rdkit pin" (#446), carrying also #443 "docs: document Pixi development setup for contributors" (new since run #320/321's `3b9ec5d`), [38200964326](https://github.com/aqlaboratory/openfold-3/actions/runs/38200964326))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 44 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 42 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 37 min | |

**2026-10-12: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Third consecutive clean night. Both CUDA legs' `start-aws-runner` launched on pool 1 in `us-east-2` without needing the `us-west-2` fallback (`Create cloud runner (us-west-2)` skipped on both legs), and `Cache download of parameters` hit cleanly on both. Main advanced from `3b9ec5d` to `2496c32` (PR #443 docs-only, PR #446 removes unused `__init__.py` files and the rdkit pin) — no regression observed from either change.

### Run #323 — secondary nightly (schedule `17 4 * * *`, main @ `2496c32`, [38204842934](https://github.com/aqlaboratory/openfold-3/actions/runs/38204842934))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #322, per this log's established convention).

---

## 2026-10-13

### Run #324 — primary nightly (schedule `17 3 * * *`, main @ `0c82648` — "Merge branch 'consortium-main' into public-main", carrying PR #471 "fix: issue #340 wrong MSAs loaded" (new since run #322/323's `2496c32`), [38342766977](https://github.com/aqlaboratory/openfold-3/actions/runs/38342766977))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 40 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 36 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 33 min | |

**2026-10-13: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Fourth consecutive clean night. Both CUDA legs' `start-aws-runner` launched on pool 1 in `us-east-2` without needing the `us-west-2` fallback (`Create cloud runner (us-west-2)` skipped on both legs), and `Cache download of parameters` hit cleanly on both. Main advanced from `2496c32` to `0c82648` (PR #471, MSA-loading bugfix) — no regression observed.

### Run #325 — secondary nightly (schedule `17 4 * * *`, main @ `0c82648`, [38347394537](https://github.com/aqlaboratory/openfold-3/actions/runs/38347394537))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #324, per this log's established convention).

---

## 2026-10-14

### Run #326 — primary nightly (schedule `17 3 * * *`, main @ `8b0244c` — merge PR #448 "Workflow for nightly CI on Brev runners" (new since run #324/325's `0c82648`); adds a `test-pixi-cuda-brev` job running the same cuda12/cuda13 matrix on Brev-provisioned runners alongside the existing AWS-provisioned `test-pixi-cuda` — not one of the three tracked jobs, [38506642028](https://github.com/aqlaboratory/openfold-3/actions/runs/38506642028))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 37 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 38 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 36 min | |

**2026-10-14: 3/3 passed · 0 skipped · 0 queued · 0 need attention**

Fifth consecutive clean night. Both CUDA legs' `start-aws-runner` launched on pool 1 in `us-east-2` without needing the `us-west-2` fallback (`Create cloud runner (us-west-2)` skipped on both legs), and `Cache download of parameters` hit cleanly on both. Main advanced from `0c82648` to `8b0244c` (PR #448, adds parallel Brev-runner nightly jobs) — no regression observed on the tracked AWS/self-hosted jobs. The new `test-pixi-cuda-brev (openfold3-cuda12)` and `(openfold3-cuda13)` jobs also passed tonight (~40 min and ~39 min respectively) but sit outside this log's tracked-three-jobs scope and are not included in the coverage line.

### Run #327 — secondary nightly (schedule `17 4 * * *`, main @ `8b0244c`, [38511802779](https://github.com/aqlaboratory/openfold-3/actions/runs/38511802779))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event.schedule == vars.NIGHTLY_CRON` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #326, per this log's established convention).

---

## 2026-10-14 — scope update (same-day addendum)

The monitoring brief now tracks five jobs per night instead of three, effective immediately: `test-pixi-cuda (openfold3-cuda12)`, `test-pixi-cuda (openfold3-cuda13)`, `test-pixi-cuda-brev (openfold3-cuda12)`, `test-pixi-cuda-brev (openfold3-cuda13)`, and `test-pixi-amd (openfold3-rocm7)`. The `test-pixi-cuda-brev` legs were excluded from the coverage line above because they were new that same night (run #326) and out of scope under the prior three-job convention; re-stated under the new five-job scope, using the same run #326 data already recorded above:

| Job | State | Duration |
|-----|-------|----------|
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 37 min |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 35 min |
| test-pixi-cuda-brev (openfold3-cuda12) | **PASSED** | 38 min |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 35 min |
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 37 min |

**2026-10-14: 5/5 passed · 0 skipped · 0 queued · 0 need attention**

No new workflow run occurred in the scan window beyond run #326/#327, already logged above — this addendum only restates that same night's result under the expanded scope so the five-job convention starts cleanly from 2026-10-14 onward. Brev vs. AWS agreement so far: 2/2 nights (10-14 only, first night tracked) — both legs passing on both providers every run tracked to date. Per the monitoring brief, a two-week brev-vs-AWS pass-rate comparison is due once two weeks of data have accumulated (around 2026-10-28); too early to report today.

---

## 2026-10-15

### Run #328 — primary nightly (schedule `17 3 * * *`, main @ `a5867f8` — merge PR #451 "expect-freed" (AMD-Ecosystem) and PR #445 "cyclic-relpos-ring-fix" (new since run #326/327's `8b0244c`), [38676626674](https://github.com/aqlaboratory/openfold-3/actions/runs/38676626674))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 37 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 35 min | |
| test-pixi-cuda-brev (openfold3-cuda12) | **PASSED** | 49 min | |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 39 min | |
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 41 min | |

**2026-10-15: 5/5 passed · 0 skipped · 0 queued · 0 need attention**

Second consecutive clean night under the five-job scope (first: 2026-10-14, run #326). Both CUDA legs' `start-aws-runner` and both Brev legs' `start-brev-runner` completed without incident; `Cache download of parameters` hit cleanly on all four CUDA/Brev legs. Main advanced from `8b0244c` to `a5867f8` (PRs #451, #445) — no regression observed on any tracked job.

### Run #329 — secondary nightly (schedule `17 4 * * *`, main @ `a5867f8`, [38681684175](https://github.com/aqlaboratory/openfold-3/actions/runs/38681684175))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda-brev | **SKIPPED** | — | same guard; job named plain `test-pixi-cuda-brev`, confirming it never expanded |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #328, per this log's established convention).

Brev vs. AWS agreement so far: 2/2 nights tracked (10-14, 10-15) — both legs passing on both providers every run tracked to date under the five-job scope. Per the monitoring brief, a two-week brev-vs-AWS pass-rate comparison is due once two weeks of data have accumulated (brev jobs added 2026-10-14 in PR #448; due ~2026-10-28); too early to report today.

---

## 2026-10-16

### Run #330 — primary nightly (schedule `17 3 * * *`, main @ `32c9ce3` — merge PR #355 "fix: use bin centers, not left edges, for pLDDT/PAE/PDE classification targets" (new since run #328/329's `a5867f8`), [38845884578](https://github.com/aqlaboratory/openfold-3/actions/runs/38845884578))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 39 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 40 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 36 min | |
| test-pixi-cuda-brev (openfold3-cuda12) | **PASSED** | 39 min | |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 37 min | |

**2026-10-16: 5/5 passed · 0 skipped · 0 queued · 0 need attention**

Third consecutive clean night under the five-job scope (following 10-14 run #326 and 10-15 run #328). Both AWS CUDA legs' `start-aws-runner` launched on pool 1 in `us-east-2` without needing the `us-west-2` fallback (`Create cloud runner (us-west-2)` skipped on both legs), both Brev legs' `start-brev-runner` completed cleanly, and `Cache download of parameters` hit on all four CUDA/Brev legs. Main advanced from `a5867f8` to `32c9ce3` (PR #355, fixes a systematic half-bin-width miscalibration in the pLDDT/PAE/PDE confidence losses) — no regression observed on any tracked job.

### Run #331 — secondary nightly (schedule `17 4 * * *`, main @ `32c9ce3`, [38850844753](https://github.com/aqlaboratory/openfold-3/actions/runs/38850844753))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda-brev | **SKIPPED** | — | same guard; job named plain `test-pixi-cuda-brev`, confirming it never expanded |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #330, per this log's established convention).

Brev vs. AWS agreement so far: 3/3 nights tracked (10-14, 10-15, 10-16) — both legs passing on both providers every run tracked to date under the five-job scope. Per the monitoring brief, a two-week brev-vs-AWS pass-rate comparison is due once two weeks of data have accumulated (brev jobs added 2026-10-14 in PR #448; due ~2026-10-28); still too early to report today (2 days in).

---

## 2026-10-17

### Run #332 — primary nightly (schedule `17 3 * * *`, main @ `0d25195` — merge PR #457 "triton-precision" (AMD-Ecosystem) (new since run #330/331's `32c9ce3`), [38998919417](https://github.com/aqlaboratory/openfold-3/actions/runs/38998919417))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 36 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 33 min | |
| test-pixi-cuda-brev (openfold3-cuda12) | **PASSED** | 35 min | |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 33 min | |
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 41 min | |

**2026-10-17: 5/5 passed · 0 skipped · 0 queued · 0 need attention**

Fourth consecutive clean night under the five-job scope (following 10-14 run #326, 10-15 run #328, 10-16 run #330). Both AWS CUDA legs' `start-aws-runner` launched on pool 1 in `us-east-2` without needing the `us-west-2` fallback (`Create cloud runner (us-west-2)` skipped on both legs), both Brev legs' `start-brev-runner` completed cleanly, and `Cache download of parameters` hit on all four CUDA/Brev legs (16–21s each). Main advanced from `32c9ce3` to `0d25195` (PR #457, AMD-Ecosystem fp32 matmul precision fix for `tl.dot`) — no regression observed on any tracked job.

### Run #333 — secondary nightly (schedule `17 4 * * *`, main @ `0d25195`, [39002647337](https://github.com/aqlaboratory/openfold-3/actions/runs/39002647337))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda-brev | **SKIPPED** | — | same guard; job named plain `test-pixi-cuda-brev`, confirming it never expanded |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #332, per this log's established convention).

Brev vs. AWS agreement so far: 4/4 nights tracked (10-14, 10-15, 10-16, 10-17) — both legs passing on both providers every run tracked to date under the five-job scope. Per the monitoring brief, a two-week brev-vs-AWS pass-rate comparison is due once two weeks of data have accumulated (brev jobs added 2026-10-14 in PR #448; due ~2026-10-28); still too early to report today (4 days in).

---

## 2026-10-18

### Run #334 — primary nightly (schedule `17 3 * * *`, main @ `0d25195` — same commit as run #332/333 (10-17); no new merges to main since yesterday's primary nightly, [39093832772](https://github.com/aqlaboratory/openfold-3/actions/runs/39093832772))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 50 min | |
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 56 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 47 min | |
| test-pixi-cuda-brev (openfold3-cuda12) | **PASSED** | 56 min | |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 49 min | |

**2026-10-18: 5/5 passed · 0 skipped · 0 queued · 0 need attention**

Fifth consecutive clean night under the five-job scope (following 10-14 run #326, 10-15 run #328, 10-16 run #330, 10-17 run #332). Both AWS CUDA legs' `start-aws-runner` launched on pool 1 in `us-east-2` without needing the `us-west-2` fallback (`Create cloud runner (us-west-2)` skipped on both legs), both Brev legs' `start-brev-runner` completed cleanly, and `Cache download of parameters` hit on all four CUDA/Brev legs. Main stayed at `0d25195` (PR #457) — no new commit exercised tonight.

### Run #335 — secondary nightly (schedule `17 4 * * *`, main @ `0d25195`, [39097187219](https://github.com/aqlaboratory/openfold-3/actions/runs/39097187219))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda-brev | **SKIPPED** | — | same guard; job named plain `test-pixi-cuda-brev`, confirming it never expanded |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #334, per this log's established convention).

Brev vs. AWS agreement so far: 5/5 nights tracked (10-14 through 10-18) — both legs passing on both providers every run tracked to date under the five-job scope. Two-week mark (brev jobs added 2026-10-14 in PR #448) falls on 2026-10-28 — 4 days tracked so far this stretch toward it; still too early to report the comparison today.

---

## 2026-10-19

### Run #336 — primary nightly (schedule `17 3 * * *`, main @ `0d25195` — same commit as run #334/335 (10-18); no new merges to main since yesterday's primary nightly, [39195251631](https://github.com/aqlaboratory/openfold-3/actions/runs/39195251631))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 49 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 46 min | |
| test-pixi-cuda-brev (openfold3-cuda12) | **PASSED** | 51 min | |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 46 min | |
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 36 min | |

**2026-10-19: 5/5 passed · 0 skipped · 0 queued · 0 need attention**

Sixth consecutive clean night under the five-job scope (following 10-14 run #326, 10-15 run #328, 10-16 run #330, 10-17 run #332, 10-18 run #334). Both AWS CUDA legs' `start-aws-runner` launched on pool 1 in `us-east-2` without needing the `us-west-2` fallback (`Create cloud runner (us-west-2)` skipped on both legs), both Brev legs' `start-brev-runner` completed cleanly, and `Cache download of parameters` hit on all four CUDA/Brev legs. Main stayed at `0d25195` (PR #457) — no new commit exercised tonight. Note: this session's `gh` CLI had an invalid `GH_TOKEN` (unauthenticated), so job/run data for tonight's scan was fetched via the GitHub MCP server's Actions tools instead of `gh run list`/`gh run view`/`gh api`.

### Run #337 — secondary nightly (schedule `17 4 * * *`, main @ `0d25195`, [39199928059](https://github.com/aqlaboratory/openfold-3/actions/runs/39199928059))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda-brev | **SKIPPED** | — | same guard; job named plain `test-pixi-cuda-brev`, confirming it never expanded |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #336, per this log's established convention).

Brev vs. AWS agreement so far: 6/6 nights tracked (10-14 through 10-19) — both legs passing on both providers every run tracked to date under the five-job scope. Two-week mark (brev jobs added 2026-10-14 in PR #448) falls on 2026-10-28 — 6 days tracked so far this stretch toward it; still too early to report the comparison today (8 days remain).

---

## 2026-10-20

### Run #338 — primary nightly (schedule `17 3 * * *`, main @ `7bebb7a` — merge PR #459 "fix: configurable cluster environment; stop SLURM autodetection hangs" (new since run #336/337's `0d25195`), [39364702695](https://github.com/aqlaboratory/openfold-3/actions/runs/39364702695))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 40 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 36 min | |
| test-pixi-cuda-brev (openfold3-cuda12) | **FAILED** | 0 min | `unclassified` — `start-brev-runner` / "Create Brev runner" step: `Skipping: instance type "g5.4xlarge" is not a recognized type; run 'brev search' to see available types` → `Warning: Only created 0/1 instances` → `subprocess.CalledProcessError: Command '[... 'brev', 'create', 'runner-a5ccc9cw', '--type', 'g5.4xlarge', ...]' returned non-zero exit status 1.` |
| test-pixi-cuda-brev (openfold3-cuda13) | **FAILED** | 0 min | `unclassified` — same signature: `Skipping: instance type "g5.4xlarge" is not a recognized type; run 'brev search' to see available types` → `Warning: Only created 0/1 instances` → `subprocess.CalledProcessError: Command '[... 'brev', 'create', 'runner-mkn0lc8h', '--type', 'g5.4xlarge', ...]' returned non-zero exit status 1.` |
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 52 min | |

**2026-10-20: 3/5 passed · 0 skipped · 0 queued · 2 need attention**

First non-pass for either Brev leg since their introduction on 2026-10-14 (run #326), breaking a 6-consecutive-night clean streak. Both Brev legs failed identically and immediately (under 15s) at the `start-brev-runner` / "Create Brev runner" step: Brev's own CLI (`brev create ... --type g5.4xlarge`) rejected `g5.4xlarge` as "not a recognized type," i.e. Brev's instance-type catalog no longer offers (or has renamed) the type this workflow requests — not a capacity exhaustion (no `InsufficientInstanceCapacity`-style message), not an AWS issue at all (the two AWS `test-pixi-cuda` legs provisioned and ran normally on pool 1 in `us-east-2`), and not a GPU-busy or runner-offline condition either. Doesn't match any of this log's six defined signatures (`aws-capacity`, `gpu-unavailable`, `runner-offline`, `msa-hang`, `parameter-cache`, `build-push`, `code`), so recorded as `unclassified` per this log's explicit fallback rule rather than guessed at. See `test-failures.md` for full detail. **Flagged for a human:** confirm whether Brev deprecated/renamed the `g5.4xlarge` SKU (run `brev search` against the `OMSF-Brev` org) and update the `brev_instance_type` input in `.github/workflows/ci-integration-test-pixi-cuda-brev-reusable.yml` accordingly; until fixed, the Brev legs will fail every night regardless of code changes. AMD leg and both AWS CUDA legs unaffected.

### Run #339 — secondary nightly (schedule `17 4 * * *`, main @ `7bebb7a`, [39369835149](https://github.com/aqlaboratory/openfold-3/actions/runs/39369835149))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda-brev | **SKIPPED** | — | same guard; job named plain `test-pixi-cuda-brev`, confirming it never expanded |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #338, per this log's established convention).

Brev vs. AWS agreement: 6/7 nights fully agreeing (10-14 through 10-19); tonight (10-20) is the first divergence — both Brev legs failed (`unclassified`, Brev-side instance-type rejection, see above) while both AWS `test-pixi-cuda` legs passed. Running Brev leg-tally toward the two-week comparison (brev jobs added 2026-10-14 in PR #448; two-week mark 2026-10-28): 12/14 brev job-runs passed to date (6 nights × 2 legs clean, tonight's 2 legs failed) = 85.7% so far, with 8 days of the two-week window remaining — too early to apply the brief's 95%-pass-rate replace-AWS threshold, and today's result is a reminder the comparison should exclude/annotate infra-only non-passes (an unrelated Brev catalog issue) rather than count them against Brev's reliability at face value when the formal report is due.

---

## 2026-10-21

### Run #340 — primary nightly (schedule `17 3 * * *`, main @ `b2b9f93` — merge PR #467 "chore(deps): bump prefix-dev/setup-pixi" (dependabot, new since run #338/339's `7bebb7a`), [39539067117](https://github.com/aqlaboratory/openfold-3/actions/runs/39539067117))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 20 min | code — `FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin] - requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.` (6 retries exhausted) |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 16 min | code — same signature: `FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin] - requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.` |
| test-pixi-amd (openfold3-rocm7) | **FAILED** | 14 min | code — same signature: `FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin] - requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.` |
| test-pixi-cuda-brev (openfold3-cuda12) | **FAILED** | 0 min | unclassified — `start-brev-runner` / "Create Brev runner" step, same signature as run #338 (10-20), second consecutive night: `Skipping: instance type "g5.4xlarge" is not a recognized type; run 'brev search' to see available types` → `subprocess.CalledProcessError: ... 'brev', 'create', ..., '--type', 'g5.4xlarge', ...` |
| test-pixi-cuda-brev (openfold3-cuda13) | **FAILED** | 0 min | unclassified — identical signature, second consecutive night |

**2026-10-21: 0/5 passed · 0 skipped · 0 queued · 5 need attention**

First night in this log where all five tracked jobs are simultaneously non-passing (previously the worst was 0/3 under the three-job scope, on 09-17 and 09-19). Two unrelated causes converged: (1) `api.colabfold.com` (the ColabFold MSA-submission endpoint) read-timed-out on all three non-Brev legs' `msa-no_templates-ubiquitin` inference test, each after 6 retries — the same recurring external-service signature seen on 08-09, 08-11, 08-12, 09-08, and 09-19, classified `code` per this log's rules even though the trigger is an external API outage, not an OF3 regression; (2) both Brev legs failed a second consecutive night at `start-brev-runner` with the identical "instance type `g5.4xlarge` is not a recognized type" error first seen on 10-20 (run #338) — still `unclassified`, still unresolved, still flagged for a human to fix the `brev_instance_type` input or confirm the SKU rename with Brev. Both AWS CUDA legs' `start-aws-runner` succeeded on pool 1 in `us-east-2` with no `aws-capacity` signature — tonight's CUDA failures are purely the ColabFold `code` issue, independent of the AWS-capacity history earlier this month. See `test-failures.md` for full detail.

### Run #341 — secondary nightly (schedule `17 4 * * *`, main @ `b2b9f93`, [39544412270](https://github.com/aqlaboratory/openfold-3/actions/runs/39544412270))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda-brev | **SKIPPED** | — | same guard; job named plain `test-pixi-cuda-brev`, confirming it never expanded |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #340, per this log's established convention).

Brev vs. AWS agreement: second consecutive divergence (10-20, 10-21) — both Brev legs failed on `unclassified` instance-type rejection both nights while the AWS `test-pixi-cuda` legs ran clean both nights (tonight's AWS legs failing only on the unrelated `code`/ColabFold issue that also hit the AMD leg). Running Brev leg-tally toward the two-week comparison (brev jobs added 2026-10-14 in PR #448; two-week mark 2026-10-28): 12/16 brev job-runs passed to date (6 clean nights × 2 legs, then 2 consecutive nights of 2 failures each) = 75.0% so far, with 7 days of the two-week window remaining. The brief's 95%-pass-rate bar for recommending AWS-workflow retirement looks increasingly unlikely to be met by 10-28 unless the Brev-side `g5.4xlarge` catalog issue is fixed and the legs return to a clean streak soon — but since both failures to date trace to a Brev-catalog/config problem rather than underlying instance reliability, the final report should still weigh that distinction rather than count it at face value, per the 10-20 entry's note.

---

## 2026-10-22

### Run #342 — primary nightly (schedule `17 3 * * *`, main @ `e242b404` — merge PR #469 "adds g6.4xlarge to allowed instance types for brev integration tests" (new since run #340/341's `b2b9f93`), [39717308935](https://github.com/aqlaboratory/openfold-3/actions/runs/39717308935))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **CANCELLED** | 40 min | matrix fail-fast — sibling (cuda13) failed at 04:07:05Z; this leg was cancelled mid-test (`##[error]The operation was canceled.`) at 04:14:11Z, ~7 min later (consistent with this log's historical fail-fast propagation delay, e.g. runs #268, #273), 1m54s after its own last completed test (`test_cuda_allocator.py::test_matches_snapshot PASSED` at 04:12:17Z) — not a timeout-minutes cap (only ~41 of 60 allotted minutes elapsed when cancelled), not superseded by a newer run (no other run in this concurrency group tonight — confirmed via `list_workflow_runs` filtered to `workflow_dispatch`/`main`), not manual |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 33 min | code — `FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin] - requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.` (6 retries exhausted) |
| test-pixi-cuda-brev (openfold3-cuda12) | **CANCELLED** | 60 min | hit the `timeout-minutes: 60` cap — job ran 03:37:43Z–04:38:06Z (60m23s). Last completed test was `test_cuda_allocator.py::test_matches_snapshot PASSED` at 04:29:56Z, then an ~8-minute silent gap (no further pytest nodeid printed at all) before `##[error]The operation was canceled.` at 04:37:59Z. This resembles this log's `msa-hang` pattern (cancelled at the cap, long gap after the CUDA-allocator tests which run just before the MSA-dependent inference tests in file order) but does not literally match the defined signature — the required line (a `test_inference_writes_outputs[msa-...]` nodeid left hanging with no PASSED/FAILED, or `TimeoutError: MSA server`) never appears, because whatever test started next never got far enough to print its own nodeid before the cap killed the job. Recording as `unclassified` per this log's explicit "don't guess" rule rather than assuming `msa-hang`; flagged for a human to confirm test collection order and whether an MSA-dependent test was in fact running when the cap hit |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 59 min | |
| test-pixi-amd (openfold3-rocm7) | **FAILED** | 23 min | code — `FAILED openfold3/tests/inference/test_pocket_constraints.py::test_pocket_constraint_localizes_ligand - requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.` |

**2026-10-22: 1/5 passed · 0 skipped · 0 queued · 4 need attention**

First night since the 10-20/10-21 Brev `g5.4xlarge`-rejection streak where both Brev legs successfully provisioned a runner: main advanced from `b2b9f93` to `e242b404` (PR #469, adds `g6.4xlarge` as a fallback type in `.github/workflows/ci-integration-test-pixi-cuda-brev-reusable.yml`'s `brev_instance_type` input, now `g5.4xlarge,g6.4xlarge`), and neither Brev leg hit the "not a recognized type" error tonight — the fix flagged on 10-20 and 10-21 appears effective. `test-pixi-cuda-brev (openfold3-cuda13)` passed clean in 59 of 60 allotted minutes; `test-pixi-cuda-brev (openfold3-cuda12)` instead ran the full 60-minute job timeout and was killed by the cap (see table note on the near-miss `msa-hang` match) — recorded `unclassified` and flagged for a human. Separately, `test-pixi-cuda (openfold3-cuda13)` hit the recurring `api.colabfold.com` MSA-submission read-timeout (same signature as 09-19, 10-21, and others — `code` per this log's rules even though the trigger is an external API, not an OF3 regression), and `test-pixi-cuda (openfold3-cuda12)` was taken out by GitHub's default matrix fail-fast once its cuda13 sibling failed (`test-pixi-cuda`'s `strategy:` block carries no `fail-fast: false` override) — the same mechanism as runs #254, #266, #268, #273, and #279. `test-pixi-amd (openfold3-rocm7)` failed on the identical ColabFold signature but on a different test (`test_pocket_constraint_localizes_ligand` rather than `test_inference_writes_outputs[msa-no_templates-ubiquitin]`), suggesting `api.colabfold.com` was broadly degraded tonight rather than one specific query being slow. Both AWS CUDA legs' `start-aws-runner` launched on pool 1 in `us-east-2` with no `aws-capacity` signature. See `test-failures.md` for full detail.

### Run #343 — secondary nightly (schedule `17 4 * * *`, main @ `e242b404`, [39722672379](https://github.com/aqlaboratory/openfold-3/actions/runs/39722672379))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda-brev | **SKIPPED** | — | same guard; job named plain `test-pixi-cuda-brev`, confirming it never expanded |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #342, per this log's established convention).

Brev vs. AWS agreement: the 10-20/10-21 divergence (both Brev legs failing on the `g5.4xlarge` catalog issue) appears resolved by PR #469 — tonight one of two Brev legs passed clean and the other ran to its own 60-minute cap (not a repeat of the instance-type rejection). Running Brev leg-tally toward the two-week comparison (brev jobs added 2026-10-14 in PR #448; two-week mark 2026-10-28): 13/18 brev job-runs passed to date (6 clean nights × 2 legs = 12, then 2 nights × 2 legs failed = 0, tonight 1 of 2 legs passed) = 72.2%, with 6 days of the two-week window remaining. Still too early to apply the monitoring brief's 95%-pass-rate replace-AWS threshold, and — per the 10-20/10-21 notes' precedent for infra-only non-passes — tonight's cap-kill on `test-pixi-cuda-brev (openfold3-cuda12)` should be investigated before the 10-28 report counts it at face value against Brev's reliability, since it's not yet clear whether it reflects genuine Brev-instance slowness/variance or something else (e.g. the `msa-hang`-like gap noted above).

---

## 2026-10-23

### Run #345 — primary nightly (schedule `17 3 * * *`, main @ `a83cb11` — merge PR #465 "fix/biotite-2-unpin" (Jennifer Wei, new since run #342/343's `e242b404`), [39895507300](https://github.com/aqlaboratory/openfold-3/actions/runs/39895507300))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 2 min | aws-capacity — `start-aws-runner`: `Failed to launch in us-east-2a/b/c: ... InsufficientInstanceCapacity ... g5.4xlarge` then `us-west-2a/b/c` identically, plus the standing `us-west-2d` `Unsupported`-AZ rejection → `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 3 min | aws-capacity — identical signature: `Failed to launch in us-east-2a/b/c` then `us-west-2a/b/c`, all `InsufficientInstanceCapacity` for `g5.4xlarge` (plus `us-west-2d` `Unsupported`) → `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |
| test-pixi-cuda-brev (openfold3-cuda12) | **PASSED** | 41 min | |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 39 min | |
| test-pixi-amd (openfold3-rocm7) | **QUEUED** | — (3h40m+ as of scan) | started (job `120224903978`) 2026-10-23T03:31:52Z with zero steps recorded; still queued at scan time (07:12Z), past this log's 2h `QUEUED` threshold. Required runner label: `[self-hosted, amd-gpu]` (`.github/workflows/ci-integration-test-pixi-amd-reusable.yml`). Candidate `runner-offline`, unconfirmed — `GET /repos/aqlaboratory/openfold-3/actions/runners` returns `403 Access to this GitHub Actions path is not permitted through this proxy` in this execution environment, so the runner's live online/offline state could not be checked directly (same limitation flagged in `weekly-summary.md`'s data-source notes) |

**2026-10-23: 2/5 passed · 0 skipped · 1 queued · 2 need attention**

First `aws-capacity` night since 2026-10-09 (run #316) — both AWS `test-pixi-cuda` legs failed identically on `g5.4xlarge` `InsufficientInstanceCapacity` across all three `us-east-2` AZs, then all three `us-west-2` AZs, plus the standing `us-west-2d` `Unsupported`-AZ rejection, via the same `start-aws-runner` → `stop-aws-runner` cascade seen in every prior `aws-capacity` entry in this log (09-09, 09-10, 09-24, 09-27, 09-29, 10-01, 10-08, 10-09); full detail in `aws-outage-failures.md`. Both Brev legs ran clean on the same commit (`a83cb11`, PR #465 "biotite 2 unpin," new since 10-22's `e242b404`) in 39–41 minutes — since the AWS-capacity outage that took out both AWS CUDA legs didn't touch Brev's underlying provider/region, tonight's split (2/2 AWS failed, 2/2 Brev passed) is further evidence Brev isn't sharing AWS's `g5.4xlarge` capacity pool. Most notably, `test-pixi-amd (openfold3-rocm7)` never got a runner at all: it has sat `QUEUED` since 03:31:52Z with no steps ever recorded, and is still queued as of this scan (07:12Z, 3h40m+) — well past this log's 2h `QUEUED` threshold. This is a candidate `runner-offline`, but it's unconfirmed: the `/actions/runners` endpoint needed to check the self-hosted `amd-gpu` runner's live status is blocked in this execution environment (`403`, same limitation noted in `weekly-summary.md`). **Flagged for a human:** check the self-hosted AMD runner's status directly under Settings → Actions → Runners and restart/re-register it if offline; also worth re-running `test-pixi-amd` manually once the runner is confirmed up, since tonight's result otherwise leaves this leg completely untested. Running Brev leg-tally toward the two-week comparison (brev jobs added 2026-10-14 in PR #448; two-week mark 2026-10-28, 5 days remaining): 15/20 brev job-runs passed to date (72.2% → 75.0% with tonight's 2/2 clean legs).

### Run #346 — secondary nightly (schedule `17 4 * * *`, main @ `a83cb11`, [39900676696](https://github.com/aqlaboratory/openfold-3/actions/runs/39900676696))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda-brev | **SKIPPED** | — | same guard; job named plain `test-pixi-cuda-brev`, confirming it never expanded |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #345, per this log's established convention).

---

## 2026-10-24

### Run #347 — primary nightly (schedule `17 3 * * *`, main @ `a83cb11` (unchanged since run #345/346's `a83cb11`, 10-23 — no new merges to main since yesterday's primary nightly), [40058531049](https://github.com/aqlaboratory/openfold-3/actions/runs/40058531049))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 2 min | aws-capacity — `start-aws-runner`: `Failed to launch in us-east-2a/b/c: ... InsufficientInstanceCapacity ... g5.4xlarge` then identically across `us-west-2a/b/c`, plus the standing `us-west-2d` `Unsupported`-AZ rejection → `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |
| test-pixi-cuda (openfold3-cuda13) | **CANCELLED** | 9 min | matrix fail-fast, tied to cuda12's aws-capacity failure above — cuda12's `start-aws-runner` failed at 03:31:45Z; cuda13's own `start-aws-runner` had actually just succeeded (launched `runner-o095hzqq` in `us-east-2` after 2 AZ retries, "Launched in us-east-2", all steps reporting `success`) but the job's own `conclusion` still reports `cancelled` once the sibling's failure propagated through the matrix; the downstream `test-openfold-docker-pixi` step was cancelled immediately at 03:32:25Z, before it ever started — not a timeout-minutes cap, not superseded by a newer run (no other run in this concurrency group tonight), not manual |
| test-pixi-cuda-brev (openfold3-cuda12) | **PASSED** | 50 min | |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 48 min | |
| test-pixi-amd (openfold3-rocm7) | **QUEUED** | — (3h42m+ as of scan) | started 2026-10-24T03:29:52Z with zero steps recorded; still queued at scan time (~07:12Z). Required runner label: `[self-hosted, amd-gpu]` (`.github/workflows/ci-integration-test-pixi-amd-reusable.yml`). Candidate `runner-offline`, unconfirmed — `GET /repos/aqlaboratory/openfold-3/actions/runners` still returns `403 Access to this GitHub Actions path is not permitted through this proxy` in this execution environment, same limitation as every prior occurrence in this log. **Second consecutive night with zero AMD coverage:** yesterday's (10-23) AMD job (job 120224903978) sat queued for the full 24 hours and was never picked up by a runner at all — it was only auto-cancelled at 2026-10-24T03:29:52Z, the instant tonight's run opened a new job in the same `cancel-in-progress: true` concurrency group. A runner that hasn't claimed a job in over 24 hours across two consecutive nightly runs is strong circumstantial evidence the self-hosted `amd-gpu` box is actually offline, even though the `/actions/runners` endpoint can't confirm it directly from here. |

**2026-10-24: 2/5 passed · 0 skipped · 1 queued · 2 need attention**

First `aws-capacity` night since run #345 (10-23, which itself broke a 14-night clean streak) — second consecutive night of AWS `g5.4xlarge` capacity exhaustion, though tonight only cuda12 hit it directly; cuda13 is the first case in this log where a leg's own `start-aws-runner` fully succeeded (provisioned an instance in `us-east-2` on the second AZ retry) only for the leg to still be marked `cancelled` once its sibling failed — see `test-failures.md` for full detail on the fail-fast mechanics. Both Brev legs ran clean on the same commit (`a83cb11`, unchanged since 10-23) in 48–50 minutes, same pattern as 10-23: the AWS capacity outage doesn't touch Brev's provider/region. The AMD runner situation is now the most pressing item in this log: `test-pixi-amd (openfold3-rocm7)` has gone two full nights (10-23, 10-24) without executing a single test — not a transient 2h blip but an apparently-stuck self-hosted runner. **Flagged for a human, now urgently:** check Settings → Actions → Runners for the `amd-gpu` self-hosted box directly and restart/re-register it — AMD coverage has been completely dark for over 24 hours. Running Brev leg-tally toward the two-week comparison (brev jobs added 2026-10-14 in PR #448; two-week mark 2026-10-28, 4 days remaining): 17/22 brev job-runs passed to date (75.0% → 77.3% with tonight's 2/2 clean legs).

### Run #348 — secondary nightly (schedule `17 4 * * *`, main @ `a83cb11`, [40062442244](https://github.com/aqlaboratory/openfold-3/actions/runs/40062442244))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda-brev | **SKIPPED** | — | same guard; job named plain `test-pixi-cuda-brev`, confirming it never expanded |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #347, per this log's established convention).

Brev vs. AWS agreement: AWS `test-pixi-cuda` failed/cancelled both legs tonight (one direct `aws-capacity`, one fail-fast cascade) while both Brev legs passed clean — same split as 10-23. Running Brev leg-tally toward the two-week comparison (brev jobs added 2026-10-14 in PR #448; two-week mark 2026-10-28, 4 days remaining): 17/22 brev job-runs passed to date (77.3%), up from 75.0% after 10-23. Still below the monitoring brief's 95%-pass-rate bar for recommending AWS-workflow retirement, and — per this log's running notes since 10-20 — the two non-passes behind that gap (10-20, 10-21) were a Brev-catalog/config issue (fixed by PR #469 on 10-22), not underlying instance unreliability, so the final 10-28 report should weigh that distinction rather than count it at face value.

## 2026-10-25

### Run #349 — primary nightly (schedule `17 3 * * *`, main @ `4597b944` — new merge since 10-24's `a83cb11` (PR #462, feature-dict-mem-refactor), [40158569484](https://github.com/aqlaboratory/openfold-3/actions/runs/40158569484))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 40 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 42 min | |
| test-pixi-cuda-brev (openfold3-cuda12) | **PASSED** | 46 min | |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 43 min | |
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 43 min | runner `omsf-amd-aupcloud` picked the job up immediately (queued 0 min) — recovered after sitting dark for the prior two nights (10-23, 10-24) |

**2026-10-25: 5/5 passed · 0 skipped · 0 queued · 0 need attention**

First fully clean night since 10-22, breaking the two-night `aws-capacity` streak (10-23, 10-24): both AWS CUDA legs launched and passed cleanly this time (no capacity exhaustion in `start-aws-runner`), and the AMD self-hosted runner — dark for the two prior nights per `aws-outage-failures.md`'s `runner-offline` entries — picked up tonight's job the instant it was queued, with no human intervention recorded in this log between 10-24's scan and tonight's run. No new entries required in `aws-outage-failures.md` or `test-failures.md` tonight. Running Brev leg-tally toward the two-week comparison (brev jobs added 2026-10-14 in PR #448; two-week mark 2026-10-28, 3 days remaining): 19/24 brev job-runs passed to date (79.2%), up from 77.3% after 10-24. Still below the monitoring brief's 95%-pass-rate bar for recommending AWS-workflow retirement; per this log's running notes since 10-20, the non-passes behind that gap predate the 10-22 Brev-catalog fix (PR #469) and 10-23/10-24's AWS-side (not Brev-side) capacity outage — the 10-28 report should weigh that distinction rather than count it at face value.

### Run #350 — secondary nightly (schedule `17 4 * * *`, main @ `4597b944`, [40162032591](https://github.com/aqlaboratory/openfold-3/actions/runs/40162032591))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda-brev | **SKIPPED** | — | same guard; job named plain `test-pixi-cuda-brev`, confirming it never expanded |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #349, per this log's established convention).

---

## 2026-10-26

### Run #351 — primary nightly (schedule `17 3 * * *`, main @ `f87904d` — merge PR #478 "Mark `test_shape_large_bf16_train` test as slow" (new since run #349/10-25's `4597b944`), [40264862911](https://github.com/aqlaboratory/openfold-3/actions/runs/40264862911))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 2 min | aws-capacity — `start-aws-runner`: `Failed to launch in us-east-2a/b/c: ... InsufficientInstanceCapacity ... g5.4xlarge` then identically across `us-west-2a/b/c`, plus the standing `us-west-2d` `Unsupported`-AZ rejection → `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 2 min | aws-capacity — identical signature, hit directly and independently of cuda12 (not a fail-fast cascade tonight — unlike 10-24, both legs' own `start-aws-runner` failed on their own merits): `Failed to launch in us-east-2a/b/c` then `us-west-2a/b/c`, all `InsufficientInstanceCapacity` for `g5.4xlarge` (plus `us-west-2d` `Unsupported`) → `##[error]No capacity for g5.4xlarge in any zone of us-east-2 or us-west-2` |
| test-pixi-cuda-brev (openfold3-cuda12) | **PASSED** | 42 min | |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 39 min | |
| test-pixi-amd (openfold3-rocm7) | **PASSED** | 40 min | runner `omsf-amd-aupcloud` picked the job up immediately (queued 0 min) — second consecutive clean night (10-25, 10-26) after the 10-23/10-24 dark streak |

**2026-10-26: 3/5 passed · 0 skipped · 0 queued · 2 need attention**

Third `aws-capacity` night in the last four (10-23, 10-24, 10-26) — tonight both AWS CUDA legs hit `g5.4xlarge` `InsufficientInstanceCapacity` directly and independently (unlike 10-24, where only cuda12 hit it directly and cuda13 was taken out by fail-fast after its own runner had already launched); full detail in `aws-outage-failures.md`. Both `stop-aws-runner` steps failed as the same cascading consequence seen in every prior occurrence of this signature (`TypeError: StopAWS.__init__() missing 1 required positional argument: 'instance_mapping'` — no instance was ever created to tear down). Both Brev legs ran clean on the new commit (`f87904d`, PR #478 — a test-marker-only change, no functional impact) in 39–42 minutes, same pattern as every prior `aws-capacity` night: the outage doesn't touch Brev's provider/region. The AMD self-hosted runner picked up tonight's job immediately, its second consecutive clean night after the 10-23/10-24 two-night dark streak. No new entry required in `test-failures.md` tonight (both non-passes are `aws-capacity`, scoped to `aws-outage-failures.md`). Running Brev leg-tally toward the two-week comparison (brev jobs added 2026-10-14 in PR #448; two-week mark 2026-10-28, 2 days remaining): 21/26 brev job-runs passed to date (80.8%), up from 79.2% after 10-25. Still below the monitoring brief's 95%-pass-rate bar for recommending AWS-workflow retirement; per this log's running notes since 10-20, the two non-passes behind most of that gap (10-20, 10-21, 4 job-runs) predate the 10-22 Brev-catalog fix (PR #469), so the 10-28 report should weigh that distinction rather than count it at face value — excluding those pre-fix job-runs, Brev is 21/22 (95.5%) since the fix.

### Run #352 — secondary nightly (schedule `17 4 * * *`, main @ `f87904d`, [40269664282](https://github.com/aqlaboratory/openfold-3/actions/runs/40269664282))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda-brev | **SKIPPED** | — | same guard; job named plain `test-pixi-cuda-brev`, confirming it never expanded |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #351, per this log's established convention).

---

## 2026-10-27

### Run #353 — primary nightly (schedule `17 3 * * *`, main @ `244136c8` — merge PR #346 "test: add msa_mask/padding tests for create_msa_feature_precursor_of3" (new since run #351/10-26's `f87904d`), [40444773458](https://github.com/aqlaboratory/openfold-3/actions/runs/40444773458))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **PASSED** | 52 min | |
| test-pixi-cuda (openfold3-cuda13) | **PASSED** | 46 min | |
| test-pixi-cuda-brev (openfold3-cuda12) | **PASSED** | 52 min | |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 45 min | |
| test-pixi-amd (openfold3-rocm7) | **CANCELLED** | 61 min | superseded by a newer run (concurrency) — not the `timeout-minutes: 60` cap (the "Run integration test" step ran 03:45:24–04:33:17, 47m53s, under the cap) and not a manual cancellation. `##[error]The operation was canceled.` fired at 04:33:17Z, ~3 minutes after tonight's secondary-slot run (#354, created 04:30:14Z) opened its own (`if:`-guard-skipped) `test-pixi-amd` job entry in the identical `${{ github.workflow }}-pixi-amd-${{ github.ref }}-openfold3-rocm7` `cancel-in-progress: true` concurrency group — same mechanism flagged on 10-23/10-24 for a *queued* AMD job cancelled the instant the next run opened the group, but this is the first time in this log that mechanism has cancelled an AMD job that was actively mid-test rather than sitting queued. The last test before cancellation, `test_shape_large_eval[dtype=torch.float32]`, had already `PASSED` at 04:31:54Z; the next test (`of3_all_atom`) had only just started when the cancellation hit 1m23s later — no `msa-...` test node ID or `TimeoutError: MSA server` string anywhere near the cut, so this does not match this log's `msa-hang` signature. Recorded `unclassified` per this log's rule (none of the seven defined signatures fit); see `test-failures.md` for full detail. |

**2026-10-27: 4/5 passed · 0 skipped · 0 queued · 1 need attention**

Both AWS `test-pixi-cuda` legs and both Brev `test-pixi-cuda-brev` legs passed clean tonight — no `aws-capacity` recurrence after three such nights in the prior four (10-23, 10-24, 10-26). The sole non-pass, `test-pixi-amd`, is a new variant of the secondary-slot concurrency-cancellation pattern first seen 10-23/10-24: there the stale AMD job was sitting `queued` (never started) when the next run's same-named job entry cancelled it; tonight the AMD job had actually started, run most of the suite, and was mid-test when the identical cancellation mechanism fired. Worth a human's attention on two fronts: (1) whether `test-pixi-amd`'s concurrency group should exclude the (always-skipped) secondary schedule slot entirely, since it now has a second, more disruptive failure mode attributable to it, and (2) confirming no code regression was missed — the suite was over an hour in and apparently healthy (all visible tests `PASSED`) at the moment of cancellation. Running Brev leg-tally toward the two-week comparison (brev jobs added 2026-10-14 in PR #448; two-week mark 2026-10-28, **1 day remaining**): 23/28 brev job-runs passed to date (82.1%), up from 80.8% after 10-26. Still below the monitoring brief's 95%-pass-rate bar at face value; per this log's running notes since 10-20, most of the gap (4 job-runs, 10-20/10-21) predates the 10-22 Brev-catalog fix (PR #469) — excluding those, Brev is 23/24 (95.8%) since the fix. Tomorrow's (10-28) scan is the formal two-week mark and should resolve which framing the retirement recommendation uses.

### Run #354 — secondary nightly (schedule `17 4 * * *`, main @ `244136c8`, [40450085233](https://github.com/aqlaboratory/openfold-3/actions/runs/40450085233))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda-brev | **SKIPPED** | — | same guard; job named plain `test-pixi-cuda-brev`, confirming it never expanded |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #353, per this log's established convention). This run's `test-pixi-amd` job entry (created 04:30:14Z, immediately skipped) is itself the concurrency-cancellation trigger for run #353's still-running AMD job — see that run's notes above.

---

## 2026-10-28

**Log access note:** this scan could not fetch any job logs. Every `actions/jobs/<id>/logs` request redirects to `productionresultssa0.blob.core.windows.net`, and this session's egress proxy answered `403` to the `CONNECT` tunnel for that host (policy denial — confirmed via `/__agentproxy/status`'s `recentRelayFailures`, three identical rejections). Per this log's explicit fallback for an unavailable log, every non-pass below is recorded `unclassified (log unavailable)` rather than guessed from the job conclusion alone — a first for this log; every prior night's logs were reachable. `gh variable list` (to confirm `NIGHTLY_CRON`) hit the same pre-existing `actions/variables` proxy restriction noted since 10-23/`weekly-summary.md`, unrelated to tonight's new blob-storage block.

### Run #355 — primary nightly (schedule `17 3 * * *`, main @ `ddcaaa99` — merge PR #489 "Pin ruff version" (new since run #353/10-27's `244136c8`), [40626451630](https://github.com/aqlaboratory/openfold-3/actions/runs/40626451630))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **FAILED** | 2 min | `start-aws-runner` failed 03:24:00–03:25:45. `unclassified (log unavailable)` — cannot confirm `aws-capacity` signature tonight; see log-access note above. |
| test-pixi-cuda (openfold3-cuda13) | **FAILED** | 2 min | `start-aws-runner` failed 03:24:00–03:25:43, same pattern as cuda12. `unclassified (log unavailable)`. |
| test-pixi-cuda-brev (openfold3-cuda12) | **PASSED** | 48 min | `test-openfold-docker-pixi` 03:29:58–04:17:51 (47m53s). Cleanup `stop-brev-runner` succeeded (04:17:54–04:23:57). |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 42 min | `test-openfold-docker-pixi` 03:30:02–04:11:38 (41m36s). Note: this leg's `stop-brev-runner` cleanup step itself **failed** (04:11:41–04:22:09, job 122626868292) — doesn't change the test job's PASSED state, but worth a human's eye once logs are reachable again (teardown failure could leave a Brev instance running/billing). |
| test-pixi-amd (openfold3-rocm7) | **FAILED** | <1 min | `test-openfold-docker-pixi-amd` failed in 7s (03:23:59–03:24:06) — far too fast to have run any test; looks like an early-stage failure (image pull, runner registration, or similar) rather than a test failure, but `unclassified (log unavailable)` since the log can't be confirmed tonight. |

**2026-10-28: 2/5 passed · 0 skipped · 0 queued · 3 need attention**

First night this log could not retrieve any job logs (proxy policy block on `productionresultssa0.blob.core.windows.net`, detailed above) — all three non-passes are recorded `unclassified (log unavailable)` rather than classified from log content, including the two AWS `start-aws-runner` failures that, on every prior occurrence of a same-shaped failure in this log, turned out to be `aws-capacity`. They are **not** entered in `aws-outage-failures.md` tonight precisely because that can't be confirmed from the conclusion alone; all three live in `test-failures.md` instead, flagged for re-classification once log access is restored (either a future session with working proxy access, or a human pulling the raw logs from the Actions UI). Both Brev legs (`test-pixi-cuda-brev`) passed clean on tonight's commit (`ddcaaa99`, PR #489 — a lint-tooling version pin, no functional impact), unaffected by whatever is behind the AWS/AMD non-passes.

This is also the formal two-week mark for the Brev-vs-AWS comparison (`test-pixi-cuda-brev` jobs added 2026-10-14 in PR #448). Including tonight's two passes, the running tally is **25/30 brev job-runs passed (83.3%)** since 10-14 — still below the monitoring brief's 95%-pass-rate bar at face value. All 5 non-passes in that denominator are the 10-20/10-21 `g5.4xlarge`-catalog-rejection streak (4 job-runs, fixed by PR #469 on 10-22) plus the single 10-22 `test-pixi-cuda-brev (openfold3-cuda12)` cap-kill (`unclassified`, 1 job-run) that this log flagged at the time as possibly infra-variance rather than a Brev-specific defect. Excluding only the pre-fix 10-20/10-21 job-runs (not reflective of Brev's current reliability — the catalog issue they hit was fixed two weeks ago and hasn't recurred since): **25/26 (96.2%)**, above the 95% bar. Recommendation: on the post-fix framing, which this log's running notes since 10-20 have consistently argued is the more representative one, **Brev's two-week track record clears the bar — the AWS-hosted `test-pixi-cuda` workflows look ready to be replaced by their Brev equivalents.** Flagging for a human decision rather than acting unilaterally, since (a) the face-value number doesn't clear the bar and the exclusion is a judgment call, (b) tonight's two AWS failures are themselves unclassified pending log access and could independently be adding to the case for Brev, and (c) retiring a workflow is a repo-structure change outside this log's scope.

### Run #356 — secondary nightly (schedule `17 4 * * *`, main @ `ddcaaa99`, [40632031014](https://github.com/aqlaboratory/openfold-3/actions/runs/40632031014))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion (job named plain `test-pixi-cuda`, no matrix suffix) |
| test-pixi-cuda-brev | **SKIPPED** | — | same guard; job named plain `test-pixi-cuda-brev`, confirming it never expanded |
| test-pixi-amd | **SKIPPED** | — | same guard; job named plain `test-pixi-amd`, confirming it never expanded |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #355, per this log's established convention).

---

## 2026-10-29

**Scope change, effective tonight:** PR #493 "Remove aws GPU workflows" (merged 2026-10-28 ~10:27 UTC main `aaa9fb46`, after run #355/#356's scan but before tonight's) deleted the `test-pixi-cuda` (AWS) job and its reusable workflow (`ci-integration-test-aws-pixi-cuda.yml`) entirely from `integration-test.yml`. This is not a skip or a failure — tonight's run has no `test-pixi-cuda` job entry of any kind (confirmed against the full job list: 7/7 jobs accounted for, all `test-pixi-amd`/`test-pixi-cuda-brev` start/test/stop). This is the exact move this log's running Brev-vs-AWS comparison (tracked since `test-pixi-cuda-brev` was added 2026-10-14, PR #448) had been building toward, and the one the 10-28 entry explicitly recommended that same day. The file header above still describes the five-job scope that applied 2026-10-14 through 2026-10-28; as of tonight the live scope is three jobs (`test-pixi-cuda-brev` cuda12/cuda13 + `test-pixi-amd`). Per this routine's append-only rule the header itself is left as-is; this note is the record of the change. The two retired jobs are marked **RETIRED** below (none of this log's five defined states — PASSED/FAILED/CANCELLED/SKIPPED/QUEUED — fit "removed from the workflow"), and are excluded from tonight's coverage denominator.

### Run #357 — primary nightly (schedule `17 3 * * *`, main @ `fb656973` — on top of PR #493 "Remove aws GPU workflows" and PR #490 "refactor(templates): make the preprocessing worker pool configurable", new since run #355/10-28's `ddcaaa99`, [40811962445](https://github.com/aqlaboratory/openfold-3/actions/runs/40811962445))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda (openfold3-cuda12) | **RETIRED** | — | Job removed from `integration-test.yml` by PR #493, merged 2026-10-28. No job entry in tonight's run. |
| test-pixi-cuda (openfold3-cuda13) | **RETIRED** | — | Same — removed by PR #493. |
| test-pixi-cuda-brev (openfold3-cuda12) | **PASSED** | 50 min | `test-openfold-docker-pixi` 03:31:05–04:20:51 (49m46s). |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 43 min | `test-openfold-docker-pixi` 03:30:52–04:13:47 (42m55s). |
| test-pixi-amd (openfold3-rocm7) | **FAILED** | <1 min | `test-openfold-docker-pixi-amd` failed in 8s (03:24:30–03:24:38), in the `Run actions/checkout@v7` step's workspace-clean sub-step: `##[error]File was unable to be removed Error: EACCES: permission denied, unlink '/home/jan/actions-runner/_work/openfold-3/openfold-3/.pytest_cache/.gitignore'`. Matches none of this log's seven defined signatures — no pytest ran (checkout itself never completed), no AWS/Brev runner-start step involved, no MSA/timeout text. Recorded `unclassified`; full detail in `test-failures.md`. |

**2026-10-29: 2/3 passed · 0 skipped · 0 queued · 1 need attention** (scope now 3 jobs, not 5 — see scope-change note above; both AWS `test-pixi-cuda` legs are RETIRED, not FAILED/SKIPPED, and are not counted in this denominator)

**Two-week Brev-vs-AWS comparison — closed out tonight.** `test-pixi-cuda-brev` was added 2026-10-14 (PR #448); the formal two-week mark landed 2026-10-28, and that night's entry above computed a running tally of 25/30 brev job-runs passed (83.3% face value; 25/26 = 96.2% excluding the pre-PR#469-fix 10-20/10-21 catalog-rejection streak) and recommended retiring the AWS `test-pixi-cuda` jobs on the post-fix framing. Tonight's two Brev passes extend that tally to **27/32 (84.4% face value) / 27/28 (96.4% excluding the pre-fix streak)** — both framings now comfortably at or above the 95% bar once the fixed-and-resolved 10-20/10-21 incident is excluded. A human had already acted on the 10-28 recommendation before tonight's run: PR #493 merged 2026-10-28 ~10:27 UTC, deleting the AWS `test-pixi-cuda` jobs outright. This closes the comparison this log has tracked nightly since 10-14 — no further action needed, and this log's per-day Brev tally bookkeeping ends here since there is no longer an AWS counterpart to compare against.

### Run #358 — secondary nightly (schedule `17 4 * * *`, main @ `fb656973`, [40817507618](https://github.com/aqlaboratory/openfold-3/actions/runs/40817507618))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda-brev | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion |
| test-pixi-amd | **SKIPPED** | — | same guard |

No `test-pixi-cuda` job entry tonight (consistent with its removal from the workflow — the guard-skip entry that showed up on every prior secondary slot is also gone, since the job no longer exists to be gated). Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #357, per this log's established convention).

---

## 2026-10-30

### Run #359 — primary nightly (schedule `17 3 * * *`, main @ `ac128070` — merge PR #486 "fix: don't reassign used ground-truth copies in greedy permutation", on top of PR #498 "ci: run the test image by digest, not by a shared tag", PR #495 (templates test worker config) and PR #476 (AMD `pl_trainer_args` passthrough), new since run #357/10-29's `fb656973`, [40997176233](https://github.com/aqlaboratory/openfold-3/actions/runs/40997176233))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda-brev (openfold3-cuda12) | **CANCELLED** | 60 min | `test-openfold-docker-pixi` hit the 60-min `timeout-minutes` cap, 03:30:11–04:30:30 (60m19s). Last pytest line before cancellation: `test_templates.py::test_template_lowers_rmsd[1y57]` started with no PASSED/FAILED recorded after it — superficially `msa-hang`-shaped (cap-kill + no-verdict test) but doesn't match the defined signature, which requires a `test_inference_writes_outputs[msa-...]` case, not `test_template_lowers_rmsd`. Recorded `unclassified`; full detail in `test-failures.md`. |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 24 min | `test-openfold-docker-pixi` 03:29:57–03:53:58 (24m01s). |
| test-pixi-amd (openfold3-rocm7) | **FAILED** | <1 min | `test-openfold-docker-pixi-amd` failed in 7s (03:24:13–03:24:20), in `Run actions/checkout@v7`'s workspace-clean sub-step: `##[error]File was unable to be removed Error: EACCES: permission denied, unlink '/home/jan/actions-runner/_work/openfold-3/openfold-3/.pytest_cache/.gitignore'`. Identical signature to 10-29's AMD failure, same runner (`omsf-amd-aupcloud`), same file — the 10-29 entry's prediction that this would block every subsequent run until manually cleared has held. Matches none of this log's seven defined signatures; recorded `unclassified`. |

**2026-10-30: 1/3 passed · 0 skipped · 0 queued · 2 need attention**

Second consecutive night the AMD self-hosted runner has failed on the exact same `EACCES` unlink of `.pytest_cache/.gitignore` — this now needs a human to clear it on the runner host itself; this log cannot. The cuda12 Brev cancellation is new — first cap-hit since 10-22 — and looks unrelated to the AMD issue (different job, different machine, different cause); the run was mid-way through a template-RMSD test, not blocked on an external MSA-server call, so it is recorded `unclassified` rather than guessed as `msa-hang`. Tonight's commit (`ac128070`, PR #486 — ground-truth permutation reuse fix) is unlikely to be the cause of either non-pass: the one job that ran to completion (cuda13 Brev) passed clean on the same commit.

### Run #360 — secondary nightly (schedule `17 4 * * *`, main @ `ac128070`, [41002627203](https://github.com/aqlaboratory/openfold-3/actions/runs/41002627203))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda-brev | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion |
| test-pixi-amd | **SKIPPED** | — | same guard |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #359, per this log's established convention).

---

## 2026-10-31

### Run #361 — primary nightly (schedule `17 3 * * *`, main @ `ab517c1b` — PR #488 "fix: keep CCD stereochemistry for polymer residues built from a sequence", new since run #359/360's `ac128070`, [41163871779](https://github.com/aqlaboratory/openfold-3/actions/runs/41163871779))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda-brev (openfold3-cuda12) | **PASSED** | 25 min | `test-openfold-docker-pixi` 03:25:37–03:50:30 (24m53s). |
| test-pixi-cuda-brev (openfold3-cuda13) | **PASSED** | 24 min | `test-openfold-docker-pixi` 03:25:53–03:49:40 (23m47s). |
| test-pixi-amd (openfold3-rocm7) | **FAILED** | <1 min | `test-openfold-docker-pixi-amd` failed in 6s (03:19:30–03:19:36), in `Run actions/checkout@v7`'s workspace-clean sub-step: `##[error]File was unable to be removed Error: EACCES: permission denied, unlink '/home/jan/actions-runner/_work/openfold-3/openfold-3/.pytest_cache/.gitignore'`. Identical signature to 10-29 and 10-30's AMD failures, same runner (`omsf-amd-aupcloud`), same file — third consecutive night. Matches none of this log's seven defined signatures; recorded `unclassified`. |

**2026-10-31: 2/3 passed · 0 skipped · 0 queued · 1 need attention**

Third consecutive night the AMD self-hosted runner (`omsf-amd-aupcloud`) has failed on the exact same `EACCES` unlink of `.pytest_cache/.gitignore` during `actions/checkout`'s workspace-clean step — still needs a human to clear it on the runner host itself (`sudo rm -rf` or a container-UID-mapping fix); this log cannot. Both Brev CUDA legs (cuda12, cuda13) passed clean on tonight's commit (`ab517c1b`, PR #488 — CCD stereochemistry fix for polymer residues), so no code regression indicated.

### Run #362 — secondary nightly (schedule `17 4 * * *`, main @ `ab517c1b`, [41168069261](https://github.com/aqlaboratory/openfold-3/actions/runs/41168069261))

| Job | State | Duration | Notes |
|-----|-------|----------|-------|
| test-pixi-cuda-brev | **SKIPPED** | — | `if:` guard: `github.event_name != 'schedule' \|\| (vars.RUN_NIGHTLY == 'true' && github.event.schedule == vars.NIGHTLY_CRON)` evaluated false on this slot (`17 4 * * *`) — job-level skip before matrix expansion |
| test-pixi-amd | **SKIPPED** | — | same guard |

Expected skip on the secondary slot (intended for other repos, not `aqlaboratory/openfold-3`) — same pattern as every prior night in this log. Does not affect the day's coverage line above (based on the primary slot, run #361, per this log's established convention).
