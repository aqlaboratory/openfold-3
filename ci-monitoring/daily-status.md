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
