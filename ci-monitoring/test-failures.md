# CI Test Failures (Non-AWS-Outage)

This file records CI failures that appear to be related to OF3 code or infrastructure
(not AWS GPU unavailability).

---

## 2026-08-04

**Run:** [30914218305](https://github.com/aqlaboratory/openfold-3/actions/runs/30914218305)  
**Branch:** `jandom/2026-08/feat/mps-inference`  
**Triggered:** 2026-08-04T13:31:10Z  

### Failed Jobs

| Job | Test | Error |
|-----|------|-------|
| `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml) / test-openfold-docker` | `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin]` | `RuntimeError: DataLoader worker (pid(s) 11342) exited unexpectedly` |
| `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` | (logs unavailable — 404) | — |

### Root Cause

The DataLoader worker crashed with a bus error inside the Docker container:

```
ERROR: Unexpected bus error encountered in worker. This might be caused by insufficient shared memory (shm).
```

This is **not** an AWS GPU outage. The failure is consistent with the Docker container
having insufficient `/dev/shm` for the DataLoader's shared-memory IPC. Occurs on the
`msa-no_templates-ubiquitin` inference test variant that fetches MSAs from the
ColabFold server before running prediction.

---

## 2026-08-09

**Cause:** External service timeout — ColabFold API (`api.colabfold.com`) connection timed out during the `msa-no_templates-ubiquitin` test. Not an AWS GPU outage; not an OF3 code bug. Two consecutive runs affected.

### Run 1
- **Run ID:** [31293281898](https://github.com/aqlaboratory/openfold-3/actions/runs/31293281898)
- **Time:** 2026-08-09T03:50:49Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 93194112488)
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin]`
- **Error:** `requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.`

### Run 2
- **Run ID:** [31295107418](https://github.com/aqlaboratory/openfold-3/actions/runs/31295107418)
- **Time:** 2026-08-09T04:41:42Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 93198826302)
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin]`
- **Error:** `requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.`

---

## 2026-08-11

**Cause:** External service timeout — ColabFold API (`api.colabfold.com`) connection timed out during the `msa-no_templates-ubiquitin` test. Not an AWS GPU outage; not an OF3 code bug. Two consecutive scheduled runs affected (same pattern as 2026-08-09).

### Run 1
- **Run ID:** [31456689857](https://github.com/aqlaboratory/openfold-3/actions/runs/31456689857)
- **Time:** 2026-08-11T03:52:02Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 93671838497)
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin]`
- **Error:** `requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.`

### Run 2
- **Run ID:** [31459386217](https://github.com/aqlaboratory/openfold-3/actions/runs/31459386217)
- **Time:** 2026-08-11T04:43:45Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 93679698753)
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin]`
- **Error:** `requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.`

---

## 2026-08-12

**Cause:** External service timeout — ColabFold API (`api.colabfold.com`) connection timed out during the `msa-no_templates-ubiquitin` test. Not an AWS GPU outage; not an OF3 code bug. Two consecutive scheduled runs affected (same recurring pattern as 2026-08-09 and 2026-08-11).

### Run 1
- **Run ID:** [31561679174](https://github.com/aqlaboratory/openfold-3/actions/runs/31561679174)
- **Time:** 2026-08-12T03:57:06Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 94005121395)
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin]`
- **Error:** `requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.`

### Run 2
- **Run ID:** [31564740462](https://github.com/aqlaboratory/openfold-3/actions/runs/31564740462)
- **Time:** 2026-08-12T04:53:46Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 94014160549)
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin]`
- **Error:** `requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.`

---

## 2026-08-18

**Cause:** Docker image build failure on the AMD GPU runner. Tests never ran. Logs unavailable (HTTP 404). Not an AWS GPU outage (self-hosted AMD runner, not CUDA/EC2). Single scheduled run affected.

- **Run ID:** [32099159272](https://github.com/aqlaboratory/openfold-3/actions/runs/32099159272)
- **Time:** 2026-08-18T04:26:34Z – 04:36:16Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 95596199605)
- **Runner:** `omsf-amd-aupcloud` (self-hosted AMD GPU runner)
- **Failed Step:** `Build and push test image` (step 5 of 9; all subsequent steps pending/skipped)
- **Skipped Jobs:** `test-conda`, `test-pixi-cuda` (both skipped — unrelated to this failure)
- **Error:** Logs unavailable (HTTP 404); exact error message not retrievable
- **Assessment:** Failure during Docker build phase before any test code executed. Could be a transient AMD runner/GHCR push issue or a Dockerfile/dependency change that broke the build. Warrants manual inspection if it recurs.

---
## 2026-08-19

**Cause:** Docker image build failure on the AMD GPU runner — same failure pattern as 2026-08-18. Tests never ran. Logs unavailable (HTTP 404). Not an AWS GPU outage (self-hosted AMD runner, not CUDA/EC2). There was also a cancelled run earlier the same day (run 32212447147, triggered at 03:31, cancelled before starting).

- **Run ID:** [32215774633](https://github.com/aqlaboratory/openfold-3/actions/runs/32215774633)
- **Triggered:** 2026-08-19T04:26:36Z (queued for ~10 hours; started 2026-08-19T14:27:09Z)
- **Completed:** 2026-08-19T14:36:43Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 95956906036)
- **Runner:** `omsf-amd-aupcloud` (self-hosted AMD GPU runner)
- **Failed Step:** `Build and push test image` (step 5; all subsequent steps pending/skipped)
- **Skipped Jobs:** `test-conda`, `test-pixi-cuda` (both skipped — unrelated to this failure)
- **Error:** Logs unavailable (HTTP 404); exact error message not retrievable
- **Assessment:** Same Docker build failure pattern as Aug 18 — two consecutive days. The long queue time (~10 hours) may indicate the runner was busy or unstable. Strongly warrants investigation of the AMD runner and Docker build pipeline.

---
## 2026-08-22

**Cause:** Missing default checkpoint on the AMD/ROCm runner — `openbind-2025-06-30-174k` not found in `/root/.openfold3`. The test fails at config validation (pydantic `ValidationError`) before any model inference runs, indicating `setup_openfold` was not run or failed silently in the CI environment. Not an AWS GPU outage. Two consecutive scheduled runs affected, same error in both.

### Run 1
- **Run ID:** [32549099790](https://github.com/aqlaboratory/openfold-3/actions/runs/32549099790)
- **Time:** 2026-08-22T03:29:13Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 96972765376)
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[no_msa-no_templates-protein_only]`
- **Error:** `pydantic_core._pydantic_core.ValidationError: 1 validation error for InferenceExperimentConfig — Value error, Default checkpoint openbind-2025-06-30-174k not found in /root/.openfold3, cowardly refusing to perform inference. Please run setup_openfold to download the current default parameters or specify a valid checkpoint path with --inference-ckpt-path`

### Run 2
- **Run ID:** [32551686139](https://github.com/aqlaboratory/openfold-3/actions/runs/32551686139)
- **Time:** 2026-08-22T04:26:04Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 96979367313)
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[no_msa-no_templates-protein_only]`
- **Error:** `pydantic_core._pydantic_core.ValidationError: 1 validation error for InferenceExperimentConfig — Value error, Default checkpoint openbind-2025-06-30-174k not found in /root/.openfold3, cowardly refusing to perform inference. Please run setup_openfold to download the current default parameters or specify a valid checkpoint path with --inference-ckpt-path`

---

## 2026-08-23

**Cause:** Same missing checkpoint error as 2026-08-22 — `openbind-2025-06-30-174k` not found in `/root/.openfold3` on the AMD/ROCm runner. Third consecutive day with this failure. Two scheduled runs affected.

### Run 1
- **Run ID:** [32615678684](https://github.com/aqlaboratory/openfold-3/actions/runs/32615678684)
- **Time:** 2026-08-23T03:34:11Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 97135888268)
- **Runner:** `omsf-amd-aupcloud` (self-hosted AMD GPU)
- **Commit:** `f9649cce7de32382bc1100e8e9e1de2301adf2c2`
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[no_msa-no_templates-protein_only]`
- **Error:** `pydantic_core._pydantic_core.ValidationError: 1 validation error for InferenceExperimentConfig — Value error, Default checkpoint openbind-2025-06-30-174k not found in /root/.openfold3, cowardly refusing to perform inference.`
- **Skipped Jobs:** `test-conda`, `test-pixi-cuda`

### Run 2
- **Run ID:** [32617889227](https://github.com/aqlaboratory/openfold-3/actions/runs/32617889227)
- **Time:** 2026-08-23T04:26:46Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 97141446677)
- **Runner:** `omsf-amd-aupcloud` (self-hosted AMD GPU)
- **Commit:** `f9649cce7de32382bc1100e8e9e1de2301adf2c2`
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[no_msa-no_templates-protein_only]`
- **Error:** `pydantic_core._pydantic_core.ValidationError: 1 validation error for InferenceExperimentConfig — Value error, Default checkpoint openbind-2025-06-30-174k not found in /root/.openfold3, cowardly refusing to perform inference.`
- **Skipped Jobs:** `test-conda`, `test-pixi-cuda`

---

## 2026-08-24

**Cause:** Same missing checkpoint error as 2026-08-22 and 2026-08-23 — `openbind-2025-06-30-174k` not found in `/root/.openfold3` on the AMD/ROCm runner. Fourth consecutive day with this identical failure. Two scheduled runs affected. Persistent OF3 CI-infrastructure issue on the AMD runner, not an AWS GPU outage.

### Run 1
- **Run ID:** [32686976308](https://github.com/aqlaboratory/openfold-3/actions/runs/32686976308)
- **Time:** 2026-08-24T03:35:26Z (run_attempt 2)
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 97316237360)
- **Runner:** `omsf-amd-aupcloud` (self-hosted AMD GPU)
- **Commit:** `f9649cce7de32382bc1100e8e9e1de2301adf2c2`
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[no_msa-no_templates-protein_only]`
- **Error:** `pydantic_core._pydantic_core.ValidationError: 1 validation error for InferenceExperimentConfig — Value error, Default checkpoint openbind-2025-06-30-174k not found in /root/.openfold3, cowardly refusing to perform inference. Please run setup_openfold to download the current default parameters or specify a valid checkpoint path with --inference-ckpt-path`
- **Skipped Jobs:** `test-conda`, `test-pixi-cuda`

### Run 2
- **Run ID:** [32690260716](https://github.com/aqlaboratory/openfold-3/actions/runs/32690260716)
- **Time:** 2026-08-24T04:30:19Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 97322529883)
- **Runner:** `omsf-amd-aupcloud` (self-hosted AMD GPU)
- **Commit:** `f9649cce7de32382bc1100e8e9e1de2301adf2c2`
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[no_msa-no_templates-protein_only]`
- **Error:** `pydantic_core._pydantic_core.ValidationError: 1 validation error for InferenceExperimentConfig — Value error, Default checkpoint openbind-2025-06-30-174k not found in /root/.openfold3, cowardly refusing to perform inference. Please run setup_openfold to download the current default parameters or specify a valid checkpoint path with --inference-ckpt-path`
- **Skipped Jobs:** `test-conda`, `test-pixi-cuda`

---

## 2026-08-25

**Cause:** Same missing checkpoint error as 2026-08-22 through 2026-08-24 — `openbind-2025-06-30-174k` not found in `/root/.openfold3` on the AMD/ROCm runner. Fifth consecutive day with this identical failure. One scheduled run observed (only run today; `test-conda` and `test-pixi-cuda` were skipped). Not an AWS GPU outage.

- **Run ID:** [32809003809](https://github.com/aqlaboratory/openfold-3/actions/runs/32809003809)
- **Time:** 2026-08-25T04:27:01Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 97684623800)
- **Runner:** `omsf-amd-aupcloud` (self-hosted AMD GPU)
- **Commit:** `f9649cce7de32382bc1100e8e9e1de2301adf2c2`
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[no_msa-no_templates-protein_only]`
- **Error:** `pydantic_core._pydantic_core.ValidationError: 1 validation error for InferenceExperimentConfig — Value error, Default checkpoint openbind-2025-06-30-174k not found in /root/.openfold3, cowardly refusing to perform inference. Please run setup_openfold to download the current default parameters or specify a valid checkpoint path with --inference-ckpt-path`
- **Skipped Jobs:** `test-conda`, `test-pixi-cuda`

---

## 2026-08-26

**Cause:** Same missing checkpoint error as 2026-08-22 through 2026-08-25 — `openbind-2025-06-30-174k` not found in `/root/.openfold3` on the AMD/ROCm runner. Sixth consecutive day with this identical failure. Two scheduled runs affected. Not an AWS GPU outage.

### Run 1
- **Run ID:** [32927057774](https://github.com/aqlaboratory/openfold-3/actions/runs/32927057774)
- **Time:** 2026-08-26T03:36:05Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 98051979359)
- **Runner:** `omsf-amd-aupcloud` (self-hosted AMD GPU)
- **Commit:** `f9649cce7de32382bc1100e8e9e1de2301adf2c2`
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[no_msa-no_templates-protein_only]`
- **Error:** `pydantic_core._pydantic_core.ValidationError: 1 validation error for InferenceExperimentConfig — Value error, Default checkpoint openbind-2025-06-30-174k not found in /root/.openfold3, cowardly refusing to perform inference. Please run setup_openfold to download the current default parameters or specify a valid checkpoint path with --inference-ckpt-path`
- **Skipped Jobs:** `test-conda`, `test-pixi-cuda`

### Run 2
- **Run ID:** [32930290344](https://github.com/aqlaboratory/openfold-3/actions/runs/32930290344)
- **Time:** 2026-08-26T04:27:26Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 98061091635)
- **Runner:** `omsf-amd-aupcloud` (self-hosted AMD GPU)
- **Commit:** `f9649cce7de32382bc1100e8e9e1de2301adf2c2`
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin]`
- **Error:** `pydantic_core._pydantic_core.ValidationError: 1 validation error for InferenceExperimentConfig — Value error, Default checkpoint openbind-2025-06-30-174k not found in /root/.openfold3, cowardly refusing to perform inference. Please run setup_openfold to download the current default parameters or specify a valid checkpoint path with --inference-ckpt-path`
- **Skipped Jobs:** `test-conda`, `test-pixi-cuda`

---

## 2026-08-27

**Cause:** `curl` parameter download failure on the AMD runner — the output destination `/home/jan/.openfold3/` is a directory rather than a file. Tests never ran. Not an AWS GPU outage. The error indicates a CI runner state issue (the path exists as a directory, blocking the download). This is a new failure mode compared to the preceding checkpoint-not-found errors (Aug 22–26).

- **Run ID:** [33045827431](https://github.com/aqlaboratory/openfold-3/actions/runs/33045827431)
- **Run #:** 220
- **Time:** 2026-08-27T06:25:49Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 98429469520)
- **Runner:** `omsf-amd-aupcloud` (self-hosted AMD GPU)
- **Commit:** `042c74f340c18cff6d35ff9dc00c444a335aa5be`
- **Failed Step:** `Download OpenFold parameters` (step 8)
- **Error:** `Warning: Failed to open the file /home/jan/.openfold3/: Is a directory` / `curl: (23) Failure writing output to destination`
- **Skipped Jobs:** `test-conda`, `test-pixi-cuda`
- **Assessment:** The curl output target path already exists as a directory on the AMD self-hosted runner. This prevents parameter download and causes the integration test step to be skipped entirely. Likely requires manual cleanup of `/home/jan/.openfold3/` on the `omsf-amd-aupcloud` runner.

---

## 2026-09-03

**Cause:** Same checkpoint validation error (`openbind-2025-06-30-174k` not found in `/root/.openfold3`) — now appearing on CUDA runners as well as AMD. Failure on PR branch `claude/pr-393-test-import-error-zac4a1` (PR #394, workflow_dispatch). The scheduled main branch run #236 (2026-09-03T05:22:45Z) **passed**. The AMD job passed in this PR run; three CUDA test jobs failed at the "Run integration test" step. AWS instance provisioning succeeded for all jobs (not an AWS capacity issue).

- **Run ID:** [33714692716](https://github.com/aqlaboratory/openfold-3/actions/runs/33714692716)
- **Run #:** 234
- **Branch:** `claude/pr-393-test-import-error-zac4a1` (PR #394)
- **Event:** workflow_dispatch
- **Time:** 2026-09-03T04:20:40Z – 04:59:25Z
- **Commit:** `8cba31c782c9e5a07271184bc2618f4e8883c825` (Source deepspeed from conda-forge and regenerate the production lock)
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[no_msa-no_templates-protein_only]`
- **Error:** `pydantic_core._pydantic_core.ValidationError: 1 validation error for InferenceExperimentConfig — Value error, Default checkpoint openbind-2025-06-30-174k not found in /root/.openfold3, cowardly refusing to perform inference. Please run setup_openfold to download the current default parameters or specify a valid checkpoint path with --inference-ckpt-path`

### Failed Jobs

| Job | Job ID | Runner |
|-----|--------|--------|
| `test-pixi-cuda (openfold3-cuda12) / test-openfold-docker-pixi` | 100521573034 | AWS EC2 (runner-63hpalu7) |
| `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml) / test-openfold-docker` | 100521607984 | AWS EC2 (runner-xxjq2fkq) |
| `test-pixi-cuda (openfold3-cuda13) / test-openfold-docker-pixi` | 100521682476 | AWS EC2 (runner-jmj2u817) |

### Passing Jobs

- `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` — **passed** on `omsf-amd-aupcloud`
- All `start-aws-runner` and `stop-aws-runner` jobs — **passed**

### Assessment

Not an AWS GPU outage. The parameter cache was restored (step 7 succeeded) but did not contain the `openbind-2025-06-30-174k` checkpoint name required by the PR's code. This PR changes the conda environment (deepspeed from PyPI to conda-forge, pytorch 2.5.1→2.10.0), which may have caused a cache miss or served a stale cache lacking the new checkpoint. The same checkpoint name mismatch drove the AMD runner failures from 2026-08-22 through 2026-08-27. The scheduled main run passed on the same day, suggesting the issue is specific to this PR's cache state or environment.

---

## 2026-09-08

**Cause:** External service timeout — ColabFold API (`api.colabfold.com`) connection timed out during the `msa-no_templates-ubiquitin` test. Recurring pattern previously seen on 2026-08-09, 2026-08-11, and 2026-08-12. Not an AWS GPU outage; not an OF3 code bug. Two consecutive scheduled runs affected.

### Run 1
- **Run ID:** [34183685544](https://github.com/aqlaboratory/openfold-3/actions/runs/34183685544)
- **Run #:** 248
- **Time:** 2026-09-08T03:29:28Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 101927687126)
- **Runner:** `omsf-amd-aupcloud` (self-hosted AMD GPU)
- **Commit:** `e47adbf95448956574603c9e8eb703cbb0c3bb0b`
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin]`
- **Error:** `requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.`
- **Skipped Jobs:** `test-conda`, `test-pixi-cuda`

### Run 2
- **Run ID:** [34187127942](https://github.com/aqlaboratory/openfold-3/actions/runs/34187127942)
- **Run #:** 249
- **Time:** 2026-09-08T04:28:47Z
- **Failed Job:** `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` (job ID: 101937614761)
- **Runner:** `omsf-amd-aupcloud` (self-hosted AMD GPU)
- **Commit:** `e47adbf95448956574603c9e8eb703cbb0c3bb0b`
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin]`
- **Error:** `requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.`
- **Skipped Jobs:** `test-conda`, `test-pixi-cuda`

---

## 2026-09-11

**Cause:** Default checkpoint `openbind-2025-06-30-174k` not found in `/root/.openfold3` on the CUDA EC2 runners. The `actions/cache` restore succeeded (step 7: "Cache download of parameters" — hit) but delivered files to the wrong path. The AMD cache entry was stored with `HOME=/home/jan` (the AMD runner's persistent user); EC2 runners have `HOME=/home/ubuntu`. Because `actions/cache` stores absolute paths (tar -P), the restore unpacked files to `/home/jan/.openfold3` on the EC2 runner — a path not mounted into Docker — while the cache-hit gate skipped the S3 download, leaving `/root/.openfold3` empty. Fixed in PR #404 (`feature/split-parameter-cache-by-runner`); subsequent run #255 passed.

- **Run ID:** [34558633289](https://github.com/aqlaboratory/openfold-3/actions/runs/34558633289)
- **Run #:** 254
- **Branch:** main @ `346c0efe3b4d4f08c3f8c8ad481fe3dc999287cf`
- **Time:** 2026-09-11T03:29:31Z – 04:19:54Z

### Failed / Cancelled Jobs

| Job | Status | Job ID | Runner |
|-----|--------|--------|--------|
| `test-pixi-cuda (openfold3-cuda13) / test-openfold-docker-pixi` | **failure** | 103136993054 | ip-172-31-5-56 (runner-03f3rsj7) |
| `test-pixi-cuda (openfold3-cuda12) / test-openfold-docker-pixi` | cancelled | 103136966042 | ip-172-31-1-25 (runner-xjaxxdi2) |

### Passing Jobs

- `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` — **passed** (AMD runner uses local disk, not `actions/cache`)

### Failed Test

```
FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[no_msa-no_templates-protein_only]
pydantic_core._pydantic_core.ValidationError: 1 validation error for InferenceExperimentConfig
  Value error, Default checkpoint openbind-2025-06-30-174k not found in /root/.openfold3,
  cowardly refusing to perform inference. Please run `setup_openfold` to download the current
  default parameters or specify a valid checkpoint path with `--inference-ckpt-path`
```

---

## 2026-09-16

**Cause:** External service timeout — RCSB PDB GraphQL API (`data.rcsb.org`) read-timed-out (30s) while `fetch_label_to_author_chain_ids` (`openfold3/core/data/tools/rscb.py`) was remapping ColabFold template chain IDs for 218 PDB entries. A new external-dependency timeout signature for this log — distinct from the recurring `api.colabfold.com` MSA-submission timeouts seen on 2026-08-09, 2026-08-11, 2026-08-12, and 2026-09-08 (this run's log also shows an unrelated, non-fatal `api.colabfold.com` retry warning earlier in the same job, which the code recovered from). Classifies as `code` per this log's rules (pytest FAILED line + exception from library code), same as the ColabFold timeouts, even though the underlying trigger is an external-service flake, not an OF3 logic bug. Not an AWS GPU outage — AWS instance provisioning succeeded for both CUDA legs. Same commit (`6569fcc7edd4afd5f887bf924ae0d2f613977763`) passed 3/3 the previous night (run #264), confirming this is not a regression from a code or workflow change.

- **Run ID:** [35052050368](https://github.com/aqlaboratory/openfold-3/actions/runs/35052050368)
- **Run #:** 266
- **Branch:** main @ `6569fcc7edd4afd5f887bf924ae0d2f613977763`
- **Time:** 2026-09-16T03:30:22Z – 04:03:47Z
- **Failed Job:** `test-pixi-cuda (openfold3-cuda12) / test-openfold-docker-pixi` (job ID: 104654864474)
- **Failed Test:** `openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin]`
- **Error:** `RuntimeError: Failed to fetch chain ID mappings from RCSB for 218 entries. Cannot proceed without chain ID re-mapping.` — chained from `requests.exceptions.ReadTimeout: HTTPSConnectionPool(host='data.rcsb.org', port=443): Read timed out. (read timeout=30)` (`openfold3/core/data/tools/rscb.py:70`, request originates at `rscb.py:60`)

### Cascading cancellation

- `test-pixi-cuda (openfold3-cuda13) / test-openfold-docker-pixi` (job ID: 104654836117) — **cancelled** (`##[error]The operation was canceled.`) at 2026-09-16T03:50:58Z, ~5 minutes after the cuda12 sibling failed. `test-pixi-cuda`'s `strategy:` block does not set `fail-fast: false`, so GitHub's default matrix fail-fast cancelled this leg mid-test (submitting `test_pocket_constraint_localizes_ligand` to the ColabFold MSA server) once cuda12 failed — not a timeout-minutes cap, not superseded by a newer run, not manual. Same fail-fast mechanism as run #254 (2026-09-11), with the failing/cancelled legs reversed.

### Passing Jobs

- `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` — **passed** (33 min)

---

## 2026-09-17

**Cause:** Same RCSB PDB GraphQL API (`data.rcsb.org`) chain-ID-mapping failure first seen on 2026-09-16 (`fetch_label_to_author_chain_ids` in `openfold3/core/data/tools/rscb.py`), same commit (`6569fcc7edd4afd5f887bf924ae0d2f613977763`), but tonight it hit two of the three tracked jobs directly and took out the third via matrix fail-fast — the first night in this log where all three tracked jobs are simultaneously non-passing. Classifies as `code` per this log's rules (pytest FAILED line + exception from library code), same as 09-16, even though the underlying trigger is an external-service dependency.

- **Run ID:** [35178424759](https://github.com/aqlaboratory/openfold-3/actions/runs/35178424759)
- **Run #:** 268
- **Branch:** main @ `6569fcc7edd4afd5f887bf924ae0d2f613977763`
- **Time:** 2026-09-17T03:30:29Z – 03:53:05Z

### Failed Jobs

| Job | Status | Job ID | Runner |
|-----|--------|--------|--------|
| `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` | **failure** | 105065157600 | omsf-amd-aupcloud |
| `test-pixi-cuda (openfold3-cuda13) / test-openfold-docker-pixi` | **failure** | 105065627596 | ip-172-31-4-68 (runner-cuy5rzak) |

### Failed Test (both jobs, identical signature)

```
FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin]
RuntimeError: Failed to fetch chain ID mappings from RCSB for 218 entries. Cannot proceed without chain ID re-mapping.
```

(`openfold3/core/data/tools/rscb.py`, same code path as the 09-16 failure.)

### Cascading cancellation

- `test-pixi-cuda (openfold3-cuda12) / test-openfold-docker-pixi` (job ID: 105065614045) — **cancelled** (`##[error]The operation was canceled.`) at 2026-09-17T03:48:10Z, ~5 minutes after the cuda13 sibling failed at 03:42:48Z. `test-pixi-cuda`'s `strategy:` block does not set `fail-fast: false`, so GitHub's default matrix fail-fast cancelled this leg mid-test (submitting `test_pocket_constraint_localizes_ligand` to the ColabFold MSA server, which had just completed) once cuda13 failed — not a timeout-minutes cap, not superseded by a newer run, not manual. Same fail-fast mechanism as runs #254 (2026-09-11) and #266 (2026-09-16).

### Passing Jobs

None. First night in this log where all three tracked jobs (`test-pixi-amd`, `test-pixi-cuda (openfold3-cuda12)`, `test-pixi-cuda (openfold3-cuda13)`) are simultaneously non-passing.

---

## 2026-09-19

**Cause:** `api.colabfold.com` MSA-submission read-timeout — the same external-service signature first seen on 2026-09-08, now recurring after a six-night gap (09-09 through 09-18 all showed a different or no failure signature). First nightly run on `7de748b7bc93adb5af0a4032d5e9f208b6e1325f` (PR #417 `feature/rcsb-template-call-fix`, merged 2026-09-18T08:09:10Z — validated clean via `workflow_dispatch` run #272 the night before). The RCSB template-fetch code path touched by PR #417 was not implicated in tonight's failure; the exception originates from ColabFold MSA submission (`openfold3/core/data/tools/colabfold_msa_server.py`), an unrelated code path. Classifies as `code` per this log's rules (pytest FAILED line + exception from library code), even though the underlying trigger is an external ColabFold API outage/flake, not an OF3 regression.

- **Run ID:** [35418661454](https://github.com/aqlaboratory/openfold-3/actions/runs/35418661454)
- **Run #:** 273
- **Branch:** main @ `7de748b7bc93adb5af0a4032d5e9f208b6e1325f`
- **Time:** 2026-09-19T03:29:00Z – 03:55:44Z

### Failed Jobs

| Job | Status | Job ID | Runner |
|-----|--------|--------|--------|
| `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` | **failure** | 105832086384 | omsf-amd-aupcloud |
| `test-pixi-cuda (openfold3-cuda13) / test-openfold-docker-pixi` | **failure** | 105832419584 | ip-172-31 (AWS) |

### Failed Test (both jobs, identical signature)

```
FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin]
requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.
```

Both jobs show 6 retries against the ColabFold server, each read-timing-out, before pytest recorded the failure (`openfold3/core/data/tools/colabfold_msa_server.py:279`/`283`, raised from `download()` at `colabfold_msa_server.py:269`).

### Cascading cancellation

- `test-pixi-cuda (openfold3-cuda12) / test-openfold-docker-pixi` (job ID: 105832422791) — **cancelled** (`##[error]The operation was canceled.`) at 2026-09-19T03:50:34Z, ~5 minutes after the cuda13 sibling failed at 03:45:13Z, and only 19 of the 60 allotted `timeout-minutes` elapsed. `test-pixi-cuda`'s `strategy:` block does not set `fail-fast: false`, so GitHub's default matrix fail-fast cancelled this leg mid-test (submitting `test_inference_writes_outputs[msa-templates-ubiquitin]` to the ColabFold MSA server) once cuda13 failed — not a timeout-minutes cap, not superseded by a newer run (no other run in this concurrency group tonight), not manual. Same fail-fast mechanism as runs #254 (09-11), #266 (09-16), and #268 (09-17).

### Passing Jobs

None. Second night in this log (after 09-17) where all three tracked jobs are simultaneously non-passing.

---

## 2026-10-05

**Cause:** `unclassified (log unavailable)` for all three tracked jobs — third night in this log (after 09-17 and 09-19) where all three are simultaneously non-passing, and the first where the classification itself could not be completed. All three legs' setup steps (checkout, GHCR login, Docker Buildx, build-and-push test image, `start-aws-runner`, parameter cache) succeeded; each failed specifically at the "Run integration test" step. That shared failure point is suggestive of a common cause, but it could not be confirmed against any of this log's signatures (`aws-capacity`, `gpu-unavailable`, `runner-offline`, `msa-hang`, `parameter-cache`, `build-push`, `code`) because the job logs could not be fetched this session.

**Why the logs were unavailable:** every attempt to fetch the job logs — `gh api repos/aqlaboratory/openfold-3/actions/jobs/<id>/logs`, `gh run view --log`, and a direct `curl` on the Azure Blob Storage redirect target returned by the GitHub API — returned `403 Forbidden`. This session's egress-proxy status endpoint (`http://127.0.0.1:40399/__agentproxy/status`) confirms these are organization network-policy denials at the CONNECT level (`"connect_rejected"`, `"gateway answered 403 to CONNECT (policy denial or upstream failure)"`), not a GitHub-side error or an expired SAS signature:

```
"recentRelayFailures": [
  {"kind": "connect_rejected", "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)", "host": "productionresultssa1.blob.core.windows.net:443"},
  {"kind": "connect_rejected", "detail": "gateway answered 403 to CONNECT (policy denial or upstream failure)", "host": "results-receiver.actions.githubusercontent.com:443"}
]
```

Both hosts are where GitHub Actions redirects log downloads. Per this session's operating rules, a policy denial (403/407) at the proxy is not to be retried or routed around, so no further log-fetch attempts were made and this is recorded as `unclassified (log unavailable)` per this log's explicit fallback rule, rather than guessed at from the job metadata alone. **Flagged for human follow-up on two fronts:** (1) the underlying test failure itself, and (2) whether this network path needs to be allow-listed for this scanning session/environment so future nightly scans can classify failures again.

- **Run ID:** [37260402169](https://github.com/aqlaboratory/openfold-3/actions/runs/37260402169)
- **Run #:** 307
- **Branch:** main @ `1d48c84` (unchanged for 3 nights since run #303/10-03)
- **Time:** 2026-10-05T03:40:57Z – 04:04:37Z

### Failed Jobs

| Job | Status | Job ID | Failed step | Step window |
|-----|--------|--------|-------------|-------------|
| `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` | **failure** | 111606231484 | Run integration test | 03:41:17 – 03:52:51 |
| `test-pixi-cuda (openfold3-cuda12) / test-openfold-docker-pixi` | **failure** | 111606858956 | Run integration test | 03:45:11 – 04:04:28 |
| `test-pixi-cuda (openfold3-cuda13) / test-openfold-docker-pixi` | **failure** | 111606857931 | Run integration test | 03:45:01 – 04:04:33 |

### Failed Test

Log line unavailable — see "Why the logs were unavailable" above. No pytest output, assertion, or exception could be read for any of the three jobs this session.

### Passing Jobs

None. Third night in this log (after 09-17 and 09-19) where all three tracked jobs are simultaneously non-passing.

---

## 2026-10-06

**Cause:** `unclassified` for both CUDA legs — each hit the job-level `timeout-minutes: 60` cap mid-suite rather than failing on an assertion or exception, and neither matches this log's `msa-hang` signature closely enough to classify as such (see detail below). The AMD leg passed.

- **Run ID:** [37409527801](https://github.com/aqlaboratory/openfold-3/actions/runs/37409527801)
- **Run #:** 309 (primary nightly, schedule `17 3 * * *`)
- **Branch:** main @ `939ebd4b2b5e26f0b2e97cfc22291853f6be05a0` (PR #436 "Start --num_model_seeds from the configured seed instead of a fixed 42", merged since run #307–308/10-05)
- **Time:** 2026-10-06T03:33:20Z – 04:42:19Z

### Cancelled Jobs

| Job | Status | Job ID | Cancelled step | Step window | Job wall time |
|-----|--------|--------|-----------------|-------------|---------------|
| `test-pixi-cuda (openfold3-cuda12) / test-openfold-docker-pixi` | **cancelled** | 112095075518 | Run integration test | 03:40:07 – 04:35:32 | 60m28s (03:35:17–04:35:45) |
| `test-pixi-cuda (openfold3-cuda13) / test-openfold-docker-pixi` | **cancelled** | 112095193667 | Run integration test | 03:39:41 – 04:36:01 | 60m29s (03:35:46–04:36:15) |

Both job wall times match the `timeout-minutes: 60` cap set on `test-openfold-docker-pixi` in `ci-integration-test-pixi-cuda-reusable.yml` — this is a cap cancellation, not a fail-fast cascade (no sibling failure occurred to trigger one; `test-pixi-cuda`'s `strategy:` has no `fail-fast: false` but nothing failed) and not superseded by a newer run (each leg is the only run in its `${{ github.workflow }}-pixi-cuda-...-${{ matrix.pixi_env }}` concurrency group tonight).

### Classification detail (why neither is `msa-hang`)

This log's `msa-hang` signature requires the last pytest line to be a `test_inference_writes_outputs[msa-...]` case with no `PASSED`/`FAILED` after it, or a literal `TimeoutError: MSA server` string. Neither job's log matches:

- **cuda12** — last line before `##[error]The operation was canceled.` (04:35:32.8795952Z):
  ```
  openfold3/tests/inference/test_templates.py::test_template_lowers_rmsd[1y57]
  ```
  This is `test_template_lowers_rmsd`, not `test_inference_writes_outputs[msa-...]`, and no `TimeoutError: MSA server` string appears anywhere in the log.

- **cuda13** — last line before `##[error]The operation was canceled.` (04:36:01.9591332Z) is simply:
  ```
  PASSED
  ```
  (timestamped 04:32:37.4853688Z, for the previous test) followed by **no further output at all** for ~3m24s until the cancellation. No next-test node ID was ever printed — in every other test transition in both jobs' logs, the next node ID appears within single-digit seconds of the prior `PASSED`. This gap is anomalous and worth a human look, but it doesn't contain either literal signature string, so it is not binned as `msa-hang` on a guess.

Both are therefore recorded `unclassified` per this log's rule ("none of the above — flag as needing a human").

**Context, not cause:** earlier in the cuda12 job, a different (passing) test — `test_pocket_constraint_localizes_ligand`, which does call the ColabFold MSA server — spent ~10m44s rate-limited (`Sleeping for Ns. Reason: RATELIMIT`, ~80 occurrences, 03:21:18Z–03:32:01Z) before completing and passing. That test did not fail, but it shows the 60-minute per-job budget has little slack on nights when ColabFold responds slowly, which may be a contributing factor to why the suite ran out of time rather than evidence of a specific hang.

### Passing Jobs

| Job | Status | Job ID | Duration |
|-----|--------|--------|----------|
| `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` | **success** | 112094591348 | 47 min |

---

## 2026-10-07

**Cause:** `code` — `test_cuda_allocator.py::test_matches_snapshot` failed on both CUDA legs with `AssertionError: Values are not sufficiently close.` Both legs' logs carry the identical `UserWarning` (`openfold3/tests/conftest.py:261`) already seen on 2026-09-25/26 (run #288, off-schedule): `Snapshot environment mismatch in nvidia/: torch_version: stored=2.12.1, current=2.10.0, gpu_name: stored=NVIDIA GB10, current=NVIDIA A10G` — the stored allocator snapshot was captured on different hardware/torch version than the CI runner. Not an AWS GPU outage: `start-aws-runner` succeeded on both legs; the failure is isolated to pytest inside the "Run integration test" step. **New context this time:** main advanced from `939ebd4` (run #309/10-06) to `332d3a1`, carrying PR #414, "Sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` by default" (merged 2026-10-06T12:23:40Z) — a direct change to CUDA allocator behavior, the exact subsystem this snapshot test exercises. Classified `code` per this log's rules (pytest FAILED line + exception from test/library code), but flagged for a human to confirm whether PR #414 changed the allocator's actual byte-count behavior (a real regression) versus this being the same pre-existing stored-snapshot staleness as run #288 coincidentally recurring now that the test finally ran on CUDA again. First time `test_matches_snapshot` has failed on the regular nightly schedule (previously only seen on the off-schedule run #288).

- **Run ID:** [37567531088](https://github.com/aqlaboratory/openfold-3/actions/runs/37567531088)
- **Run #:** 311
- **Branch:** main @ `332d3a1` (PR #414, merged 2026-10-06T12:23:40Z)
- **Time:** 2026-10-07T03:36:27Z – 04:18:52Z

### Failed Jobs

| Job | Status | Job ID | Failed step | Duration |
|-----|--------|--------|--------------|----------|
| `test-pixi-cuda (openfold3-cuda12) / test-openfold-docker-pixi` | **failure** | 112619031745 | Run integration test | 23 min (03:38:25–04:01:43) |
| `test-pixi-cuda (openfold3-cuda13) / test-openfold-docker-pixi` | **failure** | 112619056940 | Run integration test | 22 min (03:38:31–04:00:15) |

### Failed Test (both jobs, identical signature)

```
FAILED openfold3/tests/test_cuda_allocator.py::test_matches_snapshot - AssertionError: Values are not sufficiently close.
```

```
openfold3/tests/test_cuda_allocator.py::test_matches_snapshot
  /opt/openfold3/openfold3/tests/conftest.py:261: UserWarning: Snapshot environment mismatch in nvidia/:
    torch_version: stored=2.12.1, current=2.10.0
    gpu_name: stored=NVIDIA GB10, current=NVIDIA A10G
  Snapshot tests may fail. Regenerate with: pytest --force-regen
```

### Passing Jobs

| Job | Status | Job ID | Duration |
|-----|--------|--------|----------|
| `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` | **success** | 112618559356 | 42 min |

---

## 2026-10-20

**Cause:** `unclassified` — both Brev legs failed at `start-brev-runner` / "Create Brev runner" because Brev's own CLI rejected the requested instance type: `Skipping: instance type "g5.4xlarge" is not a recognized type; run 'brev search' to see available types`. Not a capacity-exhaustion message (no `InsufficientInstanceCapacity`), not an AWS issue (both AWS `test-pixi-cuda` legs on the same run provisioned via `start-aws-runner` and passed normally), not GPU-busy, and not a runner going offline mid-queue — doesn't match any of this log's six defined signatures (`aws-capacity`, `gpu-unavailable`, `runner-offline`, `msa-hang`, `parameter-cache`, `build-push`, `code`). First non-pass for either Brev leg since their introduction 2026-10-14 (run #326), breaking a 6-night clean streak. The failure is in Brev's own instance-type catalog/provisioning, not in OF3 code or the test suite — `test-openfold-docker-pixi` never ran on either leg (both `skipped`, cascading from the `start-brev-runner` failure).

- **Run ID:** [39364702695](https://github.com/aqlaboratory/openfold-3/actions/runs/39364702695) (run #338, primary nightly, schedule `17 3 * * *`)
- **Branch:** main @ `7bebb7ac676d9ceb1510ac87d99f9803abf01f19` (PR #459, merged since run #336/337's `0d25195`)
- **Time:** 2026-10-20T03:31:03Z – 03:31:33Z (both Brev legs failed within ~15–20s of starting)

### Failed Jobs

| Job | Status | Job ID | Failed step |
|-----|--------|--------|--------------|
| `test-pixi-cuda-brev (openfold3-cuda12) / start-brev-runner` | **failure** | 118438752342 | Create Brev runner |
| `test-pixi-cuda-brev (openfold3-cuda13) / start-brev-runner` | **failure** | 118438752477 | Create Brev runner |

### Error (both legs, identical signature)

```
Attempting to create 1 instance(s) with 1 parallel attempts
Instance types to try: g5.4xlarge

Trying g5.4xlarge for 1 instance(s)...
Skipping: instance type "g5.4xlarge" is not a recognized type; run 'brev search' to see available types

Type g5.4xlarge had failures, trying next type...

Warning: Only created 0/1 instances
could only create 0/1 instances
subprocess.CalledProcessError: Command '['brev', 'create', '<runner-id>', '--type', 'g5.4xlarge', '--startup-script', '@/tmp/<tmpfile>.sh', '--timeout', '900', '--jupyter=false']' returned non-zero exit status 1.
```

(`start-brev-runner` input `brev_instance_type: g5.4xlarge`, same value used successfully every prior night since 10-14 — this appears to be a change on Brev's side, not a workflow-config regression that just happened to surface tonight.)

### Cascading skip

- `test-pixi-cuda-brev (openfold3-cuda12) / test-openfold-docker-pixi` (job ID: 118438852796) — **skipped**, no runner available
- `test-pixi-cuda-brev (openfold3-cuda13) / test-openfold-docker-pixi` (job ID: 118438894773) — **skipped**, no runner available

### Passing Jobs

| Job | Status | Job ID | Duration |
|-----|--------|--------|----------|
| `test-pixi-cuda (openfold3-cuda12) / test-openfold-docker-pixi` | **success** | 118439677412 | 40 min |
| `test-pixi-cuda (openfold3-cuda13) / test-openfold-docker-pixi` | **success** | 118439697678 | 36 min |
| `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` | **success** | 118438752790 | 52 min |

### Assessment

**Flagged for a human:** confirm whether Brev deprecated or renamed the `g5.4xlarge` SKU under the `OMSF-Brev` org (`brev search` would list current types) and update the `brev_instance_type` input in `.github/workflows/ci-integration-test-pixi-cuda-brev-reusable.yml` to match. Until fixed, both Brev legs will fail every night regardless of code changes on `main` — this is an infrastructure/config issue, not a regression to bisect. Also relevant to the pending two-week Brev-vs-AWS pass-rate comparison (due 2026-10-28, per the monitoring brief): this failure should be weighed as a Brev-catalog/config issue rather than counted at face value against Brev's underlying reliability when that report is written.

---

## 2026-10-21

**Cause:** Two unrelated non-passes the same night, first time this log has recorded all five tracked jobs non-passing simultaneously. (1) `code` — recurring `api.colabfold.com` MSA-submission read-timeout (same signature as 08-09, 08-11, 08-12, 09-08, and 09-19) hit all three non-Brev tracked jobs. (2) `unclassified` — both Brev legs failed a second consecutive night at `start-brev-runner` with the same "`g5.4xlarge` is not a recognized type" error first seen 10-20 (run #338), still unresolved.

- **Run ID:** [39539067117](https://github.com/aqlaboratory/openfold-3/actions/runs/39539067117)
- **Run #:** 340 (primary nightly, schedule `17 3 * * *`)
- **Branch:** main @ `b2b9f932c050f22568dcb843ef388f1dd8aceec9` (PR #467, dependabot bump of `prefix-dev/setup-pixi` 0.11.0→0.11.2, merged since run #338/339's `7bebb7a`)
- **Time:** 2026-10-21T03:31:44Z – 04:00:55Z

### ColabFold timeout (`code`) — all three non-Brev legs, identical signature

| Job | Job ID | Failed step | Duration |
|-----|--------|--------------|----------|
| `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` | 119022380282 | Run integration test | 14 min (03:37:06–03:46:05) |
| `test-pixi-cuda (openfold3-cuda12) / test-openfold-docker-pixi` | 119023227800 | Run integration test | 20 min (03:34:38–03:55:04) |
| `test-pixi-cuda (openfold3-cuda13) / test-openfold-docker-pixi` | 119023278489 | Run integration test | 16 min (03:34:49–03:50:51) |

```
FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin] - requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.
```

All three jobs show 6 retries against the ColabFold server (`openfold3/core/data/tools/colabfold_msa_server.py:279`/`283`, raised from `download()` at `colabfold_msa_server.py:269`), each read-timing-out, before pytest recorded the failure — identical to the pattern on 09-19. Both AWS CUDA legs' `start-aws-runner` and all four CUDA/Brev legs' parameter-cache steps succeeded; the failure is isolated to the ColabFold network call inside "Run integration test."

### Brev instance-type rejection (`unclassified`) — both Brev legs, second consecutive night

| Job | Job ID | Failed step |
|-----|--------|--------------|
| `test-pixi-cuda-brev (openfold3-cuda12) / start-brev-runner` | 119022380639 | Create Brev runner |
| `test-pixi-cuda-brev (openfold3-cuda13) / start-brev-runner` | 119022380796 | Create Brev runner |

```
Trying g5.4xlarge for 1 instance(s)...
Skipping: instance type "g5.4xlarge" is not a recognized type; run 'brev search' to see available types

Type g5.4xlarge had failures, trying next type...

Warning: Only created 0/1 instances
could only create 0/1 instances
subprocess.CalledProcessError: Command '['brev', 'create', 'runner-t8pcglck', '--type', 'g5.4xlarge', '--startup-script', '@/tmp/tmpwony48s8.sh', '--timeout', '900', '--jupyter=false']' returned non-zero exit status 1.
```

Identical signature to run #338 (10-20) — the `brev_instance_type: g5.4xlarge` input in `.github/workflows/ci-integration-test-pixi-cuda-brev-reusable.yml` is still rejected by Brev's own CLI, two nights running. Cascading `test-openfold-docker-pixi` skip on both legs (no runner provisioned, jobs never expanded).

### Cascading skip

- `test-pixi-cuda-brev (openfold3-cuda12) / test-openfold-docker-pixi` (job ID: 119022486605) — **skipped**, no runner available
- `test-pixi-cuda-brev (openfold3-cuda13) / test-openfold-docker-pixi` (job ID: 119022513755) — **skipped**, no runner available

### Passing Jobs

None. First night in this log where all five tracked jobs (three non-Brev + two Brev) are simultaneously non-passing.

### Assessment

**Still flagged for a human:** the Brev `g5.4xlarge` catalog issue has not been fixed since yesterday's entry — confirm the correct instance-type SKU via `brev search` against the `OMSF-Brev` org and update the workflow input. The ColabFold timeout is an external-service flake per this log's established pattern, not expected to need a code change; recommend watching the next clean night to confirm, same as every prior ColabFold-timeout entry.

---

## 2026-10-22

**Cause:** Three unrelated non-passes the same night. (1) `code` — recurring `api.colabfold.com` MSA-submission read-timeout (same signature as 09-19, 10-21, and others) hit `test-pixi-cuda (openfold3-cuda13)` and `test-pixi-amd (openfold3-rocm7)`, on two different tests. (2) matrix fail-fast — `test-pixi-cuda (openfold3-cuda12)` was cancelled when its cuda13 sibling failed. (3) `unclassified` — `test-pixi-cuda-brev (openfold3-cuda12)` ran to its 60-minute job timeout and was killed, with a suspicious silent gap before cancellation that does not quite match this log's `msa-hang` signature. Both Brev legs provisioned a runner successfully tonight for the first time since 10-19 — PR #469 (today's head commit) added `g6.4xlarge` as a fallback instance type, which appears to have resolved the 10-20/10-21 "`g5.4xlarge` is not a recognized type" rejection.

- **Run ID:** [39717308935](https://github.com/aqlaboratory/openfold-3/actions/runs/39717308935)
- **Run #:** 342 (primary nightly, schedule `17 3 * * *`)
- **Branch:** main @ `e242b404e0a4c811a4bbe5b29a0ab39a2f3edb82` (PR #469, "adds g6.4xlarge to allowed instance types for brev integration tests", merged since run #340/341's `b2b9f93`)
- **Time:** 2026-10-22T03:31:30Z – 04:45:08Z

### ColabFold timeout (`code`) — two legs, two different tests, same server signature

| Job | Job ID | Failed step | Duration |
|-----|--------|--------------|----------|
| `test-pixi-amd (openfold3-rocm7) / test-openfold-docker-pixi-amd` | 119622406560 | Run integration test | 23 min (03:31:49–03:54:21) |
| `test-pixi-cuda (openfold3-cuda13) / test-openfold-docker-pixi` | 119623170212 | Run integration test | 33 min (03:38:12–04:06:53) |

```
FAILED openfold3/tests/inference/test_pocket_constraints.py::test_pocket_constraint_localizes_ligand - requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.
FAILED openfold3/tests/inference/test_inference_full.py::test_inference_writes_outputs[msa-no_templates-ubiquitin] - requests.exceptions.ConnectionError: HTTPSConnectionPool(host='api.colabfold.com', port=443): Read timed out.
```

Both failures are `requests.exceptions.ConnectionError` read-timeouts against `api.colabfold.com`, same host and error class as every prior ColabFold entry in this log (08-09, 08-11, 08-12, 09-08, 09-19, 10-21), but on two different tests this time (`test_pocket_constraint_localizes_ligand` on AMD vs. `test_inference_writes_outputs[msa-no_templates-ubiquitin]` on cuda13) — consistent with a broader ColabFold-side degradation tonight rather than one specific query being slow.

### Matrix fail-fast cancellation — `test-pixi-cuda (openfold3-cuda12)`

| Job | Job ID | Cancelled step | Duration |
|-----|--------|--------------|----------|
| `test-pixi-cuda (openfold3-cuda12) / test-openfold-docker-pixi` | 119623131888 | Run integration test | 41 min (03:39:19–04:14:11) |

```
openfold3/tests/test_cuda_allocator.py::test_matches_snapshot PASSED
##[error]The operation was canceled.
```

`test-pixi-cuda`'s `strategy:` block (`.github/workflows/integration-test.yml`) carries no `fail-fast: false` override, so GitHub's default matrix fail-fast cancelled this leg once its cuda13 sibling failed (above) at 04:07:05Z — cancellation landed here at 04:14:11Z, ~7 minutes later, consistent with this log's historical fail-fast propagation delay (runs #268, #273). Not a timeout-minutes cap (only 41 of 60 minutes elapsed) and not superseded by a newer run (`list_workflow_runs` shows no other `workflow_dispatch`/`main` run in this concurrency group tonight).

### Timeout-cap cancellation, near-miss on `msa-hang` (`unclassified`) — `test-pixi-cuda-brev (openfold3-cuda12)`

| Job | Job ID | Cancelled step | Duration |
|-----|--------|--------------|----------|
| `test-pixi-cuda-brev (openfold3-cuda12) / test-openfold-docker-pixi` | 119624255440 | Run integration test | 60 min (03:58:48–04:37:59) |

```
openfold3/tests/test_cuda_allocator.py::test_matches_snapshot PASSED
##[error]The operation was canceled.
```

(identical last pytest line to the fail-fast case above, but on this job the gap before cancellation is ~8 minutes — 04:29:56Z to 04:37:59Z — and the job's total runtime (60m23s) exceeds `timeout-minutes: 60`, i.e. GitHub's own cap killed it, not fail-fast.) This log's `msa-hang` signature requires the last pytest line to be a hanging `test_inference_writes_outputs[msa-...]` nodeid, or a `TimeoutError: MSA server` line. Neither appears: the last printed nodeid is the same completed `test_cuda_allocator.py::test_matches_snapshot`, and whatever test ran next (if any did) never printed its own nodeid before the cap fired — pytest's `-vvv` normally flushes a test's nodeid immediately on start (confirmed from the same log's earlier lines), so its total absence here is itself unusual. Recording `unclassified` per this log's "don't guess" rule. **Flagged for a human:** check `openfold3/tests/test_cuda_allocator.py`'s position in the collected test order relative to `openfold3/tests/inference/test_inference_full.py` (MSA-dependent) to confirm whether an MSA-server call was in fact in flight when the cap hit.

### Passing jobs

| Job | Status | Job ID | Duration |
|-----|--------|--------|----------|
| `test-pixi-cuda-brev (openfold3-cuda13) / test-openfold-docker-pixi` | **success** | 119624327964 | 59 min |

### Assessment

The ColabFold timeouts are an external-service flake per this log's established pattern (08-09 through 10-21) — no code change expected; watch for the next clean night. The cuda12 AWS cancellation is fully explained by matrix fail-fast and needs no action. **Still flagged for a human:** the `test-pixi-cuda-brev (openfold3-cuda12)` cap-kill, which doesn't cleanly fit any defined signature — worth a second look once the next Brev `msa-hang`-shaped cancellation (if any) gives more log detail to compare against. Separately, good news: PR #469's `g6.4xlarge` fallback appears to have fixed the two-night Brev `g5.4xlarge` catalog rejection (10-20, 10-21) — both Brev legs provisioned a runner cleanly tonight.

---

---

## 2026-10-24

**Cause:** One non-pass class, one leg. `test-pixi-cuda (openfold3-cuda13)`'s own `start-aws-runner` succeeded (launched an instance in `us-east-2` on the second AZ retry), but the leg was cancelled by matrix fail-fast once its cuda12 sibling hit `aws-capacity` (full detail in `aws-outage-failures.md`) — the downstream `test-openfold-docker-pixi` step never started. Recorded here, not in `aws-outage-failures.md`, because the aws-capacity signature itself belongs to cuda12 only; cuda13's own non-pass is the fail-fast cascade, consistent with this log's established split (09-16, 09-17, 09-19, 10-22).

- **Run ID:** [40058531049](https://github.com/aqlaboratory/openfold-3/actions/runs/40058531049)
- **Run #:** 347 (primary nightly, schedule `17 3 * * *`)
- **Branch:** main @ `a83cb119f58a86306f9d66ca528ef2a0fffad250` (unchanged since run #345/346's `a83cb11`, 10-23)
- **Time:** 2026-10-24T03:29:51Z – 03:38:44Z

### Matrix fail-fast cancellation — `test-pixi-cuda (openfold3-cuda13)`

| Job | Job ID | Cancelled step | Duration |
|-----|--------|--------------|----------|
| `test-pixi-cuda (openfold3-cuda13) / test-openfold-docker-pixi` | 120767336406 | (never started — cancelled at dispatch) | 0 min (03:32:25–03:32:25) |

```
Launched in us-east-2
##[error]The operation was canceled.
```

`test-pixi-cuda`'s `strategy:` block (`.github/workflows/integration-test.yml`) carries no `fail-fast: false` override, so GitHub's default matrix fail-fast cancelled this leg's downstream test job the moment its cuda12 sibling's `start-aws-runner` failed at 03:31:45Z. Unlike every prior fail-fast cancellation in this log (runs #254, #266, #268, #273, #279, #340, #342 — all of which were cancelled mid-test, 5–7 minutes after the sibling's failure, or taken out before a runner was even provisioned), this is the first case where the cancelled leg's own infrastructure step (`start-aws-runner`) had already fully succeeded — `runner-o095hzqq` launched in `us-east-2` and the step reported `Launched in us-east-2` with every step conclusion `success` — before the job's overall `conclusion` still came back `cancelled` and the test step was killed before it could start. Not a timeout-minutes cap (the leg ran only ~2.5 minutes), not superseded by a newer run (no other run in this concurrency group tonight), not manual. The corresponding `stop-aws-runner` for this leg ran and succeeded normally (120767335401), tearing down the instance that had just been provisioned for a test that never ran.

### Passing jobs

| Job | Status | Job ID | Duration |
|-----|--------|--------|----------|
| `test-pixi-cuda-brev (openfold3-cuda12) / test-openfold-docker-pixi` | **success** | 120768228526 | 37 min |
| `test-pixi-cuda-brev (openfold3-cuda13) / test-openfold-docker-pixi` | **success** | 120768202969 | 34 min |

### Assessment

The fail-fast cancellation needs no code-side action — it's a direct consequence of cuda12's `aws-capacity` outage (see `aws-outage-failures.md`), and cuda13 would almost certainly have passed (its own runner had already launched cleanly). Both Brev legs passed clean on the unchanged commit, unaffected by the AWS capacity pool exhaustion. The AMD runner situation (now 2 consecutive dark nights, `test-pixi-amd (openfold3-rocm7)` queued since 2026-10-23T03:31:52Z with no runner ever picking it up) is tracked in `aws-outage-failures.md`'s `runner-offline` section, not here, but is the most urgent open item from tonight's scan.

---
