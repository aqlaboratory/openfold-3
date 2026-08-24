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
