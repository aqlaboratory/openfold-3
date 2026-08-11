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
