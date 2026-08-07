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
