# CI Failures Due to AWS Outages

This file records CI failures caused by AWS infrastructure issues (capacity, GPU unavailability, etc.) rather than OpenFold3 code defects.

| Date | Run ID | Branch | Error Type | Failed Jobs |
|------|--------|--------|------------|-------------|
| 2026-07-14 | [29305629949](https://github.com/aqlaboratory/openfold-3/actions/runs/29305629949) | main | InsufficientInstanceCapacity | test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml), test-pixi (openfold3-cuda13), test-pixi (openfold3-cuda12) |
| 2026-07-17 | [29554609928](https://github.com/aqlaboratory/openfold-3/actions/runs/29554609928) | main | InsufficientInstanceCapacity | test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml), test-pixi (openfold3-cuda13), test-pixi (openfold3-cuda12) |
| 2026-07-21 | [29801095059](https://github.com/aqlaboratory/openfold-3/actions/runs/29801095059) | main | InsufficientInstanceCapacity | test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml), test-pixi-cuda (openfold3-cuda13), test-pixi-cuda (openfold3-cuda12) |
| 2026-09-09 | [34307353805](https://github.com/aqlaboratory/openfold-3/actions/runs/34307353805) | main | InsufficientInstanceCapacity | test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml), test-pixi-cuda (openfold3-cuda12), test-pixi-cuda (openfold3-cuda13) |
| 2026-09-09 | [34307353805](https://github.com/aqlaboratory/openfold-3/actions/runs/34307353805) | main | AMD runner — Docker build failure (non-AWS infra) | test-pixi-amd (openfold3-rocm7) |
| 2026-09-09 | [34311151668](https://github.com/aqlaboratory/openfold-3/actions/runs/34311151668) | main | AMD runner — Docker build failure (non-AWS infra) | test-pixi-amd (openfold3-rocm7) |

---

## Entries

### 2026-07-14 — Run [29305629949](https://github.com/aqlaboratory/openfold-3/actions/runs/29305629949)

- **Branch:** main
- **Scan date:** 2026-07-15
- **Error:** `botocore.exceptions.ClientError: An error occurred (InsufficientInstanceCapacity) when calling the RunInstances operation (reached max retries: 4): Insufficient capacity.`
- **Root cause:** AWS could not provision GPU EC2 instances (capacity exhaustion). All `start-aws-runner` steps failed; corresponding `stop-aws-runner` steps failed as a cascading consequence (no instances to stop).
- **Failed jobs:**
  - `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml)` — start-aws-runner
  - `test-pixi (openfold3-cuda13)` — start-aws-runner
  - `test-pixi (openfold3-cuda12)` — start-aws-runner

---

### 2026-07-17 — Run [29554609928](https://github.com/aqlaboratory/openfold-3/actions/runs/29554609928)

- **Branch:** main
- **Scan date:** 2026-07-17
- **Time:** 2026-07-17T04:20:27Z – 04:21:50Z UTC
- **Error:** `botocore.exceptions.ClientError: An error occurred (InsufficientInstanceCapacity) when calling the RunInstances operation (reached max retries: 4): Insufficient capacity.`
- **Root cause:** AWS could not provision GPU EC2 instances (capacity exhaustion in us-east-2). All `start-aws-runner` steps failed; corresponding `stop-aws-runner` and test jobs failed/were skipped as a cascading consequence.
- **Failed jobs (start-aws-runner):**
  - `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml)` — start-aws-runner (job 87804052095)
  - `test-pixi (openfold3-cuda13)` — start-aws-runner (job 87804052113)
  - `test-pixi (openfold3-cuda12)` — start-aws-runner (job 87804052127)
- **Cascading failures (stop-aws-runner — no instance to stop):**
  - `test-pixi (openfold3-cuda13)` — stop-aws-runner (job 87804142583)
  - `test-pixi (openfold3-cuda12)` — stop-aws-runner (job 87804147478)
  - `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml)` — stop-aws-runner (job 87804148890)
- **Skipped (no runner available):**
  - `test-pixi (openfold3-cuda13)` — test-openfold-docker-pixi
  - `test-pixi (openfold3-cuda12)` — test-openfold-docker-pixi
  - `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml)` — test-openfold-docker

---

### 2026-07-21 — Run [29801095059](https://github.com/aqlaboratory/openfold-3/actions/runs/29801095059)

- **Branch:** main
- **Scan date:** 2026-07-21
- **Time:** 2026-07-21T04:22:58Z – 04:24:53Z UTC
- **Error:** `botocore.exceptions.ClientError: An error occurred (InsufficientInstanceCapacity) when calling the RunInstances operation (reached max retries: 4): Insufficient capacity.`
- **Root cause:** AWS could not provision GPU EC2 instances (capacity exhaustion in us-east-2). All `start-aws-runner` steps failed; corresponding `stop-aws-runner` and test jobs failed/were skipped as a cascading consequence. AMD GPU job (`test-pixi-amd openfold3-rocm7`) succeeded on separate non-AWS runner.
- **Failed jobs (start-aws-runner):**
  - `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml)` — start-aws-runner (job 88542175000)
  - `test-pixi-cuda (openfold3-cuda13)` — start-aws-runner (job 88542174989)
  - `test-pixi-cuda (openfold3-cuda12)` — start-aws-runner (job 88542174977)
- **Cascading failures (stop-aws-runner — no instance to stop):**
  - `test-pixi-cuda (openfold3-cuda12)` — stop-aws-runner (job 88542258038)
  - `test-pixi-cuda (openfold3-cuda13)` — stop-aws-runner (job 88542263150)
  - `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml)` — stop-aws-runner (job 88542296059)
- **Skipped (no runner available):**
  - `test-pixi-cuda (openfold3-cuda12)` — test-openfold-docker-pixi
  - `test-pixi-cuda (openfold3-cuda13)` — test-openfold-docker-pixi
  - `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml)` — test-openfold-docker

---

### 2026-09-09 — Run [34307353805](https://github.com/aqlaboratory/openfold-3/actions/runs/34307353805) (run #250)

- **Branch:** main
- **Scan date:** 2026-09-09
- **Time:** 2026-09-09T03:30:02Z – 03:35:58Z UTC
- **Error:** `botocore.exceptions.ClientError: An error occurred (InsufficientInstanceCapacity) when calling the RunInstances operation (reached max retries: 4): Insufficient capacity.`
- **Root cause:** AWS could not provision GPU EC2 instances (capacity exhaustion in us-east-2). All `start-aws-runner` steps failed; corresponding `stop-aws-runner` and test jobs failed/were skipped as a cascading consequence.
- **Failed jobs (start-aws-runner):**
  - `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml)` — start-aws-runner (job 102326710443)
  - `test-pixi-cuda (openfold3-cuda12)` — start-aws-runner (job 102326710500)
  - `test-pixi-cuda (openfold3-cuda13)` — start-aws-runner (job 102326710522)
- **Cascading failures (stop-aws-runner — no instance to stop):**
  - `test-pixi-cuda (openfold3-cuda13)` — stop-aws-runner (job 102326856612)
  - `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml)` — stop-aws-runner (job 102326873473)
  - `test-pixi-cuda (openfold3-cuda12)` — stop-aws-runner (job 102326874024)
- **Skipped (no runner available):**
  - `test-pixi-cuda (openfold3-cuda13)` — test-openfold-docker-pixi
  - `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml)` — test-openfold-docker
  - `test-pixi-cuda (openfold3-cuda12)` — test-openfold-docker-pixi
- **Additional failure (self-hosted AMD runner):**
  - `test-pixi-amd (openfold3-rocm7)` (job 102326710252) — failed at "Build and push test image" step on `omsf-amd-aupcloud` runner; logs not available (HTTP 404). Not AWS-related.

---

### 2026-09-09 — Run [34311151668](https://github.com/aqlaboratory/openfold-3/actions/runs/34311151668) (run #251)

- **Branch:** main
- **Scan date:** 2026-09-09
- **Time:** 2026-09-09T04:29:14Z – 04:31:13Z UTC
- **Note:** AWS CUDA/conda jobs were skipped in this run (no `start-aws-runner` step launched). Only the AMD self-hosted runner job ran and failed.
- **Failed jobs:**
  - `test-pixi-amd (openfold3-rocm7)` (job 102337915477) — failed at "Build and push test image" step on `omsf-amd-aupcloud` runner; logs not available (HTTP 404). Not AWS-related; consecutive AMD Docker build failure.
- **Skipped:**
  - `test-pixi-cuda` — skipped (no AWS runner launched)
  - `test-conda` — skipped (no AWS runner launched)
