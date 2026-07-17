# CI Failures Due to AWS Outages

This file records CI failures caused by AWS infrastructure issues (capacity, GPU unavailability, etc.) rather than OpenFold3 code defects.

| Date | Run ID | Branch | Error Type | Failed Jobs |
|------|--------|--------|------------|-------------|
| 2026-07-14 | [29305629949](https://github.com/aqlaboratory/openfold-3/actions/runs/29305629949) | main | InsufficientInstanceCapacity | test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml), test-pixi (openfold3-cuda13), test-pixi (openfold3-cuda12) |
| 2026-07-17 | [29554609928](https://github.com/aqlaboratory/openfold-3/actions/runs/29554609928) | main | InsufficientInstanceCapacity | test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml), test-pixi (openfold3-cuda13), test-pixi (openfold3-cuda12) |

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
