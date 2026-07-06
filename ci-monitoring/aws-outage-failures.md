# CI Failures Due to AWS Outage / Infrastructure Issues

This log records integration test failures caused by AWS-side issues (GPU unavailability, vCPU quota limits, network timeouts) rather than OpenFold-3 code problems.

Workflow: [integration-test.yml](https://github.com/aqlaboratory/openfold-3/actions/workflows/integration-test.yml)

---

## 2026-07-06

**Run:** [#28768427331](https://github.com/aqlaboratory/openfold-3/actions/runs/28768427331) | Branch: `main` | Time: 04:45 UTC

**Error:** `VcpuLimitExceeded` — AWS EC2 vCPU limit of 64 exceeded for `g5.4xlarge` instance bucket.

**Failing jobs:**
- `test-pixi (openfold3-cuda13) / start-aws-runner`
- `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml) / start-aws-runner`

---

## 2026-07-05

**Run:** [#28729724398](https://github.com/aqlaboratory/openfold-3/actions/runs/28729724398) | Branch: `main` | Time: 04:40 UTC

**Error:** `VcpuLimitExceeded` — AWS EC2 vCPU limit of 64 exceeded for `g5.4xlarge` instance bucket.

**Failing jobs:**
- `test-pixi (openfold3-cuda12) / start-aws-runner`
- `test-pixi (openfold3-cuda13) / start-aws-runner`

---

## 2026-07-04

**Run:** [#28694885677](https://github.com/aqlaboratory/openfold-3/actions/runs/28694885677) | Branch: `main` | Time: 04:27 UTC

**Error:** `VcpuLimitExceeded` — AWS EC2 vCPU limit of 64 exceeded for `g5.4xlarge` instance bucket.

**Failing jobs:**
- `test-pixi (openfold3-cuda12) / start-aws-runner`
- `test-pixi (openfold3-cuda13) / start-aws-runner`

---

## 2026-07-03

**Run:** [#28638552070](https://github.com/aqlaboratory/openfold-3/actions/runs/28638552070) | Branch: `main` | Time: 04:31 UTC

**Error:** `VcpuLimitExceeded` — AWS EC2 vCPU limit of 64 exceeded for `g5.4xlarge` instance bucket.

**Failing jobs:**
- `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml) / start-aws-runner`
- `test-pixi (openfold3-cuda12) / start-aws-runner`

---

## 2026-07-02

**Run:** [#28565870069](https://github.com/aqlaboratory/openfold-3/actions/runs/28565870069) | Branch: `main` | Time: 04:40 UTC

**Error (conda):** Docker Hub network timeout pulling `python:3.12` — `dial tcp ... i/o timeout` (infrastructure network issue).
**Error (pixi):** `VcpuLimitExceeded` — AWS EC2 vCPU limit of 64 exceeded for `g5.4xlarge` instance bucket.

**Failing jobs:**
- `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml) / start-aws-runner` (Docker Hub timeout)
- `test-pixi (openfold3-cuda12) / start-aws-runner` (VcpuLimitExceeded)

---

## 2026-07-01

**Run:** [#28494335026](https://github.com/aqlaboratory/openfold-3/actions/runs/28494335026) | Branch: `main` | Time: 04:50 UTC

**Error:** `VcpuLimitExceeded` — AWS EC2 vCPU limit of 64 exceeded for `g5.4xlarge` instance bucket.

**Failing jobs:**
- `test-pixi (openfold3-cuda12) / start-aws-runner`
- `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml) / start-aws-runner`

---

## 2026-06-30

**Run:** [#28420797925](https://github.com/aqlaboratory/openfold-3/actions/runs/28420797925) | Branch: `main` | Time: 04:41 UTC

**Error:** `VcpuLimitExceeded` — AWS EC2 vCPU limit of 64 exceeded for `g5.4xlarge` instance bucket.

**Failing jobs:**
- `test-conda (12.1.1-cudnn8-devel-ubuntu22.04, yaml) / start-aws-runner`
- `test-pixi (openfold3-cuda13) / start-aws-runner`

---

## 2026-06-29

**Run:** [#28349485369](https://github.com/aqlaboratory/openfold-3/actions/runs/28349485369) | Branch: `main` | Time: 04:53 UTC

**Error:** `VcpuLimitExceeded` — AWS EC2 vCPU limit of 64 exceeded for `g5.4xlarge` instance bucket.

**Failing jobs:**
- `test-pixi (openfold3-cuda12) / start-aws-runner`
- `test-pixi (openfold3-cuda13) / start-aws-runner`

---

*Last updated by automated daily scan: 2026-07-06*
