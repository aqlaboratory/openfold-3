# Integration-Test CI Failures Attributed to AWS GPU Outages

This file is a running record of `integration-test.yml` workflow failures on
`aqlaboratory/openfold-3` that were classified as AWS infrastructure outages
(GPU unavailable, or AWS could not provision the EC2 instance at all), not
OpenFold-3 code regressions. Maintained by the daily CI scan run from a
Claude Code on the web scheduled trigger.

## How a failure is classified as an AWS outage

A failed job is classified as an AWS outage (and recorded here without
sending a notification) if any of the following error signatures appear in
the job logs:

- **GPU-busy (in the test job, after the runner is up)**:
  - `CUDA error: CUDA-capable device(s) is/are busy or unavailable`
  - `cudaErrorDevicesUnavailable`
  - `torch.AcceleratorError: CUDA error` accompanied by a device-busy message
  - `CUDA error: no CUDA-capable device is detected`
  - `CUDA driver` / `nvidia-smi` failures before any test code runs
- **EC2-capacity (in the `start-aws-runner` job)**:
  - `botocore.exceptions.ClientError: ... (InsufficientInstanceCapacity) when calling the RunInstances operation` — `start-aws-gha-runner` could not provision the GPU EC2 instance, so test code never ran.
  - `botocore.exceptions.ClientError: ... (VcpuLimitExceeded) when calling the RunInstances operation` — vCPU quota exhausted, instance could not be provisioned.
  - Other `botocore.exceptions.ClientError` raised by `start-aws-gha-runner`
    before any GitHub runner registers.

Any other failure is treated as a potential code regression and surfaces a
notification to the maintainer instead of being recorded here.

## Failures

`Time to fail` = `failing step end` − `failing job start` (i.e. wall clock
on the *failing* job only, excluding the `start-aws-runner` preamble). It
approximates the GPU instance time that a successful early-failure smoke
check could save per outage — the AWS instance is alive across the whole
window.

Each test matrix entry (`test-conda`, `test-pixi (cuda12)`, `test-pixi
(cuda13)`) runs on its own AWS GPU runner, so multiple failures within a
single run are independent events on different instances. Test jobs that
were *cancelled* by matrix `fail-fast` (rather than failing on their own
GPU) are omitted; they don't tell us whether that runner was healthy.

| Date (UTC) | Run ID | Failed job(s) | Time to fail | Signature |
| --- | --- | --- | --- | --- |
| 2026-06-24 | [28075708064](https://github.com/aqlaboratory/openfold-3/actions/runs/28075708064) | `start-aws-runner` × 2 (test-pixi openfold3-cuda13, test-conda) | 0m 02s | EC2 `VcpuLimitExceeded` for `g5.4xlarge` in `us-east-2` — vCPU quota of 64 exhausted. No GPU runner provisioned; `test-pixi cuda12` cancelled by fail-fast. |
| 2026-06-23 | [28002658928](https://github.com/aqlaboratory/openfold-3/actions/runs/28002658928) | `start-aws-runner` × 2 (test-pixi cuda12, test-pixi cuda13) | 0m 02s | EC2 `VcpuLimitExceeded` for `g5.4xlarge` in `us-east-2` — vCPU quota of 64 exhausted. No GPU runner provisioned for pixi jobs. `test-conda` succeeded. |
| 2026-06-22 | [27930489254](https://github.com/aqlaboratory/openfold-3/actions/runs/27930489254) | `start-aws-runner` × 3 (test-conda, test-pixi cuda12, test-pixi cuda13) | 0m 02s | EC2 `VcpuLimitExceeded` for `g5.4xlarge` in `us-east-2` — vCPU quota of 64 exhausted. No GPU runner ever provisioned. |
| 2026-06-19 | [27806279583](https://github.com/aqlaboratory/openfold-3/actions/runs/27806279583) | `start-aws-runner` × 3 (test-conda, test-pixi cuda12, test-pixi cuda13) | 0m 01s (all three) | `botocore.exceptions.ClientError: (InvalidAMIID.NotFound) ... The image id '[ami-00839c71d8f6096b4]' does not exist` — AMI referenced by workflow has been deleted/deregistered in AWS. No GPU runner ever provisioned. New signature class (AMI deletion vs. capacity exhaustion); will recur every run until the workflow's AMI reference is updated. |
| 2026-06-16 | [27595142555](https://github.com/aqlaboratory/openfold-3/actions/runs/27595142555) | `test-conda` | 5m 24s | `CUDA-capable device(s) is/are busy or unavailable` in `TestKernels::test_dsk_forward_bf16` |
| 2026-06-14 | [27488342368](https://github.com/aqlaboratory/openfold-3/actions/runs/27488342368) | `test-pixi (cuda13)` | 4m 27s | `cudaErrorDevicesUnavailable` in `TestKernels::test_dsk_forward_bf16` |
| 2026-06-13 | [27456584864](https://github.com/aqlaboratory/openfold-3/actions/runs/27456584864) | `start-aws-runner` × 3 (test-conda, test-pixi cuda12, test-pixi cuda13) | 0m 09s (all three) | EC2 `InsufficientInstanceCapacity` for `g5.4xlarge` in `us-east-1` (no GPU runner ever provisioned) |
| 2026-06-12 | [27394591508](https://github.com/aqlaboratory/openfold-3/actions/runs/27394591508) | `test-pixi (cuda12)` | 4m 56s | `torch.AcceleratorError: cudaErrorDevicesUnavailable` |
| 2026-06-11 | [27323934636](https://github.com/aqlaboratory/openfold-3/actions/runs/27323934636) | `test-pixi (cuda13)` | 4m 28s | `CUDA-capable device(s) is/are busy or unavailable` |
| 2026-06-10 | [27253274562](https://github.com/aqlaboratory/openfold-3/actions/runs/27253274562) | `test-pixi (cuda12)` | 5m 04s | `CUDA-capable device(s) is/are busy or unavailable` |
| 2026-06-09 | [27183771604](https://github.com/aqlaboratory/openfold-3/actions/runs/27183771604) | `test-pixi (cuda13)` | 4m 27s | `CUDA-capable device(s) is/are busy or unavailable` |
| 2026-06-07 | [27082700605](https://github.com/aqlaboratory/openfold-3/actions/runs/27082700605) | `test-conda` | 5m 09s | `CUDA-capable device(s) is/are busy or unavailable` |
| 2026-06-06 | [27052461098](https://github.com/aqlaboratory/openfold-3/actions/runs/27052461098) | `test-pixi (cuda12)` | 5m 06s | `torch.AcceleratorError: cudaErrorDevicesUnavailable` |

> Earlier failures (2026-05-26 → 2026-06-05) are visible in the GitHub
> Actions history but have not been individually triaged. Backfill on
> request.

## Smoke-test effectiveness (running record)

The pre-build `GPU smoke check` step (host `nvidia-smi` + container
`docker run --gpus all ... nvidia-smi`) was first exercised on a feature
branch on 2026-06-13. Tracking whether it catches the same outages the
integration test catches:

| Date (UTC) | Run | Failed job(s) | Smoke result | Test result | Effective? |
| --- | --- | --- | --- | --- | --- |
| 2026-06-13 | [27456889723](https://github.com/aqlaboratory/openfold-3/actions/runs/27456889723) (feature branch `infra/add-smoketest-to-workflow`) | `test-conda`, `test-pixi (cuda13)` (sibling `test-pixi (cuda12)` passed) | passed in 11s on all three runners | `test-conda` failed at 5m 26s, `test-pixi (cuda13)` failed at 4m 27s — both with `cudaErrorDevicesUnavailable` | **No** — two of the three runners passed NVML/container smoke yet still hit the failure inside pytest. The CUDA-runtime context-creation path isn't exercised by `nvidia-smi`. |

### Cost/benefit as of 2026-06-15

- Per-outage **upper bound** on GPU time saved by a perfectly effective
  smoke check: ~4m 30s – 5m 10s of the test job, plus avoided
  `stop-aws-runner` ramp (which still runs either way, so net is just the
  test-job window).
- Smoke check **overhead** on every healthy run: ~11s × (number of test
  jobs per run). At three test jobs/run × ~365 runs/year ≈ ~3.3 hours/year
  of added wall time.
- Observed smoke check **catch rate** so far: 0/1 (the one outage that hit
  the smoke-enabled branch slipped past it).
- Failure-mode coverage: smoke catches the *NVIDIA Container Runtime
  injection / driver visibility* class of failure. It does not catch the
  *CUDA runtime can't create a context* class, which is the class we have
  actually been seeing. It also doesn't catch the EC2 capacity class
  (which already fails fast in the runner-start job in ~10s, so no
  intervention needed).

Given the above, the smoke check as currently designed is probably not
worth keeping unless we start seeing failures of the class it would
catch. Revisit before/after the next planned AMI bump or AWS-runner
action upgrade.

## Weekly report

Ask "give me the weekly AWS-outage CI report" (or similar) and the
assistant will read this file plus the last 7 days of `integration-test.yml`
runs and produce a short summary: total runs, failed runs, share
attributed to AWS outages, and any non-outage failures that surfaced.
