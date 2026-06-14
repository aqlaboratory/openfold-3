# Integration-Test CI Failures Attributed to AWS GPU Outages

This file is a running record of `integration-test.yml` workflow failures on
`aqlaboratory/openfold-3` that were classified as AWS GPU unavailability
(infrastructure outage), not OpenFold-3 code regressions. Maintained by the
daily CI scan run from a Claude Code on the web scheduled trigger.

## How a failure is classified as an AWS outage

A failed job is classified as an AWS GPU outage (and recorded here without
sending a notification) if any of the following error signatures appear in the
job logs:

- `CUDA error: CUDA-capable device(s) is/are busy or unavailable`
- `cudaErrorDevicesUnavailable`
- `torch.AcceleratorError: CUDA error` accompanied by a device-busy message
- `CUDA error: no CUDA-capable device is detected`
- `CUDA driver` / `nvidia-smi` failures before any test code runs
- `botocore.exceptions.ClientError: ... (InsufficientInstanceCapacity) when calling the RunInstances operation` — the `start-aws-gha-runner` action could not provision a GPU EC2 instance, so test code never ran

Any other failure is treated as a potential code regression and surfaces a
notification to the maintainer instead of being recorded here.

## Failures

| Date (UTC)  | Run ID                                                                                         | Failed test(s)                                | Signature                                                              |
| ----------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------- |
| 2026-06-14  | [27488342368](https://github.com/aqlaboratory/openfold-3/actions/runs/27488342368)             | `TestKernels::test_dsk_forward_bf16` (cuda13) | `torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable` (`cudaErrorDevicesUnavailable`) |
| 2026-06-13  | [27456584864](https://github.com/aqlaboratory/openfold-3/actions/runs/27456584864)             | all 3 `start-aws-runner` jobs (cuda12, cuda13, conda 12.1) — no test code ran | `botocore.exceptions.ClientError: An error occurred (InsufficientInstanceCapacity) when calling the RunInstances operation (reached max retries: 4): Insufficient capacity.` (g5.4xlarge in us-east-1) |
| 2026-06-12  | [27394591508](https://github.com/aqlaboratory/openfold-3/actions/runs/27394591508)             | `TestKernels::test_dsk_forward_bf16` (cuda12) | `torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable` (`cudaErrorDevicesUnavailable`) |
| 2026-06-11  | [27323934636](https://github.com/aqlaboratory/openfold-3/actions/runs/27323934636)             | `TestKernels::test_dsk_forward_bf16` (cuda13) | `CUDA error: CUDA-capable device(s) is/are busy or unavailable`        |
| 2026-06-10  | [27253274562](https://github.com/aqlaboratory/openfold-3/actions/runs/27253274562)             | `TestKernels::test_dsk_forward_bf16` (cuda12) | `CUDA error: CUDA-capable device(s) is/are busy or unavailable`        |
| 2026-06-09  | [27183771604](https://github.com/aqlaboratory/openfold-3/actions/runs/27183771604)             | `TestKernels::test_dsk_forward_bf16`          | `CUDA error: CUDA-capable device(s) is/are busy or unavailable`        |
| 2026-06-07  | [27082700605](https://github.com/aqlaboratory/openfold-3/actions/runs/27082700605)             | `TestKernels::test_dsk_forward_bf16`          | `CUDA error: CUDA-capable device(s) is/are busy or unavailable` (per user-supplied example) |
| 2026-06-06  | [27052461098](https://github.com/aqlaboratory/openfold-3/actions/runs/27052461098)             | `TestKernels::test_dsk_forward_bf16` (cuda12) | `torch.AcceleratorError: CUDA error: CUDA-capable device(s) is/are busy or unavailable` (`cudaErrorDevicesUnavailable`) |

> Earlier failures (2026-05-26 → 2026-06-05) are visible in the GitHub Actions
> history but have not been individually triaged. Backfill on request.

## Weekly report

Ask "give me the weekly AWS-outage CI report" (or similar) and the assistant
will read this file plus the last 7 days of `integration-test.yml` runs and
produce a short summary: total runs, failed runs, share attributed to AWS
outages, and any non-outage failures that surfaced.
