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

Any other failure is treated as a potential code regression and surfaces a
notification to the maintainer instead of being recorded here.

## Failures

| Date (UTC)  | Run ID                                                                                         | Failed test(s)                                | Signature                                                              |
| ----------- | ---------------------------------------------------------------------------------------------- | --------------------------------------------- | ---------------------------------------------------------------------- |
| 2026-06-09  | [27183771604](https://github.com/aqlaboratory/openfold-3/actions/runs/27183771604)             | `TestKernels::test_dsk_forward_bf16`          | `CUDA error: CUDA-capable device(s) is/are busy or unavailable`        |
| 2026-06-07  | [27082700605](https://github.com/aqlaboratory/openfold-3/actions/runs/27082700605)             | `TestKernels::test_dsk_forward_bf16`          | `CUDA error: CUDA-capable device(s) is/are busy or unavailable` (per user-supplied example) |

> Earlier failures (2026-05-26 → 2026-06-06) are visible in the GitHub Actions
> history but have not been individually triaged. Backfill on request.

## Weekly report

Ask "give me the weekly AWS-outage CI report" (or similar) and the assistant
will read this file plus the last 7 days of `integration-test.yml` runs and
produce a short summary: total runs, failed runs, share attributed to AWS
outages, and any non-outage failures that surfaced.
