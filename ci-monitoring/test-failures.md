# CI Failures Potentially Related to OpenFold Code

This log records `integration-test.yml` failures that do **not** appear to
be caused by AWS infrastructure issues. These may indicate a code regression
or flaky test in the OpenFold-3 codebase and should be investigated by
maintainers.

Contrast with `aws-outage-failures.md`, which records failures attributed to
AWS GPU unavailability or EC2 capacity limits.

---

## Failures

| Date (UTC) | Run ID | Failed job | Failed test(s) | Error summary |
| --- | --- | --- | --- | --- |
| 2026-07-09 | [28994362612](https://github.com/aqlaboratory/openfold-3/actions/runs/28994362612) | `test-pixi (openfold3-cuda12) / test-openfold-docker-pixi` | `openfold3/tests/test_kernels.py::TestKernels::test_compare_template_stack_triton_bf16` | Numerical precision assertion failure: got `0.0234375`, expected `≤ 0.02` (absolute tolerance). 32 other tests passed. |
| 2026-06-21 | [27894030689](https://github.com/aqlaboratory/openfold-3/actions/runs/27894030689) | `test-pixi (openfold3-cuda13) / test-openfold-docker-pixi` | `openfold3/tests/test_kernels.py::TestKernels::test_compare_template_stack_triton_bf16` | Numerical precision assertion failure: got `0.021484375`, expected `≤ 0.02` (absolute tolerance). 32 other tests passed. |

---

## Detail

### 2026-07-09 — `test_compare_template_stack_triton_bf16` precision failure (cuda12)

**Run:** [28994362612](https://github.com/aqlaboratory/openfold-3/actions/runs/28994362612)  
**Job:** `test-pixi (openfold3-cuda12) / test-openfold-docker-pixi` ([job 86042054743](https://github.com/aqlaboratory/openfold-3/actions/runs/28994362612/job/86042054743))  
**Branch:** main  
**Environment:** `openfold3-cuda12` (CUDA 12 pixi environment)  
**Result:** 1 failed, 32 passed, 706 deselected — total runtime 9m 36s

**Error:**
```
FAILED openfold3/tests/test_kernels.py::TestKernels::test_compare_template_stack_triton_bf16
AssertionError: Scalars are not close!

Expected 0.0 but got 0.0234375.
Absolute difference: 0.0234375 (up to 0.02 allowed)
Relative difference: inf (up to 0.016 allowed)
```

Same test as the 2026-06-21 failure but now occurring in the `openfold3-cuda12`
environment (previously only seen on `cuda13`). The triton kernel numerical
precision is now exceeding the `eps=0.02` tolerance on both CUDA environments,
suggesting this may be a regression rather than a flaky bfloat16 rounding issue.

---

### 2026-06-21 — `test_compare_template_stack_triton_bf16` precision failure

**Run:** [27894030689](https://github.com/aqlaboratory/openfold-3/actions/runs/27894030689)  
**Job:** `test-pixi (openfold3-cuda13) / test-openfold-docker-pixi` ([job 82542481004](https://github.com/aqlaboratory/openfold-3/actions/runs/27894030689/job/82542481004))  
**Environment:** `openfold3-cuda13` (CUDA 13 pixi environment)  
**Result:** 1 failed, 32 passed, 646 deselected — total runtime 8m 17s

**Error:**
```
FAILED openfold3/tests/test_kernels.py::TestKernels::test_compare_template_stack_triton_bf16
AssertionError: Scalars are not close!

Expected 0.0 but got 0.021484375.
Absolute difference: 0.021484375 (up to 0.02 allowed)
Relative difference: inf (up to 0.016 allowed)
```

The test compares a triton kernel output against a reference, asserting the
max absolute difference is within `eps=0.02`. The result exceeded the
tolerance by ~7% (`0.021484375` vs. `0.02`). This is a borderline numerical
precision failure that could be flaky (bfloat16 rounding varies by GPU
microarchitecture) or could indicate a regression in the triton template
stack kernel. Worth checking whether this reproduces consistently on a
`cuda13` runner.
