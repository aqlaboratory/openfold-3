# Copyright 2026 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Validation script for Intel XPU inference with OpenFold3.

Run after installing openfold3 on a system with an Intel GPU to verify that
the environment is correctly configured for the Triton kernels:

    validate-openfold3-xpu
"""

import sys


def _check(label: str, ok: bool, detail: str = "") -> bool:
    status = "PASS" if ok else "FAIL"
    line = f"  [{status}] {label}"
    if detail:
        line += f": {detail}"
    print(line)
    return ok


def main() -> None:
    print("OpenFold3 Intel XPU environment check\n")
    all_ok = True

    # 1. PyTorch importable
    try:
        import torch

        torch_version = str(torch.__version__)
        torch_ok = True
    except ImportError:
        torch_version = "not found"
        torch_ok = False
    all_ok &= _check("PyTorch installed", torch_ok, torch_version)

    if not torch_ok:
        print("\nInstall PyTorch for Intel XPU first:")
        print(
            "  pip install torch torchvision torchaudio"
            " --index-url https://download.pytorch.org/whl/xpu"
        )
        sys.exit(1)

    # 2. PyTorch built with XPU support
    xpu_module_ok = hasattr(torch, "xpu")
    all_ok &= _check(
        "PyTorch built with XPU support",
        xpu_module_ok,
        "torch.xpu module present" if xpu_module_ok else "torch has no xpu module",
    )

    if not xpu_module_ok:
        print("\nRe-install PyTorch for Intel XPU:")
        print(
            "  pip install torch torchvision torchaudio"
            " --index-url https://download.pytorch.org/whl/xpu"
        )
        sys.exit(1)

    # 3. XPU GPU visible
    gpu_ok = torch.xpu.is_available()
    device_name = torch.xpu.get_device_name(0) if gpu_ok else "none"
    all_ok &= _check("Intel XPU device visible", gpu_ok, device_name)

    # 4. Triton importable
    try:
        import triton

        triton_version = triton.__version__
        triton_ok = True
    except ImportError:
        triton_version = "not found"
        triton_ok = False
    all_ok &= _check("Triton installed", triton_ok, triton_version)

    if not triton_ok:
        print("\nTriton (pytorch-triton-xpu) should be bundled with the XPU wheel.")
        print("Re-install PyTorch for Intel XPU:")
        print(
            "  pip install torch torchvision torchaudio"
            " --index-url https://download.pytorch.org/whl/xpu"
        )
        sys.exit(1)

    # 5. Triton backend targets XPU
    try:
        import triton.runtime

        backend = triton.runtime.driver.active.get_current_target().backend
        xpu_backend_ok = backend == "xpu"
        all_ok &= _check("Triton backend is XPU", xpu_backend_ok, backend)
    except Exception as e:
        all_ok &= _check("Triton backend is XPU", False, str(e))

    # 6. OpenFold3 Triton evoformer kernel loads
    try:
        from openfold3.core.kernels.triton.evoformer import TritonEvoformer

        kernel_ok = TritonEvoformer is not None
        all_ok &= _check("Triton evoformer kernel loaded", kernel_ok)
    except Exception as e:
        all_ok &= _check("Triton evoformer kernel loaded", False, str(e))

    # 7. PyTorch Lightning XPU accelerator plugin registers correctly
    try:
        from pytorch_lightning.accelerators import AcceleratorRegistry

        import openfold3.core.utils.xpu_accelerator  # noqa: F401 - self-registers

        accelerator_ok = "xpu" in AcceleratorRegistry
        all_ok &= _check("Lightning XPUAccelerator registered", accelerator_ok)
    except Exception as e:
        all_ok &= _check("Lightning XPUAccelerator registered", False, str(e))

    # Summary
    print()
    if all_ok:
        print(
            "All checks passed. OpenFold3 Intel XPU inference is correctly configured."
        )
    else:
        print("One or more checks failed. See above for details.")
        print(
            "Installation instructions: "
            "https://github.com/aqlaboratory/openfold-3/blob/main/docs/source/Installation.md"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
