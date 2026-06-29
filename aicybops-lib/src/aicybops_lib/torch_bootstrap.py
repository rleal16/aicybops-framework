"""
Configure PyTorch before ``import torch``.

- **CPU deployment (default):** disable Dynamo/Triton/CUDA probing so training
  does not segfault on hosts without an NVIDIA driver.
- **GPU deployment:** build with ``TORCH_VARIANT=cuda`` and set ``DAM_DEVICE=cuda``;
  this module leaves the environment unchanged when ``AICYBOPS_TORCH_VARIANT=cuda``.
"""

from __future__ import annotations

import os

_VARIANT = os.environ.get("AICYBOPS_TORCH_VARIANT", "cpu").strip().lower()

_CPU_ONLY_DEFAULTS = {
    "TORCHDYNAMO_DISABLE": "1",
    "TORCH_COMPILE_DISABLE": "1",
    "PYTORCH_DISABLE_AOT_AUTOGRAD": "1",
    "CUDA_VISIBLE_DEVICES": "",
    "TRITON_DISABLE_LINE_INFO": "1",
}

if _VARIANT != "cuda":
    for _key, _value in _CPU_ONLY_DEFAULTS.items():
        os.environ[_key] = _value
    os.environ.setdefault("DAM_DEVICE", "cpu")
