#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#
"""Mock heavy dependencies (torch, vllm, etc.) for ascend_store unit tests.

IMPORTANT: This module MUST be imported before any vllm_ascend or vllm
imports in each test file.

Usage at the top of each test file:
    import tests.ut.distributed.ascend_store._mock_deps  # noqa: F401, E402
"""

import importlib
import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock

# ---------------------------------------------------------------------------
# Mock torch / torch_npu
# ---------------------------------------------------------------------------
if "torch" not in sys.modules and importlib.util.find_spec("torch") is None:
    _torch = types.ModuleType("torch")
    _torch.Tensor = MagicMock  # type: ignore[attr-defined]
    _torch.bool = "bool"  # type: ignore[attr-defined]
    _torch.float16 = "float16"  # type: ignore[attr-defined]
    _torch.float32 = "float32"  # type: ignore[attr-defined]
    _torch.int8 = "int8"  # type: ignore[attr-defined]
    _torch.dtype = str  # type: ignore[attr-defined]
    _torch.zeros = MagicMock(return_value=MagicMock())  # type: ignore[attr-defined]
    _torch.empty = MagicMock()  # type: ignore[attr-defined]
    _torch.uint8 = "uint8"  # type: ignore[attr-defined]
    _torch.sum = MagicMock(return_value=0)  # type: ignore[attr-defined]
    _torch.device = MagicMock()  # type: ignore[attr-defined]
    _torch.distributed = MagicMock()  # type: ignore[attr-defined]
    _npu = MagicMock()
    _npu.Event = MagicMock
    _npu.current_device = MagicMock(return_value=0)
    _npu.set_device = MagicMock()
    _torch.npu = _npu  # type: ignore[attr-defined]
    sys.modules["torch"] = _torch
    sys.modules["torch.distributed"] = _torch.distributed  # type: ignore[attr-defined]

_torch_module = sys.modules.get("torch") or importlib.import_module("torch")
if not hasattr(_torch_module, "npu"):
    _torch_module.npu = MagicMock()  # type: ignore[attr-defined]

if "torch_npu" not in sys.modules:
    sys.modules["torch_npu"] = MagicMock()
    sys.modules["torch_npu._inductor"] = MagicMock()

# ---------------------------------------------------------------------------
# Mock external backends
# ---------------------------------------------------------------------------
for _mod_name in [
    "mooncake",
    "mooncake.engine",
    "mooncake.store",
    "memcache_hybrid",
    "yr",
    "yr.datasystem",
    "yr.datasystem.hetero_client",
    "yr.datasystem.kv_client",
    "yr.datasystem.object_client",
    "zmq",
]:
    if _mod_name not in sys.modules and (_mod_name != "zmq" or importlib.util.find_spec("zmq") is None):
        sys.modules[_mod_name] = MagicMock()

# ---------------------------------------------------------------------------
# Mock vllm_ascend transitive imports
# ---------------------------------------------------------------------------


def _make_pkg(name, path=""):
    mod = types.ModuleType(name)
    mod.__path__ = [path]  # type: ignore[attr-defined]
    mod.__package__ = name  # type: ignore[attr-defined]
    return mod


_vllm_ascend_real_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "vllm_ascend"))
_vllm_ascend_package_paths = {
    "vllm_ascend": _vllm_ascend_real_path,
    "vllm_ascend.distributed": os.path.join(_vllm_ascend_real_path, "distributed"),
}
for _pkg, _path in _vllm_ascend_package_paths.items():
    if _pkg not in sys.modules:
        sys.modules[_pkg] = _make_pkg(_pkg, _path)

_distributed_utils = types.ModuleType("vllm_ascend.distributed.utils")
_distributed_utils.all_gather_async = MagicMock()  # type: ignore[attr-defined]
_distributed_utils.get_decode_context_model_parallel_rank = MagicMock(  # type: ignore[attr-defined]
    return_value=0
)
_distributed_utils.get_decode_context_model_parallel_world_size = MagicMock(  # type: ignore[attr-defined]
    return_value=1
)
sys.modules["vllm_ascend.distributed.utils"] = _distributed_utils

_kv_transfer_real_path = os.path.join(_vllm_ascend_real_path, "distributed", "kv_transfer")
importlib.import_module("vllm_ascend.distributed.kv_transfer")

_kv_pool_pkg = _make_pkg(
    "vllm_ascend.distributed.kv_transfer.kv_pool",
    os.path.join(_kv_transfer_real_path, "kv_pool"),
)
sys.modules["vllm_ascend.distributed.kv_transfer.kv_pool"] = _kv_pool_pkg

_ascend_store_real_path = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "vllm_ascend",
    "distributed",
    "kv_transfer",
    "kv_pool",
    "ascend_store",
)
_ascend_store_pkg = _make_pkg(
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store",
    os.path.abspath(_ascend_store_real_path),
)
sys.modules["vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store"] = _ascend_store_pkg

_backend_pkg = _make_pkg(
    "vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend",
    os.path.join(os.path.abspath(_ascend_store_real_path), "backend"),
)
sys.modules["vllm_ascend.distributed.kv_transfer.kv_pool.ascend_store.backend"] = _backend_pkg

# Execute the real backend/__init__.py inside the stub package so that
# backend_map always stays in sync with the source of truth. The stub
# package shadows the real package for these tests; without this, any
# ``from ...backend import backend_map`` in the real modules would fail
# with ImportError under the stub. (Submodules such as
# backend.memcache_backend resolve through the stub package's __path__ and
# need no special casing.)
# Scheduler tests mock create_scheduler_client on real backend classes;
# worker tests patch pool_worker.importlib locally. Do NOT install fake
# backend modules in sys.modules or globally wrap import_module here:
# test_backend.py imports the real backend classes and also relies on
# ``mock.patch`` (which itself calls importlib.import_module) resolving
# those real modules.
_backend_init_path = os.path.join(_backend_pkg.__path__[0], "__init__.py")  # type: ignore[attr-defined]
with open(_backend_init_path, encoding="utf-8") as _backend_init_file:
    exec(compile(_backend_init_file.read(), _backend_init_path, "exec"), vars(_backend_pkg))  # type: ignore[arg-type]

if "vllm_ascend.utils" not in sys.modules or not hasattr(sys.modules["vllm_ascend.utils"], "AscendDeviceType"):
    _ascend_utils = MagicMock()
    _ascend_utils.AscendDeviceType = MagicMock()
    _ascend_utils.get_ascend_device_type = MagicMock()
    sys.modules["vllm_ascend.utils"] = _ascend_utils

# NOTE: vllm_ascend.{ascend_config, attention_fence} and their helpers
# (get_ascend_config, AttentionComputeStartGate, ...) are intentionally NOT
# mocked here. Doing so by mutating these real modules leaks into every other
# UT in the same pytest session (breaking test_ascend_config / test_platform,
# which collect after ascend_store and bind the polluted symbols at import).
# These helpers are mocked per-test, scoped to the ascend_store tests only,
# via the autouse fixture in tests/ut/conftest.py.
