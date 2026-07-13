# SPDX-License-Identifier: Apache-2.0
"""KV-cache tuple slot indices for BF16 SFA sparse KV offload.

Offload five-tuple:
  [0] main_k  [1] main_v  [2] indexer_k  [3] resident_k  [4] resident_v
"""

OFFLOAD_MAIN_K = 0
OFFLOAD_MAIN_V = 1
OFFLOAD_INDEXER_K = 2
OFFLOAD_RESIDENT_K = 3
OFFLOAD_RESIDENT_V = 4
OFFLOAD_TUPLE_LEN = 5
