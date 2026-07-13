# SPDX-License-Identifier: Apache-2.0
"""KV-cache tuple slot indices for BF16 SFA sparse KV offload.

Registered offload five-tuple:
  [0] main_k  [1] main_v  [2] indexer placeholder
  [3] resident_k  [4] resident_v

For a real indexer owner, SFA forward composes the separately bound resident
indexer alias into slot 2. The connector only consumes slots 0/1 and 3/4.
"""

OFFLOAD_MAIN_K = 0
OFFLOAD_MAIN_V = 1
OFFLOAD_INDEXER_K = 2
OFFLOAD_RESIDENT_K = 3
OFFLOAD_RESIDENT_V = 4
OFFLOAD_TUPLE_LEN = 5
