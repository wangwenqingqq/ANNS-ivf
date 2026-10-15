#!/usr/bin/env python3
"""Generate count-only instrumentation without editing upstream files."""
from pathlib import Path
import sys

root, output = map(Path, sys.argv[1:])
source = (root / "faiss/gpu/impl/SIVFAppend.cu").read_text()


def replace_once(old, new):
    global source
    assert source.count(old) == 1, old
    source = source.replace(old, new)


replace_once("namespace gpu {", "namespace gpu {\n__device__ unsigned long long audit_counts[5];")
replace_once("    atomicOr(\n", "    atomicAdd(&audit_counts[3], 1ULL);\n    atomicOr(\n")
replace_once("            return; // OOM:", "            atomicAdd(&audit_counts[4], 1ULL);\n            return; // OOM:")
replace_once("        int new_slab = manager.free_list[free_idx - 1];", "        atomicAdd(&audit_counts[1], 1ULL);\n        int new_slab = manager.free_list[free_idx - 1];")
replace_once("        // TODO Failure Handling:", "        atomicAdd(&audit_counts[0], 1ULL);\n        // TODO Failure Handling:")
replace_once("    }\n}\n\n/**\n * @brief Launches", "    }\n    atomicAdd(&audit_counts[4], 1ULL);\n}\n\n/**\n * @brief Launches")
output.write_text(source)
