#!/usr/bin/env python3
"""Inject a bounded scheduling pause, preserving the original free-stack operations."""
from pathlib import Path
import sys

root, destination = map(Path, sys.argv[1:])
source = (root / "faiss/gpu/impl/SlabManager.cuh").read_text()
assert source.count("namespace gpu {") == 1
source = source.replace("namespace gpu {", "namespace gpu {\n__device__ int audit_release_ready, audit_release_resume;")
old = "        int old_top = atomicAdd(free_list_top, 1);"
assert source.count(old) == 1
source = source.replace(old, old + """
        atomicExch(&audit_release_ready, 1);
        auto audit_start = clock64();
        while (atomicAdd(&audit_release_resume, 0) == 0 &&
               clock64() - audit_start < 1000000000ULL) {}
""")
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(source)
