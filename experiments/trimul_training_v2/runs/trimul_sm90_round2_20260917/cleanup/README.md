# Workspace compiler-dump cleanup

Deleted the66 untracked `cutlass*.ptx` / `cutlass*.cubin` files directly in
`/home/psk6950/MiniWorld` (3,512,388bytes). The user confirmed this exact prefix.
No training files, kernel sources, cache entries, SVGs, or evidence directories
were removed. Exact deleted names and sizes are recorded in `root-dumps.json`.

Round2 GPU commands use an experiment directory as their working directory.
CuTe dump capture sets `CUTE_DSL_DUMP_DIR` inside that directory where supported.
The workspace root is checked again before completion.
