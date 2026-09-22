# Two-CTA TMA multicast experiment

Not selected. Initial fixed-tolerance correctness passes atL64 and training buckets; production dispatch unchanged.

Adjacent dW CTAs process two hidden groups of the same row tile, so their x_n is identical. Adjacent dX CTAs process neighboring row tiles and share projection/gate weights. `make_multicast_weights.py` adds static2-CTA clusters,132active clusters verified through CUDA occupancy API, and multicasts these common operands. All CTAs set expected transaction bytes before the leader issues multicast. The first implementation uses a cluster barrier; `make_multicast_async.py` replaces per-transfer full-cluster barriers with thread0 ready mbarriers, remote release arrival and leader acquire wait. Initial/final cluster synchronization remains. The paired CTAs have equal round counts for supported even training tile counts.

L768 sweep: plain1418µs, weights multicast2160µs, x_n1686µs, both2228µs. Thread0 ready-handshake version: plain1418µs, weights2115µs, x_n1892µs, both2208µs. L384 weights multicast~545–549µs versus plain~362–367µs. These are comparative sweep medians, not selected600sample paired timings. Some direct variants spilled and were rejected before execution. The added transfer/coordination cost outweighed reuse benefits; no promotion or full sanitizer qualification.

Sources: NVIDIA PTX ISA, https://docs.nvidia.com/cuda/parallel-thread-execution/index.html ; NVIDIA CUTLASS multicast instruction wrappers, https://github.com/NVIDIA/cutlass/blob/main/include/cute/arch/copy_sm90_tma.hpp . Code uses explicit PTX TMA multicast and preserves WGMMA arithmetic, BF16 boundaries, forward saves and one-launch final reductions.

Additional failed experiment: prefetched LN mean/rstd in an extra512shared bytes exceeded two-CTA cooperative residency. Reusing the now-dead G3 region at57344 avoids this allocation problem and passes initial correctness, but gives~396µs L384 and~1445µs L768, not a reliable win. Source `*_lnstats_gap`. Do not reuse the extra-shared variant at264CTAs; CUDA correctly rejects that launch.
