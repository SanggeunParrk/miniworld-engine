import re, sys
F = sys.argv[1]
s = open(F).read()
# kv body only
i0 = s.index("__device__ __forceinline__ void triattn_bwd_kv_body")
i1 = s.index("__global__ void __launch_bounds__(KB_THREADS, 1) triattn_bwd_kv_sm100(")
body = s[i0:i1]
a = body.index("      // ---- dK / dV of row gi ----")
b_end_marker = "    }\n    if (lead) bulk_wait<0>();\n"
b = body.index(b_end_marker)
drain_txt = body[a:b]          # includes the comment line and the drain code up to (excl.) the closing brace of the task loop
# the drain code (without the comment line)
lines = drain_txt.split("\n")
assert lines[0].strip().startswith("// ---- dK / dV of row gi")
code = "\n".join(lines[1:])
# lambda: parameters of the drained task
lam = ("      // (N1) the drain of a task runs after the NEXT task's first stage (its dfr wait overlaps that stage's math); N1_DRAIN_EARLY keeps the old order\n"
       "      auto drain = [&](const int lt_, const int g, const int b_, const int h_, const int kt) {\n"
       + code.replace("lt & 1", "lt_ & 1") + "      };\n")
# insert lambda before the task loop of the group warps
loop = "    for (int t = bid; t < ntasks; t += nb, ++lt) {\n      int kt, g, bh; decode(t, kt, g, bh);\n      const int b_ = bh / H, h_ = bh % H;\n      for (int qs = 0; qs < NQS; ++qs, ++js) {"
assert body.count(loop) == 1, body.count(loop)
new_loop = ("    bool pend = false; int plt = 0, pg = 0, pb = 0, ph = 0, pkt = 0;   // the task whose drain is pending\n" + lam +
            loop)
body = body.replace(loop, new_loop)
a = body.index("      // ---- dK / dV of row gi ----")
b = body.index(b_end_marker)
tail = ("#ifdef N1_DRAIN_EARLY\n      drain(lt, g, b_, h_, kt);\n#else\n      pend = true; plt = lt; pg = g; pb = b_; ph = h_; pkt = kt;\n#endif\n")
body = body[:a] + tail + body[b:]
body = body.replace(b_end_marker, "    }\n#ifndef N1_DRAIN_EARLY\n    if (pend) drain(plt, pg, pb, ph, pkt);\n#endif\n    if (lead) bulk_wait<0>();\n", 1)
# hook after the first stage's pf arrive
hook_anchor = "        if (lane == 0) arrive(A(pf + gi));\n        if (lane == 0 && qq == 2) TL(7 + gi * 6, y);\n"
assert body.count(hook_anchor) == 1
body = body.replace(hook_anchor, hook_anchor + "#ifndef N1_DRAIN_EARLY\n        if (qs == 0 && pend) { drain(plt, pg, pb, ph, pkt); pend = false; }\n#endif\n")
s = s[:i0] + body + s[i1:]
open(F, "w").write(s)
print("patched")
