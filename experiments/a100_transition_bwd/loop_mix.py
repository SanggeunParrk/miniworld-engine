"""loop_mix.py <sass.txt> [min_hmma]: opcode mix of each loop (backward branch) with many HMMAs, from cuobjdump -sass output."""
import collections, re, sys
L = open(sys.argv[1]).read().split('\n')
mh = int(sys.argv[2]) if len(sys.argv) > 2 else 100
lab, pend, ins = {}, [], []
for l in L:
    m = re.match(r'\s*(\.L_x_\d+):', l)
    if m: pend.append(m.group(1))
    m = re.search(r'/\*([0-9a-f]{4,})\*/\s+(.*?);', l)
    if m:
        a = int(m.group(1), 16)
        for k in pend: lab[k] = a
        pend = []
        ins.append((a, m.group(2).strip()))
for a, t in ins:
    m = re.search(r'BRA\s.*?\((\.L_x_\d+)\)', t) or re.search(r'BRA\s+(0x[0-9a-f]+)', t)
    if not m: continue
    tgt = lab.get(m.group(1)) if m.group(1).startswith('.L') else int(m.group(1), 16)
    if tgt is None or tgt >= a: continue
    body = [x[1] for x in ins if tgt <= x[0] <= a]
    ops = collections.Counter()
    for t2 in body:
        tok = t2.split(); op = tok[1] if tok[0].startswith('@') else tok[0]
        ops[op if op.startswith(('HMMA', 'F2FP', 'LDSM', 'LDS', 'STG', 'LDL', 'STL')) else op.split('.')[0]] += 1
    if sum(v for k, v in ops.items() if k.startswith('HMMA')) >= mh:
        print(f"loop {tgt:#x}-{a:#x}: {len(body)} instr  " + ", ".join(f"{k} {v}" for k, v in ops.most_common(22)))
