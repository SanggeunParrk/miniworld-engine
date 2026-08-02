import sqlite3, sys
db = sys.argv[1]; N = 100
c = sqlite3.connect(db)
# nvtx REPLAY range window
def sval(q, *a):
    r = c.execute(q, a).fetchone()
    return r[0] if r else None
# find REPLAY nvtx range (NVTX_EVENTS text table via StringIds)
row = c.execute("""
  SELECT n.start, n.end FROM NVTX_EVENTS n
  JOIN StringIds s ON n.textId = s.id WHERE s.value='REPLAY' LIMIT 1""").fetchone()
if row is None:
    # some schemas store text directly
    row = c.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='REPLAY' LIMIT 1").fetchone()
t0, t1 = row
rows = c.execute("""
  SELECT s.value, COUNT(*), SUM(k.end-k.start)
  FROM CUPTI_ACTIVITY_KIND_KERNEL k
  JOIN StringIds s ON k.shortName = s.id
  WHERE k.start>=? AND k.start<? GROUP BY s.value ORDER BY 3 DESC""", (t0, t1)).fetchall()
tot = sum(r[2] for r in rows)
ntotal = sum(r[1] for r in rows)
print(f"REPLAY window: {(t1-t0)/1e6/N:.4f} ms/iter wall, {ntotal//N} kernels/iter")
print(f"GPU kernel-time SUM: {tot/1e6/N:.4f} ms/iter  ({ntotal} kernels over {N} iters)")
print(f"{'kernel':<52} {'cnt/it':>7} {'ms/it':>9}")
for name, cnt, dur in rows[:18]:
    print(f"{name[:52]:<52} {cnt//N:>7} {dur/1e6/N:>9.4f}")
