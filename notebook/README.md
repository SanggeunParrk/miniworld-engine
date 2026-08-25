# Lab notebook

Per-round optimization logs: what was tried on a kernel, what the profiler said, and what the
number did. `<round>/vN.md` is the write-up and `<round>/vN/` holds its captures — ncu text, the
scratch `.py` a version was measured with, csv.

**These are records, not documentation.** They describe the state of a kernel at the moment they
were written and are not updated when it changes. A reader who wants to know how something works
today should use `docs/`; a reader who wants to know why it ended up that way is in the right
place.

They used to live under `docs/`, where 101 of 124 files were logs and the two dozen pages written
for a consumer were lost among them. Their profiler captures lived in a third tree, `profiles/`,
split from the prose that explained them — plus twelve `.gitkeep` files holding open directories
whose captures are gitignored, nine of them empty. One tree now.

Nothing here is maintained, linted, or expected to run. `tests/test_performance_claims.py` treats
`notebook/**/v*.md` as dated records and deliberately exempts them from the traceability rule that
governs prose in `README.md` and `docs/`.
