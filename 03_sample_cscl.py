"""Step 3: a time-stratified sample of cs.CL abstracts, 2015-2026.

For every month we take the first K abstracts submitted in that month
(K = 250, or all of them where a month has fewer). The sample is therefore
stratified by month but not random within a month; see the methods note.
"""
import json
from pathlib import Path

from arxiv import fetch, window

DATA = Path(__file__).parent / "data"
K = 250
recs = {}
for y in range(2015, 2027):
    n_y = 0
    for m in range(1, 13):
        if (y, m) > (2026, 10):  # last complete month at time of harvest
            break
        rs, tot = fetch(f"cat:cs.CL AND {window(y, m, m)}", page=K, cap=K)
        for r in rs:
            r["month_total"] = tot
            recs[r["id"]] = r
        n_y += len(rs)
    print(f"{y}: sampled {n_y}", flush=True)

with open(DATA / "cscl_sample.jsonl", "w") as f:
    for r in sorted(recs.values(), key=lambda r: r["published"]):
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"== {len(recs)} abstracts", flush=True)
