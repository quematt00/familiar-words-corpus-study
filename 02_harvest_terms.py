"""Step 2: harvest every arXiv abstract that the search engine matches to the
stems of *hallucination* and *confabulation* (exact matching is done locally later).

Years with many hits are sliced by month so that paging stays shallow.
"""
import json
from pathlib import Path

from arxiv import count, fetch, window

DATA = Path(__file__).parent / "data"
TERMS = {"hallucin": "abs:hallucination", "confab": "abs:confabulation"}

for name, q in TERMS.items():
    recs, expected = {}, 0
    for y in range(2000, 2027):
        n = count(f"{q} AND {window(y)}")
        if n == 0:
            continue
        slices = [(1, 12)] if n <= 1200 else [(m, m) for m in range(1, 13)]
        got_y = 0
        for m0, m1 in slices:
            rs, tot = fetch(f"{q} AND {window(y, m0, m1)}")
            expected += tot
            got_y += len(rs)
            for r in rs:
                recs[r["id"]] = r
        print(f"{name} {y}: api={n} fetched={got_y}", flush=True)
    with open(DATA / f"{name}.jsonl", "w") as f:
        for r in sorted(recs.values(), key=lambda r: r["published"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"== {name}: {len(recs)} unique records (API reported {expected})", flush=True)
