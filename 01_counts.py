"""Step 1: yearly record counts, to size the harvest and provide denominators."""
import json
from pathlib import Path

from arxiv import count, window

YEARS = range(2010, 2027)
QUERIES = {
    "hallucin_all": "abs:hallucination",  # arXiv search stems: matches hallucinate(s/d), hallucination(s)
    "confab_all": "abs:confabulation",
    "cs_core": "(cat:cs.CL OR cat:cs.AI OR cat:cs.LG OR cat:cs.CV)",
    "cs_CL": "cat:cs.CL",
}

out = {}
for name, q in QUERIES.items():
    out[name] = {}
    for y in YEARS:
        out[name][y] = count(f"{q} AND {window(y)}")
    print(name, out[name], flush=True)

Path(__file__).parent.joinpath("data", "counts_by_year.json").write_text(json.dumps(out, indent=1))
