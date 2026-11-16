"""Step 3b: a second, conversational register - Hacker News comments, 2012-2026.

Uses the public HN Search API (hn.algolia.com). The API returns at most 1,000
hits per query, so each time window is bisected until it fits. Author names are
not stored. Exact matching and sense classification happen locally later.

The comment texts are not redistributed. `data/hn_hallucin_ids.csv` lists the
comments harvested on 13 November 2026; when that file is present, a re-harvest
is restricted to those IDs, so that later runs analyse the same comments
(minus any that have since been deleted).

No denominator is harvested: the API's hit counts for unfiltered queries are
estimates, not counts, so all HN results are reported as within-sample shares.
"""
import csv
import datetime as dt
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATA = Path(__file__).parent / "data"
API = "https://hn.algolia.com/api/v1/search_by_date"
FORMS = ["hallucinate", "hallucinates", "hallucinated", "hallucinating", "hallucination", "hallucinations"]
T0 = int(dt.datetime(2012, 1, 1, tzinfo=dt.timezone.utc).timestamp())
T1 = int(dt.datetime(2026, 11, 1, tzinfo=dt.timezone.utc).timestamp())
IDS = DATA / "hn_hallucin_ids.csv"
n_requests = [0]


def get(params: dict) -> dict:
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            time.sleep(0.6)
            req = urllib.request.Request(url, headers={"User-Agent": "corpus-pilot/0.1"})
            with urllib.request.urlopen(req, timeout=60) as r:
                n_requests[0] += 1
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:
            print(f"  retry {attempt + 1}: {type(e).__name__} {e}", flush=True)
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(url)


def clean(s: str) -> str:
    s = re.sub(r"<i>(.*?)</i>", r"*\1*", s or "", flags=re.S)  # keep italics as *...*
    s = re.sub(r"<[^>]+>", " ", s)
    return " ".join(html.unescape(s).split())


def harvest(form: str, a: int, b: int, out: dict) -> None:
    """Collect all comments matching `form` created in [a, b)."""
    base = {
        "query": form,
        "tags": "comment",
        "numericFilters": f"created_at_i>={a},created_at_i<{b}",
        "hitsPerPage": 1000,
        "typoTolerance": "false",
    }
    res = get(base)
    if res["nbHits"] > 1000 and b - a > 3600:
        mid = (a + b) // 2
        harvest(form, a, mid, out)
        harvest(form, mid, b, out)
        return
    for h in res["hits"]:
        out[h["objectID"]] = {
            "id": h["objectID"],
            "created": h["created_at"][:10],
            "story": h.get("story_title") or "",
            "text": clean(h.get("comment_text")),
        }


def restrict_to_published(out: dict) -> dict:
    """Keep only the comments listed in the published ID file, if there is one."""
    if not IDS.exists():
        return out
    with open(IDS) as f:
        keep = {row["id"] for row in csv.DictReader(f)}
    missing = keep - out.keys()
    print(f"restricted to {len(keep)} published IDs; {len(missing)} no longer retrievable", flush=True)
    return {k: v for k, v in out.items() if k in keep}


if __name__ == "__main__":
    out: dict = {}
    for form in FORMS:
        before = len(out)
        harvest(form, T0, T1, out)
        print(f"{form}: +{len(out) - before} (total {len(out)}, {n_requests[0]} requests)", flush=True)
    out = restrict_to_published(out)
    with open(DATA / "hn_hallucin.jsonl", "w") as f:
        for r in sorted(out.values(), key=lambda r: r["created"]):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"== {len(out)} comments, {n_requests[0]} requests", flush=True)
