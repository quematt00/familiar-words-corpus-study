"""Minimal, polite client for the arXiv API (export.arxiv.org/api/query).

- One request at a time, >= 3.2 s apart (arXiv asks for 3 s).
- Every response is cached on disk, so re-running an analysis never re-queries.
- Records are sliced by submission date, because deep paging is unreliable.
"""
from __future__ import annotations

import calendar
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

API = "https://export.arxiv.org/api/query"
UA = "corpus-pilot/0.1"
NS = {
    "a": "http://www.w3.org/2005/Atom",
    "os": "http://a9.com/-/spec/opensearch/1.1/",
    "x": "http://arxiv.org/schemas/atom",
}
CACHE = Path(__file__).parent / "data" / "raw"
CACHE.mkdir(parents=True, exist_ok=True)
_last = [0.0]


def _get(params: dict, use_cache: bool = True) -> str:
    key = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()
    f = CACHE / f"{key}.xml"
    if use_cache and f.exists():
        return f.read_text()
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        wait = 3.2 - (time.time() - _last[0])
        if wait > 0:
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=120) as r:
                txt = r.read().decode("utf-8")
            _last[0] = time.time()
            if "<opensearch:totalResults>" in txt:
                f.write_text(txt)
                return txt
        except Exception as e:  # network hiccup or 5xx: back off and retry
            _last[0] = time.time()
            print(f"  retry {attempt + 1}: {type(e).__name__} {e}", flush=True)
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"arXiv API failed for {params}")


def _total(txt: str) -> int:
    return int(re.search(r"<opensearch:totalResults>(\d+)", txt).group(1))


def count(query: str) -> int:
    """Number of records matching a query."""
    return _total(_get({"search_query": query, "start": 0, "max_results": 1}))


def _parse(txt: str) -> list[dict]:
    out = []
    for e in ET.fromstring(txt).findall("a:entry", NS):
        ident = e.findtext("a:id", "", NS)
        if "/abs/" not in ident:
            continue
        prim = e.find("x:primary_category", NS)
        out.append(
            {
                "id": re.sub(r"v\d+$", "", ident.split("/abs/")[1]),
                "published": e.findtext("a:published", "", NS)[:10],
                "title": " ".join(e.findtext("a:title", "", NS).split()),
                "abstract": " ".join(e.findtext("a:summary", "", NS).split()),
                "primary": prim.get("term") if prim is not None else "",
                "cats": [c.get("term") for c in e.findall("a:category", NS)],
            }
        )
    return out


def window(y: int, m0: int = 1, m1: int = 12) -> str:
    """submittedDate filter covering months m0..m1 of year y."""
    last = calendar.monthrange(y, m1)[1]  # arXiv returns HTTP 500 for dates like 31 February
    return f"submittedDate:[{y}{m0:02d}010000 TO {y}{m1:02d}{last}2359]"


def fetch(query: str, page: int = 400, cap: int | None = None) -> tuple[list[dict], int]:
    """All records for a query (oldest first), up to `cap`. Returns (records, total)."""
    first = _get({"search_query": query, "start": 0, "max_results": 1})
    total = _total(first)
    want = total if cap is None else min(cap, total)
    recs: dict[str, dict] = {}
    start = 0
    while start < want:
        params = {
            "search_query": query,
            "start": start,
            "max_results": min(page, want - start),
            "sortBy": "submittedDate",
            "sortOrder": "ascending",
        }
        got = _parse(_get(params))
        if not got:  # arXiv sometimes returns an empty page: retry once uncached
            got = _parse(_get(params, use_cache=False))
        for r in got:
            recs[r["id"]] = r
        start += page
    return list(recs.values()), total
