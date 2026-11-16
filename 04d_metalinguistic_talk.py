"""Study A'' (Hacker News): do people keep arguing about what the word means?

Among comments that use *hallucinat\\** of an AI system, the share in which the word occurs
within 80 characters of explicit talk about words and meanings ("the term", "definition",
"misnomer", "what ... means", "anthropomorphi-"). Unlike scare quotes, this does not mark
the word as a stretch; it marks open disagreement or uncertainty about its meaning.
"""
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import RESULTS, load, wilson

TERM = re.compile(r"\bhallucinat\w*", re.I)
TALK = re.compile(
    r"\b(?:definitions?|defin(?:e|es|ed|ing)|redefin\w+|meaning of|what (?:\w+ ){0,3}means?|the (?:term|word|label|concept|phrase)|"
    r"misnomer|euphemism|terminology|semantics|anthropomorphi\w+|wrong (?:word|term)|bad (?:word|term|name)|"
    r"better (?:word|term|name)|loaded (?:word|term)|technical term|term of art|jargon|marketing (?:word|term|speak))\b", re.I)
rec = pd.read_csv(RESULTS / "A2_hn_records.csv", dtype={"id": str})
ai = rec[(rec.sense == "AI") & (rec.year >= 2022)].set_index("id")
talk = {}
for r in load("hn_hallucin"):
    if r["id"] in ai.index:
        win = " ".join(r["text"][max(0, m.start() - 80): m.end() + 80] for m in TERM.finditer(r["text"]))
        talk[r["id"]] = bool(TALK.search(win))
ai["talk"] = pd.Series(talk)
out = []
for y, g in ai.groupby("year"):
    for col in ["talk", "flagged"]:
        k, n = int(g[col].sum()), len(g)
        lo, hi = wilson(k, n)
        out.append({"year": y, "measure": col, "k": k, "n": n, "share": k / n, "lo": lo, "hi": hi})
t = pd.DataFrame(out)
t.to_csv(RESULTS / "A4_hn_metalinguistic_talk.csv", index=False)
print(t.round(4).to_string(index=False))
d = ai[ai.year >= 2023].assign(t=lambda x: pd.to_datetime(x.created).map(lambda v: v.year + (v.dayofyear - 1) / 365.25) - 2023)
for col in ["talk", "flagged"]:
    fit = smf.logit(f"y ~ t", data=d.assign(y=d[col].astype(int))).fit(disp=0)
    lo, hi = np.exp(fit.conf_int().loc["t"])
    print(f"logit {col} ~ time (2023+): OR/year = {np.exp(fit.params['t']):.3f}  95% CI [{lo:.3f}, {hi:.3f}]  p = {fit.pvalues['t']:.2g}")
