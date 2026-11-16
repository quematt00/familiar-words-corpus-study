"""Study C (arXiv): differentiation and competition.

C1  How do specialists subdivide the word? Type-modifiers of the noun
    *hallucination(s)* (object hallucination, factual hallucination, ...):
    share of tokens modified, and number of distinct types in a fixed-size
    random subsample (so that years of different size are comparable).
C2  Is *hallucination* becoming an object of institutional practice? Share of
    tokens used as the modifier of another noun (hallucination detection, ...).
C3  Do proposed replacements gain ground? *confabulat\\** against *hallucinat\\**.
"""
import json
import re
from collections import Counter

import numpy as np
import pandas as pd
import spacy
import statsmodels.formula.api as smf

from common import RESULTS, is_ai_context, load, wilson

H = re.compile(r"\bhallucinat\w*", re.I)
C = re.compile(r"\bconfabulat\w*", re.I)
# adjectives of degree / quantity / evaluation: not sense-differentiating
DEGREE = set("""severe significant frequent more fewer less many such potential possible various different
certain common serious subtle minimal reduced lower higher harmful new other several numerous substantial
persistent pervasive occasional increased unwanted undesirable critical main major minor overall total
these those same similar additional further specific particular diverse multiple general typical notable
prevalent widespread inherent inevitable unavoidable problematic dangerous costly risky stubborn rare
low high little much most least fine coarse own so-called called existing remaining resulting observed
detected generated induced related known unknown obvious apparent hidden""".split())

BEARER = set("llm lm model lvlm mllm vlm lmm ai agent system chatbot gpt chatgpt machine network generator slm plm".split())
SIZE = 300  # rarefaction sample size (tokens)


def main():
    recs = [r for r in load("hallucin") if is_ai_context(r) and H.search(r["abstract"])]
    sents = []
    for r in recs:
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"“`(\[])", r["abstract"]):
            if re.search(r"\bhallucinations?\b", s, re.I) and len(s) < 1200:
                sents.append((s, int(r["published"][:4])))
    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf", disable=["ner"])
    rows = []
    for doc, (s, year) in zip(nlp.pipe((x[0] for x in sents), batch_size=48), sents):
        for t in doc:
            if t.lower_ not in ("hallucination", "hallucinations") or t.pos_ not in ("NOUN", "PROPN"):
                continue
            mods, bearer = [], False
            for c in t.children:
                if c.dep_ not in ("amod", "compound", "nmod") or c.i >= t.i or c.pos_ == "VERB":
                    continue  # participles such as "induced", "generated" are not type labels
                phrase = " ".join(w.lemma_.lower() for w in c.subtree if w.is_alpha)  # "token level", "cross modal"
                if c.lemma_.lower() in BEARER or c.text.lower().rstrip("s") in BEARER:
                    bearer = True  # "LLM hallucination" says whose, not which kind
                elif c.lemma_.lower() not in DEGREE and re.fullmatch(r"[a-z][a-z ]+", phrase) and len(phrase.split()) <= 3:
                    mods.append(phrase)
            head = t.head.lemma_.lower() if t.dep_ == "compound" and t.head.pos_ == "NOUN" else ""
            rows.append({"year": year, "modifier": " ".join(mods), "has_type_modifier": bool(mods), "head": head})
    tok = pd.DataFrame(rows)
    tok.to_csv(RESULTS / "C_hallucination_noun_tokens.csv", index=False)

    rng = np.random.default_rng(3)
    out = []
    for y, g in tok.groupby("year"):
        n = len(g)
        k = int(g.has_type_modifier.sum())
        lo, hi = wilson(k, n)
        kh = int((g["head"] != "").sum())
        lo_h, hi_h = wilson(kh, n)
        types = np.nan
        if n >= SIZE:  # mean number of distinct modifier types in 200 random subsamples of SIZE tokens
            vals = g.modifier.to_numpy()
            types = float(np.mean([len(set(v for v in rng.choice(vals, SIZE, replace=False) if v)) for _ in range(200)]))
        out.append({"year": y, "noun_tokens": n, "modified_share": k / n, "mod_lo": lo, "mod_hi": hi,
                    f"types_per_{SIZE}_tokens": types, "distinct_types_raw": g[g.has_type_modifier].modifier.nunique(),
                    "as_modifier_share": kh / n, "asmod_lo": lo_h, "asmod_hi": hi_h})
    c1 = pd.DataFrame(out)
    c1.to_csv(RESULTS / "C1_modifiers_by_year.csv", index=False)
    pd.set_option("display.width", 200)
    print(c1.round(3).to_string(index=False))

    with open(RESULTS / "C1_top_modifiers.txt", "w") as f:
        for label, sub in [("2016-2020", tok[tok.year.between(2016, 2020)]), ("2021-2022", tok[tok.year.between(2021, 2022)]),
                           ("2023-2024", tok[tok.year.between(2023, 2024)]), ("2025-2026", tok[tok.year >= 2025])]:
            cm = Counter(sub[sub.has_type_modifier].modifier).most_common(15)
            ch = Counter(sub[sub["head"] != ""]["head"]).most_common(10)
            line = f"{label} (n={len(sub)} noun tokens)\n  type-modifiers: " + ", ".join(f"{m} ({c})" for m, c in cm) + \
                   "\n  heads it modifies: " + ", ".join(f"{m} ({c})" for m, c in ch) + "\n"
            f.write(line)
            print(line)

    # ---- C3: confabulat* vs hallucinat* --------------------------------------
    conf = [r for r in load("confab") if is_ai_context(r) and C.search(r["abstract"])]
    hy = Counter(int(r["published"][:4]) for r in recs)
    cy = Counter(int(r["published"][:4]) for r in conf)
    both = Counter(int(r["published"][:4]) for r in conf if H.search(r["abstract"]))
    comp = []
    for y in sorted(set(hy) | set(cy)):
        union = hy[y] + cy[y] - both[y]
        lo, hi = wilson(cy[y], union)
        comp.append({"year": y, "hallucinat": hy[y], "confabulat": cy[y], "both_in_same_abstract": both[y],
                     "confab_share_of_either": cy[y] / union if union else np.nan, "lo": lo, "hi": hi})
    comp = pd.DataFrame(comp)
    comp.to_csv(RESULTS / "C3_confabulate_vs_hallucinate.csv", index=False)
    print(comp.round(4).to_string(index=False))

    # abstract-level trend 2023+: does an abstract that uses either word use confabulat*?
    ids_c = {r["id"] for r in conf}
    d = pd.DataFrame([{"t": int(r["published"][:4]) - 2023 + (int(r["published"][5:7]) - 1) / 12, "conf": int(r["id"] in ids_c)}
                      for r in {x["id"]: x for x in recs + conf}.values() if r["published"] >= "2023"])
    fit = smf.logit("conf ~ t", data=d).fit(disp=0)
    or_, (lo, hi) = float(np.exp(fit.params["t"])), np.exp(fit.conf_int().loc["t"])
    print(f"logit confabulat* ~ time (2023+): OR/year = {or_:.2f}  95% CI [{lo:.2f}, {hi:.2f}]  p = {fit.pvalues['t']:.2g}  n = {len(d)}")
    (RESULTS / "C3_trend.json").write_text(json.dumps({"odds_ratio_per_year": or_, "ci95": [float(lo), float(hi)],
                                                        "p": float(fit.pvalues["t"]), "n": len(d)}, indent=1))


if __name__ == "__main__":
    main()
