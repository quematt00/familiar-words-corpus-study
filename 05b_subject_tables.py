"""Study B tables: share of cognitive-verb subjects that are artificial systems, by year.

artificial (strict) = a model noun (model, LLM, network, agent, ...) or a named system (GPT-4)
artificial (broad)  = strict + method nouns (method, approach, framework)
share               = artificial / (artificial + human), over verb tokens whose subject was resolved
"""
import json
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import DATA, RESULTS, load, wilson

df = pd.read_csv(RESULTS / "B_verb_tokens.csv").fillna({"subject": ""})
recs = load("cscl_sample")
n_abs = pd.Series([int(r["published"][:4]) for r in recs]).value_counts().sort_index()
df["strict"] = df["class"].isin(["model", "named"])
df["broad"] = df["class"].isin(["model", "named", "method"])
df["human"] = df["class"] == "human"
df["group"] = df.kind.where(df.kind != "contested", "contested (understand, know, believe, think, reason)")
df["group"] = df.group.replace({"conventional": "learn", "newcomer": "hallucinate", "baseline": "baseline (generate, predict)"})

print("abstracts per year:", n_abs.to_dict())
print("\nsubject classes per verb (all years):")
print(df.groupby(["lemma", "class"]).size().unstack(fill_value=0).to_string())


def table(key: str) -> pd.DataFrame:
    out = []
    for (g, y), s in df.groupby([key, "year"]):
        a, h = int(s.strict.sum()), int(s.human.sum())
        lo, hi = wilson(a, a + h)
        out.append({key: g, "year": y, "verb_tokens": len(s), "artificial": a, "human": h,
                    "artificial_share": a / (a + h) if a + h else np.nan, "lo": lo, "hi": hi,
                    "artificial_broad_share": s.broad.sum() / (s.broad.sum() + h) if s.broad.sum() + h else np.nan,
                    "artificial_per_1000_abstracts": 1000 * a / n_abs[y]})
    return pd.DataFrame(out)


tg, tl = table("group"), table("lemma")
tg.to_csv(RESULTS / "B_subject_share_by_group_year.csv", index=False)
tl.to_csv(RESULTS / "B_subject_share_by_verb_year.csv", index=False)
pd.set_option("display.width", 220)
for g, s in tg.groupby("group"):
    print(f"\n== {g}")
    print(s[["year", "verb_tokens", "artificial", "human", "artificial_share", "lo", "hi", "artificial_per_1000_abstracts"]].round(3).to_string(index=False))

print("\nper verb, pooled periods: artificial share [n resolved]")
df["period"] = pd.cut(df.year, [2014, 2019, 2022, 2024, 2026], labels=["2015-19", "2020-22", "2023-24", "2025-26"])
res = df[df.strict | df.human]
pv = res.groupby(["lemma", "period"], observed=True).agg(share=("strict", "mean"), n=("strict", "size"))
print(pv.assign(cell=lambda d: (100 * d.share).round(0).astype(int).astype(str) + "% [" + d.n.astype(str) + "]").cell.unstack().to_string())
pv.reset_index().to_csv(RESULTS / "B_subject_share_by_verb_period.csv", index=False)

# ---- trend tests: does the artificial share rise, and faster for contested verbs than for baselines? ----
tests = {}
res = res.assign(t=res.year - 2015, art=res.strict.astype(int))
for g, s in res.groupby("group"):
    if s.art.nunique() < 2:
        continue
    fit = smf.logit("art ~ t", data=s).fit(disp=0)
    o, (lo, hi) = float(np.exp(fit.params["t"])), np.exp(fit.conf_int().loc["t"])
    tests[g] = {"odds_ratio_per_year": o, "ci95": [float(lo), float(hi)], "p": float(fit.pvalues["t"]), "n": int(fit.nobs)}
    print(f"logit artificial ~ year [{g}]: OR/year = {o:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  p = {fit.pvalues['t']:.2g}  n = {int(fit.nobs)}")
two = res[res.kind.isin(["contested", "baseline"])].assign(contested=lambda d: (d.kind == "contested").astype(int))
fit = smf.logit("art ~ t * contested", data=two).fit(disp=0)
o, (lo, hi) = float(np.exp(fit.params["t:contested"])), np.exp(fit.conf_int().loc["t:contested"])
tests["interaction: contested vs baseline"] = {"ratio_of_odds_ratios_per_year": o, "ci95": [float(lo), float(hi)], "p": float(fit.pvalues["t:contested"])}
print(f"interaction year x contested (vs baseline): ratio of ORs = {o:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  p = {fit.pvalues['t:contested']:.2g}")
(RESULTS / "B_trend_tests.json").write_text(json.dumps(tests, indent=1))

# ---- how are model-subject uses of the contested verbs framed? ------------------------------------
ms = df[df.strict & (df.kind == "contested")]
st = ms.groupby(["period", "stance"], observed=True).size().unstack(fill_value=0)
st["n"] = st.sum(axis=1)
for c in [c for c in st.columns if c != "n"]:
    st[f"{c}_share"] = st[c] / st.n
st.reset_index().to_csv(RESULTS / "B_stance_by_period.csv", index=False)
print("\nframing of model-subject uses of contested verbs:")
print(st.round(3).to_string())

# ---- scare quotes around cognitive words in the whole sample (regex, no parsing) --------------------
stems = {"understand": r"understand\w*|understood", "know": r"know(?:s|n|ing|ledge)?|knew", "think": r"think\w*|thought", "reason": r"reason\w*",
         "learn": r"learn\w*", "believe": r"belie[fv]\w*", "memor": r"memor\w*|remember\w*|forget\w*", "aware": r"aware\w*", "intelligen": r"intelligen\w*"}
Q = "\"“”‘’`'"
rows = []
for r in recs:
    y = int(r["published"][:4])
    for k, pat in stems.items():
        for m in re.finditer(rf"\b(?:{pat})\b", r["abstract"], re.I):
            a, b = m.start(), m.end()
            quoted = (a > 0 and r["abstract"][a - 1] in Q and (a < 2 or not r["abstract"][a - 2].isalnum())) and (b < len(r["abstract"]) and r["abstract"][b] in Q + ",.")
            rows.append((y, k, quoted))
q = pd.DataFrame(rows, columns=["year", "stem", "quoted"])
qt = q.groupby("year").agg(tokens=("quoted", "size"), quoted=("quoted", "sum"))
qt["quoted_per_10k_tokens"] = 1e4 * qt.quoted / qt.tokens
qt["quoted_per_1000_abstracts"] = 1000 * qt.quoted / n_abs
qt.reset_index().to_csv(RESULTS / "B_scare_quoted_cognitive_words.csv", index=False)
print("\nscare-quoted cognitive words (single quoted word), whole sample:")
print(qt.round(2).to_string())
print(q[q.quoted].groupby("stem").size().sort_values(ascending=False).to_string())

# ---- check on the arXiv search index: local rate of hallucinat* in the sample vs API-derived rate ----
H = re.compile(r"\bhallucinat\w*", re.I)
local = pd.Series({y: np.mean([bool(H.search(r["abstract"])) for r in recs if int(r["published"][:4]) == y]) for y in n_abs.index})
api_tot = json.loads((DATA / "counts_by_year.json").read_text())["cs_CL"]
hall = pd.Series([int(r["published"][:4]) for r in load("hallucin") if "cs.CL" in r["cats"] and H.search(r["abstract"])]).value_counts()
chk = pd.DataFrame({"sample_rate_per_1000": 1000 * local, "api_rate_per_1000": [1000 * hall.get(y, 0) / api_tot[str(y)] for y in n_abs.index]}, index=n_abs.index)
chk.to_csv(RESULTS / "check_search_index.csv")
print("\nsearch-index check (cs.CL abstracts containing hallucinat*, per 1,000):")
print(chk.round(1).to_string())

# ---- validation sample ---------------------------------------------------------------------------
val = df[df["class"].isin(["model", "named", "human", "method"])].sample(60, random_state=9)
with open(RESULTS / "B_validation_sample.txt", "w") as f:
    for _, r in val.iterrows():
        f.write(f"[{r.year}] {r.lemma:11s} subj={r.subject!s:14.14s} {r['class']:6s} {r.route:14s} {r.stance:12s} | {r.sentence[:210]}\n")
