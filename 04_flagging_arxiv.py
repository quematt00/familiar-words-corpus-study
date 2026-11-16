"""Study A (arXiv): does metalinguistic flagging of *hallucinat\\** decay as the word spreads?

Unit of analysis: the abstract. An abstract counts as flagged if at least one
occurrence of the word is in quotation marks or introduced by a naming formula
("so-called", "known as", ...). Emphasis and explanatory glosses are reported
separately as weaker forms of marking.
"""
import json
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import DATA, RESULTS, arxiv_sense, flags, is_ai_context, load, wilson

TERM = re.compile(r"\bhallucinat\w*", re.I)
CORE = {"cs.CL", "cs.AI", "cs.LG", "cs.CV"}

rows = []
for r in load("hallucin"):
    if not TERM.search(r["abstract"]):
        continue  # search-engine stemming matched something else
    f = flags(r["abstract"], "hallucinat")
    prim = r["primary"]
    rows.append(
        {
            "id": r["id"],
            "year": int(r["published"][:4]),
            "month": r["published"][:7],
            "primary": prim,
            "group": prim if prim in CORE else ("other cs/ML" if prim.startswith(("cs.", "stat.ML", "eess.")) else "non-computing"),
            "ai": is_ai_context(r),
            "core": bool(CORE & set(r["cats"])),
            "n_tokens": len(TERM.findall(r["abstract"])),
            **f,
            "flagged": f["quote"] or f["meta"],
            "flagged_base": f["quote_base"] or f["meta_base"],
            "sense": arxiv_sense(r["abstract"]),
            "any_mark": any(f.values()),
            "abstract": r["abstract"],
        }
    )
df = pd.DataFrame(rows)
df.drop(columns="abstract").to_csv(RESULTS / "A_arxiv_records.csv", index=False)
print(f"{len(df)} abstracts contain hallucinat*; {df.ai.sum()} are in computing/ML categories")

ai = df[df.ai].copy()
den = json.loads((DATA / "counts_by_year.json").read_text())["cs_core"]

# ---- yearly table -----------------------------------------------------------
tab = []
for y, g in ai.groupby("year"):
    n = len(g)
    row = {"year": y, "n_abstracts": n, "per_1000_core": 1000 * g.core.sum() / den[str(y)] if den.get(str(y)) else np.nan}
    for col in ["flagged", "flagged_base", "quote", "meta", "emph", "gloss", "any_mark"]:
        k = int(g[col].sum())
        lo, hi = wilson(k, n)
        row |= {f"{col}_k": k, f"{col}_share": k / n, f"{col}_lo": lo, f"{col}_hi": hi}
    tab.append(row)
tab = pd.DataFrame(tab)
tab.to_csv(RESULTS / "A_arxiv_flagging_by_year.csv", index=False)
pd.set_option("display.width", 200)
print(tab[["year", "n_abstracts", "per_1000_core", "flagged_k", "flagged_share", "flagged_lo", "flagged_hi", "flagged_base_k", "flagged_base_share", "gloss_share", "any_mark_share"]].round(3).to_string(index=False))

# ---- trend tests ------------------------------------------------------------
m = ai[ai.year >= 2016].copy()
m["t"] = m.year - 2016
m["flag"] = m.flagged.astype(int)
m["mark"] = m.any_mark.astype(int)
m["base"] = m.flagged_base.astype(int)
out = {}
for dv in ["flag", "base", "mark"]:
    for label, formula in [("year only", f"{dv} ~ t"), ("year + primary category", f"{dv} ~ t + C(group)")]:
        fit = smf.logit(formula, data=m).fit(disp=0)
        or_, (lo, hi) = float(np.exp(fit.params["t"])), np.exp(fit.conf_int().loc["t"])
        out[f"{dv} | {label}"] = {"odds_ratio_per_year": or_, "ci95": [float(lo), float(hi)], "p": float(fit.pvalues["t"]), "n": int(fit.nobs)}
        print(f"logit {dv:4s} [{label:24s}] OR/year = {or_:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  p = {fit.pvalues['t']:.2g}  n = {int(fit.nobs)}")

# Before / after the public release of ChatGPT (30 Nov 2022)
pre, post = m[m.month <= "2022-11"], m[m.month > "2022-11"]
for name, g in [("2016-01..2022-11", pre), ("2022-12..2026-10", post)]:
    k, n = int(g.flag.sum()), len(g)
    lo, hi = wilson(k, n)
    out[f"flag share {name}"] = {"k": k, "n": n, "share": k / n, "ci95": [lo, hi]}
    print(f"flagged {name}: {k}/{n} = {k / n:.3f}  [{lo:.3f}, {hi:.3f}]")
(RESULTS / "A_arxiv_trend_tests.json").write_text(json.dumps(out, indent=1))

# ---- sense: feat vs failure -----------------------------------------------------
st = ai.groupby(["year", "sense"]).agg(n=("flagged", "size"), flagged=("flagged", "sum")).unstack(fill_value=0)
st.columns = [f"{a}_{b}" for a, b in st.columns]
cls = st.n_failure + st.n_feat
st["failure_share_of_classified"] = st.n_failure / cls
st[["fs_lo", "fs_hi"]] = [wilson(int(a), int(b)) for a, b in zip(st.n_failure, cls)]
for s_ in ["failure", "feat"]:
    st[f"flag_share_{s_}"] = st[f"flagged_{s_}"] / st[f"n_{s_}"].where(st[f"n_{s_}"] > 0)
st.reset_index().to_csv(RESULTS / "A_arxiv_sense_by_year.csv", index=False)
print(st[["n_feat", "n_failure", "n_unclear", "failure_share_of_classified", "flagged_feat", "flagged_failure", "flag_share_feat", "flag_share_failure"]].round(3).to_string())

# decay phase only: failure-sense abstracts from the 2023 peak onwards
dec = m[(m.sense == "failure") & (m.year >= 2023)].copy()
dec["t"] = dec.year - 2023 + (dec.month.str[5:7].astype(int) - 1) / 12
for dv in ["flag", "base"]:
    fit = smf.logit(f"{dv} ~ t", data=dec).fit(disp=0)
    or_, (lo, hi) = float(np.exp(fit.params["t"])), np.exp(fit.conf_int().loc["t"])
    out[f"{dv} | failure sense, 2023+"] = {"odds_ratio_per_year": or_, "ci95": [float(lo), float(hi)], "p": float(fit.pvalues["t"]), "n": int(fit.nobs)}
    print(f"logit {dv:4s} [failure sense, 2023+     ] OR/year = {or_:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  p = {fit.pvalues['t']:.2g}  n = {int(fit.nobs)}")
(RESULTS / "A_arxiv_trend_tests.json").write_text(json.dumps(out, indent=1))

rng0 = np.random.default_rng(5)
with open(RESULTS / "A_arxiv_sense_examples.txt", "w") as f:
    for s_ in ["feat", "failure", "unclear"]:
        g = ai[ai.sense == s_]
        pick = g.iloc[rng0.choice(len(g), size=min(25, len(g)), replace=False)].sort_values("year")
        f.write(f"===== classified {s_}: {len(pick)} of {len(g)} =====\n")
        for _, r in pick.iterrows():
            mm = TERM.search(r.abstract)
            f.write(f"[{r.year}] {r.id} ({r.primary}) ...{r.abstract[max(0, mm.start() - 130): mm.end() + 110]}...\n")

# ---- examples for manual validation ----------------------------------------
rng = np.random.default_rng(7)
with open(RESULTS / "A_arxiv_flag_examples.txt", "w") as f:
    for label, sub in [("FLAGGED", ai[ai.flagged]), ("UNFLAGGED", ai[~ai.flagged])]:
        pick = sub.iloc[rng.choice(len(sub), size=min(40, len(sub)), replace=False)].sort_values("year")
        f.write(f"===== {label}: {len(pick)} random abstracts (context around first occurrence) =====\n")
        for _, r in pick.iterrows():
            mm = TERM.search(r.abstract)
            ctx = r.abstract[max(0, mm.start() - 150) : mm.end() + 90]
            kinds = ",".join(k for k in ["quote", "meta", "emph", "gloss"] if r[k]) or "-"
            f.write(f"[{r.year}] {r.id} ({r.primary}) <{kinds}> ...{ctx}...\n")
print("wrote examples for manual validation")
