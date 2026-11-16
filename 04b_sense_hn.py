"""Study A' (Hacker News): the sense shift of *hallucinat\\** in a conversational register.

1. Which sense? Each comment is classified by lexical cues in the comment and in
   the title of the story it belongs to:
     AI      - said of an artificial system
     human   - perceptual / clinical / drug sense, said of people
     unclear - no cue either way (includes hyperbolic uses about people)
2. Within AI-sense comments: is the word metalinguistically flagged?
"""
import json
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import RESULTS, flags, load, wilson

TERM = re.compile(r"\bhallucinat\w*", re.I)
# Cues are searched in a window of +-100 characters around each occurrence, not in the whole
# comment: early comments often contain a stray "model" or "algorithm" far from the word.
AI_CI = re.compile(
    r"\b(?:ai|a\.i\.|agi|llms?|gpt[-\w.]*|chatgpt|openai|anthropic|claude|gemini|copilot|llama|mistral|deepseek|"
    r"perplexity|codex|language models?|neural (?:net|network)s?|chatbots?|prompts?|prompt(?:ed|ing)|models?|"
    r"transformers?|machine learning|deep learning|generative|gen-?ai|training (?:data|set)|tokens?|agents?|agentic|"
    r"bots?|stable diffusion|midjourney|dall-?e|diffusion|embeddings?|fine-?tun\w+|inference|context window|"
    r"autocomplete|dlss|upscal\w+|deepdream|deep dream|self-driving|vibe[- ]cod\w+|weights|benchmarks?|"
    r"super-?resolution|classifiers?|algorithms?|citations?|apis?|functions?|librar(?:y|ies))\b",
    re.I,
)
AI_CS = re.compile(r"\b(?:ML|RAG|FSD|GANs?|Opus|Cursor|Bard|Sonnet|Whisper|Grok|Bing|Siri|Alexa)\b")
HUMAN_CUE = re.compile(
    r"\b(?:lsd|psychedelic\w*|shrooms?|mushrooms?|psilocybin|dmt|acid|drugs?|trip(?:s|ping|ped)?|schizophreni\w*|"
    r"psychosis|psychotic|sleep[- ]depriv\w*|delirium|fever|dementia|migraines?|ketamine|salvia|ayahuasca|mescaline|"
    r"hallucinogen\w*|hearing voices|charles bonnet|sensory depriv\w*|mental (?:illness|health)|patients?|anesthe\w*|"
    r"ambien|datura|benadryl|alcohol|withdrawal|narcolep\w*|sleep paralysis|hypnagogic|auditory|medication\w*|"
    r"antipsychotic\w*|parkinson\w*|seizures?|epilep\w*|stoned|high on|cannabis|weed|thc|opioid\w*|bipolar|insomnia|"
    r"ptsd|neurolog\w*|psychiatr\w*|doctors?|hospital\w*|dream(?:s|ing|t|ed)?|delusion\w*|paranoi\w*|"
    r"(?:mass|collective|shared|consensual|group|consensus|mutual) hallucinat\w*)\b",
    re.I,
)
# a person (not "it"/"they", which may be a model) as the grammatical subject of the verb
HUMAN_SUBJ = re.compile(
    r"\b(?:I|you|we|he|she|people|humans?|brains?|minds?|someone|somebody|person|everyone|nobody|OP|author|"
    r"I'm|you're|we're|he's|she's)\s+(?:(?:am|are|is|was|were|must|might|may|can|could|would|will|have|had|has|did|do|"
    r"does|just|really|actually|also|often|sometimes|literally|probably|definitely|not|be|been|started|began|always|"
    r"constantly|all|still|never|ever|basically|essentially|collectively|"
    r"didn't|don't|doesn't|wasn't|weren't)\s+){0,4}hallucinat",
    re.I,
)
STORY_AI = re.compile(
    r"\b(?:AI|A\.I\.|AGI|LLMs?|GPT[-\w.]*|ChatGPT|OpenAI|Anthropic|Claude|Gemini|Copilot|Llama|Mistral|DeepSeek|"
    r"Grok|Perplexity|Bard|language models?|neural|chatbots?|prompts?|transformers?|machine learning|"
    r"deep learning|generative|Stable Diffusion|Midjourney|DALL-?E|diffusion|agents?|agentic|vibe[- ]cod\w+)\b",
    re.I,
)


def sense(text: str, story: str) -> str:
    win = " ".join(text[max(0, m.start() - 100): m.end() + 100] for m in TERM.finditer(text))
    if HUMAN_SUBJ.search(win):
        return "human"
    ai, hu = bool(AI_CI.search(win) or AI_CS.search(win)), bool(HUMAN_CUE.search(win))
    if ai and not hu:
        return "AI"
    if hu and not ai:
        return "human"
    if not ai and not hu and STORY_AI.search(story):
        return "AI"  # no local cue, but the thread is about an AI system
    return "unclear"


rows = []
for r in load("hn_hallucin"):
    if not TERM.search(r["text"]):
        continue
    f = flags(r["text"], "hallucinat")
    y, mth = int(r["created"][:4]), int(r["created"][5:7])
    rows.append({"id": r["id"], "created": r["created"], "year": y, "quarter": f"{y}Q{(mth - 1) // 3 + 1}",
                 "half": f"{y}H{1 if mth <= 6 else 2}", "sense": sense(r["text"], r["story"]),
                 **f, "flagged": f["quote"] or f["meta"], "flagged_base": f["quote_base"] or f["meta_base"], "text": r["text"], "story": r["story"]})
df = pd.DataFrame(rows)
df.drop(columns=["text", "story"]).to_csv(RESULTS / "A2_hn_records.csv", index=False)
print(f"{len(df)} HN comments contain hallucinat*")
print(df.groupby(["year", "sense"]).size().unstack(fill_value=0).to_string())

# ---- sense share by quarter and by year --------------------------------------

def sense_table(key: str) -> pd.DataFrame:
    out = []
    for k, g in df.groupby(key):
        n, a, h = len(g), int((g.sense == "AI").sum()), int((g.sense == "human").sum())
        lo, hi = wilson(a, n)
        lo2, hi2 = wilson(a, a + h)
        row = {key: k, "n": n, "ai": a, "human": h, "unclear": n - a - h, "ai_share_all": a / n,
               "ai_all_lo": lo, "ai_all_hi": hi, "ai_share_classified": a / (a + h) if a + h else np.nan,
               "ai_cls_lo": lo2, "ai_cls_hi": hi2}
        out.append(row)
    return pd.DataFrame(out)


sense_table("quarter").to_csv(RESULTS / "A2_hn_sense_by_quarter.csv", index=False)
ty = sense_table("year")
ty.to_csv(RESULTS / "A2_hn_sense_by_year.csv", index=False)
print(ty[["year", "n", "ai", "human", "unclear", "ai_share_all", "ai_share_classified"]].round(3).to_string(index=False))

# ---- flagging among AI-sense comments ----------------------------------------
ai = df[df.sense == "AI"].copy()
tab = []
for k, g in ai.groupby("year"):
    n = len(g)
    row = {"year": k, "n": n}
    for col in ["flagged", "flagged_base", "quote", "meta", "emph"]:
        kk = int(g[col].sum())
        lo, hi = wilson(kk, n)
        row |= {f"{col}_k": kk, f"{col}_share": kk / n, f"{col}_lo": lo, f"{col}_hi": hi}
    tab.append(row)
tab = pd.DataFrame(tab)
tab.to_csv(RESULTS / "A2_hn_flagging_by_year.csv", index=False)
print(tab[["year", "n", "flagged_k", "flagged_share", "flagged_lo", "flagged_hi", "flagged_base_k", "flagged_base_share"]].round(3).to_string(index=False))

m = ai[ai.year >= 2016].copy()
m = m.assign(t=pd.to_datetime(m.created).map(lambda d: d.year + (d.dayofyear - 1) / 365.25) - 2016)
m["flag"] = m.flagged.astype(int)
fit = smf.logit("flag ~ t", data=m).fit(disp=0)
or_, (lo, hi) = float(np.exp(fit.params["t"])), np.exp(fit.conf_int().loc["t"])
print(f"logit flag ~ time: OR/year = {or_:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  p = {fit.pvalues['t']:.2g}  n = {int(fit.nobs)}")
m["base"] = m.flagged_base.astype(int)
dec = m[m.created >= "2023-01-01"].copy()
dec["t"] = dec.t - 7
res_dec = {}
for dv in ["flag", "base"]:
    fd = smf.logit(f"{dv} ~ t", data=dec).fit(disp=0)
    o, (l, h) = float(np.exp(fd.params["t"])), np.exp(fd.conf_int().loc["t"])
    res_dec[f"{dv} ~ time (AI-sense, 2023+)"] = {"odds_ratio_per_year": o, "ci95": [float(l), float(h)], "p": float(fd.pvalues["t"]), "n": int(fd.nobs)}
    print(f"logit {dv} ~ time, 2023+: OR/year = {o:.3f}  95% CI [{l:.3f}, {h:.3f}]  p = {fd.pvalues['t']:.2g}  n = {int(fd.nobs)}")
res = {**res_dec, "flag ~ time (AI-sense, 2016+)": {"odds_ratio_per_year": or_, "ci95": [float(lo), float(hi)], "p": float(fit.pvalues["t"]), "n": int(fit.nobs)}}
for name, g in [("2016-01..2022-11", m[m.created <= "2022-11-30"]), ("2022-12..2026-10", m[m.created > "2022-11-30"])]:
    k, n = int(g.flag.sum()), len(g)
    lo, hi = wilson(k, n)
    res[f"flag share {name}"] = {"k": k, "n": n, "share": k / n if n else None, "ci95": [lo, hi]}
    print(f"flagged {name}: {k}/{n} = {k / max(n, 1):.3f}  [{lo:.3f}, {hi:.3f}]")
(RESULTS / "A2_hn_trend_tests.json").write_text(json.dumps(res, indent=1))

# ---- samples for manual validation of the sense classifier ---------------------
rng = np.random.default_rng(11)
with open(RESULTS / "A2_hn_sense_examples.txt", "w") as f:
    for period, sub in [("<=2021", df[df.year <= 2021]), (">=2023", df[df.year >= 2023])]:
        for s in ["AI", "human", "unclear"]:
            g = sub[sub.sense == s]
            pick = g.iloc[rng.choice(len(g), size=min(20, len(g)), replace=False)] if len(g) else g
            f.write(f"===== {period} | classified {s} | {len(pick)} of {len(g)} =====\n")
            for _, r in pick.iterrows():
                mm = TERM.search(r.text)
                f.write(f"[{r.created[:4]}] <{r.story[:38]}> ...{r.text[max(0, mm.start() - 110): mm.end() + 90]}...\n")
print("wrote sense examples for manual validation")
