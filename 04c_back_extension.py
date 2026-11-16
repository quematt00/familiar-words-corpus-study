"""Exploratory: is *hallucinate* said of people more often, outside clinical or drug contexts?

A comment counts as "person, non-clinical" if a person is the grammatical subject of the verb
("I / you / people ... hallucinated") and no clinical, drug or dream cue occurs within 100
characters. Because no reliable count of all Hacker News comments is available, the yardstick
is internal: the number of comments in the same year that use the word in its clinical or
drug sense. The ratio is informative only if that older sense is used at a roughly stable rate.
The pattern also captures an older hyperbolic idiom ("did I hallucinate that?"), so only the
change over time is of interest.
"""
import importlib

import pandas as pd

from common import RESULTS, load

hn = importlib.import_module("04b_sense_hn")  # re-uses its patterns (this re-runs that analysis)
rows = []
for r in load("hn_hallucin"):
    win = " ".join(r["text"][max(0, m.start() - 100): m.end() + 100] for m in hn.TERM.finditer(r["text"]))
    if not win:
        continue
    person, clinical = bool(hn.HUMAN_SUBJ.search(win)), bool(hn.HUMAN_CUE.search(win))
    ai = bool(hn.AI_CI.search(win) or hn.AI_CS.search(win))
    rows.append({"year": int(r["created"][:4]), "person_nonclinical": person and not clinical,
                 "clinical": clinical and not ai, "ai_thread": bool(hn.STORY_AI.search(r["story"])), "text": win[:260]})
d = pd.DataFrame(rows)
t = d.groupby("year").agg(person_nonclinical=("person_nonclinical", "sum"), clinical_sense=("clinical", "sum")).reset_index()
t["of_which_in_ai_threads"] = d[d.person_nonclinical].groupby("year").ai_thread.sum().reindex(t.year).fillna(0).astype(int).values
t["per_100_clinical"] = 100 * t.person_nonclinical / t.clinical_sense
t.to_csv(RESULTS / "A3_hn_person_subject_nonclinical.csv", index=False)
print(t.round(1).to_string(index=False))
d[d.person_nonclinical & (d.year >= 2024)].sample(25, random_state=4).to_csv(RESULTS / "A3_examples.csv", index=False)
