# Corpus study: *hallucinate* and the cognitive verbs, 2015–2026

Code, data, results and validation samples for the corpus study reported in §3 of Matthieu Queloz, "Familiar Words, Unfamiliar Minds: Conventionalization without Calibration in the Vocabulary of Mind".

Archived at Zenodo: https://doi.org/10.5281/zenodo.24279389 (version 1.0.0, as cited in the paper); https://doi.org/10.5281/zenodo.24279388 always resolves to the latest version.

Data were harvested on 13 November 2026. Everything here can be re-run from the scripts in this folder.

The figures are numbered differently here and in the paper:

| Paper | This folder |
|---|---|
| Figure 1 | `figures/fig1_takeover_then_decay` |
| Figure 2 | `figures/fig3_verb_subjects` |
| not shown | `figures/fig2_differentiation` (the counts are reported in the text of §3.2) |

"Fig." below refers to the numbering of this folder.

## What was measured

Three corpora, all public:

| Corpus | Size | Register |
|---|---|---|
| arXiv abstracts containing *hallucinat\** | 8,568 (all of them, 2013 to 13 Nov 2026) | specialists |
| Hacker News comments containing *hallucinat\** | 31,866 (all of them, 2012 to Oct 2026) | technical public |
| arXiv cs.CL abstracts, sampled by month | 31,123 (2015 to Oct 2026) | specialists |

"Marking" below means that an abstract or comment puts the word in quotation marks or introduces it with a naming formula (*so-called*, *known as*, *termed*, *referred to as*).

## Findings

**1. The new sense took over within a few years (Fig. 1a, 1b).**
- arXiv: the sense in which hallucinating is a failure, not a feat, was 18% of classifiable uses in 2019, 74% in 2022 and 96% in 2026.
- Hacker News: uses about an AI system were about 10% of classifiable uses from 2015 to 2021, 43% in 2022 and 89% in 2023.

**2. The marking then wore off, faster among specialists (Fig. 1c, 1d).**
- arXiv: 11.4% of abstracts marked the word in 2023 (63 of 551) and 2.3% in 2026 (76 of 3,313). Over the same years the word became three times as frequent (8 to 24 per 1,000 abstracts in the four core categories).
- Counting only cases where the bare word is marked, not a new compound, the fall is from 11.3% to 1.3%.
- For failure-sense abstracts from 2023 on, the odds of marking halve each year (odds ratio 0.50, 95% CI 0.44–0.58).
- Hacker News: 12.1% in 2024, 5.6% in 2026 (odds ratio 0.81 per year, CI 0.78–0.85).
- Before 2023 the rates are low and noisy, on small counts. Marking did not decline steadily from first use; it peaked when the word reached a mass audience.

**3. Specialists subdivided the word as the marking disappeared (Fig. 2).**
- Distinct kinds of hallucination named per 300 uses of the noun (*object*, *factual*, *package*, *faithfulness* …): 27 in 2023, 50 in 2026.
- Uses in compounds such as *hallucination detection*, *rate*, *benchmark*: 18% in 2023, 28% in 2026.
- The nouns it modifies shift from *network, method* (2016–20) to *problem, phenomenon* (2021–22) to *detection, rate, mitigation, risk* (2025–26).

**4. *Confabulate* is growing but marginal.**
- It appears in 0.5% of abstracts using either word in 2023 and 1.8% in 2026 (60 abstracts; odds ratio 1.97 per year, CI 1.46–2.65).
- In 22 sampled contexts it mostly names something narrower (fabricated accounts, false self-reports), not a replacement for *hallucinate*. This is an impression from reading, not a count.

**5. AI systems became the usual subject of cognitive verbs, but not as plain assertion (Fig. 3).**
- For *understand, know, believe, think, reason*, AI systems were 28% of classifiable subjects in 2015–2023 and 65% in 2026.
- Much of this is not specific to cognitive verbs. *Generate* and *predict* drift the same way (72% to 88%). With each verb compared only to itself, the cognitive verbs rise at 1.11 per year (CI 1.06–1.16) and the baseline verbs at 1.09; the difference is not significant (p = 0.41).
- The pooled rise is partly a change of mix: fewer *we believe*, more *reason*. *Know* (14% to 50%) and *think* (11% to 85%) rise steeply on small counts; *understand* barely moves (37% to 44%).
- Attributions to AI systems are increasingly questioned, negated, hedged or scare-quoted: 9% in 2015–19, 22% in 2025–26 (odds ratio 1.14 per year, CI 1.05–1.24). This is the opposite of what happened to *hallucinate*.

**6. Exploratory: the word is returning to people (Hacker News).**
- Non-clinical uses with a person as subject (*you hallucinated additional context*) ran at 14–32 per 100 clinical or drug uses every year from 2012 to 2022, then 81 in 2023 and 137 in 2026.

## What this does and does not show

- It shows conventionalization: the stretch stopped being marked within about three years of mass adoption.
- It does not show miscalibration. No corpus measure here captures whether the inferences the word invites are sound.
- One check came out against the idea that dispute persists while marking fades: explicit talk about the word's meaning on Hacker News (*the term*, *definition*, *misnomer*) also fell, from 4.5% of AI-sense comments in 2024 to 2.1% in 2026.
- The contrast between finding 2 and finding 5 is the most interesting result for the paper: a failure label conventionalized fast, while capacity attributions spread and became more contested. It rests on one word against five and should be offered as a hypothesis.

## Methods

- **arXiv terms.** The arXiv API's abstract search stems word forms, so one query returns every abstract with a form of the word. Exact matching is then done locally. Records were fetched in date slices and the number retrieved equals the number the API reports (8,569 of 8,569).
- **arXiv sample.** For each month, the first 250 cs.CL abstracts submitted (all of them where fewer). Stratified by month, not random within a month.
- **Hacker News.** The public search API, queried for each word form and bisected by date until every window returned under 1,000 hits. Author names were not stored.
- **Senses.** Rule-based, from cue words. arXiv: failure cues (*mitigate, detect, prone to* … *hallucination*) against feat cues (*face hallucination*, *hallucinate details*). Hacker News: cues within 100 characters of the word, plus the story title when there is no local cue.
- **Verb subjects.** Sentences containing a target verb form were parsed with spaCy's transformer model. The subject is recovered through coordination, control (*enables the model to understand*), relative clauses, and *ability of X to …*. Subjects are classed as person, AI system, method, pronoun or other; shares use person and AI system only.
- **Intervals and tests.** 95% Wilson intervals; logistic regression on year.

## Validation (by hand, small samples)

| Check | Result |
|---|---|
| Marking detector, arXiv, flagged items | about 32 of 40 mark the bare word; 6 more mark a new compound or method name; 2 unclear |
| Marking detector, arXiv, unflagged items | 0 of 40 missed |
| Marking detector, Hacker News, flagged items | 22 of 24 correct |
| arXiv sense, "feat" | 20 of 25 correct (18 of 18 before 2023) |
| arXiv sense, "failure" | 24 of 25 correct |
| Hacker News sense, "AI", before 2022 | 15 of 20 correct, so the 10% baseline is an upper estimate |
| Hacker News sense, "person", from 2023 | 18 of 20 correct |
| Verb subjects, cognitive verbs, 2024–26 | 40 of 40 correct |
| Verb subjects, all verbs, all years | about 38 of 44 correct; errors lean towards "person" for *generate* |
| Search index | *hallucinat\** rate in the cs.CL sample and the rate implied by the search agree within sampling error (largest gap: 22 against 28 per 1,000 in 2023) |

All validation was done by a single annotator, the AI assistant Claude. There was no second annotator, so no inter-annotator agreement is reported.

## Limitations

- No general news corpus; full-text news archives are licensed. Hacker News is a technical public, not a lay one.
- One word carries findings 1–4.
- The communities grew enormously. Falling marking may reflect newcomers adopting an existing convention more than individuals changing habits.
- About a third of recent arXiv abstracts and a seventh of recent Hacker News comments could not be assigned a sense. Roughly 11 of 15 sampled unassigned comments were about AI, so the Hacker News AI share is conservative.
- 2026 is a partial year.
- The selection-versus-drift test proposed earlier (Newberry et al. 2017) was not run. *Hallucinate* is at over 98% against *confabulate* throughout, and monthly counts of the rival are in single digits, so the test would be uninformative.
- Hacker News results are within-sample shares. The search API does not give a reliable count of all comments, so no per-comment rates are reported.

## Data and what is redistributed

- **arXiv.** `data/hallucin.jsonl`, `data/confab.jsonl` and `data/cscl_sample.jsonl` hold the arXiv metadata (identifier, date, title, abstract, categories) that the analyses use. arXiv makes its metadata available under CC0 1.0. Thank you to arXiv for use of its open access interoperability.
- **Hacker News.** The comment texts are not redistributed. `data/hn_hallucin_ids.csv` lists the 31,895 comments harvested, and `results/A2_hn_records.csv` gives the derived annotations (date, sense, marking) for the 31,866 that contain a form of the word after exact matching. `03b_harvest_hn.py` fetches the texts again from the public search API and keeps only the listed IDs; comments deleted since the harvest will be missing. Author names were never stored.
- **Excerpts.** The validation samples in `results/` (`*_examples.txt`, `A3_examples.csv`, `B_validation_sample.txt`) quote short excerpts of abstracts and of about 150 comments, without author names, so that the hand-checks can be inspected.
- **Not included.** The cache of raw arXiv API responses (`data/raw/`) is regenerated by the harvest scripts.

## Reproducing

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt && .venv/bin/python -m spacy download en_core_web_trf
```

Then run the numbered scripts in order with `.venv/bin/python`. To re-run only the analyses on the published data, start at `04_flagging_arxiv.py`; the Hacker News analyses (`04b`–`04d`) need the comment texts, so run `03b_harvest_hn.py` first. Responses from arXiv are cached in `data/raw`, so re-running the arXiv harvest does not re-query. The harvest takes about 40 minutes; the analyses take about five.

## Files

- `figures/`: three figures, PNG and PDF.
- `results/`: one CSV per table, trend tests as JSON, and the validation samples as text.
- `data/`: the arXiv records, the Hacker News ID list, and the harvest logs.
- `01`–`03b` harvest; `04`–`04d` *hallucinate*; `05`–`05b` verb subjects; `06` subdivision and rivals; `07` figures.

## Licence

- Code: MIT (see `LICENSE`).
- Results tables, figures and this documentation: CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/).
- arXiv metadata is in the public domain under CC0 1.0; quoted Hacker News excerpts remain their authors'.
