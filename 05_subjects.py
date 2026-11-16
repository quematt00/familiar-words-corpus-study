"""Study B: who is the grammatical subject of cognitive verbs in cs.CL abstracts?

For each finite or infinitival use of a target verb we recover its (possibly
understood) subject from a dependency parse and classify the subject's head as
HUMAN, MODEL, METHOD, or unresolved. The quantity of interest is the share of
resolved subjects that are artificial systems, by year.

Targets
  contested    understand, know, believe, think, reason
  conventional learn            (extended to machines since the 1950s)
  newcomer     hallucinate
  baseline     generate, predict (non-mental verbs that models routinely take)
"""
import re
import sys

import pandas as pd
import spacy

from common import RESULTS, load

FORMS = {
    "understand": "understand|understands|understood",
    "know": "know|knows|knew",
    "believe": "believe|believes|believed",
    "think": "think|thinks",
    "reason": "reason|reasons|reasoned",
    "learn": "learn|learns|learned|learnt",
    "hallucinate": "hallucinate|hallucinates|hallucinated",
    "generate": "generate|generates|generated",
    "predict": "predict|predicts|predicted",
}
KIND = {"understand": "contested", "know": "contested", "believe": "contested", "think": "contested",
        "reason": "contested", "learn": "conventional", "hallucinate": "newcomer",
        "generate": "baseline", "predict": "baseline"}
ANY = re.compile(r"\b(" + "|".join(FORMS.values()) + r")\b", re.I)

HUMAN = {"we", "i", "human", "people", "person", "user", "researcher", "speaker", "reader", "annotator",
         "student", "child", "practitioner", "participant", "expert", "one", "author", "developer",
         "listener", "clinician", "doctor", "patient", "community", "scientist", "linguist", "teacher",
         "individual", "adult", "infant", "worker", "crowdworker", "engineer", "public", "reviewer",
         "audience", "judge", "physician", "everyone", "anyone", "someone", "nobody", "you", "programmer",
         "writer", "educator", "consumer", "customer", "citizen", "professional", "novice", "men", "women"}
MODEL = {"model", "llm", "lm", "plm", "vlm", "mllm", "slm", "lmm", "network", "net", "system", "agent",
         "machine", "transformer", "ai", "algorithm", "classifier", "parser", "encoder", "decoder",
         "architecture", "chatbot", "assistant", "bot", "computer", "generator", "retriever", "tagger",
         "neuron", "layer", "head", "module", "lstm", "rnn", "cnn", "gan", "autoencoder", "embedding",
         "representation", "probe", "predictor", "policy", "reasoner", "solver", "planner", "ranker",
         "summarizer", "translator", "recognizer", "detector", "nlg", "nmt", "seq2seq"}
METHOD = {"method", "approach", "framework", "technique", "pipeline", "tool", "mechanism", "strategy",
          "scheme", "procedure", "paradigm", "objective", "loss", "baseline", "variant", "solution"}
NAMED = re.compile(r"^(?:Chat)?GPT|BERT|RoBERTa|LLaMA|Llama|Claude|Gemini|Bard|Mistral|DeepSeek|Qwen|Grok|"
                   r"Gemma|ELMo|XLNet|Codex|Copilot|PaLM|Vicuna|Alpaca|CLIP|T5$|BART|LM$|Former$|Net$|^o[134]$")
ABILITY = {"ability", "capacity", "capability", "inability", "failure", "tendency", "potential",
           "competence", "capabilities", "abilities", "power", "skill"}
RELPRON = {"that", "which", "who"}
PRON = {"it", "they", "this", "these", "those", "both", "each", "some", "many", "most", "all", "others", "he", "she"}
NEG_GOV = {"fail", "struggle", "unable", "cannot", "lack", "inability", "failure", "incapable", "hard", "difficult"}
HEDGE_ADV = {"really", "truly", "actually", "genuinely", "indeed", "fully", "merely", "simply"}
QUOTES = "\"“”‘’'`"


def subject_of(tok, depth=0):
    """Return (subject token, route) for a verb, following coordination and control."""
    if depth > 4:
        return None, "none"
    subs = [c for c in tok.children if c.dep_ == "nsubj"]
    if subs:
        s = subs[0]
        if s.lower_ in RELPRON and tok.dep_ == "relcl":
            return tok.head, "relative"
        return s, "nsubj"
    if any(c.dep_ == "nsubjpass" for c in tok.children):
        return None, "passive"
    if tok.dep_ == "relcl" and tok.tag_ in ("VBZ", "VBP", "VBD", "VB"):
        return tok.head, "relative"
    if tok.dep_ == "conj" and tok.head.pos_ in ("VERB", "AUX"):
        return subject_of(tok.head, depth + 1)
    if tok.dep_ == "xcomp":
        h = tok.head
        objs = [c for c in h.children if c.dep_ in ("dobj", "obj") and c.i < tok.i]
        if objs and h.pos_ == "VERB":
            return objs[0], "control-object"
        return subject_of(h, depth + 1)
    if tok.dep_ in ("acomp", "attr", "oprd", "advcl", "ccomp") and depth > 0:
        return subject_of(tok.head, depth + 1)
    if tok.dep_ == "advcl":  # purpose clause: "To understand X, we ..."
        return subject_of(tok.head, depth + 1)
    if tok.dep_ == "acl" and tok.head.lemma_.lower() in ABILITY:
        h = tok.head
        for c in h.children:
            if c.dep_ == "poss":
                return c, "ability-of"
            if c.dep_ == "prep" and c.lower_ == "of":
                pobj = [g for g in c.children if g.dep_ == "pobj"]
                if pobj:
                    return pobj[0], "ability-of"
    return None, "none"


def norm(tok) -> str:
    t = tok.text
    if t.isupper() and t.endswith("S") is False and t.lower().endswith("s") and len(t) > 2:
        return t[:-1].lower()
    if re.fullmatch(r"[A-Z]{2,}s", t):  # LLMs -> llm
        return t[:-1].lower()
    return tok.lemma_.lower()


def classify(s) -> str:
    if s is None:
        return "none"
    if s.lower_ in PRON or (s.pos_ == "PRON" and s.lower_ not in HUMAN):
        return "pronoun"
    lem = norm(s)
    if lem in HUMAN or s.lower_ in HUMAN:
        return "human"
    if lem in MODEL or s.lower_ in MODEL:
        return "model"
    if NAMED.search(s.text):
        return "model"
    if lem in METHOD:
        return "method"
    if s.pos_ == "PROPN" or re.search(r"[A-Z].*[A-Z0-9]", s.text):
        return "named"  # system names in abstracts: reported separately
    return "other"


def stance(tok, sent_text: str, sent_start: int) -> str:
    """How is the predication framed? (first matching category wins)"""
    a, b = tok.idx - sent_start, tok.idx - sent_start + len(tok.text)
    if (a > 0 and sent_text[a - 1] in QUOTES) or (b < len(sent_text) and sent_text[b] in QUOTES):
        return "scare-quoted"
    chain, t = [tok], tok
    while t.dep_ in ("xcomp", "conj", "acl", "acomp", "ccomp") and t.head is not t and len(chain) < 5:
        t = t.head
        chain.append(t)
    if any(c.dep_ == "neg" for x in chain for c in x.children) or any(x.lemma_.lower() in NEG_GOV for x in chain[1:]):
        return "negated"
    if sent_text.rstrip().endswith("?") or any(c.lower_ in ("whether", "if") and c.dep_ == "mark" for x in chain for c in x.children):
        return "questioned"
    if any(c.lower_ in HEDGE_ADV and c.dep_ == "advmod" for c in tok.children):
        return "intensified"
    return "plain"


def main():
    recs = load("cscl_sample")
    print(f"{len(recs)} abstracts", flush=True)
    sents = []
    for r in recs:
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\"“`(\[])", r["abstract"]):
            if ANY.search(s) and len(s) < 1200:
                sents.append((s, r["id"], int(r["published"][:4])))
    print(f"{len(sents)} sentences contain a target form", flush=True)

    spacy.prefer_gpu()
    nlp = spacy.load("en_core_web_trf", disable=["ner"])
    form2lemma = {f: lem for lem, fs in FORMS.items() for f in fs.split("|")}
    rows = []
    for i, (doc, (s, pid, year)) in enumerate(zip(nlp.pipe((x[0] for x in sents), batch_size=48), sents)):
        if i % 5000 == 0:
            print(f"  parsed {i}", flush=True)
        for tok in doc:
            lem = form2lemma.get(tok.lower_)
            if lem is None or tok.pos_ != "VERB":
                continue
            if tok.tag_ == "VBN" and not any(c.lemma_ == "have" and c.dep_ == "aux" for c in tok.children):
                continue  # passive / adjectival participle ("learned representations")
            subj, route = subject_of(tok)
            cls = classify(subj)
            rows.append({"id": pid, "year": year, "lemma": lem, "kind": KIND[lem], "form": tok.text,
                         "subject": subj.text if subj is not None else "", "route": route, "class": cls,
                         "stance": stance(tok, s, 0), "sentence": s})
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "B_verb_tokens.csv", index=False)
    print(f"{len(df)} verb tokens written", flush=True)
    print(df.groupby(["lemma", "class"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    sys.exit(main())
