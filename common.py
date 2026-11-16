"""Shared helpers: loading, date handling, Wilson intervals, flag detection."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

HERE = Path(__file__).parent
DATA = HERE / "data"
RESULTS = HERE / "results"
FIGS = HERE / "figures"
for d in (RESULTS, FIGS):
    d.mkdir(exist_ok=True)

AI_CATS = ("cs.", "stat.ML", "eess.")


def load(name: str) -> list[dict]:
    with open(DATA / f"{name}.jsonl") as f:
        return [json.loads(line) for line in f]


def is_ai_context(rec: dict) -> bool:
    """Record is listed in a computing / ML category (primary or cross-list)."""
    return any(c.startswith(AI_CATS) for c in rec["cats"])


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, mid - half), min(1.0, mid + half))


# ---------------------------------------------------------------- flag detection
def quoted_spans(text: str) -> list[str]:
    """Short spans (<= 4 words) enclosed in quotation marks of any common style."""
    pats = [
        r"``\s*([^`']{1,60}?)\s*''",  # LaTeX ``...''
        r"“\s*([^“”]{1,60}?)\s*”",  # curly double
        r"\"\s*([^\"]{1,60}?)\s*\"",  # straight double
        r"‘\s*([^‘’]{1,60}?)\s*’",  # curly single
        r"(?<!\w)'\s*([^']{1,60}?)\s*'(?!\w)",  # straight single (not apostrophes)
        r"(?<!\w)`\s*([^`']{1,60}?)\s*'(?!\w)",  # LaTeX `...'
    ]
    out = []
    for p in pats:
        for m in re.finditer(p, text):
            if len(m.group(1).split()) <= 4:
                out.append(m.group(1))
    return out


def emph_spans(text: str) -> list[str]:
    """Short spans marked as italic / emphasised (LaTeX or markdown)."""
    pats = [
        r"\\(?:emph|textit|textsl|textbf)\{([^}]{1,60})\}",
        r"\{\\(?:em|it|sl|bf)\s+([^}]{1,60})\}",
        r"(?<![\w*])\*([^*\s][^*]{0,58})\*(?![\w*])",
        r"(?<![\w_])_([^_\s][^_]{0,58})_(?![\w_])",
    ]
    out = []
    for p in pats:
        for m in re.finditer(p, text):
            if len(m.group(1).split()) <= 4:
                out.append(m.group(1))
    return out


META_CUE = r"(?:so[- ]called|known as|referred to as|refer to as|termed|called|dubbed|coined|a\.k\.a\.?|aka)"


def flags(text: str, stem: str) -> dict[str, bool]:
    """Which kinds of metalinguistic marking attach to words beginning with `stem`?

    quote : the word sits inside quotation marks                  ("hallucinate")
    meta  : it is introduced by a naming formula                  (so-called / known as ...)
    emph  : it is italicised / emphasised                         (\\emph{hallucination})
    gloss : it is immediately followed by an explanatory gloss    (hallucination, i.e. ...)
    """
    term = re.compile(rf"\b{stem}\w*", re.I)
    quote = any(term.search(s) for s in quoted_spans(text))
    emph = any(term.search(s) for s in emph_spans(text))
    meta = bool(
        re.search(
            rf"{META_CUE}\s+(?:the\s+|an?\s+)?[\"“‘`'{{\\\w]*?\s?(?:[\w-]+\s){{0,2}}[\"“‘`']{{0,2}}{stem}",
            text,
            re.I,
        )
    )
    gloss = bool(
        re.search(
            rf"{stem}\w*[\"”’']{{0,2}}\s*(?:,|\(|--|—|–|:)\s*"
            r"(?:i\.e\.|that is|namely|where|which (?:is|are|refers?)|defined as|meaning|"
            r"(?:a|the) (?:phenomenon|tendency|generation|problem|issue))",
            text,
            re.I,
        )
    )
    # strict variants: the bare word itself is marked, not a new compound or a method name
    quote_base = any(re.fullmatch(rf"(?:the |an? )?{stem}\w*[.,;:]?", s.strip(), re.I) for s in quoted_spans(text))
    meta_base = bool(re.search(rf"{META_CUE}\s+(?:as\s+)?(?:the\s+|an?\s+)?[\"“‘`']{{0,2}}{stem}", text, re.I))
    return {"quote": quote, "meta": meta, "emph": emph, "gloss": gloss, "quote_base": quote_base, "meta_base": meta_base}


# ---------------------------------------------------------------- senses of *hallucinate* (arXiv)
_FEAT = re.compile(
    r"face hallucinat|hallucinat\w+ (?:network|module|branch|loss)|hallucinator|super[- ]resol|"
    r"(?:feature|data|sample|image|depth|motion|view|scene|texture|detail|modality|pose|flow|example)s? hallucinat|"
    r"learn(?:s|ing)? to hallucinate|by hallucinating|hallucination[- ]based|"
    r"hallucinat(?:e|es|ed|ing) (?:(?:the|a|an|new|novel|missing|additional|plausible|realistic|high[- ]resolution|"
    r"fine|unseen|occluded|future|extra|diverse|more|virtual|synthetic|pseudo)\s+){0,3}"
    r"(?:details?|features?|samples?|examples?|images?|views?|depth|textures?|pixels?|regions?|frames?|scenes?|data|"
    r"modalit\w+|poses?|motions?|structures?|shapes?|geometry|representations?|obstacles?|faces?|parts?|appearance)",
    re.I,
)
_FAIL_TIED = re.compile(
    r"(?:mitigat|alleviat|reduc|suppress|detect|avoid|prevent|tackl|address|combat|eliminat|minimi[sz]|curb|fewer|less|"
    r"without|free of|prone to|suffer\w* from|susceptible to|vulnerable to|risk of)\w*\s+(?:the\s+|such\s+|these\s+)?"
    r"(?:\w+[- ]){0,2}hallucinat|hallucinations? (?:detection|mitigation|rates?|problems?|issues?|benchmarks?|"
    r"evaluation|errors?)|hallucination[- ]free|object hallucination",
    re.I,
)
_FAIL_LOOSE = re.compile(
    r"unfaithful|factual|incorrect|fabricat|nonsensical|trustworth|misinformation|inaccura|misleading|erroneous|"
    r"ungrounded|unsupported|faithful|non-?existent|not (?:present|supported|grounded)",
    re.I,
)


def arxiv_sense(text: str) -> str:
    """feat    = hallucinating as something a system is built to do (fill in plausible detail)
    failure = hallucinating as something a system should not do (unsupported or false output)"""
    if _FAIL_TIED.search(text):
        return "failure"
    if _FEAT.search(text):
        return "feat"
    if _FAIL_LOOSE.search(text):
        return "failure"
    return "unclear"
