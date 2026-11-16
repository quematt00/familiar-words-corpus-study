"""Figures for the pilot. Static, print-oriented (white surface), one y-scale per panel.

Colour follows the corpus throughout: arXiv = blue, Hacker News = orange.
Palette validated with the dataviz skill's validator (all-pairs, light, white surface).
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter, MaxNLocator

from common import FIGS, RESULTS

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"], "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK2, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.titlelocation": "left", "axes.titlepad": 8,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
    "axes.axisbelow": True, "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "pdf.fonttype": 42,
})
PCT = FuncFormatter(lambda v, _: f"{100 * v:.0f}%")


def series(ax, x, y, lo, hi, color, markers=True):
    ax.fill_between(x, lo, hi, color=color, alpha=0.12, linewidth=0)
    ax.plot(x, y, color=color, linewidth=2, solid_capstyle="round", solid_joinstyle="round",
            marker="o" if markers else None, markersize=6, markeredgecolor="white", markeredgewidth=1.5)


def label(ax, x, y, text, dx=0, dy=9, ha="center"):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center", color=INK, fontsize=8)


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(FIGS / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# =============================================================== Figure 1: takeover, then decay
def fig1():
    ax_s = pd.read_csv(RESULTS / "A_arxiv_sense_by_year.csv")
    ax_f = pd.read_csv(RESULTS / "A_arxiv_flagging_by_year.csv")
    hn_q = pd.read_csv(RESULTS / "A2_hn_sense_by_quarter.csv")
    hn_f = pd.read_csv(RESULTS / "A2_hn_flagging_by_year.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.6, 6.1), gridspec_kw={"hspace": 0.62, "wspace": 0.3})
    fig.subplots_adjust(top=0.83, bottom=0.1)
    for x0, name, col in [(0.125, "arXiv abstracts (specialists)", BLUE), (0.557, "Hacker News comments (technical public)", ORANGE)]:
        fig.patches.append(matplotlib.patches.Rectangle((x0, 0.952), 0.012, 0.016, transform=fig.transFigure, color=col, linewidth=0))
        fig.text(x0 + 0.02, 0.96, name, color=INK, fontsize=10.5, fontweight="bold", va="center")

    # (a) arXiv: failure sense as share of sense-classified abstracts
    a = axes[0, 0]
    d = ax_s[(ax_s.year >= 2016) & (ax_s.n_failure + ax_s.n_feat >= 5)]
    series(a, d.year, d.failure_share_of_classified, d.fs_lo, d.fs_hi, BLUE)
    a.set_title("a  The new sense takes over\n", loc="left")
    a.text(0, 1.03, "Share of uses in which hallucinating is a failure, not a feat", transform=a.transAxes, color=INK2, fontsize=8)
    label(a, 2016, d.failure_share_of_classified.iloc[0], f"{100 * d.failure_share_of_classified.iloc[0]:.0f}%", dy=10)
    label(a, 2026, d.failure_share_of_classified.iloc[-1], f"{100 * d.failure_share_of_classified.iloc[-1]:.0f}%", dy=-11)

    # (b) HN: AI sense as share of sense-classified comments, by quarter
    b = axes[0, 1]
    q = hn_q[(hn_q.quarter >= "2015Q1") & (hn_q.quarter <= "2026Q3")].copy()
    q["x"] = q.quarter.str[:4].astype(int) + (q.quarter.str[5].astype(int) - 0.5) / 4
    series(b, q.x, q.ai_share_classified, q.ai_cls_lo, q.ai_cls_hi, ORANGE, markers=False)
    b.plot(q.x.iloc[-1], q.ai_share_classified.iloc[-1], "o", color=ORANGE, markersize=6, markeredgecolor="white", markeredgewidth=1.5)
    b.set_title("b  The new sense takes over\n", loc="left")
    b.text(0, 1.03, "Share of uses that are about an AI system, by quarter", transform=b.transAxes, color=INK2, fontsize=8)
    b.axvline(2022.92, color=AXIS, linewidth=0.8)
    b.text(2022.75, 0.60, "ChatGPT\nreleased", color=INK2, fontsize=7.5, ha="right", va="center")
    label(b, q.x.iloc[-1], q.ai_share_classified.iloc[-1], f"{100 * q.ai_share_classified.iloc[-1]:.0f}%", dy=-11, dx=-2)
    pre = q[q.x < 2022].ai_share_classified.mean()
    b.text(2018.3, pre + 0.13, f"{100 * pre:.0f}% on average\nbefore 2022", color=INK, fontsize=8, ha="center")

    # (c) arXiv flagging by year
    c = axes[1, 0]
    d = ax_f[ax_f.year >= 2018]
    series(c, d.year, d.flagged_share, d.flagged_lo, d.flagged_hi, BLUE)
    c.set_title("c  Then the marking wears off\n", loc="left")
    c.text(0, 1.03, "Share of abstracts marking the word (quotes, “so-called”)", transform=c.transAxes, color=INK2, fontsize=8)
    pk = d.loc[d.flagged_share.idxmax()] if False else d[d.year == 2023].iloc[0]
    label(c, pk.year, pk.flagged_share, f"{100 * pk.flagged_share:.0f}%", dy=10, dx=-12)
    label(c, 2026, d.flagged_share.iloc[-1], f"{100 * d.flagged_share.iloc[-1]:.0f}%", dy=10)

    # (d) HN flagging by year (AI-sense comments)
    e = axes[1, 1]
    d = hn_f[hn_f.year >= 2018]
    series(e, d.year, d.flagged_share, d.flagged_lo, d.flagged_hi, ORANGE)
    e.set_title("d  Then the marking wears off\n", loc="left")
    e.text(0, 1.03, "Same measure, comments about an AI system", transform=e.transAxes, color=INK2, fontsize=8)
    label(e, 2024, d[d.year == 2024].flagged_share.iloc[0], f"{100 * d[d.year == 2024].flagged_share.iloc[0]:.0f}%", dy=10)
    label(e, 2026, d.flagged_share.iloc[-1], f"{100 * d.flagged_share.iloc[-1]:.0f}%", dy=-11)

    for a_ in (axes[0, 0], axes[0, 1]):
        a_.set_ylim(0, 1.05); a_.yaxis.set_major_formatter(PCT); a_.set_yticks([0, .25, .5, .75, 1])
    for a_ in (axes[1, 0], axes[1, 1]):
        a_.set_ylim(0, 0.42); a_.yaxis.set_major_formatter(PCT); a_.set_yticks([0, .1, .2, .3, .4])
    axes[0, 0].set_xticks(range(2016, 2027, 2)); axes[0, 1].set_xticks(range(2016, 2027, 2))
    axes[1, 0].set_xticks(range(2018, 2027, 2)); axes[1, 1].set_xticks(range(2018, 2027, 2))
    axes[0, 1].set_xlim(2014.8, 2026.9)
    fig.text(0.125, 0.025, "Shaded bands: 95% Wilson intervals. 2026 covers January to October. "
             "Early years rest on few items (see tables).", color=MUTED, fontsize=7.5)
    save(fig, "fig1_takeover_then_decay")


# =============================================================== Figure 2: differentiation (arXiv)
def fig2():
    c1 = pd.read_csv(RESULTS / "C1_modifiers_by_year.csv")
    d = c1[c1.year >= 2023]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 2.9), gridspec_kw={"wspace": 0.3})
    fig.subplots_adjust(top=0.78, bottom=0.2)
    y = d.types_per_300_tokens
    a.plot(d.year, y, color=BLUE, linewidth=2, marker="o", markersize=6, markeredgecolor="white", markeredgewidth=1.5)
    a.set_title("a  Specialists subdivide the word\n", loc="left")
    a.text(0, 1.03, "Distinct kinds of hallucination named per 300 uses of the noun", transform=a.transAxes, color=INK2, fontsize=8)
    a.set_ylim(0, 62); a.set_yticks([0, 20, 40, 60])
    label(a, 2023, y.iloc[0], f"{y.iloc[0]:.0f}", dy=11); label(a, 2026, y.iloc[-1], f"{y.iloc[-1]:.0f}", dy=11)

    series(b, d.year, d.as_modifier_share, d.asmod_lo, d.asmod_hi, BLUE)
    b.set_title("b  It becomes something to measure and manage\n", loc="left")
    b.text(0, 1.03, "Share of uses in compounds such as hallucination detection or rate", transform=b.transAxes, color=INK2, fontsize=8)
    b.set_ylim(0, 0.42); b.yaxis.set_major_formatter(PCT); b.set_yticks([0, .1, .2, .3, .4])
    label(b, 2023, d.as_modifier_share.iloc[0], f"{100 * d.as_modifier_share.iloc[0]:.0f}%", dy=11)
    label(b, 2026, d.as_modifier_share.iloc[-1], f"{100 * d.as_modifier_share.iloc[-1]:.0f}%", dy=11)
    for ax in (a, b):
        ax.set_xticks([2023, 2024, 2025, 2026]); ax.set_xlim(2022.6, 2026.4)
    fig.text(0.125, 0.03, "arXiv abstracts in computing categories. Left: mean over 200 random subsamples of 300 noun tokens per year. "
             "Right: 95% Wilson intervals.", color=MUTED, fontsize=7.5)
    save(fig, "fig2_differentiation")


# =============================================================== Figure 3: who understands, knows, reasons?
def fig3():
    g = pd.read_csv(RESULTS / "B_subject_share_by_group_year.csv")
    mk = pd.read_csv(RESULTS / "B_marked_share_by_period.csv")
    GREY1, GREY2 = "#898781", "#b5b3aa"
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.6, 3.7), gridspec_kw={"wspace": 0.3, "width_ratios": [1.4, 1]})
    fig.subplots_adjust(top=0.82, bottom=0.3)
    names = {"contested": "understand, know, believe, think, reason", "learn": "learn", "baseline": "generate, predict"}
    handles = {}
    for key, col in [("baseline", GREY2), ("learn", GREY1), ("contested", BLUE)]:
        d = g[g.group.str.startswith(key)]
        if key == "contested":
            series(a, d.year, d.artificial_share, d.lo, d.hi, col)
        else:
            a.plot(d.year, d.artificial_share, color=col, linewidth=2, solid_capstyle="round")
        handles[key] = a.plot([], [], color=col, linewidth=2)[0]
    c = g[g.group.str.startswith("contested")]
    le, ba = g[g.group.str.startswith("learn")], g[g.group.str.startswith("baseline")]
    label(a, 2026, le.artificial_share.iloc[-1], "learn", dy=9, dx=2, ha="right")
    label(a, 2026.4, 0.775, "generate, predict", dy=0, dx=0, ha="right")
    label(a, 2024.6, 0.60, "five cognitive verbs", dy=0, dx=0, ha="right")
    label(a, 2023, c[c.year == 2023].artificial_share.iloc[0], "31%", dy=-12)
    label(a, 2026, c.artificial_share.iloc[-1], "65%", dy=-13, dx=2)
    a.set_title("a  AI systems overtake people as subjects of cognitive verbs\n", loc="left")
    a.text(0, 1.03, "Share of verb subjects that are AI systems, not people", transform=a.transAxes, color=INK2, fontsize=8)
    a.set_ylim(0, 1.05); a.yaxis.set_major_formatter(PCT); a.set_yticks([0, .25, .5, .75, 1]); a.set_xticks(range(2016, 2027, 2)); a.set_xlim(2014.6, 2026.5)
    a.legend([handles[k] for k in ("contested", "learn", "baseline")], [names[k] for k in ("contested", "learn", "baseline")],
             loc="upper left", bbox_to_anchor=(-0.02, -0.13), ncol=3, frameon=False, fontsize=7.5, handlelength=1.4, labelcolor=INK2, columnspacing=1.2, borderaxespad=0)

    x = np.arange(len(mk))
    series(b, x, mk.share, mk.lo, mk.hi, BLUE)
    b.set_xticks(x); b.set_xticklabels(mk.period)
    b.set_title("b  But less often as a plain assertion\n", loc="left")
    b.text(0, 1.03, "Share questioned, negated, hedged or in scare quotes", transform=b.transAxes, color=INK2, fontsize=8)
    b.set_ylim(0, 0.42); b.yaxis.set_major_formatter(PCT); b.set_yticks([0, .1, .2, .3, .4]); b.set_xlim(-0.35, len(mk) - 0.65)
    label(b, 0, mk.share.iloc[0], f"{100 * mk.share.iloc[0]:.0f}%", dy=-12); label(b, x[-1], mk.share.iloc[-1], f"{100 * mk.share.iloc[-1]:.0f}%", dy=-12)
    fig.text(0.125, 0.02, "31,123 cs.CL abstracts, 2015 to October 2026. Shares are of subjects classified as an AI system or a person.\n"
             "Panel b: the five cognitive verbs, AI-system subjects only. Bands: 95% Wilson intervals.", color=MUTED, fontsize=7.5, va="bottom")
    save(fig, "fig3_verb_subjects")


if __name__ == "__main__":
    import sys
    which = sys.argv[1:] or ["1", "2", "3"]
    for w in which:
        globals()[f"fig{w}"]()
