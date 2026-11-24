#!/usr/bin/env python3
"""Generate reproducible publication-grade figures for CRISOL-MBA-x64 directly from raw JSON/CSV data."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
PAPER_DIR = REPO_ROOT / "paper"
PAPER_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 300,
})


def fig1_vtest_ablation() -> None:
    s_test = json.loads((DATA_DIR / "summary_test.json").read_text())
    s_gamba = json.loads((DATA_DIR / "summary_gamba_vtest.json").read_text())
    h_rows = [json.loads(l) for l in (DATA_DIR / "results_test_H_hydrogen.jsonl").read_text().splitlines() if l.strip()]

    fams = sorted(s_test["per_family"].keys())
    short_labels = [
        f.replace("test_mba_", "").replace("_64", "\n(64b)").replace("_32", "\n(32b)")
        for f in fams
    ]

    a_vals = [100.0 * s_test["per_family"][f]["A"] / 20 for f in fams]
    g_vals = [100.0 * s_gamba["by_family"][f]["gamba_simplified"] / 20 for f in fams]
    c_vals = [100.0 * s_test["per_family"][f]["C"] / 20 for f in fams]
    b_vals = [100.0 * s_test["per_family"][f]["B"] / 20 for f in fams]
    d_vals = [100.0 * s_test["per_family"][f]["D"] / 20 for f in fams]
    h_vals = [
        100.0 * sum(1 for r in h_rows if r["family"] == f and r["passed"]) / 20
        for f in fams
    ]

    x = np.arange(len(fams))
    w = 0.13
    fig, ax = plt.subplots(figsize=(11.5, 4.6))

    ax.bar(x - 2.5*w, a_vals, w, label="A: SiMBA + Enum", color="#94a3b8")
    ax.bar(x - 1.5*w, g_vals, w, label="GAMBA (Official, uncert.)", color="#64748b")
    ax.bar(x - 0.5*w, c_vals, w, label="C: MCMC-CEGIS (No-LLM)", color="#cbd5e1", edgecolor="#475569", linewidth=0.6)
    ax.bar(x + 0.5*w, b_vals, w, label="B: LLM One-Shot (K=8)", color="#60a5fa")
    ax.bar(x + 1.5*w, d_vals, w, label="D: CRISOL CEGIS (4×2)", color="#2563eb")
    ax.bar(x + 2.5*w, h_vals, w, label="H: CRISOL Hybrid (6→2×2)", color="#0f766e")

    ax.set_ylabel("Pass / Simplification Rate (%)\n(LRAT-certified; GAMBA uncert.)")
    ax.set_title("Benchmark 1 (V_test, N = 160): Certified Deobfuscation Rate Across 8 Disjoint x86-64 Families")
    ax.set_xticks(x)
    ax.set_xticklabels(short_labels, fontsize=8.5)
    ax.set_ylim(0, 112)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=9)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "fig1_vtest_ablation.png", bbox_inches="tight")
    fig.savefig(PAPER_DIR / "fig1_vtest_ablation.pdf", bbox_inches="tight")
    plt.close(fig)


def fig2_vhard_lean4() -> None:
    import csv
    vh_rows = list(csv.DictReader((DATA_DIR / "results_vhard_cert.csv").open()))
    fams = ["test_mba_poly_mul_32", "test_mba_poly_mul_64"]
    labels = ["32-bit IMUL + MBA\n(n = 20)", "64-bit IMUL + MBA\n(n = 20)"]

    def pct_strict30(fam: str, ok_col: str, sec_col: str) -> float:
        sub = [r for r in vh_rows if r["family"] == fam]
        return 100.0 * sum(1 for r in sub if r[ok_col] == "True" and float(r[sec_col]) <= 30.0) / 20.0

    z3_v = [pct_strict30(f, "z3_unsat", "z3_sec") for f in fams]
    lrat_v = [pct_strict30(f, "cadical_lrat_ok", "cadical_lrat_sec") for f in fams]
    lean_v = [pct_strict30(f, "lean4_bv_decide_ok", "lean4_bv_decide_sec") for f in fams]
    bz_v = [pct_strict30(f, "bitwuzla_unsat", "bitwuzla_sec") for f in fams]

    x = np.arange(len(fams))
    w = 0.18
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    b1 = ax.bar(x - 1.5*w, z3_v, w, label="Z3 QF_BV (unverified)", color="#ef4444")
    b2 = ax.bar(x - 0.5*w, lrat_v, w, label="Z3 bit-blast + CaDiCaL LRAT", color="#f97316")
    b3 = ax.bar(x + 0.5*w, lean_v, w, label="Lean 4 bv_decide (certified)", color="#0f766e")
    b4 = ax.bar(x + 1.5*w, bz_v, w, label="Bitwuzla QF_BV (unverified)", color="#94a3b8", hatch="//", edgecolor="#334155")

    for bars in (b1, b2, b3, b4):
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.0f}%", (bar.get_x() + bar.get_width()/2, h + 1.5), ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    ax.set_ylabel("Equivalence Proofs Closed (%)")
    ax.set_title("Benchmark 2 (V_hard, N = 40 IMUL Slices): Certified Lean 4 bv_decide vs Raw SAT Bit-Blasting")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 116)
    ax.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18), frameon=False, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "fig2_vhard_lean4.png", bbox_inches="tight")
    fig.savefig(PAPER_DIR / "fig2_vhard_lean4.pdf", bbox_inches="tight")
    plt.close(fig)


def fig3_real_and_tradeoff() -> None:
    s_real = json.loads((DATA_DIR / "summary_real_o2.json").read_text())
    s_test = json.loads((DATA_DIR / "summary_test.json").read_text())
    s_h = json.loads((DATA_DIR / "summary_test_H.json").read_text())
    t_rows = [json.loads(l) for l in (DATA_DIR / "results_test.jsonl").read_text().splitlines() if l.strip()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.2, 4.2))

    # Left panel: Real compiled C (-O2) V_real
    cfgs = ["A\n(SiMBA+Enum)", "C\n(MCMC-CEGIS)", "H-Sol\n(GPT-6.2-Sol)", "H-Hydrogen\n(Gemini 4)"]
    vals = [
        s_real["configs"]["A"]["rate_pct"],
        s_real["configs"]["C"]["rate_pct"],
        s_real["configs"]["H_sol"]["rate_pct"],
        s_real["configs"]["H_hydrogen"]["rate_pct"],
    ]
    colors = ["#94a3b8", "#cbd5e1", "#3b82f6", "#0f766e"]
    bars = ax1.bar(cfgs, vals, color=colors, width=0.55, edgecolor="#334155", linewidth=0.6)
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", (bar.get_x() + bar.get_width()/2, h + 1.8), ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_ylabel("Certified Pass Rate (%)\n(A, C: LRAT; H: Lean 4 bv_decide)")
    ax1.set_title(f"Benchmark 3: Real gcc/clang -O2 C Corpus (N = {s_real['n_v_real']})")
    ax1.set_ylim(0, 115)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.35)
    ax1.set_axisbelow(True)

    # Right panel: Pass Rate vs Mean Wall-Clock Latency on V_test (N=160)
    pts = []
    for cfg, lbl, col, mk in [
        ("A", "A (SiMBA+Enum)", "#64748b", "s"),
        ("C", "C (MCMC-CEGIS)", "#94a3b8", "^"),
        ("B", "B (One-Shot K=8)", "#60a5fa", "o"),
        ("D", "D (CEGIS 4×2)", "#2563eb", "D"),
    ]:
        crows = [r for r in t_rows if r["config"] == cfg]
        avg_t = sum(r["elapsed_sec"] for r in crows) / len(crows)
        pts.append((avg_t, s_test["pass_rates_pct"][cfg], lbl, col, mk))
    pts.append((
        s_h["H_hydrogen_V_test"]["avg_time_sec"],
        s_h["H_hydrogen_V_test"]["rate_pct"],
        "H (Hybrid 6→2×2)",
        "#0f766e",
        "*",
    ))

    for t, r, lbl, col, mk in pts:
        ms = 14 if mk == "*" else 9
        ax2.plot(t, r, marker=mk, markersize=ms, color=col, linestyle="None", label=lbl)
        ax2.annotate(lbl.split(" ")[0], (t + 1.2, r - 2.0), fontsize=9, fontweight="bold", color=col)

    ax2.set_xlabel("Mean Wall-Clock Time per Slice (s, loop-entry 60s budget)")
    ax2.set_ylabel("LRAT-Certified Pass Rate on V_test (%, in-flight)")
    ax2.set_title("Accuracy vs. Latency Trade-off on V_test (N = 160)")
    ax2.set_xlim(-2, 62)
    ax2.set_ylim(0, 108)
    ax2.grid(True, linestyle="--", alpha=0.35)
    ax2.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.22), frameon=False, fontsize=8)

    fig.tight_layout()
    fig.savefig(PAPER_DIR / "fig3_real_o2_and_tradeoff.png", bbox_inches="tight")
    fig.savefig(PAPER_DIR / "fig3_real_o2_and_tradeoff.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    fig1_vtest_ablation()
    fig2_vhard_lean4()
    fig3_real_and_tradeoff()
    print("FIGURES_OK:", sorted(p.name for p in PAPER_DIR.glob("fig*")))
