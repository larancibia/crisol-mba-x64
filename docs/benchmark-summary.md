# Empirical Benchmark Summary — CRISOL-MBA-x64 (v1.0.0)

Canonical summary of experimental outcomes across Benchmarks 1–3 (full analysis, Trusted Computing Base boundaries, and statistical tests in [`paper/paper-en.pdf`](../paper/paper-en.pdf)).

---

## 1. Benchmark 1 — Held-Out x86-64 Ablation (`V_test`, `N = 160`, 8 Disjoint Templates)

Wall-clock budget: `60.0 s` loop-entry cap (`Strict <= 60 s` applies a post-hoc `elapsed_sec <= 60.0 s` cutoff; `In-Flight` counts slices whose final verification step started before `60.0 s`).

| Configuration | Description | Strict `<= 60 s` | Strict (%) | In-Flight | In-Flight (%) | Mean / Max (s) | Mean LLM Calls |
|---|---|---:|---:|---:|---:|---:|---:|
| `A` | SiMBA + cost-bounded enumerator (LRAT) | 20 / 160 | 12.50 | 20 / 160 | 12.50 | 0.88 / 3.27 | 0.00 |
| `GAMBA` | Official Denuvo GAMBA (`simplified` on 2 AST families, unverified) | 40 / 160 | 25.00 | 40 / 160 | 25.00 | --- | 0.00 |
| `C` | Xyntia/STOKE MCMC-CEGIS (No-LLM, LRAT) | 39 / 160 | 24.38 | 42 / 160 | 26.25 | 50.50 / 65.75 | 0.00 |
| `A ∪ C` | Virtual best non-LLM baseline (certified) | 39 / 160 | 24.38 | 42 / 160 | 26.25 | --- | 0.00 |
| `A ∪ C ∪ GAMBA` | Union including GAMBA `simplified` (unverified) | 55 / 160 | 34.38 | 56 / 160 | 35.00 | --- | 0.00 |
| `B` | LLM One-Shot (`gemini-4-hydrogen`, `K = 8`, LRAT) | 154 / 160 | 96.25 | 154 / 160 | 96.25 | 25.29 / 68.30* | 1.00 |
| `D` | Pre-registered CEGIS (`4 x 2`, LRAT) | 145 / 160 | 90.62 | 146 / 160 | 91.25 | 26.65 / 63.96 | 1.32 |
| `H` (Hydrogen) | Exploratory Hybrid (`6 -> 2 x 2`, LRAT) | 159 / 160 | 99.38 | 160 / 160 | 100.00 | 29.60 / 74.29 | 1.01 |
| `H` (GPT-6.2-Sol) | Exploratory stratified cross-family (`N = 40`, LRAT) | 40 / 40 | 100.00 | 40 / 40 | 100.00 | 28.51 / 38.71 | 1.00 |

- **Primary Hypothesis $H_1$ (`D` vs. `A ∪ C`): CONFIRMED** (`+66.25 pp` strict, exact McNemar $p = 2.47 \times 10^{-32}$; `121/129` vs. `32/129`, $p = 3.23 \times 10^{-27}$ on `gt_cert_sec <= 10.0 s`; `7` family wins, `1` tie, `0` losses, two-sided sign test $p = 0.0156$; `105/120` vs. `19/120`, $p = 2.58 \times 10^{-26}$ excluding `nand_nor_add_64` and `sar_neg_32`).
- **Secondary Hypothesis $H_2$ (`D > B`): NOT CONFIRMED** (`145/160` strict vs. `154/160`, McNemar $p = 0.078$ strict; $p = 0.115$ in-flight). Of the `6` slices solved by `D` and missed by `B`, `4` succeeded in Round 1 and `2` in Round 2 following a Round 1 call with `0` parseable candidates. The union **`B ∪ D` reached `160/160 (100.0%)` under both strict `<= 60.0 s` and in-flight cutoffs**, motivating exploratory Hybrid `H` (`159/160` strict `<= 60.0 s`; `160/160` in-flight).

---

## 2. Benchmark 2 — Multiplication Certification Frontier (`V_hard`, `N = 40` `IMUL` Slices)

Ground-truth equivalence proofs ($F_B \equiv g$) under equalized `30.0 s` solver caps (strict `<= 30.0 s` recorded solver/certifier time; in-flight LRAT count in brackets):

| Family (`n = 20` each) | Z3 `QF_BV` | CaDiCaL LRAT (in-flight) | Bitwuzla (*uncertified*) | Lean 4 `bv_decide` | Mean Lean 4 (all / cert.) |
|---|---:|---:|---:|---:|---:|
| `test_mba_poly_mul_32` | 0 / 20 (0%) | 2 / 20 (10%) [3/20] | 20 / 20 (100%) | **20 / 20 (100%)** | 5.41 s / 5.41 s |
| `test_mba_poly_mul_64` | 0 / 20 (0%) | 0 / 20 (0%) [0/20] | 20 / 20 (100%) | **15 / 20 (75%)** | 10.57 s / 8.70 s |
| **Total `V_hard` (`N = 40`)** | **0 / 40 (0%)** | **2 / 40 (5.0%) [3/40]** | **40 / 40 (100%)** | **35 / 40 (87.5%)** | **7.99 s / 6.82 s** |

---

## 3. Benchmark 3 — Real `gcc -O2` / `clang -O2` Compiled C Corpus (`V_real`, `N = 22`)

| Configuration | Certificate Oracle | PASS / 22 | Rate (%) | Mean Time (s) | Exact McNemar vs. `A ∪ C` |
|---|---|---:|---:|---:|---|
| `A` (SiMBA + Enum) | LRAT | 0 / 22 | 0.00 | 2.28 | --- |
| `C` (MCMC-CEGIS) | LRAT | 0 / 22 | 0.00 | 60.02* | --- |
| `H-sol` (`gpt-6.2-sol`) | Lean 4 `bv_decide` | 6 / 22 | 27.27 | 55.12 | $p = 0.0313$ (n.s. at $\alpha = 0.01$) |
| **`H-hydrogen`** (`gemini-4-hydrogen`) | **Lean 4 `bv_decide`** | **22 / 22** | **100.00** | **23.44** | **$p = 4.77 \times 10^{-7}$** |
