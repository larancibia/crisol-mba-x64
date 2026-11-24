# Experimental Protocol Specification — CRISOL-MBA-x64

*Document version: 1.0 (Study Protocol & Post-Freeze Implementation Notes) · Author: Luis Alejandro Arancibia*

---

## 1. Research Question and Falsifiable Hypotheses

### Central Question
Given a straight-line, register-only **x86-64** machine-code byte slice ($\le 24$ instructions, no memory operands, branches, or system calls) encoding a linear or degree-$\le 2$ polynomial Mixed Boolean-Arithmetic (MBA) computation in 32- or 64-bit arithmetic with Carry Flag (`CF`) propagation and 32-to-64-bit zero-extension, does **neuro-symbolic CEGIS (`D`)** recover a certified simplified expression $e \in \mathcal{G}$ with canonical cost $\mathrm{Cost}(e) \le (1 + \varepsilon)\,\mathrm{Cost}(g)$ and an independently checked **LRAT** proof certificate on a significantly higher fraction of instances than the instance-wise virtual best of non-neural baselines (`A ∪ C`) under an identical wall-clock and solver-call budget?

### Primary Hypothesis ($H_1$)
On the held-out frozen benchmark $V_{\mathrm{test}}$ ($N \ge 150$ instances from identity templates disjoint from development):
- **Minimum effect size:** $\mathrm{Rate}(D) - \mathrm{Rate}(A \cup C) \ge +15.0$ percentage points, where $A \cup C$ denotes the instance-wise union of algebraic/enumerative baseline `A` and non-neural MCMC-CEGIS baseline `C`.
- **Statistical test:** Exact two-sided paired **McNemar** test at significance level $\alpha = 0.01$ (paired with a family-level exact sign test across disjoint templates).
- **Canonical cost margin:** $\varepsilon = 0.10$ relative to the un-obfuscated reference expression $g$, measured after Common Subexpression Elimination (CSE) and constant folding on both sides.

### Secondary Hypothesis ($H_2$ — Counterexample Feedback Ablation)
Under identical wall-clock budget, sequential CEGIS with concrete SMT counterexamples (`D`, $4 \times 2$ candidates) significantly outperforms single-call multi-sample synthesis without counterexample feedback (`B`, $1 \times 8$ candidates) under an exact two-sided paired McNemar test at $\alpha = 0.01$.

### Refutation Criteria
$H_1$ is refuted if any of the following holds:
1. $\mathrm{Rate}(D) - \mathrm{Rate}(A \cup C) < 15.0\text{ pp}$ or paired McNemar $p \ge 0.01$.
2. Non-neural search `C` matches or exceeds `D` under identical concrete pre-filtering and verification budgets.
3. Even a single **false `PASS`** occurs on semantic mutants or pre-filter-evading negatives, or an unresolved discrepancy occurs against native x86-64 execution.

---

## 2. Closed x86-64 Execution Contract

- **Architecture:** x86-64 (System V register convention).
- **Length:** At most **24** straight-line instructions (`JMP`, `Jcc`, `CALL`, `RET`, `SYSCALL`, memory operands, and `RSP`/`ESP`/`SP`/`SPL` operands are deterministically rejected at decode time via `capstone`).
- **Inputs and outputs:** Inputs `(rdi, rsi, rdx, rcx, cf)` with the 11 remaining general-purpose registers (`rax`, `rbx`, `rbp`, `r8`–`r15`) initialized to `0`; output `rax` (64-bit).
- **Closed 17-mnemonic allowlist ($\mathcal{M}$):**
  - Data movement and register-only effective address computation: `MOV`, `LEA` (including `0x67` 32-bit address-size truncation/zero-extension per Intel SDM Table 3-54).
  - Arithmetic with and without carry/borrow: `ADD`, `ADC`, `SUB`, `SBB`, `NEG`, `IMUL` (non-nested on 64-bit).
  - Bitwise logic: `AND`, `OR`, `XOR`, `NOT`.
  - Immediate shifts and rotations: `SHL`/`SAL`, `SHR`, `SAR`, `ROL`, `ROR`.

---

## 3. Search vs. Verification Separation and Anti-Leakage Split

1. **Dataset partitions:** Four development identity templates (`V_dev`, $N = 40$) are strictly disjoint from ten held-out test templates: eight general MBA/carry/shift/rotate templates (`V_test`, $N = 160$) and two `IMUL` polynomial MBA templates (`V_hard`, $N = 40$).
2. **Cryptographic dataset freeze:** Assembled `.text` byte sequences, ground-truth ASTs, and instance metadata are frozen in `data/manifest_V.json` (`data/manifest_V.sha256`) and `data/manifest_V_real.json` (`data/manifest_V_real.sha256`).
3. **Ablation configurations (`60.0 s` budget, max `16` SMT calls):**
   - `A`: Exact linear SiMBA basis solver + cost-bounded bottom-up enumerator.
   - `B`: LLM One-Shot ($K = 8$ candidates in a single call, zero counterexamples).
   - `C`: Non-neural Xyntia/STOKE-style MCMC-CEGIS guided by concrete I/O distance and SMT counterexamples.
   - `D`: Sequential LLM CEGIS ($4$ rounds $\times 2$ candidates with concrete `QF_BV` counterexample feedback).
   - `H` (*exploratory post-hoc*): Wide One-Shot ($K = 6$) followed by $2 \times 2$ CEGIS fallback only if needed.

---

## 4. Zero-LLM-Trust Verification Pipeline

1. **Lifter Sanity Gate:** Every closed-list instruction test ($31 \times 1{,}024 = 31{,}744$ checks) and reference slice check ($50 \times 1{,}024 = 51{,}200$ checks across 28 generated and 22 `-O2` C slices) must match across native x86-64 execution, Unicorn emulation, and lifted semantics (`0` discrepancies across all `82,944` checks).
2. **Vectorized Concrete Pre-Filter (`numpy`):** Rejects candidates failing on boundary values, one-hot bit patterns, seeded random inputs, `CF` $\in \{0,1\}$, or accumulated counterexamples (rejects only; never approves).
3. **Dual `QF_BV` Miter + Native Counterexample Replay:** Checks $\exists \vec{x}.\ F_B(\vec{x}) \neq [\![e]\!](\vec{x})$ in both `Bitwuzla` and `Z3`; every `SAT` counterexample is re-executed natively on the x86-64 CPU (`0` `LIFTER_BUG` verdicts).
4. **Checkable Proof Certificates:** Every `PASS` requires either an independently verified **LRAT** proof (`CaDiCaL` + `lrat-check`) or a **Lean 4.36.0 `bv_decide`** theorem verification.

---

## 5. Disclosed Post-Freeze Implementation Deviations (Section 4 of Paper)

As documented in Section 4 of `paper/arxiv-v1/main.tex`:
1. During freeze, `run_experiment.py` admitted all `160` LRAT-certified slices into `V_test` (`129/160` had `gt_cert_sec <= 10.0 s`, while `31` took `10.0–16.45 s`); `H_1` is therefore reported on both `N = 160` and the strict `N = 129` subset.
2. `generator.py` instantiated `20` randomized immediate-constant variants per template with fixed register operands and byte-level deduplication; instance-level McNemar tests are therefore paired with a template-level sign test (`N = 8` families).
3. Baseline `A` implemented linear SiMBA + cost-bounded enumeration without degree-2 polynomial interpolation over $\mathbb{F}_2$ or solver simplification, and `A`/`C`'s search grammars omitted `ashr` (affecting `sar_neg_32`).
4. The prompt's syntax example (`xor(add(rdi, rsi), 0x1234)`) shares outer structure with `nand_nor_add_64`; a joint `N = 120` sensitivity analysis excluding both `nand_nor_add_64` and `sar_neg_32` is reported in Section 4.1.
