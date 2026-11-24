# Related Work Survey (2013–2026) — CRISOL-MBA-x64

## 1. Scope and Positioning

Mixed Boolean-Arithmetic (MBA) deobfuscation spans three methodological lines: algebraic AST simplifiers, stochastic/black-box program synthesis, and neural sequence/decompilation models. None of the prior systems surveyed below simultaneously combines:

1. **Direct x86-64 machine-code byte input** preserving explicit Carry Flag (`CF`) chains (`ADC`/`SBB`), 32-to-64-bit implicit zero-extension, arithmetic shifts (`SAR`), and bit rotations (`ROL`/`ROR`), empirically cross-checked against native x86-64 execution and Unicorn emulation.
2. **Counterexample-Guided Inductive Synthesis (CEGIS)** pairing neural candidate generation with concrete distinguishing inputs extracted from dual `QF_BV` SMT solvers (`Bitwuzla` + `Z3`) under controlled ablation against non-neural search.
3. **Explicit separation between search hardness and equivalence-verification hardness** (`V_test` vs. `V_hard`).
4. **Solver-independent proof certificates** checked either via `CaDiCaL` + `lrat-check` (LRAT) or inside **Lean 4** via `bv_decide` (removing bit-blasting from the external solver TCB).

---

## 2. Comparative Literature Matrix

### A. Algebraic and Synthesis-Based MBA Deobfuscators (2021–2026)

| System | Authors | Venue / Year | Approach | Limitation Relative to CRISOL-MBA-x64 |
|---|---|---|---|---|
| **MBA-Blast** | Liu, Shen, Ming, Zheng, Li, Xu | USENIX Security 2021 | 1-bit truth-table reduction for linear MBA expressions. | Linear AST expressions only; no x86-64 binary semantics (`CF`, 32-to-64-bit zero-extension) or proof certificates. |
| **Xyntia** | Menguy, Bardin, Bonichon, de Souza Lima | ACM CCS 2021 | Black-box local/MCMC search guided by concrete I/O traces. | No neural prior and probabilistic I/O equivalence unless paired with external verification (modeled in baseline `C`). |
| **SiMBA** | Reichenwallner, Meerwald-Stadler | ACM CCS CheckMATE 2022 | Linear MBA simplification over $n$-bit space without 1-bit expansion. | Restricted to linear MBA expression ASTs (modeled in baseline `A`). |
| **GAMBA** | Reichenwallner, Meerwald-Stadler | IEEE EuroS&PW 2023 | Extends SiMBA to nonlinear/general MBA ASTs via temporary substitution. | Operates on symbolic expression ASTs rather than x86-64 byte slices with flags/rotations; optional uncertified Z3 check. |
| **ProMBA** | Lee, Lee, Cho, Lee | ACM CCS 2023 / ACM TOSEM 2026 | Program synthesis + term rewriting / equality saturation on MBA ASTs. | Operates on symbolic MBA ASTs without binary flag lifting or checkable SAT/Lean certificates. |
| **RUMBA** | Royer, Tronel, Vinçont | HAL 2026 (`hal-05578742`) | Unified algebraic framework for linear and polynomial MBA simplification. | Purely algebraic AST simplifier; does not lift x86-64 machine-code slices or emit formal proof certificates. |

### B. Neural Decompilation and MBA Simplification (2024–2025)

| System | Authors | Venue / Year | Approach | Limitation Relative to CRISOL-MBA-x64 |
|---|---|---|---|---|
| **LLM4Decompile** | Tan, Luo, Li, Zhang | EMNLP 2024 | Open-weight LLMs translating assembly into readable C code. | Evaluates finite test-case re-executability rather than universal bit-vector equivalence across all $2^{64}$ inputs. |
| **gMBA** | Roh, Paik, Kwon, Cho | Findings of ACL 2025 | Transformer encoder-decoder guided by internal truth-table representations. | Operates on MBA expression strings without SMT counterexample feedback, x86-64 flag/register lifting, or proof certificates. |

### C. x86-64 Superoptimization and Certified Bit-Vector Reasoning

| System | Authors | Venue / Year | Relevance to CRISOL-MBA-x64 |
|---|---|---|---|
| **STOKE** | Schkufza, Sharma, Aiken | ASPLOS 2013 | Foundational MCMC superoptimization over straight-line x86-64 with SMT validation; motivates baseline `C`. |
| **Lean 4 `bv_decide` + LRAT** | Moura, Ullrich et al. / Cruz-Filipe et al. / Biere et al. | CADE 2021 / TACAS 2017 / CAV 2024 | Verified bit-blasting + CaDiCaL LRAT checking inside Lean 4 (`bv_decide`), enabling certified `IMUL` proofs on `V_hard` and `V_real`. |
