# CRISOL-MBA-x64 — Artifact Evaluation & Reproduction Package (v1.0.0)

Official artifact repository for **"CRISOL-MBA-x64: Certified Neuro-Symbolic Deobfuscation of x86-64 Machine-Code Slices via Hybrid CEGIS and Lean 4 `bv_decide`"**.

- **Author:** Luis Alejandro Arancibia (Computer Engineer, FIUBA / UBA · Independent Researcher)
- **Canonical Paper PDF:** [`paper/paper-en.pdf`](paper/paper-en.pdf) (compiled directly from [`paper/arxiv-v1/main.tex`](paper/arxiv-v1/main.tex))
- **Reserved Zenodo DOI:** `10.5281/zenodo.24476603` (resolves upon Zenodo publication)
- **License:** MIT ([`LICENSE`](LICENSE))

---

## 1. System Requirements & Toolchain Setup (Linux x86-64)

### Python Environment (`>= 3.11`)
```bash
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -e .
```

### External Formal Verifiers (`build/`)
`replay_offline.py` and `run_phase2.py` expect the following local binaries under `build/` (excluded from Git via `.gitignore`):
1. **`CaDiCaL` (`build/cadical/build/cadical`)** — CaDiCaL `>= 2.0` / `3.1.0` compiled from source (`git clone https://github.com/arminbiere/cadical build/cadical && cd build/cadical && ./configure && make`).
2. **`lrat-check` (`build/drat-trim/lrat-check`)** — Independent LRAT proof checker (`git clone https://github.com/marijnheule/drat-trim build/drat-trim && make -C build/drat-trim lrat-check`).
3. **Lean 4.36.0 (`build/elan`)** — Install `elan` under `build/elan` and set the default toolchain to `leanprover/lean4:v4.36.0` (`ELAN_HOME=$PWD/build/elan ./build/elan/bin/elan default leanprover/lean4:v4.36.0`).

---

## 2. Deterministic Paper & Checksum Verification (Python Stdlib + Git; Zero External Solvers)

Verify every cell of Tables 2–4, strict wall-clock cutoffs, subset counts, all 20 document $p$-values (17 McNemar + 3 family sign-test values), all 58 fractions and 67 percentages across `main.tex`, cross-consistency across all raw `.jsonl`/`.csv` logs and `data/summary_gamba_vtest.json`, byte-for-byte `arxiv-crisol-mba-x64-v1.tar.gz` integrity, and all three SHA-256 manifests (`data/manifest_V.sha256`, `data/manifest_V_real.sha256`, `docs/SHA256SUMS.txt`).

By default, `verify_paper_claims.py` also invokes `git cat-file -e` to verify that commit `8008d30` cited in `main.tex` exists in repository history (run inside a full Git clone of `larancibia/crisol-mba-x64`, not `--depth 1`; if running from an extracted Zenodo `.zip` snapshot without `.git` history, pass `--allow-no-git`):

```bash
python3 verify_paper_claims.py
# Or from an extracted Zenodo archive without .git history:
python3 verify_paper_claims.py --allow-no-git
```

---

## 3. Test Suite, Offline Certificate Replay & Figure Regeneration (Zero LLM API Keys)

Using the Python `.venv` and `build/` verifiers from Section 1, run the 38-test suite (lifter, Unicorn/CPU 3-way checks, `0x67` `LEA` wrapping, LRAT, and Lean 4 `bv_decide`) and re-verify cached `Hybrid H` outputs (`--mode smoke`: 12 slices in ~30 s; `--mode full`: all 228 archived `H` passes — 182 `H`-Hydrogen + 46 `H`-Sol — across `V_test` and `V_real`, checking canonical cost, LRAT with 60 s Z3/Bitwuzla/CaDiCaL solver caps [in-process Z3 `goal.dimacs()` bit-blasting runs uncapped], and Lean 4 `bv_decide` with 60 s theorem caps across 2 worker processes by default) while regenerating `paper/fig1..fig3`:

```bash
.venv/bin/pytest -q
.venv/bin/python replay_offline.py --mode smoke
```
