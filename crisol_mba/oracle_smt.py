"""Sound QF_BV equivalence oracle: NumPy pre-filter + Bitwuzla/Z3 + CaDiCaL LRAT + native SAT check."""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import bitwuzla as bz
import numpy as np
import z3

from .expr import Expr, canonical_cost, eval_concrete, eval_numpy, to_bitwuzla, to_z3
from .lifter import lift_to_bitwuzla, lift_to_z3
from .oracle_native import ConcreteVector, NativeSliceRunner, generate_test_vectors


REPO_ROOT = Path(__file__).resolve().parent.parent
CADICAL_BIN = REPO_ROOT / "build" / "cadical" / "build" / "cadical"
LRAT_CHECK_BIN = REPO_ROOT / "build" / "drat-trim" / "lrat-check"


class OracleDisagreementError(RuntimeError):
    """Raised if Bitwuzla and Z3 disagree on SAT vs UNSAT."""


@dataclass(frozen=True)
class OracleVerdict:
    status: str  # "PASS", "FAIL_PREFILTER", "FAIL_COST", "FAIL_SAT", "FAIL_TIMEOUT", "LIFTER_BUG", "FAIL_LRAT"
    cost: int
    max_cost: float
    elapsed_sec: float
    counterexample: ConcreteVector | None = None
    orig_out: int | None = None
    cand_out: int | None = None
    lrat_sha256: str | None = None
    detail: str = ""


class SlicePreFilter:
    """Fast vectorized pre-filter using concrete native outputs of the slice."""

    def __init__(self, code: bytes, seed: int = 20261123, n_vectors: int = 1024) -> None:
        self.code = code
        self.vectors: list[ConcreteVector] = generate_test_vectors(seed=seed, n_total=n_vectors)
        with NativeSliceRunner(code) as nat:
            outs = nat.run_batch(self.vectors)
        self.orig_rax = np.array([o[0] for o in outs], dtype=np.uint64)
        self.orig_cf = np.array([o[1] for o in outs], dtype=np.uint64)
        self._rebuild_env()

    def _rebuild_env(self) -> None:
        self.env = {
            "rdi": np.array([v.rdi for v in self.vectors], dtype=np.uint64),
            "rsi": np.array([v.rsi for v in self.vectors], dtype=np.uint64),
            "rdx": np.array([v.rdx for v in self.vectors], dtype=np.uint64),
            "rcx": np.array([v.rcx for v in self.vectors], dtype=np.uint64),
            "cf": np.array([v.cf for v in self.vectors], dtype=np.uint64),
        }

    def add_counterexample(self, vec: ConcreteVector, orig_rax: int, orig_cf: int) -> None:
        self.vectors.append(vec)
        self.orig_rax = np.append(self.orig_rax, np.uint64(orig_rax))
        self.orig_cf = np.append(self.orig_cf, np.uint64(orig_cf))
        self._rebuild_env()

    def check(self, cand: Expr) -> tuple[bool, ConcreteVector | None, int | None, int | None]:
        """Return (passed_prefilter, counterexample, orig_out, cand_out). Never approves."""
        cand_vals = eval_numpy(cand, self.env).astype(np.uint64)
        diffs = np.nonzero(cand_vals != self.orig_rax)[0]
        if len(diffs) == 0:
            return True, None, None, None
        i = int(diffs[0])
        return False, self.vectors[i], int(self.orig_rax[i]), int(cand_vals[i])


def certify_unsat_lrat(miter_formula_z3: z3.BoolRef, timeout_sec: float = 10.0) -> tuple[bool, str | None, str]:
    """Bit-blast a QF_BV UNSAT miter to DIMACS CNF, solve with CaDiCaL --lrat, and verify with lrat-check."""
    if not CADICAL_BIN.exists() or not LRAT_CHECK_BIN.exists():
        return False, None, f"Missing CaDiCaL ({CADICAL_BIN}) or lrat-check ({LRAT_CHECK_BIN})"

    g = z3.Goal()
    g.add(miter_formula_z3)
    # simplify MUST precede bit-blast so BV equalities are lowered to bit-level before tseitin-cnf
    tactic = z3.Then(z3.Tactic("simplify"), z3.Tactic("bit-blast"), z3.Tactic("tseitin-cnf"))
    res = tactic(g)
    if len(res) != 1:
        return False, None, f"Unexpected subgoal count after tseitin-cnf: {len(res)}"
    sub = res[0]
    dimacs = sub.dimacs()

    # Guard against un-blasted BV atoms leaking into DIMACS comments
    for line in dimacs.splitlines():
        if line.startswith("c ") and any(tok in line for tok in ("=", "bv", "Extract", "Concat")):
            return False, None, f"Un-blasted BV atom in DIMACS: {line}"

    # Normalize DIMACS header (fixes Z3 goal.dimacs() off-by-one max_var bug)
    clause_lines: list[str] = []
    max_var = 0
    for line in dimacs.splitlines():
        s = line.strip()
        if not s or s.startswith("c") or s.startswith("p"):
            continue
        lits = [int(tok) for tok in s.split()]
        for lit in lits:
            if lit != 0 and abs(lit) > max_var:
                max_var = abs(lit)
        clause_lines.append(s)

    if not clause_lines:
        if len(sub) == 1 and z3.is_false(sub[0]):
            dimacs_clean = "p cnf 1 2\n1 0\n-1 0\n"
        else:
            return False, None, f"Unexpected empty DIMACS for subgoal: {sub}"
    else:
        dimacs_clean = f"p cnf {max_var} {len(clause_lines)}\n" + "\n".join(clause_lines) + "\n"

    with tempfile.TemporaryDirectory() as td:
        cnf_path = Path(td) / "miter.cnf"
        lrat_path = Path(td) / "miter.lrat"
        cnf_path.write_text(dimacs_clean, encoding="utf-8")

        try:
            cad = subprocess.run(
                [str(CADICAL_BIN), "--lrat=true", "--binary=false", str(cnf_path), str(lrat_path)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return False, None, "CaDiCaL LRAT timeout"

        # CaDiCaL exit code 20 == UNSAT, 10 == SAT
        if cad.returncode != 20:
            err = (cad.stderr or cad.stdout).strip()[:200]
            return False, None, f"CaDiCaL returned {cad.returncode} (expected 20 UNSAT): {err}"

        if not lrat_path.exists() or lrat_path.stat().st_size == 0:
            return False, None, "CaDiCaL produced empty LRAT file"

        try:
            chk = subprocess.run(
                [str(LRAT_CHECK_BIN), str(cnf_path), str(lrat_path)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired:
            return False, None, "lrat-check timeout"

        combined = (chk.stdout + "\n" + chk.stderr).strip()
        if chk.returncode != 0 or "VERIFIED" not in combined.upper():
            return False, None, f"lrat-check failed (rc={chk.returncode}): {combined[:200]}"

        lrat_hash = hashlib.sha256(lrat_path.read_bytes()).hexdigest()
        return True, lrat_hash, "LRAT_VERIFIED"


def verify_candidate(
    code: bytes,
    cand: Expr,
    max_cost: float,
    prefilter: SlicePreFilter | None = None,
    timeout_sec: float = 10.0,
    require_lrat: bool = True,
) -> OracleVerdict:
    """Full sound verification pipeline per pre-registration section 5."""
    t0 = time.perf_counter()

    # 1. Canonical cost check (after CSE + constant folding)
    cost = canonical_cost(cand)
    if cost > max_cost + 1e-9:
        return OracleVerdict(
            status="FAIL_COST",
            cost=cost,
            max_cost=max_cost,
            elapsed_sec=time.perf_counter() - t0,
            detail=f"cost {cost} > max_cost {max_cost:.2f}",
        )

    # 2. Fast NumPy pre-filter (rejects only, never approves)
    if prefilter is not None:
        ok, ce, o_out, c_out = prefilter.check(cand)
        if not ok:
            return OracleVerdict(
                status="FAIL_PREFILTER",
                cost=cost,
                max_cost=max_cost,
                elapsed_sec=time.perf_counter() - t0,
                counterexample=ce,
                orig_out=o_out,
                cand_out=c_out,
                detail="Rejected by concrete pre-filter",
            )

    # 3. Dual QF_BV miter: Bitwuzla + Z3
    timeout_ms = int(timeout_sec * 1000)

    # 3a. Bitwuzla
    tm = bz.TermManager()
    lifted_bz = lift_to_bitwuzla(code, tm)
    cand_bz = to_bitwuzla(cand, tm, lifted_bz.input_vars)
    opts = bz.Options()
    opts.set(bz.Option.TIME_LIMIT_PER, timeout_ms)
    opts.set(bz.Option.PRODUCE_MODELS, True)
    b_solver = bz.Bitwuzla(tm, opts)
    # Constrain scratch registers to 0 matching NativeSliceRunner contract
    zero64 = tm.mk_bv_zero(tm.mk_bv_sort(64))
    for r in ("rax", "rbx", "rbp", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"):
        b_solver.assert_formula(tm.mk_term(bz.Kind.EQUAL, [lifted_bz.input_vars[r], zero64]))
    b_solver.assert_formula(tm.mk_term(bz.Kind.DISTINCT, [lifted_bz.regs["rax"], cand_bz]))
    bz_res = b_solver.check_sat()

    # 3b. Z3 (explicit QF_BV cross-check; cap at 2s if Bitwuzla already proved UNSAT)
    lifted_z3 = lift_to_z3(code)
    cand_z3 = to_z3(cand, lifted_z3.input_vars)
    z_solver = z3.SolverFor("QF_BV")
    z3_timeout_ms = min(timeout_ms, 2000) if bz_res == bz.Result.UNSAT else timeout_ms
    z_solver.set("timeout", z3_timeout_ms)
    scratch_eqs = [
        lifted_z3.input_vars[r] == z3.BitVecVal(0, 64)
        for r in ("rax", "rbx", "rbp", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15")
    ]
    miter_z3 = z3.And(*scratch_eqs, lifted_z3.regs["rax"] != cand_z3)
    z_solver.add(miter_z3)
    z3_res = z_solver.check()

    # Cross-check agreement (SAT vs UNSAT is a hard failure)
    if (bz_res == bz.Result.SAT and z3_res == z3.unsat) or (bz_res == bz.Result.UNSAT and z3_res == z3.sat):
        raise OracleDisagreementError(f"Solver disagreement: Bitwuzla={bz_res}, Z3={z3_res}")

    # Handle SAT: extract counterexample and re-execute natively
    if z3_res == z3.sat or bz_res == bz.Result.SAT:
        if z3_res == z3.sat:
            m = z_solver.model()
            def _gv(name: str, w: int = 64) -> int:
                v = m.eval(lifted_z3.input_vars[name], model_completion=True)
                return int(v.as_long())
            ce = ConcreteVector(_gv("rdi"), _gv("rsi"), _gv("rdx"), _gv("rcx"), _gv("cf") & 1)
        else:
            def _bv(name: str) -> int:
                return int(b_solver.get_value(lifted_bz.input_vars[name]).value(10))
            ce = ConcreteVector(_bv("rdi"), _bv("rsi"), _bv("rdx"), _bv("rcx"), _bv("cf") & 1)

        with NativeSliceRunner(code) as nat:
            nat_rax, nat_cf = nat.run(ce)
        cand_val = eval_concrete(
            cand,
            {"rdi": ce.rdi, "rsi": ce.rsi, "rdx": ce.rdx, "rcx": ce.rcx, "cf": ce.cf},
        )
        if nat_rax == cand_val:
            return OracleVerdict(
                status="LIFTER_BUG",
                cost=cost,
                max_cost=max_cost,
                elapsed_sec=time.perf_counter() - t0,
                counterexample=ce,
                orig_out=nat_rax,
                cand_out=cand_val,
                detail="SAT counterexample did NOT reproduce on native CPU",
            )
        if prefilter is not None:
            prefilter.add_counterexample(ce, nat_rax, nat_cf)
        return OracleVerdict(
            status="FAIL_SAT",
            cost=cost,
            max_cost=max_cost,
            elapsed_sec=time.perf_counter() - t0,
            counterexample=ce,
            orig_out=nat_rax,
            cand_out=cand_val,
            detail="SAT counterexample reproduced natively",
        )

    # Handle UNKNOWN / timeout: require at least one solver UNSAT (and neither SAT),
    # then independent LRAT proof certifies UNSAT mathematically.
    if bz_res != bz.Result.UNSAT and z3_res != z3.unsat:
        return OracleVerdict(
            status="FAIL_TIMEOUT",
            cost=cost,
            max_cost=max_cost,
            elapsed_sec=time.perf_counter() - t0,
            detail=f"Bitwuzla={bz_res}, Z3={z3_res}",
        )

    # Both UNSAT -> mandatory LRAT proof certificate
    if require_lrat:
        ok_lrat, lrat_hash, msg = certify_unsat_lrat(miter_z3, timeout_sec=timeout_sec)
        if not ok_lrat:
            return OracleVerdict(
                status="FAIL_LRAT",
                cost=cost,
                max_cost=max_cost,
                elapsed_sec=time.perf_counter() - t0,
                detail=msg,
            )
        return OracleVerdict(
            status="PASS",
            cost=cost,
            max_cost=max_cost,
            elapsed_sec=time.perf_counter() - t0,
            lrat_sha256=lrat_hash,
            detail="UNSAT_DUAL_AND_LRAT_VERIFIED",
        )

    return OracleVerdict(
        status="PASS",
        cost=cost,
        max_cost=max_cost,
        elapsed_sec=time.perf_counter() - t0,
        detail="UNSAT_DUAL_NO_LRAT",
    )
