"""Ablation configurations A, B, C, D sharing identical grammar, cost metric, budget, and LRAT oracle."""

from __future__ import annotations

import itertools
import random
import re
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from .expr import Binary, Const, Expr, Ternary, Unary, Var, canonical_cost, eval_numpy, mask_for
from .lifter import decode_and_validate
from .oracle_native import ConcreteVector
from .oracle_smt import OracleVerdict, SlicePreFilter, verify_candidate
from .parser import ParseError, format_expr, parse_expr


@dataclass
class RunResult:
    config: str  # "A", "B", "C", "D"
    instance_id: str
    family: str
    passed: bool
    status: str
    elapsed_sec: float
    smt_calls: int
    llm_calls: int
    candidates_tried: int
    best_expr: str | None = None
    best_cost: int | None = None
    ground_truth_cost: int = 0
    lrat_sha256: str | None = None
    detail: str = ""


def extract_slice_constants(code: bytes) -> list[int]:
    """Extract immediate constants appearing in the slice disassembly (0 tokens)."""
    insns = decode_and_validate(code)
    consts: list[int] = [0, 1, 2]
    for d in insns:
        for m in re.findall(r"0x[0-9a-fA-F]+|\b\d+\b", d.op_str):
            v = int(m, 0)
            if v not in consts:
                consts.append(v)
    return consts[:12]


def extract_active_inputs(code: bytes) -> tuple[list[str], bool, int]:
    """Determine which input registers (rdi, rsi, rdx, rcx) and cf are read, plus output width hint."""
    insns = decode_and_validate(code)
    text = "\n".join(f"{d.mnemonic} {d.op_str}" for d in insns).lower()
    regs: list[str] = []
    for canon, aliases in [
        ("rdi", ("rdi", "edi", "di", "dil")),
        ("rsi", ("rsi", "esi", "si", "sil")),
        ("rdx", ("rdx", "edx", "dx", "dl", "dh")),
        ("rcx", ("rcx", "ecx", "cx", "cl", "ch")),
    ]:
        if any(re.search(rf"\b{a}\b", text) for a in aliases):
            regs.append(canon)
    if not regs:
        regs = ["rdi"]
    uses_cf = any(d.mnemonic in {"adc", "sbb"} for d in insns)
    # Check if final write to rax is 32-bit (eax)
    width_hint = 64
    for d in reversed(insns):
        first_op = d.op_str.split(",")[0].strip().lower() if d.op_str else ""
        if first_op == "eax":
            width_hint = 32
            break
        if first_op == "rax":
            width_hint = 64
            break
    return regs, uses_cf, width_hint


def _simba_linear_candidates(
    pf: SlicePreFilter,
    active_regs: list[str],
    width: int,
) -> list[Expr]:
    """SiMBA-style exact linear MBA solver over {0, 1}^k corners (Reichenwallner & Meerwald-Stadler 2022)."""
    # Works for 1, 2, or 3 variables over bitwise basis modulo 2^width
    if len(active_regs) > 3:
        return []
    k = len(active_regs)
    m = mask_for(width)
    # Evaluate slice on all 2^k corners with cf=0 using NativeSliceRunner via pf.code
    from .oracle_native import NativeSliceRunner
    corners = list(itertools.product((0, 1), repeat=k))
    vals: list[int] = []
    with NativeSliceRunner(pf.code) as nat:
        for bits in corners:
            kwargs = {"rdi": 0, "rsi": 0, "rdx": 0, "rcx": 0, "cf": 0}
            for r, b in zip(active_regs, bits):
                kwargs[r] = b
            rax, _ = nat.run(ConcreteVector(**kwargs))
            vals.append(rax & m)

    # Möbius transform on {0,1}^k gives exact linear coefficients c_S for conjunctions AND_{i in S} x_i
    # f(x) = sum_S c_S * (AND_{i in S} x_i) mod 2^w
    coeffs = list(vals)
    for i in range(k):
        bit = 1 << (k - 1 - i)
        for mask_idx in range(1 << k):
            if mask_idx & bit:
                coeffs[mask_idx] = (coeffs[mask_idx] - coeffs[mask_idx ^ bit]) & m

    # Build simplified AST from non-zero signed coefficients (keep small coefficients only)
    def to_signed_small(c: int) -> int | None:
        if c == 0:
            return 0
        if c <= 32:
            return c
        if (1 << width) - c <= 32:
            return c - (1 << width)
        return None

    def make_var(name: str) -> Expr:
        return Unary(32, "trunc32", Var(64, name)) if width == 32 else Var(64, name)

    terms: list[tuple[int, Expr | None]] = []
    const_term = coeffs[0]
    for mask_idx in range(1, 1 << k):
        sc = to_signed_small(coeffs[mask_idx])
        if sc is None:
            return []  # Non-linear or huge coeff; skip
        if sc == 0:
            continue
        vars_in_conj = [make_var(active_regs[i]) for i in range(k) if mask_idx & (1 << (k - 1 - i))]
        conj = vars_in_conj[0]
        for v in vars_in_conj[1:]:
            conj = Binary(width, "and", conj, v)
        terms.append((sc, conj))

    if not terms:
        e: Expr = Const(width, const_term)
        return [Unary(64, "zext64", e) if width == 32 else e]

    # Check common 2-term XOR recognition: x + y - 2*(x & y) == x ^ y
    # Also build direct sum of terms
    cands: list[Expr] = []
    acc: Expr | None = None
    for sc, term in terms:
        assert term is not None
        scaled = term if abs(sc) == 1 else Binary(width, "mul", term, Const(width, abs(sc)))
        if acc is None:
            acc = scaled if sc > 0 else Unary(width, "neg", scaled)
        else:
            acc = Binary(width, "add" if sc > 0 else "sub", acc, scaled)
    if acc is not None and const_term != 0:
        acc = Binary(width, "add", acc, Const(width, const_term))
    if acc is not None:
        cands.append(Unary(64, "zext64", acc) if width == 32 else acc)

    # Also recognize pure XOR combinations (e.g. coeff(x)=1, coeff(y)=1, coeff(x&y)=-2 => x^y)
    if k == 2:
        c1, c2, c12 = to_signed_small(coeffs[2]), to_signed_small(coeffs[1]), to_signed_small(coeffs[3])
        if c1 == 1 and c2 == 1 and c12 == -2:
            xr = Binary(width, "xor", make_var(active_regs[0]), make_var(active_regs[1]))
            if const_term != 0:
                xr = Binary(width, "add", xr, Const(width, const_term))
            cands.append(Unary(64, "zext64", xr) if width == 32 else xr)
    return cands


def run_config_A(
    instance_id: str,
    family: str,
    code: bytes,
    gt_cost: int,
    epsilon: float = 0.10,
    budget_sec: float = 30.0,
    max_smt_calls: int = 16,
) -> RunResult:
    """Configuration A: Deterministic baseline (SiMBA linear + F2 + cost-bounded bottom-up enumerator)."""
    t0 = time.perf_counter()
    max_cost = gt_cost * (1.0 + epsilon)
    pf = SlicePreFilter(code)
    active_regs, uses_cf, width_hint = extract_active_inputs(code)
    consts = extract_slice_constants(code)
    smt_calls = 0
    tried = 0

    def try_cand(e: Expr) -> OracleVerdict | None:
        nonlocal smt_calls, tried
        if time.perf_counter() - t0 >= budget_sec or smt_calls >= max_smt_calls:
            return None
        tried += 1
        if canonical_cost(e) > max_cost + 1e-9:
            return None
        ok, _, _, _ = pf.check(e)
        if not ok:
            return None
        smt_calls += 1
        rem = max(1.0, min(10.0, budget_sec - (time.perf_counter() - t0)))
        return verify_candidate(code, e, max_cost=max_cost, prefilter=pf, timeout_sec=rem, require_lrat=True)

    # 1. SiMBA linear candidates (64-bit and 32-bit)
    for w in ([32, 64] if width_hint == 32 else [64, 32]):
        for cand in _simba_linear_candidates(pf, active_regs, w):
            v = try_cand(cand)
            if v is not None and v.status == "PASS":
                return RunResult(
                    config="A", instance_id=instance_id, family=family, passed=True,
                    status="PASS", elapsed_sec=time.perf_counter() - t0,
                    smt_calls=smt_calls, llm_calls=0, candidates_tried=tried,
                    best_expr=format_expr(cand), best_cost=v.cost,
                    ground_truth_cost=gt_cost, lrat_sha256=v.lrat_sha256, detail="SiMBA",
                )

    # 2. Cost-bounded bottom-up enumerator over active regs + slice constants (cost 1..3)
    w = width_hint
    atoms: list[Expr] = [
        (Unary(32, "trunc32", Var(64, r)) if w == 32 else Var(64, r)) for r in active_regs
    ]
    const_atoms: list[Expr] = [Const(w, c) for c in consts[:8]]
    bin_ops = ("add", "sub", "xor", "and", "or", "mul", "shl", "lshr", "rol", "ror")

    level1: list[Expr] = []
    if uses_cf and len(atoms) >= 2:
        for op in ("adc", "sbb"):
            level1.append(Ternary(w, op, atoms[0], atoms[1], Var(1, "cf")))
    for op in ("add", "sub", "xor", "and", "or"):
        for a, b in itertools.combinations_with_replacement(atoms, 2):
            if a == b and op in {"sub", "xor", "and", "or"}:
                continue
            level1.append(Binary(w, op, a, b))
    for op in bin_ops:
        for a in atoms:
            for c in const_atoms:
                if op in {"shl", "lshr", "rol", "ror"} and (c.value == 0 or c.value >= w):
                    continue
                level1.append(Binary(w, op, a, c))

    for e in level1:
        cand = Unary(64, "zext64", e) if w == 32 else e
        v = try_cand(cand)
        if v is not None and v.status == "PASS":
            return RunResult(
                config="A", instance_id=instance_id, family=family, passed=True,
                status="PASS", elapsed_sec=time.perf_counter() - t0,
                smt_calls=smt_calls, llm_calls=0, candidates_tried=tried,
                best_expr=format_expr(cand), best_cost=v.cost,
                ground_truth_cost=gt_cost, lrat_sha256=v.lrat_sha256, detail="Enum-L1",
            )

    # Level 2: combine level1 with atoms or constants (bounded by budget_sec)
    for e1 in level1:
        if time.perf_counter() - t0 >= budget_sec or smt_calls >= max_smt_calls:
            break
        for op in ("add", "sub", "xor", "mul", "shl", "lshr", "rol", "ror"):
            rhs_pool = const_atoms if op in {"mul", "shl", "lshr", "rol", "ror"} else (atoms + const_atoms[:5])
            for rhs in rhs_pool:
                if isinstance(rhs, Const) and op in {"shl", "lshr", "rol", "ror"} and (rhs.value == 0 or rhs.value >= w):
                    continue
                e2 = Binary(w, op, e1, rhs)
                cand = Unary(64, "zext64", e2) if w == 32 else e2
                v = try_cand(cand)
                if v is not None and v.status == "PASS":
                    return RunResult(
                        config="A", instance_id=instance_id, family=family, passed=True,
                        status="PASS", elapsed_sec=time.perf_counter() - t0,
                        smt_calls=smt_calls, llm_calls=0, candidates_tried=tried,
                        best_expr=format_expr(cand), best_cost=v.cost,
                        ground_truth_cost=gt_cost, lrat_sha256=v.lrat_sha256, detail="Enum-L2",
                    )

    return RunResult(
        config="A", instance_id=instance_id, family=family, passed=False,
        status="FAIL_EXHAUSTED", elapsed_sec=time.perf_counter() - t0,
        smt_calls=smt_calls, llm_calls=0, candidates_tried=tried,
        ground_truth_cost=gt_cost,
    )


def _bit_error_score(cand: Expr, pf: SlicePreFilter) -> float:
    """Bit-level Hamming distance + exact mismatch penalty (Xyntia/STOKE objective)."""
    vals = eval_numpy(cand, pf.env).astype(np.uint64)
    diff = np.bitwise_xor(vals, pf.orig_rax)
    # Count mismatched elements + approximate popcount on low/high bytes
    mismatches = int(np.count_nonzero(diff))
    if mismatches == 0:
        return 0.0
    # Fast vectorized popcount via uint8 view
    pop = int(np.unpackbits(diff.view(np.uint8)).sum())
    return float(pop) / float(len(vals) * 64) + 0.1 * (mismatches / len(vals))


def _random_expr(
    rng: random.Random,
    atoms: list[Expr],
    consts: list[Expr],
    uses_cf: bool,
    w: int,
    depth: int,
) -> Expr:
    if depth <= 0:
        return rng.choice(atoms if rng.random() < 0.7 else (atoms + consts))
    r = rng.random()
    if uses_cf and r < 0.18 and len(atoms) >= 2:
        op = rng.choice(("adc", "sbb"))
        return Ternary(
            w, op,
            _random_expr(rng, atoms, consts, False, w, depth - 1),
            _random_expr(rng, atoms, consts, False, w, depth - 1),
            Var(1, "cf"),
        )
    if r < 0.25:
        op = rng.choice(("not", "neg"))
        return Unary(w, op, _random_expr(rng, atoms, consts, uses_cf, w, depth - 1))
    op = rng.choice(("add", "sub", "xor", "and", "or", "mul", "shl", "lshr", "rol", "ror"))
    left = _random_expr(rng, atoms, consts, uses_cf, w, depth - 1)
    if op in {"shl", "lshr", "rol", "ror"}:
        shift_c = [c for c in consts if 1 <= c.value < w] or [Const(w, rng.randint(1, min(31, w - 1)))]
        right = rng.choice(shift_c)
    else:
        right = _random_expr(rng, atoms, consts, False, w, depth - 1)
    return Binary(w, op, left, right)


def _mutate_expr(
    rng: random.Random,
    e: Expr,
    atoms: list[Expr],
    consts: list[Expr],
    uses_cf: bool,
    w: int,
) -> Expr:
    if rng.random() < 0.3:
        return _random_expr(rng, atoms, consts, uses_cf, w, rng.randint(1, 2))
    if isinstance(e, Binary):
        choice = rng.random()
        if choice < 0.35:
            new_op = rng.choice(("add", "sub", "xor", "and", "or", "mul"))
            return Binary(w, new_op, e.left, e.right)
        if choice < 0.68:
            return Binary(w, e.op, _mutate_expr(rng, e.left, atoms, consts, uses_cf, w), e.right)
        return Binary(w, e.op, e.left, _mutate_expr(rng, e.right, atoms, consts, False, w))
    if isinstance(e, Unary) and e.op in {"not", "neg"}:
        return Unary(w, e.op, _mutate_expr(rng, e.arg, atoms, consts, uses_cf, w))
    if isinstance(e, Ternary):
        op = "sbb" if e.op == "adc" else "adc"
        return Ternary(w, op, e.a, e.b, e.c)
    return _random_expr(rng, atoms, consts, uses_cf, w, 1)


def run_config_C(
    instance_id: str,
    family: str,
    code: bytes,
    gt_cost: int,
    epsilon: float = 0.10,
    budget_sec: float = 30.0,
    max_smt_calls: int = 16,
    seed: int = 20261123,
) -> RunResult:
    """Configuration C: CEGIS without LLM (Xyntia/STOKE-style MCMC stochastic search + LRAT oracle)."""
    t0 = time.perf_counter()
    max_cost = gt_cost * (1.0 + epsilon)
    pf = SlicePreFilter(code)
    active_regs, uses_cf, width_hint = extract_active_inputs(code)
    consts = [Const(width_hint, c) for c in extract_slice_constants(code)]
    atoms: list[Expr] = [
        (Unary(32, "trunc32", Var(64, r)) if width_hint == 32 else Var(64, r)) for r in active_regs
    ]
    rng = random.Random(seed ^ hash(instance_id))
    smt_calls = 0
    tried = 0

    best_e = _random_expr(rng, atoms, consts, uses_cf, width_hint, 2)
    best_score = _bit_error_score(Unary(64, "zext64", best_e) if width_hint == 32 else best_e, pf)
    restarts = 0

    while time.perf_counter() - t0 < budget_sec and smt_calls < max_smt_calls:
        if tried % 150 == 0 and tried > 0:
            # Periodic random restart to escape local minima
            restarts += 1
            best_e = _random_expr(rng, atoms, consts, uses_cf, width_hint, rng.randint(1, 3))
            best_score = _bit_error_score(Unary(64, "zext64", best_e) if width_hint == 32 else best_e, pf)

        cand_inner = _mutate_expr(rng, best_e, atoms, consts, uses_cf, width_hint)
        cand = Unary(64, "zext64", cand_inner) if width_hint == 32 else cand_inner
        tried += 1
        if canonical_cost(cand) > max_cost + 1e-9:
            continue
        score = _bit_error_score(cand, pf)
        if score == 0.0:
            smt_calls += 1
            rem = max(1.0, min(10.0, budget_sec - (time.perf_counter() - t0)))
            v = verify_candidate(code, cand, max_cost=max_cost, prefilter=pf, timeout_sec=rem, require_lrat=True)
            if v.status == "PASS":
                return RunResult(
                    config="C", instance_id=instance_id, family=family, passed=True,
                    status="PASS", elapsed_sec=time.perf_counter() - t0,
                    smt_calls=smt_calls, llm_calls=0, candidates_tried=tried,
                    best_expr=format_expr(cand), best_cost=v.cost,
                    ground_truth_cost=gt_cost, lrat_sha256=v.lrat_sha256,
                    detail=f"MCMC-CEGIS (restarts={restarts})",
                )
            # Counterexample was added to pf by verify_candidate; recompute score
            best_score = _bit_error_score(Unary(64, "zext64", best_e) if width_hint == 32 else best_e, pf)
        elif score < best_score or rng.random() < 0.05:
            best_e = cand_inner
            best_score = score

    return RunResult(
        config="C", instance_id=instance_id, family=family, passed=False,
        status="FAIL_EXHAUSTED", elapsed_sec=time.perf_counter() - t0,
        smt_calls=smt_calls, llm_calls=0, candidates_tried=tried,
        ground_truth_cost=gt_cost,
    )


# ---------------------------------------------------------------------------
# LLM Proponent Adapter (CLI-based, zero external API keys needed)
# ---------------------------------------------------------------------------

GRAMMAR_SPEC = """Use ONLY this functional grammar (one expression per line, NO prose, NO markdown fences):
- 64-bit vars: rdi, rsi, rdx, rcx | 1-bit carry flag: cf | constants: decimal or 0xHEX
- 64-bit ops: add(a,b), sub(a,b), mul(a,b), and(a,b), or(a,b), xor(a,b), not(a), neg(a), shl(a,k), lshr(a,k), ashr(a,k), rol(a,k), ror(a,k), adc(a,b,cf), sbb(a,b,cf)
- 32-bit ops (MUST wrap final 32-bit result in zext64(...) and inputs in trunc32(...)):
  add32, sub32, mul32, and32, or32, xor32, not32, neg32, shl32, lshr32, ashr32, rol32, ror32, adc32(a,b,cf), sbb32(a,b,cf)
Example 64-bit: xor(add(rdi, rsi), 0x1234)
Example 32-bit: zext64(add32(adc32(trunc32(rdi), trunc32(rsi), cf), 0x42))"""


def call_llm_cli(prompt: str, backend: str = "codex-sol", timeout_sec: float = 25.0) -> str:
    """Invoke a local CLI model cleanly in read-only / no-tools mode."""
    try:
        if backend == "codex-sol":
            proc = subprocess.run(
                [
                    "codex", "exec", "-m", "gpt-6.2-sol",
                    "-c", 'model_reasoning_effort="low"',
                    "--skip-git-repo-check", "--sandbox", "read-only",
                    prompt,
                ],
                capture_output=True, text=True, timeout=timeout_sec,
            )
            return proc.stdout
        if backend == "agy-hydrogen":
            proc = subprocess.run(
                [
                    "agy", "--model", "gemini-4-hydrogen-medium",
                    "--disable-slash-commands", "-p", prompt,
                ],
                capture_output=True, text=True, timeout=timeout_sec,
            )
            return proc.stdout
        raise ValueError(f"Unknown backend: {backend}")
    except subprocess.TimeoutExpired:
        return ""


def parse_llm_candidates(raw_output: str, max_k: int = 8) -> list[tuple[str, Expr]]:
    """Extract valid candidate expressions from raw LLM text."""
    results: list[tuple[str, Expr]] = []
    seen: set[Expr] = set()
    for line in raw_output.splitlines():
        s = line.strip()
        if not s or s.startswith("#") or s.startswith("```"):
            continue
        # Strip leading numbering like "1. " or "- "
        s = re.sub(r"^(?:\d+[.)]\s*|[-*]\s*)", "", s).strip()
        s = s.strip("`").strip()
        try:
            e = parse_expr(s, expected_width=64)
            if e not in seen:
                seen.add(e)
                results.append((s, e))
                if len(results) >= max_k:
                    break
        except ParseError:
            continue
    return results


def _build_initial_prompt(asm: str, active_regs: list[str], uses_cf: bool, width_hint: int, k: int) -> str:
    return (
        f"Simplify this obfuscated x86-64 register slice (computing RAX; active inputs={active_regs}, "
        f"uses_cf={uses_cf}, output_width_hint={width_hint}-bit) into its simplest mathematically equivalent expression.\n"
        f"Output ONLY {k} distinct candidate expressions (most likely first), one per line.\n\n"
        f"{GRAMMAR_SPEC}\n\n"
        f"Assembly slice:\n{asm}\n"
    )


def run_config_B(
    instance_id: str,
    family: str,
    asm: str,
    code: bytes,
    gt_cost: int,
    epsilon: float = 0.10,
    budget_sec: float = 45.0,
    max_smt_calls: int = 16,
    k_samples: int = 8,
    backend: str = "agy-hydrogen",
) -> RunResult:
    """Configuration B: LLM One-Shot (K samples, NO counterexample feedback) + LRAT oracle."""
    t0 = time.perf_counter()
    max_cost = gt_cost * (1.0 + epsilon)
    pf = SlicePreFilter(code)
    active_regs, uses_cf, width_hint = extract_active_inputs(code)
    prompt = _build_initial_prompt(asm, active_regs, uses_cf, width_hint, k_samples)

    raw = call_llm_cli(prompt, backend=backend, timeout_sec=max(5.0, budget_sec - 5.0))
    cands = parse_llm_candidates(raw, max_k=k_samples)
    smt_calls = 0
    tried = 0

    for _, expr in cands:
        if time.perf_counter() - t0 >= budget_sec or smt_calls >= max_smt_calls:
            break
        tried += 1
        if canonical_cost(expr) > max_cost + 1e-9:
            continue
        ok, _, _, _ = pf.check(expr)
        if not ok:
            continue
        smt_calls += 1
        rem = max(1.0, min(10.0, budget_sec - (time.perf_counter() - t0)))
        v = verify_candidate(code, expr, max_cost=max_cost, prefilter=pf, timeout_sec=rem, require_lrat=True)
        if v.status == "PASS":
            return RunResult(
                config="B", instance_id=instance_id, family=family, passed=True,
                status="PASS", elapsed_sec=time.perf_counter() - t0,
                smt_calls=smt_calls, llm_calls=1, candidates_tried=tried,
                best_expr=format_expr(expr), best_cost=v.cost,
                ground_truth_cost=gt_cost, lrat_sha256=v.lrat_sha256, detail=backend,
            )

    return RunResult(
        config="B", instance_id=instance_id, family=family, passed=False,
        status="FAIL_EXHAUSTED", elapsed_sec=time.perf_counter() - t0,
        smt_calls=smt_calls, llm_calls=1, candidates_tried=tried,
        ground_truth_cost=gt_cost, detail=f"{backend} parsed={len(cands)}",
    )


def run_config_D(
    instance_id: str,
    family: str,
    asm: str,
    code: bytes,
    gt_cost: int,
    epsilon: float = 0.10,
    budget_sec: float = 45.0,
    max_smt_calls: int = 16,
    max_rounds: int = 4,
    k_per_round: int = 2,
    backend: str = "agy-hydrogen",
) -> RunResult:
    """Configuration D (CRISOL-MBA-x64): Neuro-Symbolic CEGIS with concrete counterexample feedback + LRAT."""
    t0 = time.perf_counter()
    max_cost = gt_cost * (1.0 + epsilon)
    pf = SlicePreFilter(code)
    active_regs, uses_cf, width_hint = extract_active_inputs(code)
    smt_calls = 0
    llm_calls = 0
    tried = 0
    feedback_lines: list[str] = []

    for r in range(1, max_rounds + 1):
        rem_total = budget_sec - (time.perf_counter() - t0)
        if rem_total <= 3.0 or smt_calls >= max_smt_calls:
            break
        prompt = _build_initial_prompt(asm, active_regs, uses_cf, width_hint, k_per_round)
        if feedback_lines:
            prompt += (
                "\nPrevious candidates FAILED on these exact concrete counterexamples "
                "(fix the semantics so your new candidates match expected_rax):\n"
                + "\n".join(feedback_lines[-6:])
                + "\n"
            )
        llm_calls += 1
        raw = call_llm_cli(prompt, backend=backend, timeout_sec=max(4.0, min(20.0, rem_total - 2.0)))
        cands = parse_llm_candidates(raw, max_k=k_per_round)
        if not cands:
            feedback_lines.append(f"Round {r}: no parseable expressions returned. Follow the exact functional grammar.")
            continue

        for expr_str, expr in cands:
            if time.perf_counter() - t0 >= budget_sec or smt_calls >= max_smt_calls:
                break
            tried += 1
            cost = canonical_cost(expr)
            if cost > max_cost + 1e-9:
                feedback_lines.append(f"- `{expr_str}` REJECTED: too complex (cost={cost} > max={max_cost:.1f}). Simplify further.")
                continue
            ok, ce, o_out, c_out = pf.check(expr)
            if not ok and ce is not None:
                delta = (o_out or 0) ^ (c_out or 0)
                feedback_lines.append(
                    f"- `{expr_str}` FAILED concrete test: rdi={hex(ce.rdi)}, rsi={hex(ce.rsi)}, "
                    f"rdx={hex(ce.rdx)}, rcx={hex(ce.rcx)}, cf={ce.cf} => expected_rax={hex(o_out or 0)}, "
                    f"got={hex(c_out or 0)}, xor_diff={hex(delta)}"
                )
                continue
            smt_calls += 1
            rem = max(1.0, min(10.0, budget_sec - (time.perf_counter() - t0)))
            v = verify_candidate(code, expr, max_cost=max_cost, prefilter=pf, timeout_sec=rem, require_lrat=True)
            if v.status == "PASS":
                return RunResult(
                    config="D", instance_id=instance_id, family=family, passed=True,
                    status="PASS", elapsed_sec=time.perf_counter() - t0,
                    smt_calls=smt_calls, llm_calls=llm_calls, candidates_tried=tried,
                    best_expr=format_expr(expr), best_cost=v.cost,
                    ground_truth_cost=gt_cost, lrat_sha256=v.lrat_sha256,
                    detail=f"{backend} round={r}",
                )
            if v.counterexample is not None:
                ce = v.counterexample
                delta = (v.orig_out or 0) ^ (v.cand_out or 0)
                feedback_lines.append(
                    f"- `{expr_str}` FAILED SMT ({v.status}): rdi={hex(ce.rdi)}, rsi={hex(ce.rsi)}, "
                    f"rdx={hex(ce.rdx)}, rcx={hex(ce.rcx)}, cf={ce.cf} => expected_rax={hex(v.orig_out or 0)}, "
                    f"got={hex(v.cand_out or 0)}, xor_diff={hex(delta)}"
                )

    return RunResult(
        config="D", instance_id=instance_id, family=family, passed=False,
        status="FAIL_EXHAUSTED", elapsed_sec=time.perf_counter() - t0,
        smt_calls=smt_calls, llm_calls=llm_calls, candidates_tried=tried,
        ground_truth_cost=gt_cost, detail=f"{backend} rounds={llm_calls}",
    )


def run_config_H(
    instance_id: str,
    family: str,
    asm: str,
    code: bytes,
    gt_cost: int,
    epsilon: float = 0.10,
    budget_sec: float = 60.0,
    max_smt_calls: int = 16,
    round1_k: int = 6,
    cegis_rounds: int = 2,
    cegis_k: int = 2,
    backend: str = "agy-hydrogen",
    use_lean4: bool = False,
) -> RunResult:
    """Configuration H (CRISOL-MBA-x64 Hybrid): Wide One-Shot Round 1 (K=6) -> CEGIS fallback (2x2).

    Combines B's parallel sample diversity with D's counterexample-guided repair under a single budget.
    When use_lean4=True, certifies UNSAT with Lean 4 `bv_decide` (zero untrusted bit-blaster in TCB).
    """
    from .oracle_lean import certify_with_lean_bv_decide

    t0 = time.perf_counter()
    max_cost = gt_cost * (1.0 + epsilon)
    pf = SlicePreFilter(code)
    active_regs, uses_cf, width_hint = extract_active_inputs(code)
    smt_calls = 0
    llm_calls = 0
    tried = 0
    feedback_lines: list[str] = []

    schedule = [round1_k] + [cegis_k] * cegis_rounds
    for r, k in enumerate(schedule, 1):
        rem_total = budget_sec - (time.perf_counter() - t0)
        if rem_total <= 4.0 or smt_calls >= max_smt_calls:
            break
        prompt = _build_initial_prompt(asm, active_regs, uses_cf, width_hint, k)
        if feedback_lines:
            prompt += (
                "\nPrevious candidates FAILED on these exact concrete counterexamples "
                "(fix the semantics so your new candidates match expected_rax):\n"
                + "\n".join(feedback_lines[-6:])
                + "\n"
            )
        llm_calls += 1
        call_to = max(5.0, min(35.0 if r == 1 else 20.0, rem_total - 3.0))
        raw = call_llm_cli(prompt, backend=backend, timeout_sec=call_to)
        cands = parse_llm_candidates(raw, max_k=k)
        if not cands:
            feedback_lines.append(f"Round {r}: no parseable expressions returned. Follow the exact functional grammar.")
            continue

        for expr_str, expr in cands:
            if time.perf_counter() - t0 >= budget_sec or smt_calls >= max_smt_calls:
                break
            tried += 1
            cost = canonical_cost(expr)
            if cost > max_cost + 1e-9:
                feedback_lines.append(f"- `{expr_str}` REJECTED: too complex (cost={cost} > max={max_cost:.1f}). Simplify further.")
                continue
            ok, ce, o_out, c_out = pf.check(expr)
            if not ok and ce is not None:
                delta = (o_out or 0) ^ (c_out or 0)
                feedback_lines.append(
                    f"- `{expr_str}` FAILED concrete test: rdi={hex(ce.rdi)}, rsi={hex(ce.rsi)}, "
                    f"rdx={hex(ce.rdx)}, rcx={hex(ce.rcx)}, cf={ce.cf} => expected_rax={hex(o_out or 0)}, "
                    f"got={hex(c_out or 0)}, xor_diff={hex(delta)}"
                )
                continue

            smt_calls += 1
            rem = max(2.0, min(15.0, budget_sec - (time.perf_counter() - t0)))
            v = verify_candidate(
                code,
                expr,
                max_cost=max_cost,
                prefilter=pf,
                timeout_sec=rem,
                require_lrat=(not use_lean4),
            )
            if v.status == "PASS":
                cert_hash = v.lrat_sha256
                cert_tag = "lrat"
                if use_lean4:
                    lv = certify_with_lean_bv_decide(code, expr, timeout_sec=max(5.0, min(25.0, budget_sec - (time.perf_counter() - t0) + 5.0)))
                    if not lv.verified:
                        feedback_lines.append(f"- `{expr_str}` Lean 4 bv_decide failed: {lv.detail}")
                        continue
                    cert_hash = lv.theorem_sha256
                    cert_tag = "lean4_bv_decide"
                return RunResult(
                    config="H", instance_id=instance_id, family=family, passed=True,
                    status="PASS", elapsed_sec=time.perf_counter() - t0,
                    smt_calls=smt_calls, llm_calls=llm_calls, candidates_tried=tried,
                    best_expr=format_expr(expr), best_cost=v.cost,
                    ground_truth_cost=gt_cost, lrat_sha256=cert_hash,
                    detail=f"{backend} round={r} cert={cert_tag}",
                )
            if v.counterexample is not None:
                ce = v.counterexample
                delta = (v.orig_out or 0) ^ (v.cand_out or 0)
                feedback_lines.append(
                    f"- `{expr_str}` FAILED SMT ({v.status}): rdi={hex(ce.rdi)}, rsi={hex(ce.rsi)}, "
                    f"rdx={hex(ce.rdx)}, rcx={hex(ce.rcx)}, cf={ce.cf} => expected_rax={hex(v.orig_out or 0)}, "
                    f"got={hex(v.cand_out or 0)}, xor_diff={hex(delta)}"
                )

    return RunResult(
        config="H", instance_id=instance_id, family=family, passed=False,
        status="FAIL_EXHAUSTED", elapsed_sec=time.perf_counter() - t0,
        smt_calls=smt_calls, llm_calls=llm_calls, candidates_tried=tried,
        ground_truth_cost=gt_cost, detail=f"{backend} rounds={llm_calls}",
    )
