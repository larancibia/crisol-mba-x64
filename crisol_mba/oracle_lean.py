"""Lean 4 `bv_decide` certified bit-vector equivalence oracle (removes bit-blaster from TCB).

Translates a validated x86-64 byte slice and a candidate Expr directly into a Lean 4
BitVec theorem and discharges it with `by bv_decide` (Bitwuzla-style normalization +
verified bit-blasting + CaDiCaL + Lean's formally verified LRAT kernel checker).
"""

from __future__ import annotations

import hashlib
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from capstone import x86 as cs_x86

from .expr import Binary, Const, Expr, Ternary, Unary, Var, mask_for
from .lifter import CANONICAL_REGS, LiftError, REG_INFO, decode_and_validate


REPO_ROOT = Path(__file__).resolve().parent.parent
LEAN_BIN = REPO_ROOT / "build" / "elan" / "bin" / "lean"


@dataclass(frozen=True)
class LeanVerdict:
    verified: bool
    elapsed_sec: float
    theorem_sha256: str | None = None
    theorem_source: str | None = None
    detail: str = ""


def _bv_lit(val: int, width: int) -> str:
    return f"({val & mask_for(width)}#{(width)})"


def _lean_read_reg(state: dict[str, str], rname: str) -> tuple[str, int]:
    canon, width, offset = REG_INFO[rname]
    full = state[canon]
    if width == 64:
        return full, 64
    if offset == 0:
        return f"(BitVec.truncate {width} {full})", width
    # High 8-bit register (ah/bh/ch/dh): bits [15:8]
    return f"(BitVec.extractLsb' {offset} {width} {full})", width


def _lean_write_reg(state: dict[str, str], rname: str, val: str, let_lines: list[str], idx: int) -> None:
    canon, width, offset = REG_INFO[rname]
    vname = f"r_{canon}_{idx}"
    if width == 64:
        let_lines.append(f"  let {vname} : BitVec 64 := {val}")
    elif width == 32:
        let_lines.append(f"  let {vname} : BitVec 64 := BitVec.zeroExtend 64 ({val})")
    elif width == 16:
        hi = f"(BitVec.extractLsb' 16 48 {state[canon]})"
        let_lines.append(f"  let {vname} : BitVec 64 := BitVec.cast (by decide) ({hi} ++ ({val}))")
    elif width == 8 and offset == 0:
        hi = f"(BitVec.extractLsb' 8 56 {state[canon]})"
        let_lines.append(f"  let {vname} : BitVec 64 := BitVec.cast (by decide) ({hi} ++ ({val}))")
    elif width == 8 and offset == 8:
        hi = f"(BitVec.extractLsb' 16 48 {state[canon]})"
        lo = f"(BitVec.extractLsb' 0 8 {state[canon]})"
        let_lines.append(f"  let {vname} : BitVec 64 := BitVec.cast (by decide) (({hi} ++ ({val})) ++ {lo})")
    else:
        raise LiftError(f"Unsupported register write: {rname}")
    state[canon] = vname


def _lean_read_op(state: dict[str, str], insn, op, target_width: int) -> str:
    if op.type == cs_x86.X86_OP_REG:
        val, w = _lean_read_reg(state, insn.reg_name(op.reg).lower())
        if w != target_width:
            raise LiftError(f"Width mismatch {w} != {target_width}")
        return val
    if op.type == cs_x86.X86_OP_IMM:
        return _bv_lit(op.imm, target_width)
    raise LiftError(f"Unsupported operand type {op.type}")


def lift_x64_to_lean_lets(code: bytes) -> tuple[list[str], str, str]:
    """Lift validated x86-64 slice into Lean 4 let-bindings returning (let_lines, rax_var, cf_var)."""
    decoded = decode_and_validate(code)
    state: dict[str, str] = {}
    for r in CANONICAL_REGS:
        if r in {"rdi", "rsi", "rdx", "rcx"}:
            state[r] = r
        else:
            state[r] = "(0#64)"
    cf = "cf"
    let_lines: list[str] = []

    for step, d in enumerate(decoded):
        insn = d.raw_insn
        mnem = d.mnemonic
        ops = insn.operands
        dst_name = insn.reg_name(ops[0].reg).lower()
        dst_canon, dst_w, _ = REG_INFO[dst_name]

        if mnem == "mov":
            src = _lean_read_op(state, insn, ops[1], dst_w)
            _lean_write_reg(state, dst_name, src, let_lines, step)

        elif mnem == "lea":
            mem = ops[1].mem
            addr = _bv_lit(mem.disp, 64)
            if mem.base != 0:
                bv, bw = _lean_read_reg(state, insn.reg_name(mem.base).lower())
                if bw == 32:
                    bv = f"(BitVec.zeroExtend 64 {bv})"
                addr = f"({addr} + {bv})"
            if mem.index != 0:
                iv, iw = _lean_read_reg(state, insn.reg_name(mem.index).lower())
                if iw == 32:
                    iv = f"(BitVec.zeroExtend 64 {iv})"
                addr = f"({addr} + ({iv} * {_bv_lit(mem.scale, 64)}))"
            if getattr(insn, "addr_size", 8) == 4:
                addr = f"(BitVec.zeroExtend 64 (BitVec.truncate 32 {addr}))"
            if dst_w == 64:
                _lean_write_reg(state, dst_name, addr, let_lines, step)
            else:
                _lean_write_reg(state, dst_name, f"(BitVec.truncate {dst_w} {addr})", let_lines, step)

        elif mnem in {"add", "adc", "sub", "sbb"}:
            a, _ = _lean_read_reg(state, dst_name)
            b = _lean_read_op(state, insn, ops[1], dst_w)
            ext_w = dst_w + 1
            ae = f"(BitVec.zeroExtend {ext_w} {a})"
            be = f"(BitVec.zeroExtend {ext_w} {b})"
            ce = f"(BitVec.zeroExtend {ext_w} {cf})"
            ext_name = f"ext_{step}"
            if mnem == "add":
                let_lines.append(f"  let {ext_name} : BitVec {ext_w} := {ae} + {be}")
            elif mnem == "adc":
                let_lines.append(f"  let {ext_name} : BitVec {ext_w} := {ae} + {be} + {ce}")
            elif mnem == "sub":
                let_lines.append(f"  let {ext_name} : BitVec {ext_w} := {ae} - {be}")
            else:
                let_lines.append(f"  let {ext_name} : BitVec {ext_w} := {ae} - {be} - {ce}")
            _lean_write_reg(state, dst_name, f"(BitVec.truncate {dst_w} {ext_name})", let_lines, step)
            cf_name = f"cf_{step}"
            let_lines.append(f"  let {cf_name} : BitVec 1 := BitVec.extractLsb' {dst_w} 1 {ext_name}")
            cf = cf_name

        elif mnem == "neg":
            a, _ = _lean_read_reg(state, dst_name)
            _lean_write_reg(state, dst_name, f"(-{a})", let_lines, step)
            cf_name = f"cf_{step}"
            let_lines.append(f"  let {cf_name} : BitVec 1 := if {a} == {_bv_lit(0, dst_w)} then (0#1) else (1#1)")
            cf = cf_name

        elif mnem == "not":
            a, _ = _lean_read_reg(state, dst_name)
            _lean_write_reg(state, dst_name, f"(~~~{a})", let_lines, step)

        elif mnem in {"and", "or", "xor"}:
            a, _ = _lean_read_reg(state, dst_name)
            b = _lean_read_op(state, insn, ops[1], dst_w)
            op_sym = {"and": "&&&", "or": "|||", "xor": "^^^"}[mnem]
            _lean_write_reg(state, dst_name, f"({a} {op_sym} {b})", let_lines, step)
            cf = "(0#1)"

        elif mnem == "imul":
            if len(ops) == 2:
                a, _ = _lean_read_reg(state, dst_name)
                b = _lean_read_op(state, insn, ops[1], dst_w)
            elif len(ops) == 3:
                a = _lean_read_op(state, insn, ops[1], dst_w)
                b = _lean_read_op(state, insn, ops[2], dst_w)
            else:
                raise LiftError(f"Unsupported IMUL form: {d.op_str}")
            low_name = f"mul_{step}"
            let_lines.append(f"  let {low_name} : BitVec {dst_w} := {a} * {b}")
            _lean_write_reg(state, dst_name, low_name, let_lines, step)
            # CF for IMUL: signed overflow check
            prod_w = dst_w * 2
            cf_name = f"cf_{step}"
            let_lines.append(
                f"  let {cf_name} : BitVec 1 := "
                f"if BitVec.signExtend {prod_w} {low_name} == "
                f"(BitVec.signExtend {prod_w} {a} * BitVec.signExtend {prod_w} {b}) then (0#1) else (1#1)"
            )
            cf = cf_name

        elif mnem in {"shl", "sal", "shr", "sar"}:
            a, _ = _lean_read_reg(state, dst_name)
            count_mask = 63 if dst_w == 64 else 31
            c = ops[1].imm & count_mask
            if c == 0 or c >= dst_w:
                raise LiftError(f"Invalid shift count {c}")
            if mnem in {"shl", "sal"}:
                res = f"({a} <<< {c})"
                cf_expr = f"BitVec.extractLsb' {dst_w - c} 1 {a}"
            elif mnem == "shr":
                res = f"({a} >>> {c})"
                cf_expr = f"BitVec.extractLsb' {c - 1} 1 {a}"
            else:
                res = f"(BitVec.sshiftRight {a} {c})"
                cf_expr = f"BitVec.extractLsb' {c - 1} 1 {a}"
            _lean_write_reg(state, dst_name, res, let_lines, step)
            cf_name = f"cf_{step}"
            let_lines.append(f"  let {cf_name} : BitVec 1 := {cf_expr}")
            cf = cf_name

        elif mnem in {"rol", "ror"}:
            a, _ = _lean_read_reg(state, dst_name)
            count_mask = 63 if dst_w == 64 else 31
            c = ops[1].imm & count_mask
            if c == 0:
                raise LiftError("Rotate by 0 disallowed")
            r = c % dst_w
            res_name = f"rot_{step}"
            if r == 0:
                let_lines.append(f"  let {res_name} : BitVec {dst_w} := {a}")
            elif mnem == "rol":
                let_lines.append(f"  let {res_name} : BitVec {dst_w} := BitVec.rotateLeft {a} {r}")
            else:
                let_lines.append(f"  let {res_name} : BitVec {dst_w} := BitVec.rotateRight {a} {r}")
            _lean_write_reg(state, dst_name, res_name, let_lines, step)
            cf_name = f"cf_{step}"
            bit_idx = 0 if mnem == "rol" else (dst_w - 1)
            let_lines.append(f"  let {cf_name} : BitVec 1 := BitVec.extractLsb' {bit_idx} 1 {res_name}")
            cf = cf_name
        else:
            raise LiftError(f"Unhandled mnemonic in Lean lifter: {mnem}")

    return let_lines, state["rax"], cf


def expr_to_lean(e: Expr) -> str:
    """Translate canonical Expr AST into Lean 4 BitVec term."""
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Const):
        return _bv_lit(e.value, e.width)
    if isinstance(e, Unary):
        a = expr_to_lean(e.arg)
        if e.op == "neg":
            return f"(-{a})"
        if e.op == "not":
            return f"(~~~{a})"
        if e.op == "zext64":
            return f"(BitVec.zeroExtend 64 {a})"
        if e.op == "trunc32":
            return f"(BitVec.truncate 32 {a})"
        raise ValueError(e.op)
    if isinstance(e, Binary):
        l = expr_to_lean(e.left)
        r = expr_to_lean(e.right)
        if e.op == "add":
            return f"({l} + {r})"
        if e.op == "sub":
            return f"({l} - {r})"
        if e.op == "mul":
            return f"({l} * {r})"
        if e.op == "and":
            return f"({l} &&& {r})"
        if e.op == "or":
            return f"({l} ||| {r})"
        if e.op == "xor":
            return f"({l} ^^^ {r})"
        if e.op == "shl":
            if isinstance(e.right, Const):
                return f"({l} <<< {e.right.value & (e.width - 1)})"
            return f"({l} <<< ({r} &&& {_bv_lit(e.width - 1, e.width)}))"
        if e.op == "lshr":
            if isinstance(e.right, Const):
                return f"({l} >>> {e.right.value & (e.width - 1)})"
            return f"({l} >>> ({r} &&& {_bv_lit(e.width - 1, e.width)}))"
        if e.op == "ashr":
            if isinstance(e.right, Const):
                return f"(BitVec.sshiftRight {l} {e.right.value & (e.width - 1)})"
            return f"(BitVec.sshiftRight' {l} ({r} &&& {_bv_lit(e.width - 1, e.width)}))"
        if e.op == "rol":
            if isinstance(e.right, Const):
                return f"(BitVec.rotateLeft {l} {e.right.value % e.width})"
            raise ValueError("Non-constant rotate not supported in Lean bv_decide")
        if e.op == "ror":
            if isinstance(e.right, Const):
                return f"(BitVec.rotateRight {l} {e.right.value % e.width})"
            raise ValueError("Non-constant rotate not supported in Lean bv_decide")
        raise ValueError(e.op)
    if isinstance(e, Ternary):
        a = expr_to_lean(e.a)
        b = expr_to_lean(e.b)
        c = f"(BitVec.zeroExtend {e.width} {expr_to_lean(e.c)})"
        if e.op == "adc":
            return f"({a} + {b} + {c})"
        if e.op == "sbb":
            return f"({a} - {b} - {c})"
        raise ValueError(e.op)
    raise TypeError(type(e))


def certify_with_lean_bv_decide(
    code: bytes,
    cand: Expr,
    timeout_sec: float = 20.0,
) -> LeanVerdict:
    """Emit a Lean 4 theorem and prove equivalence with `by bv_decide`."""
    t0 = time.perf_counter()
    if not LEAN_BIN.exists():
        return LeanVerdict(False, time.perf_counter() - t0, detail=f"Missing Lean binary at {LEAN_BIN}")

    try:
        lets, rax_var, _ = lift_x64_to_lean_lets(code)
        cand_lean = expr_to_lean(cand)
    except Exception as exc:
        return LeanVerdict(False, time.perf_counter() - t0, detail=f"Lean translation error: {exc}")

    lets_block = "\n".join(lets)
    src = (
        "import Std.Tactic.BVDecide\n\n"
        "def crisol_lifted (rdi rsi rdx rcx : BitVec 64) (cf : BitVec 1) : BitVec 64 :=\n"
        f"{lets_block}\n"
        f"  {rax_var}\n\n"
        "theorem crisol_equiv (rdi rsi rdx rcx : BitVec 64) (cf : BitVec 1) :\n"
        f"    crisol_lifted rdi rsi rdx rcx cf = {cand_lean} := by\n"
        "  unfold crisol_lifted\n"
        "  bv_decide\n"
    )
    th_hash = hashlib.sha256(src.encode("utf-8")).hexdigest()

    import os
    env = dict(os.environ)
    env["ELAN_HOME"] = str(REPO_ROOT / "build" / "elan")

    with tempfile.TemporaryDirectory() as td:
        lean_file = Path(td) / "CrislCheck.lean"
        lean_file.write_text(src, encoding="utf-8")
        try:
            proc = subprocess.run(
                [str(LEAN_BIN), str(lean_file)],
                capture_output=True,
                text=True,
                timeout=timeout_sec,
                env=env,
            )
        except subprocess.TimeoutExpired:
            return LeanVerdict(False, time.perf_counter() - t0, th_hash, src, detail="Lean bv_decide timeout")

        combined = (proc.stdout + "\n" + proc.stderr).strip()
        if proc.returncode == 0 and "sorry" not in combined.lower() and "error" not in combined.lower():
            return LeanVerdict(
                verified=True,
                elapsed_sec=time.perf_counter() - t0,
                theorem_sha256=th_hash,
                theorem_source=src,
                detail="LEAN4_BV_DECIDE_VERIFIED",
            )
        return LeanVerdict(
            verified=False,
            elapsed_sec=time.perf_counter() - t0,
            theorem_sha256=th_hash,
            theorem_source=src,
            detail=f"Lean rc={proc.returncode}: {combined[:300]}",
        )
