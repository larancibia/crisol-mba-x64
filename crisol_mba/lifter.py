"""Minimalist x86-64 register-only lifter to QF_BV (Z3 and Bitwuzla).

Strictly enforces the pre-registered closed instruction allowlist:
  MOV, LEA (register-only), ADD, ADC, SUB, SBB, NEG, IMUL (non-nested 64-bit),
  AND, OR, XOR, NOT, SHL/SAL, SHR, SAR, ROL, ROR.

Explicitly models:
  - Carry Flag (CF) across ADD, ADC, SUB, SBB, NEG, IMUL, AND/OR/XOR, shifts/rotates.
  - Zero-extension of 32-bit register writes into 64-bit registers.
  - Upper-bit preservation for 16-bit and 8-bit partial register writes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from capstone import Cs, CS_ARCH_X86, CS_MODE_64
from capstone import x86 as cs_x86
import z3


ALLOWED_MNEMONICS = frozenset({
    "mov", "lea",
    "add", "adc", "sub", "sbb", "neg", "imul",
    "and", "or", "xor", "not",
    "shl", "sal", "shr", "sar", "rol", "ror",
})

MAX_SLICE_INSNS = 24

# Map x86-64 register names -> (canonical_64bit_reg, bit_width, bit_offset)
REG_INFO: dict[str, tuple[str, int, int]] = {}
_REG_FAMILIES = [
    ("rax", "eax", "ax", "al", "ah"),
    ("rbx", "ebx", "bx", "bl", "bh"),
    ("rcx", "ecx", "cx", "cl", "ch"),
    ("rdx", "edx", "dx", "dl", "dh"),
    ("rsi", "esi", "si", "sil", None),
    ("rdi", "edi", "di", "dil", None),
    ("rbp", "ebp", "bp", "bpl", None),
    ("rsp", "esp", "sp", "spl", None),
    ("r8", "r8d", "r8w", "r8b", None),
    ("r9", "r9d", "r9w", "r9b", None),
    ("r10", "r10d", "r10w", "r10b", None),
    ("r11", "r11d", "r11w", "r11b", None),
    ("r12", "r12d", "r12w", "r12b", None),
    ("r13", "r13d", "r13w", "r13b", None),
    ("r14", "r14d", "r14w", "r14b", None),
    ("r15", "r15d", "r15w", "r15b", None),
]
for r64, r32, r16, r8l, r8h in _REG_FAMILIES:
    REG_INFO[r64] = (r64, 64, 0)
    REG_INFO[r32] = (r64, 32, 0)
    REG_INFO[r16] = (r64, 16, 0)
    REG_INFO[r8l] = (r64, 8, 0)
    if r8h is not None:
        REG_INFO[r8h] = (r64, 8, 8)

CANONICAL_REGS = tuple(f[0] for f in _REG_FAMILIES)


class LiftError(ValueError):
    """Raised when a slice violates the pre-registered admission contract."""


@dataclass(frozen=True)
class DecodedInsn:
    address: int
    size: int
    mnemonic: str
    op_str: str
    raw_insn: Any


def decode_and_validate(code: bytes) -> list[DecodedInsn]:
    """Decode x86-64 bytes with Capstone and enforce the closed allowlist."""
    if not code:
        raise LiftError("Empty byte slice")
    md = Cs(CS_ARCH_X86, CS_MODE_64)
    md.detail = True
    insns = list(md.disasm(code, 0))
    total_bytes = sum(i.size for i in insns)
    if total_bytes != len(code):
        raise LiftError(f"Undecoded trailing bytes: decoded {total_bytes} of {len(code)}")
    if len(insns) > MAX_SLICE_INSNS:
        raise LiftError(f"Slice has {len(insns)} instructions > max {MAX_SLICE_INSNS}")

    decoded: list[DecodedInsn] = []
    for insn in insns:
        mnem = insn.mnemonic.lower()
        if mnem not in ALLOWED_MNEMONICS:
            raise LiftError(f"Disallowed mnemonic: {mnem} ({insn.op_str})")
        # Validate operands
        for idx, op in enumerate(insn.operands):
            if op.type == cs_x86.X86_OP_MEM:
                if mnem != "lea" or idx != 1:
                    raise LiftError(f"Memory operand disallowed in {mnem} {insn.op_str}")
                if op.mem.segment != 0:
                    raise LiftError(f"Segment override disallowed in LEA: {insn.op_str}")
                if op.mem.base == cs_x86.X86_REG_RIP:
                    raise LiftError(f"RIP-relative LEA disallowed: {insn.op_str}")
                for mreg in (op.mem.base, op.mem.index):
                    if mreg != 0:
                        mrname = insn.reg_name(mreg).lower()
                        if mrname not in REG_INFO or REG_INFO[mrname][0] == "rsp":
                            raise LiftError(f"Disallowed register {mrname} in LEA: {insn.op_str}")
            elif op.type == cs_x86.X86_OP_REG:
                rname = insn.reg_name(op.reg).lower()
                if rname not in REG_INFO or REG_INFO[rname][0] == "rsp":
                    raise LiftError(f"Unsupported/disallowed register {rname} in {mnem} {insn.op_str}")
            elif op.type == cs_x86.X86_OP_IMM:
                if mnem in {"shl", "sal", "shr", "sar", "rol", "ror"} and idx == 1:
                    pass
            else:
                raise LiftError(f"Unsupported operand type {op.type} in {mnem} {insn.op_str}")
        # Shifts/rotates must be by immediate (pre-registration contract)
        if mnem in {"shl", "sal", "shr", "sar", "rol", "ror"}:
            if len(insn.operands) != 2 or insn.operands[1].type != cs_x86.X86_OP_IMM:
                raise LiftError(f"Shift/rotate must use immediate count: {mnem} {insn.op_str}")
        decoded.append(DecodedInsn(insn.address, insn.size, mnem, insn.op_str, insn))
    return decoded


@dataclass
class LiftedZ3:
    regs: dict[str, z3.BitVecRef]
    cf: z3.BitVecRef  # 1-bit BitVec
    input_vars: dict[str, z3.BitVecRef]
    insns: list[DecodedInsn]


def _z3_read_reg(state: dict[str, z3.BitVecRef], rname: str) -> tuple[z3.BitVecRef, int]:
    canon, width, offset = REG_INFO[rname]
    full = state[canon]
    if width == 64:
        return full, 64
    return z3.Extract(offset + width - 1, offset, full), width


def _z3_write_reg(state: dict[str, z3.BitVecRef], rname: str, val: z3.BitVecRef) -> None:
    canon, width, offset = REG_INFO[rname]
    if width == 64:
        state[canon] = val
    elif width == 32:
        # x86-64 architectural rule: 32-bit write zero-extends to 64 bits
        state[canon] = z3.ZeroExt(32, val)
    elif width == 16:
        # Preserves upper 48 bits
        state[canon] = z3.Concat(z3.Extract(63, 16, state[canon]), val)
    elif width == 8 and offset == 0:
        # Low 8-bit register preserves upper 56 bits
        state[canon] = z3.Concat(z3.Extract(63, 8, state[canon]), val)
    elif width == 8 and offset == 8:
        # High 8-bit register (ah/bh/ch/dh) preserves [63:16] and [7:0]
        state[canon] = z3.Concat(
            z3.Extract(63, 16, state[canon]),
            val,
            z3.Extract(7, 0, state[canon]),
        )
    else:
        raise LiftError(f"Unexpected register slice: {rname}")


def _z3_read_op(state: dict[str, z3.BitVecRef], insn: Any, op: Any, target_width: int) -> z3.BitVecRef:
    if op.type == cs_x86.X86_OP_REG:
        val, w = _z3_read_reg(state, insn.reg_name(op.reg).lower())
        if w != target_width:
            raise LiftError(f"Register width mismatch {w} != {target_width} in {insn.mnemonic} {insn.op_str}")
        return val
    if op.type == cs_x86.X86_OP_IMM:
        return z3.BitVecVal(op.imm & ((1 << target_width) - 1), target_width)
    raise LiftError(f"Cannot read operand type {op.type}")


def lift_to_z3(code: bytes, initial_vars: dict[str, z3.BitVecRef] | None = None) -> LiftedZ3:
    """Lift a validated x86-64 slice to Z3 QF_BV formulas."""
    decoded = decode_and_validate(code)
    if initial_vars is None:
        input_vars: dict[str, z3.BitVecRef] = {r: z3.BitVec(r, 64) for r in CANONICAL_REGS}
        input_vars["cf"] = z3.BitVec("cf", 1)
    else:
        input_vars = dict(initial_vars)
        for r in CANONICAL_REGS:
            if r not in input_vars:
                input_vars[r] = z3.BitVec(r, 64)
        if "cf" not in input_vars:
            input_vars["cf"] = z3.BitVec("cf", 1)

    state: dict[str, z3.BitVecRef] = {r: input_vars[r] for r in CANONICAL_REGS}
    cf: z3.BitVecRef = input_vars["cf"]
    # Track 64-bit IMUL taint to reject nested 64-bit multiplications per pre-registration
    mul64_tainted: set[str] = set()

    for d in decoded:
        insn = d.raw_insn
        mnem = d.mnemonic
        ops = insn.operands
        if len(ops) < 1 or ops[0].type != cs_x86.X86_OP_REG:
            raise LiftError(f"Destination must be register: {mnem} {d.op_str}")
        dst_name = insn.reg_name(ops[0].reg).lower()
        dst_canon, dst_w, _ = REG_INFO[dst_name]

        if mnem == "mov":
            src = _z3_read_op(state, insn, ops[1], dst_w)
            _z3_write_reg(state, dst_name, src)
            if ops[1].type == cs_x86.X86_OP_REG:
                src_canon = REG_INFO[insn.reg_name(ops[1].reg).lower()][0]
                if src_canon in mul64_tainted:
                    mul64_tainted.add(dst_canon)
                else:
                    mul64_tainted.discard(dst_canon)
            else:
                mul64_tainted.discard(dst_canon)

        elif mnem == "lea":
            mem = ops[1].mem
            addr = z3.BitVecVal(mem.disp & ((1 << 64) - 1), 64)
            if mem.base != 0:
                base_val, base_w = _z3_read_reg(state, insn.reg_name(mem.base).lower())
                if base_w == 32:
                    base_val = z3.ZeroExt(32, base_val)
                addr = addr + base_val
            if mem.index != 0:
                idx_val, idx_w = _z3_read_reg(state, insn.reg_name(mem.index).lower())
                if idx_w == 32:
                    idx_val = z3.ZeroExt(32, idx_val)
                addr = addr + idx_val * z3.BitVecVal(mem.scale, 64)
            if getattr(insn, "addr_size", 8) == 4:
                addr = z3.ZeroExt(32, z3.Extract(31, 0, addr))
            if dst_w == 64:
                _z3_write_reg(state, dst_name, addr)
            elif dst_w == 32:
                _z3_write_reg(state, dst_name, z3.Extract(31, 0, addr))
            elif dst_w == 16:
                _z3_write_reg(state, dst_name, z3.Extract(15, 0, addr))
            else:
                raise LiftError(f"Unsupported LEA width {dst_w}")

        elif mnem == "add":
            a, _ = _z3_read_reg(state, dst_name)
            b = _z3_read_op(state, insn, ops[1], dst_w)
            ext = z3.ZeroExt(1, a) + z3.ZeroExt(1, b)
            _z3_write_reg(state, dst_name, z3.Extract(dst_w - 1, 0, ext))
            cf = z3.Extract(dst_w, dst_w, ext)

        elif mnem == "adc":
            a, _ = _z3_read_reg(state, dst_name)
            b = _z3_read_op(state, insn, ops[1], dst_w)
            ext = z3.ZeroExt(1, a) + z3.ZeroExt(1, b) + z3.ZeroExt(dst_w, cf)
            _z3_write_reg(state, dst_name, z3.Extract(dst_w - 1, 0, ext))
            cf = z3.Extract(dst_w, dst_w, ext)

        elif mnem == "sub":
            a, _ = _z3_read_reg(state, dst_name)
            b = _z3_read_op(state, insn, ops[1], dst_w)
            ext = z3.ZeroExt(1, a) - z3.ZeroExt(1, b)
            _z3_write_reg(state, dst_name, z3.Extract(dst_w - 1, 0, ext))
            cf = z3.Extract(dst_w, dst_w, ext)

        elif mnem == "sbb":
            a, _ = _z3_read_reg(state, dst_name)
            b = _z3_read_op(state, insn, ops[1], dst_w)
            ext = z3.ZeroExt(1, a) - z3.ZeroExt(1, b) - z3.ZeroExt(dst_w, cf)
            _z3_write_reg(state, dst_name, z3.Extract(dst_w - 1, 0, ext))
            cf = z3.Extract(dst_w, dst_w, ext)

        elif mnem == "neg":
            a, _ = _z3_read_reg(state, dst_name)
            _z3_write_reg(state, dst_name, -a)
            cf = z3.If(a == z3.BitVecVal(0, dst_w), z3.BitVecVal(0, 1), z3.BitVecVal(1, 1))

        elif mnem == "not":
            a, _ = _z3_read_reg(state, dst_name)
            _z3_write_reg(state, dst_name, ~a)
            # NOT does not affect CF

        elif mnem in {"and", "or", "xor"}:
            a, _ = _z3_read_reg(state, dst_name)
            b = _z3_read_op(state, insn, ops[1], dst_w)
            res = (a & b) if mnem == "and" else ((a | b) if mnem == "or" else (a ^ b))
            _z3_write_reg(state, dst_name, res)
            cf = z3.BitVecVal(0, 1)

        elif mnem == "imul":
            if len(ops) == 2:
                a, _ = _z3_read_reg(state, dst_name)
                b = _z3_read_op(state, insn, ops[1], dst_w)
                src_regs = [dst_canon]
                if ops[1].type == cs_x86.X86_OP_REG:
                    src_regs.append(REG_INFO[insn.reg_name(ops[1].reg).lower()][0])
            elif len(ops) == 3:
                a = _z3_read_op(state, insn, ops[1], dst_w)
                b = _z3_read_op(state, insn, ops[2], dst_w)
                src_regs = []
                if ops[1].type == cs_x86.X86_OP_REG:
                    src_regs.append(REG_INFO[insn.reg_name(ops[1].reg).lower()][0])
            else:
                raise LiftError(f"Single-operand IMUL disallowed in MVP: {d.op_str}")
            if dst_w == 64:
                if any(r in mul64_tainted for r in src_regs):
                    raise LiftError(f"Nested 64-bit IMUL disallowed in MVP: {mnem} {d.op_str}")
                mul64_tainted.add(dst_canon)
            prod_full = z3.SignExt(dst_w, a) * z3.SignExt(dst_w, b)
            low = z3.Extract(dst_w - 1, 0, prod_full)
            _z3_write_reg(state, dst_name, low)
            cf = z3.If(z3.SignExt(dst_w, low) == prod_full, z3.BitVecVal(0, 1), z3.BitVecVal(1, 1))

        elif mnem in {"shl", "sal", "shr", "sar"}:
            a, _ = _z3_read_reg(state, dst_name)
            count_mask = 63 if dst_w == 64 else 31
            c = ops[1].imm & count_mask
            if c >= dst_w:
                raise LiftError(f"Shift count {c} >= width {dst_w} has undefined CF in x86-64: {mnem} {d.op_str}")
            if c > 0:
                if mnem in {"shl", "sal"}:
                    cf = z3.Extract(dst_w - c, dst_w - c, a)
                    res = a << z3.BitVecVal(c, dst_w)
                elif mnem == "shr":
                    cf = z3.Extract(c - 1, c - 1, a)
                    res = z3.LShR(a, z3.BitVecVal(c, dst_w))
                else:  # sar
                    cf = z3.Extract(c - 1, c - 1, a)
                    res = a >> z3.BitVecVal(c, dst_w)
                _z3_write_reg(state, dst_name, res)
            else:
                # Count 0: result unchanged, CF unchanged; still enforce 32-bit zero-ext on register write?
                # Wait: x86-64 SHL eax, 0 does NOT clear upper 32 bits? Actually on x86-64,
                # shift by 0 writes nothing to the register or writes 32-bit zero-ext depending on microarch;
                # reject shift-by-0 as degenerate/non-canonical.
                raise LiftError(f"Degenerate shift by 0 disallowed: {mnem} {d.op_str}")

        elif mnem in {"rol", "ror"}:
            a, _ = _z3_read_reg(state, dst_name)
            count_mask = 63 if dst_w == 64 else 31
            c = ops[1].imm & count_mask
            if c == 0:
                raise LiftError(f"Degenerate rotate by 0 disallowed: {mnem} {d.op_str}")
            r = c % dst_w
            if r == 0:
                res = a
            elif mnem == "rol":
                res = z3.RotateLeft(a, r)
            else:
                res = z3.RotateRight(a, r)
            _z3_write_reg(state, dst_name, res)
            if mnem == "rol":
                cf = z3.Extract(0, 0, res)
            else:
                cf = z3.Extract(dst_w - 1, dst_w - 1, res)
        else:
            raise LiftError(f"Unhandled mnemonic: {mnem}")

    return LiftedZ3(regs=state, cf=cf, input_vars=input_vars, insns=decoded)


@dataclass
class LiftedBitwuzla:
    regs: dict[str, Any]
    cf: Any
    input_vars: dict[str, Any]
    insns: list[DecodedInsn]


def _bz_read_reg(tm: Any, state: dict[str, Any], rname: str) -> tuple[Any, int]:
    import bitwuzla as bz
    canon, width, offset = REG_INFO[rname]
    full = state[canon]
    if width == 64:
        return full, 64
    return tm.mk_term(bz.Kind.BV_EXTRACT, [full], [offset + width - 1, offset]), width


def _bz_write_reg(tm: Any, state: dict[str, Any], rname: str, val: Any) -> None:
    import bitwuzla as bz
    canon, width, offset = REG_INFO[rname]
    if width == 64:
        state[canon] = val
    elif width == 32:
        state[canon] = tm.mk_term(bz.Kind.BV_ZERO_EXTEND, [val], [32])
    elif width == 16:
        hi = tm.mk_term(bz.Kind.BV_EXTRACT, [state[canon]], [63, 16])
        state[canon] = tm.mk_term(bz.Kind.BV_CONCAT, [hi, val])
    elif width == 8 and offset == 0:
        hi = tm.mk_term(bz.Kind.BV_EXTRACT, [state[canon]], [63, 8])
        state[canon] = tm.mk_term(bz.Kind.BV_CONCAT, [hi, val])
    elif width == 8 and offset == 8:
        hi = tm.mk_term(bz.Kind.BV_EXTRACT, [state[canon]], [63, 16])
        lo = tm.mk_term(bz.Kind.BV_EXTRACT, [state[canon]], [7, 0])
        state[canon] = tm.mk_term(bz.Kind.BV_CONCAT, [tm.mk_term(bz.Kind.BV_CONCAT, [hi, val]), lo])
    else:
        raise LiftError(f"Unexpected register slice: {rname}")


def _bz_read_op(tm: Any, state: dict[str, Any], insn: Any, op: Any, target_width: int) -> Any:
    if op.type == cs_x86.X86_OP_REG:
        val, w = _bz_read_reg(tm, state, insn.reg_name(op.reg).lower())
        if w != target_width:
            raise LiftError(f"Register width mismatch {w} != {target_width}")
        return val
    if op.type == cs_x86.X86_OP_IMM:
        return tm.mk_bv_value(tm.mk_bv_sort(target_width), op.imm & ((1 << target_width) - 1))
    raise LiftError(f"Cannot read operand type {op.type}")


def lift_to_bitwuzla(code: bytes, tm: Any, initial_vars: dict[str, Any] | None = None) -> LiftedBitwuzla:
    """Lift a validated x86-64 slice directly to Bitwuzla QF_BV terms."""
    import bitwuzla as bz

    decoded = decode_and_validate(code)
    s64 = tm.mk_bv_sort(64)
    s1 = tm.mk_bv_sort(1)
    if initial_vars is None:
        input_vars = {r: tm.mk_const(s64, r) for r in CANONICAL_REGS}
        input_vars["cf"] = tm.mk_const(s1, "cf")
    else:
        input_vars = dict(initial_vars)
        for r in CANONICAL_REGS:
            if r not in input_vars:
                input_vars[r] = tm.mk_const(s64, r)
        if "cf" not in input_vars:
            input_vars["cf"] = tm.mk_const(s1, "cf")

    state = {r: input_vars[r] for r in CANONICAL_REGS}
    cf = input_vars["cf"]
    mul64_tainted: set[str] = set()

    for d in decoded:
        insn = d.raw_insn
        mnem = d.mnemonic
        ops = insn.operands
        dst_name = insn.reg_name(ops[0].reg).lower()
        dst_canon, dst_w, _ = REG_INFO[dst_name]

        if mnem == "mov":
            src = _bz_read_op(tm, state, insn, ops[1], dst_w)
            _bz_write_reg(tm, state, dst_name, src)
            if ops[1].type == cs_x86.X86_OP_REG:
                src_canon = REG_INFO[insn.reg_name(ops[1].reg).lower()][0]
                if src_canon in mul64_tainted:
                    mul64_tainted.add(dst_canon)
                else:
                    mul64_tainted.discard(dst_canon)
            else:
                mul64_tainted.discard(dst_canon)

        elif mnem == "lea":
            mem = ops[1].mem
            addr = tm.mk_bv_value(s64, mem.disp & ((1 << 64) - 1))
            if mem.base != 0:
                base_val, base_w = _bz_read_reg(tm, state, insn.reg_name(mem.base).lower())
                if base_w == 32:
                    base_val = tm.mk_term(bz.Kind.BV_ZERO_EXTEND, [base_val], [32])
                addr = tm.mk_term(bz.Kind.BV_ADD, [addr, base_val])
            if mem.index != 0:
                idx_val, idx_w = _bz_read_reg(tm, state, insn.reg_name(mem.index).lower())
                if idx_w == 32:
                    idx_val = tm.mk_term(bz.Kind.BV_ZERO_EXTEND, [idx_val], [32])
                scaled = tm.mk_term(bz.Kind.BV_MUL, [idx_val, tm.mk_bv_value(s64, mem.scale)])
                addr = tm.mk_term(bz.Kind.BV_ADD, [addr, scaled])
            if getattr(insn, "addr_size", 8) == 4:
                addr32 = tm.mk_term(bz.Kind.BV_EXTRACT, [addr], [31, 0])
                addr = tm.mk_term(bz.Kind.BV_ZERO_EXTEND, [addr32], [32])
            if dst_w == 64:
                _bz_write_reg(tm, state, dst_name, addr)
            elif dst_w == 32:
                _bz_write_reg(tm, state, dst_name, tm.mk_term(bz.Kind.BV_EXTRACT, [addr], [31, 0]))
            elif dst_w == 16:
                _bz_write_reg(tm, state, dst_name, tm.mk_term(bz.Kind.BV_EXTRACT, [addr], [15, 0]))
            else:
                raise LiftError(f"Unsupported LEA width {dst_w}")

        elif mnem in {"add", "adc", "sub", "sbb"}:
            a, _ = _bz_read_reg(tm, state, dst_name)
            b = _bz_read_op(tm, state, insn, ops[1], dst_w)
            ae = tm.mk_term(bz.Kind.BV_ZERO_EXTEND, [a], [1])
            be = tm.mk_term(bz.Kind.BV_ZERO_EXTEND, [b], [1])
            if mnem == "add":
                ext = tm.mk_term(bz.Kind.BV_ADD, [ae, be])
            elif mnem == "adc":
                ce = tm.mk_term(bz.Kind.BV_ZERO_EXTEND, [cf], [dst_w])
                ext = tm.mk_term(bz.Kind.BV_ADD, [tm.mk_term(bz.Kind.BV_ADD, [ae, be]), ce])
            elif mnem == "sub":
                ext = tm.mk_term(bz.Kind.BV_SUB, [ae, be])
            else:
                ce = tm.mk_term(bz.Kind.BV_ZERO_EXTEND, [cf], [dst_w])
                ext = tm.mk_term(bz.Kind.BV_SUB, [tm.mk_term(bz.Kind.BV_SUB, [ae, be]), ce])
            _bz_write_reg(tm, state, dst_name, tm.mk_term(bz.Kind.BV_EXTRACT, [ext], [dst_w - 1, 0]))
            cf = tm.mk_term(bz.Kind.BV_EXTRACT, [ext], [dst_w, dst_w])

        elif mnem == "neg":
            a, _ = _bz_read_reg(tm, state, dst_name)
            _bz_write_reg(tm, state, dst_name, tm.mk_term(bz.Kind.BV_NEG, [a]))
            is_zero = tm.mk_term(bz.Kind.EQUAL, [a, tm.mk_bv_zero(tm.mk_bv_sort(dst_w))])
            cf = tm.mk_term(bz.Kind.ITE, [is_zero, tm.mk_bv_zero(s1), tm.mk_bv_one(s1)])

        elif mnem == "not":
            a, _ = _bz_read_reg(tm, state, dst_name)
            _bz_write_reg(tm, state, dst_name, tm.mk_term(bz.Kind.BV_NOT, [a]))

        elif mnem in {"and", "or", "xor"}:
            a, _ = _bz_read_reg(tm, state, dst_name)
            b = _bz_read_op(tm, state, insn, ops[1], dst_w)
            kind = {"and": bz.Kind.BV_AND, "or": bz.Kind.BV_OR, "xor": bz.Kind.BV_XOR}[mnem]
            _bz_write_reg(tm, state, dst_name, tm.mk_term(kind, [a, b]))
            cf = tm.mk_bv_zero(s1)

        elif mnem == "imul":
            if len(ops) == 2:
                a, _ = _bz_read_reg(tm, state, dst_name)
                b = _bz_read_op(tm, state, insn, ops[1], dst_w)
                src_regs = [dst_canon]
                if ops[1].type == cs_x86.X86_OP_REG:
                    src_regs.append(REG_INFO[insn.reg_name(ops[1].reg).lower()][0])
            elif len(ops) == 3:
                a = _bz_read_op(tm, state, insn, ops[1], dst_w)
                b = _bz_read_op(tm, state, insn, ops[2], dst_w)
                src_regs = []
                if ops[1].type == cs_x86.X86_OP_REG:
                    src_regs.append(REG_INFO[insn.reg_name(ops[1].reg).lower()][0])
            else:
                raise LiftError(f"Single-operand IMUL disallowed in MVP: {d.op_str}")
            if dst_w == 64:
                if any(r in mul64_tainted for r in src_regs):
                    raise LiftError(f"Nested 64-bit IMUL disallowed in MVP: {mnem} {d.op_str}")
                mul64_tainted.add(dst_canon)
            prod = tm.mk_term(
                bz.Kind.BV_MUL,
                [
                    tm.mk_term(bz.Kind.BV_SIGN_EXTEND, [a], [dst_w]),
                    tm.mk_term(bz.Kind.BV_SIGN_EXTEND, [b], [dst_w]),
                ],
            )
            low = tm.mk_term(bz.Kind.BV_EXTRACT, [prod], [dst_w - 1, 0])
            _bz_write_reg(tm, state, dst_name, low)
            fits = tm.mk_term(bz.Kind.EQUAL, [tm.mk_term(bz.Kind.BV_SIGN_EXTEND, [low], [dst_w]), prod])
            cf = tm.mk_term(bz.Kind.ITE, [fits, tm.mk_bv_zero(s1), tm.mk_bv_one(s1)])

        elif mnem in {"shl", "sal", "shr", "sar"}:
            a, _ = _bz_read_reg(tm, state, dst_name)
            count_mask = 63 if dst_w == 64 else 31
            c = ops[1].imm & count_mask
            if c == 0 or c >= dst_w:
                raise LiftError(f"Invalid shift count {c} for width {dst_w}")
            c_term = tm.mk_bv_value(tm.mk_bv_sort(dst_w), c)
            if mnem in {"shl", "sal"}:
                cf = tm.mk_term(bz.Kind.BV_EXTRACT, [a], [dst_w - c, dst_w - c])
                res = tm.mk_term(bz.Kind.BV_SHL, [a, c_term])
            elif mnem == "shr":
                cf = tm.mk_term(bz.Kind.BV_EXTRACT, [a], [c - 1, c - 1])
                res = tm.mk_term(bz.Kind.BV_SHR, [a, c_term])
            else:
                cf = tm.mk_term(bz.Kind.BV_EXTRACT, [a], [c - 1, c - 1])
                res = tm.mk_term(bz.Kind.BV_ASHR, [a, c_term])
            _bz_write_reg(tm, state, dst_name, res)

        elif mnem in {"rol", "ror"}:
            a, _ = _bz_read_reg(tm, state, dst_name)
            count_mask = 63 if dst_w == 64 else 31
            c = ops[1].imm & count_mask
            if c == 0:
                raise LiftError("Degenerate rotate by 0 disallowed")
            r = c % dst_w
            if r == 0:
                res = a
            elif mnem == "rol":
                res = tm.mk_term(bz.Kind.BV_ROLI, [a], [r])
            else:
                res = tm.mk_term(bz.Kind.BV_RORI, [a], [r])
            _bz_write_reg(tm, state, dst_name, res)
            if mnem == "rol":
                cf = tm.mk_term(bz.Kind.BV_EXTRACT, [res], [0, 0])
            else:
                cf = tm.mk_term(bz.Kind.BV_EXTRACT, [res], [dst_w - 1, dst_w - 1])

    return LiftedBitwuzla(regs=state, cf=cf, input_vars=input_vars, insns=decoded)
