"""Canonical bit-vector expression AST, CSE cost counter, and SMT builders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np
import z3


def mask_for(width: int) -> int:
    return (1 << width) - 1


def to_signed(val: int, width: int) -> int:
    val &= mask_for(width)
    sign_bit = 1 << (width - 1)
    return val - (1 << width) if (val & sign_bit) else val


@dataclass(frozen=True)
class Expr:
    width: int


@dataclass(frozen=True)
class Var(Expr):
    name: str


@dataclass(frozen=True)
class Const(Expr):
    value: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", self.value & mask_for(self.width))


@dataclass(frozen=True)
class Unary(Expr):
    op: str  # "neg", "not", "zext64", "trunc32"
    arg: Expr


@dataclass(frozen=True)
class Binary(Expr):
    op: str  # "add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr", "rol", "ror"
    left: Expr
    right: Expr


@dataclass(frozen=True)
class Ternary(Expr):
    op: str  # "adc", "sbb"
    a: Expr
    b: Expr
    c: Expr  # 1-bit carry


COMMUTATIVE_OPS = frozenset({"add", "mul", "and", "or", "xor"})


def _eval_op_int(op: str, vals: tuple[int, ...], width: int) -> int:
    m = mask_for(width)
    if op == "neg":
        return (-vals[0]) & m
    if op == "not":
        return (~vals[0]) & m
    if op == "zext64":
        return vals[0] & mask_for(64)
    if op == "trunc32":
        return vals[0] & mask_for(32)
    if op == "add":
        return (vals[0] + vals[1]) & m
    if op == "sub":
        return (vals[0] - vals[1]) & m
    if op == "mul":
        return (vals[0] * vals[1]) & m
    if op == "and":
        return (vals[0] & vals[1]) & m
    if op == "or":
        return (vals[0] | vals[1]) & m
    if op == "xor":
        return (vals[0] ^ vals[1]) & m
    if op == "shl":
        sh = vals[1] & (width - 1)
        return (vals[0] << sh) & m
    if op == "lshr":
        sh = vals[1] & (width - 1)
        return (vals[0] & m) >> sh
    if op == "ashr":
        sh = vals[1] & (width - 1)
        return (to_signed(vals[0], width) >> sh) & m
    if op == "rol":
        sh = vals[1] % width
        v = vals[0] & m
        return ((v << sh) | (v >> (width - sh))) & m if sh else v
    if op == "ror":
        sh = vals[1] % width
        v = vals[0] & m
        return ((v >> sh) | (v << (width - sh))) & m if sh else v
    if op == "adc":
        return (vals[0] + vals[1] + (vals[2] & 1)) & m
    if op == "sbb":
        return (vals[0] - vals[1] - (vals[2] & 1)) & m
    raise ValueError(f"Unsupported op: {op}")


def fold_and_normalize(expr: Expr) -> Expr:
    """Constant-fold and normalize commutative operands for canonical CSE cost."""
    if isinstance(expr, (Var, Const)):
        return expr
    if isinstance(expr, Unary):
        arg = fold_and_normalize(expr.arg)
        if isinstance(arg, Const):
            return Const(expr.width, _eval_op_int(expr.op, (arg.value,), expr.width))
        if expr.op == "not" and isinstance(arg, Unary) and arg.op == "not":
            return arg.arg
        if expr.op == "neg" and isinstance(arg, Unary) and arg.op == "neg":
            return arg.arg
        return Unary(expr.width, expr.op, arg)
    if isinstance(expr, Binary):
        left = fold_and_normalize(expr.left)
        right = fold_and_normalize(expr.right)
        if isinstance(left, Const) and isinstance(right, Const):
            return Const(expr.width, _eval_op_int(expr.op, (left.value, right.value), expr.width))
        if expr.op in COMMUTATIVE_OPS and repr(left) > repr(right):
            left, right = right, left
        # Algebraic identity folds (zero/one/idempotent)
        if expr.op == "add":
            if isinstance(left, Const) and left.value == 0:
                return right
            if isinstance(right, Const) and right.value == 0:
                return left
        elif expr.op == "sub":
            if isinstance(right, Const) and right.value == 0:
                return left
            if left == right:
                return Const(expr.width, 0)
        elif expr.op == "xor":
            if isinstance(left, Const) and left.value == 0:
                return right
            if isinstance(right, Const) and right.value == 0:
                return left
            if left == right:
                return Const(expr.width, 0)
        elif expr.op == "or":
            if isinstance(left, Const) and left.value == 0:
                return right
            if isinstance(right, Const) and right.value == 0:
                return left
            if left == right:
                return left
        elif expr.op == "and":
            m = mask_for(expr.width)
            if isinstance(left, Const) and left.value == 0:
                return Const(expr.width, 0)
            if isinstance(right, Const) and right.value == 0:
                return Const(expr.width, 0)
            if isinstance(left, Const) and left.value == m:
                return right
            if isinstance(right, Const) and right.value == m:
                return left
            if left == right:
                return left
        elif expr.op == "mul":
            if isinstance(left, Const) and left.value == 0:
                return Const(expr.width, 0)
            if isinstance(right, Const) and right.value == 0:
                return Const(expr.width, 0)
            if isinstance(left, Const) and left.value == 1:
                return right
            if isinstance(right, Const) and right.value == 1:
                return left
        return Binary(expr.width, expr.op, left, right)
    if isinstance(expr, Ternary):
        a = fold_and_normalize(expr.a)
        b = fold_and_normalize(expr.b)
        c = fold_and_normalize(expr.c)
        if isinstance(a, Const) and isinstance(b, Const) and isinstance(c, Const):
            return Const(expr.width, _eval_op_int(expr.op, (a.value, b.value, c.value), expr.width))
        return Ternary(expr.width, expr.op, a, b, c)
    raise TypeError(f"Unknown expr type: {type(expr)}")


def canonical_cost(expr: Expr) -> int:
    """Count unique non-leaf DAG operation nodes after constant folding and CSE."""
    norm = fold_and_normalize(expr)
    seen: set[Expr] = set()

    def visit(node: Expr) -> None:
        if isinstance(node, (Var, Const)) or node in seen:
            return
        seen.add(node)
        if isinstance(node, Unary):
            visit(node.arg)
        elif isinstance(node, Binary):
            visit(node.left)
            visit(node.right)
        elif isinstance(node, Ternary):
            visit(node.a)
            visit(node.b)
            visit(node.c)

    visit(norm)
    return len(seen)


def eval_concrete(expr: Expr, env: dict[str, int]) -> int:
    """Evaluate expression on concrete integer inputs."""
    if isinstance(expr, Var):
        return env[expr.name] & mask_for(expr.width)
    if isinstance(expr, Const):
        return expr.value & mask_for(expr.width)
    if isinstance(expr, Unary):
        return _eval_op_int(expr.op, (eval_concrete(expr.arg, env),), expr.width)
    if isinstance(expr, Binary):
        return _eval_op_int(
            expr.op,
            (eval_concrete(expr.left, env), eval_concrete(expr.right, env)),
            expr.width,
        )
    if isinstance(expr, Ternary):
        return _eval_op_int(
            expr.op,
            (
                eval_concrete(expr.a, env),
                eval_concrete(expr.b, env),
                eval_concrete(expr.c, env),
            ),
            expr.width,
        )
    raise TypeError(f"Unknown expr: {type(expr)}")


def eval_numpy(expr: Expr, env: dict[str, np.ndarray]) -> np.ndarray:
    """Vectorized concrete evaluation in uint64/uint32 for fast pre-filtering."""
    dtype = np.uint64 if expr.width == 64 else np.uint32
    if isinstance(expr, Var):
        return env[expr.name].astype(dtype, copy=False)
    if isinstance(expr, Const):
        n = next(iter(env.values())).shape[0]
        return np.full(n, expr.value, dtype=dtype)
    if isinstance(expr, Unary):
        v = eval_numpy(expr.arg, env)
        if expr.op == "neg":
            return (dtype(0) - v).astype(dtype)
        if expr.op == "not":
            return np.bitwise_not(v).astype(dtype)
        if expr.op == "zext64":
            return v.astype(np.uint64)
        if expr.op == "trunc32":
            return v.astype(np.uint32)
        raise ValueError(expr.op)
    if isinstance(expr, Binary):
        a = eval_numpy(expr.left, env)
        b = eval_numpy(expr.right, env)
        with np.errstate(over="ignore"):
            if expr.op == "add":
                return (a + b).astype(dtype)
            if expr.op == "sub":
                return (a - b).astype(dtype)
            if expr.op == "mul":
                return (a * b).astype(dtype)
            if expr.op == "and":
                return np.bitwise_and(a, b).astype(dtype)
            if expr.op == "or":
                return np.bitwise_or(a, b).astype(dtype)
            if expr.op == "xor":
                return np.bitwise_xor(a, b).astype(dtype)
            sh = (b & dtype(expr.width - 1)).astype(dtype)
            if expr.op == "shl":
                return np.left_shift(a, sh).astype(dtype)
            if expr.op == "lshr":
                return np.right_shift(a, sh).astype(dtype)
            if expr.op == "ashr":
                sdtype = np.int64 if expr.width == 64 else np.int32
                return np.right_shift(a.view(sdtype), sh.view(sdtype)).view(dtype)
            if expr.op == "rol":
                w = dtype(expr.width)
                r = (b % w).astype(dtype)
                return np.where(r == 0, a, np.bitwise_or(np.left_shift(a, r), np.right_shift(a, w - r))).astype(dtype)
            if expr.op == "ror":
                w = dtype(expr.width)
                r = (b % w).astype(dtype)
                return np.where(r == 0, a, np.bitwise_or(np.right_shift(a, r), np.left_shift(a, w - r))).astype(dtype)
        raise ValueError(expr.op)
    if isinstance(expr, Ternary):
        a = eval_numpy(expr.a, env)
        b = eval_numpy(expr.b, env)
        c = (eval_numpy(expr.c, env) & np.uint64(1)).astype(dtype)
        with np.errstate(over="ignore"):
            if expr.op == "adc":
                return (a + b + c).astype(dtype)
            if expr.op == "sbb":
                return (a - b - c).astype(dtype)
        raise ValueError(expr.op)
    raise TypeError(f"Unknown expr: {type(expr)}")


def to_z3(expr: Expr, z3_vars: dict[str, z3.BitVecRef]) -> z3.BitVecRef:
    """Convert Expr to a Z3 BitVecRef."""
    if isinstance(expr, Var):
        return z3_vars[expr.name]
    if isinstance(expr, Const):
        return z3.BitVecVal(expr.value, expr.width)
    if isinstance(expr, Unary):
        v = to_z3(expr.arg, z3_vars)
        if expr.op == "neg":
            return -v
        if expr.op == "not":
            return ~v
        if expr.op == "zext64":
            return z3.ZeroExt(64 - v.size(), v)
        if expr.op == "trunc32":
            return z3.Extract(31, 0, v)
        raise ValueError(expr.op)
    if isinstance(expr, Binary):
        a = to_z3(expr.left, z3_vars)
        b = to_z3(expr.right, z3_vars)
        if expr.op == "add":
            return a + b
        if expr.op == "sub":
            return a - b
        if expr.op == "mul":
            return a * b
        if expr.op == "and":
            return a & b
        if expr.op == "or":
            return a | b
        if expr.op == "xor":
            return a ^ b
        sh = b & z3.BitVecVal(expr.width - 1, expr.width)
        if expr.op == "shl":
            return a << sh
        if expr.op == "lshr":
            return z3.LShR(a, sh)
        if expr.op == "ashr":
            return a >> sh
        if expr.op == "rol":
            if isinstance(expr.right, Const):
                return z3.RotateLeft(a, expr.right.value % expr.width)
            return (a << sh) | z3.LShR(a, z3.BitVecVal(expr.width, expr.width) - sh)
        if expr.op == "ror":
            if isinstance(expr.right, Const):
                return z3.RotateRight(a, expr.right.value % expr.width)
            return z3.LShR(a, sh) | (a << (z3.BitVecVal(expr.width, expr.width) - sh))
        raise ValueError(expr.op)
    if isinstance(expr, Ternary):
        a = to_z3(expr.a, z3_vars)
        b = to_z3(expr.b, z3_vars)
        c = z3.ZeroExt(expr.width - 1, to_z3(expr.c, z3_vars))
        if expr.op == "adc":
            return a + b + c
        if expr.op == "sbb":
            return a - b - c
        raise ValueError(expr.op)
    raise TypeError(f"Unknown expr: {type(expr)}")


def to_bitwuzla(expr: Expr, tm: Any, bw_vars: dict[str, Any]) -> Any:
    """Convert Expr to a Bitwuzla Term using bitwuzla.TermManager."""
    import bitwuzla as bz

    if isinstance(expr, Var):
        return bw_vars[expr.name]
    if isinstance(expr, Const):
        sort = tm.mk_bv_sort(expr.width)
        return tm.mk_bv_value(sort, expr.value)
    if isinstance(expr, Unary):
        v = to_bitwuzla(expr.arg, tm, bw_vars)
        if expr.op == "neg":
            return tm.mk_term(bz.Kind.BV_NEG, [v])
        if expr.op == "not":
            return tm.mk_term(bz.Kind.BV_NOT, [v])
        if expr.op == "zext64":
            w_in = expr.arg.width
            return tm.mk_term(bz.Kind.BV_ZERO_EXTEND, [v], [64 - w_in])
        if expr.op == "trunc32":
            return tm.mk_term(bz.Kind.BV_EXTRACT, [v], [31, 0])
        raise ValueError(expr.op)
    if isinstance(expr, Binary):
        a = to_bitwuzla(expr.left, tm, bw_vars)
        b = to_bitwuzla(expr.right, tm, bw_vars)
        if expr.op == "add":
            return tm.mk_term(bz.Kind.BV_ADD, [a, b])
        if expr.op == "sub":
            return tm.mk_term(bz.Kind.BV_SUB, [a, b])
        if expr.op == "mul":
            return tm.mk_term(bz.Kind.BV_MUL, [a, b])
        if expr.op == "and":
            return tm.mk_term(bz.Kind.BV_AND, [a, b])
        if expr.op == "or":
            return tm.mk_term(bz.Kind.BV_OR, [a, b])
        if expr.op == "xor":
            return tm.mk_term(bz.Kind.BV_XOR, [a, b])
        sort = tm.mk_bv_sort(expr.width)
        sh = tm.mk_term(bz.Kind.BV_AND, [b, tm.mk_bv_value(sort, expr.width - 1)])
        if expr.op == "shl":
            return tm.mk_term(bz.Kind.BV_SHL, [a, sh])
        if expr.op == "lshr":
            return tm.mk_term(bz.Kind.BV_SHR, [a, sh])
        if expr.op == "ashr":
            return tm.mk_term(bz.Kind.BV_ASHR, [a, sh])
        if expr.op == "rol":
            if isinstance(expr.right, Const):
                return tm.mk_term(bz.Kind.BV_ROLI, [a], [expr.right.value % expr.width])
            return tm.mk_term(bz.Kind.BV_ROL, [a, sh])
        if expr.op == "ror":
            if isinstance(expr.right, Const):
                return tm.mk_term(bz.Kind.BV_RORI, [a], [expr.right.value % expr.width])
            return tm.mk_term(bz.Kind.BV_ROR, [a, sh])
        raise ValueError(expr.op)
    if isinstance(expr, Ternary):
        a = to_bitwuzla(expr.a, tm, bw_vars)
        b = to_bitwuzla(expr.b, tm, bw_vars)
        c_raw = to_bitwuzla(expr.c, tm, bw_vars)
        c = tm.mk_term(bz.Kind.BV_ZERO_EXTEND, [c_raw], [expr.width - 1])
        if expr.op == "adc":
            return tm.mk_term(bz.Kind.BV_ADD, [tm.mk_term(bz.Kind.BV_ADD, [a, b]), c])
        if expr.op == "sbb":
            return tm.mk_term(bz.Kind.BV_SUB, [tm.mk_term(bz.Kind.BV_SUB, [a, b]), c])
        raise ValueError(expr.op)
    raise TypeError(f"Unknown expr: {type(expr)}")
