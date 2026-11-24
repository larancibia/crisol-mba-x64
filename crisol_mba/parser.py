"""Strict canonical functional grammar parser & pretty-printer for CRISOL-MBA-x64.

Shared identically by A (deterministic), B (LLM one-shot), C (CEGIS no-LLM), and D (CEGIS LLM).
Grammar examples:
  64-bit: add(rdi, rsi), sub(rdi, rsi), mul(add(rdi, rsi), 7), xor(rdi, 0x1234),
          and(rdi, rsi), or(rdi, rsi), not(rdi), neg(rdi),
          shl(rdi, 5), lshr(rdi, 7), ashr(rdi, 9), rol(rdi, 11), ror(rdi, 13),
          adc(rdi, rsi, cf), sbb(rdi, rsi, cf)
  32-bit: zext64(add32(adc32(trunc32(rdi), trunc32(rsi), cf), 0x123))
          (all 32-bit ops have `32` suffix: add32, sub32, mul32, and32, or32, xor32,
           not32, neg32, shl32, lshr32, ashr32, rol32, ror32, adc32, sbb32)
"""

from __future__ import annotations

import re
from .expr import Binary, Const, Expr, Ternary, Unary, Var


class ParseError(ValueError):
    """Raised when a candidate expression violates the shared grammar."""


_TOKEN_RE = re.compile(r"\s*(?:(0x[0-9a-fA-F]+|-?\d+)|([a-zA-Z_][a-zA-Z0-9_]*)|([(),]))")

_VARS_64 = frozenset({"rdi", "rsi", "rdx", "rcx"})
_VARS_1 = frozenset({"cf"})

_UNARY_64 = frozenset({"neg", "not"})
_UNARY_32 = frozenset({"neg32", "not32"})
_BINARY_64 = frozenset({"add", "sub", "mul", "and", "or", "xor", "shl", "lshr", "ashr", "rol", "ror"})
_BINARY_32 = frozenset({f"{op}32" for op in _BINARY_64})
_TERNARY_64 = frozenset({"adc", "sbb"})
_TERNARY_32 = frozenset({"adc32", "sbb32"})


def _tokenize(text: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    pos = 0
    n = len(text)
    while pos < n:
        if text[pos].isspace():
            pos += 1
            continue
        m = _TOKEN_RE.match(text, pos)
        if not m:
            raise ParseError(f"Unexpected character at pos {pos}: {text[pos:pos+16]!r}")
        if m.group(1) is not None:
            tokens.append(("NUM", m.group(1)))
        elif m.group(2) is not None:
            tokens.append(("ID", m.group(2)))
        else:
            tokens.append(("PUNC", m.group(3)))
        pos = m.end()
    return tokens


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> tuple[str, str] | None:
        return self.tokens[self.pos] if self.pos < len(self.tokens) else None

    def consume(self, expected_val: str | None = None) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise ParseError("Unexpected end of expression")
        if expected_val is not None and tok[1] != expected_val:
            raise ParseError(f"Expected {expected_val!r}, got {tok[1]!r}")
        self.pos += 1
        return tok

    def parse_expr(self, expected_width: int = 64) -> Expr:
        tok = self.peek()
        if tok is None:
            raise ParseError("Empty expression")
        kind, val = tok
        if kind == "NUM":
            self.consume()
            iv = int(val, 0)
            return Const(expected_width, iv)
        if kind == "ID":
            self.consume()
            if self.peek() == ("PUNC", "("):
                return self._parse_call(val, expected_width)
            low = val.lower()
            if low in _VARS_64:
                if expected_width != 64:
                    raise ParseError(f"64-bit var {low} used where {expected_width}-bit expected (use trunc32({low}))")
                return Var(64, low)
            if low in _VARS_1:
                if expected_width != 1:
                    raise ParseError("1-bit var cf used where non-1-bit expected")
                return Var(1, "cf")
            raise ParseError(f"Unknown identifier: {val}")
        raise ParseError(f"Unexpected token: {tok}")

    def _parse_call(self, name: str, expected_width: int) -> Expr:
        fn = name.lower()
        self.consume("(")
        args: list[Expr] = []

        if fn == "zext64":
            if expected_width != 64:
                raise ParseError("zext64 produces 64-bit value")
            args.append(self.parse_expr(32))
            self.consume(")")
            return Unary(64, "zext64", args[0])

        if fn == "trunc32":
            if expected_width != 32:
                raise ParseError("trunc32 produces 32-bit value")
            args.append(self.parse_expr(64))
            self.consume(")")
            return Unary(32, "trunc32", args[0])

        if fn in _UNARY_64 or fn in _UNARY_32:
            w = 32 if fn.endswith("32") else 64
            if w != expected_width:
                raise ParseError(f"{fn} produces {w}-bit, expected {expected_width}-bit")
            op = fn[:-2] if w == 32 else fn
            a = self.parse_expr(w)
            self.consume(")")
            return Unary(w, op, a)

        if fn in _BINARY_64 or fn in _BINARY_32:
            w = 32 if fn.endswith("32") else 64
            if w != expected_width:
                raise ParseError(f"{fn} produces {w}-bit, expected {expected_width}-bit")
            op = fn[:-2] if w == 32 else fn
            a = self.parse_expr(w)
            self.consume(",")
            b = self.parse_expr(w)
            self.consume(")")
            return Binary(w, op, a, b)

        if fn in _TERNARY_64 or fn in _TERNARY_32:
            w = 32 if fn.endswith("32") else 64
            if w != expected_width:
                raise ParseError(f"{fn} produces {w}-bit, expected {expected_width}-bit")
            op = fn[:-2] if w == 32 else fn
            a = self.parse_expr(w)
            self.consume(",")
            b = self.parse_expr(w)
            self.consume(",")
            c = self.parse_expr(1)
            self.consume(")")
            return Ternary(w, op, a, b, c)

        raise ParseError(f"Unknown function: {name}")


def parse_expr(text: str, expected_width: int = 64) -> Expr:
    tokens = _tokenize(text.strip())
    if not tokens:
        raise ParseError("Empty expression")
    p = _Parser(tokens)
    expr = p.parse_expr(expected_width)
    if p.peek() is not None:
        raise ParseError(f"Trailing tokens: {p.tokens[p.pos:]}")
    return expr


def format_expr(e: Expr) -> str:
    """Pretty-print an Expr into the canonical functional grammar."""
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Const):
        return hex(e.value) if e.value >= 16 else str(e.value)
    if isinstance(e, Unary):
        if e.op in {"zext64", "trunc32"}:
            return f"{e.op}({format_expr(e.arg)})"
        suf = "32" if e.width == 32 else ""
        return f"{e.op}{suf}({format_expr(e.arg)})"
    if isinstance(e, Binary):
        suf = "32" if e.width == 32 else ""
        return f"{e.op}{suf}({format_expr(e.left)}, {format_expr(e.right)})"
    if isinstance(e, Ternary):
        suf = "32" if e.width == 32 else ""
        return f"{e.op}{suf}({format_expr(e.a)}, {format_expr(e.b)}, {format_expr(e.c)})"
    raise TypeError(type(e))
