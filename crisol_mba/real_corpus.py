"""Real compiled C (-O2) MBA-obfuscated benchmark corpus (`V_real`).

Compiles genuine C functions (crypto/hash/bit-twiddling primitives with Tigress-style
EncodeArithmetic MBA obfuscation) using real `gcc -O2` and `clang -O2`, extracts the
stripped `.text` machine code bytes, strips trailing `ret`, validates against the
closed-list x86-64 lifter, and verifies 0 discrepancies on 1024 vectors vs Native + Unicorn.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path

from .expr import Binary, Const, Expr, Unary, Var, canonical_cost, eval_concrete
from .generator import SliceInstance, expr_to_dict
from .lifter import decode_and_validate
from .oracle_native import NativeSliceRunner, UnicornSliceRunner, generate_test_vectors


def compile_c_function_x64(c_source: str, compiler: str = "gcc", opt: str = "-O2") -> bytes:
    """Compile a single C function to stripped x86-64 .text bytes (trailing `ret` removed)."""
    with tempfile.TemporaryDirectory() as td:
        c_path = Path(td) / "fn.c"
        o_path = Path(td) / "fn.o"
        b_path = Path(td) / "fn.bin"
        c_path.write_text(c_source, encoding="utf-8")
        cmd = [
            compiler, opt, "-c",
            "-fno-inline", "-fno-stack-protector", "-fomit-frame-pointer",
            "-mno-red-zone", "-fno-asynchronous-unwind-tables",
            "-fno-ident", "-march=x86-64",
            str(c_path), "-o", str(o_path),
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        subprocess.run(["objcopy", "-O", "binary", "-j", ".text", str(o_path), str(b_path)], check=True, capture_output=True)
        raw = b_path.read_bytes()
        # Strip trailing alignment NOPs (0x90) / endbr64 (f3 0f 1e fa) and final ret (0xc3)
        while raw.endswith(b"\x90"):
            raw = raw[:-1]
        if raw.startswith(b"\xf3\x0f\x1e\xfa"):
            raw = raw[4:]
        if not raw.endswith(b"\xc3"):
            raise ValueError(f"Expected trailing ret (0xc3), got {raw.hex()}")
        return raw[:-1]


# Tigress EncodeArithmetic-style C macros that force real MBA arithmetic even under -O2
# (using volatile-free algebraic disguises that gcc/clang -O2 do NOT collapse!)
_C_HEADER = """#include <stdint.h>
static inline uint64_t ror64(uint64_t v, unsigned s) { return (v >> s) | (v << (64 - s)); }
static inline uint64_t rol64(uint64_t v, unsigned s) { return (v << s) | (v >> (64 - s)); }
static inline uint32_t ror32(uint32_t v, unsigned s) { return (v >> s) | (v << (32 - s)); }
static inline uint32_t rol32(uint32_t v, unsigned s) { return (v << s) | (v >> (32 - s)); }
"""


def _real_c_hash_mix_64(rng: random.Random) -> tuple[str, Expr, int]:
    # Hash mixer: ((x ^ y) + k) ^ (z >> sh), obfuscated with Tigress-style MBA identities
    k = rng.randint(0x101, 0x7FFF)
    sh = rng.randint(3, 19)
    c_src = _C_HEADER + f"""
uint64_t target(uint64_t x, uint64_t y, uint64_t z) {{
    uint64_t t1 = (x | y) - (x & y);                /* x ^ y */
    uint64_t t2 = (t1 ^ {k}ULL) + 2ULL * (t1 & {k}ULL); /* t1 + k */
    uint64_t t3 = z >> {sh};
    return (t2 | t3) - (t2 & t3);                   /* t2 ^ t3 */
}}
"""
    gt = Binary(
        64, "xor",
        Binary(64, "add", Binary(64, "xor", Var(64, "rdi"), Var(64, "rsi")), Const(64, k)),
        Binary(64, "lshr", Var(64, "rdx"), Const(64, sh)),
    )
    return c_src, gt, 64


def _real_c_arx_round_64(rng: random.Random) -> tuple[str, Expr, int]:
    # Speck/TEA-like ARX round step: rol64(x + y, r) ^ z, obfuscated with 2-layer MBA that -O2 does not collapse
    r = rng.randint(5, 29)
    k = rng.randint(0x11, 0x7FFF)
    c_src = _C_HEADER + f"""
uint64_t target(uint64_t x, uint64_t y, uint64_t z) {{
    uint64_t a = (x ^ y) + 2ULL * (x & y);
    uint64_t b = 2ULL * (a | {k}ULL) - (a ^ {k}ULL);  /* (x + y) + k */
    uint64_t rot = rol64(b, {r});
    return (rot | z) - (rot & z);                     /* rot ^ z */
}}
"""
    gt = Binary(
        64, "xor",
        Binary(64, "rol", Binary(64, "add", Binary(64, "add", Var(64, "rdi"), Var(64, "rsi")), Const(64, k)), Const(64, r)),
        Var(64, "rdx"),
    )
    return c_src, gt, 64


def _real_c_sub_rotate_32(rng: random.Random) -> tuple[str, Expr, int]:
    # 32-bit checksum step (zero-extended to 64-bit): ror32(x - y, r) ^ (z + k)
    r = rng.randint(3, 23)
    k = rng.randint(0x11, 0x7FFF)
    c_src = _C_HEADER + f"""
uint32_t target(uint32_t x, uint32_t y, uint32_t z) {{
    uint32_t ny = ~y;
    uint32_t d = (x ^ ny) + 2U * (x & ny) + 1U;       /* x - y */
    uint32_t rot = ror32(d, {r});
    uint32_t zk = (z ^ {k}U) + 2U * (z & {k}U);       /* z + k */
    return (rot | zk) - (rot & zk);                   /* rot ^ (z + k) */
}}
"""
    gt = Unary(
        64, "zext64",
        Binary(
            32, "xor",
            Binary(32, "ror", Binary(32, "sub", Unary(32, "trunc32", Var(64, "rdi")), Unary(32, "trunc32", Var(64, "rsi"))), Const(32, r)),
            Binary(32, "add", Unary(32, "trunc32", Var(64, "rdx")), Const(32, k)),
        ),
    )
    return c_src, gt, 32


def _real_c_signed_shift_32(rng: random.Random) -> tuple[str, Expr, int]:
    # Signed arithmetic shift + subtraction + XOR mask (32-bit -> zext64)
    sh = rng.randint(2, 15)
    k = rng.randint(0x101, 0x7FFF)
    c_src = _C_HEADER + f"""
uint32_t target(int32_t x, uint32_t y) {{
    uint32_t a = (uint32_t)(x >> {sh});
    uint32_t s = (a ^ ~y) + 2U * (a & ~y) + 1U;     /* a - y */
    return (s | {k}U) - (s & {k}U);                 /* s ^ k */
}}
"""
    gt = Unary(
        64, "zext64",
        Binary(
            32, "xor",
            Binary(32, "sub", Binary(32, "ashr", Unary(32, "trunc32", Var(64, "rdi")), Const(32, sh)), Unary(32, "trunc32", Var(64, "rsi"))),
            Const(32, k),
        ),
    )
    return c_src, gt, 32


def _real_c_affine_mul_64(rng: random.Random) -> tuple[str, Expr, int]:
    # Hash finalizer with non-nested 64-bit IMUL + MBA: ((x + y) * c) ^ z
    c = rng.choice([3, 5, 7, 9, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47])
    c_src = _C_HEADER + f"""
uint64_t target(uint64_t x, uint64_t y, uint64_t z) {{
    uint64_t s = (x ^ y) + 2ULL * (x & y);          /* x + y */
    uint64_t p = s * {c}ULL;
    return (p | z) - (p & z);                       /* p ^ z */
}}
"""
    gt = Binary(
        64, "xor",
        Binary(64, "mul", Binary(64, "add", Var(64, "rdi"), Var(64, "rsi")), Const(64, c)),
        Var(64, "rdx"),
    )
    return c_src, gt, 64


def _real_c_4var_token_64(rng: random.Random) -> tuple[str, Expr, int]:
    # 4-variable token mixer: ((x + y) ^ z) - w + k, MBA-obfuscated
    k = rng.randint(0x11, 0x7FFF)
    c_src = _C_HEADER + f"""
uint64_t target(uint64_t x, uint64_t y, uint64_t z, uint64_t w) {{
    uint64_t s = 2ULL * (x | y) - (x ^ y);          /* x + y */
    uint64_t t = (s | z) - (s & z);                 /* s ^ z */
    uint64_t nw = ~w;
    uint64_t u = (t ^ nw) + 2ULL * (t & nw) + 1ULL; /* t - w */
    return (u ^ {k}ULL) + 2ULL * (u & {k}ULL);      /* u + k */
}}
"""
    gt = Binary(
        64, "add",
        Binary(64, "sub", Binary(64, "xor", Binary(64, "add", Var(64, "rdi"), Var(64, "rsi")), Var(64, "rdx")), Var(64, "rcx")),
        Const(64, k),
    )
    return c_src, gt, 64


REAL_C_FAMILIES = {
    "real_c_hash_mix_64": _real_c_hash_mix_64,
    "real_c_arx_round_64": _real_c_arx_round_64,
    "real_c_sub_rotate_32": _real_c_sub_rotate_32,
    "real_c_signed_shift_32": _real_c_signed_shift_32,
    "real_c_affine_mul_64": _real_c_affine_mul_64,
    "real_c_4var_token_64": _real_c_4var_token_64,
}


def generate_real_compiled_dataset(
    seed: int = 20261124,
    n_per_family: int = 10,
    compilers: tuple[str, ...] = ("gcc", "clang"),
) -> list[SliceInstance]:
    """Generate real gcc -O2 / clang -O2 compiled MBA slices, verified on 1024 vectors vs Native + Unicorn."""
    rng = random.Random(seed)
    vectors = generate_test_vectors(seed=seed, n_total=1024)
    seen: set[str] = set()
    out: list[SliceInstance] = []

    for fam, gen_fn in REAL_C_FAMILIES.items():
        produced = 0
        attempts = 0
        while produced < n_per_family and attempts < n_per_family * 8:
            attempts += 1
            comp = compilers[produced % len(compilers)]
            c_src, gt, width = gen_fn(rng)
            try:
                code = compile_c_function_x64(c_src, compiler=comp, opt="-O2")
                insns = decode_and_validate(code)
            except Exception:
                continue
            # Ensure compiler did NOT already collapse the MBA to trivial length (< 4 insns)
            if len(insns) < 4:
                continue
            h = hashlib.sha256(code).hexdigest()
            if h in seen:
                continue
            # Sanity gate: 0 discrepancies on 1024 vectors vs Native + Unicorn + GT
            uni = UnicornSliceRunner(code)
            with NativeSliceRunner(code) as nat:
                for vec in vectors:
                    nr, nc = nat.run(vec)
                    ur, uc = uni.run(vec)
                    gv = eval_concrete(gt, {"rdi": vec.rdi, "rsi": vec.rsi, "rdx": vec.rdx, "rcx": vec.rcx, "cf": vec.cf})
                    if not (nr == ur == gv and nc == uc):
                        raise AssertionError(f"Sanity mismatch on {fam} ({comp})")
            seen.add(h)
            produced += 1
            out.append(
                SliceInstance(
                    instance_id=f"real_{fam}_{comp}_{produced:03d}",
                    split="real_o2",
                    family=f"{fam}_{comp}",
                    width=width,
                    check_cf=False,
                    asm="\n".join(f"{i.mnemonic} {i.op_str}".strip() for i in insns),
                    bytes_hex=code.hex(),
                    bytes_sha256=h,
                    insn_count=len(insns),
                    ground_truth_expr=expr_to_dict(gt),
                    ground_truth_cost=canonical_cost(gt),
                )
            )
    return out
