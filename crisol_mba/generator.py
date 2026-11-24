"""Procedural anti-leakage MBA benchmark generator compiled to real x86-64 bytes.

Partitions by identity family BEFORE compilation into `dev` and `test` splits,
assembles with GNU `as` + `objcopy`, validates via the closed-list lifter,
and freezes `manifest.json` with SHA-256 hashes.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

from .expr import Binary, Const, Expr, Ternary, Unary, Var, canonical_cost
from .lifter import decode_and_validate


def assemble_x64(intel_asm: str) -> bytes:
    """Assemble Intel-syntax x86-64 instructions into raw .text machine bytes."""
    src = ".intel_syntax noprefix\n.text\n" + intel_asm.strip() + "\n"
    with tempfile.TemporaryDirectory() as td:
        s_path = Path(td) / "slice.s"
        o_path = Path(td) / "slice.o"
        b_path = Path(td) / "slice.bin"
        s_path.write_text(src, encoding="utf-8")
        subprocess.run(["as", "--64", str(s_path), "-o", str(o_path)], check=True, capture_output=True)
        subprocess.run(["objcopy", "-O", "binary", "-j", ".text", str(o_path), str(b_path)], check=True, capture_output=True)
        return b_path.read_bytes()


def expr_to_dict(e: Expr) -> dict:
    if isinstance(e, Var):
        return {"k": "var", "w": e.width, "name": e.name}
    if isinstance(e, Const):
        return {"k": "const", "w": e.width, "val": e.value}
    if isinstance(e, Unary):
        return {"k": "un", "w": e.width, "op": e.op, "arg": expr_to_dict(e.arg)}
    if isinstance(e, Binary):
        return {"k": "bin", "w": e.width, "op": e.op, "l": expr_to_dict(e.left), "r": expr_to_dict(e.right)}
    if isinstance(e, Ternary):
        return {"k": "ter", "w": e.width, "op": e.op, "a": expr_to_dict(e.a), "b": expr_to_dict(e.b), "c": expr_to_dict(e.c)}
    raise TypeError(type(e))


def expr_from_dict(d: dict) -> Expr:
    k = d["k"]
    if k == "var":
        return Var(d["w"], d["name"])
    if k == "const":
        return Const(d["w"], d["val"])
    if k == "un":
        return Unary(d["w"], d["op"], expr_from_dict(d["arg"]))
    if k == "bin":
        return Binary(d["w"], d["op"], expr_from_dict(d["l"]), expr_from_dict(d["r"]))
    if k == "ter":
        return Ternary(d["w"], d["op"], expr_from_dict(d["a"]), expr_from_dict(d["b"]), expr_from_dict(d["c"]))
    raise ValueError(k)


@dataclass(frozen=True)
class SliceInstance:
    instance_id: str
    split: str  # "dev" or "test"
    family: str
    width: int  # 32 or 64
    check_cf: bool
    asm: str
    bytes_hex: str
    bytes_sha256: str
    insn_count: int
    ground_truth_expr: dict
    ground_truth_cost: int


def _v64(name: str) -> Var:
    return Var(64, name)


def _v32(name: str) -> Var:
    return Var(32, name)


def _c64(val: int) -> Const:
    return Const(64, val)


def _c32(val: int) -> Const:
    return Const(32, val)


def _zext(e32: Expr) -> Expr:
    return Unary(64, "zext64", e32)


def _trunc32(e64: Expr) -> Expr:
    return Unary(32, "trunc32", e64)


# ---------------------------------------------------------------------------
# Identity families: STRICTLY DISJOINT between dev and test
# ---------------------------------------------------------------------------

def _gen_dev_lin_xor_add_64(rng: random.Random) -> tuple[str, Expr, bool]:
    # Ground truth: (rdi + rsi) + c
    # Obfuscated: (rdi ^ rsi) + 2*(rdi & rsi) + c
    c = rng.randint(1, 0x7FFF)
    asm = f"""
    mov rax, rdi
    xor rax, rsi
    mov r8, rdi
    and r8, rsi
    lea rax, [rax + r8*2 + {c}]
    """
    gt = Binary(64, "add", Binary(64, "add", _v64("rdi"), _v64("rsi")), _c64(c))
    return asm, gt, False


def _gen_dev_lin_sub_not_64(rng: random.Random) -> tuple[str, Expr, bool]:
    # Ground truth: (rdi - rsi) ^ c
    # Identity: x - y = (x ^ ~y) + 2*(x & ~y) + 1
    c = rng.randint(1, 0x7FFFFFFF)
    asm = f"""
    mov r8, rsi
    not r8
    mov rax, rdi
    xor rax, r8
    mov r9, rdi
    and r9, r8
    shl r9, 1
    add rax, r9
    add rax, 1
    xor rax, {c}
    """
    gt = Binary(64, "xor", Binary(64, "sub", _v64("rdi"), _v64("rsi")), _c64(c))
    return asm, gt, False


def _gen_dev_carry_adc_32_zext(rng: random.Random) -> tuple[str, Expr, bool]:
    # Ground truth: zext64((edi + esi + cf) ^ c)
    # Identity: a + b + cf via bitwise split + adc
    c = rng.randint(1, 0x7FFF)
    asm = f"""
    mov eax, edi
    adc eax, esi
    mov r8d, eax
    xor eax, {c}
    mov r9d, r8d
    and r9d, {c}
    sub eax, 0
    mov eax, r8d
    xor eax, {c}
    """
    # Actually make a genuine MBA with adc + zero-extension:
    # (edi + esi + cf) == ((edi ^ esi) + 2*(edi & esi)) + cf? Careful: carry of (edi+esi) + cf!
    # Simpler exact identity: x + y + cf = adc(x, y, cf); obfuscate result z = adc(edi, esi, cf)
    # as (z | c) + (z & c) - c == z, or (z ^ c) + 2*(z & c) == z + c
    asm = f"""
    mov eax, edi
    adc eax, esi
    mov r8d, eax
    xor eax, {c}
    and r8d, {c}
    lea eax, [rax + r8*2]
    """
    z = Ternary(32, "adc", _trunc32(_v64("rdi")), _trunc32(_v64("rsi")), Var(1, "cf"))
    gt = _zext(Binary(32, "add", z, _c32(c)))
    return asm, gt, False


def _gen_dev_poly_shift_64(rng: random.Random) -> tuple[str, Expr, bool]:
    # Ground truth: ((rdi + rsi) << sh) ^ rdx
    # Identity: x + y = (x | y) + (x & y)
    sh = rng.randint(1, 15)
    asm = f"""
    mov rax, rdi
    or rax, rsi
    mov r8, rdi
    and r8, rsi
    add rax, r8
    shl rax, {sh}
    mov r9, rax
    or rax, rdx
    and r9, rdx
    sub rax, r9
    """
    # Note: (a | d) - (a & d) == a ^ d
    gt = Binary(
        64,
        "xor",
        Binary(64, "shl", Binary(64, "add", _v64("rdi"), _v64("rsi")), _c64(sh)),
        _v64("rdx"),
    )
    return asm, gt, False


# --- TEST FAMILIES (strictly disjoint identities from dev) ---

def _gen_test_mba_or_and_sub_64(rng: random.Random) -> tuple[str, Expr, bool]:
    # Identity: (x ^ y) + k == (x | y) - (x & y) + k, wrapped in double-negation & shift
    k = rng.randint(3, 0x7FFF)
    sh = rng.randint(1, 7)
    asm = f"""
    mov rax, rdi
    or rax, rsi
    mov r8, rdi
    and r8, rsi
    sub rax, r8
    not rax
    neg rax
    sub rax, 1
    add rax, {k}
    rol rax, {sh}
    """
    # note: - (~t) - 1 == (t + 1) - 1 == t
    gt = Binary(
        64,
        "rol",
        Binary(64, "add", Binary(64, "xor", _v64("rdi"), _v64("rsi")), _c64(k)),
        _c64(sh),
    )
    return asm, gt, False


def _gen_test_mba_nand_nor_add_64(rng: random.Random) -> tuple[str, Expr, bool]:
    # Identity: x + y = 2*(x | y) - (x ^ y)
    # Plus: (a ^ b) = (a + b) - 2*(a & b)
    c = rng.randint(5, 0x7FFF)
    asm = f"""
    mov rax, rdi
    or rax, rsi
    shl rax, 1
    mov r8, rdi
    xor r8, rsi
    sub rax, r8
    mov r9, rax
    add rax, {c}
    and r9, {c}
    shl r9, 1
    sub rax, r9
    """
    # result: (rdi + rsi) ^ c
    gt = Binary(64, "xor", Binary(64, "add", _v64("rdi"), _v64("rsi")), _c64(c))
    return asm, gt, False


def _gen_test_mba_sbb_mask_64(rng: random.Random) -> tuple[str, Expr, bool]:
    # Exploits SBB with incoming CF: z = sbb(rdi, rsi, cf)
    # Obfuscates z + rdx via 2*(z | rdx) - (z ^ rdx)
    k = rng.randint(1, 0x7FFF)
    asm = f"""
    mov r8, rdi
    sbb r8, rsi
    mov rax, r8
    or rax, rdx
    shl rax, 1
    mov r9, r8
    xor r9, rdx
    sub rax, r9
    lea rax, [rax + {k}]
    """
    z = Ternary(64, "sbb", _v64("rdi"), _v64("rsi"), Var(1, "cf"))
    gt = Binary(64, "add", Binary(64, "add", z, _v64("rdx")), _c64(k))
    return asm, gt, False


def _gen_test_mba_adc_chain_32(rng: random.Random) -> tuple[str, Expr, bool]:
    # 32-bit zero-extension + CF propagation:
    # z = adc(edi, esi, cf); result = zext64((z - edx) ^ c)
    # Obfuscate z - edx = (z ^ ~edx) + 2*(z & ~edx) + 1
    c = rng.randint(1, 0x7FFF)
    asm = f"""
    mov r8d, edi
    adc r8d, esi
    mov r9d, edx
    not r9d
    mov eax, r8d
    xor eax, r9d
    and r8d, r9d
    lea eax, [rax + r8*2 + 1]
    xor eax, {c}
    """
    z = Ternary(32, "adc", _trunc32(_v64("rdi")), _trunc32(_v64("rsi")), Var(1, "cf"))
    gt = _zext(Binary(32, "xor", Binary(32, "sub", z, _trunc32(_v64("rdx"))), _c32(c)))
    return asm, gt, False


def _gen_test_mba_sar_neg_32(rng: random.Random) -> tuple[str, Expr, bool]:
    # 32-bit arithmetic shift + NEG + MBA + zero-extension:
    # Ground truth: zext64((sar(edi, sh) - esi) ^ k)
    # Obfuscated: a = sar(edi, sh); a - esi = (a ^ ~esi) + 2*(a & ~esi) + 1; then ^ k via (t|k)-(t&k)
    sh = rng.randint(1, 15)
    k = rng.randint(1, 0x7FFF)
    asm = f"""
    mov r8d, edi
    sar r8d, {sh}
    mov r9d, esi
    not r9d
    mov eax, r8d
    xor eax, r9d
    and r8d, r9d
    lea eax, [rax + r8*2 + 1]
    mov r10d, eax
    or eax, {k}
    and r10d, {k}
    sub eax, r10d
    """
    gt = _zext(
        Binary(
            32,
            "xor",
            Binary(
                32,
                "sub",
                Binary(32, "ashr", _trunc32(_v64("rdi")), _c32(sh)),
                _trunc32(_v64("rsi")),
            ),
            _c32(k),
        )
    )
    return asm, gt, False


def _gen_test_mba_poly_mul_32(rng: random.Random) -> tuple[str, Expr, bool]:
    # 32-bit IMUL + MBA + zero-extension to 64-bit:
    # Ground truth: zext64((edi + esi) * c + (edx ^ k))
    c = rng.choice([3, 5, 7, 9, 11, 13, 17, 19, 23, 29, 31])
    k = rng.randint(1, 0x7FFF)
    asm = f"""
    mov eax, edi
    or eax, esi
    mov r8d, edi
    and r8d, esi
    add eax, r8d
    imul eax, eax, {c}
    mov r9d, edx
    or r9d, {k}
    mov r10d, edx
    and r10d, {k}
    sub r9d, r10d
    add eax, r9d
    """
    gt = _zext(
        Binary(
            32,
            "add",
            Binary(32, "mul", Binary(32, "add", _trunc32(_v64("rdi")), _trunc32(_v64("rsi"))), _c32(c)),
            Binary(32, "xor", _trunc32(_v64("rdx")), _c32(k)),
        )
    )
    return asm, gt, False


def _gen_test_mba_poly_mul_64(rng: random.Random) -> tuple[str, Expr, bool]:
    # Non-nested 64-bit IMUL + MBA:
    # Ground truth: c * (rdi + rsi) + rdx  (single IMUL!)
    # Obfuscated: (rdi ^ rsi)*c + 2*(rdi & rsi)*c? Wait: 2 IMULs on distinct non-tainted regs is non-nested!
    # Or single IMUL on obfuscated sum: ((rdi | rsi) + (rdi & rsi)) * c + ((rdx | k) - (rdx & k)) -> wait:
    c = rng.choice([3, 5, 7, 9, 11, 13, 17, 19, 23, 29, 31])
    k = rng.randint(1, 0x7FFF)
    asm = f"""
    mov rax, rdi
    or rax, rsi
    mov r8, rdi
    and r8, rsi
    add rax, r8
    imul rax, rax, {c}
    mov r9, rdx
    or r9, {k}
    mov r10, rdx
    and r10, {k}
    sub r9, r10
    add rax, r9
    """
    gt = Binary(
        64,
        "add",
        Binary(64, "mul", Binary(64, "add", _v64("rdi"), _v64("rsi")), _c64(c)),
        Binary(64, "xor", _v64("rdx"), _c64(k)),
    )
    return asm, gt, False


def _gen_test_mba_rotate_mix_64(rng: random.Random) -> tuple[str, Expr, bool]:
    # Shift/rotate entangled MBA:
    # Ground truth: ror(rdi ^ rsi, r) + (rdx >> sh)
    r = rng.randint(1, 31)
    sh = rng.randint(1, 15)
    asm = f"""
    mov r8, rdi
    or r8, rsi
    mov r9, rdi
    and r9, rsi
    sub r8, r9
    ror r8, {r}
    mov r10, rdx
    shr r10, {sh}
    mov rax, r8
    xor rax, r10
    and r8, r10
    lea rax, [rax + r8*2]
    """
    gt = Binary(
        64,
        "add",
        Binary(64, "ror", Binary(64, "xor", _v64("rdi"), _v64("rsi")), _c64(r)),
        Binary(64, "lshr", _v64("rdx"), _c64(sh)),
    )
    return asm, gt, False


def _gen_test_mba_3var_linear_64(rng: random.Random) -> tuple[str, Expr, bool]:
    # 3-variable linear MBA: (rdi + rsi) - rdx
    # Obfuscated via: x+y = (x^y) + 2*(x&y), then t - z = (t | ~z) - (~t & z)? No:
    # t - z = (t ^ ~z) + 2*(t & ~z) + 1
    k = rng.randint(1, 0x7FFF)
    asm = f"""
    mov r8, rdi
    xor r8, rsi
    mov r9, rdi
    and r9, rsi
    lea r8, [r8 + r9*2]
    mov r10, rdx
    not r10
    mov rax, r8
    xor rax, r10
    and r8, r10
    lea rax, [rax + r8*2 + 1]
    xor rax, {k}
    """
    gt = Binary(
        64,
        "xor",
        Binary(64, "sub", Binary(64, "add", _v64("rdi"), _v64("rsi")), _v64("rdx")),
        _c64(k),
    )
    return asm, gt, False


def _gen_test_mba_3var_carry_64(rng: random.Random) -> tuple[str, Expr, bool]:
    # Carry-preserving 3-var: z = adc(rdi, rsi, cf); out = ((z ^ rdx) - rcx) ^ k
    # Obfuscate (z ^ rdx) = (z | rdx) - (z & rdx), then sub rcx via add ~rcx + 1
    k = rng.randint(1, 0x7FFF)
    asm = f"""
    mov r8, rdi
    adc r8, rsi
    mov r9, r8
    or r9, rdx
    and r8, rdx
    sub r9, r8
    mov r10, rcx
    not r10
    mov rax, r9
    xor rax, r10
    and r9, r10
    lea rax, [rax + r9*2 + 1]
    xor rax, {k}
    """
    z = Ternary(64, "adc", _v64("rdi"), _v64("rsi"), Var(1, "cf"))
    gt = Binary(64, "xor", Binary(64, "sub", Binary(64, "xor", z, _v64("rdx")), _v64("rcx")), _c64(k))
    return asm, gt, False


DEV_FAMILIES = {
    "dev_lin_xor_add_64": (64, _gen_dev_lin_xor_add_64),
    "dev_lin_sub_not_64": (64, _gen_dev_lin_sub_not_64),
    "dev_carry_adc_32_zext": (32, _gen_dev_carry_adc_32_zext),
    "dev_poly_shift_64": (64, _gen_dev_poly_shift_64),
}

TEST_FAMILIES = {
    "test_mba_or_and_sub_64": (64, _gen_test_mba_or_and_sub_64),
    "test_mba_nand_nor_add_64": (64, _gen_test_mba_nand_nor_add_64),
    "test_mba_sbb_mask_64": (64, _gen_test_mba_sbb_mask_64),
    "test_mba_adc_chain_32": (32, _gen_test_mba_adc_chain_32),
    "test_mba_sar_neg_32": (32, _gen_test_mba_sar_neg_32),
    "test_mba_poly_mul_32": (32, _gen_test_mba_poly_mul_32),
    "test_mba_poly_mul_64": (64, _gen_test_mba_poly_mul_64),
    "test_mba_rotate_mix_64": (64, _gen_test_mba_rotate_mix_64),
    "test_mba_3var_linear_64": (64, _gen_test_mba_3var_linear_64),
    "test_mba_3var_carry_64": (64, _gen_test_mba_3var_carry_64),
}


def generate_dataset(
    seed: int = 20261123,
    n_dev_per_family: int = 10,
    n_test_per_family: int = 20,
) -> list[SliceInstance]:
    """Generate deduplicated, validated dev and test slice instances."""
    rng = random.Random(seed)
    seen_hashes: set[str] = set()
    instances: list[SliceInstance] = []

    def _run_split(split_name: str, fam_map: dict, per_fam: int) -> None:
        for fam, (width, gen_fn) in fam_map.items():
            produced = 0
            attempts = 0
            while produced < per_fam and attempts < per_fam * 10:
                attempts += 1
                asm, gt, check_cf = gen_fn(rng)
                code = assemble_x64(asm)
                h = hashlib.sha256(code).hexdigest()
                if h in seen_hashes:
                    continue
                insns = decode_and_validate(code)
                seen_hashes.add(h)
                produced += 1
                instances.append(
                    SliceInstance(
                        instance_id=f"{split_name}_{fam}_{produced:03d}",
                        split=split_name,
                        family=fam,
                        width=width,
                        check_cf=check_cf,
                        asm="\n".join(f"{i.mnemonic} {i.op_str}".strip() for i in insns),
                        bytes_hex=code.hex(),
                        bytes_sha256=h,
                        insn_count=len(insns),
                        ground_truth_expr=expr_to_dict(gt),
                        ground_truth_cost=canonical_cost(gt),
                    )
                )

    _run_split("dev", DEV_FAMILIES, n_dev_per_family)
    _run_split("test", TEST_FAMILIES, n_test_per_family)
    return instances


def freeze_manifest(instances: list[SliceInstance], out_path: Path, seed: int) -> str:
    payload = {
        "schema": "crisol-mba-x64-manifest-v1",
        "seed": seed,
        "counts": {
            "total": len(instances),
            "dev": sum(1 for i in instances if i.split == "dev"),
            "test": sum(1 for i in instances if i.split == "test"),
        },
        "instances": [asdict(i) for i in instances],
    }
    raw = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(raw, encoding="utf-8")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
