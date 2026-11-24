#!/usr/bin/env python3
"""Evaluate official Denuvo GAMBA (build/gamba/src/simplify_general.py) on frozen V_test (N=160)."""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from crisol_mba.lifter import decode_and_validate


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"
GAMBA_SCRIPT = REPO_ROOT / "build" / "gamba" / "src" / "simplify_general.py"
PYTHON_BIN = REPO_ROOT / ".venv" / "bin" / "python"

# Operators that official GAMBA supports on pure algebraic expressions (no carry, no shifts/rotates, no partial-width zero-ext)
GAMBA_UNSUPPORTED_MNEMS = frozenset({"adc", "sbb", "rol", "ror", "shr", "sar"})


def lift_to_gamba_expr(code: bytes) -> tuple[str | None, str]:
    """Attempt to express a slice in GAMBA's pure algebraic syntax; return (expr_or_None, reason)."""
    insns = decode_and_validate(code)
    for d in insns:
        if d.mnemonic in GAMBA_UNSUPPORTED_MNEMS:
            return None, f"UNSUPPORTED_MNEM:{d.mnemonic}"
        # Check for 32-bit partial register writes (zero-extension not representable in GAMBA)
        first = d.op_str.split(",")[0].strip().lower() if d.op_str else ""
        if first in {"eax", "ebx", "ecx", "edx", "esi", "edi", "r8d", "r9d", "r10d", "r11d"}:
            return None, "UNSUPPORTED_32BIT_ZEXT"

    # Symbolic string substitution for pure 64-bit MOV/ADD/SUB/AND/OR/XOR/NOT/NEG/SHL-by-1/LEA
    regs = {"rdi": "x", "rsi": "y", "rdx": "z", "rcx": "w", "rax": "0", "r8": "0", "r9": "0", "r10": "0"}
    for d in insns:
        parts = [p.strip() for p in d.op_str.split(",")]
        dst = parts[0].lower()
        src = parts[1].lower() if len(parts) > 1 else ""
        def val(tok: str) -> str:
            if tok in regs:
                return f"({regs[tok]})"
            return str(int(tok, 0))
        m = d.mnemonic
        if m == "mov":
            regs[dst] = val(src)
        elif m == "add":
            regs[dst] = f"({regs[dst]} + {val(src)})"
        elif m == "sub":
            regs[dst] = f"({regs[dst]} - {val(src)})"
        elif m == "xor":
            regs[dst] = f"({regs[dst]} ^ {val(src)})"
        elif m == "and":
            regs[dst] = f"({regs[dst]} & {val(src)})"
        elif m == "or":
            regs[dst] = f"({regs[dst]} | {val(src)})"
        elif m == "not":
            regs[dst] = f"(~({regs[dst]}))"
        elif m == "neg":
            regs[dst] = f"(-({regs[dst]}))"
        elif m in {"shl", "sal"}:
            sh = int(src, 0)
            regs[dst] = f"({1 << sh} * ({regs[dst]}))"
        elif m == "lea":
            # Parse simple [a + b*2 + c]
            inner = src.strip("[]").strip()
            # Replace register names with their symbolic strings
            expr = inner
            for r in ("r10", "r9", "r8", "rax", "rdi", "rsi", "rdx", "rcx"):
                if r in expr:
                    expr = expr.replace(r, f"({regs[r]})")
            regs[dst] = f"({expr})"
        else:
            return None, f"UNSUPPORTED_MNEM:{m}"
    return regs["rax"], "OK"


def main() -> None:
    frozen = json.loads((DATA_DIR / "manifest_V.json").read_text(encoding="utf-8"))
    v_test = frozen["V_test"]
    rows = []
    by_fam = {}
    for d in v_test:
        fam = d["family"]
        by_fam.setdefault(fam, {"n": 0, "supported": 0, "gamba_simplified": 0, "reason": ""})
        by_fam[fam]["n"] += 1
        code = bytes.fromhex(d["bytes_hex"])
        g_expr, reason = lift_to_gamba_expr(code)
        if g_expr is None:
            by_fam[fam]["reason"] = reason
            rows.append({"instance_id": d["instance_id"], "family": fam, "supported": False, "status": reason, "output": None})
            continue
        by_fam[fam]["supported"] += 1
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [str(PYTHON_BIN), str(GAMBA_SCRIPT), "-b", "64", g_expr],
                capture_output=True, text=True, timeout=15.0,
            )
            dt = round(time.perf_counter() - t0, 3)
            # Extract simplified line
            out_lines = [l.strip() for l in proc.stdout.splitlines() if "simplified" in l.lower()]
            simp = out_lines[-1] if out_lines else proc.stdout.strip()[-120:]
            ok = (proc.returncode == 0 and bool(out_lines))
            if ok:
                by_fam[fam]["gamba_simplified"] += 1
            rows.append({"instance_id": d["instance_id"], "family": fam, "supported": True, "status": "OK" if ok else f"ERR:{proc.returncode}", "sec": dt, "output": simp})
        except subprocess.TimeoutExpired:
            rows.append({"instance_id": d["instance_id"], "family": fam, "supported": True, "status": "TIMEOUT", "output": None})

    total_sup = sum(v["supported"] for v in by_fam.values())
    total_ok = sum(v["gamba_simplified"] for v in by_fam.values())
    summary = {
        "n_v_test": len(v_test),
        "gamba_supported_slices": total_sup,
        "gamba_simplified_slices": total_ok,
        "gamba_coverage_pct": round(100.0 * total_ok / len(v_test), 2),
        "by_family": by_fam,
    }
    (DATA_DIR / "summary_gamba_vtest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
