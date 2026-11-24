"""Unit test for Intel SDM Table 3-54 32-bit address-size override LEA (`0x67` prefix)."""
import bitwuzla as bz
import z3

from crisol_mba.expr import to_bitwuzla, to_z3
from crisol_mba.lifter import lift_to_bitwuzla, lift_to_z3
from crisol_mba.oracle_lean import certify_with_lean_bv_decide
from crisol_mba.oracle_native import ConcreteVector, NativeSliceRunner, UnicornSliceRunner
from crisol_mba.parser import parse_expr


def test_lea_addr32_dest64_wrapping():
    # 67 48 8d 04 37  ->  lea rax, [edi + esi]  (AddressSize=32, OperandSize=64)
    # When edi=0xffffffff and esi=0x1, 32-bit effective address wraps to 0x0 and zero-extends to rax=0x0!
    code = bytes.fromhex("67488d0437")
    vec = ConcreteVector(rdi=0xFFFF_FFFF, rsi=1, rdx=0, rcx=0, cf=0)

    with NativeSliceRunner(code) as nr:
        nat, _ = nr.run(vec)
    uni, _ = UnicornSliceRunner(code).run(vec)
    assert nat == 0 and uni == 0

    expr = parse_expr("zext64(add32(trunc32(rdi), trunc32(rsi)))", expected_width=64)

    # Universal Z3 equivalence check across all 2^128 (rdi, rsi) pairs!
    lz = lift_to_z3(code)
    s = z3.SolverFor("QF_BV")
    s.add(lz.regs["rax"] != to_z3(expr, lz.input_vars))
    assert s.check() == z3.unsat

    # Universal Bitwuzla equivalence check across all 2^128 (rdi, rsi) pairs!
    tm = bz.TermManager()
    lb = lift_to_bitwuzla(code, tm)
    bs = bz.Bitwuzla(tm)
    bs.assert_formula(tm.mk_term(bz.Kind.DISTINCT, [lb.regs["rax"], to_bitwuzla(expr, tm, lb.input_vars)]))
    assert str(bs.check_sat()) == "unsat"

    # Lean 4 bv_decide universal equivalence check (allow 60s under background CPU load)
    lv = certify_with_lean_bv_decide(code, expr, timeout_sec=60.0)
    assert lv.verified, lv.detail


def test_rsp_operands_rejected():
    import pytest
    from crisol_mba.lifter import LiftError, decode_and_validate
    # 48 89 e0 -> mov rax, rsp (must be rejected by decode_and_validate)
    with pytest.raises(LiftError):
        decode_and_validate(bytes.fromhex("4889e0"))
    # 48 8d 04 24 -> lea rax, [rsp] (must be rejected by decode_and_validate)
    with pytest.raises(LiftError):
        decode_and_validate(bytes.fromhex("488d0424"))
