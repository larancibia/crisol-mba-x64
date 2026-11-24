"""Oracle soundness suite: Bitwuzla+Z3 miter, CaDiCaL LRAT verification, and mutation testing."""

from __future__ import annotations

from crisol_mba.expr import Binary, Const, Ternary, Unary, Var, canonical_cost
from crisol_mba.generator import assemble_x64, expr_from_dict, generate_dataset
from crisol_mba.oracle_smt import SlicePreFilter, verify_candidate


def test_lrat_verified_pass_on_all_generated_families() -> None:
    """All V-eligible families must PASS with a verified LRAT proof; V_hard (IMUL) may timeout on SAT bit-blasting."""
    instances = generate_dataset(seed=20261123, n_dev_per_family=1, n_test_per_family=1)
    assert len(instances) == 14
    lrat_passed = 0
    for inst in instances:
        code = bytes.fromhex(inst.bytes_hex)
        gt = expr_from_dict(inst.ground_truth_expr)
        pf = SlicePreFilter(code, seed=20261123, n_vectors=1024)
        verdict = verify_candidate(
            code,
            gt,
            max_cost=canonical_cost(gt) * 1.10,
            prefilter=pf,
            timeout_sec=10.0,
            require_lrat=True,
        )
        if inst.family in {"test_mba_poly_mul_32", "test_mba_poly_mul_64"}:
            assert verdict.status in {"PASS", "FAIL_LRAT"}, f"Unexpected status on V_hard: {verdict}"
        else:
            assert verdict.status == "PASS", f"Expected PASS on {inst.instance_id}, got {verdict}"
            assert verdict.lrat_sha256 is not None and len(verdict.lrat_sha256) == 64
            lrat_passed += 1
    assert lrat_passed == 12


def test_mutation_testing_zero_false_pass() -> None:
    """All semantic mutants must fail (zero false PASS verdicts):
    - Constant perturbed by +1
    - ADD <-> ADC substitution
    - Inverted incoming CF
    - Omitted 32-bit zero-extension
    - Canonical cost bound violation
    """
    code1 = assemble_x64("mov rax, rdi\nxor rax, rsi\nmov r8, rdi\nand r8, rsi\nlea rax, [rax + r8*2 + 17]")
    pf1 = SlicePreFilter(code1)
    mut_const = Binary(64, "add", Binary(64, "add", Var(64, "rdi"), Var(64, "rsi")), Const(64, 18))
    v1 = verify_candidate(code1, mut_const, max_cost=10, prefilter=pf1)
    assert v1.status != "PASS" and v1.status != "LIFTER_BUG"

    code2 = assemble_x64("mov rax, rdi\nadc rax, rsi")
    pf2 = SlicePreFilter(code2)
    mut_add = Binary(64, "add", Var(64, "rdi"), Var(64, "rsi"))
    v2 = verify_candidate(code2, mut_add, max_cost=10, prefilter=pf2)
    assert v2.status in {"FAIL_PREFILTER", "FAIL_SAT"}

    mut_inv_cf = Ternary(64, "adc", Var(64, "rdi"), Var(64, "rsi"), Unary(1, "not", Var(1, "cf")))
    v3 = verify_candidate(code2, mut_inv_cf, max_cost=10, prefilter=None)
    assert v3.status == "FAIL_SAT", f"Expected native-reproduced FAIL_SAT, got {v3}"

    code4 = assemble_x64("mov eax, edi\nadd eax, esi")
    pf4 = SlicePreFilter(code4)
    mut_no_zext = Binary(64, "add", Var(64, "rdi"), Var(64, "rsi"))
    v4 = verify_candidate(code4, mut_no_zext, max_cost=10, prefilter=pf4)
    assert v4.status in {"FAIL_PREFILTER", "FAIL_SAT"}

    gt_ok = Binary(64, "add", Binary(64, "add", Var(64, "rdi"), Var(64, "rsi")), Const(64, 17))
    v5 = verify_candidate(code1, gt_ok, max_cost=1.0, prefilter=pf1)
    assert v5.status == "FAIL_COST"


def test_prefilter_evading_negative_caught_by_smt_and_native() -> None:
    """A candidate agreeing on all 1,024 concrete pre-filter vectors yet differing on a single
    point in 2^128 must still be rejected by Bitwuzla+Z3 and reproduced natively."""
    code = assemble_x64("mov rax, rdi\nadd rax, rsi")
    pf = SlicePreFilter(code, seed=20261123, n_vectors=1024)

    x = Var(64, "rdi")
    y = Var(64, "rsi")
    t = Binary(64, "and", Binary(64, "xor", x, Const(64, 0x5A5A5A5A5A5A5A5A)), Binary(64, "xor", y, Const(64, 0xA5A5A5A5A5A5A5A5)))
    rare = t
    for sh in (1, 2, 4, 8, 16, 32):
        rare = Binary(64, "and", rare, Binary(64, "ror", rare, Const(64, sh)))
    sneaky_cand = Binary(64, "xor", Binary(64, "add", x, y), rare)

    ok_pf, _, _, _ = pf.check(sneaky_cand)
    assert ok_pf is True, "Point-difference mutant must evade the 1,024-vector concrete pre-filter"

    verdict = verify_candidate(code, sneaky_cand, max_cost=50, prefilter=pf, timeout_sec=10.0)
    assert verdict.status == "FAIL_SAT", f"SMT+native must reject prefilter-evading mutant; got {verdict}"
    assert verdict.counterexample is not None
    assert verdict.orig_out != verdict.cand_out
