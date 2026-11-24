"""Lifter fidelity suite: 3-way zero-discrepancy checks across Z3, Bitwuzla, Native x86-64, and Unicorn."""

from __future__ import annotations

import pytest
import z3
import bitwuzla as bz

from crisol_mba.expr import eval_concrete, to_bitwuzla, to_z3
from crisol_mba.generator import (
    assemble_x64,
    expr_from_dict,
    generate_dataset,
)
from crisol_mba.lifter import (
    LiftError,
    decode_and_validate,
    lift_to_bitwuzla,
    lift_to_z3,
)
from crisol_mba.oracle_native import (
    ConcreteVector,
    NativeSliceRunner,
    UnicornSliceRunner,
    generate_test_vectors,
)


def _eval_lifted_z3(lifted, vec: ConcreteVector) -> tuple[int, int]:
    subs = [
        (lifted.input_vars["rdi"], z3.BitVecVal(vec.rdi, 64)),
        (lifted.input_vars["rsi"], z3.BitVecVal(vec.rsi, 64)),
        (lifted.input_vars["rdx"], z3.BitVecVal(vec.rdx, 64)),
        (lifted.input_vars["rcx"], z3.BitVecVal(vec.rcx, 64)),
        (lifted.input_vars["cf"], z3.BitVecVal(vec.cf & 1, 1)),
    ]
    for r in ("rax", "rbx", "rbp", "r8", "r9", "r10", "r11", "r12", "r13", "r14", "r15"):
        subs.append((lifted.input_vars[r], z3.BitVecVal(0, 64)))
    rax_val = z3.simplify(z3.substitute(lifted.regs["rax"], subs)).as_long()
    cf_val = z3.simplify(z3.substitute(lifted.cf, subs)).as_long()
    return int(rax_val), int(cf_val)


def _assert_three_way_zero_discrepancies(asm: str, n_vectors: int = 1024, check_cf: bool = True) -> None:
    code = assemble_x64(asm)
    lifted_z3 = lift_to_z3(code)
    vectors = generate_test_vectors(seed=20261123, n_total=n_vectors)

    tm = bz.TermManager()
    lifted_bz = lift_to_bitwuzla(code, tm)
    assert len(lifted_bz.insns) == len(lifted_z3.insns)

    uni = UnicornSliceRunner(code)
    with NativeSliceRunner(code) as nat:
        for idx, vec in enumerate(vectors):
            nat_rax, nat_cf = nat.run(vec)
            uni_rax, uni_cf = uni.run(vec)
            z3_rax, z3_cf = _eval_lifted_z3(lifted_z3, vec)
            assert nat_rax == uni_rax == z3_rax, (
                f"RAX mismatch at vec #{idx} ({vec}): native={hex(nat_rax)} unicorn={hex(uni_rax)} z3={hex(z3_rax)}"
            )
            if check_cf:
                assert nat_cf == uni_cf == z3_cf, (
                    f"CF mismatch at vec #{idx} ({vec}): native={nat_cf} unicorn={uni_cf} z3={z3_cf}"
                )


INSTRUCTION_TEST_CASES = [
    ("mov_64", "mov rax, rdi"),
    ("mov_32_zext", "mov rax, rdi\nmov eax, esi"),
    ("mov_16_preserve", "mov rax, rdi\nmov ax, si"),
    ("mov_8l_preserve", "mov rax, rdi\nmov al, sil"),
    ("mov_8h_preserve", "mov rax, rdi\nmov ah, dh"),
    ("lea_scale_disp", "lea rax, [rdi + rsi*4 + 17]"),
    ("lea_32_zext", "mov rax, rdi\nlea eax, [rsi + rdx*2 + 9]"),
    ("add_64_cf", "mov rax, rdi\nadd rax, rsi"),
    ("add_32_zext_cf", "mov rax, rdi\nadd eax, esi"),
    ("adc_64_cf", "mov rax, rdi\nadc rax, rsi"),
    ("adc_32_zext_cf", "mov rax, rdi\nadc eax, esi"),
    ("sub_64_cf", "mov rax, rdi\nsub rax, rsi"),
    ("sbb_64_cf", "mov rax, rdi\nsbb rax, rsi"),
    ("sbb_32_zext_cf", "mov rax, rdi\nsbb eax, esi"),
    ("neg_64_cf", "mov rax, rdi\nneg rax"),
    ("neg_32_zext_cf", "mov rax, rdi\nneg eax"),
    ("not_64_cf_preserve", "mov rax, rdi\nnot rax"),
    ("and_64_cf_clear", "mov rax, rdi\nand rax, rsi"),
    ("or_64_cf_clear", "mov rax, rdi\nor rax, rsi"),
    ("xor_64_cf_clear", "mov rax, rdi\nxor rax, rsi"),
    ("imul_2op_64", "mov rax, rdi\nimul rax, rsi"),
    ("imul_3op_64_imm", "imul rax, rdi, 13"),
    ("imul_32_zext", "mov rax, rdi\nimul eax, esi, 7"),
    ("shl_64_imm", "mov rax, rdi\nshl rax, 5"),
    ("shl_32_zext", "mov rax, rdi\nshl eax, 3"),
    ("shr_64_imm", "mov rax, rdi\nshr rax, 7"),
    ("sar_64_imm", "mov rax, rdi\nsar rax, 9"),
    ("rol_64_imm", "mov rax, rdi\nrol rax, 11"),
    ("ror_64_imm", "mov rax, rdi\nror rax, 13"),
    ("rol_32_zext", "mov rax, rdi\nrol eax, 5"),
    ("ror_32_zext", "mov rax, rdi\nror eax, 7"),
]


@pytest.mark.parametrize("name,asm", INSTRUCTION_TEST_CASES, ids=[c[0] for c in INSTRUCTION_TEST_CASES])
def test_all_closed_instructions_1024_vectors(name: str, asm: str) -> None:
    """Verify zero discrepancies across Native CPU, Unicorn, and Z3 lifter on 1,024 concrete vectors."""
    _assert_three_way_zero_discrepancies(asm, n_vectors=1024, check_cf=True)


def test_disallowed_instructions_rejected() -> None:
    """Ensure memory ops, control flow, nested 64-bit IMUL, and >24 instructions fail admission."""
    with pytest.raises(LiftError):
        decode_and_validate(assemble_x64("mov rax, [rdi]"))
    with pytest.raises(LiftError):
        decode_and_validate(assemble_x64("ret"))
    with pytest.raises(LiftError):
        lift_to_z3(assemble_x64("mov rax, rdi\nimul rax, rsi\nimul rax, rdx"))
    too_long = "\n".join(["add rax, rdi"] * 25)
    with pytest.raises(LiftError):
        decode_and_validate(assemble_x64(too_long))


def test_generated_dataset_sanity_and_ground_truth_equivalence() -> None:
    """Verify generated dev+test families pass 3-way native/Unicorn/reference checks and SMT equivalence."""
    instances = generate_dataset(seed=20261123, n_dev_per_family=2, n_test_per_family=2)
    assert len(instances) == 28  # 4 dev*2 + 10 test*2
    vectors = generate_test_vectors(seed=20261123, n_total=1024)

    for inst in instances:
        code = bytes.fromhex(inst.bytes_hex)
        gt = expr_from_dict(inst.ground_truth_expr)
        lifted_z3 = lift_to_z3(code)

        tm = bz.TermManager()
        lifted_bz = lift_to_bitwuzla(code, tm)
        gt_bz = to_bitwuzla(gt, tm, lifted_bz.input_vars)
        opts = bz.Options()
        opts.set(bz.Option.TIME_LIMIT_PER, 10000)
        b_solver = bz.Bitwuzla(tm, opts)
        b_solver.assert_formula(tm.mk_term(bz.Kind.DISTINCT, [lifted_bz.regs["rax"], gt_bz]))
        assert b_solver.check_sat() == bz.Result.UNSAT, f"Bitwuzla failed obf == gt for {inst.instance_id}"

        gt_z3 = to_z3(gt, lifted_z3.input_vars)
        s = z3.SolverFor("QF_BV")
        s.set("timeout", 2000)
        s.add(lifted_z3.regs["rax"] != gt_z3)
        assert s.check() != z3.sat, f"Z3 disagreed (SAT!) on {inst.instance_id}"

        uni = UnicornSliceRunner(code)
        with NativeSliceRunner(code) as nat:
            for vec in vectors:
                nat_rax, nat_cf = nat.run(vec)
                uni_rax, uni_cf = uni.run(vec)
                gt_val = eval_concrete(
                    gt,
                    {"rdi": vec.rdi, "rsi": vec.rsi, "rdx": vec.rdx, "rcx": vec.rcx, "cf": vec.cf},
                )
                assert nat_rax == uni_rax == gt_val, (
                    f"Concrete mismatch on {inst.instance_id} for {vec}: "
                    f"nat={hex(nat_rax)} uni={hex(uni_rax)} gt={hex(gt_val)}"
                )
                assert nat_cf == uni_cf
