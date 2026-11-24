#!/usr/bin/env python3
"""CRISOL-MBA-x64 Phase 2 Publication Benchmark Orchestrator.

Benchmarks:
  1. `v_hard_cert` : Compare Z3, Bitwuzla, CaDiCaL-LRAT, and Lean 4 `bv_decide` on V_hard (N=40 IMUL slices).
  2. `v_test_H`    : Evaluate Hybrid H (One-Shot K=6 -> CEGIS 2x2) with Gemini 4 Hydrogen on frozen V_test (N=160)
                     and GPT-6.2-Sol on stratified V_test (N=40, 5 per family) for cross-family generalization.
  3. `v_real`      : Freeze real gcc -O2 / clang -O2 compiled MBA C corpus (N=60) with Lean 4 `bv_decide`
                     certificates, and evaluate A, C, H-hydrogen, and H-sol.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing
import time
from dataclasses import asdict
from pathlib import Path

import bitwuzla as bz
import z3
from scipy.stats import binomtest

from crisol_mba.expr import canonical_cost, to_bitwuzla, to_z3
from crisol_mba.generator import expr_from_dict
from crisol_mba.lifter import lift_to_bitwuzla, lift_to_z3
from crisol_mba.oracle_lean import certify_with_lean_bv_decide
from crisol_mba.oracle_smt import certify_unsat_lrat
from crisol_mba.real_corpus import generate_real_compiled_dataset
from crisol_mba.solvers import run_config_A, run_config_C, run_config_H


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"


# ---------------------------------------------------------------------------
# 1. V_hard Certification Benchmark (Z3 vs Bitwuzla vs CaDiCaL-LRAT vs Lean 4 bv_decide)
# ---------------------------------------------------------------------------

_LEAN_SEM: multiprocessing.synchronize.Semaphore | None = None


def _init_vhard_worker(sem) -> None:
    global _LEAN_SEM
    _LEAN_SEM = sem


def _cert_one_vhard(d: dict) -> dict:
    code = bytes.fromhex(d["bytes_hex"])
    gt = expr_from_dict(d["ground_truth_expr"])

    # Bitwuzla
    t0 = time.perf_counter()
    tm = bz.TermManager()
    lb = lift_to_bitwuzla(code, tm)
    gb = to_bitwuzla(gt, tm, lb.input_vars)
    # All 4 backends evaluated under an identical 30.0 s timeout budget
    opts = bz.Options()
    opts.set(bz.Option.TIME_LIMIT_PER, 30000)
    bs = bz.Bitwuzla(tm, opts)
    bs.assert_formula(tm.mk_term(bz.Kind.DISTINCT, [lb.regs["rax"], gb]))
    bres = str(bs.check_sat())
    bt = round(time.perf_counter() - t0, 4)

    # Z3 QF_BV (30.0 s)
    t0 = time.perf_counter()
    lz = lift_to_z3(code)
    gz = to_z3(gt, lz.input_vars)
    zs = z3.SolverFor("QF_BV")
    zs.set("timeout", 30000)
    miter = (lz.regs["rax"] != gz)
    zs.add(miter)
    zres = str(zs.check())
    zt = round(time.perf_counter() - t0, 4)

    # CaDiCaL LRAT (raw Z3 bit-blast, 30.0 s)
    t0 = time.perf_counter()
    ok_lrat, _, lmsg = certify_unsat_lrat(miter, timeout_sec=30.0)
    lt = round(time.perf_counter() - t0, 4)

    # Lean 4 bv_decide (30.0 s, max 2 concurrent Lean processes to avoid multi-thread CPU contention)
    if _LEAN_SEM is not None:
        with _LEAN_SEM:
            lv = certify_with_lean_bv_decide(code, gt, timeout_sec=30.0)
    else:
        lv = certify_with_lean_bv_decide(code, gt, timeout_sec=30.0)

    return {
        "instance_id": d["instance_id"],
        "family": d["family"],
        "width": d["width"],
        "bitwuzla_unsat": (bres == "unsat"),
        "bitwuzla_sec": bt,
        "z3_unsat": (zres == "unsat"),
        "z3_sec": zt,
        "cadical_lrat_ok": ok_lrat,
        "cadical_lrat_sec": lt,
        "lean4_bv_decide_ok": lv.verified,
        "lean4_bv_decide_sec": round(lv.elapsed_sec, 4),
        "lean4_theorem_sha256": lv.theorem_sha256,
        "lean4_detail": lv.detail[:120],
    }


def cmd_vhard_cert(args: argparse.Namespace) -> None:
    frozen = json.loads((DATA_DIR / "manifest_V.json").read_text(encoding="utf-8"))
    v_hard = frozen["V_hard"]
    print(f"[v_hard_cert] Benchmarking 4 certification backends on {len(v_hard)} V_hard IMUL slices ({args.workers} workers)...")
    rows: list[dict] = []
    lean_sem = multiprocessing.Semaphore(2)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers, initializer=_init_vhard_worker, initargs=(lean_sem,)) as ex:
        futs = [ex.submit(_cert_one_vhard, d) for d in v_hard]
        for idx, fut in enumerate(concurrent.futures.as_completed(futs), 1):
            rows.append(fut.result())
            if idx % 10 == 0 or idx == len(v_hard):
                print(f"  [v_hard_cert] {idx}/{len(v_hard)} done")

    rows.sort(key=lambda x: x["instance_id"])
    csv_path = DATA_DIR / "results_vhard_cert.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    summary = {"total": len(rows), "by_family": {}}
    for fam in sorted(set(r["family"] for r in rows)):
        fr = [r for r in rows if r["family"] == fam]
        summary["by_family"][fam] = {
            "n": len(fr),
            "bitwuzla_unsat": sum(1 for r in fr if r["bitwuzla_unsat"]),
            "z3_unsat": sum(1 for r in fr if r["z3_unsat"]),
            "cadical_lrat_ok": sum(1 for r in fr if r["cadical_lrat_ok"]),
            "lean4_bv_decide_ok": sum(1 for r in fr if r["lean4_bv_decide_ok"]),
            "lean4_avg_sec": round(sum(r["lean4_bv_decide_sec"] for r in fr) / len(fr), 2),
        }
    (DATA_DIR / "summary_vhard_cert.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# 2. Hybrid H on V_test (N=160 Gemini 4 Hydrogen + N=40 stratified GPT-6.2-Sol)
# ---------------------------------------------------------------------------

def _run_H_task(task: tuple[dict, str, float, bool]) -> dict:
    d, backend, budget_sec, use_lean4 = task
    code = bytes.fromhex(d["bytes_hex"])
    res = run_config_H(
        d["instance_id"],
        d["family"],
        d["asm"],
        code,
        d["ground_truth_cost"],
        epsilon=0.10,
        budget_sec=budget_sec,
        max_smt_calls=16,
        round1_k=6,
        cegis_rounds=2,
        cegis_k=2,
        backend=backend,
        use_lean4=use_lean4,
    )
    row = asdict(res)
    row["backend"] = backend
    row["use_lean4"] = use_lean4
    return row


def cmd_vtest_H(args: argparse.Namespace) -> None:
    frozen = json.loads((DATA_DIR / "manifest_V.json").read_text(encoding="utf-8"))
    v_test = frozen["V_test"]

    # A) H-hydrogen on full V_test (N=160)
    h_hyd_path = DATA_DIR / "results_test_H_hydrogen.jsonl"
    done_hyd = set()
    rows_hyd: list[dict] = []
    if h_hyd_path.exists():
        for line in h_hyd_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows_hyd.append(r)
                done_hyd.add(r["instance_id"])

    todo_hyd = [(d, "agy-hydrogen", args.budget, False) for d in v_test if d["instance_id"] not in done_hyd]
    if todo_hyd:
        print(f"[v_test_H] Running H (agy-hydrogen) on {len(todo_hyd)} V_test instances ({args.llm_workers} workers)...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.llm_workers) as ex, h_hyd_path.open("a", encoding="utf-8") as f:
            futs = [ex.submit(_run_H_task, t) for t in todo_hyd]
            for idx, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                r = fut.result()
                rows_hyd.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                if idx % 20 == 0 or idx == len(todo_hyd):
                    print(f"  [H-hydrogen] {idx}/{len(todo_hyd)} done (last: {r['instance_id']} -> {r['status']} {r['elapsed_sec']:.1f}s)")

    # B) Cross-family generalization: H (codex-sol / GPT-6.2-Sol) on stratified 40 instances (5 per family)
    by_fam: dict[str, list[dict]] = {}
    for d in v_test:
        by_fam.setdefault(d["family"], []).append(d)
    strat40 = [d for fam in sorted(by_fam) for d in by_fam[fam][:5]]

    h_sol_path = DATA_DIR / "results_test_H_sol40.jsonl"
    done_sol = set()
    rows_sol: list[dict] = []
    if h_sol_path.exists():
        for line in h_sol_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows_sol.append(r)
                done_sol.add(r["instance_id"])

    todo_sol = [(d, "codex-sol", args.budget, False) for d in strat40 if d["instance_id"] not in done_sol]
    if todo_sol:
        print(f"[v_test_H] Running H (codex-sol / GPT-6.2-Sol) on {len(todo_sol)} stratified V_test instances ({args.llm_workers} workers)...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.llm_workers) as ex, h_sol_path.open("a", encoding="utf-8") as f:
            futs = [ex.submit(_run_H_task, t) for t in todo_sol]
            for idx, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                r = fut.result()
                rows_sol.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                if idx % 10 == 0 or idx == len(todo_sol):
                    print(f"  [H-sol] {idx}/{len(todo_sol)} done (last: {r['instance_id']} -> {r['status']} {r['elapsed_sec']:.1f}s)")

    # Summarize H vs A u C
    base_rows = [json.loads(l) for l in (DATA_DIR / "results_test.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    ac_ok = {r["instance_id"] for r in base_rows if r["config"] in {"A", "C"} and r["passed"]}
    h_ok = {r["instance_id"] for r in rows_hyd if r["passed"]}
    n10 = len(h_ok - ac_ok)
    n01 = len(ac_ok - h_ok)
    p_val = float(binomtest(n10, n10 + n01, 0.5).pvalue) if (n10 + n01) else 1.0

    sol_ok = sum(1 for r in rows_sol if r["passed"])
    summary = {
        "H_hydrogen_V_test": {
            "passed": len(h_ok),
            "total": len(v_test),
            "rate_pct": round(100.0 * len(h_ok) / len(v_test), 2),
            "avg_time_sec": round(sum(r["elapsed_sec"] for r in rows_hyd) / len(rows_hyd), 2),
            "avg_llm_calls": round(sum(r["llm_calls"] for r in rows_hyd) / len(rows_hyd), 2),
            "vs_AuC_diff_pp": round(100.0 * (len(h_ok) - len(ac_ok)) / len(v_test), 2),
            "mcnemar_p": p_val,
        },
        "H_sol_strat40": {
            "passed": sol_ok,
            "total": len(strat40),
            "rate_pct": round(100.0 * sol_ok / len(strat40), 2),
            "avg_time_sec": round(sum(r["elapsed_sec"] for r in rows_sol) / len(rows_sol), 2) if rows_sol else 0.0,
        },
    }
    (DATA_DIR / "summary_test_H.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


# ---------------------------------------------------------------------------
# 3. Real Compiled C (-O2) Corpus (`V_real`, N=60) + Evaluation (A, C, H-hydrogen, H-sol)
# ---------------------------------------------------------------------------

def _run_real_det(task: tuple[str, dict, float]) -> dict:
    cfg, d, budget = task
    code = bytes.fromhex(d["bytes_hex"])
    if cfg == "A":
        r = run_config_A(d["instance_id"], d["family"], code, d["ground_truth_cost"], epsilon=0.10, budget_sec=budget)
    else:
        r = run_config_C(d["instance_id"], d["family"], code, d["ground_truth_cost"], epsilon=0.10, budget_sec=budget)
    return asdict(r)


def cmd_vreal(args: argparse.Namespace) -> None:
    manifest_path = DATA_DIR / "manifest_V_real.json"
    if not manifest_path.exists():
        print("[v_real] Generating real gcc -O2 / clang -O2 compiled MBA corpus...")
        insts = generate_real_compiled_dataset(seed=20261124, n_per_family=args.n_per_family)
        dicts = [asdict(i) for i in insts]
        # Certify each ground truth with Lean 4 bv_decide in parallel
        print(f"[v_real] Certifying {len(dicts)} real -O2 slices with Lean 4 bv_decide...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as ex:
            futs = {
                ex.submit(certify_with_lean_bv_decide, bytes.fromhex(d["bytes_hex"]), expr_from_dict(d["ground_truth_expr"]), 25.0): d
                for d in dicts
            }
            for fut in concurrent.futures.as_completed(futs):
                d = futs[fut]
                lv = fut.result()
                d["lean4_gt_verified"] = lv.verified
                d["lean4_gt_sec"] = round(lv.elapsed_sec, 3)
                d["lean4_gt_sha256"] = lv.theorem_sha256
        dicts.sort(key=lambda x: x["instance_id"])
        raw = json.dumps({"schema": "crisol-mba-x64-real-o2-v1", "count": len(dicts), "instances": dicts}, indent=2) + "\n"
        manifest_path.write_text(raw, encoding="utf-8")
        sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        (DATA_DIR / "manifest_V_real.sha256").write_text(f"{sha}  manifest_V_real.json\n", encoding="utf-8")
        lean_ok = sum(1 for d in dicts if d["lean4_gt_verified"])
        print(f"[v_real] Frozen {len(dicts)} real -O2 slices (Lean 4 GT verified: {lean_ok}/{len(dicts)}, SHA-256: {sha[:16]}...)")

    real_data = json.loads(manifest_path.read_text(encoding="utf-8"))["instances"]
    # Keep all Lean4-verified slices for primary V_real evaluation (report any non-Lean4 separately)
    v_real = [d for d in real_data if d.get("lean4_gt_verified")]
    jsonl_path = DATA_DIR / "results_real_o2.jsonl"
    done = set()
    rows: list[dict] = []
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["config"], r["instance_id"]))

    # A & C
    det_tasks = [(cfg, d, args.budget) for cfg in ("A", "C") for d in v_real if (cfg, d["instance_id"]) not in done]
    if det_tasks:
        print(f"[v_real] Running {len(det_tasks)} A/C tasks on {len(v_real)} real -O2 slices...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=12) as ex, jsonl_path.open("a", encoding="utf-8") as f:
            for r in ex.map(_run_real_det, det_tasks):
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()

    # H-hydrogen and H-sol with Lean 4 bv_decide certification!
    for tag, backend in (("H_hydrogen", "agy-hydrogen"), ("H_sol", "codex-sol")):
        llm_tasks = [(d, backend, args.budget, True) for d in v_real if (tag, d["instance_id"]) not in done]
        if llm_tasks:
            print(f"[v_real] Running {len(llm_tasks)} {tag} (Lean 4 bv_decide certified) tasks...")
            with concurrent.futures.ProcessPoolExecutor(max_workers=args.llm_workers) as ex, jsonl_path.open("a", encoding="utf-8") as f:
                futs = [ex.submit(_run_H_task, t) for t in llm_tasks]
                for idx, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                    r = fut.result()
                    r["config"] = tag
                    rows.append(r)
                    f.write(json.dumps(r) + "\n")
                    f.flush()
                    if idx % 10 == 0 or idx == len(llm_tasks):
                        print(f"  [{tag}] {idx}/{len(llm_tasks)} done (last: {r['instance_id']} -> {r['status']} {r['elapsed_sec']:.1f}s)")

    # Summarize V_real
    n = len(v_real)
    summary = {"n_v_real": n, "total_generated": len(real_data), "configs": {}}
    for cfg in ("A", "C", "H_hydrogen", "H_sol"):
        crows = [r for r in rows if r["config"] == cfg]
        p = sum(1 for r in crows if r["passed"])
        summary["configs"][cfg] = {
            "passed": p,
            "rate_pct": round(100.0 * p / n, 2) if n else 0.0,
            "avg_sec": round(sum(r["elapsed_sec"] for r in crows) / len(crows), 2) if crows else 0.0,
        }
    ac_ids = {r["instance_id"] for r in rows if r["config"] in {"A", "C"} and r["passed"]}
    hh_ids = {r["instance_id"] for r in rows if r["config"] == "H_hydrogen" and r["passed"]}
    hs_ids = {r["instance_id"] for r in rows if r["config"] == "H_sol" and r["passed"]}
    summary["A_u_C"] = {"passed": len(ac_ids), "rate_pct": round(100.0 * len(ac_ids) / n, 2) if n else 0.0}
    for name, s_ids in (("H_hydrogen", hh_ids), ("H_sol", hs_ids)):
        n10, n01 = len(s_ids - ac_ids), len(ac_ids - s_ids)
        summary[f"mcnemar_{name}_vs_AuC"] = {
            "n10": n10, "n01": n01,
            "p": float(binomtest(n10, n10 + n01, 0.5).pvalue) if (n10 + n01) else 1.0,
        }
    (DATA_DIR / "summary_real_o2.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("vhard_cert")
    p1.add_argument("--workers", type=int, default=8)

    p2 = sub.add_parser("vtest_H")
    p2.add_argument("--budget", type=float, default=60.0)
    p2.add_argument("--llm-workers", type=int, default=6)

    p3 = sub.add_parser("vreal")
    p3.add_argument("--n-per-family", type=int, default=10)
    p3.add_argument("--budget", type=float, default=60.0)
    p3.add_argument("--llm-workers", type=int, default=4)

    args = ap.parse_args()
    if args.cmd == "vhard_cert":
        cmd_vhard_cert(args)
    elif args.cmd == "vtest_H":
        cmd_vtest_H(args)
    elif args.cmd == "vreal":
        cmd_vreal(args)


if __name__ == "__main__":
    main()
