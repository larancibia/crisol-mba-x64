#!/usr/bin/env python3
"""CRISOL-MBA-x64 Experiment Runner: freeze, pilot (V_dev), blind eval (V_test), and reverify."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from scipy.stats import binomtest

from crisol_mba.expr import canonical_cost
from crisol_mba.generator import SliceInstance, expr_from_dict, freeze_manifest, generate_dataset
from crisol_mba.oracle_smt import SlicePreFilter, verify_candidate
from crisol_mba.parser import parse_expr
from crisol_mba.solvers import RunResult, run_config_A, run_config_B, run_config_C, run_config_D


REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = REPO_ROOT / "data"


def _certify_one_gt(inst_dict: dict) -> tuple[str, bool, str, str | None, float]:
    code = bytes.fromhex(inst_dict["bytes_hex"])
    gt = expr_from_dict(inst_dict["ground_truth_expr"])
    pf = SlicePreFilter(code, seed=20261123, n_vectors=1024)
    v = verify_candidate(
        code,
        gt,
        max_cost=canonical_cost(gt) * 1.10,
        prefilter=pf,
        timeout_sec=10.0,
        require_lrat=True,
    )
    return inst_dict["instance_id"], (v.status == "PASS"), v.status, v.lrat_sha256, v.elapsed_sec


def cmd_freeze(args: argparse.Namespace) -> None:
    instances = generate_dataset(
        seed=args.seed,
        n_dev_per_family=args.n_dev_per_family,
        n_test_per_family=args.n_test_per_family,
    )
    raw_manifest = DATA_DIR / "manifest_raw.json"
    raw_sha = freeze_manifest(instances, raw_manifest, seed=args.seed)
    print(f"Generated {len(instances)} slices -> {raw_manifest} (SHA-256: {raw_sha[:16]}...)")

    payload = json.loads(raw_manifest.read_text(encoding="utf-8"))
    inst_list = payload["instances"]

    # Certify obf == g with LRAT in parallel (max 12 workers per pre-registration)
    v_dev: list[dict] = []
    v_test: list[dict] = []
    v_hard: list[dict] = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_certify_one_gt, d): d for d in inst_list}
        for fut in concurrent.futures.as_completed(futs):
            d = futs[fut]
            iid, ok, status, lhash, elapsed = fut.result()
            d["gt_cert_status"] = status
            d["gt_lrat_sha256"] = lhash
            d["gt_cert_sec"] = round(elapsed, 4)
            if ok:
                if d["split"] == "dev":
                    v_dev.append(d)
                else:
                    v_test.append(d)
            else:
                v_hard.append(d)

    v_dev.sort(key=lambda x: x["instance_id"])
    v_test.sort(key=lambda x: x["instance_id"])
    v_hard.sort(key=lambda x: x["instance_id"])

    frozen = {
        "schema": "crisol-mba-x64-frozen-V-v1",
        "seed": args.seed,
        "raw_manifest_sha256": raw_sha,
        "counts": {
            "V_dev": len(v_dev),
            "V_test": len(v_test),
            "V_hard": len(v_hard),
        },
        "V_dev": v_dev,
        "V_test": v_test,
        "V_hard": v_hard,
    }
    v_path = DATA_DIR / "manifest_V.json"
    txt = json.dumps(frozen, indent=2, sort_keys=True) + "\n"
    v_path.write_text(txt, encoding="utf-8")
    v_sha = hashlib.sha256(txt.encode("utf-8")).hexdigest()
    (DATA_DIR / "manifest_V.sha256").write_text(f"{v_sha}  manifest_V.json\n", encoding="utf-8")
    print(f"Frozen V: V_dev={len(v_dev)}, V_test={len(v_test)}, V_hard={len(v_hard)} (SHA-256: {v_sha[:16]}...)")


def _run_one_config(task: tuple[str, dict, float, int, str]) -> dict:
    cfg, d, budget_sec, max_smt_calls, backend = task
    code = bytes.fromhex(d["bytes_hex"])
    iid = d["instance_id"]
    fam = d["family"]
    gt_cost = d["ground_truth_cost"]
    asm = d["asm"]

    if cfg == "A":
        res = run_config_A(iid, fam, code, gt_cost, epsilon=0.10, budget_sec=budget_sec, max_smt_calls=max_smt_calls)
    elif cfg == "C":
        res = run_config_C(iid, fam, code, gt_cost, epsilon=0.10, budget_sec=budget_sec, max_smt_calls=max_smt_calls)
    elif cfg == "B":
        res = run_config_B(iid, fam, asm, code, gt_cost, epsilon=0.10, budget_sec=budget_sec, max_smt_calls=max_smt_calls, backend=backend)
    elif cfg == "D":
        res = run_config_D(iid, fam, asm, code, gt_cost, epsilon=0.10, budget_sec=budget_sec, max_smt_calls=max_smt_calls, backend=backend)
    else:
        raise ValueError(cfg)
    return asdict(res)


def _compute_stats(rows: list[dict], out_summary: Path) -> dict:
    by_cfg: dict[str, dict[str, dict]] = {"A": {}, "B": {}, "C": {}, "D": {}}
    for r in rows:
        by_cfg[r["config"]][r["instance_id"]] = r

    ids = sorted(set(r["instance_id"] for r in rows))
    n = len(ids)

    def rate(passed_count: int) -> float:
        return round(100.0 * passed_count / n, 2) if n else 0.0

    counts = {c: sum(1 for i in ids if by_cfg[c].get(i, {}).get("passed")) for c in ("A", "B", "C", "D")}
    ac_pass = {i: bool(by_cfg["A"].get(i, {}).get("passed") or by_cfg["C"].get(i, {}).get("passed")) for i in ids}
    d_pass = {i: bool(by_cfg["D"].get(i, {}).get("passed")) for i in ids}
    b_pass = {i: bool(by_cfg["B"].get(i, {}).get("passed")) for i in ids}
    ac_count = sum(1 for v in ac_pass.values() if v)

    # Exact McNemar: D vs (A u C)
    n10_h1 = sum(1 for i in ids if d_pass[i] and not ac_pass[i])
    n01_h1 = sum(1 for i in ids if not d_pass[i] and ac_pass[i])
    p_h1 = float(binomtest(n10_h1, n10_h1 + n01_h1, 0.5).pvalue) if (n10_h1 + n01_h1) > 0 else 1.0
    diff_h1_pp = rate(counts["D"]) - rate(ac_count)

    # Exact McNemar: D vs B (H2)
    n10_h2 = sum(1 for i in ids if d_pass[i] and not b_pass[i])
    n01_h2 = sum(1 for i in ids if not d_pass[i] and b_pass[i])
    p_h2 = float(binomtest(n10_h2, n10_h2 + n01_h2, 0.5).pvalue) if (n10_h2 + n01_h2) > 0 else 1.0
    diff_h2_pp = rate(counts["D"]) - rate(counts["B"])

    # Per-family breakdown
    fams = sorted(set(r["family"] for r in rows))
    fam_table = {}
    for fam in fams:
        f_ids = [i for i in ids if by_cfg["A"].get(i, by_cfg["D"].get(i, {})).get("family") == fam]
        fn = len(f_ids)
        fam_table[fam] = {
            "n": fn,
            "A": sum(1 for i in f_ids if by_cfg["A"].get(i, {}).get("passed")),
            "B": sum(1 for i in f_ids if by_cfg["B"].get(i, {}).get("passed")),
            "C": sum(1 for i in f_ids if by_cfg["C"].get(i, {}).get("passed")),
            "D": sum(1 for i in f_ids if by_cfg["D"].get(i, {}).get("passed")),
            "A_u_C": sum(1 for i in f_ids if ac_pass[i]),
        }

    go_h1 = (diff_h1_pp >= 15.0) and (p_h1 < 0.01) and (counts["D"] > counts["C"])
    summary = {
        "n_instances": n,
        "pass_counts": {**counts, "A_u_C": ac_count},
        "pass_rates_pct": {
            "A": rate(counts["A"]),
            "B": rate(counts["B"]),
            "C": rate(counts["C"]),
            "D": rate(counts["D"]),
            "A_u_C": rate(ac_count),
        },
        "H1_D_vs_AuC": {
            "diff_pp": round(diff_h1_pp, 2),
            "n10_D_only": n10_h1,
            "n01_AuC_only": n01_h1,
            "mcnemar_exact_p": p_h1,
            "pass_H1_alpha_0_01": go_h1,
        },
        "H2_D_vs_B": {
            "diff_pp": round(diff_h2_pp, 2),
            "n10_D_only": n10_h2,
            "n01_B_only": n01_h2,
            "mcnemar_exact_p": p_h2,
            "pass_H2_alpha_0_01": (diff_h2_pp > 0) and (p_h2 < 0.01),
        },
        "per_family": fam_table,
        "verdict": "GO" if go_h1 else "NO-GO",
    }
    out_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def _run_suite(
    instances: list[dict],
    tag: str,
    budget_sec: float,
    max_smt_calls: int,
    backend: str,
    det_workers: int,
    llm_workers: int,
) -> dict:
    jsonl_path = DATA_DIR / f"results_{tag}.jsonl"
    csv_path = DATA_DIR / f"results_{tag}.csv"
    summary_path = DATA_DIR / f"summary_{tag}.json"

    # Resume support: load already completed (config, instance_id) pairs
    rows: list[dict] = []
    done: set[tuple[str, str]] = set()
    if jsonl_path.exists():
        for line in jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append(r)
                done.add((r["config"], r["instance_id"]))

    # Phase 1: Deterministic configs A & C in parallel (up to det_workers)
    det_tasks = [
        (cfg, d, budget_sec, max_smt_calls, backend)
        for cfg in ("A", "C")
        for d in instances
        if (cfg, d["instance_id"]) not in done
    ]
    if det_tasks:
        print(f"[{tag}] Running {len(det_tasks)} deterministic tasks (A, C) with {det_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=det_workers) as ex, jsonl_path.open("a", encoding="utf-8") as f:
            futs = [ex.submit(_run_one_config, t) for t in det_tasks]
            for idx, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                r = fut.result()
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                if idx % 20 == 0 or idx == len(det_tasks):
                    print(f"  [A/C] {idx}/{len(det_tasks)} done")

    # Phase 2: LLM configs B & D with bounded concurrency (llm_workers)
    llm_tasks = [
        (cfg, d, budget_sec, max_smt_calls, backend)
        for cfg in ("B", "D")
        for d in instances
        if (cfg, d["instance_id"]) not in done
    ]
    if llm_tasks:
        print(f"[{tag}] Running {len(llm_tasks)} LLM tasks (B, D; backend={backend}) with {llm_workers} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=llm_workers) as ex, jsonl_path.open("a", encoding="utf-8") as f:
            futs = [ex.submit(_run_one_config, t) for t in llm_tasks]
            for idx, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                r = fut.result()
                rows.append(r)
                f.write(json.dumps(r) + "\n")
                f.flush()
                if idx % 10 == 0 or idx == len(llm_tasks):
                    print(f"  [B/D] {idx}/{len(llm_tasks)} done (last: {r['config']} {r['instance_id']} -> {r['status']} {r['elapsed_sec']:.1f}s)")

    # Write CSV
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(sorted(rows, key=lambda x: (x["config"], x["instance_id"])))

    summary = _compute_stats(rows, summary_path)
    print(json.dumps(summary, indent=2))
    return summary


def cmd_pilot(args: argparse.Namespace) -> None:
    frozen = json.loads((DATA_DIR / "manifest_V.json").read_text(encoding="utf-8"))
    # Select 20 instances from V_dev (5 per family)
    v_dev = frozen["V_dev"]
    by_fam: dict[str, list[dict]] = {}
    for d in v_dev:
        by_fam.setdefault(d["family"], []).append(d)
    pilot_set = [d for fam in sorted(by_fam) for d in by_fam[fam][:5]][:20]
    print(f"Running pilot on {len(pilot_set)} V_dev instances (backend={args.backend}, budget={args.budget}s)...")
    _run_suite(
        pilot_set,
        tag="pilot_dev",
        budget_sec=args.budget,
        max_smt_calls=args.max_smt,
        backend=args.backend,
        det_workers=args.det_workers,
        llm_workers=args.llm_workers,
    )


def cmd_eval(args: argparse.Namespace) -> None:
    frozen = json.loads((DATA_DIR / "manifest_V.json").read_text(encoding="utf-8"))
    v_test = frozen["V_test"]
    print(f"Running BLIND evaluation on {len(v_test)} V_test instances (backend={args.backend}, budget={args.budget}s)...")
    _run_suite(
        v_test,
        tag="test",
        budget_sec=args.budget,
        max_smt_calls=args.max_smt,
        backend=args.backend,
        det_workers=args.det_workers,
        llm_workers=args.llm_workers,
    )


def _reverify_one(task: tuple[dict, dict]) -> tuple[str, str, str]:
    r, d = task
    code = bytes.fromhex(d["bytes_hex"])
    expr = parse_expr(r["best_expr"], expected_width=64)
    pf = SlicePreFilter(code)
    v = verify_candidate(
        code,
        expr,
        max_cost=d["ground_truth_cost"] * 1.10,
        prefilter=pf,
        timeout_sec=30.0,
        require_lrat=True,
    )
    return r["config"], r["instance_id"], v.status


def cmd_reverify(args: argparse.Namespace) -> None:
    frozen = json.loads((DATA_DIR / "manifest_V.json").read_text(encoding="utf-8"))
    by_id = {d["instance_id"]: d for d in frozen["V_dev"] + frozen["V_test"]}
    jsonl_path = DATA_DIR / f"results_{args.tag}.jsonl"
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    passes = [r for r in rows if r["passed"]]
    print(f"Re-verifying {len(passes)} PASS verdicts from {jsonl_path} independently with LRAT (timeout=30s, 4 workers)...")
    ok_count = 0
    fails = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as ex:
        for cfg, iid, st in ex.map(_reverify_one, [(r, by_id[r["instance_id"]]) for r in passes]):
            if st == "PASS":
                ok_count += 1
            else:
                fails.append((cfg, iid, st))
    if fails:
        print(f"REVERIFY_FAIL: {len(fails)} failures: {fails[:10]}")
        sys.exit(1)
    print(f"REVERIFY_OK: {ok_count}/{len(passes)} PASS verdicts independently re-verified with LRAT!")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_f = sub.add_parser("freeze")
    p_f.add_argument("--seed", type=int, default=20261123)
    p_f.add_argument("--n-dev-per-family", type=int, default=10)
    p_f.add_argument("--n-test-per-family", type=int, default=20)
    p_f.add_argument("--workers", type=int, default=12)

    for name in ("pilot", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--budget", type=float, default=35.0)
        p.add_argument("--max-smt", type=int, default=16)
        p.add_argument("--backend", type=str, default="agy-hydrogen")
        p.add_argument("--det-workers", type=int, default=12)
        p.add_argument("--llm-workers", type=int, default=4)

    p_r = sub.add_parser("reverify")
    p_r.add_argument("--tag", type=str, default="test")

    args = ap.parse_args()
    if args.cmd == "freeze":
        cmd_freeze(args)
    elif args.cmd == "pilot":
        cmd_pilot(args)
    elif args.cmd == "eval":
        cmd_eval(args)
    elif args.cmd == "reverify":
        cmd_reverify(args)


if __name__ == "__main__":
    main()
