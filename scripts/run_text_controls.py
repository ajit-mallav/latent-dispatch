"""Paired text-report pilot. Importing this file never loads a model or starts a run.

Default: 20 toy hop1 + 30 MuSiQue 2hop sources, AB/BA, seed 0, thinking off.
Uses the existing model, worker prompts, answer function, scorer and result writer.
See scripts/README.md for setup, saved report reuse, smoke and resume commands.
"""
import argparse
import collections
import fcntl
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from latent_dispatch import text_controls as controls


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_bank(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pin_manifest(path, identity, commit):
    if path.exists():
        previous = json.loads(path.read_text(encoding="utf-8"))
        if previous["identity"] != identity:
            raise ValueError("Run inputs/settings/code changed. Use a new --run-name; existing results are preserved.")
    else:
        write_json(path, {"identity": identity, "commit": commit, "created_unix": time.time()})


def arguments():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=("Qwen/Qwen3-4B", "Qwen/Qwen3-8B"), default="Qwen/Qwen3-4B")
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--thinking", nargs="+", choices=("off", "forced"), default=["off"])
    p.add_argument("--think-budget", type=int, default=1024)
    p.add_argument("--answer-tokens", type=int, default=256)
    p.add_argument("--report-tokens", type=int, default=1024)
    p.add_argument("--toy-count", type=int, default=20)
    p.add_argument("--musique-count", type=int, default=30)
    p.add_argument("--seeds", nargs="+", type=int, choices=(0, 1, 2), default=[0])
    p.add_argument("--placements", nargs="+", choices=("AB", "BA", "BB"), default=["AB", "BA"])
    p.add_argument("--run-name", default="text_controls_v1")
    p.add_argument("--report-source", type=Path, action="append", default=[],
                   help="Existing T1 JSONL or this runner's report-bank JSON; may repeat.")
    p.add_argument("--require-saved-reports", action="store_true",
                   help="Fail before loading weights if a selected question lacks a saved worker report.")
    p.add_argument("--smoke-only", action="store_true", help="One question per dataset/placement/seed cell, all arms.")
    p.add_argument("--max-hours", type=float, default=8, help="Soft runtime limit, checked between questions.")
    args = p.parse_args()
    for key in ("think_budget", "answer_tokens", "report_tokens", "toy_count", "musique_count", "max_hours"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_name):
        p.error("--run-name may contain only letters, digits, underscores and hyphens")
    for key in ("thinking", "seeds", "placements"):
        if len(set(getattr(args, key))) != len(getattr(args, key)):
            p.error(f"--{key} must not contain duplicates")
    return args


def select_questions(args, data):
    expected = {}
    for line in (data.BENCHMARK / "SHA256SUMS").read_text().splitlines():
        sha, name = line.split(maxsplit=1)
        expected[Path(name.lstrip("*")).name] = sha
    hashes = {}
    for name in ("toy.jsonl", "musique.jsonl"):
        path = data.BENCHMARK / name
        if not path.exists():
            raise ValueError(f"Missing {path}. Build benchmarks first: python -m latent_dispatch.data toy musique")
        hashes[name] = file_sha(path)
        if expected.get(name) != hashes[name]:
            raise ValueError(f"Benchmark checksum mismatch: {name}")
    qs = (data.per_cell(data.load("toy", seed=args.seeds, qtype="hop1", placement=args.placements), args.toy_count)
          + data.per_cell(data.load("musique", seed=args.seeds, qtype="2hop", placement=args.placements), args.musique_count))
    wanted = (args.toy_count + args.musique_count) * len(args.seeds) * len(args.placements)
    if len(qs) != wanted or len({q["id"] for q in qs}) != wanted:
        raise ValueError(f"Expected {wanted} unique layouts, found {len(qs)}; check counts and benchmarks")
    for dataset in {q["dataset"] for q in qs}:
        for seed in args.seeds:
            sources = [{q["source_id"] for q in qs if q["dataset"] == dataset and q["seed"] == seed
                        and q["placement"] == placement} for placement in args.placements]
            if any(s != sources[0] for s in sources):
                raise ValueError("Placements do not contain identical source questions")
    return qs, hashes


def import_reports(paths, bank, questions, args, results, prompt_sha, commit):
    """Copy reports, never answer text; old answers may have different EOS semantics."""
    by_id, checked = {q["id"]: q for q in questions}, set()
    for path in paths:
        if not path.is_file():
            raise ValueError(f"Report source not found: {path}")
        from_bank = path.suffix == ".json"
        rows = load_bank(path) if from_bank else results._read(path)
        for qid, row in rows.items():
            if qid not in by_id or qid in bank:
                continue
            if from_bank:
                if row.get("question_sha256") != controls.question_digest(by_id[qid]) or row.get("report_prompt_sha256") != prompt_sha:
                    raise ValueError(f"Report-bank input fingerprint mismatch: {qid}")
            elif row.get("arm") != "t1":
                raise ValueError(f"Expected an original T1 export, got {row.get('arm')!r}: {path}")
            controls.validate_report(row, by_id[qid], args.model, args.dtype, args.report_tokens)
            source_commit = row.get("commit", "unknown")
            if source_commit not in checked:
                results._same_inputs(source_commit, str(path))
                checked.add(source_commit)
            bank[qid] = {"id": qid, "model": args.model, "dtype": args.dtype,
                         "report_tokens": args.report_tokens, "question_sha256": controls.question_digest(by_id[qid]),
                         "report_prompt_sha256": prompt_sha, "commit": commit,
                         **{k: row[k] for k in ("reports", "worker_tokens", "truncated")},
                         "origin": {"kind": "imported_t1", "path": str(path.resolve()),
                                    "source_commit": source_commit, "source_row_sha256": controls.digest(row)}}


def summarize(qs, specs, args, results, folder, prefix):
    from latent_dispatch import analysis
    from latent_dispatch.score import normalize
    import pandas as pd

    df = results.load_results(qs, specs, args.model, args.dtype)
    if df.empty:
        return
    # Keep every completed row; a missing final answer is incorrect, not filtered out.
    df.loc[~df.final_answer_present.astype(bool), ["em", "f1"]] = 0.0
    abstentions = {"unknown", "not stated", "not provided", "cannot determine", "cannot be determined",
                   "insufficient information", "not enough information"}
    df["abstention_heuristic"] = df.pred.map(lambda p: normalize(p) in abstentions)
    df["edited_report"] = df.report_edits.map(lambda edits: any(edits))
    metrics = ["em", "f1", "think_tokens", "gen_tokens", "input_tokens", "seconds", "cut",
               "think_cut", "final_answer_present", "final_boxed", "abstention_heuristic", "edited_report"]
    group = df.groupby(["arm", "dataset", "qtype", "placement"])
    table = group[metrics].mean()
    table["completed_layouts"] = group.size()
    table["sources"] = group.source_id.nunique()
    expected = collections.Counter((q["dataset"], q["qtype"], q["placement"]) for q in qs)
    table["expected_layouts"] = [expected[tuple(index[1:])] for index in table.index]
    table.to_csv(folder / f"{prefix}_summary.csv")
    comparisons = []
    for thinking in args.thinking:
        for a, b in (("reports", "joint"), ("cleaned", "reports"), ("gold_reports", "gold_joint"),
                     ("compose_prompt", "reports")):
            names = [f"{prefix}_{c}_{thinking}" for c in (a, b)]
            t = analysis.paired(df, *names, by=("dataset", "qtype"))
            if not t.empty:
                t = t.reset_index().assign(contrast=f"{a} - {b}", thinking=thinking)
                # Exploratory intervals only; don't imply a globally corrected test family.
                comparisons.append(t.drop(columns=["p", "p_holm"], errors="ignore"))
    if comparisons:
        pd.concat(comparisons, ignore_index=True).to_csv(folder / f"{prefix}_paired.csv", index=False)
    write_json(folder / f"{prefix}_completion.json", {
        "expected_rows": len(qs) * len(specs), "completed_rows": len(df),
        "complete": len(df) == len(qs) * len(specs),
        "ci_unit": "source question; average placements/data seeds before bootstrapping",
        "gold_unit": "annotated paragraph, not gold sentence",
        "abstention_metric": "exact normalized phrase heuristic, not human labels",
    })


def check_saved_answers(saved, bank, questions):
    """Refuse resume if the report bank was lost or replaced after answers were saved."""
    by_id = {q["id"]: q for q in questions}
    for qid, row in saved.items():
        if qid not in by_id or qid not in bank:
            raise ValueError(f"Saved answer lacks its original question/report bank: {qid}")
        if (row.get("reports_sha256") != controls.digest(bank[qid]["reports"])
                or row.get("question_sha256") != controls.question_digest(by_id[qid])):
            raise ValueError(f"Saved answer used different reports or evidence: {qid}")


def main():
    args = arguments()
    from latent_dispatch import data, results
    from latent_dispatch.model import Model

    qs, benchmark_hashes = select_questions(args, data)
    prefix = args.run_name
    folder = results.results_path(args.model, args.dtype, prefix).parent
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / f"{prefix}.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("This run name is already active; do not share a result namespace concurrently.")
    source_files = ["latent_dispatch/text_controls.py", "latent_dispatch/arms.py", "latent_dispatch/model.py",
                    "latent_dispatch/prompts.py", "latent_dispatch/data.py", "scripts/run_text_controls.py"]
    code = {name: file_sha(ROOT / name) for name in source_files}
    identity = {"version": controls.VERSION, "model": args.model, "dtype": args.dtype,
                "thinking": args.thinking, "think_budget": args.think_budget, "answer_tokens": args.answer_tokens,
                "report_tokens": args.report_tokens, "conditions": list(controls.CONDITIONS),
                "benchmark_sha256": benchmark_hashes, "code_sha256": code,
                "question_sha256": {q["id"]: controls.question_digest(q) for q in qs},
                "decoding": "existing greedy decoder; no sampling seed", "gold_unit": "paragraph"}
    commit = results._commit()
    pin_manifest(folder / f"{prefix}_manifest.json", identity, commit)
    bank_path = folder / f"{prefix}_report_bank.json"
    bank = load_bank(bank_path)
    prompt_sha = code["latent_dispatch/prompts.py"]
    for q in qs:
        if q["id"] in bank:
            controls.validate_report(bank[q["id"]], q, args.model, args.dtype, args.report_tokens)
            if bank[q["id"]].get("report_prompt_sha256") != prompt_sha:
                raise ValueError("Saved report prompt changed; use a new run name")
    sources = args.report_source or [p for p in (
        folder / "thinking_pilot_t1_forced.jsonl", folder / "thinking_pilot_t1_off.jsonl",
        folder / "t1_1024.jsonl") if p.is_file()]
    import_reports(sources, bank, qs, args, results, prompt_sha, commit)
    write_json(bank_path, bank)
    smoke = data.per_cell(qs, 1)
    needed = smoke if args.smoke_only else qs
    missing = [q for q in needed if q["id"] not in bank]
    if args.require_saved_reports and missing:
        raise ValueError(f"{len(missing)} selected layouts lack saved reports. Nothing was generated.")
    deadline = time.monotonic() + args.max_hours * 3600

    def check_time():
        if time.monotonic() >= deadline:
            raise TimeoutError("Soft runtime limit reached; rerun the same command to resume")

    def ensure_reports(model, questions):
        for q in questions:
            if q["id"] in bank:
                continue
            check_time()
            started = time.monotonic()
            bank[q["id"]] = {"id": q["id"], "model": args.model, "dtype": args.dtype,
                             "report_tokens": args.report_tokens, "question_sha256": controls.question_digest(q),
                             "report_prompt_sha256": prompt_sha, "commit": commit,
                             **controls.generate_reports(model, q, args.report_tokens),
                             "origin": {"kind": "generated_once", "seconds": time.monotonic() - started}}
            write_json(bank_path, bank)
            print(f"REPORT_SAVED {q['id']}", flush=True)

    def text_control(model, q, condition, **opts):
        check_time()
        return controls.evaluate(model, q, bank[q["id"]], condition, **opts)

    specs = {f"{prefix}_{condition}_{thinking}": (text_control, dict(
        condition=condition, thinking=thinking, think_budget=args.think_budget,
        max_answer_tokens=args.answer_tokens)) for thinking in args.thinking for condition in controls.CONDITIONS}
    for name, (arm, opts) in specs.items():
        for suffix in ("", "_smoke"):
            path = results.results_path(args.model, args.dtype, name + suffix)
            saved = results._read(path)
            results._check(path, saved, {"arm": arm.__name__, "options": opts})
            check_saved_answers(saved, bank, qs)

    def pack():
        bundle = folder / f"{prefix}.tar.gz"
        temporary = bundle.with_suffix(".tmp")
        with tarfile.open(temporary, "w:gz") as tar:
            for path in sorted(folder.glob(f"{prefix}*")):
                if path.suffix in (".json", ".jsonl", ".csv"):
                    tar.add(path, arcname=str(path.relative_to(ROOT)))
        temporary.replace(bundle)
        print(f"ARTIFACTS {bundle}", flush=True)

    print(json.dumps({"model": args.model, "dtype": args.dtype, "commit": commit,
                      "question_layouts": len(qs), "source_questions": len({q['source_id'] for q in qs}),
                      "main_answer_calls": len(qs) * len(specs), "thinking": args.thinking,
                      "reports_reused": len(qs) - len([q for q in qs if q['id'] not in bank]),
                      "new_worker_calls_at_most": sum(len(q["workers"]) for q in missing),
                      "smoke_only": args.smoke_only, "max_hours": args.max_hours}, indent=2), flush=True)
    status = "failed"
    try:
        model = Model.load(args.model, dtype=args.dtype, device=args.device)
        runtime_path = folder / f"{prefix}_runtime.json"
        runtime = {
            "model_config_commit": getattr(model.hf.config, "_commit_hash", None), "device": str(model.device),
            "note": "Imported historical reports may not record a weight revision; source provenance is in the bank."}
        if runtime_path.exists():
            old = json.loads(runtime_path.read_text(encoding="utf-8"))
            if old["model_config_commit"] != runtime["model_config_commit"]:
                raise ValueError("Model revision changed on resume; use a new run name")
        write_json(runtime_path, runtime)
        ensure_reports(model, smoke)
        # Separate smoke files: errors cannot silently enter the main result table.
        for name, (arm, opts) in specs.items():
            results.run(model, smoke, name + "_smoke", arm, **opts)
            saved = results._read(results.results_path(args.model, args.dtype, name + "_smoke"))
            for q in smoke:
                row = saved[q["id"]]
                print("SMOKE", name, q["id"], json.dumps(row, ensure_ascii=False), flush=True)
                if row["cut"] or not row["final_answer_present"] or not row["final_boxed"]:
                    raise ValueError(f"Broken/truncated final answer in {name}; inspect smoke before continuing")
        print("SMOKE PASSED (format/caps, not factual accuracy)", flush=True)
        if args.smoke_only:
            status = "smoke_complete"
            return
        ensure_reports(model, qs)
        for name, (arm, opts) in specs.items():
            results.run(model, qs, name, arm, **opts)
            pack()
        status = "complete"
    except TimeoutError:
        status = "paused_at_runtime_limit"
        raise
    finally:
        try:
            summarize(qs, specs, args, results, folder, prefix)
        except Exception:
            status = "summary_failed"
            raise
        finally:
            write_json(folder / f"{prefix}_status.json", {"status": status, "updated_unix": time.time(), "commit": commit})
            pack()
            lock.close()


if __name__ == "__main__":
    main()
