"""Text handoff controls; no existing arm, prompt, or decoding behavior is changed.

The report bank freezes worker output across orchestrator conditions and thinking
modes. Gold controls copy annotated PARAGRAPHS verbatim, in original worker/document
order. The benchmark does not annotate exact supporting sentences (even the toy's
gold shards contain other facts), so these are not gold-sentence controls.
"""
import hashlib
import json
import re

from . import prompts

VERSION = 1
CONDITIONS = ("joint", "reports", "cleaned", "gold_reports", "gold_joint", "compose_prompt")
COMPOSE = (
    "Each report covers only part of the evidence. Combine facts across reports: "
    "an entity identified in one report may connect to a fact in another report. "
    "A worker saying its own material is insufficient does not mean the combined "
    "evidence is insufficient. Base the answer on the factual statements provided.\n\n"
)

# Deliberately conservative: only source-scoped absence/abstention sentences.
# Keep mixed clauses instead of risking removal of a factual assertion. No answer
# labels, bridges, or model calls are used to decide which sentences to remove.
_SOURCE = r"(?:(?:the|this|these|my|our|provided|given|available)\s+)*(?:documents?|texts?|material|reports?|information|context|passages?)"
_ABSENCE = [re.compile(p, re.IGNORECASE) for p in (
    rf"{_SOURCE}\s+(?:does|do)\s+not\s+(?:provide|contain|mention|state|include|specify|identify|give)\b.+",
    r"(?:I|we)\s+(?:cannot|can't|can’t|am unable to|are unable to)\s+(?:determine|answer|identify|establish|infer|conclude)\b.+",
    r"(?:there is\s+)?no\s+(?:information|evidence|mention|details?)\b.+\b(?:provided|given|available|included|mentioned|documents?|text|material|context)\b.*",
    rf"{_SOURCE}\s+(?:is|are)\s+(?:not provided|not available|insufficient|missing)\b.*",
)]
_MIXED = re.compile(r"\b(?:and|but|however|although|while|because|since|which|whereas|yet)\b|;", re.IGNORECASE)
_SENTENCE = re.compile(r".+?(?:[.!?](?=\s|$)|\n|$)", re.DOTALL)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                    separators=(",", ":")).encode()).hexdigest()


def question_digest(q):
    return digest(q)


def strip_abstentions(report):
    """Return edited text and exact deletion spans; unmatched text is byte-preserved.

    This is a narrow wording intervention, not a general fact-preserving semantic
    editor. Inspect the logged deletions and report how often the rule fired.
    """
    edits = []
    for match in _SENTENCE.finditer(report):
        sentence = match.group().strip().lstrip("-*# ").strip()
        sentence = sentence.rstrip(".!?").strip()
        if not _MIXED.search(sentence) and any(p.fullmatch(sentence) for p in _ABSENCE):
            edits.append({"start": match.start(), "end": match.end(), "text": match.group(),
                          "reason": "source_scoped_abstention"})
    out, cursor = [], 0
    for edit in edits:
        out.append(report[cursor:edit["start"]])
        cursor = edit["end"]
    out.append(report[cursor:])
    return "".join(out), edits


def gold_paragraphs(q):
    """Select annotated evidence without reordering it into reasoning-chain order."""
    refs = {(g["worker"], g["index"]) for g in q["gold"]}
    if not refs or any(w < 0 or w >= len(q["workers"]) or i < 0 or i >= len(q["workers"][w])
                       for w, i in refs):
        raise ValueError(f"Invalid gold paragraph references: {q['id']}")
    return [[p for i, p in enumerate(worker) if (w, i) in refs]
            for w, worker in enumerate(q["workers"])]


def evidence_blocks(q, reports, condition):
    """Return exact prompt segments and an audit of the intervention."""
    if condition not in CONDITIONS:
        raise ValueError(f"Unknown text condition: {condition}")
    if len(reports) != len(q["workers"]) or not all(isinstance(r, str) for r in reports):
        raise ValueError(f"Need one report per worker: {q['id']}")
    audit = {"condition": condition, "report_edits": [], "gold_unit": None}
    if condition == "joint":
        return [prompts.block(w) for w in q["workers"] if w], audit
    if condition in ("gold_reports", "gold_joint"):
        evidence = gold_paragraphs(q)
        audit.update(gold_unit="paragraph", gold_paragraphs=evidence)
        if condition == "gold_joint":
            return [prompts.block([p for w in evidence for p in w])], audit
        reports = ["\n\n".join(w) for w in evidence]
    elif condition == "cleaned":
        cleaned = [strip_abstentions(r) for r in reports]
        reports = [r for r, _ in cleaned]
        audit["report_edits"] = [edits for _, edits in cleaned]
    audit["presented_reports"] = reports
    return [prompts.report_block(w, report) for w, report in enumerate(reports)], audit


def generate_reports(model, q, report_tokens):
    """Exactly T1's worker prompt and greedy generation; no orchestrator call."""
    reports, tokens, truncated = [], [], []
    for worker in q["workers"]:
        g = model.greedy(model.ids(prompts.report_prompt(model.fmt, q["question"], worker)),
                         max_new=report_tokens)
        reports.append(g.text.strip())
        tokens.append(g.tokens)
        truncated.append(g.cut)
    return {"reports": reports, "worker_tokens": tokens, "truncated": truncated}


def validate_report(row, q, model_name, dtype, report_tokens):
    """Accept only same-model, same-budget T1 worker outputs for this question."""
    if row.get("id") != q["id"] or row.get("model") != model_name or row.get("dtype") != dtype:
        raise ValueError(f"Report identity/model/dtype mismatch: {q['id']}")
    if row.get("report_tokens", row.get("options", {}).get("report_tokens")) != report_tokens:
        raise ValueError(f"Report budget mismatch: {q['id']}")
    reports = row.get("reports")
    if not isinstance(reports, list) or len(reports) != len(q["workers"]) or not all(
            isinstance(r, str) for r in reports):
        raise ValueError(f"Malformed worker reports: {q['id']}")
    for key in ("worker_tokens", "truncated"):
        if not isinstance(row.get(key), list) or len(row[key]) != len(reports):
            raise ValueError(f"Missing worker audit {key}: {q['id']}")
    if "question_sha256" in row and row["question_sha256"] != question_digest(q):
        raise ValueError(f"Question changed since report generation: {q['id']}")


def final_output_flags(text, thinking):
    """Never regard an unfinished reasoning trace as a final answer."""
    closed = "</think>" in text
    final = text.rsplit("</think>", 1)[-1] if closed else text
    has_final = bool(final.strip()) and (thinking != "forced" or closed)
    start = final.find("\\boxed{")
    depth, boxed = 0, False
    if start >= 0:
        for char in final[start + len("\\boxed{"):]:
            if char == "{":
                depth += 1
            elif char == "}":
                if depth == 0:
                    boxed = True
                    break
                depth -= 1
    return {"final_answer_present": has_final, "final_boxed": has_final and boxed}


def evaluate(model, q, report_row, condition, **answer_options):
    from .arms import answer

    blocks, audit = evidence_blocks(q, report_row["reports"], condition)
    block_ids = [model.ids(block) for block in blocks]
    ids = model.ids(prompts.prefix(model.fmt, q["question"])) + [t for b in block_ids for t in b]
    cache, _ = model.prefill(ids)
    pre = COMPOSE if condition == "compose_prompt" else ""
    out = answer(model, cache, len(ids), pre=pre, **answer_options)
    return {**out, **audit, **final_output_flags(out["text"], answer_options.get("thinking", "off")),
            "input_tokens": len(ids) + len(model.ids(prompts.tail(model.fmt, answer_options.get("thinking", "off"), pre))),
            "worker_tokens": [len(b) for b in block_ids],
            "source_report_tokens": report_row["worker_tokens"],
            "truncated": report_row["truncated"] if condition in ("reports", "cleaned", "compose_prompt") else [],
            "reports_sha256": digest(report_row["reports"]),
            "question_sha256": question_digest(q), "timing_scope": "orchestrator_only"}
