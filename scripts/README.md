# T4 thinking pilot

Run `scripts/run_thinking_vm.sh` on the existing T4. It runs fast tests,
Qwen3-4B/float16 real-model tests, and then `run_thinking_pilot.py`.
The branch includes fix/think-eos so EOS during reasoning closes thinking and
continues to a final answer.

Scope: seed 0, first 50 rows per cell: toy hop1/hop2 AB/BA and MuSiQue 2hop
AB/BA/BB (350 question layouts; source questions repeat across placements).
All six arms run on those same layouts: B1, K0, T1 with thinking forced/off.
Thinking budget 1024, answer cap 256, T1 report budget 1024 per worker.
Forced arms run first. T1 enables thinking only in the orchestrator; worker
reports use the existing report prompt. No masking or composition-filter run.

The smoke pass checks one toy and one MuSiQue question in every arm, prints raw
outputs, and requires an untruncated boxed answer after the closing think tag.
Results resume under unique thinking_pilot names. The runner prints thinking
length and cap counts, writes a summary CSV, and checkpoints an archive after
each arm and on Python exit. Raw rows flush after every question.

The VM wrapper records success/failure, emits the existing Monitoring serial
marker, and schedules shutdown five minutes after exit for artifact retrieval. A separate guest shutdown is scheduled at 24h.
No new email policy is created. Artifacts remain on the boot disk on shutdown.

Paths on the VM:
- checkout: /home/elkalt/latent-dispatch-thinking
- log/status: /home/elkalt/thinking-experiment.log and .status
- bundle in checkout: results-Qwen3-4B-float16-thinking.tar.gz

The bundle contains scripts as well as results and is not directly accepted by
results.load_archive; inspect/extract its named files explicitly.

# Text-report controls for Elliott / Josh

Published in `ajit-mallav/latent-dispatch` (Ajit's existing fork), branch
`codex/text-report-controls`, based on `codex/t4-thinking-pilot` at
`9d9fffe`, including `fix/think-eos`. The existing thinking and masking runners
are unchanged. These additions were prepared by static review only: **no tests,
model calls, benchmark builds, or experiments were executed by the author**.
The new entrypoint runs its regression tests and a format/cap smoke before the
main comparison. Model validation for the chosen model/dtype remains required
as described in the root README.

Use a separate checkout if a current experiment is using the existing checkout:

```bash
git fetch https://github.com/ajit-mallav/latent-dispatch.git codex/text-report-controls
git worktree add --detach ../latent-dispatch-text-controls FETCH_HEAD
cd ../latent-dispatch-text-controls
```

Use the existing environment with the dependencies from `pyproject.toml`.
Set `TEXT_CONTROLS_PYTHON` to its Python executable, or activate it so `python`
resolves to that executable. Copy the existing verified `benchmark/toy.jsonl`
and `benchmark/musique.jsonl` into this checkout; if missing, the normal build
command is `python -m latent_dispatch.data toy musique`. The runner verifies
both files against the committed checksums before loading weights.

Elliott, Qwen3-4B on T4 (float16), both orchestrator thinking settings:

```bash
bash scripts/run_text_controls.sh --thinking off forced
```

Josh, Qwen3-8B on A100 (bfloat16), the identical comparison:

```bash
bash scripts/run_text_controls.sh --model Qwen/Qwen3-8B --dtype bfloat16 --thinking off forced
```

Each command runs the unit tests, then a smoke on one question from every
dataset/placement cell, then the pilot only if all smoke outputs have a complete
boxed final answer and do not hit the answer cap. It never provisions machines
or schedules a shutdown. Run it after the GPU is available. Add `--smoke-only`
to stop after smoke; rerun the same command without that flag to proceed.
Runtime is bounded by `--max-hours 8` by default, checked between questions.
A timeout preserves results and exits nonzero; rerunning resumes. This is a
soft boundary, not a GPU-hour estimate or a guarantee to finish in eight hours.

Reports can be reused from Elliott's existing run in another checkout:

```bash
bash scripts/run_text_controls.sh --thinking off forced \
  --report-source /path/to/thinking-checkout/results/Qwen3-4B-float16/thinking_pilot_t1_forced.jsonl
```

The source must match worker model, dtype, report budget, benchmark and worker
prompt. Multiple `--report-source` arguments are allowed; the first available
report per question wins. Without explicit sources, the runner looks in the
current model/dtype result folder for `thinking_pilot_t1_forced.jsonl`, then
`thinking_pilot_t1_off.jsonl`, then `t1_1024.jsonl`. Missing reports are generated
once, saved immediately, and shared across **every condition and thinking mode**.
`--require-saved-reports` refuses missing selected reports before loading weights.
A prior `*_report_bank.json` is also accepted as a source when using a new run
name. Historical reports do not always record the model weight revision; their
source path, commit and row hash are retained rather than claiming that it is known.

Scope and parameters:

| Setting | Default |
|---|---|
| Sources | First 20 toy_d2 hop1 + first 30 MuSiQue 2hop; selected without inspecting answers |
| Placements | AB and BA on the same sources: 100 question layouts, 50 independent source questions |
| Data seeds | 0; `--seeds 0 1 2` expands this explicitly |
| Conditions | Joint original evidence; original reports; cleaned reports; gold-paragraph reports; gold paragraphs together; original reports with a composition instruction |
| Generation | Existing greedy decoder, same settings for all conditions; no sampling seed |
| Worker reports | Thinking off, 1,024 tokens per worker, using the original T1 prompt |
| Orchestrator | CLI defaults to off; commands above request off and forced |
| Forced thinking | 1,024 tokens, existing fixed EOS handling |
| Final answer cap | 256 tokens for every condition; inspect smoke and cap rates |
| Main answer calls | 600 for off alone; 1,200 for off + forced, plus separate smoke calls |
| Worker calls | At most 200 for the default cohort; zero if all reports are reused |

This is an inference-only ablation, not training. Each invocation loads one model
once. The original-report and joint answers are recomputed under the fixed
decoder for a controlled comparison; historical worker reports are reused, not
historical answer scores. The default does not add three-hop questions or API
models. `--placements AB BA BB` adds the colocated control to both datasets.

Scientific interpretation:

- `cleaned - reports` tests a narrow wording intervention. The deterministic
  filter removes only recognizable, standalone source-scoped abstentions. It
  leaves mixed clauses unchanged and logs exact deletion spans; inspect them.
  Zero edits are a valid result, not a successful removal of all uncertainty.
- **Gold means annotated paragraphs, not exact gold sentences.** Both the toy
  and MuSiQue paragraphs can include other facts. No answer label is inserted
  as new evidence. Both gold conditions use identical paragraphs in the same
  worker/document order, including BA; only the report framing differs.
- `compose_prompt - reports` changes only the orchestrator instruction. Worker
  generation and evidence remain fixed.
- Text reports are jointly re-encoded by the orchestrator. A text-report failure
  does not by itself establish the same mechanism as isolated/stitched KV caches.
- Greedy decoding is inherited to reproduce the existing experiment. This is
  not a claim about optimal Qwen thinking settings or all reasoning budgets.

All outputs are under `results/<model>-<dtype>/<run-name>*`:
manifest with input/code hashes; immutable-per-question report bank and source
provenance; raw answers and edits; per-cell EM/F1, tokens, latency, cap and
format rates; paired source-bootstrap intervals; completion/status JSON; and
`<run-name>.tar.gz`. The archive checkpoints after each arm and on normal Python
exit/error. Raw answers flush per question; forced VM termination may leave the
archive behind the raw files. No notebook execution is required.

Summary tables distinguish completed from expected layouts. Missing final
answers count as wrong rather than being dropped. An abstention phrase count is
explicitly a heuristic. Report latency is logged separately; answer latency is
orchestrator-only. The paired table averages placements/seeds per source before
bootstrapping; its intervals are exploratory and are not a multiple-comparison
corrected confirmatory test.

Run identity is pinned. Changing the cohort, model/dtype, code, mode list or
token budgets requires a new `--run-name`; use `--report-source` to reuse the
old bank. A missing or changed bank cannot silently regenerate reports while
keeping old answers. If smoke hits the answer cap, inspect it and rerun under a
new name with the same higher `--answer-tokens` for every condition. Do not
report a partially completed pilot as the full comparison.
