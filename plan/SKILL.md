---
name: modelwiki-plan
description: >-
  Plan the optimization of one model on one served tree commit. Read the
  model page, its buckets, the serving skill, the tree at that commit, and a
  ruppell profile of that commit; audit every row, map the profile to buckets,
  say how each row changes the step, and let the CLI project served metrics at
  the workload and order the work.
  Use when `modelwiki plan` has prepared a plan directory whose features.json
  is unfilled.
---

# Plan a served tree commit

`modelwiki plan <model> --tree <tree> --commit <commit> --profile <ruppell run> --steps A:B --workload <name> --bench-file <bench.json> --out <folder>` makes a plan directory from scratch (temporary; `--work <dir>` keeps it), runs you on it, then checks, projects and writes the handoff folder `<folder>`. It holds `request.yaml` (tree, commit, the ruppell profile, workload, bench, launch), `brief.md` (the model's buckets and rows), the profile tables (below), `features.json` (one entry per row, unfilled), and `bucket_map.yaml` (empty).

Three inputs set the projection:

- **The profile:** a ruppell run of the served tree at the same commit, under the workload (the `profiler` skill takes it and runs `ruppell check`).
- **The workload:** concurrency, ISL, OSL, prefix-hit rate, speculative drafts and acceptance length, the metric to rank by, and the requirements (`not_worse`, or a bound). It is the load the profile ran. Give it as flags, as a YAML (`--workload <path>`; `--workload agentx-c32` reads the sample in `examples/`), or both, the flags overriding:

  ```bash
  modelwiki plan ... --concurrency 32 --isl 105314 --osl 472 \
      --prefix-hit 0.946 --drafts 5 --acceptance 3.51 --require p90_interactivity=not_worse
  modelwiki plan ... --workload agentx-c32 --prefix-hit 0.9
  ```

  `--active-decodes` is the requests decoding at once, which agentic loads push above the client's concurrency: a number or a range (`39-46`); left out, it is the profile's p10-p90. ISL and OSL are means, in tokens. `--set key=value` sets any other field (`speculative.drafts=3`). Concurrency, ISL, OSL and prefix hits are required. `--target` adds a load to project at: a YAML, or changes to this one (`--target concurrency=64,osl=800`).
- **The bench** of the profiled run (`--bench-file`, or `--bench key=value`): its measured throughput, interactivity and TTFT calibrate the step model.

`modelwiki plan` queries the profile once over the step window and rank:

- `kernels.csv`: every kernel by step kind, module scope (layer index folded to `layers.*`), op, category and name, with its time and launches weighted over all steps of its kind, and the bytes, FLOPs and link bytes its op calls moved, read from their recorded shapes (`shaped_ms` is the time that has shapes; composite ops have none).
- `steps.csv`: each kind's step count and host time.
- `step_shapes.csv`: every step's host time and shape: prefill requests, tokens, context and attention work; decode requests, rows and context.
- `rates.yaml`: what the machine that served the workload reached in this profile: bandwidth, compute per dtype and link rate (p90, time-weighted, over its large calls; over all its calls, a low bound, when it has no large one), the shortest kernels' time per step kind (the launch floor), and the collective world size. Floors price against these, not a datasheet; a dtype the profile never ran has no compute floor.

Read before the tree, from the paths in `brief.md`:

- The model page, `models/<family>/<model>/model.md`: its feature rows, with their IDs.
- Its `serving.md`: the model's serving rows.
- Its `buckets.md`: the blocks of the step and the rows each maps to.
- `shared/how-to-work.md` (statuses, the launch check, pricing) and `shared/profile-buckets.md` (how to derive the map).

You fill `bucket_map.yaml` and `features.json`. The CLI does the arithmetic and the order. Work from the tree at the plan's commit and from its profile. Never search upstream projects or other trees for an implementation: a row that needs one is `port` (the code exists in this tree for another model or dtype) or `write`.

## 1. Read the tree

For each row in `brief.md`, find the module, kernel and setting that serve it in the tree at the commit. In a git tree, read files with `git -C <tree> show <commit>:<path>`; the working copy may differ.

Follow the forward path the served launch takes (`request.yaml` launch and environment), not every path in the tree: a kernel behind a flag the launch does not set is `off-by-config`.

## 2. Map the profile to buckets

From the code you read, write `bucket_map.yaml` as rules: a bucket from `buckets.md` plus regexes on `scope`, `op`, `kernel`, `category` or `kind` (omitted: any). The first match wins, so put the most specific first: a scope that does two blocks' work splits by kernel, and collectives go before every scope. Map by scope first; kernel names alone mislead, since one generic kernel serves many call sites.

```yaml
rules:
- {bucket: Communication, op: '^(vllm::all_reduce|nccl:_all_gather_base)$'}
- {bucket: Selection, scope: '\.layers\.\*\.attn\.indexer\.indexer_op$'}
- {bucket: Sparse attention, scope: '\.layers\.\*\.attn$', kernel: '_sparse_attn_|_dequantize_and_gather_k'}
- {bucket: Residual mixing, scope: '^model\.language_model\.model\.layers\.\*$'}
```

Then run:

```bash
modelwiki weigh <plan dir>
```

It writes `weights.yaml`: each bucket's share of measured host step time, overall and per step kind, with its top scopes and kernels, and its floor: the time left at the profile's rates (`headroom`, what bounds it, and the share of its time that has shapes); host gaps are host time less kernel time. Tracing adds time per kernel: when a kind's kernel time exceeds its host time, give the kind's measured GPU-busy fraction (`--busy decode=0.962`, or `ruppell.busy` in `request.yaml`), and its kernel time is scaled to host time × busy. Extend the map until `Other` is under 5% of the step; a bucket the page does not have (step inputs, say) is named `Unplaced: <what>`.

## 3. Audit every row

Set, per row in `features.json`:

| Field | What |
| --- | --- |
| `status` | `on`: served and in the profile. `off-by-config`: in the tree, not enabled by the launch. `unwired`: kernel exists, the model does not call it. `port`: exists in this tree for another model or dtype. `write`: nothing to build on. `partial`: part of the row's technique is on. `n/a`: the row does not apply to this launch. |
| `evidence` | `path:line` or `path:first-last` at the commit, for every status but `write` and `n/a`. |
| `kernels` | Regexes for the row's kernels. An `on` row's kernels must appear in `kernels.csv`; if one does not, the row is `partial` or `off-by-config`. |
| `buckets` | Prefilled from `buckets.md`. Change only when the tree computes the row elsewhere. |
| `weight` | `{share, basis}` only for a row no bucket measures, such as an overlay of mixed or prefill steps. |
| `change` | How the row changes the step: `kind`, plus `fraction`, `basis`, `kernels`, `step_kinds` and `why` as below. |
| `depends_on` | Rows that must land first (a fused kernel that needs a layout another row sets). |
| `launch_blocker` | The server cannot run the bench without it. |
| `config` | The flags or environment the row sets when it is a setting. |
| `notes` | For `write`: what you searched and where. Anything the next reader needs. |

`change.kind` picks how the CLI computes the time the row removes, per step kind. `kernels` (regexes) narrows the row's buckets to the kernels it changes, `scope` to a scope regex, and `step_kinds` to the kinds it touches (default: all):

| Kind | Removes | Fraction |
| --- | --- | --- |
| `optimize` | Each matched kernel down to its floor. | computed |
| `fuse` | The matched kernels become one at the largest floor among them; the launches it drops take their share of host gaps. | computed |
| `remove_work` | `fraction` of the matched kernels' time and launches (a gather or a cast the row no longer needs). | given |
| `overlap` | `fraction` of the matched kernels' time, at most the rest of the step's kernel time. | given |
| `host` | `fraction` of the step kinds' host gaps (graphs, compilation, scheduling). | given |
| `estimate` | `fraction` of the row's bucket (or `weight`) time, with `basis`: `reference` (the block measured on another engine or platform), `budget` (a kernel or byte budget), or `estimate`. | given |

Prefer `optimize` and `fuse` where the matched kernels have shapes: their numbers come from the profile, not from you. Composite ops (attention modules, the indexer) have no shapes and no floor: use `remove_work` for the work the row drops, or `estimate`. `why` says in one sentence what goes, with its numbers. An `on` row whose kernels sit far from their floor gets `optimize` too: it goes to the optimize round.

## 4. Check

```bash
modelwiki check <plan dir>
```

Fix every error; a warning needs a reason in the row's notes. Check warns on a computed change that matches no kernel or no shapes, on an estimate that claims more than its buckets' floor headroom, and on a workload whose active decodes do not hold the profile's median.

Stop there: `modelwiki plan` (or, with `--work`, `modelwiki project <plan dir>`) writes `plan.json`, `plan.md` and the handoff folder.

How the CLI projects: it fits each step kind's host time to its shape over the profile's own steps (untraced steps where there are enough), takes the tracing overhead per kernel out, removes each row's time from the steps of the kinds it touches, and replays the step sequence. Throughput is the tokens the steps decode (decode requests × acceptance length) over their time; interactivity is a request's tokens over the steps it spans (OSL ÷ acceptance length of them), at p50 and p90; TTFT scales with the prefill steps' time. Each metric is then scaled by measured over modeled at the profile, so "now" matches the bench. A target workload rescales each step's decode rows and prefill work through the fit, at its active decodes (a range is projected at both ends and the middle, and reported as a range; left out, the profile's range scaled by concurrency), and names every shape pushed past the profiled range: such a number is a guess.

The scheduler settings get their own projection, from the same profile: the running-sequence cap (requests decoding at once) and the step token budget (the prefill chunk plus the decode rows). With the cap kept full, a request finishes every OSL ÷ acceptance steps and brings its prompt steps, so the mean step is the decode step plus the share of steps that carry a prefill times what a prompt step adds; interactivity is the acceptance length over it, throughput the cap times that per GPU, both scaled so the profiled point gives the bench. Inputs, all read from the profile:

- the decode step at another row count: the decode fit when the profile varies rows; otherwise a range, from flat to all time above the launch floor (launches a step × the kernel floor, plus host gaps) scaling with rows;
- the prompt step: the fit of mixed (or prefill) steps, with its decode rows at the cap;
- each prompt's new tokens and context, rebuilt from the prompt steps (a step that fills the budget continues into the next), re-chunked at each budget through the fit, which sets the prompt steps per request and their time;
- the clients as a closed loop: the workload's concurrency, with the time between a reply and the next request fitted at the profile (none when the profile decodes at least that many requests, so the waits are an upper bound). Requests past the cap wait, and TTFT counts the wait.

It projects every cap and budget now and with the plan's cut on decode and prompt steps, picks the one that meets the requirements (else the fewest failed, then the metric), and gives the mean step the requirements need, and the most kernels a decode step can launch at the launch floor to reach it. A budget above the profiled one was not served: its buffers come out of the KV pool.

`plan.md` is the order of the work: the changes projecting 2% or more in order (features, optimize and serving rows together), then those under 2% to build and serve together at the end, each with what it changes and where. `projection.md` has the numbers behind it: the metrics now and planned at the workload (and at each target), the step model's fit, the full scheduler sweep, the buckets with their floors, every row's audit, and per round the rows in order: launch blockers, then dependencies, then the gain in the workload's metric (rows that break a requirement last), then page order, with rows under 2% gain bundled.

The handoff folder is what the agent that works the plan reads: `plan.md`, `features.json`, `workload.yaml`, copies of `model.md`, `serving.md` and `how-to-work.md`, and a `SKILL.md` with the commit to work from, the launch and its baseline, the order of the work, and how to read the numbers.
