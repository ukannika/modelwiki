# ModelWiki

Plans the optimization of one model on one served tree commit. Each model has skill pages (`model.md`,
`serving.md`, `buckets.md`) that list the features it needs. `modelwiki plan` audits every feature against the tree,
weighs it with a ruppell profile of that commit, projects what it gives at a workload, and orders the work. The
same command writes the result into a folder for the agent that does the optimizing.

## Layout

```
models/<family>/<model>/   model.md (feature rows), serving.md (serving rows), buckets.md (profile blocks)
shared/                    how-to-work.md (the procedure), profile-buckets.md (mapping a profile to buckets)
plan/                      the CLI (plan/modelwiki), the plan agent's steps (plan/SKILL.md), tests
examples/<name>.yaml       sample workloads
```

## Install

```bash
pip install -e plan          # puts `modelwiki` on PATH
# or, without installing:
cd plan && PYTHONPATH=. python3 -m modelwiki --help
```

Python 3.10+, with `pyyaml` and `numpy`. Querying the profile needs ruppell (`--python` gives its PYTHONPATH).
Set `MODELWIKI_ROOT` to use model pages from another checkout.

## What you need

| Input | Flag | What |
| --- | --- | --- |
| The model | `<model>` | A folder under `models/`. `modelwiki rows <model>` lists its rows and buckets. |
| The served tree | `--tree`, `--commit` | The engine source the model was served from, and the commit to plan. Evidence is read at that commit. |
| A profile | `--profile`, `--steps`, `--rank` | A ruppell run of that commit under the workload, and its steady-state step window. |
| The workload | `--workload` and/or flags | Concurrency, ISL, OSL, prefix-hit rate, speculative drafts and acceptance, the metric, the requirements. |
| The bench | `--bench-file` or `--bench k=v` | The profiled run's measured throughput, interactivity and TTFT; they calibrate the projection. |
| A plan agent | `--agent` | `claude` or `cursor-agent` on PATH (auto-detected), `--agent-cmd`, or `none` to fill the plan yourself. |

## Generate a plan

```bash
modelwiki plan <model> \
    --tree <served tree> --commit <commit> \
    --profile <ruppell run> --steps A:B --rank 0 --busy decode=<GPU busy fraction> \
    --workload examples/<workload>.yaml \
    --bench-file <bench.json> \
    --out <folder>
```

Every run starts from scratch and writes one thing, the handoff folder `<folder>` for the optimizing agent. The
working files go to a temporary directory that is removed when the run ends. The command:

1. Prepares the temporary plan directory: `request.yaml`, `brief.md`, the profile tables (`kernels.csv`, `steps.csv`,
   `step_shapes.csv`, `rates.yaml`), and an unfilled `features.json` and `bucket_map.yaml`.
2. Runs the plan agent on it, which follows `plan/SKILL.md`. The agent maps the profile to buckets, then audits every
   row against the tree: status, evidence, kernels, and how the row changes the step.
3. Weighs the buckets, checks the audit, and projects the result into `plan.md`, `projection.md` and `plan.json`.
4. Writes the handoff folder `<folder>` (below), replacing what a previous plan left there.

`--work <dir>` keeps the working files in `<dir>`, for debugging the audit or filling it by hand; a run with the same
`--work` keeps the rows already filled.

### The workload as flags

Flags set the workload, or override fields of `--workload`:

```bash
modelwiki plan <model> ... \
    --concurrency 32 --isl 105314 --osl 472 --prefix-hit 0.946 \
    --drafts 5 --acceptance 3.51 \
    --require output_tok_s_gpu=383.4 --require p90_ttft_s=2.39 --require p90_interactivity=not_worse \
    --target concurrency=64
```

- `--active-decodes N|LO-HI`: requests decoding at once. Leave it out to take the profile's p10–p90.
- `--require KEY=VALUE`: a bound (a minimum, or a maximum for `*_s`), or `not_worse` (no lower than now).
- `--target`: another load to project at. Give a YAML, or changes to this load such as `concurrency=64,osl=800`.
- `--set KEY=VALUE`: any other workload field, dotted (`speculative.drafts=3`).

### Without an agent

`--agent none` needs `--work <dir>`: `modelwiki plan` stops after preparing `<dir>`. Fill `bucket_map.yaml` and
`features.json` by following `plan/SKILL.md`, then run the same command again, or these one at a time:

```bash
modelwiki weigh   <dir>   # bucket shares and floors -> weights.yaml
modelwiki check   <dir>   # verify features.json against the tree and the profile
modelwiki project <dir>   # projections and order -> plan.md, projection.md, plan.json, and the handoff folder
```

## The handoff folder

`<folder>` holds only what the optimizing agent needs; point the agent at its `SKILL.md`:

```
SKILL.md          start here: the commit to work from, the launch, the order of work, how to read the numbers
plan.md           the rows projecting 2% or more, in order; then the rows under 2%, to build and bench together at the end
features.json     every row's evidence, kernels and projected change
workload.yaml     the workload the plan projects at, to plan again
how-to-work.md    the procedure
model/model.md    the model's feature rows
model/serving.md  its serving rows
```

After a change that cuts 20% or more of the step, profile the kept tree and plan again into a new folder, with
`--workload <folder>/workload.yaml`.

## Tests

```bash
cd plan && python3 -m pytest -q tests
```
