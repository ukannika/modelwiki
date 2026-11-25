# How to work: any model

The model page's tables are the work; Arbor's map is how it gets done. First the requested features, then
exploration. The dossier carries each stage's findings, each map round is one stage, and each iteration works one
row. The harness serves, benches, gates, and keeps the map: never write `directions.json` yourself.

A model page (`models/<family>/<model>/model.md`) holds **feature sections** (the model's components: its layers,
attention, experts, draft, and the fusions across them). Beside it, `buckets.md` holds its blocks for the profile, and `serving.md` its serving
tables, with the model's own numbers. Each feature row has
an ID, a Feature, a Want, a Technique and a Handoff (Shape tuning, Accuracy eval, or both). What follows
holds for every page; a page says only what is its own.

Rules for every row:

- Keep what already runs: a feature that is on in the served tree, and whose output matches the separate-kernel
  reference (one kernel per op, unfused), stays as it is.
- No kernel reads a tensor before the kernel that writes it has finished: across streams, inside a fusion, and in
  any reordering.

A plan ([plan/SKILL.md](../plan/SKILL.md)) does the Buckets and Audit stages ahead of a campaign, for one tree commit
and its profile, and writes each row's status, weight, projected gain and the order in the `plan.md` of its handoff
folder. A campaign that has a plan for its tree starts from it.

| Stage | Who | Chapter | Does | Leaves |
| --- | --- | --- | --- | --- |
| Launch check | master | `map.measure`, `dossier:measure` | Boot the launch, baseline, trace a short run. | Part 6: the drops and the noise floor. Part 8: the kernels that run, by module |
| Buckets | master | `map.directions`: `model-wiki:buckets`, before the audit | The profile by block: each block's share of step time, mapped to the page's rows. No GPU. | `buckets.md`: each row's weight, the Row order |
| Audit | master | `map.directions`: `model-wiki:audit`, before the directions | Every requested row against the served tree and part 8's trace, with its weight. No GPU. | `features.md`: one line per row, the Round 1 order |
| Features | worker | `iterate`: map round 1 | One direction per requested row that is not on and running, one row per iteration: implement it, check it, tune it. | Each row done or killed |
| Summary | master | `round_end`: `model-wiki:summary`, before the update round | Each row's outcome, and what the update round does with it. | `features-summary.md` |
| Optimize | master, then worker | `round_end` update, then `iterate`: map round 2 | Price the top kernels against their roofline; work the most headroom first. | 07.uK: each top kernel's bound and efficiency. Each row closed |
| Serving | master, then worker | `round_end` update, then `iterate`: map round 3 | Sweep the model's serving.md tables on the optimized tree. | The kept settings |

The requested rows are the program's REQUESTED FEATURES directive; without one, every row of the page's feature
sections. Map round 1 holds only those rows, in the order of their blocks' weight in the profile. Everything else the
directions rank waits under "Beyond this map's length" for the update round, which maps the first stage whose exit
has not passed.

## Map rows

One direction per row: the ID from the page's ID column (`m-<table>-<feature>`, lowercase with hyphens); the row's
title, `depends_on` from the audit, and `rank` in the stage's order. A one-change row is an `increment`. A row whose
correct version is a new or rebuilt kernel is a `rewrite` with its kernel plan. A top kernel that no row owns is
`m-kernel-<name>` (optimize). Tier: `feature` in the feature round, `kernel` in optimize, `config` in serving.

Each iteration cites its row's ID in its DONE line and declares its INTENT. Close a row with
`DIRECTION: {"id": ..., "status": ..., "why": ...}`; the why is the evidence: iteration, number, trace.

## Launch check (`map.measure`)

No code changes. Confirm what runs before the audit reads it.

1. Boot the launch. When it fails on a setting, drop only that setting and record it in part 6 with its row.
2. Baseline as part 6 says; the spread is the noise floor.
3. Trace a short run and read its kernels by module, in part 8. The audit reads this trace: a row on in code is on
   only when its kernels run under its module.

Exit: the launch boots, and part 8 names the kernels each module runs.

## Buckets (`model-wiki:buckets`)

No GPU, no code changes. Group the profile's time by the blocks of the model's `buckets.md`, as
[profile-buckets.md](profile-buckets.md) does: the engine's module scopes and kernels mapped to those blocks from the
served tree's code and part 8, not from names on the page; every module scope and kernel to one bucket, each bucket's share of
decode, mixed and prefill step time, weighted by how often the bench runs each kind, and each bucket's weight given
to the rows it maps to. A missing feature is weighed by the time of the block it would replace.

Exit: the buckets, host gaps and Other add to the measured step time within part 7's residual, and every feature row
has a weight.

## Audit (`model-wiki:audit`)

No GPU, no code changes. Every row of the page's feature sections against the served tree and part 8's trace, in
`features.md`: status (on, off by config, unwired, port, write, partial, not applicable), whether its kernels run, the
evidence (file and line, and the source to port from), the rows it depends on, whether the launch needs it to boot,
its weight from `buckets.md`, its Handoff, and its Map column. Code that is gated off or never called is not on.

Exit: every requested row has a status, a weight and a Map column, and `features.md` ends with the Round 1 order.

## Features (map round 1)

The map: every requested row the audit gave `round 1`, and every dropped setting, in the audit's Round 1 order:
launch blockers first, then rows whose dependencies are on, then by weight, the largest first; ties by the page's order.

1. Implement: turn it on, call the existing kernel, port it, or write a correct version.
2. Check the component against the separate-kernel reference at the model's shapes, in `eval/`. A row whose Handoff
   says Accuracy eval proves its numerics there too whenever the campaign's bench cannot read the model's output
   (a bench with synthetic acceptance, say).
3. Tune it at the served shapes along its Handoff's axis (tiles, token cutoffs, the variant), in the same row, until
   the best existing variant is live. Driving a kernel to its bound is Optimize's.
4. Declare `INTENT: FIX` with the row as evidence: neutral is a KEEP. A row whose absence gives wrong output is
   `FIX-CORRECTNESS`. A row whose correct version lands slower than the noise floor is a rewrite: build it off by
   default under FIX, and turn it on as a WIN once it pays.
5. Close it done once its setting is live in a KEEP; killed with what was tried and what would unblock it.

**A row under 5 % of the run is built off the bench,** with the pack's `subfloor_bundle_pct` set. Its weight is its
block's time over the step time now, not at the baseline: rescale `buckets.md`'s weight by the kept tree's speedup
since that profile, keeping the block's own time unless a KEEP changed it.

1. Implement, check and tune it as steps 1 to 3 do, in `eval/` against the kernel or the component at the served
   shapes. No server.
2. Save the change as `eval/bundle/<id>.patch` against the kept tree (one patch per editable tree it changes, named
   for the tree), and leave the served trees as the kept tree has them. Declare `INTENT: LEARN` with the evidence in the ledger: a PROBE, with no server and no bench.
3. Close it `below_floor` with
   `SUBFLOOR: {"id": "<id>", "speed_pct": <its weight × the share of the block's time the eval removes>, "ships": "eval/bundle/<id>.patch"}`.
4. When the program says BUNDLE DUE, apply every open patch to the kept tree (rebase one a KEEP has moved), check
   them together in `eval/`, serve them as one change under `INTENT: FIX`, and name them in `SUBFLOOR_SERVED`. A
   bundle that loses: split it in halves over the next iterations to find the row.

Served alone, whatever its weight: a launch blocker, a `FIX-CORRECTNESS` row, a row a served row depends on, and a
row whose check needs the live engine (graphs, the scheduler, the KV cache manager).

Exit: every requested row is on, killed, or served in a bundle, and the launch runs with nothing dropped that a row
brought back. A bundle still open when the round's last row closes is served then, due or not. The summary then
writes `features-summary.md` for the update round.

## Optimize (map round 2)

In the update round, read `features-summary.md`, trace the full bench on the feature tree, read its decode and mixed
prefill steps, and confirm each row brought up shows its kernels under its module; one that does not goes back on
the map. Bucket the new trace as before ([profile-buckets.md](profile-buckets.md)) and rewrite `buckets.md`: the
feature round moved time between blocks.

**Pick the top kernels,** block by block. Take buckets by weight, the largest first, and inside each its kernels by
their share of step time, decode and mixed prefill steps apart, each weighted by how often its step runs in the
bench. Take kernels until they cover 80% of step time, at most 8, none under 2%. Each taken kernel maps to the row its
bucket names; a top kernel in Other (a norm, a collective, a copy no row owns) gets its own direction,
`m-kernel-<name>`. A row with no top kernel waits.

**Price each against its roofline,** in part 7:

- Share: its time over step time.
- Work: the bytes it must move and the FLOPs it must do at the served shapes: the rows a step really runs, and for an
  expert GEMM the distinct experts the step's routing touches, not all of them. Bytes that padding or layout add are
  counted apart, as a lever of their own.
- Bound: the larger of bytes ÷ measured read bandwidth and FLOPs ÷ measured matrix rate for its dtypes. A kernel of a
  few microseconds is bound by its launch inside the graph. Measure each limit on this node at the kernel's sizes,
  never from the spec sheet.
- Efficiency: bound ÷ its time inside the captured graph.
- Headroom: share × (1 − efficiency).

Confirm the limit before building: vary the quantity the bound names (the bytes read, or the rows) with the rest
fixed; the time must follow it. A kernel whose time does not is bound by something else (tiles, dispatch, a wait),
and part 7 says what.

The map: the taken kernels' rows, the most headroom first, its headroom as `ceiling`. Each iteration:

1. Search offline at the served shapes along the Handoff's axis: tiles, fusion, dtype, or another kernel. Correctness
   first. Record the efficiency before and after, and project the server gain from the share.
2. Declare `INTENT: WIN`; the gate keeps it only when it clears the noise floor. A step of a kernel plan that serves
   nothing yet is a FIX, citing its step.
3. After a KEEP, trace again and rank again: the top kernels move. One that leaves the top closes below_floor; one
   that enters is mapped.

Close each row done when its kernels reach 90% of their bound, with the evidence: the efficiency, the bound's inputs,
and the trace. Close it below_floor (headroom under the noise floor), killed (three REVERTs with no path left), or
plan (a rewrite, its kernel plan written).

Exit: every taken kernel is at 90% of its bound or closed, the top kernels after the last KEEP are all worked, the
last KEEP's bench beats the round's starting bar on the same node, and every Accuracy eval row's check in `eval/`
still passes.

## Serving (map round 3)

On the optimized tree, one row per row of the model's `serving.md`, in order, on the served workload at its
concurrency points.
Each row's Handoff is its sweep: one `INTENT: LEARN` iteration per point, its reading in the ledger, then the best
point as a WIN.

## Measuring

- Use the rows a step really runs, after padding to the captured size, not the configured maximum.
- Time inside a captured graph. Eager timing of a short kernel measures dispatch.
- Warm every served shape before timing; kernels compile per shape. Count the compiles inside each measured window: a
  run that compiles measures the stall, not the change.
- Compare only against a baseline from the same node.
- An instrument that slows the serving path stays off.

## When a stage cannot exit

Close each row that cannot move killed, its why saying what was tried, why it cannot move (no source, a kernel or
hardware limit, accuracy, an open dependency, or budget), what would unblock it, and the ask. The update round lists
the asks at the top of part 11 with the last baseline, the profile, and the tree they ran on, and maps only rows that
can still move. A person answers there.

## A new model

`models/` holds one folder per family and one per model inside it (`models/dsv4/dsv41-flash/`). A new model in a
family starts as a copy of its closest sibling, edited from its config; a new family starts from the closest model of
another. The model's folder holds three files:

- `model.md`: the model's title and its feature sections, each row in the columns above with its ID first. Write
  on it only what this model adds: the rules above, and the flow in the top-level `SKILL.md`, stay there.
- `serving.md`: the serving tables (Topologies, Scheduler, Compilation, Graph capture, Prefix cache, KV cache, KV
  offload, Kernel selection, Compute and communication overlap), with this model's numbers in place: its experts,
  KV heads, caches by layer, dtypes, page size, memory tables, draft rows per sequence.
- `buckets.md`: the Bucket, The block and Rows table, each block by what it computes in the model's terms, and its
  rows (IDs, and serving rows by name); no engine module or kernel names, which the buckets stage derives. Overlays
  below it.

In Arbor, the pack loads `model-wiki:buckets`, `model-wiki:audit`, `model-wiki:summary` and `model-wiki:<model>`.
