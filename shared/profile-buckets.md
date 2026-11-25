# Profile buckets: where step time goes, by block

A bucket is one block of the model (a page's feature table, or a part of one). The model's `buckets.md`, beside its
page, says what each block computes, in the model's terms, and the rows it maps to. It does not name the engine's
modules or kernels: those differ by engine, backend and version, so you derive them, from the profile and the served
tree's code, and write the map you derived. Every bucket's time is the block as it runs now, fused or not, so a
missing feature is weighed by the time of what it would replace.

The map you derive goes where the work is. In a plan ([plan/SKILL.md](../plan/SKILL.md)) it is `bucket_map.yaml`,
rules on scope, op and kernel that `modelwiki weigh` applies to the profile's tables. In a campaign it is the
dossier's own `buckets.md`, beside its parts, and follows the dossier's rules: every number with its source, every
claim with its evidence.

## Read

- The profile of the served tree: part 7 and its report, `<session>/profiles/<name>/PROFILE.md`, with birdeye's
  tables over the measured window (`module_by_step_kind.csv`, `kernel_by_step_kind.csv`,
  `category_by_step_kind.csv`, `step_coverage.csv`).
- Part 8: which kernel serves each op, and the module it runs under.
- In a plan, the same profile as `kernels.csv` (every kernel by step kind, folded scope, op and name, weighted over
  all steps of its kind) and `steps.csv` (each kind's steps and host time), which `modelwiki plan` queried.
- The model's `buckets.md`: the blocks, and the rows each maps to.
- The served tree's model code: the module classes the profile's paths name, and what each one's forward computes.

On a sampled run, rank only from the `_by_step_kind` tables, which weigh each step kind by every step of that kind
(the profiler skill). For a grouping the tables do not have, query birdeye over the same window with `--scope GLOB`
(one bucket's scopes, as you derived them) rather than adding rows by hand.

## Derive the map

The map is the engine's names for each block: the module scopes, and the kernels no scope covers. Build it from
this run, not from names you expect; a name that looks right is a lead, and the code is the evidence.

1. **Module scopes.** List the module paths in `module_by_step_kind.csv`, the layer index folded (`*` for the
   layer). For each, open the class that defines it in the served tree and read what its forward computes; give it
   the bucket whose block that is. A module that does two blocks' work (a fused projection inside the indexer, say)
   gives its own kernels to one and its children to theirs: the most specific scope wins.
2. **Kernels outside a scope,** or under one that does two blocks' work: find the op that launches each, from part 8,
   or from the op or Python stack the trace records for it, or by finding where the served tree or its kernel
   library calls it. Give it the block of that op. Collectives (category `collective`) are Communication.
3. **The draft and the layers.** Find where the engine roots the draft and how it numbers the draft's layers, from
   the model code and the profile's paths, before splitting them from the target's. The page's layer ranges (an
   overlay's, say) are the model's layers; write the scope each one is in this engine.
4. **Check the map.** Every module path and every kernel over 0.5% of step time is in it, each with its evidence: the
   class or call site (file and line), or part 8's row. One that you cannot place is Other, not a guess.

## Assign

1. **Each module row to one bucket,** by the map, the most specific scope first.
2. **Each kernel with no module,** or under a scope that does two blocks' work, by the map.
3. **Host gaps:** the step period less the GPU's busy time, per step kind. It is a bucket of its own.
4. **Whatever is left is Other,** listed by kernel, the largest first. Over 2% of step time, it is a direction of its
   own (`m-kernel-<name>`).

The buckets, host gaps and Other add to the measured step time within the residual part 7 states. Say the residual.
An overlay bucket (one the page marks as counted apart, such as the prefill rows of some layers) is shown beside the
table, not in its sum.

## Weigh

Per bucket, per step kind (decode, mixed, prefill): its time per step and its share of step time. Then its
**weight**: its share of the bench's total step time, each kind weighted by how many of its steps the bench runs
(`step_coverage.csv`). A kind the trace does not cover gets no share, and the table says so.

Per row: the weight of its bucket. A bucket with several rows splits its weight by the kernels each row owns where
the trace tells them apart (the three cache stores, say), and gives the whole weight to each row, marked shared,
where it does not.

## Write the result

In a plan, `modelwiki weigh` writes `weights.yaml` from the rules, and `plan.md` carries the bucket table. In a
campaign, write the dossier's `buckets.md`. The top: the profile it reads (the report's path, the window, the tree's commit), the engine and its version, the
step kinds covered, and the residual.

Then **Map:** the map you derived, one line per scope or kernel pattern.

| Bucket | Scope or kernel | Evidence |
| --- | --- | --- |

| Bucket | Decode share | Mixed share | Prefill share | Weight | Top kernels | Rows |
| --- | --- | --- | --- | --- | --- | --- |

- **Shares** are percent of that step kind's time; **Weight** is percent of the bench's step time.
- **Top kernels:** the bucket's three largest, with their share.
- **Rows:** the IDs it maps to, each with its weight, and `shared` where the trace does not split them.

Sorted by weight, the largest first. Then **Overlays**, **Other** (by kernel), and **Unplaced**: module rows and
kernels with no block on the page, with what each computes, so the page can learn a block it lacks.

Then **Row order:** every row of the page's feature sections with its weight, the largest first. Ties go to the
page's order. The audit and the directions order the feature round by it.
