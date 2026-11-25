---
name: modelwiki
description: >-
  Optimize a model's serving with ModelWiki: find the model's files, plan one
  served tree commit from its profile, then work the plan. Use when asked to
  plan, audit, or optimize a model on a served tree.
---

# ModelWiki

## Layout

```
models/<family>/<model>/
  model.md       the features the model needs, by component; each row has an ID
  serving.md     its serving: topology, scheduler, graphs, caches, kernel selection, overlap
  buckets.md     the blocks its profile is grouped by, and the rows each maps to
shared/
  how-to-work.md      the procedure, any model
  profile-buckets.md  how to map a profile to buckets, any model
plan/                 the plan agent's steps (SKILL.md) and the modelwiki CLI
examples/<name>.yaml  a sample workload (concurrency, ISL, OSL, prefix hits, drafts, metric, requirements)
```

`modelwiki plan ... --out <folder>` writes the handoff `<folder>`: the plan, `model.md`, `serving.md`, `how-to-work.md`
and a `SKILL.md` for the agent that works it (the commit, the launch, the order of the work, how to read the numbers).
Each run starts from scratch; the working files are temporary (`--work <dir>` keeps them).

## Find the model

1. Name it from the served checkpoint's `config.json`: `architectures` gives the family folder (`DeepseekV41ForCausalLM`
   is `dsv4`), and the checkpoint's name gives the model folder (DeepSeek-V4.1-Flash is `dsv41-flash`).
2. `modelwiki rows <model>` finds `models/*/<model>/` and prints its feature rows, serving rows, buckets, and the
   paths of all three files.
3. Not there: copy its closest sibling in the family (or the closest model of another family) and edit it from the
   config, as `shared/how-to-work.md` says under "A new model".

## The flow

1. **Plan.** `modelwiki plan <model> --tree <tree> --commit <commit> --profile <ruppell run> --steps A:B
   --bench-file <bench.json> --out <folder>` with the workload as flags (`--concurrency 32 --isl ... --osl ...`) or a YAML
   (`--workload <path>`; `examples/` has a sample). The agent follows `plan/SKILL.md`: it audits every row of `model.md`
   and `serving.md` against the tree at that commit, maps the profile to the blocks in `buckets.md`, and says how
   each row changes the step. The CLI weighs, checks and projects into `<folder>/plan.md`: each row's
   status, share of step time, the throughput, interactivity and TTFT it gives at the workload, config, and the order.
2. **Features.** Work the plan's features in its order: implement each row, check it against the separate-kernel
   reference in `eval/`, tune it.
3. **Optimize.** Drive the top kernels toward their floor, the most headroom first.
4. **Serving.** Sweep `serving.md`'s rows on the optimized tree.

Each stage's detail, and the rules every row follows, are in `shared/how-to-work.md`.
