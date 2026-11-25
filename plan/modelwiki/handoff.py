"""The handoff folder: everything the agent that works a plan reads, copied out of the wiki, with a SKILL.md that says
what each file holds, the order of the work, and how to read the numbers."""
import json
import shutil
from pathlib import Path

import yaml

from . import ROOT
from .page import load as load_page
from .plan import FLOOR, _num, _pct, _start_command, load_yaml

REPLAN_CUT = 0.2
PLAN_FILES = ("plan.md", "features.json")
FILES = ("SKILL.md", "plan.md", "features.json", "workload.yaml", "how-to-work.md", "model/model.md",
         "model/serving.md")


def write(pdir, out, root=ROOT):
    """Copy the plan, the model's pages and the procedure into out, and write out/SKILL.md; a file a previous plan
    left there goes first."""
    pdir, out, root = Path(pdir), Path(out), Path(root)
    plan = json.loads((pdir / "plan.json").read_text())
    req = load_yaml(pdir / "request.yaml")
    page = load_page(plan["model"], root)
    for name in FILES:
        (out / name).unlink(missing_ok=True)
    (out / "model").mkdir(parents=True, exist_ok=True)
    for name in PLAN_FILES:
        if (pdir / name).exists():
            shutil.copy2(pdir / name, out / name)
    for src in (page["path"], page["serving_path"]):
        if src and Path(src).exists():
            shutil.copy2(src, out / "model" / Path(src).name)
    shutil.copy2(root / "shared" / "how-to-work.md", out / "how-to-work.md")
    if req.get("workload"):
        (out / "workload.yaml").write_text(yaml.safe_dump(req["workload"], sort_keys=False, allow_unicode=True))
    (out / "SKILL.md").write_text(skill_md(plan, req, page, out, root))
    return out


def skill_md(plan, req, page, out, root=ROOT):
    out = Path(out).resolve()
    metric, p = plan["metric"], plan["projected"]
    wl = plan.get("workload") or {}
    reqs = wl.get("requirements") or {}
    bench = plan.get("bench") or {}
    commit, tree = plan["commit"], req["tree"]
    rows = plan["rows"]
    planned_ids = {r["id"] for r in rows}
    unplanned = [r["id"] for r in page["features"] if r.get("id") and r["id"] not in planned_ids]
    big = [r for r in rows if r.get("cut", 0) >= REPLAN_CUT]
    sc = plan.get("scheduler") or {}
    need = sc.get("need") or {}
    cli = f"cd {root / 'plan'} && PYTHONPATH=. python3 -m modelwiki"

    goal = (f"`{metric}` is {_num(bench.get(metric, p['now'].get(metric)))} now and projects to "
            f"{_num(p['planned'].get(metric))} with every row in `plan.md`"
            + (f"; {reqs[metric]} is required" if metric in reqs else "") + ".")
    if p["fails"]:
        goal += f" Still short after the plan: {'; '.join(p['fails'])}."

    out_md = [
        "---",
        f"name: {plan['model']}-plan-{commit[:12]}",
        "description: >-",
        f"  Optimize {page['title']} served from {tree} at {commit[:12]} for the workload {wl.get('name', 'as profiled')}:",
        "  the plan, the model's rows and the procedure. Use when working this plan.",
        "---",
        "",
        f"# Work the plan: {page['title']} at {commit[:12]}",
        "",
        goal,
        "",
        "## What is here",
        "",
        "| File | What it holds | Read it |",
        "| --- | --- | --- |",
        "| `plan.md` | The work in order: the optimizations to bench one by one, then the small ones to bundle. | First, and again before each row. |",
        "| `model/model.md` | Every feature row by ID: Want, Layers, Technique, Handoff (Shape tuning, Accuracy eval). "
        "A row ID in `plan.md` is its ID here. | When you start a row. |",
        "| `model/serving.md` | The serving rows with this model's numbers: each row's settings and sweep. "
        "| When you start an `s-` row. |",
        "| `how-to-work.md` | How a row is implemented, checked, tuned, gated and closed; the bundle of small "
        "rows; how to measure. | Before the first row. |",
        "| `features.json` | Per row: status, evidence (`path:line` in the tree at the commit), the kernels it owns, "
        "the change and why, `depends_on`, its config. | When you start a row: where to look. |",
    ]
    if req.get("workload"):
        out_md.append("| `workload.yaml` | The workload the plan projects at: load, lengths, speculation, metric and "
                      "requirements. | To plan again. |")
    out_md += [
        "",
        "## Before you start",
        "",
        f"1. The plan is for `{tree}` at `{commit}`, and every `path:line` in `plan.md` and `features.json` is at "
        f"that commit. Work from it alone: if `git -C {tree} rev-parse HEAD` is anything else, commit or stash what "
        f"is there and run `git -C {tree} checkout -b <branch> {commit[:12]}`. Take nothing from later commits, other "
        "branches or upstream vLLM.",
    ]
    cmd = _start_command(plan)
    bench = ", ".join(f"{k} {_num(v)}" for k, v in (plan.get("bench") or {}).items() if k == metric or k in reqs)
    out_md += (["2. Serve the profiled launch:", "", "   ```", f"   {cmd}", "   ```", ""] if cmd
               else ["2. Serve the launch as profiled."])
    out_md.append(f"3. The profiled run gave {bench}." if bench else "")
    out_md += [
        "",
        "## Order of work",
        "",
        "1. **Baseline.** Serve the launch above and bench it three times; the median is the baseline every change is "
        "measured against, the spread the noise floor.",
        f"2. **Optimizations in order** (`plan.md` section 1), each projecting {_pct(FLOOR)} or more, top to bottom; a "
        "row under After lands first. For each row:",
        "   - Read its row in `model/model.md` (Want, Technique, Handoff) and its entry in `features.json` (evidence, "
        "kernels, why).",
        "   - Implement, check against the separate-kernel reference in `eval/`, and tune it, as "
        "`how-to-work.md` says under Features.",
        "   - Serve it and bench against the current baseline. A kept change is the new baseline: bench it again "
        f"before the next row. One that measures under {_pct(FLOOR)} end to end is not reverted: once its check "
        "passes, it joins step 3.",
        f"3. **Under {_pct(FLOOR)}** (`plan.md` section 2). An end-to-end uplift that small is inside the noise, so it "
        "decides nothing until the end. Build each row as a patch on the kept tree and check it in `eval/` (correct, "
        "and its kernels faster at the served shapes), without a bench. After step 2, apply them all, serve them as "
        "one change and bench it; a bundle that loses is split in halves to find the row.",
        f"4. **Plan again after a big cut.** A change that cuts {_pct(REPLAN_CUT)} or more of the step"
        + (f" ({', '.join(f'`{r['id']}` {_pct(r['cut'])}' for r in big)})" if big else "")
        + " moves every other row's weight. Profile the kept tree with ruppell and run:",
        "",
        "   ```",
        f"   {cli} plan {plan['model']} --tree {tree} --commit <kept commit> --profile <ruppell run> --steps A:B "
        f"--workload {out / 'workload.yaml'} --bench-file <bench.json> --out <new folder>",
        "   ```",
        "",
        "   then work from the new folder's `SKILL.md`, its start command on the kept tree.",
        "5. **Stop** when the requirements are met, or when every change left is under the noise floor.",
    ]
    if need.get("step_ms"):
        kernels = need.get("kernels_per_decode_step")
        out_md.append(f"   Still short then: report it with what the requirements need, a mean step of "
                      f"{_num(need['step_ms'])} ms"
                      + (f", at most {kernels:.0f} kernels a decode step at the launch floor" if kernels else "")
                      + ".")

    out_md += [
        "",
        "## Reading the numbers",
        "",
        "- **Weight**: the row's share of step time now, the most it can remove.",
        f"- **Projected `{metric}` (+%)**: the metric with only this row landed: its change removed from the profiled "
        "steps, calibrated so now matches the bench. Gains do not add; the plan's total composes them.",
        "- **After**: rows that must land first. **Where**: the first evidence line; `features.json` has all of it.",
        "- **Status**: `on` served and in the profile; `off-by-config` in the tree, not enabled by the launch; "
        "`unwired` the kernel exists, the model does not call it; `port` exists at the plan's commit for another model, "
        "dtype or platform (its evidence names where), so adapt that code; "
        "`write` nothing to build on; `partial` part of the technique is on.",
        "- A row that measures well under its projection: check its kernels left the trace (its `kernels` in "
        "`features.json`) before moving on.",
    ]
    if unplanned:
        out_md.append("- Rows in `model/model.md` the plan does not price yet, with no projection: "
                      + ", ".join(f"`{i}`" for i in unplanned)
                      + ". Plan again once they have evidence.")
    out_md += [
        "- This file and `plan.md` set the order; `how-to-work.md`'s rules (check, tune, gate, bundle, "
        f"measure) hold for every row; its round orders give way to `plan.md`'s, and its 5 % bundle floor to the "
        f"plan's {_pct(FLOOR)}.",
    ]
    return "\n".join(out_md) + "\n"
