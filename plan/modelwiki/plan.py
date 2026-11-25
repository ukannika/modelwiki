"""Plans: a model page plus one served tree commit and its ruppell profile give features, projections, config and order."""
import json
import re
import shlex
import subprocess
from pathlib import Path

import numpy as np
import yaml

from . import ROOT
from .page import bucket_rows, load as load_page
from . import ruppell, scheduler, stepmodel

STATUSES = ("on", "off-by-config", "unwired", "port", "write", "partial", "n/a")
NEEDS_EVIDENCE = ("on", "off-by-config", "unwired", "port", "partial")
FEATURE_STAGE = ("off-by-config", "unwired", "port", "write", "partial")
# How a row removes time. optimize and fuse are computed from the profile's floors; the rest take a fraction.
CHANGES = ("optimize", "fuse", "remove_work", "overlap", "host", "estimate")
FRACTION = ("remove_work", "overlap", "host", "estimate")
BASES = ("reference", "budget", "estimate")
BENCH_KEYS = {"output_throughput": "output_tok_s_gpu", "total_throughput_per_gpu": "total_tok_s_gpu",
              "mean_interactivity": "mean_interactivity", "p50_interactivity": "p50_interactivity",
              "p90_interactivity": "p90_interactivity", "p90_ttft_s": "p90_ttft_s", "p90_tpot_s": "p90_tpot_s"}
METRICS = ("output_tok_s_gpu", "total_tok_s_gpu", "mean_interactivity", "p50_interactivity", "p90_interactivity",
           "p90_ttft_s")
FLOOR = 0.02
PROFILER_ENV = ("RUPPELL_", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONUNBUFFERED")
MAX_CUT = 0.95


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def _git(tree, *args):
    return subprocess.run(["git", "-C", str(tree), *args], capture_output=True, text=True)


def is_git(tree):
    return _git(tree, "rev-parse", "--is-inside-work-tree").returncode == 0


def resolve_commit(tree, ref="HEAD"):
    """A git tree resolves ref to a full sha. Any other directory takes ref as its label."""
    if not is_git(tree):
        if ref == "HEAD":
            raise ValueError(f"{tree} is not a git tree: give commit and trace_commit as a label")
        return ref
    r = _git(tree, "rev-parse", ref)
    if r.returncode:
        raise ValueError(f"{tree}: cannot resolve {ref}: {r.stderr.strip()}")
    return r.stdout.strip()


def file_lines(tree, commit, path):
    """Line count of path at commit (on disk for a tree that is not git), or None when it does not exist."""
    if not is_git(tree):
        p = Path(tree) / path
        return len(p.read_text(errors="replace").splitlines()) if p.is_file() else None
    r = _git(tree, "show", f"{commit}:{path}")
    return None if r.returncode else len(r.stdout.splitlines())


def _slug(s):
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-")


def _ruppell_cfg(req, base):
    cfg = dict(req.get("ruppell") or {})
    if not (cfg.get("run") or cfg.get("tables")):
        raise ValueError("request needs ruppell.run (a ruppell run or trace) or ruppell.tables (kernels.csv, steps.csv)")
    for k in ("run", "tables"):
        if cfg.get(k) and not Path(cfg[k]).is_absolute():
            cfg[k] = str((Path(base) / cfg[k]).resolve())
    return cfg


def _row_buckets(page):
    by_id, by_name = {}, {}
    for b in page["buckets"]:
        ids, serving = bucket_rows(b["rows_md"])
        for i in ids:
            by_id.setdefault(i, []).append(b["name"])
        for n in serving:
            by_name.setdefault(n, []).append(b["name"])
    return by_id, by_name


def skeleton(page):
    by_id, by_name = _row_buckets(page)
    rows = []
    for i, r in enumerate(page["features"] + page["serving"]):
        serving = r["section"] == "serving"
        rid = f"s-{_slug(r['subsection'] or '')}-{_slug(r['Feature'])}" if serving else r["id"]
        rows.append({
            "id": rid, "stage": "serving" if serving else "features", "page_index": i,
            "section": r["section"], "subsection": r["subsection"], "feature": r["Feature"],
            "want": r.get("Want", ""), "technique": r.get("Technique", ""),
            "status": None, "evidence": [], "kernels": [],
            "buckets": (by_name.get(r["Feature"]) or by_name.get(r["subsection"], [])) if serving
            else by_id.get(rid, []),
            "weight": None, "change": {"kind": None, "fraction": None, "basis": None, "kernels": [], "why": ""},
            "depends_on": [], "launch_blocker": False, "config": {}, "notes": "",
        })
    return rows


def load_workload(spec, root=ROOT, base=None):
    """A workload: a dict, a YAML path, or a name under examples/."""
    if spec is None or isinstance(spec, dict):
        return spec
    p = Path(spec)
    if not p.is_absolute() and base is not None and (Path(base) / p).exists():
        p = Path(base) / p
    if not p.exists():
        p = Path(root) / "examples" / f"{spec}.yaml"
    if not p.exists():
        raise ValueError(f"no workload {spec}: give a YAML path or a name under {Path(root) / 'examples'}")
    return load_yaml(p)


WORKLOAD_NEEDS = {"concurrency": "--concurrency", "isl": "--isl", "osl": "--osl", "prefix_hit": "--prefix-hit"}


def set_field(wl, key, value):
    """Set a dotted key (speculative.drafts, requirements.p90_ttft_s) in a workload, in place."""
    *path, last = key.split(".")
    d = wl
    for k in path:
        d = d.setdefault(k, {})
    d[last] = yaml.safe_load(value) if isinstance(value, str) else value
    return wl


def with_fields(wl, fields):
    """A copy of a workload with fields (key, value) set. A new concurrency without active_decodes drops the old
    ones."""
    out = json.loads(json.dumps(wl or {}))
    keys = [k for k, _ in fields]
    for k, v in fields:
        set_field(out, k, v)
    if "concurrency" in keys and "active_decodes" not in keys:
        out.pop("active_decodes", None)
    return out


def decode_points(v):
    """active_decodes as the points to project at: a number, or a range's low end, middle and high end."""
    if v is None:
        return None
    if isinstance(v, str):
        v = [float(x) for x in v.split("-")] if "-" in v.strip("-") else float(v)
    if isinstance(v, (list, tuple)):
        lo, hi = float(min(v)), float(max(v))
        return [lo] if lo == hi else [lo, (lo + hi) / 2, hi]
    return [float(v)]


def workload_errors(wl):
    def has(key):
        d = wl
        for k in key.split("."):
            if not isinstance(d, dict) or d.get(k) in (None, {}):
                return False
            d = d[k]
        return True
    out = [f"workload needs {k} ({flag})" for k, flag in WORKLOAD_NEEDS.items() if not has(k)]
    spec = wl.get("speculative") or {}
    if spec.get("drafts") and not spec.get("acceptance_length"):
        out.append("workload has drafts but no speculative.acceptance_length (--acceptance)")
    return out


def read_bench(path):
    """The served metrics of the profiled run from a bench result JSON, under the plan's metric names."""
    d = json.loads(Path(path).read_text())
    return {v: float(d[k]) for k, v in BENCH_KEYS.items() if isinstance(d.get(k), (int, float))}


def plan_dir(model, commit, root=ROOT):
    return Path(root) / "plans" / model / commit[:12]


def prepare(request, root=ROOT, allow_mismatch=False, out=None):
    """Validate the request (a dict, or a YAML file), create the plan directory (out, or plans/<model>/<commit>),
    and write the skeletons the plan agent fills; on a resume, add the page's rows the plan does not have yet.
    Relative paths in a file resolve from its directory, in a dict from the working directory."""
    root = Path(root)
    if isinstance(request, dict):
        req, base = dict(request), Path.cwd()
    else:
        req, base = load_yaml(request), Path(request).resolve().parent
    tree = Path(req["tree"]).resolve()
    commit = resolve_commit(tree, req.get("commit", "HEAD"))
    trace_commit = req.get("trace_commit")
    if not trace_commit:
        raise ValueError("request needs trace_commit: the tree commit the trace was taken on")
    trace_commit = resolve_commit(tree, trace_commit)
    if trace_commit != commit and not allow_mismatch:
        raise ValueError(f"trace was taken on {trace_commit[:12]}, the tree is at {commit[:12]}: profile this commit")
    rcfg = _ruppell_cfg(req, base)
    page = load_page(req["model"], root)

    bench = {}
    if req.get("bench_file"):
        bf = Path(req["bench_file"])
        bench = read_bench(bf if bf.is_absolute() else base / bf)
    bench.update(req.get("bench") or {})
    workload = load_workload(req.get("workload"), root, base)
    targets = [load_workload(t, root, base) for t in req.get("targets") or []]
    for w in [workload, *targets] if workload else targets:
        errs = workload_errors(w)
        if errs:
            raise ValueError(f"workload {w.get('name', '')}: " + "; ".join(errs))
    resolved = dict(req, tree=str(tree), commit=commit, trace_commit=trace_commit, ruppell=rcfg, bench=bench,
                    workload=workload, targets=targets)
    out = Path(out) if out else plan_dir(req["model"], _slug(commit) if not is_git(tree) else commit, root)
    out.mkdir(parents=True, exist_ok=True)
    (out / "request.yaml").write_text(yaml.safe_dump(resolved, sort_keys=False, allow_unicode=True))
    rows = json.loads((out / "features.json").read_text()) if (out / "features.json").exists() else []
    have = {r["id"] for r in rows}
    new = [r for r in skeleton(page) if r["id"] not in have]
    if new or not rows:
        (out / "features.json").write_text(json.dumps(rows + new, indent=1, ensure_ascii=False) + "\n")
    if not (out / "kernels.csv").exists():
        ruppell.collect(rcfg, out)
    if not (out / "bucket_map.yaml").exists():
        names = [b["name"] for b in page["buckets"] if b["name"] != "Host gaps"]
        (out / "bucket_map.yaml").write_text(
            "# Rules from the served tree: bucket plus scope, op, category and kernel regexes (omitted: any).\n"
            "# First match wins; put the most specific first. Buckets: " + ", ".join(names) + ".\n"
            "rules: []\n")
    (out / "brief.md").write_text(brief(page, resolved, root))
    return out


def brief(page, req, root=ROOT):
    lines = [f"# Plan brief: {page['title']} at {req['commit'][:12]}", "",
             f"Procedure: `{Path(root) / 'plan/SKILL.md'}`.", "",
             f"- Model page: `{page['path']}`. Serving: `{page['serving_path']}`. Buckets: `{page['buckets_path']}`.",
             f"- Bucket rules: `{Path(root) / 'shared/profile-buckets.md'}`.",
             f"- Tree: `{req['tree']}` at `{req['commit']}`. Read files at this commit only.",
             f"- Profile: ruppell `{req['ruppell'].get('run') or req['ruppell'].get('tables')}`"
             f" (steps {req['ruppell'].get('steps', 'all')}, rank {req['ruppell'].get('rank', 0)}),"
             " as `kernels.csv` (with each kernel's bytes, FLOPs and launches), `steps.csv`, `step_shapes.csv` and"
             " `rates.yaml` in this directory.",
             f"- Workload: `{json.dumps(req.get('workload'))}`.",
             f"- Bench: `{json.dumps(req.get('bench', {}))}`.", "",
             "## Buckets", "", "| Bucket | The block | Rows |", "| --- | --- | --- |"]
    for b in page["buckets"]:
        ids, serving = bucket_rows(b["rows_md"])
        lines.append(f"| {b['name']} | {b['block']} | {', '.join(ids + serving)} |")
    lines += ["", "## Rows", ""]
    for r in page["features"] + page["serving"]:
        where = r.get("Layers") or r.get("Where") or r.get("Allocation") or ""
        rid = r.get("id") or f"Serving: {r['subsection']}: {r['Feature']}"
        lines.append(f"- **{rid}** ({r['Feature']}). Want: {r.get('Want', '')} Technique: {r.get('Technique', '')}"
                     + (f" Layers: {where}" if where else ""))
    return "\n".join(lines) + "\n"


def load_plan(pdir):
    pdir = Path(pdir)
    req = load_yaml(pdir / "request.yaml")
    rows = json.loads((pdir / "features.json").read_text())
    return req, rows


def run_weigh(pdir):
    pdir = Path(pdir)
    req, _ = load_plan(pdir)
    rules = load_yaml(pdir / "bucket_map.yaml").get("rules") or []
    rates = load_yaml(pdir / "rates.yaml") if (pdir / "rates.yaml").exists() else None
    w = ruppell.weigh(ruppell.read(pdir / "kernels.csv"), ruppell.read(pdir / "steps.csv"), rules,
                      busy=req["ruppell"].get("busy"), rates=rates)
    (pdir / "weights.yaml").write_text(yaml.safe_dump(w, sort_keys=False, allow_unicode=True, width=200))
    return w


def context(pdir):
    """What projections read: the request, rows, weights, each kernel row's bucket and floor (scaled to host time),
    host and gap time per step kind, the step shapes and their fits, and the profile's rates."""
    pdir = Path(pdir)
    req, rows = load_plan(pdir)
    weights = load_yaml(pdir / "weights.yaml")
    rules = ruppell.compile_rules(load_yaml(pdir / "bucket_map.yaml").get("rules") or [])
    rates = load_yaml(pdir / "rates.yaml") if (pdir / "rates.yaml").exists() else None
    kinds = weights["kinds"]
    kernels = []
    for k in ruppell.read(pdir / "kernels.csv"):
        sc = kinds.get(k["kind"], {}).get("scale", 1.0)
        f, bound = ruppell.row_floor(k, rates)
        kernels.append({"kind": k["kind"], "scope": k["scope"], "op": k["op"], "name": k["name"],
                        "bucket": ruppell.classify(k, rules) or "Other", "ms": float(k["ms"]) * sc,
                        "floor": f * sc, "bound": bound, "shaped": float(k.get("shaped_ms") or 0) > 0,
                        "launches": float(k.get("launches") or 0), "i": len(kernels)})
    ctx = {"req": req, "rows": rows, "weights": weights, "kernels": kernels, "rates": rates,
           "host": {k: float(v["host_ms"]) for k, v in kinds.items()},
           "gaps": {k: float(v["gaps_ms"]) for k, v in kinds.items()},
           "kernel_ms": {k: float(v["kernel_ms"]) * float(v["scale"]) for k, v in kinds.items()},
           "launches": {k: sum(x["launches"] for x in kernels if x["kind"] == k) for k in kinds},
           "bench": req.get("bench") or {}, "shapes": None}
    if (pdir / "step_shapes.csv").exists() and req.get("workload"):
        steps = stepmodel.load(ruppell.read(pdir / "step_shapes.csv"))
        models = stepmodel.fits(steps)
        launches = {}
        for k in ruppell.read(pdir / "kernels.csv"):
            launches[k["kind"]] = launches.get(k["kind"], 0.0) + float(k.get("per_step") or 0)
        per_kernel = stepmodel.tracing(steps, models, launches)
        wl = dict(req["workload"])
        dec = [s["dec_reqs"] for s in steps if s["kind"] == "decode"]
        if dec:
            ctx["decodes_mean"] = sum(dec) / len(dec)
            ctx["decodes_profile"] = [float(np.percentile(dec, q)) for q in (10, 50, 90)]
            if wl.get("active_decodes") is None:
                wl["active_decodes"] = [ctx["decodes_profile"][0], ctx["decodes_profile"][2]]
        gpus = int(req.get("gpus") or (rates or {}).get("world") or 1)
        ctx.update(shapes=steps, models=models, tracing_ms_per_kernel=per_kernel, workload=wl, gpus=gpus,
                   launches_per_step=launches,
                   traced_ms=[s["host_ms"] for s in steps],
                   untraced_ms=stepmodel.untraced_ms(steps, per_kernel, launches))
        model = stepmodel.metrics(steps, ctx["traced_ms"], wl, gpus)
        ctx["model_traced"] = model
        ctx["ratio"] = stepmodel.ratio(ctx["bench"], model)
    return ctx


def _matched(r, ctx):
    ch = r.get("change") or {}
    pats = [re.compile(k) for k in (ch.get("kernels") or r.get("kernels") or [])]
    scope = re.compile(ch["scope"]) if ch.get("scope") else None
    buckets = set(r.get("buckets") or [])
    return [k for k in ctx["kernels"]
            if (not buckets or k["bucket"] in buckets) and (not pats or any(p.search(k["name"]) for p in pats))
            and (not scope or scope.search(k["scope"])) and (pats or buckets)]


def removal_parts(r, ctx):
    """What the row removes over the profile's window: host ms per kernel row (by index),
    per step kind's host gaps, and per step kind for an overlay that names no kernels. Launches a fusion or a removal
    drops take their share of the kind's host gaps (steps that launch eagerly wait on the host)."""
    ch = r.get("change") or {}
    kind, f = ch.get("kind"), float(ch.get("fraction") or 0)
    parts = {"kernels": {}, "gaps": {}, "overlay": {}}
    if r.get("status") == "n/a" or kind not in CHANGES:
        return parts
    kinds = [k for k in ch.get("step_kinds") or list(ctx["host"]) if k in ctx["host"]]
    kinds_host = sum(ctx["host"][k] for k in kinds)
    if kind in ("optimize", "fuse", "remove_work", "overlap"):
        matched = [x for x in _matched(r, ctx) if x["kind"] in kinds]
    elif kind == "estimate" and not r.get("weight"):
        buckets = set(r.get("buckets") or [])
        matched = [x for x in ctx["kernels"] if x["bucket"] in buckets and x["kind"] in kinds]
    else:
        matched = []
    for k in kinds:
        m = [x for x in matched if x["kind"] == k]
        cut, dropped = {}, 0.0
        if kind == "optimize":
            cut = {x["i"]: x["ms"] - x["floor"] for x in m}
        elif kind == "fuse":
            shaped = [x for x in m if x["shaped"]]
            if shaped:
                top = max(shaped, key=lambda x: x["floor"])
                cut = {x["i"]: x["ms"] - (x["floor"] if x is top else 0.0) for x in shaped}
                dropped = sum(x["launches"] for x in shaped) - top["launches"]
        elif kind in ("remove_work", "estimate"):
            cut = {x["i"]: f * x["ms"] for x in m}
            dropped = f * sum(x["launches"] for x in m) if kind == "remove_work" else 0.0
        elif kind == "overlap":
            mine = sum(x["ms"] for x in m)
            room = max(0.0, ctx["kernel_ms"].get(k, 0.0) - mine)
            scale = min(1.0, room / (f * mine)) if f and mine else 0.0
            cut = {x["i"]: f * scale * x["ms"] for x in m}
        parts["kernels"].update({i: max(0.0, v) for i, v in cut.items()})
        gaps = ctx["gaps"].get(k, 0.0)
        if kind == "host" or (kind == "estimate" and "Host gaps" in (r.get("buckets") or [])):
            parts["gaps"][k] = f * gaps
        elif dropped and ctx["launches"].get(k):
            parts["gaps"][k] = gaps * min(1.0, dropped / ctx["launches"][k])
        if kind == "estimate" and r.get("weight"):
            parts["overlay"][k] = f * float(r["weight"]["share"]) * sum(ctx["host"].values()) * ctx["host"][k] / kinds_host
    return parts


def removal(r, ctx, parts=None):
    """Host ms the row removes per step kind over the profile's window."""
    p = parts or removal_parts(r, ctx)
    out = dict.fromkeys(ctx["host"], 0.0)
    for i, v in p["kernels"].items():
        out[ctx["kernels"][i]["kind"]] += v
    for part in ("gaps", "overlay"):
        for k, v in p[part].items():
            out[k] += v
    return {k: v for k, v in out.items() if v}


def compose(parts_list, ctx):
    """Host ms per step kind that several rows remove together. On each kernel row, and on each kind's gaps, every
    row takes its share of what the rows before it left; overlays then take their share of what is left of the kind."""
    kept = {}
    gaps_kept = {}
    for p in parts_list:
        for i, v in p["kernels"].items():
            ms = ctx["kernels"][i]["ms"]
            if ms > 0:
                kept[i] = kept.get(i, 1.0) * max(0.0, 1 - v / ms)
        for k, v in p["gaps"].items():
            g = ctx["gaps"].get(k, 0.0)
            if g > 0:
                gaps_kept[k] = gaps_kept.get(k, 1.0) * max(0.0, 1 - v / g)
    out = dict.fromkeys(ctx["host"], 0.0)
    for i, frac in kept.items():
        out[ctx["kernels"][i]["kind"]] += ctx["kernels"][i]["ms"] * (1 - frac)
    for k, frac in gaps_kept.items():
        out[k] += ctx["gaps"][k] * (1 - frac)
    for p in parts_list:
        for k, v in p["overlay"].items():
            left = ctx["host"][k] - out[k]
            out[k] += left * min(1.0, v / ctx["host"][k])
    return {k: min(MAX_CUT * ctx["host"][k], v) for k, v in out.items() if v}


def _basis(r):
    ch = r.get("change") or {}
    return (ch.get("basis") or "estimate") if ch.get("kind") == "estimate" else ch.get("kind") or "estimate"


def evaluate(ctx, removed, target=None, decodes=None):
    """Served metrics with removed[kind] host ms taken out, at the profiled workload or at a target one with decodes
    active decodes: the step model's numbers scaled by measured over modeled at the profile. Without step shapes, the
    bench scaled by the step-time speedup."""
    host = ctx["host"]
    bench = ctx["bench"]
    if not ctx.get("shapes"):
        h = sum(host.values())
        sp = h / max(h * (1 - MAX_CUT), h - sum(removed.values()))
        return ({k: (v / sp if k.endswith("_s") else v * sp) for k, v in bench.items() if k in METRICS}, [])
    steps, base = ctx["shapes"], ctx["untraced_ms"]
    times = [max(u * (1 - MAX_CUT), u - removed.get(s["kind"], 0.0) * s["host_ms"] / host[s["kind"]])
             if host.get(s["kind"]) else u for s, u in zip(steps, base)]
    wl, past = ctx["workload"], []
    if target:
        base = ctx.get("decodes_mean") or 1.0
        target = dict(target, active_decodes=decodes or base)
        steps, factor, past = stepmodel.retarget(steps, ctx["models"], dict(wl, active_decodes=base), target)
        times = [t * x for t, x in zip(times, factor)]
        if abs(target["active_decodes"] / base - 1) > 0.01:
            past += [f"{k}: step time does not vary with rows in the profile" for k, m in ctx["models"].items()
                     if k == "decode" and not any(m["coef"].values())]
        wl = target
    m = stepmodel.metrics(steps, times, wl, ctx["gpus"])
    out = {k: m[k] * ctx["ratio"].get(k, 1.0) for k in ("output_tok_s_gpu", "mean_interactivity",
                                                         "p50_interactivity", "p90_interactivity") if k in m}
    model = ctx["model_traced"]
    if "total_tok_s_gpu" in bench and model["output_tok_s_gpu"]:
        out["total_tok_s_gpu"] = bench["total_tok_s_gpu"] * m["output_tok_s_gpu"] / model["output_tok_s_gpu"]
    if "p90_ttft_s" in bench and model["mean_prefill_step_ms"]:
        out["p90_ttft_s"] = bench["p90_ttft_s"] * m["mean_prefill_step_ms"] / model["mean_prefill_step_ms"]
    return out, past


def _lower_better(key):
    return key.endswith("_s")


def requirements_failed(metrics, now, reqs):
    fails = []
    for key, want in (reqs or {}).items():
        v = metrics.get(key)
        if v is None:
            continue
        ref = now.get(key) if want == "not_worse" else float(want)
        if ref is None:
            continue
        bad = v > ref * (1.01 if want == "not_worse" else 1) if _lower_better(key) else \
            v < ref * (0.99 if want == "not_worse" else 1)
        if bad:
            fails.append(f"{key} {v:.2f} against {want}")
    return fails


def check(pdir):
    """Errors and warnings in the plan agent's features.json, against the tree commit and the trace."""
    pdir = Path(pdir)
    req, rows = load_plan(pdir)
    tree, commit = req["tree"], req["commit"]
    weights = load_yaml(pdir / "weights.yaml") if (pdir / "weights.yaml").exists() else None
    names = ruppell.kernel_names(ruppell.read(pdir / "kernels.csv"))
    ctx = context(pdir) if weights else None
    ids = {r["id"] for r in rows}
    errors, warnings = [], []
    for r in rows:
        rid, st = r["id"], r.get("status")
        if st not in STATUSES:
            errors.append(f"{rid}: status {st!r} is not one of {', '.join(STATUSES)}")
            continue
        if st in NEEDS_EVIDENCE and not r.get("evidence"):
            errors.append(f"{rid}: status {st} needs file:line evidence")
        if st == "write" and not r.get("notes"):
            errors.append(f"{rid}: status write needs notes saying what was searched")
        for ev in r.get("evidence", []):
            m = re.match(r"^(.+?):(\d+)(?:-(\d+))?$", ev)
            if not m:
                errors.append(f"{rid}: evidence {ev!r} is not path:line")
                continue
            n = file_lines(tree, commit, m.group(1))
            last = int(m.group(3) or m.group(2))
            if n is None:
                errors.append(f"{rid}: {m.group(1)} does not exist at {commit[:12]}")
            elif last > n:
                errors.append(f"{rid}: {ev} is past the end of the file ({n} lines)")
        if st == "on":
            if not r.get("kernels") and r.get("stage") != "serving":
                errors.append(f"{rid}: on, but names no kernel to find in the trace")
            for k in r.get("kernels", []):
                if not any(re.search(k, n) for n in names):
                    errors.append(f"{rid}: on, but kernel /{k}/ is not in the trace: partial or off")
        for d in r.get("depends_on", []):
            if d not in ids:
                errors.append(f"{rid}: depends on unknown row {d}")
        ch = r.get("change") or {}
        kind = ch.get("kind")
        if st != "n/a" and kind:
            if kind not in CHANGES:
                errors.append(f"{rid}: change {kind!r} is not one of {', '.join(CHANGES)}")
            elif kind in FRACTION:
                f = ch.get("fraction")
                if f is None:
                    errors.append(f"{rid}: change {kind} needs a fraction")
                elif not 0 <= f <= 1:
                    errors.append(f"{rid}: fraction {f} is outside 0..1")
                if kind == "estimate" and ch.get("basis") not in BASES:
                    errors.append(f"{rid}: estimate basis {ch.get('basis')!r} is not one of {', '.join(BASES)}")
            if ctx and kind in ("optimize", "fuse", "remove_work", "overlap"):
                m = _matched(r, ctx)
                if not m:
                    warnings.append(f"{rid}: change {kind} matches no kernel time in its buckets: projected at zero")
                elif kind in ("optimize", "fuse") and not any(x["shaped"] for x in m):
                    warnings.append(f"{rid}: its kernels have no recorded shapes, so no floor: give an estimate")
            if ctx and kind == "estimate" and ctx.get("rates") and not r.get("weight"):
                claim = sum(removal(r, ctx).values())
                bs = [ctx["weights"]["buckets"].get(b, {}) for b in r.get("buckets") or []]
                room = sum(b.get("ms", 0) - b.get("floor_ms", 0) for b in bs)
                measured = bs and all(b.get("shaped", 0) >= 0.8 for b in bs)
                if measured and claim > room * 1.05 and "Host gaps" not in (r.get("buckets") or []):
                    warnings.append(f"{rid}: claims {claim / 1e3:.1f} s, above its buckets' floor headroom "
                                    f"{room / 1e3:.1f} s: say what work it removes")
        elif st in FEATURE_STAGE:
            warnings.append(f"{rid}: {st} with no change: projected at zero")
        if weights:
            for b in r.get("buckets", []):
                if b not in weights["buckets"]:
                    warnings.append(f"{rid}: bucket {b} has no time in the trace")
        if not r.get("buckets") and not r.get("weight") and st in FEATURE_STAGE and kind != "host":
            warnings.append(f"{rid}: no bucket and no weight: projected at zero")
    if weights and weights["buckets"].get("Other", {}).get("share", 0) > 0.05:
        warnings.append(f"Other holds {weights['buckets']['Other']['share']:.1%} of the step: extend bucket_map.yaml")
    if ctx:
        if not ctx["bench"]:
            warnings.append("no bench for the profiled run: metrics are the step model's, uncalibrated")
        if not ctx.get("shapes"):
            warnings.append("no step shapes or workload: metrics are the bench scaled by the step-time cut")
        else:
            pts = decode_points((req.get("workload") or {}).get("active_decodes"))
            prof = ctx.get("decodes_profile")
            if pts and prof:
                lo, hi = (min(pts) * 0.8, max(pts) * 1.2) if len(pts) == 1 else (min(pts), max(pts))
                if not lo <= prof[1] <= hi:
                    warnings.append(f"workload active_decodes {(req.get('workload') or {})['active_decodes']} does not "
                                    f"hold the profile's (p50 {prof[1]:g}, p10-p90 {prof[0]:g}-{prof[2]:g}): the "
                                    "profile ran another load")
    report = {"commit": commit, "errors": errors, "warnings": warnings}
    (pdir / "check.json").write_text(json.dumps(report, indent=1) + "\n")
    return report


def _gain(m, now, key):
    if not now.get(key) or key not in m:
        return 0.0
    g = m[key] / now[key] - 1
    return -g if _lower_better(key) else g


def project(pdir, root=ROOT):
    """Each row's removed time per step kind, the served metrics it gives at the workload, then the order: launch
    blockers, dependencies, projected gain in the workload's metric, page order."""
    pdir = Path(pdir)
    ctx = context(pdir)
    req, rows = ctx["req"], ctx["rows"]
    wl = ctx.get("workload") or req.get("workload") or {}
    metric = wl.get("metric", "output_tok_s_gpu")
    reqs = wl.get("requirements") or {}
    now, _ = evaluate(ctx, {})
    host_total = sum(ctx["host"].values())
    for r in rows:
        r["basis"] = _basis(r)
        r["_parts"] = removal_parts(r, ctx)
        rem = {k: min(MAX_CUT * ctx["host"][k], v) for k, v in removal(r, ctx, r["_parts"]).items()}
        r["removed_ms"] = {k: round(v, 1) for k, v in rem.items() if v}
        r["cut"] = sum(rem.values()) / host_total if host_total else 0.0
        r["share"] = sum(ctx["weights"]["buckets"].get(b, {}).get("share", 0.0) for b in r.get("buckets") or []) \
            if not r.get("weight") else float(r["weight"]["share"])
        r["metrics"], _ = evaluate(ctx, rem)
        r["gain"] = _gain(r["metrics"], now, metric) if now else r["cut"] / (1 - r["cut"])
        r["fails"] = requirements_failed(r["metrics"], now, reqs)

    by_id = {r["id"]: r for r in rows}
    stages = {
        "features": [r for r in rows if r["stage"] == "features" and r.get("status") in FEATURE_STAGE],
        "optimize": [r for r in rows if r["stage"] == "features" and r.get("status") == "on" and r["cut"] > 0],
        "serving": [r for r in rows if r["stage"] == "serving" and r.get("status") != "n/a"
                    and (r.get("status") != "on" or r["cut"] > 0)],
    }
    ordered = {k: _order(v, by_id) for k, v in stages.items()}
    total = compose([r["_parts"] for v in stages.values() for r in v], ctx)
    planned, _ = evaluate(ctx, total)
    targets = []
    for t in req.get("targets") or []:
        targets.append(_target(ctx, t, total, reqs))
    sched = _scheduler(ctx, total)
    plan = {
        "model": req["model"], "commit": req["commit"], "bench": ctx["bench"],
        "workload": wl, "metric": metric,
        "step_model": _step_model(ctx),
        "projected": {"step_cut": sum(total.values()) / host_total if host_total else 0.0,
                      "removed_ms": {k: round(v, 1) for k, v in total.items()}, "now": now, "planned": planned,
                      "fails": requirements_failed(planned, now, reqs), "targets": targets},
        "scheduler": sched,
        "launch": req.get("launch") or _profiled_launch(req, rows),
        "config": {r["id"]: r["config"] for r in rows if r.get("config")},
        "stages": {k: [_summary(r) for r in v] for k, v in ordered.items()},
        "rows": [_summary(r) for r in rows],
    }
    (pdir / "plan.json").write_text(json.dumps(plan, indent=1, ensure_ascii=False) + "\n")
    (pdir / "plan.md").write_text(plan_md(plan))
    (pdir / "projection.md").write_text(projection_md(plan, ctx["weights"]))
    return plan


def _profiled_launch(req, rows):
    """The profiled run's serve command (the ruppell run's run.json) and the environment its launch script exported,
    less the profiler's own; a row's config is the value to try, not the one served."""
    run = (req.get("ruppell") or {}).get("run")
    path = Path(run) / "run.json" if run else None
    cmd = json.loads(path.read_text()).get("command") if path and path.exists() else None
    env = {}
    for sh in sorted(path.parent.parent.glob("*.sh")) if path else []:
        text = sh.read_text(errors="replace")
        if "vllm serve" not in text:
            continue
        for line in text.splitlines():
            if not line.strip().startswith("export "):
                continue
            for word in shlex.split(line.strip()[7:], comments=True):
                k, _, v = word.partition("=")
                if v and "$" not in v and not k.startswith(PROFILER_ENV):
                    env[k] = v
        break
    return {"command": cmd, "env": env} if cmd or env else {}


def _target(ctx, t, total, reqs):
    """Metrics now and planned at a target workload, at each point of its active decodes (scaled from the profile's
    range by concurrency when not given); the middle point is the headline, the ends the range."""
    pts = decode_points(t.get("active_decodes"))
    if not pts and ctx.get("decodes_profile"):
        r = float(t.get("concurrency") or 1) / float(ctx["workload"].get("concurrency") or t.get("concurrency") or 1)
        pts = decode_points([ctx["decodes_profile"][0] * r, ctx["decodes_profile"][2] * r])
    pts = pts or [None]
    runs, past, fails = [], set(), []
    for d in pts:
        now, p1 = evaluate(ctx, {}, t, d)
        planned, _ = evaluate(ctx, total, t, d)
        runs.append((now, planned))
        past.update(p1)
        fails += [f"at {d:g} active decodes: {f}" if d else f
                  for f in requirements_failed(planned, now, t.get("requirements") or reqs)]
    mid = runs[len(runs) // 2]

    def span(i):
        return {k: [min(r[i][k] for r in runs), max(r[i][k] for r in runs)] for k in mid[i]}

    return {"workload": t.get("name"), "active_decodes": pts, "now": mid[0], "planned": mid[1],
            "now_range": span(0), "planned_range": span(1), "extrapolated": sorted(past), "fails": fails}


def _scheduler(ctx, total):
    """The running-sequence cap and step token budget, now and with the plan's cut on decode and prompt steps."""
    if not ctx.get("shapes"):
        return None
    n, untraced = {}, {}
    for st, u in zip(ctx["shapes"], ctx["untraced_ms"]):
        n[st["kind"]] = n.get(st["kind"], 0) + 1
        untraced[st["kind"]] = untraced.get(st["kind"], 0.0) + u
    T = scheduler.terms(ctx["shapes"], ctx["untraced_ms"], ctx["models"], ctx["workload"], ctx["launches_per_step"],
                        {k: ctx["gaps"].get(k, 0.0) / c for k, c in n.items()},
                        (ctx["rates"] or {}).get("kernel_floor_us") or 0.0)
    if not T:
        return None
    prompt = [k for k in untraced if k != "decode"]
    p_ms = sum(untraced[k] for k in prompt)
    cut = {"decode": min(MAX_CUT, total.get("decode", 0.0) / untraced["decode"]) if untraced.get("decode") else 0.0,
           "prompt": min(MAX_CUT, sum(total.get(k, 0.0) for k in prompt) / p_ms) if p_ms else 0.0}
    return scheduler.sweep(T, ctx["bench"], ctx["workload"], ctx["gpus"], cut, requirements_failed)


def _step_model(ctx):
    if not ctx.get("shapes"):
        return None
    return {"fits": {k: {x: m[x] for x in ("n", "traced", "r2", "mape", "intercept", "coef")}
                     for k, m in ctx["models"].items()},
            "tracing_ms_per_kernel": ctx["tracing_ms_per_kernel"], "gpus": ctx["gpus"],
            "modeled_at_profile": ctx["model_traced"], "measured_over_modeled": ctx["ratio"],
            "rates": ctx["rates"]}


def _order(rows, by_id):
    """Launch blockers first. Then repeatedly the ready row with the largest gain, ties to page order."""
    placed, out = set(), []
    pending = list(rows)
    in_stage = {r["id"] for r in rows}

    def ready(r):
        return all(d in placed or d not in in_stage or by_id.get(d, {}).get("status") == "on"
                   for d in r.get("depends_on", []))

    def key(r):
        return (not r.get("launch_blocker"), bool(r["fails"]), r["gain"] < FLOOR, -r["gain"], r["page_index"])

    while pending:
        cands = [r for r in pending if ready(r)] or pending
        best = min(cands, key=key)
        out.append(best)
        placed.add(best["id"])
        pending.remove(best)
    return out


def _summary(r):
    keys = ("id", "feature", "section", "subsection", "status", "buckets", "share", "change", "basis",
            "removed_ms", "cut", "gain", "metrics", "fails", "depends_on", "launch_blocker", "evidence", "kernels",
            "config", "notes")
    s = {k: r.get(k) for k in keys}
    s["subfloor"] = r.get("gain", 0) < FLOOR
    return s


def _pct(x):
    return f"{100 * x:.1f}%"


def _num(v):
    return "" if v is None else f"{v:.2f}" if abs(v) < 100 else f"{v:.1f}"


def _span(v, fmt=None):
    lo, hi = v
    f = (lambda x: fmt.format(round(x, 1))) if fmt else _num
    return f(lo) if abs(hi - lo) < 1e-9 * max(1.0, abs(lo)) else f"{f(lo)}-{f(hi)}"


def projection_md(plan, weights):
    """The numbers behind plan.md: metrics at the workload and its targets, the step model, the scheduler sweep, the
    buckets, every row's projection and the audit."""
    p = plan["projected"]
    metric = plan["metric"]
    now, planned, bench = p["now"], p["planned"], plan["bench"]
    out = [f"# Projection: {plan['model']} at {plan['commit'][:12]}", "", "The numbers behind `plan.md`.", "",
           f"Workload `{(plan['workload'] or {}).get('name', 'as profiled')}`, ranked by `{metric}`. "
           f"Projected step-time cut {_pct(p['step_cut'])}.", ""]
    keys = [k for k in METRICS if k in now or k in bench]
    if keys:
        out += ["| Metric | Bench (profiled run) | Now (untraced model) | Planned |", "| --- | ---: | ---: | ---: |"]
        out += [f"| {k} | {_num(bench.get(k))} | {_num(now.get(k))} | {_num(planned.get(k))} |" for k in keys]
        if any(k not in bench for k in keys):
            out += ["", "A metric with no bench value is the step model's own, not calibrated to a measurement."]
    if p["fails"]:
        out += ["", "Requirements not met: " + "; ".join(p["fails"]) + "."]
    for t in p["targets"]:
        pts = [x for x in t["active_decodes"] if x is not None]
        dec = f" ({_span([min(pts), max(pts)], '{:g}')} active decodes)" if pts else ""
        out += ["", f"At workload `{t['workload']}`{dec}: " + ", ".join(
            f"{k} {_span(t['now_range'][k])} to {_span(t['planned_range'][k])}" for k in keys if k in t["planned"]) + "."]
        if t["extrapolated"]:
            out.append("Outside the profile's range, so a guess: " + ", ".join(t["extrapolated"]) + ".")
        if t["fails"]:
            out.append("Requirements not met: " + "; ".join(t["fails"]) + ".")
    sm = plan.get("step_model")
    if sm:
        out += ["", "Step model: " + "; ".join(f"{k} steps n={f['n']}, R² {f['r2']:.2f}, mean error {_pct(f['mape'])}"
                                              for k, f in sm["fits"].items())
                + f". Tracing adds {sm['tracing_ms_per_kernel'] * 1e3:.2f} us per kernel."]
    if plan.get("scheduler"):
        out += _scheduler_md(plan["scheduler"], metric, (plan["workload"] or {}).get("requirements") or {})
    out += ["", "## Buckets", "", "| Bucket | Share | Floor headroom | Bound | Top scope / kernel |",
            "| --- | --- | --- | --- | --- |"]
    for b, w in weights["buckets"].items():
        top = f"`{w['top'][0]['scope']} / {w['top'][0]['kernel'][:60]}`" if w.get("top") else ""
        head = ""
        if "headroom" in w and b != "Host gaps":
            head = f"{_pct(w['headroom'])}" + (f" ({_pct(w['shaped'])} has shapes)" if w["shaped"] < 0.8 else "")
        out.append(f"| {b} | {_pct(w['share'])} | {head} | {w.get('bound', '') if head else ''} | {top} |")
    titles = {"features": "Features (map round 1)", "optimize": "Optimize (map round 2)", "serving": "Serving (map round 3)"}
    for stage, rows in plan["stages"].items():
        out += ["", f"## {titles[stage]}", ""]
        if not rows:
            out.append("Nothing to do.")
            continue
        out += [f"| # | Row | Status | Change | Share | Cut | {metric} | Depends on | Why |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
        main = [r for r in rows if not r["subfloor"] or r["launch_blocker"]]
        sub = [r for r in rows if r["subfloor"] and not r["launch_blocker"]]
        for i, r in enumerate(main, 1):
            out.append(_row_md(i, r, metric))
        if sub:
            out += ["", f"Subfloor bundle (each under {_pct(FLOOR)} gain): " + ", ".join(f"`{r['id']}`" for r in sub)]
    out += ["", "## Config", ""]
    launch = plan["launch"]
    out += [f"- Launch: `{json.dumps(launch)}`" if launch else "- Launch: as the request."]
    sc = plan.get("scheduler")
    if sc:
        out.append(f"- Scheduler (projected with the plan): `{json.dumps(_sched_config(sc['planned']))}`")
    out += [f"- `{rid}`: `{json.dumps(cfg)}`" for rid, cfg in plan["config"].items()]
    out += ["", "## Audit", "", "| Row | Status | Evidence | Kernels |", "| --- | --- | --- | --- |"]
    for r in plan["rows"]:
        ev = ", ".join(f"`{e}`" for e in r["evidence"] or [])
        ks = ", ".join(f"`{k}`" for k in r["kernels"] or [])
        out.append(f"| `{r['id']}` | {r['status']} | {ev} | {ks} |")
    return "\n".join(out) + "\n"


def _cap(c):
    return "none" if c is None else f"{c:g}"


def _sched_config(p):
    return {"running_sequences": p["cap"], "step_token_budget": p["budget"]}


def _scheduler_md(sc, metric, reqs):
    a, need = sc["assumptions"], sc["need"]
    keys = [metric] + [k for k in reqs if k != metric]
    keys = [k for k in keys if k in sc["planned"]["metrics"]]
    decode = ("Decode time against rows is the profile's fit." if a["decode_fit"] else
              f"The profile does not vary decode rows, so a decode step at another cap lies between flat and all its "
              f"time above the launch floor ({a['decode_floor_ms']:.1f} ms) scaling with rows: the numbers are the "
              f"middle, the ranges the two ends.")
    clients = (f"Clients: {a['population']:.1f} in a closed loop, {a['think_s']:.1f} s between a reply and the next "
               f"request." if a["think_s"] else
               f"The profile decodes at least as many requests as the workload has clients, so all {a['population']:.1f} "
               f"are taken as always at the server, with no time between requests: the waits are an upper bound.")
    out = ["", "## Scheduler: running sequences and step token budget", "",
           f"Projected in closed form from the profile's steps, scaled so the profiled point gives the bench. The profile "
           f"ran {a['decoding']:.1f} requests decoding at {a['rows_per_request']:.0f} rows each, "
           f"{a['prompt_steps_per_request']:.2f} prompt steps a request ({a['prompts']} prompts rebuilt) and a "
           f"{a['budget']:.0f}-token step budget; decode steps took {a['decode_ms']:.1f} ms and prompt steps "
           f"{a['prompt_ms']:.1f} ms. {clients} Requests past the cap wait, and TTFT counts the wait. {decode} "
           f"A budget above the profiled one was not served: its buffers come out of the KV pool.", ""]
    for name in ("now", "planned"):
        p = sc[name]
        vals = ", ".join(f"{k} {_num(p['metrics'][k])}" for k in keys)
        fails = f" Fails: {'; '.join(p['fails'])}." if p["fails"] else " Meets the requirements."
        out.append(f"- {name.capitalize()}: running sequences {_cap(p['cap'])}, budget {p['budget']}: {vals}.{fails}")
    b = sc["planned"]["budget"]
    out += ["", f"At a budget of {b}:", "",
            "| Running sequences | Decoding | Waiting | Step ms now | Step ms planned | "
            + " | ".join(f"{k} planned" for k in keys) + " |",
            "| --- | ---: | ---: | ---: | ---: | " + " | ".join("---:" for _ in keys) + " |"]
    now = {p["cap"]: p for p in sc["points"]["now"] if p["budget"] == b}
    for p in (p for p in sc["points"]["planned"] if p["budget"] == b):
        n = now.get(p["cap"])
        out.append(f"| {_cap(p['cap'])} | {p['decoding']:.1f} | {p['waiting']:.1f} | "
                   f"{_num(n['step_ms']) if n else ''} | {_num(p['step_ms'])} | "
                   + " | ".join(_span(p["range"][k]) for k in keys) + " |")
    c = sc["planned"]["cap"]
    out += ["", f"At running sequences {_cap(c)}, planned:", "",
            f"| Step token budget | Prompt-step share | Prompt step ms | Step ms | {metric} |",
            "| ---: | ---: | ---: | ---: | ---: |"]
    for p in (p for p in sc["points"]["planned"] if p["cap"] == c):
        out.append(f"| {p['budget']} | {_pct(p['prompt_share'])} | {_num(p['prompt_ms'])} | {_num(p['step_ms'])} | "
                   f"{_span(p['range'][metric])} |")
    if need["step_ms"]:
        by = ", ".join(f"{k} {v:.2f} ms" for k, v in need["by_requirement"].items())
        kern = (f"; at the launch floor that is at most {need['kernels_per_decode_step']:.0f} kernels a decode step"
                if need["kernels_per_decode_step"] else "")
        out += ["", f"The requirements need a mean step of {need['step_ms']:.2f} ms ({by}){kern}."]
    return out


def plan_md(plan):
    """The handoff, in the order of the work: the optimizations worth a bench each, then the ones under the floor
    to build and serve together at the end."""
    p, metric = plan["projected"], plan["metric"]
    wl = plan["workload"] or {}
    reqs = wl.get("requirements") or {}
    now, planned = p["now"], p["planned"]
    head = (f"Workload `{wl.get('name', 'as profiled')}`. `{metric}`: {_num(plan['bench'].get(metric, now.get(metric)))} "
            f"now, {_num(planned.get(metric))} projected with everything below"
            + (f", {reqs[metric]} required" if metric in reqs else "") + ".")
    if p["fails"]:
        head += f" Still short: {'; '.join(p['fails'])}."
    out = [f"# Plan: {plan['model']} at {plan['commit'][:12]}", "", head]

    rows = [r for st in ("features", "optimize", "serving") for r in plan["stages"].get(st) or []]
    gaining = sorted((r for r in rows if r["gain"] > 0.0005), key=lambda r: (-r["gain"], -r["share"]))
    big = [r for r in gaining if r["gain"] >= FLOOR]
    small = [r for r in gaining if r["gain"] < FLOOR]
    out += ["", f"## 1. Optimizations in order: each projecting {_pct(FLOOR)} or more", "",
            "Serve each and bench it against the current baseline; a kept change is the new baseline. A row under "
            f"After lands first. One that measures under {_pct(FLOOR)} end to end joins section 2 instead of being "
            "reverted, once its check in `eval/` passes.", ""]
    out += _rows_md(big, metric)
    out += ["", f"## 2. Under {_pct(FLOOR)}: build now, bench at the end", "",
            f"An end-to-end uplift under {_pct(FLOOR)} is inside the bench's noise: do not keep or revert these on it. "
            "Build each as a patch on the kept tree and check it in `eval/` (correct, and its kernels faster at the "
            "served shapes); after section 1, serve them together as one change and bench that.", ""]
    out += _rows_md(small, metric)
    rest = [r for r in rows if r not in gaining]
    if rest:
        out += ["", "No projected gain yet: " + ", ".join(f"`{r['id']}`" for r in rest) + "."]
    out += ["", "Weight is the row's share of step time; the projection removes its change from the profiled steps. "
            "Every row's evidence, kernels and change are in `features.json`."]
    return "\n".join(out) + "\n"


def _rows_md(rows, metric):
    if not rows:
        return ["None."]
    out = [f"| # | Row | Status | Weight | {metric} | After | What | Where |",
           "| --- | --- | --- | ---: | --- | --- | --- | --- |"]
    for i, r in enumerate(rows, 1):
        where = f"`{r['evidence'][0]}`" if r.get("evidence") else (f"`{json.dumps(r['config'])}`" if r.get("config")
                                                                   else "")
        after = ", ".join(f"`{d}`" for d in r.get("depends_on") or [])
        out.append(f"| {i} | `{r['id']}` | {r['status']} | {_pct(r['share'])} | {_gain_md(r, metric)} | {after} | "
                   f"{_what(r)} | {where} |")
    return out


def _start_command(plan):
    launch = plan.get("launch") or {}
    cmd = launch.get("command")
    if not cmd:
        return None
    env = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (launch.get("env") or {}).items())
    return (env + " " if env else "") + shlex.join(cmd)


def _gain_md(r, metric):
    v = (r.get("metrics") or {}).get(metric)
    sign = "+" if r["gain"] >= 0 else ""
    return f"{_num(v)} ({sign}{_pct(r['gain'])})" if v is not None else f"{sign}{_pct(r['gain'])}"


def _what(r):
    return (r.get("change") or {}).get("why") or r.get("notes") or ""


def _row_md(i, r, metric):
    ch = r["change"] or {}
    kind = ch.get("kind") or ""
    if ch.get("fraction") is not None and kind in FRACTION:
        kind += f" {_pct(ch['fraction'])}" + (f" ({ch.get('basis')})" if kind.startswith("estimate") else "")
    flag = " (launch blocker)" if r["launch_blocker"] else ""
    val = (r.get("metrics") or {}).get(metric)
    gain = f"{_num(val)} ({'+' if r['gain'] >= 0 else ''}{_pct(r['gain'])})" if val is not None else _pct(r["gain"])
    fails = f" Fails: {'; '.join(r['fails'])}." if r.get("fails") else ""
    return (f"| {i} | `{r['id']}`{flag} | {r['status']} | {kind} | {_pct(r['share'])} | {_pct(r['cut'])} | {gain} | "
            f"{', '.join(r['depends_on'] or [])} | {ch.get('why', '')}{fails} |")
