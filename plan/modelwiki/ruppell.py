"""Step time by bucket from a ruppell run: its kernels by step kind, module scope, op and name, with the work their
op calls did (from the recorded shapes), each step's shape, and its host time."""
import csv
import io
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import yaml

from . import floors

FOLD = "regexp_replace(coalesce(k.scope, ''), '\\.layers\\.[0-9]+', '.layers.*', 'g')"

# Per (step kind, folded scope, op, category, kernel, op signature) over the traced steps.
DETAIL_SQL = f"""
SELECT s.kind, {FOLD} AS scope, coalesce(k.op, '') AS op, coalesce(k.category, '') AS category, k.name,
       coalesce(k.sig_id, -1) AS sig_id, sum(k.dur_ns) AS ns, count(*) AS kernels, count(DISTINCT k.inst) AS calls
FROM kernels k JOIN steps s USING (rank, step) WHERE k.rank = {{rank}} GROUP BY ALL
"""

SIGS_SQL = "SELECT sig_id, dims, types FROM sigs"

STEPS_DETAIL_SQL = """
SELECT step, kind, traced, host_ns, fields FROM all_steps
WHERE rank = {rank} ORDER BY step
"""

# The shortest kernels per step kind: the launch floor a kernel cannot go under on this machine.
KERNEL_FLOOR_SQL = """
SELECT s.kind, quantile_cont(k.dur_ns, 0.05) / 1e3 AS us FROM kernels k JOIN steps s USING (rank, step)
WHERE k.rank = {rank} GROUP BY ALL
"""

WORLD_SQL = "SELECT world, count(*) AS n FROM colls GROUP BY world ORDER BY n DESC LIMIT 1"

ANNOTATION = re.compile(r"execute_(\d+)_context_(\d+)\(sq(\d+)sk(\d+)sqsq(\d+)sqsk(\d+)\)"
                        r"_generation_(\d+)\(sq(\d+)sk(\d+)sqsq(\d+)sqsk(\d+)\)")
SHAPE_COLS = ("step", "kind", "traced", "host_ms", "rows", "pf_reqs", "pf_tok", "pf_ctx", "pf_attn", "dec_reqs",
              "dec_rows", "dec_ctx")


def sql(src, query, steps=None, fmt=None, python_path=None):
    cmd = [sys.executable, "-m", "ruppell", "sql", *([src] if isinstance(src, str) else src),
           "--out-format", "csv", "--max-rows", "0", "-e", query]
    if steps:
        cmd += ["--steps", str(steps)]
    if fmt:
        cmd += ["--format", fmt]
    env = dict(os.environ)
    if python_path:
        env["PYTHONPATH"] = python_path + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if r.returncode:
        raise RuntimeError(f"ruppell sql failed: {r.stderr.strip()[-800:]}")
    return list(csv.DictReader(io.StringIO(r.stdout)))


def _write(path, rows):
    with open(path, "w", newline="") as f:
        if rows:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def read(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def step_shape(row):
    """One step's shape: prefill requests, tokens, context and attention work; decode requests, rows and context."""
    f = json.loads(row.get("fields") or "{}")
    out = {"step": int(row["step"]), "kind": row["kind"], "traced": str(row["traced"]).lower() == "true",
           "host_ms": int(row["host_ns"]) / 1e6, "rows": int(f.get("rows") or 0)}
    m = ANNOTATION.search(f.get("annotation") or "")
    if m:
        g = [int(x) for x in m.groups()]
        out.update(pf_reqs=g[1], pf_tok=g[2], pf_ctx=g[3], pf_attn=g[5], dec_reqs=g[6], dec_rows=g[7], dec_ctx=g[8])
    else:
        ctx = f.get("context_lengths") or []
        dec = out["kind"] == "decode"
        out.update(pf_reqs=0 if dec else 1, pf_tok=0 if dec else out["rows"], pf_ctx=0, pf_attn=0,
                   dec_reqs=int(f.get("running_requests") or len(ctx)) if dec else 0,
                   dec_rows=out["rows"] if dec else 0, dec_ctx=sum(ctx) if dec else 0)
    return out


def kernel_rows(detail, sigs, steps, world=1):
    """kernels.csv rows (kind, scope, op, category, name) with time, launches and the work of their op calls, each
    scaled from the traced steps to all steps of the kind; and the op calls, for the profile's own rates."""
    per = defaultdict(lambda: [0, 0])
    for s in steps:
        per[s["kind"]][0] += 1
        per[s["kind"]][1] += bool(s["traced"])
    groups = defaultdict(lambda: [0, 0])
    for d in detail:
        g = groups[(d["kind"], d["scope"], d["op"], d["sig_id"])]
        g[0] += int(d["ns"])
        g[1] = max(g[1], int(d["calls"]))
    work, calls = {}, []
    for (kind, scope, op, sig), (ns, ncalls) in groups.items():
        dims, types = sigs.get(str(sig), (None, None))
        w = floors.op_work(op, dims, types, world) if dims else None
        work[(kind, scope, op, sig)] = w
        if w and ncalls:
            calls.append((w, ns / ncalls, ncalls))
    rows = {}
    for d in detail:
        kind = d["kind"]
        steps_k, traced_k = per.get(kind, (0, 0))
        if not traced_k:
            continue
        scale = steps_k / traced_k
        key = (kind, d["scope"], d["op"], d["category"], d["name"])
        r = rows.setdefault(key, {"kind": kind, "scope": d["scope"], "op": d["op"], "category": d["category"],
                                  "name": d["name"], "ms": 0.0, "per_step": 0.0, "launches": 0.0, "bytes": 0.0,
                                  "flops": 0.0, "link_bytes": 0.0, "dtype": "", "shaped_ms": 0.0})
        ns = int(d["ns"])
        r["ms"] += ns * scale / 1e6
        r["per_step"] += int(d["kernels"]) / traced_k
        r["launches"] += int(d["kernels"]) * scale
        gk = (kind, d["scope"], d["op"], d["sig_id"])
        w = work.get(gk)
        if w:
            gns, gcalls = groups[gk]
            share = ns / gns if gns else 0.0
            r["bytes"] += w["bytes"] * gcalls * share * scale
            r["flops"] += w["flops"] * gcalls * share * scale
            r["link_bytes"] += w.get("link_bytes", 0) * gcalls * share * scale
            r["shaped_ms"] += ns * scale / 1e6
            if w["dtype"] and w["flops"]:
                r["dtype"] = w["dtype"]
    out = sorted(rows.values(), key=lambda r: -r["ms"])
    for r in out:
        for k in ("ms", "per_step", "launches", "bytes", "flops", "link_bytes", "shaped_ms"):
            r[k] = round(r[k], 4) if k in ("ms", "per_step", "shaped_ms") else round(r[k])
    return out, calls


def collect(cfg, out):
    """Write kernels.csv, steps.csv, step_shapes.csv and rates.yaml into out: from cfg['tables'] when given, else by
    querying cfg['run']."""
    out = Path(out)
    if cfg.get("tables"):
        src = Path(cfg["tables"])
        for name in ("kernels.csv", "steps.csv", "step_shapes.csv", "rates.yaml"):
            if (src / name).exists():
                (out / name).write_bytes((src / name).read_bytes())
        return
    rank = int(cfg.get("rank", 0))
    args = dict(steps=cfg.get("steps"), fmt=cfg.get("format"), python_path=cfg.get("python"))
    shapes = [step_shape(r) for r in sql(cfg["run"], STEPS_DETAIL_SQL.format(rank=rank), **args)]
    world = sql(cfg["run"], WORLD_SQL, **args)
    world = int(world[0]["world"]) if world else 1
    sigs = {r["sig_id"]: (json.loads(r["dims"]) if r["dims"] else None, json.loads(r["types"]) if r["types"] else None)
            for r in sql(cfg["run"], SIGS_SQL, **args)}
    detail = sql(cfg["run"], DETAIL_SQL.format(rank=rank), **args)
    kernels, calls = kernel_rows(detail, sigs, shapes, world)
    _write(out / "kernels.csv", kernels)
    by_kind = defaultdict(lambda: {"steps": 0, "traced": 0, "host_ms": 0.0})
    for s in shapes:
        k = by_kind[s["kind"]]
        k["steps"] += 1
        k["traced"] += s["traced"]
        k["host_ms"] += s["host_ms"]
    _write(out / "steps.csv", [{"kind": kind, **v} for kind, v in sorted(by_kind.items(), key=lambda kv: -kv[1]["host_ms"])])
    _write(out / "step_shapes.csv", [{c: s[c] for c in SHAPE_COLS} for s in shapes])
    kfloor = {r["kind"]: round(float(r["us"]), 2) for r in sql(cfg["run"], KERNEL_FLOOR_SQL.format(rank=rank), **args)}
    r = floors.rates(calls, kfloor)
    r["world"] = world
    (out / "rates.yaml").write_text(yaml.safe_dump(r, sort_keys=False))


def compile_rules(rules):
    keys = ("kind", "scope", "op", "category", "kernel")
    return [(r["bucket"], {k: re.compile(r[k]) for k in keys if r.get(k)}) for r in rules]


def classify(row, rules):
    fields = {"kind": row["kind"], "scope": row["scope"], "op": row["op"], "category": row.get("category", ""),
              "kernel": row["name"]}
    for bucket, pats in rules:
        if all(p.search(fields[k]) for k, p in pats.items()):
            return bucket
    return None


def row_floor(k, rates):
    """(floor ms, bound) of one kernels.csv row at the profile's rates. The part with no recorded shapes keeps its
    measured time: no floor, no claim."""
    ms = float(k["ms"])
    shaped = min(ms, float(k.get("shaped_ms") or 0))
    if not rates or shaped <= 0 or ms <= 0:
        return ms, "none"
    f, bound = floors.floor_ms(float(k["bytes"]), float(k["flops"]), k.get("dtype") or None, float(k["link_bytes"]),
                               float(k["launches"]) * shaped / ms, rates, k["kind"])
    return min(f, shaped) + ms - shaped, bound


def weigh(kernels, steps, rules, busy=None, host_bucket="Host gaps", top=5, rates=None):
    """Each bucket's share of measured host step time, and with rates its floor: the time left at the profile's own
    bandwidth, compute, link and launch rates.

    Per step kind, kernel time beyond the kind's host time (tracing adds time per kernel) is scaled down to
    host time × busy, the GPU-busy fraction measured for that kind (1.0 when not given)."""
    busy = busy or {}
    rules = compile_rules(rules)
    host = {s["kind"]: float(s["host_ms"]) for s in steps}
    total_host = sum(host.values())
    kernel_ms = defaultdict(float)
    for k in kernels:
        kernel_ms[k["kind"]] += float(k["ms"])
    kinds = {}
    for kind, h in host.items():
        km = kernel_ms.get(kind, 0.0)
        scale = min(1.0, h * float(busy.get(kind, 1.0)) / km) if km else 1.0
        kinds[kind] = {"host_ms": round(h, 1), "kernel_ms": round(km, 1), "scale": round(scale, 4),
                       "gaps_ms": round(h - km * scale, 1)}

    ms = defaultdict(float)
    floor = defaultdict(float)
    shaped = defaultdict(float)
    bounds = defaultdict(lambda: defaultdict(float))
    by_kind = defaultdict(lambda: defaultdict(float))
    head_kind = defaultdict(lambda: defaultdict(float))
    entries = defaultdict(lambda: defaultdict(float))
    for k in kernels:
        b = classify(k, rules) or "Other"
        sc = kinds.get(k["kind"], {"scale": 1.0})["scale"]
        v = float(k["ms"]) * sc
        f, bound = row_floor(k, rates)
        ms[b] += v
        floor[b] += f * sc
        shaped[b] += min(float(k.get("shaped_ms") or 0), float(k["ms"])) * sc
        bounds[b][bound] += v
        by_kind[b][k["kind"]] += v
        head_kind[b][k["kind"]] += (float(k["ms"]) - f) * sc
        entries[b][(k["scope"], k["op"], k["name"][:120])] += v
    for kind, d in kinds.items():
        ms[host_bucket] += d["gaps_ms"]
        floor[host_bucket] += d["gaps_ms"]
        by_kind[host_bucket][kind] += d["gaps_ms"]

    buckets = {}
    for b in sorted(ms, key=lambda b: -ms[b]):
        top_entries = sorted(entries[b].items(), key=lambda kv: -kv[1])[:top]
        buckets[b] = {
            "ms": round(ms[b], 1), "share": ms[b] / total_host,
            "by_kind": {kind: by_kind[b][kind] / host[kind] for kind in host if by_kind[b].get(kind)},
            **({"floor_ms": round(floor[b], 1), "shaped": shaped[b] / ms[b] if ms[b] else 0.0, "headroom": 1 - floor[b] / ms[b] if ms[b] else 0.0,
                "headroom_by_kind": {kind: head_kind[b][kind] / host[kind] for kind in host if head_kind[b].get(kind)},
                "bound": max(bounds[b], key=bounds[b].get) if bounds[b] else "none"} if rates else {}),
            "top": [{"scope": s, "op": o, "kernel": n, "ms": round(v, 1)} for (s, o, n), v in top_entries],
        }
    return {"host_ms": round(total_host, 1), "kinds": kinds, "buckets": buckets}


def kernel_names(kernels):
    return {k["name"] for k in kernels}
