import argparse
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from . import ROOT, handoff
from . import plan as P
from .page import bucket_rows, load

AGENTS = {"claude": ["claude", "-p", "{prompt}"], "cursor-agent": ["cursor-agent", "-p", "{prompt}"]}


def _pairs(items, flag):
    out = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep:
            raise SystemExit(f"{flag} takes key=value, got {it!r}")
        try:
            out[k] = float(v)
        except ValueError:
            out[k] = v
    return out


def _request(a):
    if a.request:
        req = P.load_yaml(a.request)
        req.setdefault("model", a.model)
        return req
    if not a.tree or not (a.profile or a.tables):
        raise SystemExit("modelwiki plan needs --tree and --profile (or --tables), or --request")
    rcfg = {k: v for k, v in (("run", a.profile), ("tables", a.tables), ("steps", a.steps), ("rank", a.rank),
                                ("python", a.python), ("format", a.format)) if v is not None}
    if a.busy:
        rcfg["busy"] = _pairs(a.busy, "--busy")
    req = {"model": a.model, "tree": a.tree, "commit": a.commit, "trace_commit": a.profile_commit or a.commit,
           "ruppell": rcfg}
    if a.bench:
        req["bench"] = _pairs(a.bench, "--bench")
    for k, v in (("bench_file", a.bench_file), ("gpus", a.gpus)):
        if v:
            req[k] = str(Path(v).resolve()) if k == "bench_file" else v
    wl = _workload(a)
    if wl:
        req["workload"] = wl
    if a.target:
        req["targets"] = [_target(t, wl) for t in a.target]
    return req


def _decodes(v):
    lo, sep, hi = v.partition("-")
    return [float(lo), float(hi)] if sep else float(v)


def _workload(a):
    """The workload from --workload (a name or YAML) with the workload flags set on top, or None."""
    fields = [(k, v) for k, v in (("concurrency", a.concurrency), ("active_decodes", a.active_decodes and _decodes(a.active_decodes)),
                                  ("isl", a.isl), ("osl", a.osl),
                                  ("prefix_hit", a.prefix_hit), ("speculative.drafts", a.drafts),
                                  ("speculative.acceptance_length", a.acceptance), ("metric", a.metric))
              if v is not None]
    fields += [("requirements." + k, v) for k, v in _pairs(a.require, "--require").items()]
    fields += [tuple(f.split("=", 1)) for f in a.set or [] if "=" in f]
    if a.workload is None and not fields:
        return None
    base = P.load_workload(a.workload, ROOT, Path.cwd()) if a.workload else {}
    wl = P.with_fields(base, fields)
    name = base.get("name") or (f"c{wl['concurrency']:g}" if wl.get("concurrency") else "cli")
    wl["name"] = f"{name} ({', '.join(k for k, _ in fields)} set)" if base and fields else name
    return wl


def _target(t, wl):
    """A name or YAML, or key=value[,key=value] changes to the profiled workload."""
    if "=" not in t:
        return t
    if not wl:
        raise SystemExit(f"--target {t} changes the workload: give --workload or the workload flags too")
    out = P.with_fields(wl, [tuple(p.split("=", 1)) for p in re.split(r",(?![^\[]*\])", t)])
    out["name"] = t
    return out


def _summary(plan):
    p, metric = plan["projected"], plan["metric"]
    now, planned = p["now"].get(metric), p["planned"].get(metric)
    line = f"step cut {100 * p['step_cut']:.1f}%"
    if now and planned:
        line += f", {metric} {now:.1f} to {planned:.1f}"
    print(line + (f"; requirements not met: {'; '.join(p['fails'])}" if p["fails"] else ""))


def _agent(a, out):
    """The agent command for this machine, or None."""
    prompt = (f"Follow {ROOT / 'plan/SKILL.md'} to fill the plan in {out}. Start with {out / 'brief.md'}. "
              f"Stop once `modelwiki check {out}` has no errors.")
    if a.agent_cmd:
        return [w.replace("{prompt}", prompt) for w in shlex.split(a.agent_cmd)]
    names = list(AGENTS) if a.agent == "auto" else [] if a.agent == "none" else [a.agent]
    for n in names:
        if shutil.which(n):
            return [w.replace("{prompt}", prompt) for w in AGENTS[n]]
    if a.agent not in ("auto", "none"):
        raise SystemExit(f"{a.agent} is not installed: pass --agent-cmd, or --agent none")
    return None


def _filled(out):
    _, rows = P.load_plan(out)
    return all(r.get("status") for r in rows)


def _has_rules(out):
    return bool((P.load_yaml(out / "bucket_map.yaml") or {}).get("rules"))


def _report(r):
    for e in r["errors"]:
        print(f"error: {e}")
    for w in r["warnings"]:
        print(f"warning: {w}")
    print(f"{len(r['errors'])} errors, {len(r['warnings'])} warnings")


def _project(out):
    """Project the plan and write its handoff folder, the one request.yaml names."""
    _summary(P.project(out, root=ROOT))
    folder = P.load_yaml(Path(out) / "request.yaml").get("handoff")
    print(handoff.write(out, folder, root=ROOT) / "SKILL.md" if folder else Path(out) / "plan.md")


def cmd_plan(a):
    """Prepare, run the plan agent, check, project, and write the handoff folder, from scratch each time: the working
    files go to a temporary directory, removed after, unless --work keeps them."""
    if a.agent == "none" and not a.work:
        raise SystemExit("--agent none fills the plan by hand: give --work <dir> to keep it between runs")
    folder = Path(a.out).resolve()
    work = Path(a.work).resolve() if a.work else Path(tempfile.mkdtemp(prefix="modelwiki-"))
    try:
        return _plan(a, folder, work)
    finally:
        if not a.work:
            shutil.rmtree(work, ignore_errors=True)


def _plan(a, folder, work):
    req = dict(_request(a), handoff=str(folder))
    out = P.prepare(req, root=ROOT, allow_mismatch=a.allow_mismatch, out=work)
    print(f"plan directory: {out}")
    if not _filled(out):
        cmd = _agent(a, out)
        if cmd is None and not a.work:
            raise SystemExit(f"no plan agent here: install one of {', '.join(AGENTS)}, pass --agent-cmd, or fill the "
                             "plan by hand with --agent none --work <dir>")
        if cmd is None:
            print(f"\nNo plan agent here. Fill the plan by following {ROOT / 'plan/SKILL.md'},"
                  f" starting with {out / 'brief.md'}; then run this command again to write {folder}.")
            return 0
        print("agent: " + " ".join(shlex.quote(w) for w in cmd[:1]) + " ...")
        if subprocess.run(cmd, cwd=out).returncode:
            print("the agent failed", file=sys.stderr)
            return 1
        if not _filled(out):
            print(f"the agent left rows without a status in {out / 'features.json'}", file=sys.stderr)
            return 1
    if not _has_rules(out):
        print(f"{out / 'bucket_map.yaml'} has no rules: derive them as shared/profile-buckets.md says", file=sys.stderr)
        return 1
    P.run_weigh(out)
    r = P.check(out)
    _report(r)
    if r["errors"]:
        return 1
    _project(out)
    return 0


def cmd_rows(a):
    page = load(a.model, ROOT)
    print(f"{page['title']}: {page['path']}")
    print(f"\nBuckets ({len(page['buckets'])}): {page['buckets_path']}")
    for b in page["buckets"]:
        ids, serving = bucket_rows(b["rows_md"])
        print(f"  {b['name']}: {', '.join(ids + serving)}")
    print(f"\nFeature rows ({len(page['features'])})")
    for r in page["features"]:
        print(f"  {r['id']}  {r['Feature']}")
    print(f"\nServing rows ({len(page['serving'])}): {page['serving_path']}")
    for r in page["serving"]:
        print(f"  {r['subsection']}: {r['Feature']}")


def cmd_weigh(a):
    w = P.run_weigh(a.plan)
    kinds = list(w["kinds"])
    print(f"{'share':>7}  {'bucket':24s} " + " ".join(f"{k:>8s}" for k in kinds) + "  top scope / kernel")
    for b, v in w["buckets"].items():
        top = f"{v['top'][0]['scope'] or '-'} / {v['top'][0]['kernel'][:50]}" if v["top"] else ""
        per = " ".join(f"{100 * v['by_kind'].get(k, 0):7.2f}%" for k in kinds)
        head = (f" headroom {100 * v['headroom']:3.0f}% {v['bound']:7s} shapes {100 * v['shaped']:3.0f}%"
                if "headroom" in v and b != "Host gaps" else "")
        print(f"{100 * v['share']:6.2f}%  {b:24s} {per}{head}  {top}")


def cmd_check(a):
    r = P.check(a.plan)
    _report(r)
    return 1 if r["errors"] else 0


def cmd_project(a):
    if not a.skip_check and P.check(a.plan)["errors"]:
        print("check has errors: fix them, or pass --skip-check", file=sys.stderr)
        return 1
    _project(a.plan)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="modelwiki")
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("plan", help="plan a model on one served tree commit and its profile: writes plan.md")
    s.add_argument("model")
    s.add_argument("--tree", help="the served tree (git, or a plain directory with --commit as a label)")
    s.add_argument("--commit", default="HEAD", help="the tree commit to plan (default HEAD)")
    s.add_argument("--profile-commit", help="the commit the profile was taken on (default --commit)")
    s.add_argument("--profile", help="a ruppell run, or a trace with --format")
    s.add_argument("--tables", help="a directory with kernels.csv and steps.csv, instead of --profile")
    s.add_argument("--steps", help="the steady-state steps, A:B")
    s.add_argument("--rank", type=int)
    s.add_argument("--format", help="the profile's format, when not a ruppell run (kineto, ...)")
    s.add_argument("--python", help="PYTHONPATH for ruppell")
    s.add_argument("--busy", action="append", metavar="KIND=FRAC", help="GPU busy fraction of a step kind")
    s.add_argument("--workload", help="the workload the profile ran: a YAML, or a name under examples/; the flags "
                                      "below set it, or override its fields")
    s.add_argument("--concurrency", type=float, help="client concurrency")
    s.add_argument("--active-decodes", metavar="N|LO-HI",
                   help="requests decoding at once: a number or a range (default: the profile's p10-p90)")
    s.add_argument("--isl", type=float, help="mean input length, tokens")
    s.add_argument("--osl", type=float, help="mean output length, tokens")
    s.add_argument("--prefix-hit", type=float, help="share of prompt tokens served from the prefix cache")
    s.add_argument("--drafts", type=int, help="speculative draft tokens per step")
    s.add_argument("--acceptance", type=float, help="tokens accepted per request per decode step")
    s.add_argument("--metric", help="what to rank rows by (default output_tok_s_gpu)")
    s.add_argument("--require", action="append", metavar="KEY=VALUE",
                   help="a requirement: p90_interactivity=not_worse, p90_ttft_s=5, ... (repeatable)")
    s.add_argument("--set", action="append", metavar="KEY=VALUE", help="any workload field, dotted: speculative.drafts=3")
    s.add_argument("--target", action="append",
                   help="another workload to project at: a name, a YAML, or changes such as concurrency=64,osl=800 "
                        "(repeatable)")
    s.add_argument("--bench-file", help="the profiled run's bench result JSON")
    s.add_argument("--bench", action="append", metavar="KEY=VALUE",
                   help="the profiled run's metrics: output_tok_s_gpu, p50_interactivity, p90_interactivity, "
                        "p90_ttft_s, ... (override --bench-file)")
    s.add_argument("--gpus", type=int, help="GPUs serving the model (default: the profile's collective world)")
    s.add_argument("--request", help="a request YAML instead of the flags")
    s.add_argument("--out", required=True,
                   help="the handoff folder: the plan, the model's pages and a SKILL.md for the agent that works it")
    s.add_argument("--work", help="keep the plan's working files (profile tables, audit, projection) here; run again "
                                  "with it to resume (default: a temporary directory, removed after)")
    s.add_argument("--allow-mismatch", action="store_true", help="accept a profile from another commit")
    s.add_argument("--agent", default="auto", choices=["auto", *AGENTS, "none"])
    s.add_argument("--agent-cmd", help="the agent command, with {prompt} where the prompt goes")
    s.set_defaults(fn=cmd_plan)

    s = sub.add_parser("rows", help="the buckets and rows a plan reads for a model")
    s.add_argument("model")
    s.set_defaults(fn=cmd_rows)
    for name, fn, hlp in (("weigh", cmd_weigh, "bucket shares from the profile tables and bucket_map.yaml"),
                          ("check", cmd_check, "verify features.json against the tree and the profile")):
        s = sub.add_parser(name, help=hlp)
        s.add_argument("plan")
        s.set_defaults(fn=fn)
    s = sub.add_parser("project", help="projections, order and config: plan.json, plan.md and the handoff folder")
    s.add_argument("plan")
    s.add_argument("--skip-check", action="store_true")
    s.set_defaults(fn=cmd_project)

    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
