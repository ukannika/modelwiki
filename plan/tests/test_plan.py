import json
import subprocess
from pathlib import Path

import pytest
import yaml

from modelwiki import cli
from modelwiki import plan as P


def _tables(d, kernels, host_ms=100.0, kind="decode"):
    """Ruppell tables as the plan step reads them. kernels: (scope, op, name, ms)."""
    d.mkdir(parents=True, exist_ok=True)
    rows = ["kind,scope,op,category,name,ms,per_step"]
    rows += [f"{kind},{sc},{op},x,{n},{ms},1" for sc, op, n, ms in kernels]
    (d / "kernels.csv").write_text("\n".join(rows) + "\n")
    (d / "steps.csv").write_text(f"kind,steps,traced,host_ms\n{kind},10,10,{host_ms}\n")
    return d


RULES = {"rules": [{"bucket": "Communication", "op": "all_reduce"},
                   {"bucket": "Residual mixing", "scope": r"layers\.\*$"},
                   {"bucket": "Routed experts", "scope": r"ffn\.experts$"}]}


@pytest.fixture
def plan_dir(root, tree, tmp_path):
    # 100 ms of host time: 30 mix, 40 experts, 10 all-reduce, 5 unknown, 15 idle.
    _tables(tmp_path / "prof", [("model.layers.*", "aten::add", "mhc_pre_kernel", 30),
                                ("model.layers.*.ffn.experts", "aiter::fused_moe", "moe_gemm1", 40),
                                ("model.layers.*.ffn.experts", "vllm::all_reduce", "allreduce_1stage", 10),
                                ("runner", "aten::fill_", "mystery", 5)])
    req = {"model": "dsv41-flash", "tree": str(tree), "trace_commit": "HEAD", "ruppell": {"tables": "prof"},
           "bench": {"output_tok_s_gpu": 100.0, "p90_interactivity": 20.0}}
    (tmp_path / "request.yaml").write_text(yaml.safe_dump(req))
    return P.prepare(tmp_path / "request.yaml", root)


def _set(pdir, **rows):
    data = json.loads((pdir / "features.json").read_text())
    for r in data:
        r.update(rows.get(r["id"].replace("-", "_"), {"status": "n/a"}))
    (pdir / "features.json").write_text(json.dumps(data))


def test_prepare_writes_skeletons(plan_dir):
    rows = json.loads((plan_dir / "features.json").read_text())
    by = {r["id"]: r for r in rows}
    assert by["m-mix-seam"]["buckets"] == ["Residual mixing", "Communication"]
    assert by["s-compilation-compile"]["buckets"] == ["Host gaps"]
    assert "Residual mixing" in (plan_dir / "bucket_map.yaml").read_text()
    assert (plan_dir / "kernels.csv").exists() and (plan_dir / "steps.csv").exists()
    assert (plan_dir / "brief.md").read_text().startswith("# Plan brief: DeepSeek-V4.1-Flash")


def test_trace_from_another_commit_is_refused(root, tree, tmp_path):
    first = P.resolve_commit(tree)
    (tree / "model/new.py").write_text("x\n")
    subprocess.run(["git", "-C", str(tree), "add", "."], check=True)
    subprocess.run(["git", "-C", str(tree), "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "2"],
                   check=True)
    _tables(tmp_path / "prof", [("s", "o", "k", 1)])
    req = {"model": "dsv41-flash", "tree": str(tree), "trace_commit": first, "ruppell": {"tables": "prof"}}
    (tmp_path / "req.yaml").write_text(yaml.safe_dump(req))
    with pytest.raises(ValueError, match="profile this commit"):
        P.prepare(tmp_path / "req.yaml", root)


def test_weigh_shares_and_host_gaps(plan_dir):
    (plan_dir / "bucket_map.yaml").write_text(yaml.safe_dump(RULES))
    w = P.run_weigh(plan_dir)["buckets"]
    assert w["Routed experts"]["share"] == pytest.approx(0.40)
    assert w["Residual mixing"]["share"] == pytest.approx(0.30)
    assert w["Other"]["share"] == pytest.approx(0.05)
    assert w["Host gaps"]["share"] == pytest.approx(0.15)


def test_check_then_project(plan_dir, root):
    (plan_dir / "bucket_map.yaml").write_text(yaml.safe_dump(RULES))
    P.run_weigh(plan_dir)
    bad = dict(
        m_mix_seam={"status": "on", "evidence": ["model/seam.py:99"], "kernels": ["^fused_seam$"]},
        m_moe_routed_experts={"status": "port", "evidence": [], "change": {"kind": "estimate", "fraction": 2, "basis": "guess"}},
    )
    _set(plan_dir, **bad)
    errors = "\n".join(P.check(plan_dir)["errors"])
    assert "past the end of the file" in errors
    assert "/^fused_seam$/ is not in the trace" in errors
    assert "port needs file:line evidence" in errors
    assert "outside 0..1" in errors and "basis 'guess'" in errors

    good = dict(
        m_mix_seam={"status": "partial", "evidence": ["model/seam.py:3-9"], "kernels": ["^mhc_"],
                    "buckets": ["Residual mixing"], "change": {"kind": "estimate", "fraction": 0.5, "basis": "budget", "why": "2 of 4"}},
        m_moe_routed_experts={"status": "port", "evidence": ["model/moe.py:2"], "depends_on": ["m-moe-route"],
                              "change": {"kind": "estimate", "fraction": 0.5, "basis": "reference", "why": "half of peak"}},
        m_moe_route={"status": "unwired", "evidence": ["model/moe.py:5"], "buckets": ["Routed experts"],
                     "change": {"kind": "estimate", "fraction": 0.02, "basis": "estimate", "why": "small"}},
        m_draft_cache_type={"status": "off-by-config", "evidence": ["model/moe.py:1"], "launch_blocker": True,
                            "config": {"kv_cache_dtype": "fp8"}},
        s_compilation_compile={"status": "off-by-config", "evidence": ["model/seam.py:1"],
                               "change": {"kind": "host", "fraction": 0.5, "why": "half the gaps"}},
    )
    _set(plan_dir, **good)
    assert P.check(plan_dir)["errors"] == []

    plan = P.project(plan_dir, root)
    order = [r["id"] for r in plan["stages"]["features"]]
    # The launch blocker first; route (a dependency) before experts despite a smaller cut.
    assert order[0] == "m-draft-cache-type"
    assert order.index("m-moe-route") < order.index("m-moe-routed-experts")
    by = {r["id"]: r for r in plan["rows"]}
    assert by["m-moe-routed-experts"]["cut"] == pytest.approx(0.20)
    assert by["m-mix-seam"]["cut"] == pytest.approx(0.15)
    assert [r["id"] for r in plan["stages"]["serving"]] == ["s-compilation-compile"]
    # Experts (50%) and route (2%) share the experts bucket (40%): together they leave 0.5 x 0.98 of it.
    total = 0.15 + 0.40 * (1 - 0.5 * 0.98) + 0.075
    assert plan["projected"]["step_cut"] == pytest.approx(total)
    assert plan["projected"]["planned"]["output_tok_s_gpu"] == pytest.approx(100 / (1 - total))
    assert plan["projected"]["planned"]["p90_interactivity"] == pytest.approx(20 / (1 - total))
    assert plan["config"] == {"m-draft-cache-type": {"kv_cache_dtype": "fp8"}}
    md = (plan_dir / "projection.md").read_text()
    assert "Subfloor bundle" in md and "`m-moe-route`" in md


def test_plain_directory_tree(root, tmp_path):
    t = tmp_path / "plain"
    t.mkdir()
    (t / "a.py").write_text("1\n2\n3\n")
    _tables(tmp_path / "prof", [("s", "o", "k", 1)])
    req = {"model": "dsv41-flash", "tree": str(t), "commit": "iter17", "trace_commit": "iter17",
           "ruppell": {"tables": "prof"}}
    (tmp_path / "req.yaml").write_text(yaml.safe_dump(req))
    out = P.prepare(tmp_path / "req.yaml", root)
    assert out == root / "plans/dsv41-flash/iter17"
    assert P.file_lines(t, "iter17", "a.py") == 3


def test_traced_kernel_time_is_scaled_to_host_time(tmp_path):
    from modelwiki import ruppell
    d = _tables(tmp_path / "p", [("model.layers.*", "aten::add", "mhc", 120)], host_ms=100.0)
    w = ruppell.weigh(ruppell.read(d / "kernels.csv"), ruppell.read(d / "steps.csv"), RULES["rules"],
                      busy={"decode": 0.9})
    assert w["kinds"]["decode"]["scale"] == pytest.approx(0.75)
    assert w["buckets"]["Residual mixing"]["share"] == pytest.approx(0.90)
    assert w["buckets"]["Host gaps"]["share"] == pytest.approx(0.10)


def test_prepare_takes_a_dict(root, tree, tmp_path):
    prof = _tables(tmp_path / "prof", [("s", "o", "k", 1)])
    out = P.prepare({"model": "dsv41-flash", "tree": str(tree), "trace_commit": "HEAD",
                     "ruppell": {"tables": str(prof)}}, root)
    assert out == P.plan_dir("dsv41-flash", P.resolve_commit(tree), root)
    brief = (out / "brief.md").read_text()
    assert "buckets.md" in brief and "serving.md" in brief


def _argv(tree, prof, *more):
    return ["plan", "dsv41-flash", "--tree", str(tree), "--tables", str(prof),
            "--bench", "output_tok_s_gpu=100", "--out", str(prof.parent / "handoff"), *more]


def test_plan_command_stops_without_an_agent_then_resumes(root, tree, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", root)
    prof = _tables(tmp_path / "prof", [("model.layers.*", "aten::add", "mhc_pre_kernel", 30)])
    pdir = tmp_path / "work"
    with pytest.raises(SystemExit, match="--work"):
        cli.main(_argv(tree, prof, "--agent", "none"))
    assert cli.main(_argv(tree, prof, "--agent", "none", "--work", str(pdir))) == 0
    assert "No plan agent here" in capsys.readouterr().out
    assert not (pdir / "plan.md").exists() and not (tmp_path / "handoff").exists()

    (pdir / "bucket_map.yaml").write_text(yaml.safe_dump(RULES))
    _set(pdir, m_mix_seam={"status": "partial", "evidence": ["model/seam.py:3"], "kernels": ["^mhc_"],
                           "change": {"kind": "estimate", "fraction": 0.5, "basis": "budget", "why": "2 of 4"}})
    assert cli.main(_argv(tree, prof, "--agent", "none", "--work", str(pdir))) == 0
    assert str(tmp_path / "handoff" / "SKILL.md") in capsys.readouterr().out
    assert (tmp_path / "handoff" / "plan.md").read_text() == (pdir / "plan.md").read_text()
    assert json.loads((pdir / "plan.json").read_text())["projected"]["step_cut"] == pytest.approx(0.15)


def test_plan_command_runs_the_agent(root, tree, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", root)
    prof = _tables(tmp_path / "prof", [("model.layers.*", "aten::add", "mhc_pre_kernel", 30)])
    agent = tmp_path / "agent.py"
    agent.write_text(
        "import json, sys, yaml\n"
        "assert 'brief.md' in sys.argv[1]\n"
        f"yaml.safe_dump({RULES!r}, open('bucket_map.yaml', 'w'))\n"
        "rows = json.load(open('features.json'))\n"
        "for r in rows: r['status'] = 'n/a'\n"
        "json.dump(rows, open('features.json', 'w'))\n")
    assert cli.main(_argv(tree, prof, "--agent-cmd", f"python3 {agent} {{prompt}}")) == 0
    out = capsys.readouterr().out
    assert "0 errors" in out and out.strip().endswith("SKILL.md")
    work = Path(out.split("plan directory: ")[1].split("\n")[0])
    assert not work.exists() and (tmp_path / "handoff" / "plan.md").exists()


def test_workload_from_flags_and_target_changes(root, tree, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", root)
    prof = _tables(tmp_path / "prof", [("model.layers.*", "aten::add", "mhc_pre_kernel", 30)])
    flags = ["--concurrency", "32", "--isl", "100000", "--osl", "500", "--prefix-hit", "0.9",
             "--drafts", "5", "--acceptance", "3.5", "--require", "p90_ttft_s=5", "--set", "metric=p50_interactivity",
             "--target", "concurrency=64,osl=800", "--target", "active_decodes=[40,60],osl=900",
             "--active-decodes", "39-46", "--agent", "none", "--work", str(tmp_path / "work")]
    assert cli.main(_argv(tree, prof, *flags)) == 0
    req = yaml.safe_load((tmp_path / "work" / "request.yaml").read_text())
    wl, target = req["workload"], req["targets"][0]
    assert wl["name"] == "c32" and wl["isl"] == 100000 and wl["metric"] == "p50_interactivity"
    assert wl["speculative"] == {"drafts": 5, "acceptance_length": 3.5} and wl["requirements"] == {"p90_ttft_s": 5}
    assert target["concurrency"] == 64 and target["osl"] == 800 and target["isl"] == wl["isl"]
    assert wl["active_decodes"] == [39, 46] and "active_decodes" not in target
    assert req["targets"][1]["active_decodes"] == [40, 60] and req["targets"][1]["osl"] == 900


def test_a_yaml_workload_takes_overrides_and_must_be_complete(root, tree, tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(cli, "ROOT", root)
    prof = _tables(tmp_path / "prof", [("s", "o", "k", 1)])
    assert cli.main(_argv(tree, prof, "--workload", "agentx-c32", "--prefix-hit", "0.5", "--agent", "none", "--work", str(tmp_path / "work"))) == 0
    req = yaml.safe_load((tmp_path / "work" / "request.yaml").read_text())
    assert req["workload"]["prefix_hit"] == 0.5 and req["workload"]["osl"] == 472
    assert req["workload"]["name"] == "agentx-c32 (prefix_hit set)"
    with pytest.raises(ValueError, match=r"needs osl \(--osl\).*prefix_hit"):
        cli.main(_argv(tree, prof, "--concurrency", "8", "--isl", "1000", "--agent", "none", "--work", str(tmp_path / "work")))
