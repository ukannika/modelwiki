import json

import pytest
import yaml

from modelwiki import cli, floors, ruppell, stepmodel
from modelwiki import plan as P

RATES = {"source": "test", "hbm_tb_s": 1.0, "link_gb_s": 100.0, "tflops": {"bf16": 100.0}, "kernel_floor_us": 0.0,
         "world": 4}
WORKLOAD = {"name": "w", "concurrency": 16, "isl": 1000, "osl": 40,
            "prefix_hit": 0.5, "speculative": {"drafts": 1, "acceptance_length": 2.0},
            "requirements": {"p90_interactivity": "not_worse"}}


def test_op_work_from_shapes():
    bf = "c10::BFloat16"
    g = floors.op_work("aten::mm", [[64, 128], [128, 256]], [bf, bf])
    assert g["flops"] == 2 * 64 * 128 * 256 and g["bytes"] == 2 * (64 * 128 + 128 * 256 + 64 * 256)
    add = floors.op_work("aten::add", [[10, 10], [10, 10], []], [bf, bf, "Scalar"])
    assert add["bytes"] == 3 * 200
    assert floors.op_work("aten::add_", [[10, 10], [10, 10]], [bf, bf])["bytes"] == 2 * 200
    ar = floors.op_work("vllm::all_reduce", [[1000]], [bf], world=4)
    assert ar["link_bytes"] == pytest.approx(2000 * 2 * 3 / 4)
    moe = floors.op_work("aiter::fused_moe_", [[8, 64], [4, 256, 32], [4, 64, 64], [8, 2]],
                         [bf, "c10::Float4_e2m1fn_x2", "c10::Float4_e2m1fn_x2", "int"])
    assert moe["flops"] == 2 * 8 * 2 * (256 * 64 + 64 * 64 * 2) and moe["dtype"] == "mxfp4"
    assert floors.op_work("DeepseekAttention", None, None) is None


def test_rates_come_from_the_profile():
    big = {"kind": "elementwise", "bytes": 1e9, "flops": 0, "dtype": None}
    small = {"kind": "gemm", "bytes": 1e6, "flops": 1e9, "dtype": "bf16"}
    r = floors.rates([(big, 1e9 / 5000, 10), (small, 1e9 / 200e3, 5)], {"decode": 4.0, "mixed": 2.0})
    assert r["hbm_tb_s"] == pytest.approx(5.0) and r["tflops"] == {"bf16": pytest.approx(200.0)}
    assert "low bound" in r["source"]["tflops.bf16"] and r["link_gb_s"] is None
    f, bound = floors.floor_ms(2e9, 1e12, "bf16", 0, 10, r, "decode")
    assert bound == "compute" and f == pytest.approx(1e12 / 200e12 * 1e3)
    assert floors.floor_ms(0, 1e12, "fp8", 0, 1000, r, "mixed") == (pytest.approx(2.0), "launch")


def test_step_shape_reads_the_annotation():
    ann = "execute_300_context_1(sq60sk5000sqsq3600sqsk150000)_generation_40(sq240sk800000sqsq0sqsk0)"
    s = ruppell.step_shape({"step": "7", "kind": "mixed", "traced": "True", "host_ns": "2000000",
                            "fields": json.dumps({"rows": 300, "annotation": ann})})
    assert (s["pf_tok"], s["pf_ctx"], s["pf_attn"], s["dec_reqs"], s["dec_rows"]) == (60, 5000, 150000, 40, 240)
    assert s["host_ms"] == 2.0 and s["traced"]


def test_kernel_rows_spread_an_ops_work_over_its_kernels():
    bf = "c10::BFloat16"
    detail = [{"kind": "decode", "scope": "a", "op": "aten::add", "category": "x", "name": n, "sig_id": "1",
               "ns": ns, "kernels": "2", "calls": "2"} for n, ns in (("k1", "300"), ("k2", "100"))]
    sigs = {"1": ([[10], [10]], [bf, bf])}
    steps = [{"kind": "decode", "traced": True}, {"kind": "decode", "traced": False}]
    rows, calls = ruppell.kernel_rows(detail, sigs, steps)
    by = {r["name"]: r for r in rows}
    # 2 calls x 60 bytes, 3:1 by time, scaled by 2 steps / 1 traced.
    assert by["k1"]["bytes"] == 180 and by["k2"]["bytes"] == 60 and by["k1"]["launches"] == 4
    assert calls[0][2] == 2


def _synthetic_steps(n_dec=200, n_mixed=40):
    out = []
    for i in range(n_dec):
        rows = 60 + (i % 40)
        out.append({"step": i, "kind": "decode", "traced": i < 10, "host_ms": 20 + 0.1 * rows, "rows": rows,
                    "pf_reqs": 0, "pf_tok": 0, "pf_ctx": 0, "pf_attn": 0, "dec_reqs": rows / 2, "dec_rows": rows,
                    "dec_ctx": rows * 500})
    for i in range(n_mixed):
        tok = 200 + 50 * (i % 10)
        out.append({"step": n_dec + i, "kind": "mixed", "traced": True, "host_ms": 40 + 0.05 * tok + 0.1 * 80,
                    "rows": tok + 80, "pf_reqs": 1, "pf_tok": tok, "pf_ctx": 1000, "pf_attn": tok * 1000,
                    "dec_reqs": 40, "dec_rows": 80, "dec_ctx": 40000})
    return out


def test_step_model_fits_and_metrics():
    steps = _synthetic_steps()
    models = stepmodel.fits(steps)
    assert models["decode"]["coef"]["dec_rows"] + 500 * models["decode"]["coef"]["dec_ctx"] == pytest.approx(0.1)
    assert models["mixed"]["r2"] > 0.99
    assert stepmodel.tracing(steps, models, {"decode": 100}) == pytest.approx(0.0, abs=1e-9)
    times = [s["host_ms"] for s in steps]
    m = stepmodel.metrics(steps, times, WORKLOAD, gpus=2)
    tokens = sum(s["dec_reqs"] for s in steps) * 2.0
    assert m["output_tok_s_gpu"] == pytest.approx(tokens / (sum(times) / 1e3) / 2)
    assert m["p90_interactivity"] <= m["p50_interactivity"]
    half = stepmodel.metrics(steps, [t / 2 for t in times], WORKLOAD, gpus=2)
    assert half["p50_interactivity"] == pytest.approx(2 * m["p50_interactivity"])


def test_retarget_scales_rows_and_flags_the_range():
    steps = _synthetic_steps()
    models = stepmodel.fits(steps)
    wl = dict(WORKLOAD, active_decodes=40)
    _, factor, past = stepmodel.retarget(steps, models, wl, dict(wl, active_decodes=80))
    assert factor[0] > 1.2 and "decode.dec_rows" in past
    _, factor, past = stepmodel.retarget(steps, models, wl, wl)
    assert factor[0] == pytest.approx(1.0) and past == []


@pytest.fixture
def shaped_plan(root, tree, tmp_path):
    """Decode kernels 600 ms of 1000 ms host per window: two mixing kernels (floor 100 ms each), an all-reduce."""
    prof = tmp_path / "shaped"
    prof.mkdir()
    head = "kind,scope,op,category,name,ms,per_step,launches,bytes,flops,link_bytes,dtype,shaped_ms"
    rows = [f"decode,model.layers.*,aten::add,x,mhc_a,250,1,200,1e11,0,0,,250",
            f"decode,model.layers.*,aten::mul,x,mhc_b,250,1,200,1e11,0,0,,250",
            f"decode,model.layers.*.ffn.experts,vllm::all_reduce,x,allreduce,100,1,200,0,0,0,,0"]
    (prof / "kernels.csv").write_text("\n".join([head, *rows]) + "\n")
    steps = _synthetic_steps(n_mixed=0)
    total = sum(s["host_ms"] for s in steps)
    (prof / "steps.csv").write_text(f"kind,steps,traced,host_ms\ndecode,{len(steps)},10,{total}\n")
    cols = ruppell.SHAPE_COLS
    (prof / "step_shapes.csv").write_text(
        ",".join(cols) + "\n" + "\n".join(",".join(str(s[c]) for c in cols) for s in steps) + "\n")
    (prof / "rates.yaml").write_text(yaml.safe_dump(RATES))
    req = {"model": "dsv41-flash", "tree": str(tree), "trace_commit": "HEAD", "ruppell": {"tables": str(prof)},
           "workload": WORKLOAD, "bench": {"output_tok_s_gpu": 50.0, "p90_interactivity": 10.0}, "gpus": 2,
           "targets": [dict(WORKLOAD, name="w2", concurrency=64), dict(WORKLOAD, name="w3", active_decodes=[30, 50])]}
    out = P.prepare(req, root)
    (out / "bucket_map.yaml").write_text(yaml.safe_dump(
        {"rules": [{"bucket": "Communication", "op": "all_reduce"},
                   {"bucket": "Residual mixing", "scope": r"layers\.\*$"}]}))
    P.run_weigh(out)
    return out, total


def _rows(pdir, **rows):
    data = json.loads((pdir / "features.json").read_text())
    for r in data:
        r.update(rows.get(r["id"].replace("-", "_"), {"status": "n/a"}))
    (pdir / "features.json").write_text(json.dumps(data))


def test_change_kinds_remove_time_from_floors(shaped_plan):
    pdir, total = shaped_plan
    ctx = P.context(pdir)
    gaps = ctx["gaps"]["decode"]
    assert gaps == pytest.approx(total - 600)
    seam = {"status": "port", "evidence": ["model/seam.py:1"], "buckets": ["Residual mixing"]}
    cases = {
        "optimize": (dict(kind="optimize"), 2 * 150),
        # One kernel left at the larger floor; 200 of 600 launches dropped take a third of the gaps.
        "fuse": (dict(kind="fuse"), 500 - 100 + gaps * 200 / 600),
        "remove_work": (dict(kind="remove_work", fraction=0.5, kernels=["mhc_a"]), 125 + gaps * 100 / 600),
        "host": (dict(kind="host", fraction=0.5), gaps / 2),
    }
    for name, (change, want) in cases.items():
        r = dict(seam, change=change)
        assert sum(P.removal(r, ctx).values()) == pytest.approx(want), name
    over = {"status": "port", "evidence": ["model/seam.py:1"], "buckets": ["Communication"],
            "change": {"kind": "overlap", "fraction": 1.0}}
    assert sum(P.removal(over, ctx).values()) == pytest.approx(100)


def test_project_gives_workload_metrics(shaped_plan):
    pdir, total = shaped_plan
    _rows(pdir, m_mix_seam={"status": "port", "evidence": ["model/seam.py:1"], "buckets": ["Residual mixing"],
                            "change": {"kind": "optimize", "why": "to the floor"}})
    report = P.check(pdir)
    assert report["errors"] == []
    plan = P.project(pdir)
    now, planned = plan["projected"]["now"], plan["projected"]["planned"]
    # No tracing overhead in these steps: the model reproduces the bench, then 300 ms of the window go.
    assert now["output_tok_s_gpu"] == pytest.approx(50.0)
    assert planned["output_tok_s_gpu"] == pytest.approx(50.0 * total / (total - 300))
    assert planned["p90_interactivity"] > now["p90_interactivity"] and plan["projected"]["fails"] == []
    seam = next(r for r in plan["rows"] if r["id"] == "m-mix-seam")
    assert seam["gain"] == pytest.approx(total / (total - 300) - 1)
    target, ranged = plan["projected"]["targets"]
    assert target["workload"] == "w2" and target["extrapolated"]
    # The profile's decode steps hold 30-49.5 decodes, p10-p90 31.95-47.55; concurrency 16 to 64 scales that by 4.
    assert target["active_decodes"] == pytest.approx([127.8, 159.0, 190.2])
    assert ranged["active_decodes"] == [30, 40, 50]
    lo, hi = ranged["now_range"]["output_tok_s_gpu"]
    assert lo < ranged["now"]["output_tok_s_gpu"] < hi
    md = (pdir / "projection.md").read_text()
    assert "| output_tok_s_gpu | 50.00 | 50.00 |" in md and "Outside the profile's range" in md
    assert "`w3` (30-50 active decodes)" in md

    req = yaml.safe_load((pdir / "request.yaml").read_text())
    (pdir / "request.yaml").write_text(yaml.safe_dump(dict(req, handoff=str(pdir.parent / "handoff"))))
    assert cli.main(["project", str(pdir), "--skip-check"]) == 0
    out = pdir.parent / "handoff"
    assert sorted(str(f.relative_to(out)) for f in out.rglob("*") if f.is_file()) == [
        "SKILL.md", "features.json", "how-to-work.md", "model/model.md", "model/serving.md", "plan.md",
        "workload.yaml"]
    skill = (out / "SKILL.md").read_text()
    assert skill.startswith("---\nname: ") and "## Order of work" in skill and "## Reading the numbers" in skill
    assert str(pdir) not in skill and f"--workload {out.resolve() / 'workload.yaml'}" in skill


def test_check_flags_rows_without_a_floor(shaped_plan):
    pdir, _ = shaped_plan
    _rows(pdir, s_compute_and_communication_overlap_all_reduce_path={
        "status": "partial", "evidence": ["model/seam.py:1"], "change": {"kind": "optimize"}},
        m_mix_seam={"status": "port", "evidence": ["model/seam.py:1"], "buckets": ["Residual mixing"],
                    "change": {"kind": "remove_work"}})
    report = P.check(pdir)
    assert any("needs a fraction" in e for e in report["errors"])
    assert any("no recorded shapes" in w for w in report["warnings"])


def test_check_warns_when_active_decodes_miss_the_profile(shaped_plan):
    pdir, _ = shaped_plan
    req = yaml.safe_load((pdir / "request.yaml").read_text())
    req["workload"]["active_decodes"] = [80, 100]
    (pdir / "request.yaml").write_text(yaml.safe_dump(req))
    assert any("does not hold the profile's" in w for w in P.check(pdir)["warnings"])
    req["workload"]["active_decodes"] = "30-45"
    (pdir / "request.yaml").write_text(yaml.safe_dump(req))
    assert not any("does not hold" in w for w in P.check(pdir)["warnings"])
