import pytest

from modelwiki import plan as P
from modelwiki import scheduler as S
from modelwiki import stepmodel

BUDGET = 8192
WL = {"concurrency": 10, "osl": 300, "isl": 20000, "prefix_hit": 0.9,
      "speculative": {"drafts": 5, "acceptance_length": 3.0}, "metric": "output_tok_s_gpu"}


def _step(i, kind, pf_tok=0, pf_ctx=0, dec_reqs=10, ms=None):
    rows = dec_reqs * 6
    if ms is None:
        ms = 40 + 0.5 * rows if kind == "decode" else 100 + 0.01 * pf_tok + 0.001 * pf_ctx + 0.5 * rows
    return {"step": i, "kind": kind, "traced": False, "host_ms": ms, "rows": rows + pf_tok, "pf_reqs": int(pf_tok > 0),
            "pf_tok": pf_tok, "pf_ctx": pf_ctx, "pf_attn": pf_tok * pf_ctx, "dec_reqs": dec_reqs, "dec_rows": rows,
            "dec_ctx": dec_reqs * 20000}


def _profile():
    steps, i = [], 0
    for n in range(120):
        for _ in range(3):
            steps.append(_step(i, "decode", dec_reqs=8 + n % 5))
            i += 1
        tok = 1500 + 100 * (n % 7)
        steps.append(_step(i, "mixed", tok, 20000 + tok, dec_reqs=8 + n % 5))
        i += 1
    chunk = BUDGET - 60
    for j in range(3):
        steps.append(_step(i, "mixed", chunk, chunk * (j + 1)))
        i += 1
    steps.append(_step(i, "mixed", 2000, 3 * chunk + 2000))
    return steps


def _terms(steps=None):
    steps = steps or _profile()
    models = stepmodel.fits(steps)
    return S.terms(steps, [s["host_ms"] for s in steps], models, WL, {"decode": 400}, {"decode": 2.0}, 4.0)


def test_prompts_continue_across_full_steps():
    ps = S.prompts(_profile(), BUDGET)
    assert len(ps) == 121
    assert ps[-1]["tokens"] == pytest.approx(3 * (BUDGET - 60) + 2000)
    assert ps[-1]["ctx"] == pytest.approx(0.0)


def test_terms_read_the_profile():
    T = _terms()
    assert T["rows_per_request"] == pytest.approx(6.0)
    assert T["budget"] == pytest.approx(BUDGET)
    assert T["decode_fit"]
    assert S._decode_ms(T, 120, "mid") - S._decode_ms(T, 60, "mid") == pytest.approx(30.0, rel=1e-3)
    assert T["decode_floor_ms"] == pytest.approx(400 * 4.0 / 1e3 + 2.0)


def test_the_profiled_point_gives_the_bench_and_a_cap_trades_throughput_for_interactivity():
    T = _terms()
    bench = {"output_tok_s_gpu": 100.0, "p50_interactivity": 50.0, "p90_interactivity": 30.0, "p90_ttft_s": 2.0}
    sc = S.sweep(T, bench, WL, 4, {}, P.requirements_failed, caps=(2, 6))
    pts = {(p["cap"], p["budget"]): p for p in sc["points"]["now"]}
    base = pts[(None, BUDGET)]
    assert base["metrics"]["output_tok_s_gpu"] == pytest.approx(100.0)
    assert base["metrics"]["p50_interactivity"] == pytest.approx(50.0)
    assert base["metrics"]["p90_ttft_s"] == pytest.approx(2.0)
    capped = pts[(2, BUDGET)]
    assert capped["metrics"]["p50_interactivity"] > 50.0
    assert capped["metrics"]["output_tok_s_gpu"] < 100.0
    assert capped["waiting"] > 0 and capped["metrics"]["p90_ttft_s"] > 2.0


def test_a_larger_budget_takes_fewer_prompt_steps_and_the_plan_cut_speeds_the_step():
    T = _terms()
    loop = (T["decoding"], 0.0)
    small = S.point(T, None, 2048, {}, loop, 4)
    big = S.point(T, None, BUDGET, {}, loop, 4)
    assert small["prompt_share"] > big["prompt_share"]
    cut = S.point(T, None, BUDGET, {"decode": 0.5, "prompt": 0.5}, loop, 4)
    assert cut["step_ms"] == pytest.approx(big["step_ms"] * 0.5, rel=0.02)


def test_requirements_pick_the_cap_and_set_the_step_budget():
    T = _terms()
    bench = {"output_tok_s_gpu": 100.0, "p50_interactivity": 50.0}
    wl = dict(WL, requirements={"p50_interactivity": 55.0})
    sc = S.sweep(T, bench, wl, 4, {}, P.requirements_failed, caps=(2, 4, 6, 8))
    assert sc["now"]["cap"] is not None and not sc["now"]["fails"]
    assert sc["now"]["metrics"]["p50_interactivity"] >= 55.0
    cal = sc["calibration"]["p50_interactivity"]
    assert sc["need"]["step_ms"] == pytest.approx(3.0 * 1e3 * cal / 55.0)
