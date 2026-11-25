"""The running-sequence cap and the step token budget, projected from the profile's own steps in closed form.

With k requests decoding and the cap kept full, a request finishes every OSL / A steps (A: tokens accepted a step)
and brings g prompt steps, so a share f = k A g / OSL of steps carry prefill work. The mean step is
t_dec(k) + f (t_prompt(k) - t_dec(k)); interactivity is A over it and throughput k A over it per GPU, each scaled so
the profiled point gives the bench. Clients are a closed loop: requests past the cap wait, and the wait adds to TTFT.
"""
from .stepmodel import length, predict

CAPS = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256)
BUDGETS = (2048, 4096, 8192, 16384, 32768, 65536)
NEAR_FULL = 0.97
FIT_R2 = 0.3
SCALED = ("output_tok_s_gpu", "mean_interactivity", "p50_interactivity", "p90_interactivity")


def _accept(wl):
    return float((wl.get("speculative") or {}).get("acceptance_length") or 1.0)


def _mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def prompts(steps, budget):
    """Each prompt's new tokens and the context before it, rebuilt from the prompt steps in order: a step that fills
    the budget with one prompt continues it in the next prompt step."""
    out, run = [], None
    for s in steps:
        if s["kind"] == "decode" or s["pf_tok"] <= 0:
            continue
        n = max(1, int(s["pf_reqs"] or 1))
        full = n == 1 and s["pf_tok"] + s["dec_rows"] >= NEAR_FULL * budget
        if run is not None:
            run["tokens"] += s["pf_tok"]
            if not full:
                out.append(run)
                run = None
            continue
        tok, end = s["pf_tok"] / n, s["pf_ctx"] / n
        p = {"tokens": tok, "ctx": max(0.0, end - tok)}
        if full:
            run = p
        else:
            out += [dict(p) for _ in range(n)]
    if run:
        out.append(run)
    return out


def _prompt_cost(model, ps, chunk, rows):
    """Prompt steps and their modeled time for the prompts ps, chunked at chunk tokens beside rows decode rows."""
    n_steps, ms = 0, 0.0
    for p in ps:
        done = 0.0
        while done < p["tokens"]:
            s = min(chunk, p["tokens"] - done)
            done += s
            end = p["ctx"] + done
            n_steps += 1
            ms += predict(model, {"pf_tok": s, "pf_ctx": end, "pf_attn": s * end, "dec_rows": rows})
    return n_steps, ms


def terms(steps, times, models, workload, launches_per_step, gaps_per_step, floor_us):
    """What the closed form reads from the profile."""
    acc, osl = _accept(workload), length(workload.get("osl"))
    dec = [(s, t) for s, t in zip(steps, times) if s["kind"] == "decode"]
    pr = [(s, t) for s, t in zip(steps, times) if s["kind"] != "decode"]
    if not dec or not osl:
        return None
    reqs = sum(s["dec_reqs"] for s in steps)
    completions = reqs * acc / osl
    budget0 = max((s["pf_tok"] + s["dec_rows"] for s, _ in pr), default=0.0)
    fu = floor_us.get("decode", 0.0) if isinstance(floor_us, dict) else float(floor_us or 0)
    m_dec = models.get("decode")
    rows = sum(s["dec_rows"] for s in steps)
    return {
        "accept": acc, "osl": osl, "decoding": reqs / len(steps),
        "rows_per_request": sum(s["dec_rows"] for s in steps) / reqs if reqs else 1.0,
        "decode_ms": _mean([t for _, t in dec]), "decode_rows": _mean([s["dec_rows"] for s, _ in dec]),
        "prompt_ms": _mean([t for _, t in pr]), "prompt_rows": _mean([s["dec_rows"] for s, _ in pr]),
        "prompt_steps_per_request": len(pr) / completions if completions else 0.0,
        "budget": budget0, "ctx_per_row": sum(s["dec_ctx"] for s in steps) / rows if rows else 0.0,
        "decode_model": m_dec if m_dec and m_dec["r2"] >= FIT_R2 and any(m_dec["coef"].values()) else None,
        "decode_fit": bool(m_dec and m_dec["r2"] >= FIT_R2 and any(m_dec["coef"].values())),
        "decode_floor_ms": launches_per_step.get("decode", 0.0) * fu / 1e3 + gaps_per_step.get("decode", 0.0),
        "kernel_floor_us": fu, "prompt_model": models.get("mixed") or models.get("prefill"),
        "prompts": prompts([s for s, _ in pr], budget0) if budget0 else [],
    }


def _chunking(T, budget, rows):
    """Prompt steps per request and the mean prompt step against the profile's, at another step token budget."""
    m, ps = T["prompt_model"], T["prompts"]
    if not m or not ps or not T["budget"]:
        return 1.0, 1.0
    n0, ms0 = _prompt_cost(m, ps, max(1.0, T["budget"] - T["prompt_rows"]), T["prompt_rows"])
    n1, ms1 = _prompt_cost(m, ps, max(1.0, budget - rows), T["prompt_rows"])
    if not n0 or not n1 or ms0 <= 0:
        return 1.0, 1.0
    return n1 / n0, (ms1 / n1) / (ms0 / n0)


def _rows_delta(T, model, r0, r1):
    """What a fit adds going from r0 to r1 decode rows, their context moving with them."""
    if not model:
        return 0.0
    c = T["ctx_per_row"]
    return predict(model, {"dec_rows": r1, "dec_ctx": c * r1}) - predict(model, {"dec_rows": r0, "dec_ctx": c * r0})


def _decode_ms(T, rows, end):
    """Decode step at rows: the fit when the profile varies rows; otherwise between flat and everything above the
    launch floor scaling with rows (end: lo, mid or hi)."""
    t, r0, fl = T["decode_ms"], T["decode_rows"] or 1.0, min(T["decode_floor_ms"], T["decode_ms"])
    if T["decode_model"]:
        return max(fl, t + _rows_delta(T, T["decode_model"], r0, rows))
    flat, prop = t, fl + (t - fl) * rows / r0
    lo, hi = min(flat, prop), max(flat, prop)
    return {"lo": lo, "hi": hi}.get(end, (lo + hi) / 2)


def point(T, cap, budget, cut, loop, gpus, end="mid"):
    """Modeled step and metrics at a cap (None: no cap) and a step token budget, with cut[kind] the share of a step
    kind's time the plan removes. loop is the closed loop: population and think time."""
    acc, osl, rpr = T["accept"], T["osl"], T["rows_per_request"]
    pop, think = loop

    def at(k):
        rows = k * rpr
        g_ratio, p_ratio = _chunking(T, budget, rows)
        td = _decode_ms(T, rows, end) * (1 - cut.get("decode", 0.0))
        tp = (T["prompt_ms"] + _rows_delta(T, T["prompt_model"], T["prompt_rows"], rows)) * p_ratio
        tp = max(td, tp * (1 - cut.get("prompt", 0.0)))
        f = min(1.0, k * acc * T["prompt_steps_per_request"] * g_ratio / osl)
        t = td + f * (tp - td)
        return t, td, tp, f

    n = pop
    for _ in range(100):
        k = min(cap, n) if cap else n
        t, *_ = at(k)
        rate = k * acc / (t * osl) * 1e3
        nxt = max(1.0, pop - rate * think)
        if abs(nxt - n) < 1e-6:
            break
        n = 0.5 * (n + nxt)
    k = min(cap, n) if cap else n
    t, td, tp, f = at(k)
    rate = k * acc / (t * osl) * 1e3
    return {"cap": cap, "budget": budget, "decoding": k, "waiting": max(0.0, n - k),
            "queue_s": max(0.0, n - k) / rate if rate else 0.0, "step_ms": t, "decode_ms": td, "prompt_ms": tp,
            "prompt_share": f, "output_tok_s_gpu": k * acc / t * 1e3 / gpus, "interactivity": acc / t * 1e3}


def _metrics(p, cal):
    out = {"output_tok_s_gpu": p["output_tok_s_gpu"] * cal.get("output_tok_s_gpu", 1.0)}
    for key in SCALED[1:]:
        out[key] = p["interactivity"] * cal.get(key, 1.0)
    if "total_tok_s_gpu" in cal:
        out["total_tok_s_gpu"] = out["output_tok_s_gpu"] * cal["total_tok_s_gpu"]
    if "p90_ttft_s" in cal:
        out["p90_ttft_s"] = cal["p90_ttft_s"] * p["prompt_ms"] / cal["prompt_ms"] + p["queue_s"]
    return out


def _calibration(T, bench, loop, gpus):
    p = point(T, None, T["budget"], {}, loop, gpus)
    cal = {"output_tok_s_gpu": bench["output_tok_s_gpu"] / p["output_tok_s_gpu"]} if bench.get("output_tok_s_gpu") else {}
    cal.update({k: bench[k] / p["interactivity"] for k in SCALED[1:] if bench.get(k)})
    if bench.get("total_tok_s_gpu") and bench.get("output_tok_s_gpu"):
        cal["total_tok_s_gpu"] = bench["total_tok_s_gpu"] / bench["output_tok_s_gpu"]
    if bench.get("p90_ttft_s"):
        cal.update(p90_ttft_s=bench["p90_ttft_s"], prompt_ms=p["prompt_ms"])
    return cal


def sweep(T, bench, workload, gpus, cut_planned, failed, caps=CAPS, budgets=BUDGETS):
    """Every cap and step token budget, now and with the plan's cut; the best of each that meets the requirements
    (else the fewest failed, then the metric) among the budgets the profile served, and the mean step the
    requirements need."""
    metric = workload.get("metric", "output_tok_s_gpu")
    reqs = workload.get("requirements") or {}
    n0 = T["decoding"]
    pop = max(float(workload.get("concurrency") or 0), n0)
    t0 = point(T, None, T["budget"], {}, (n0, 0.0), gpus)
    rate0 = n0 * T["accept"] / (t0["step_ms"] * T["osl"]) * 1e3
    loop = (pop, (pop - n0) / rate0 if rate0 else 0.0)
    cal = _calibration(T, bench, loop, gpus)
    lower = metric.endswith("_s")
    grid_caps = [c for c in caps if c < pop] + [None]
    grid_budgets = sorted(set(budgets) | ({round(T["budget"])} if T["budget"] else set()))
    now_ref = _metrics(point(T, None, T["budget"], {}, loop, gpus), cal)
    out = {}
    for name, cut in (("now", {}), ("planned", cut_planned)):
        pts = []
        for c in grid_caps:
            for b in grid_budgets:
                if c and b <= c * T["rows_per_request"]:
                    continue
                mid = point(T, c, b, cut, loop, gpus)
                ends = [_metrics(point(T, c, b, cut, loop, gpus, e), cal) for e in ("lo", "hi")]
                m = _metrics(mid, cal)
                rng = {k: [min(e[k] for e in ends), max(e[k] for e in ends)] for k in m}
                pts.append(dict(mid, metrics=m, range=rng, fails=failed(m, now_ref, reqs)))
        served = [p for p in pts if p["budget"] <= round(T["budget"])] or pts
        best = min(served, key=lambda p: (len(p["fails"]), (1 if lower else -1) * p["metrics"].get(metric, 0.0)))
        out[name] = {"best": best, "points": pts}
    best = out["planned"]["best"]
    need = {}
    for key, want in reqs.items():
        if want == "not_worse" or key not in cal:
            continue
        if key.endswith("interactivity"):
            need[key] = T["accept"] * 1e3 * cal[key] / float(want)
        elif key == "output_tok_s_gpu":
            need[key] = best["decoding"] * T["accept"] * 1e3 * cal[key] / (gpus * float(want))
    step = min(need.values()) if need else None
    return {
        "assumptions": {k: T[k] for k in ("decoding", "rows_per_request", "prompt_steps_per_request", "budget",
                                          "decode_ms", "prompt_ms", "decode_floor_ms", "decode_fit")}
        | {"population": loop[0], "think_s": loop[1], "prompts": len(T["prompts"])},
        "calibration": {k: v for k, v in cal.items() if k != "prompt_ms"},
        "now": out["now"]["best"], "planned": best,
        "points": {k: v["points"] for k, v in out.items()},
        "need": {"by_requirement": need, "step_ms": step,
                 "kernels_per_decode_step": step / (T["kernel_floor_us"] / 1e3) if step and T["kernel_floor_us"] else None},
    }

