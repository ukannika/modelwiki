"""Step time from step shape, fitted on the profile's own steps, and the serving metrics a sequence of steps gives."""
import math

import numpy as np

FEATURES = {
    "decode": ("dec_rows", "dec_ctx"),
    "mixed": ("pf_tok", "pf_attn", "pf_ctx", "dec_rows", "dec_ctx"),
    "prefill": ("pf_tok", "pf_attn", "pf_ctx"),
}
NUM = ("host_ms", "rows", "pf_reqs", "pf_tok", "pf_ctx", "pf_attn", "dec_reqs", "dec_rows", "dec_ctx")


def length(v):
    """A sequence length: a number (a mapping's mean is accepted)."""
    return float((v.get("mean") if isinstance(v, dict) else v) or 0)


def load(rows):
    out = []
    for r in rows:
        s = {k: float(r.get(k) or 0) for k in NUM}
        s.update(step=int(r["step"]), kind=r["kind"], traced=str(r["traced"]).lower() == "true")
        out.append(s)
    return out


def _nnls(X, y):
    """Least squares with every coefficient but the intercept (column 0) kept at or above zero."""
    keep = list(range(X.shape[1]))
    while True:
        c, *_ = np.linalg.lstsq(X[:, keep], y, rcond=None)
        neg = [keep[i] for i, v in enumerate(c) if v < 0 and keep[i] != 0]
        if not neg:
            full = np.zeros(X.shape[1])
            full[keep] = c
            return full
        keep.remove(min(neg, key=lambda j: c[keep.index(j)]))


def fit(steps, kind):
    """host_ms ~ intercept + shape terms over the kind's untraced steps (all of them when fewer than 30 are untraced)."""
    feats = FEATURES.get(kind, ())
    pool = [s for s in steps if s["kind"] == kind]
    untraced = [s for s in pool if not s["traced"]]
    use = untraced if len(untraced) >= 30 else pool
    if not use:
        return None
    y = np.array([s["host_ms"] for s in use])
    X = np.array([[1.0] + [s[f] for f in feats] for s in use])
    c = _nnls(X, y) if len(use) > len(feats) + 1 else np.array([y.mean()] + [0.0] * len(feats))
    pred = X @ c
    ss = float(((y - y.mean()) ** 2).sum())
    rng = {f: [float(np.percentile(X[:, i + 1], 5)), float(np.percentile(X[:, i + 1], 95))] for i, f in enumerate(feats)}
    return {"kind": kind, "n": len(use), "traced": use is pool and any(s["traced"] for s in pool),
            "intercept": float(c[0]), "coef": {f: float(v) for f, v in zip(feats, c[1:])},
            "r2": 1 - float(((y - pred) ** 2).sum()) / ss if ss else 0.0,
            "mape": float(np.mean(np.abs(y - pred) / y)), "range": rng}


def predict(model, s):
    return model["intercept"] + sum(c * s.get(f, 0.0) for f, c in model["coef"].items())


def fits(steps):
    return {k: m for k in sorted({s["kind"] for s in steps}) if (m := fit(steps, k))}


def tracing(steps, models, launches_per_step):
    """Host time tracing adds per kernel: traced decode steps against the untraced fit, over their launches."""
    m = models.get("decode")
    traced = [s for s in steps if s["kind"] == "decode" and s["traced"]]
    if not m or m["traced"] or not traced or not launches_per_step.get("decode"):
        return 0.0
    extra = sum(s["host_ms"] - predict(m, s) for s in traced) / len(traced)
    return max(0.0, extra / launches_per_step["decode"])


def untraced_ms(steps, per_kernel_ms, launches_per_step):
    """Each step's host time with the tracing overhead removed from traced steps."""
    return [s["host_ms"] - (per_kernel_ms * launches_per_step.get(s["kind"], 0) if s["traced"] else 0.0)
            for s in steps]


def metrics(steps, times, workload, gpus):
    """Output tok/s/GPU, interactivity (mean, p50, p90) and the mean step, from a sequence of steps and their times.
    Each request is a window of consecutive steps: OSL / acceptance length of them."""
    spec = workload.get("speculative") or {}
    accept = float(spec.get("acceptance_length") or 1.0)
    tokens = sum(s["dec_reqs"] for s in steps) * accept
    total_s = sum(times) / 1e3
    out = {"output_tok_s_gpu": tokens / total_s / gpus if total_s else 0.0,
           "mean_step_ms": sum(times) / len(times) if times else 0.0}
    osl = length(workload.get("osl"))
    lengths = [max(1, round(osl / accept))] if osl else []
    speeds = []
    t = np.asarray(times, dtype=float)
    csum = np.concatenate([[0.0], np.cumsum(t)])
    for L in lengths:
        if L < len(t):
            win = (csum[L:] - csum[:-L]) / 1e3
            speeds.append(accept * L / win)
    if speeds:
        v = np.concatenate(speeds)
        out.update(mean_interactivity=float(v.mean()), p50_interactivity=float(np.percentile(v, 50)),
                   p90_interactivity=float(np.percentile(v, 10)))
    mixed = [x for s, x in zip(steps, times) if s["kind"] != "decode"]
    out["mean_prefill_step_ms"] = sum(mixed) / len(mixed) if mixed else 0.0
    return out


def retarget(steps, models, profiled, workload):
    """Scale each step to another workload: decode requests by the active-decodes ratio (each a number here), prefill
    work by the new prefill tokens. Returns the scaled shapes, a time factor per step, and the features pushed past the
    fitted range."""
    r_req = (float(workload.get("active_decodes") or workload.get("concurrency") or 1)
             / float(profiled.get("active_decodes") or profiled.get("concurrency") or 1))

    def pf(w):
        return length(w.get("isl")) * (1 - float(w.get("prefix_hit") or 0))

    r_isl = (length(workload.get("isl")) or 1) / (length(profiled.get("isl")) or 1)
    r_pf = pf(workload) / pf(profiled) if pf(profiled) else 1.0
    d_old = float((profiled.get("speculative") or {}).get("drafts") or 0)
    d_new = float((workload.get("speculative") or {}).get("drafts") or d_old)
    r_rows = r_req * (1 + d_new) / (1 + d_old)
    out, factor, past = [], [], set()
    for s in steps:
        n = dict(s)
        n["dec_reqs"] *= r_req
        n["dec_rows"] *= r_rows
        n["dec_ctx"] *= r_req * r_isl
        n["pf_tok"] *= r_pf
        n["pf_ctx"] *= r_isl
        n["pf_attn"] *= r_pf * r_isl
        m = models.get(s["kind"])
        if m:
            old, new = predict(m, s), predict(m, n)
            factor.append(new / old if old > 0 else 1.0)
            for f, (lo, hi) in m["range"].items():
                if not (lo * 0.8 <= n[f] <= hi * 1.25 or lo == hi == 0):
                    past.add(f"{s['kind']}.{f}")
        else:
            factor.append(1.0)
        out.append(n)
    return out, factor, sorted(past)


def ratio(measured, modeled):
    return {k: measured[k] / modeled[k] for k in modeled if measured.get(k) and modeled.get(k) and not math.isnan(modeled[k])}
