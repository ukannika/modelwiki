"""Work per op call from its recorded shapes (bytes moved, FLOPs), the rates the profile itself reached, and floors."""
import math

DTYPE_BYTES = {
    "float": 4, "double": 8, "c10::BFloat16": 2, "c10::Half": 2, "c10::Float8_e4m3fn": 1, "c10::Float8_e4m3fnuz": 1,
    "c10::Float8_e5m2": 1, "c10::Float8_e8m0fnu": 1, "c10::Float4_e2m1fn_x2": 1, "unsigned char": 1,
    "signed char": 1, "bool": 1, "short int": 2, "int": 4, "long int": 8,
}
PEAK_DTYPE = {"float": "fp32", "c10::BFloat16": "bf16", "c10::Half": "bf16", "c10::Float8_e4m3fn": "fp8",
              "c10::Float8_e4m3fnuz": "fp8", "c10::Float4_e2m1fn_x2": "mxfp4"}
GEMM = {"aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm", "aten::matmul", "aten::linear", "aten::_scaled_mm"}
REDUCE = {"aten::sum", "aten::mean", "aten::amax", "aten::amin", "aten::max", "aten::min", "aten::norm",
          "aten::linalg_vector_norm", "aten::var", "aten::argmax", "aten::topk", "aten::cumsum", "aten::softmax",
          "aten::_softmax", "aten::log_softmax", "aten::_log_softmax"}
COLLECTIVE = {"all_reduce": lambda n: 2 * (n - 1) / n, "all_gather": lambda n: (n - 1) / n,
              "reduce_scatter": lambda n: (n - 1) / n}


def _tensors(dims, types):
    out = []
    for d, t in zip(dims or [], types or []):
        if t in DTYPE_BYTES and isinstance(d, list):
            out.append((math.prod(d) if d else 1, DTYPE_BYTES[t], d, t))
    return out


def collective(op):
    return next((k for k in COLLECTIVE if k in op), None)


def op_work(op, dims, types, world=1):
    """{bytes, flops, dtype, kind} for one call, or None when the op's work cannot be read from its shapes."""
    ts = _tensors(dims, types)
    if not ts:
        return None
    coll = collective(op)
    if coll:
        n, b = ts[0][0], ts[0][1]
        return {"bytes": n * b, "link_bytes": n * b * COLLECTIVE[coll](max(world, 2)), "flops": 0, "dtype": None,
                "kind": "collective"}
    if op in GEMM:
        mats = [t for t in ts if len(t[2]) >= 2]
        if op in ("aten::addmm", "aten::baddbmm") and len(mats) >= 3:
            mats = mats[1:]
        if len(mats) < 2:
            return None
        (na, ba, a, ta), (nb, bb, b, _) = mats[0], mats[1]
        if op == "aten::linear":
            k, n_out = a[-1], b[0]
            m = na // k
        else:
            k, n_out = a[-1], b[-1]
            m = na // k
        out = m * n_out
        return {"bytes": na * ba + nb * bb + out * ba, "flops": 2 * m * k * n_out, "dtype": PEAK_DTYPE.get(ta, "bf16"),
                "kind": "gemm"}
    if "fused_moe" in op and len(ts) >= 4:
        (nx, bx, x, _), (_, _, w1, t1), (_, _, w2, t2) = ts[0], ts[1], ts[2]
        ids = next((t for t in ts[3:] if len(t[2]) == 2 and t[2][0] == x[0]), None)
        if len(x) != 2 or len(w1) != 3 or len(w2) != 3 or not ids:
            return None
        tokens, k, topk, experts = x[0], x[1], ids[2][1], w1[0]
        pack2 = 2 if t2 == "c10::Float4_e2m1fn_x2" else 1
        flops = 2 * tokens * topk * (w1[1] * k + w2[1] * w2[2] * pack2)
        per_expert = (math.prod(w1[1:]) + math.prod(w2[1:])) * DTYPE_BYTES[t1]
        touched = experts * (1 - (1 - 1 / experts) ** (tokens * topk))
        return {"bytes": nx * bx * 2 + per_expert * touched, "flops": flops, "dtype": PEAK_DTYPE.get(t1, "fp8"),
                "kind": "moe"}
    if op.startswith("aten::"):
        nbytes = sum(n * b for n, b, _, _ in ts)
        if op.endswith("_"):
            return {"bytes": nbytes, "flops": 0, "dtype": None, "kind": "elementwise"}
        if op in REDUCE:
            return {"bytes": nbytes, "flops": 0, "dtype": None, "kind": "reduce"}
        out = max(n for n, _, _, _ in ts) * max(b for _, b, _, _ in ts)
        return {"bytes": nbytes + out, "flops": 0, "dtype": None, "kind": "elementwise"}
    return None


def _pct(pairs, q):
    """Time-weighted percentile of (value, weight) pairs."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    acc = 0.0
    for v, w in pairs:
        acc += w
        if acc >= q * total:
            return v
    return pairs[-1][0] if pairs else None


def rates(calls, kernel_floor_us=None, q=0.9, min_bytes=256e6, min_flops=50e9):
    """The rates this profile reached, at its q-th time-weighted percentile: bandwidth from memory-bound calls,
    compute per dtype from GEMMs, the link from collectives, each from its large calls where the profile has them,
    else from all its calls (a low bound). calls: (work, ns_per_call, ncalls). kernel_floor_us: the shortest kernels'
    time per step kind."""
    pools = {}
    for w, ns, n in calls:
        if ns <= 0:
            continue
        if w["kind"] in ("elementwise", "reduce"):
            key, rate, big = "hbm_tb_s", w["bytes"] / ns / 1e3, w["bytes"] >= min_bytes
        elif w["kind"] == "collective":
            key, rate, big = "link_gb_s", w["link_bytes"] / ns, w["bytes"] >= min_bytes / 4
        elif w["kind"] in ("gemm", "moe") and w["dtype"] and w["flops"]:
            key, rate, big = ("tflops", w["dtype"]), w["flops"] / ns / 1e3, w["flops"] >= min_flops
        else:
            continue
        pools.setdefault(key, ([], []))[0 if big else 1].append((rate, ns * n))
    out = {"source": {}, "hbm_tb_s": None, "link_gb_s": None, "tflops": {}, "kernel_floor_us": kernel_floor_us or {}}
    for key, (big, small) in pools.items():
        use = big or small
        v = _pct(use, q)
        name = key if isinstance(key, str) else f"tflops.{key[1]}"
        out["source"][name] = f"p{int(q * 100)} of {len(use)} {'large' if big else 'small (no large: a low bound)'} op groups"
        if isinstance(key, str):
            out[key] = round(v, 2)
        else:
            out["tflops"][key[1]] = round(v, 1)
    return out


def floor_ms(bytes_, flops, dtype, link_bytes, launches, r, kind=None):
    """The fastest a group of calls could run at the profile's rates: memory, compute, link, or launches. A dtype the
    profile never ran has no compute floor."""
    mem = bytes_ / (r["hbm_tb_s"] * 1e9) if bytes_ and r.get("hbm_tb_s") else 0.0
    comp = flops / (r["tflops"].get(dtype, 0) * 1e9) if flops and dtype and r["tflops"].get(dtype) else 0.0
    lnk = link_bytes / (r["link_gb_s"] * 1e6) if link_bytes and r.get("link_gb_s") else 0.0
    kf = r.get("kernel_floor_us") or 0.0
    kf = kf.get(kind, min(kf.values(), default=0.0)) if isinstance(kf, dict) else kf
    launch = launches * kf / 1e3
    parts = {"memory": mem, "compute": comp, "link": lnk, "launch": launch}
    bound = max(parts, key=parts.get)
    return parts[bound], bound
