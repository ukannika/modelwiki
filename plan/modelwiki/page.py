"""Read a model's page, its serving and its buckets, straight from their Markdown tables."""
import re
from pathlib import Path

from . import ROOT

FEATURE_GROUPS = {"Model": "model", "Attention": "attention", "Mixture of experts": "moe", "Draft model": "draft",
                  "Fusions": "fusion"}


def model_dir(model, root=ROOT):
    """models/<family>/<model>/."""
    hits = sorted(p.parent for p in Path(root).glob(f"models/*/{model}/model.md"))
    if len(hits) != 1:
        raise FileNotFoundError(f"expected one models/*/{model}/model.md, found {len(hits)}")
    return hits[0]


def _cells(line):
    return [c.strip() for c in line.strip()[1:-1].split("|")]


def tables(text):
    """Every table with the ## and ### headings it sits under: (h2, h3, columns, rows as dicts)."""
    out, h2, h3 = [], None, None
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("## "):
            h2, h3 = line[3:].strip(), None
        elif line.startswith("### "):
            h3 = line[4:].strip()
        elif line.startswith("|") and i + 1 < len(lines) and re.match(r"^\|\s*-", lines[i + 1]):
            cols = _cells(line)
            i += 2
            rows = []
            while i < len(lines) and lines[i].startswith("|"):
                rows.append(dict(zip(cols, _cells(lines[i]))))
                i += 1
            out.append((h2, h3, cols, rows))
            continue
        i += 1
    return out


def bucket_rows(rows_md):
    """IDs and Serving row names a bucket's Rows cell names."""
    ids = re.findall(r"`(m-[a-z0-9-]+)`", rows_md)
    serving = []
    m = re.search(r"Serving: (.*)$", rows_md)
    if m:
        serving = [s.split(" (")[0].strip() for s in m.group(1).split(", ")]
    return ids, serving


def load(model, root=ROOT):
    """The model's title, features (with IDs), serving rows and buckets."""
    mdir = model_dir(model, root)
    text = (mdir / "model.md").read_text()
    title = re.search(r"(?m)^# (.+)$", text).group(1).strip()

    features = []
    for h2, h3, cols, rows in tables(text):
        if h2 in FEATURE_GROUPS and "ID" in cols:
            for r in rows:
                features.append(dict(r, id=r["ID"].strip("`"), section=FEATURE_GROUPS[h2], subsection=h3))
    ids = [r["id"] for r in features]
    dup = sorted({i for i in ids if ids.count(i) > 1})
    if dup:
        raise ValueError(f"{mdir / 'model.md'}: duplicate IDs {dup}")

    buckets = []
    bfile = mdir / "buckets.md"
    if bfile.exists():
        for _, _, cols, rows in tables(bfile.read_text()):
            if cols[:2] == ["Bucket", "The block"]:
                buckets = [{"name": r["Bucket"], "block": r["The block"], "rows_md": r.get("Rows", "")} for r in rows]

    serving = []
    sfile = mdir / "serving.md"
    if sfile.exists():
        for h2, _, _, rows in tables(sfile.read_text()):
            serving += [dict(r, section="serving", subsection=h2) for r in rows if r.get("Feature")]

    unknown = [i for b in buckets for i in bucket_rows(b["rows_md"])[0] if i not in ids]
    if unknown:
        raise ValueError(f"{bfile}: rows {unknown} are not IDs on the page")
    return {"name": model, "title": title, "path": str(mdir / "model.md"), "buckets_path": str(bfile),
            "serving_path": str(sfile), "features": features, "buckets": buckets, "serving": serving}
