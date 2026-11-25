import pytest

from modelwiki.page import bucket_rows, load, model_dir


def test_dsv41_flash_page():
    page = load("dsv41-flash")
    assert page["title"] == "DeepSeek-V4.1-Flash"
    assert model_dir("dsv41-flash").parts[-2:] == ("dsv4", "dsv41-flash")
    ids = [r["id"] for r in page["features"]]
    assert len(ids) == 23 and ids[0] == "m-mix-seam" and ids[-1] == "m-fuse-attention-prologue"
    assert [b["name"] for b in page["buckets"]][:2] == ["Residual mixing", "Engram"]
    assert page["serving_path"].endswith("dsv4/dsv41-flash/serving.md")


def test_serving_rows_come_from_the_models_serving_md():
    serving = {(r["subsection"], r["Feature"]) for r in load("dsv41-flash")["serving"]}
    assert ("Topologies", "Sizes") not in serving
    assert ("KV cache", "Page size") in serving
    assert ("Scheduler", "Running sequences") in serving
    assert ("Kernel selection", "Dense GEMMs") in serving


def test_every_bucket_row_exists():
    page = load("dsv41-flash")
    ids = {r["id"] for r in page["features"]}
    names = {r["Feature"] for r in page["serving"]} | {r["subsection"] for r in page["serving"]}
    for b in page["buckets"]:
        rids, serving = bucket_rows(b["rows_md"])
        assert set(rids) <= ids, b["name"]
        assert set(serving) <= names, b["name"]


def test_bucket_rows_must_be_page_ids(root):
    path = model_dir("dsv41-flash", root) / "buckets.md"
    path.write_text(path.read_text().replace("`m-mix-seam`", "`m-mix-nope`", 1))
    with pytest.raises(ValueError, match=r"m-mix-nope"):
        load("dsv41-flash", root)


def test_ids_are_unique(root):
    path = model_dir("dsv41-flash", root) / "model.md"
    second = load("dsv41-flash")["features"][1]["id"]
    path.write_text(path.read_text().replace("| `m-mix-seam` |", f"| `{second}` |", 1))
    (path.parent / "buckets.md").write_text("")
    with pytest.raises(ValueError, match="duplicate IDs"):
        load("dsv41-flash", root)
