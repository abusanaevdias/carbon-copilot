"""Build the hand-in zip Team<N>_DalaFoods.zip (stdlib only).

Run:  python make_zip.py <team number> [DATA_DIR]      (after python run.py)
Never packs .mask_salt, mask_map.csv, masked/, __pycache__ or raw ledgers; scans every packed file for raw
counterparty / person names and refuses to write the zip if any is found (prints counts only).
"""
import glob, os, sys, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = ["run.py", "app.py", "mask.py", "pipeline.py", "report.py", "inventory.py", "eval_hard.py", "make_zip.py",
        "llm_fallback.py", "llm_cache.json", "out/llm_eval.json"]
RESULTS = ["predictions.csv", "audit.csv", "bank_summary.pdf", "out/inventory.json", "out/bank_summary.html",
           "out/dashboard.html", "out/audit_all.csv"]
TREES = ["templates", "demo", "tests", "deck"]
FORBIDDEN = (".mask_salt", "mask_map", "masked/", "__pycache__", "ledger_train", "ledger_test", "ledger_new", "_fresh",
             ".env", "_key", ".venv")
SECRET_MARKERS = ("sk-or-", "OPENROUTER_API_KEY=", "x-api-key:")   # API keys must never ship


def members():
    files = [(f, f) for f in CODE + RESULTS if os.path.isfile(os.path.join(HERE, f))]
    files.append(("docs/SUBMISSION.md", "README.md"))
    for t in TREES:
        for p in glob.glob(os.path.join(HERE, t, "**", "*"), recursive=True):
            rel = os.path.relpath(p, HERE).replace(os.sep, "/")
            if os.path.isfile(p) and not (t == "deck" and not rel.endswith(".pdf")):
                files.append((rel, rel))
    return [(s, d) for s, d in files if not any(x in s for x in FORBIDDEN)]


def main(argv):
    if not argv:
        sys.exit("usage: python make_zip.py <team number> [DATA_DIR]")
    team = argv[0]
    data = argv[1] if len(argv) > 1 else os.path.join(HERE, "data")
    sys.path.insert(0, HERE)
    import mask
    _, cp_map, person_map, _, cores = mask.mask_rows(mask.read_raw(data), mask.salt_value())
    files = members()
    root = f"Team{team}_DalaFoods"
    leaks = 0
    for src, _ in files:
        blob = open(os.path.join(HERE, src), "rb").read().decode("utf-8", "ignore").lower()
        lk = mask.leaks(blob, cp_map, cores, person_map)
        lk.pop("patronymic_left")              # code/regex text legitimately contains suffix patterns
        if src == "mask.py":                   # its CP_KW list holds public brand keywords (fuel/bank chains) that
            lk = {"person": lk["person"]}      # equal some counterparty names by design; persons still checked
        leaks += sum(lk.values())
        raw = open(os.path.join(HERE, src), "rb").read().decode("utf-8", "ignore")
        keyf = os.path.join(HERE, ".scaledown_key")
        secrets = [open(keyf).read().strip()] if os.path.exists(keyf) else []
        markers = () if src == "make_zip.py" else SECRET_MARKERS   # this file names the markers itself
        if any(m in raw for m in markers) or any(s and s in raw for s in secrets):
            sys.exit(f"FAILED: secret-looking text in {src}; zip not written")
    if leaks:
        sys.exit(f"FAILED: {leaks} raw name hit(s) in files to pack; zip not written")
    path = os.path.join(HERE, root + ".zip")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for src, dst in files:
            z.write(os.path.join(HERE, src), f"{root}/{dst}")
    names = zipfile.ZipFile(path).namelist()
    assert not any(x in n for n in names for x in FORBIDDEN), "forbidden file in zip"
    for need in ("predictions.csv", "audit.csv", "run.py", "README.md"):
        assert f"{root}/{need}" in names, f"missing {need}"
    print(f"{path}: {len(names)} files, raw-name hits 0, no salt/map/masked files")
    print("\n".join("  " + n for n in names))


if __name__ == "__main__":
    main(sys.argv[1:])
