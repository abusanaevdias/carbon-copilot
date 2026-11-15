"""Carbon Copilot - one command from the raw 1C ledgers to predictions, audit file and reports (stdlib only).

Run:  python run.py [DATA_DIR] [--new extra_lines.csv]
      DATA_DIR = folder with ledger_train.csv, ledger_test.csv, emission_factors.csv ... (default ./data)
      --new    = extra raw ledger lines (same columns as ledger_test.csv); masked, classified, written to out/audit_new.csv
Out:  predictions.csv, audit.csv (root), out/*, masked/* (local only).
Exit code 0 = masking, pipeline and report self-checks all passed. Prints counts only, never names or rows.
No network calls: everything runs on this machine.
"""
import csv, os, shutil, subprocess, sys, time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "out")


def load(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def step(name, script, env):
    print(f"\n== {name} ==", flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, script)], cwd=HERE, env=env)
    if r.returncode:
        sys.exit(f"FAILED: {script} exit {r.returncode}")


def predict_new(env_data, new_path):
    """Classify the masked extra lines with a model trained on all labelled train lines."""
    os.environ["DALA_DATA"] = env_data
    sys.path.insert(0, HERE)
    import pipeline as P
    m = os.path.join(HERE, "masked")
    train, test, new = (P.load(os.path.join(m, n)) for n in ("ledger_train.csv", "ledger_test.csv", "ledger_new.csv"))
    clash = {r["line_id"] for r in new} & {r["line_id"] for r in train + test}
    if clash:
        sys.exit(f"FAILED: {len(clash)} line_id(s) in {new_path} already exist in the ledgers")
    dups = P.dup_ids(train + test + new)
    model = P.train_model(train, dups)
    preds = [dict(line_id=r["line_id"], **P.predict(r, model, dups)) for r in new]
    preds = __import__("llm_fallback").apply(new, preds, model, dups, train)   # second reader on unclear new lines
    fields = ["line_id"] + [k for k in preds[0] if k != "line_id"] if preds else P.AUDIT
    P.write(os.path.join(OUT, "audit_new.csv"), preds, fields)
    print(f"new lines: {len(preds)} classified -> out/audit_new.csv; "
          f"{sum(p['tco2e'] for p in preds):.1f} tCO2e; methods {dict(Counter(p['method'] for p in preds))}")


def stage_counts(audit):
    reasons = [r["reason"] for r in audit]
    flags = Counter(f for r in audit for f in r.get("flags", "").split("|") if f)
    print(f"test lines {len(audit)}: methods {dict(Counter(r['method'] for r in audit))}")
    print(f"  duplicates excluded {sum('duplicate' in x for x in reasons)}"
          f" | non-purchase excluded {sum('not an emission' in x or 'non-purchase' in x for x in reasons)}"
          f" | instruction-like text ignored {sum('instruction-like' in x for x in reasons)}")
    if flags:
        print("  flags:", dict(flags))
    print(f"  total {sum(float(r['tco2e']) for r in audit):,.1f} tCO2e")


def main(argv):
    t0 = time.time()
    new = os.path.abspath(argv[argv.index("--new") + 1]) if "--new" in argv else None
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--new")]
    data = os.path.abspath(pos[0] if pos else os.path.join(HERE, "data"))
    for f in ("ledger_train.csv", "ledger_test.csv", "emission_factors.csv", "sample_submission.csv"):
        if not os.path.exists(os.path.join(data, f)):
            sys.exit(f"FAILED: {f} not found in {data}. Usage: python run.py DATA_DIR [--new file.csv]")
    default = os.path.abspath(os.path.join(HERE, "data"))
    if data != default and "DALA_DATA" not in open(os.path.join(HERE, "pipeline.py"), encoding="utf-8").read():
        sys.exit(f"FAILED: this pipeline.py reads reference data only from {default}; put the data there")
    env = dict(os.environ, DALA_DATA=data, PYTHONIOENCODING="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")

    print("== 1. mask names (local) ==")
    sys.path.insert(0, HERE)
    import mask
    stats = mask.main(data, new)
    lk = mask.check(data, new)                         # re-read what was written to disk
    print(f"masked counterparties {stats['counterparties']} | persons {stats['persons']} | leaks {sum(lk.values())}")
    if any(lk.values()):
        sys.exit("FAILED: raw name found in masked files")

    step("2. pipeline (rules + naive Bayes, self-checks)", "pipeline.py", env)
    stale = os.path.join(OUT, "audit_new.csv")
    if new:
        print("\n== 2b. new lines ==")
        predict_new(data, new)
    elif os.path.exists(stale):
        os.remove(stale)

    shutil.copyfile(os.path.join(OUT, "submission.csv"), os.path.join(HERE, "predictions.csv"))
    shutil.copyfile(os.path.join(OUT, "audit_test.csv"), os.path.join(HERE, "audit.csv"))
    pred, sample = load(os.path.join(HERE, "predictions.csv")), load(os.path.join(data, "sample_submission.csv"))
    if [r["line_id"] for r in pred] != [r["line_id"] for r in sample]:
        sys.exit("FAILED: predictions.csv line_ids differ from sample_submission.csv")

    for script in ("report.py", "inventory.py"):       # report.py supersedes inventory.py once it exists
        if os.path.exists(os.path.join(HERE, script)):
            step(f"3. report ({script})", script, env)
            break
    else:
        print("\n== 3. report: report.py not present yet, skipped ==")

    print("\n== summary ==")
    stage_counts(load(os.path.join(HERE, "audit.csv")))
    print(f"written: predictions.csv ({len(pred)} rows), audit.csv, out/")
    print(f"runtime {time.time() - t0:.1f} s | second reader mode {os.environ.get('DALA_LLM', 'cache')} (cache/off = no network) | DONE")


if __name__ == "__main__":
    main(sys.argv[1:])
