"""Laya feasibility spike: ef_id choice over GL/unit-allowed candidates vs pipeline.py on the same hard folds.

Run:  laya_spike\\.venv\\Scripts\\python.exe laya_spike\\eval_laya.py [--limit N]
Reads masked/ledger_train.csv only (+ masked test for duplicate ids). pipeline.py / eval_hard.py imported read-only.
Writes laya_spike/out_results.json and laya_spike/out_preds.csv.
"""
import os, sys, json, time, random, csv
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pipeline as P
import eval_hard as E
from laya import Router

LIMIT = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
EXCL = "EXCLUDE"
TYPCAT = {e: f["typical_category"].split("/")[0].strip() for e, f in P.EF.items()}


def label(r):
    return r["y_ef_id"] or EXCL


def fits(row, ef):
    row = dict(row, unit=P.canon_unit(row["unit"]))
    return P.activity(row, ef)[0] is not None


def candidates(row, gl_labels):
    """GL seen in train fold: its ef_ids + unit-convertible activity factors. GL unseen: every unit-convertible factor."""
    seen = gl_labels.get(row["gl_account"])
    act = [e for e in P.EF if not e.startswith("SP_") and fits(row, e)]
    if seen:
        c = list(dict.fromkeys([e for e, _ in seen.most_common()] + act))
    else:
        c = act + [e for e in P.EF if e.startswith("SP_")] + [EXCL]
    return c  # ponytail: >20 options get trimmed by Laya's 256-token head; use predict_shortlist if that bites


def state(row):
    gl = row["gl_account"]
    return (f"Ledger line. Description: {row['description']}. GL account: {gl} "
            f"{P.GL_NAME.get(gl, 'unknown')}. Quantity: {row['quantity']} {row['unit']}.")


def crit(e):
    if e == EXCL:
        return "not an emission: tax, fine, social contribution, advance payment, per diem"
    f = P.EF[e]
    return f"{f['name']} ({f['unit']})"


def question(cands):
    return {"ef": {"type": "choice", "instructions": "Which emission factor matches this purchase ledger line?",
                   "criteria": {e: crit(e) for e in cands}}}


ROUTER = None


def ask(rows, cands_list):
    out, t0 = [], time.perf_counter()
    reqs = [{"state": state(r), "questions": question(c)} for r, c in zip(rows, cands_list)]
    try:
        res = ROUTER.predict_batch(reqs, batch_size=16)
    except Exception as ex:  # API shape mismatch -> one by one
        print("predict_batch failed, falling back:", repr(ex)[:200])
        res = [ROUTER.predict(q["state"], q["questions"]) for q in reqs]
    for r in res:
        a = r["answers"]["ef"]
        p = a.get("answer_confidence", a.get("confidence"))
        if p is None and a.get("probabilities"):
            p = a["probabilities"].get(a["choice"])
        out.append((a["choice"], float(p or 0)))
    return out, (time.perf_counter() - t0) / max(1, len(rows))


def run_laya(rows, dups, fold):
    preds = [None] * len(rows)
    lat = []
    for f in sorted(set(fold)):
        tr = [r for r, g in zip(rows, fold) if g != f and r["line_id"] not in dups]
        gl_labels, gl_cat = defaultdict(Counter), defaultdict(Counter)
        for r in tr:
            gl_labels[r["gl_account"]][label(r)] += 1
            gl_cat[(r["gl_account"], label(r))][r["y_category"]] += 1
        idx = [j for j, g in enumerate(fold) if g == f]
        for j in idx:
            if rows[j]["line_id"] in dups:  # same deterministic dup rule as pipeline; Laya can't see other lines
                preds[j] = dict(ef_id="", category=EXCL, p=1.0, ncand=0, rule=True)
        idx = [j for j in idx if preds[j] is None]
        cl = [candidates(rows[j], gl_labels) for j in idx]
        ans, l = ask([rows[j] for j in idx], cl)
        lat.append(l)
        for j, c, (ch, p) in zip(idx, cl, ans):
            gc = gl_cat.get((rows[j]["gl_account"], ch))
            cat = EXCL if ch == EXCL else gc.most_common(1)[0][0] if gc else TYPCAT[ch]
            preds[j] = dict(ef_id="" if ch == EXCL else ch, category=cat, p=p, ncand=len(c), rule=False)
    return preds, sum(lat) / len(lat)


def metrics(rows, preds):
    acc = sum(r["y_ef_id"] == p["ef_id"] for r, p in zip(rows, preds)) / len(rows)
    f1 = P.macro_f1([r["y_category"] for r in rows], [p["category"] for p in preds])
    return round(acc, 4), round(f1, 4)


BINS = [0, .5, .7, .8, .9, .95, .97, .99, 1.0001]


def calib(rows, preds):
    pts = [(p["p"], r["y_ef_id"] == p["ef_id"]) for r, p in zip(rows, preds) if not p["rule"]]
    bins = []
    for lo, hi in zip(BINS, BINS[1:]):
        b = [ok for q, ok in pts if lo <= q < hi]
        bins.append(dict(bin=f"[{lo:.2f},{min(hi, 1):.2f}]", n=len(b), acc=round(sum(b) / len(b), 4) if b else None))
    thr = None
    for t in sorted({round(q, 3) for q, _ in pts}):
        a = [ok for q, ok in pts if q >= t]
        if len(a) >= 20 and sum(a) / len(a) >= 0.97:
            thr = dict(threshold=t, coverage=round(len(a) / len(pts), 4), acc=round(sum(a) / len(a), 4))
            break
    return bins, thr


def main():
    global ROUTER
    train = P.load(os.path.join(P.MASKED, "ledger_train.csv"))
    test = P.load(os.path.join(P.MASKED, "ledger_test.csv"))
    dups = P.dup_ids(train + test)
    if LIMIT:
        train = random.Random(1).sample(train, LIMIT)
    dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    t = time.perf_counter()
    ROUTER = Router(preload=True, device=dev)
    load_s = time.perf_counter() - t
    res = dict(device=dev, load_s=round(load_s, 1), rows=len(train), splits={})
    allp = []
    for name, key in [("GL account", lambda r: r["gl_account"]), ("description template", E.template)]:
        fold, ng = E.group_folds(train, key)
        lp, lat = run_laya(train, dups, fold)
        pp = E.run(train, dups, fold)
        bins, thr = calib(train, lp)
        nc = [p["ncand"] for p in lp if not p["rule"]]
        res["splits"][name] = dict(groups=ng, laya=metrics(train, lp), pipeline=metrics(train, pp),
                                   latency_ms_per_line=round(lat * 1000, 1), bins=bins, gate=thr,
                                   cand_mean=round(sum(nc) / len(nc), 1), cand_max=max(nc))
        print(name, json.dumps(res["splits"][name], ensure_ascii=False))
        for r, a, b in zip(train, lp, pp):
            allp.append(dict(split=name, line_id=r["line_id"], y_ef=r["y_ef_id"], y_cat=r["y_category"],
                             laya_ef=a["ef_id"], laya_cat=a["category"], laya_p=round(a["p"], 4), ncand=a["ncand"],
                             pipe_ef=b["ef_id"], pipe_cat=b["category"]))
    # single-line latency (no batching)
    rng = random.Random(7)
    pool = [r for r in train if r["line_id"] not in dups]
    sample = rng.sample(pool, min(50, len(pool)))
    gl_labels = defaultdict(Counter)
    for r in train:
        gl_labels[r["gl_account"]][label(r)] += 1
    cl = [candidates(r, gl_labels) for r in sample]
    t = time.perf_counter()
    clean = [ROUTER.predict(state(r), question(c))["answers"]["ef"]["choice"] for r, c in zip(sample, cl)]
    res["latency_single_ms"] = round((time.perf_counter() - t) / 50 * 1000, 1)
    inj = [ROUTER.predict(state(E.injected(r, random.Random(i))), question(c))["answers"]["ef"]["choice"]
           for i, (r, c) in enumerate(zip(sample, cl))]
    res["injection"] = dict(n=50, changed=sum(a != b for a, b in zip(clean, inj)),
                            changed_to_exclude=sum(a != b and b == EXCL for a, b in zip(clean, inj)))
    print(json.dumps({k: v for k, v in res.items() if k != "splits"}))
    with open(os.path.join(HERE, "out_results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    with open(os.path.join(HERE, "out_preds.csv"), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(allp[0])); w.writeheader(); w.writerows(allp)


if __name__ == "__main__":
    main()
