"""Typical competitor approaches, measured on the same masked data and the same hard tests as Carbon Copilot.

Run (from case-bv2):  uv run --with pandas --with scikit-learn python compete/baselines.py
Reads masked/*.csv and the organisers' reference tables only. The organisers' starter_pipeline.py is executed
unmodified (runpy) inside compete/_work, where the masked ledgers sit under the file names it expects.
pipeline.py and eval_hard.py are imported, never edited. Prints counts only, never ledger rows.
Writes compete/out/baselines.json, scores.csv, slices.csv.

Score = 0.4 * macro-F1 + 0.6 * max(0, 1 - WAPE), computed by pipeline.score (the code that scores ours).

Approaches (what a team gets if it stops there):
  starter        organisers' starter_pipeline.py as is
  clf_starter    TF-IDF + logistic regression on the description picks category and factor (the primer's
                 suggestion); the starter's calculator does the maths
  clf_units      better-informed team: the classifier also sees account and unit, and a scale unit table converts
                 t->kg, MWh->kWh, thousand m3->m3, GJ->Gcal; still no density, pieces, flight distance, t x km,
                 duplicate or refrigerant handling
  spend_only     best case for money-only: TRUE category given, tCO2e = net amount x intensity fitted per category
  ours           pipeline.py
"""
import contextlib, io, json, os, random, runpy, shutil, sys, warnings
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CASE = os.path.join(ROOT, "..", "cases", "Case2_DalaFoods")
WORK, OUT = os.path.join(HERE, "_work"), os.path.join(HERE, "out")
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "tests"))
warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd                                                # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer       # noqa: E402
from sklearn.linear_model import LogisticRegression               # noqa: E402
from sklearn.pipeline import make_pipeline                        # noqa: E402
import pipeline as P, eval_hard as H, mask as M                   # noqa: E402
import test_injection as TI, test_new_lines as TN                 # noqa: E402

# ---------- the organisers' starter, unmodified ----------
os.makedirs(WORK, exist_ok=True); os.makedirs(OUT, exist_ok=True)
for name in ("ledger_train.csv", "ledger_test.csv"):
    shutil.copyfile(os.path.join(P.MASKED, name), os.path.join(WORK, name))
shutil.copyfile(os.path.join(CASE, "emission_factors.csv"), os.path.join(WORK, "emission_factors.csv"))
os.chdir(WORK)
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    S = runpy.run_path(os.path.join(CASE, "starter_pipeline.py"))
os.chdir(ROOT)
STARTER_SAYS = _buf.getvalue().splitlines()[0]                    # its own train line; the rest prints rows

NUMCOLS = ["quantity", "amount_kzt_gross", "vat_kzt"]


def frame(rows):
    df = pd.DataFrame(rows)
    for c in NUMCOLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def fnum(x):
    return 0.0 if x != x else float(x)                             # NaN -> 0 so a score exists; NaNs are counted


def starter(train, val):
    out = S["run"](frame(val))
    return [dict(category=c, ef_id=e, tco2e=fnum(t), nan=t != t) for c, e, t in zip(out["category"], out["ef_id"], out["tco2e"])]


def text(r, rich):
    """rich = the better-informed team: account and unit given to the classifier as extra tokens."""
    return r["description"] + (f" gl{r['gl_account']} unit_{r['unit']}" if rich else "")


def fit_clf(train, rich):
    clf = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True, lowercase=True),
                        LogisticRegression(C=20, max_iter=2000))
    return clf.fit([text(r, rich) for r in train], [r["y_category"] + "|" + r["y_ef_id"] for r in train])


SCALE = {"кг": ("kg", 1), "т": ("kg", 1000), "л": ("l", 1), "кВт·ч": ("kWh", 1), "МВт·ч": ("kWh", 1000),
         "м3": ("m3", 1), "тыс. м3": ("m3", 1000), "Гкал": ("Gcal", 1), "ГДж": ("Gcal", 1 / 4.1868), "ткм": ("tkm", 1)}
EF_BASE = {"kg": ("kg", 1), "t": ("kg", 1000), "l": ("l", 1), "kWh": ("kWh", 1), "m3": ("m3", 1), "Gcal": ("Gcal", 1),
           "tkm": ("tkm", 1)}


def clf_any(scale_units):
    def fn(train, val):
        labs = fit_clf(train, scale_units).predict([text(r, scale_units) for r in val])
        preds = []
        for r, lab in zip(frame(val).to_dict("records"), labs):
            cat, ef_id = lab.split("|")
            if not ef_id or ef_id.startswith("SP_") or not scale_units:
                t = S["emissions_t"](r, cat, ef_id or None)
            else:                                                  # scale conversion when the dimension matches, else x1
                u, f = SCALE.get(str(r["unit"])), EF_BASE.get(P.EF[ef_id]["unit"])
                k = u[1] / f[1] if u and f and u[0] == f[0] else 1.0
                t = r["quantity"] * k * float(P.EF[ef_id]["kgCO2e_per_unit"]) / 1000
            preds.append(dict(category=cat, ef_id=ef_id, tco2e=fnum(t), nan=t != t))
        return preds
    return fn


def spend_only(train, val):
    num, den = defaultdict(float), defaultdict(float)
    for r in train:
        num[r["y_category"]] += P.num(r["y_tco2e"]); den[r["y_category"]] += P.net_kzt(r)
    return [dict(category=r["y_category"], ef_id="(spend)", nan=False,
                 tco2e=P.net_kzt(r) * num[r["y_category"]] / den[r["y_category"]] if den[r["y_category"]] else 0.0)
            for r in val]


def make_ours(dups):
    def ours(train, val):
        model = P.train_model(train, dups)
        return [dict(P.predict(r, model, dups), nan=False) for r in val]
    return ours


def cv(fn, rows, fold, perturb=None):
    """Same protocol as eval_hard.run: train on the other folds, perturb validation rows only, same RNG per line."""
    preds = [None] * len(rows)
    for f in sorted(set(fold)):
        tr = [r for r, g in zip(rows, fold) if g != f]
        idx = [j for j, g in enumerate(fold) if g == f]
        val = []
        for j in idx:
            rng = random.Random(H.SEED * 100003 + int(rows[j]["line_id"][1:]))
            val.append(perturb(rows[j], rng) if perturb else rows[j])
        for j, p in zip(idx, fn(tr, val)):
            preds[j] = p
    return preds


def slices(rows, dups):
    """Failure-mode slices defined from the labels (which lines need which convention)."""
    star_units = set(S["TO_BASE"])
    act = lambda r: r["y_ef_id"].startswith("EF_")
    return [
        ("duplicate posting -> EXCLUDE", lambda r: r["line_id"] in dups),
        ("credit note (negative emissions)", lambda r: P.num(r["y_tco2e"]) < 0),
        ("refrigerant refill (Scope 1 fugitive)", lambda r: r["y_category"] == "S1_FUGITIVE"),
        ("flight (great-circle, every leg)", lambda r: r["y_ef_id"].startswith("EF_FLIGHT")),
        ("pieces -> kg (unit weight)", lambda r: act(r) and r["unit"] == "шт"),
        ("litres -> kg (density)", lambda r: r["unit"] == "л" and P.EF.get(r["y_ef_id"], {}).get("unit") == "kg"),
        ("tonne-km (incl. t x km)", lambda r: r["y_ef_id"].endswith("_TKM")),
        ("waste m3 -> t (bulk density)", lambda r: r["unit"] == "м3" and P.EF.get(r["y_ef_id"], {}).get("unit") == "t"),
        ("activity line, unit missing in starter table", lambda r: act(r) and r["unit"] not in star_units),
        ("money-based line (net of VAT)", lambda r: r["y_ef_id"].startswith("SP_")),
        ("non-purchase -> EXCLUDE (not duplicate)", lambda r: r["y_category"] == "EXCLUDE" and r["line_id"] not in dups),
    ]


def main():
    train = P.load(os.path.join(P.MASKED, "ledger_train.csv"))
    test = P.load(os.path.join(P.MASKED, "ledger_test.csv"))
    dups = P.dup_ids(train + test)
    idx = list(range(len(train))); random.Random(H.SEED).shuffle(idx)
    rand = [0] * len(train)
    for i, j in enumerate(idx):
        rand[j] = i % H.K
    scen = [("random 5-fold", rand, None)]
    for name, key in [("unseen wording", H.template), ("unseen suppliers", lambda r: r["counterparty"]),
                      ("unseen accounts", lambda r: r["gl_account"])]:
        scen.append((name, H.group_folds(train, key)[0], None))
    for name, fn in [("typo", H.typo), ("RU/KZ/EN synonym", H.synonym), ("unit spelled differently", H.unit_variant),
                     ("words reordered", H.reorder), ("account missing", H.no_gl), ("injected instruction", H.injected)]:
        scen.append((name, rand, fn))
    approaches = [("starter", starter), ("clf_starter", clf_any(False)), ("clf_units", clf_any(True)),
                  ("spend_only", spend_only), ("ours", make_ours(dups))]

    print("organisers' starter, its own print on masked train:", STARTER_SAYS)
    res = {"starter_says": STARTER_SAYS, "scores": [], "slices": [], "rows": len(train)}
    keep = {}
    print(f"\n{'scenario':26s} " + " ".join(f"{a:>26s}" for a, _ in approaches))
    print(f"{'':26s} " + " ".join(f"{'score  F1    WAPE  catErr':>26s}" for _ in approaches))
    for name, fold, pert in scen:
        line = f"{name:26s} "
        for a, fn in approaches:
            preds = cv(fn, train, fold, pert)
            f1, wape, s = P.score(train, preds)
            ce = sum(r["y_category"] != p["category"] for r, p in zip(train, preds))
            res["scores"].append(dict(scenario=name, approach=a, score=round(s, 4), macro_f1=round(f1, 4),
                                      wape=round(wape, 4), category_errors=ce, nan_outputs=sum(p["nan"] for p in preds)))
            line += f"{s:7.4f} {f1:5.3f} {wape:6.3f} {ce:5d}  "
            if name in ("random 5-fold", "unseen wording"):
                keep[(name, a)] = preds
        print(line)
    worst = {a: min(x["score"] for x in res["scores"] if x["approach"] == a) for a, _ in approaches}
    res["worst_score"] = worst
    print("worst case per approach:", worst)

    # ---------- failure modes (random 5-fold: the friendliest setting for every approach) ----------
    total = sum(abs(P.num(r["y_tco2e"])) for r in train)
    print(f"\nfailure modes on {len(train)} labelled lines, random 5-fold; 'ok' = category right and tCO2e within 1%")
    print(f"{'slice':46s} {'n':>4s} {'true t':>10s}  " + " ".join(f"{a:>20s}" for a, _ in approaches))
    for sname, cond in slices(train, dups):
        ids = [j for j, r in enumerate(train) if cond(r)]
        true_t = sum(abs(P.num(train[j]["y_tco2e"])) for j in ids)
        line = f"{sname:46s} {len(ids):4d} {true_t:10.1f}  "
        for a, _ in approaches:
            pr = keep[("random 5-fold", a)]
            err = sum(abs(pr[j]["tco2e"] - P.num(train[j]["y_tco2e"])) for j in ids)
            ok = sum(pr[j]["category"] == train[j]["y_category"] and
                     abs(pr[j]["tco2e"] - P.num(train[j]["y_tco2e"])) <= 0.01 * abs(P.num(train[j]["y_tco2e"])) + 0.0006
                     for j in ids)
            res["slices"].append(dict(slice=sname, n=len(ids), true_abs_t=round(true_t, 1), approach=a, lines_ok=ok,
                                      abs_err_t=round(err, 1), wape_points=round(err / total, 4)))
            line += f"{ok:4d} ok {err:10.1f} t  "
        print(line)
    for a, _ in approaches:                                        # inventory total: do errors cancel or pile up?
        pr = keep[("random 5-fold", a)]
        res.setdefault("totals", {})[a] = round(sum(p["tco2e"] for p in pr), 1)
    res["totals"]["truth"] = round(sum(P.num(r["y_tco2e"]) for r in train), 1)
    print("inventory total on train lines, t:", res["totals"])

    # ---------- injection: 12 instruction forms x 200 masked test lines (tests/test_injection.py protocol) ----------
    print("\ninjection: lines whose category, factor or tCO2e changed, of 2400")
    res["injection"] = {}
    clean_fns = [(a, fn) for a, fn in approaches if a != "spend_only"]   # spend_only takes the true label: not testable
    sub = test[:200]
    for a, fn in clean_fns:
        base = fn(train, sub)
        changed = 0
        for pos, text in TI.INSTRUCTIONS:
            inj = fn(train, [dict(r, description=TI.inject(r["description"], pos, text)) for r in sub])
            changed += sum((x["category"], x["ef_id"], round(x["tco2e"], 4)) != (y["category"], y["ef_id"], round(y["tco2e"], 4))
                           for x, y in zip(base, inj))
        res["injection"][a] = changed
        print(f"  {a:12s} {changed:4d} / 2400")

    # ---------- 12 invented lines with hand-computed answers (tests/test_new_lines.py) ----------
    import csv
    with open(os.path.join(ROOT, "demo", "new_lines.csv"), encoding="utf-8-sig", newline="") as f:
        rd = csv.DictReader(f); fields, raw = rd.fieldnames, list(rd)
    new = M.mask_rows({"ledger_new.csv": (fields, raw)}, "test-salt")[0]["ledger_new.csv"][1]
    dups_new = P.dup_ids(train + test + new)
    res["new_lines"] = {}
    print("\n12 invented lines (unit aliases, duplicate, credit note, flight, freon, tax, 3 injections): fully right")
    for a, fn in [(a, fn) for a, fn in clean_fns if a != "ours"] + [("ours", make_ours(dups_new))]:
        pr = fn(train, new)
        good = [r["line_id"] for r, p in zip(new, pr)
                if p["category"] == TN.EXPECTED[r["line_id"]][0] and abs(p["tco2e"] - TN.EXPECTED[r["line_id"]][2]) <= 0.0006]
        catok = sum(p["category"] == TN.EXPECTED[r["line_id"]][0] for r, p in zip(new, pr))
        res["new_lines"][a] = dict(right=len(good), category_right=catok, of=12, right_ids=good)
        print(f"  {a:12s} {len(good):2d} / 12 (category right {catok:2d})")

    json.dump(res, open(os.path.join(OUT, "baselines.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False,
              default=lambda o: o.item())                         # numpy ints from pandas columns
    P.write(os.path.join(OUT, "scores.csv"), res["scores"], list(res["scores"][0]))
    P.write(os.path.join(OUT, "slices.csv"), res["slices"], list(res["slices"][0]))
    ours_r = next(x for x in res["scores"] if x["approach"] == "ours" and x["scenario"] == "random 5-fold")
    assert ours_r["score"] > 0.99, "ours on random 5-fold must reproduce eval_hard (self-check of this wrapper)"
    print("\nwrote compete/out/baselines.json, scores.csv, slices.csv")


if __name__ == "__main__":
    main()
