"""Verifier pack (PLAN_V5 §A4): out/verifier_pack/{factors,reconcile,sample,recompute}.csv from masked outputs only.

Run:  python verifier_pack.py --seed <verifier's number> [--n-random 25]
Reads only out/audit_all.csv and out/inventory.json (both masked). Standard library. Does not import pipeline.py.
The verifier owns the sample: their seed, their size. The pack saves retrieval, not judgement.
"""
import argparse, csv, json, os, random, re

CASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(CASE, "out")
PACK = os.path.join(OUT, "verifier_pack")
QUEUE_RE = r"new_wording|outlier|injection"          # same review-queue rule as deck/build.py numbers()
BIG_SHARE = 0.0025                                      # every line >= 0.25% of the total is sampled
# Independent totals we hold today. [brief] Case2_Student_Brief.txt: "about 120,000 t of raw milk a year".
INDEPENDENT = {"EF_MILK_KG": (120_000_000, "kg", "company profile in the case brief: about 120,000 t a year")}


def write(name, fields, rows):
    with open(os.path.join(PACK, name), "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(fields)
        w.writerows(rows)


def conversion(a):
    """Ledger quantity -> factor's activity unit. Spend lines: net KZT -> thousand KZT."""
    if a["method"] == "spend":
        return "net_kzt", float(a["net_kzt"]), 0.001
    q = float(a["quantity"])
    return "quantity", q, float(a["activity_qty"]) / q


def recompute_t(base, conv, ef):
    return base * conv * ef / 1000


def main(seed, n_random):
    os.makedirs(PACK, exist_ok=True)
    inv = json.load(open(os.path.join(OUT, "inventory.json"), encoding="utf-8"))
    aud = list(csv.DictReader(open(os.path.join(OUT, "audit_all.csv"), encoding="utf-8-sig")))
    inc = [a for a in aud if a["category"] != "EXCLUDE"]
    total = inv["total"]

    # 1. factors.csv: which factors carry the tonnes
    fac = {}
    for a in inc:
        f = fac.setdefault(a["ef_id"], dict(value=a["ef_value"], unit=a["ef_unit"], act_unit=a["activity_unit"], lines=0, qty=0.0, t=0.0))
        f["lines"] += 1; f["qty"] += float(a["activity_qty"]); f["t"] += float(a["tco2e"])
    rows, cum = [], 0.0
    for k, f in sorted(fac.items(), key=lambda kv: -kv[1]["t"]):
        cum += f["t"]
        rows.append([k, f["value"], f"kgCO2e per {f['unit']}", f["lines"], f"{f['qty']:.4f}", f["act_unit"], f"{f['t']:.4f}",
                     f"{100 * f['t'] / total:.2f}", f"{100 * cum / total:.2f}"])
    write("factors.csv", ["ef_id", "ef_value", "ef_unit", "lines", "activity_qty_total", "activity_unit", "tco2e", "share_pct", "cum_share_pct"], rows)
    assert abs(sum(f["t"] for f in fac.values()) - total) < 0.1, "factors.csv does not sum to inventory total"
    assert float(rows[2][-1]) >= 90, "top 3 factors < 90%: slide claim no longer holds"
    assert len(fac) == len(inv["by_factor"]), "factor count differs from inventory.json"

    # 2. reconcile.csv: ledger quantity per factor against an independent total where one exists
    rec = []
    for k, f in sorted(fac.items(), key=lambda kv: -kv[1]["t"]):
        if k in INDEPENDENT:
            ind, unit, src = INDEPENDENT[k]
            rec.append([k, f"{f['qty']:.2f}", f["act_unit"], ind, unit, f"{100 * (f['qty'] - ind) / ind:+.1f}", src])
        else:
            rec.append([k, f"{f['qty']:.2f}", f["act_unit"], "", "", "", "independent total: to be supplied by Dala (utility bills, fuel cards, supplier statements)"])
    write("reconcile.csv", ["ef_id", "ledger_activity_qty", "unit", "independent_qty", "independent_unit", "diff_pct", "independent_source"], rec)

    # 3. sample.csv: every queue line + every big line + a seeded random stratum of the rest
    queue = [a for a in inc if re.search(QUEUE_RE, a["flags"])]
    assert len(queue) == inv["checks"]["review_queue"]["lines"], "review queue differs from inventory.json"
    big = [a for a in inc if float(a["tco2e"]) >= BIG_SHARE * total and a not in queue]
    chosen = {a["line_id"] for a in queue + big}
    rest = sorted((a for a in inc if a["line_id"] not in chosen), key=lambda a: a["line_id"])
    rnd = random.Random(seed).sample(rest, min(n_random, len(rest)))
    sample = [("review_queue", a) for a in queue] + [("big_line", a) for a in big] + \
             [("random", a) for a in sorted(rnd, key=lambda a: a["line_id"])]
    cols = ["line_id", "doc_no", "posting_date", "counterparty", "description", "category", "ef_id", "tco2e", "flags"]
    write("sample.csv", ["stratum", "seed"] + cols, [[s, seed] + [a[c] for c in cols] for s, a in sample])

    # 4. recompute.csv: for each sampled line, an Excel formula that recomputes tCO2e next to our value
    hdr = ["line_id", "doc_no", "base", "base_value", "unit", "conversion", "activity_unit", "ef_id", "ef_value", "ef_unit",
           "our_tco2e", "recomputed_tco2e", "diff_t"]
    rr, worst = [], 0.0
    for i, (_, a) in enumerate(sample, start=2):
        base, bv, conv = conversion(a)
        rr.append([a["line_id"], a["doc_no"], base, repr(bv), a["unit"] if base == "quantity" else "KZT", repr(conv), a["activity_unit"],
                   a["ef_id"], a["ef_value"], f"kgCO2e per {a['ef_unit']}", a["tco2e"], f"=D{i}*F{i}*I{i}/1000", f"=ROUND(L{i}-K{i},4)"])
    write("recompute.csv", hdr, rr)

    # self-check on ALL included lines, not only the sample: the formula above, evaluated from the written strings
    for a in inc:
        _, bv, conv = conversion(a)
        worst = max(worst, abs(recompute_t(float(repr(bv)), float(repr(conv)), float(a["ef_value"])) - float(a["tco2e"])))
    assert worst < 5e-5, f"recompute differs from tco2e by {worst}"
    print(f"verifier pack: {PACK}\n  factors {len(rows)} (top 3 = {rows[2][-1]}%, top 5 = {rows[4][-1]}%, top 10 = {rows[9][-1]}%)"
          f"\n  sample {len(sample)} = queue {len(queue)} + big {len(big)} + random {len(rnd)} (seed {seed})"
          f"\n  recompute: max |diff| on all {len(inc)} included lines = {worst:.2e} t")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True, help="chosen by the verifier")
    p.add_argument("--n-random", type=int, default=25)
    a = p.parse_args()
    main(a.seed, a.n_random)
