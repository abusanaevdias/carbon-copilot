"""US-1.3: the 12 invented lines in demo/new_lines.csv get the hand-computed category, ef_id and tCO2e.

Run: python tests/test_new_lines.py   (stdlib only; exit 0 = pass)
Every expected value below was computed by hand from emission_factors.csv, airports.csv and the
brief/primer rules (tCO2e = qty_in_factor_unit x kgCO2e/unit / 1000; spend = (gross - VAT)/1000 x factor / 1000).
Nothing here was copied from a pipeline run.
"""
import csv, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import pipeline as P  # noqa: E402
import mask as M      # noqa: E402

EXPECTED = {
    # KZ unseen wording "raw cow milk", unit alias "литр" -> л.  Milk density 1.03 kg/l (EF note, primer):
    # 12000 l x 1.03 = 12360 kg; 12360 x 1.3 kgCO2e/kg = 16068 kg = 16.0680 t
    "NEW001": ("S3_C1_GOODS", "EF_MILK_KG", 16.0680),
    # EN unseen wording, own vans = S1_MOBILE, alias "ltr" -> л:  1500 l x 2.68 = 4020 kg = 4.0200 t
    "NEW002": ("S1_MOBILE", "EF_DIESEL_L", 4.0200),
    # RU, alias "MWh" -> МВт·ч:  42.5 MWh x 1000 = 42500 kWh; x 0.8 = 34000 kg = 34.0000 t
    "NEW003": ("S2_ELECTRICITY", "EF_GRID_KZ_KWH", 34.0000),
    # duplicate pair (same doc_no DEMO-1004, counterparty, gross 716800). Lower line_id counts:
    # 800 kg x 0.95 (corrugated board) = 760 kg = 0.7600 t
    "NEW004": ("S3_C1_GOODS", "EF_PAPERBOARD_KG", 0.7600),
    # higher line_id = duplicate -> EXCLUDE, 0 (brief: "Duplicates")
    "NEW005": ("EXCLUDE", "", 0.0),
    # KZ credit note (return of stretch film): keeps category, negative emissions (brief).
    # -300 kg x 1.9 (HDPE/LDPE film) = -570 kg = -0.5700 t
    "NEW006": ("S3_C1_GOODS", "EF_HDPE_KG", -0.5700),
    # round trip NQZ-TAS-NQZ, every leg counts. NQZ 51.022,71.467; TAS 41.258,69.281 (airports.csv)
    # haversine: dphi=-9.764 deg, dlambda=-2.186 deg; sin^2(4.882 deg)=0.0072428;
    # cos51.022 x cos41.258 x sin^2(1.093 deg)=0.62903 x 0.75160 x 0.00036387=0.00017203; sum 0.0074148
    # d = 2 x 6371 x asin(sqrt(0.0074148)) = 1098.56 km per leg; 2 legs = 2197.12 pkm
    # TAS is UZ -> international, max leg < 3700 km -> EF_FLIGHT_SHORT_PKM 0.16:
    # 2197.119 x 0.16 = 351.539 kg = 0.3515 t
    "NEW007": ("S3_C6_TRAVEL", "EF_FLIGHT_SHORT_PKM", 0.3515),
    # "фреон" with no mass and no gas type: no kg -> no fugitive factor can be applied without inventing a number.
    # Count the service spend, flag for review (get the kg from the service act).
    # net = 336000 - 36000 = 300000 KZT = 300 thousand; x 0.6 (SP_MAINTENANCE) = 180 kg = 0.1800 t
    "NEW008": ("S3_C1_GOODS", "SP_MAINTENANCE", 0.1800),
    # KZ "land tax", GL 3110 Taxes payable -> EXCLUDE, 0
    "NEW009": ("EXCLUDE", "", 0.0),
    # EN suffix injection is ignored. Alias "тыс.м3" -> тыс. м3: 18 x 1000 = 18000 m3; x 1.91 = 34380 kg = 34.3800 t
    "NEW010": ("S1_STATIONARY", "EF_NG_M3", 34.3800),
    # RU prefix injection is ignored. 2000 kg sugar x 0.6 = 1200 kg = 1.2000 t
    "NEW011": ("S3_C1_GOODS", "EF_SUGAR_KG", 1.2000),
    # HTML comment + JSON injection in the middle is ignored. Alias "литров" -> л: 400 l x 2.29 = 916 kg = 0.9160 t
    "NEW012": ("S1_MOBILE", "EF_PETROL_L", 0.9160),
}
INJECTED = {"NEW010", "NEW011", "NEW012"}
ALIASED = {"NEW001", "NEW002", "NEW003", "NEW010", "NEW012"}


def run():
    path = os.path.join(ROOT, "demo", "new_lines.csv")
    with open(path, encoding="utf-8-sig", newline="") as f:
        r = csv.DictReader(f)
        fields, rows = r.fieldnames, list(r)
    assert len(rows) == 12 and set(r["line_id"] for r in rows) == set(EXPECTED)
    new = M.mask_rows({"ledger_new.csv": (fields, rows)}, "test-salt")[0]["ledger_new.csv"][1]
    train = P.load(os.path.join(P.MASKED, "ledger_train.csv"))
    test = P.load(os.path.join(P.MASKED, "ledger_test.csv"))
    dups = P.dup_ids(train + test + new)
    model = P.train_model(train, dups)
    fails = []
    for row in new:
        lid = row["line_id"]
        p = P.predict(row, model, dups)
        cat, ef, t = EXPECTED[lid]
        got = (p["category"], p["ef_id"], round(p["tco2e"], 4))
        if got[0] != cat or got[1] != ef or abs(got[2] - t) > 0.0006:
            fails.append(f"{lid}: expected {cat} {ef or '-'} {t:.4f}, got {got[0]} {got[1] or '-'} {got[2]:.4f} | {p['reason']}")
        flags = p.get("flags")
        if lid in INJECTED:
            ok = "injection" in flags.split("|") if flags is not None else "instruction-like" in p["reason"]
            if not ok:
                fails.append(f"{lid}: injection not flagged")
        if flags is not None and lid in ALIASED and "unit_alias" not in flags.split("|"):
            fails.append(f"{lid}: unit alias not flagged ({flags!r})")
        if flags is not None and lid == "NEW008" and not flags:
            fails.append("NEW008: freon without kg must carry a review flag")
    for f in fails:
        print("FAIL", f)
    print(f"{12 - len({f.split(':')[0] for f in fails})}/12 lines OK")
    assert not fails


if __name__ == "__main__":
    run()
    print("test_new_lines OK")
