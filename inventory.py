"""Dala Foods 2025 GHG inventory summary for the bank (train + test = the whole 2025 ledger).

Run:  python pipeline.py && python inventory.py
In:   masked/ledger_train.csv (organiser labels y_*), masked/ledger_test.csv + out/audit_test.csv (our predictions)
Out:  out/inventory_summary.json, out/inventory_summary.md, out/inventory_summary.csv (by category),
      out/inventory_lines.csv (every line: source, category, factor, quantity, method, tCO2e, data quality)
Exit code 0 = self-checks passed (scopes sum to total, every ledger line counted once, EXCLUDE = 0 t).
"""
import os, re, json, math
from collections import Counter, defaultdict
import pipeline as P

PRODUCT_T, RAW_MILK_T, TRUCKS = 105_000, 120_000, 45          # Company_Profile sheet / brief p.2, p.4
MATERIAL_SHARE = 0.01                                          # S3 category material if >= 1% of total
SCOPE = {"S1_STATIONARY": "Scope 1", "S1_MOBILE": "Scope 1", "S1_FUGITIVE": "Scope 1",
         "S2_ELECTRICITY": "Scope 2", "S2_HEAT": "Scope 2"}
DIRECT_UNIT = {"kWh": "кВт·ч", "m3": "м3", "l": "л", "kg": "кг", "t": "т", "Gcal": "Гкал", "tkm": "ткм"}
ENERGY = {"EF_DIESEL_L", "EF_DIESEL_T", "EF_PETROL_L", "EF_LPG_KG", "EF_NG_M3", "EF_GRID_KZ_KWH", "EF_HEAT_GCAL",
          "EF_R404A_KG", "EF_R134A_KG", "EF_R410A_KG", "EF_R744_KG"}
LOGISTICS = {"EF_ROAD_TKM", "EF_REEFER_TKM", "EF_RAIL_TKM", "EF_FLIGHT_DOM_PKM", "EF_FLIGHT_SHORT_PKM",
             "EF_FLIGHT_LONG_PKM", "EF_HOTEL_NIGHT", "EF_WASTE_LANDFILL_T", "EF_WASTEWATER_M3", "EF_WATER_M3"}
# data-quality score 1 (best) .. 5 and the +/- uncertainty assumed for each (activity data and factor combined, 95%)
DQ = {1: ("metered energy/fuel/refrigerant in the factor's own unit", 0.05),
      2: ("activity data needing a conversion (density, t x km, great-circle km, m3->t, piece weight) "
          "or a logistics/utility factor", 0.10),
      3: ("activity data x generic cradle-to-gate product factor (milk, packaging, chemicals, food)", 0.30),
      4: ("spend-based, item-specific factor, net of VAT", 0.50),
      5: ("spend-based fallback (unseen wording or no usable quantity)", 1.00)}
UNC_SOURCES = ("Assumed ranges, not a statistical result. Informed by IPCC 2006 Guidelines Vol.1 Ch.3 (Uncertainties; "
               "metered commercial fuel/energy activity data within a few %), IPCC 2006 Vol.2 Ch.2 (CO2 factors for "
               "fuels +/-5% or better), GHG Protocol Scope 3 Standard Ch.7 and its Quantitative Uncertainty Guidance "
               "(secondary/spend-based EEIO data: +/-50% or more). Lines sharing one factor are treated as fully "
               "correlated (their uncertainties add), different factors as independent (root-sum-square).")
INBOX = [
    ("Bank Alatau, Sustainable Finance", "Full 2025 GHG inventory for the KZT 5 bn green loan (rooftop solar + heat "
     "pumps): Scope 1 and 2 complete, all material Scope 3 categories, GHG Protocol; every figure traceable to a "
     "document and a factor; say which lines are activity- vs spend-based; data-quality view; limited assurance "
     "from next year."),
    ("EuroMart Central Asia", "Carbon footprint per litre / per kg of product for 2026 supplier scorecards."),
    ("CFO", "AI that reads the ledger every year, tool budget <= KZT 15 m/yr, a number he can trust (bank audit)."),
    ("IT & Security", "No counterparty/employee names to external cloud AI unless masked; local models fine."),
    ("Chief Accountant", "Double-posted invoices: higher line_id is the duplicate; advances (1610) are later closed "
     "by delivery documents that are also in the ledger (so advances would double count)."),
    ("Sustainability Officer", "Last year's Excel missed refrigerants and counted flights one-way."),
]


def dq_score(ef, unit, method, reason=""):
    if method == "excluded":
        return None
    if method == "spend":
        return 5 if re.search(r"new wording|fallback", reason) else 4
    direct = DIRECT_UNIT.get(P.EF[ef]["unit"]) == unit
    if ef in ENERGY:
        return 1 if direct else 2
    if ef in LOGISTICS:
        return 2
    return 3


def lines():
    train = P.load(os.path.join(P.MASKED, "ledger_train.csv"))
    test = {r["line_id"]: r for r in P.load(os.path.join(P.MASKED, "ledger_test.csv"))}
    audit = P.load(os.path.join(P.OUTD, "audit_test.csv"))
    dups = P.dup_ids(train + list(test.values()))
    out = []
    for r in train:                                           # organiser labels
        ef = r["y_ef_id"]
        method = "excluded" if r["y_category"] == "EXCLUDE" else ("spend" if ef.startswith("SP_") else "activity")
        why = ("duplicate" if r["line_id"] in dups else
               f"GL {r['gl_account']} {P.EXCLUDE_GL[r['gl_account']]}" if r["gl_account"] in P.EXCLUDE_GL else
               "labelled EXCLUDE") if method == "excluded" else "organiser label"
        out.append(dict(line_id=r["line_id"], source="train_label", category=r["y_category"], ef_id=ef,
                        activity_qty=P.num(r["y_activity_qty"]), activity_unit=r["y_activity_unit"], method=method,
                        tco2e=P.num(r["y_tco2e"]), reason=why, unit=P.canon_unit(r["unit"]), desc=r["description"],
                        gross=P.num(r["amount_kzt_gross"]), vat=P.num(r["vat_kzt"]), gl=r["gl_account"]))
    for a in audit:                                           # our predictions
        r = test[a["line_id"]]
        out.append(dict(line_id=a["line_id"], source="test_prediction", category=a["category"], ef_id=a["ef_id"],
                        activity_qty=P.num(a["activity_qty"]), activity_unit=a["activity_unit"], method=a["method"],
                        tco2e=P.num(a["tco2e"]), reason=a["reason"], unit=P.canon_unit(r["unit"]),
                        desc=r["description"], gross=P.num(r["amount_kzt_gross"]), vat=P.num(r["vat_kzt"]),
                        gl=r["gl_account"]))
    for l in out:
        l["dq"] = dq_score(l["ef_id"], l["unit"], l["method"], l["reason"])
        l["unc_pct"] = DQ[l["dq"]][1] * 100 if l["dq"] else 0
    return out, len(train), len(test)


def excl_kind(l):
    r = l["reason"]
    for k in ("duplicate", "tax", "social", "advance", "fine", "per diem"):
        if k in r:
            return k
    return "other non-purchase"


def main():
    L, n_train, n_test = lines()
    inc = [l for l in L if l["method"] != "excluded"]
    total = sum(l["tco2e"] for l in inc)
    by_cat = Counter()
    for l in inc:
        by_cat[l["category"]] += l["tco2e"]
    s1 = sum(v for c, v in by_cat.items() if c.startswith("S1"))
    s2 = sum(v for c, v in by_cat.items() if c.startswith("S2"))
    s3 = sum(v for c, v in by_cat.items() if c.startswith("S3"))

    # uncertainty: within one factor fully correlated, across factors independent
    by_ef_unc = defaultdict(float)
    for l in inc:
        by_ef_unc[l["ef_id"]] += abs(l["tco2e"]) * l["unc_pct"] / 100
    unc_t = math.sqrt(sum(v * v for v in by_ef_unc.values()))
    unc_scope = {}
    for sc in ("S1", "S2", "S3"):
        d = defaultdict(float)
        for l in inc:
            if l["category"].startswith(sc):
                d[l["ef_id"]] += abs(l["tco2e"]) * l["unc_pct"] / 100
        unc_scope[sc] = math.sqrt(sum(v * v for v in d.values()))

    act = [l for l in inc if l["method"] == "activity"]; spd = [l for l in inc if l["method"] == "spend"]
    abs_tot = sum(abs(l["tco2e"]) for l in inc)
    dq_w = sum(abs(l["tco2e"]) * l["dq"] for l in inc) / abs_tot
    dq_tiers = {t: dict(meaning=DQ[t][0], uncertainty_pct=DQ[t][1] * 100,
                        lines=sum(l["dq"] == t for l in inc),
                        tco2e=round(sum(l["tco2e"] for l in inc if l["dq"] == t), 1)) for t in DQ}

    src = Counter()
    for l in inc:
        src[(l["category"], l["ef_id"])] += l["tco2e"]
    hot = [dict(rank=i + 1, category=c, ef_id=e, factor=P.EF[e]["name"], tco2e=round(v, 1),
                share_pct=round(100 * v / total, 1), lines=sum(1 for l in inc if l["ef_id"] == e and l["category"] == c))
           for i, ((c, e), v) in enumerate(src.most_common(10))]
    top_lines = [dict(line_id=l["line_id"], source=l["source"], category=l["category"], ef_id=l["ef_id"],
                      activity=f"{l['activity_qty']:g} {l['activity_unit']}", tco2e=round(l["tco2e"], 1))
                 for l in sorted(inc, key=lambda l: -abs(l["tco2e"]))[:10]]

    exc = [l for l in L if l["method"] == "excluded"]
    exc_k = defaultdict(lambda: [0, 0.0])
    for l in exc:
        k = excl_kind(l); exc_k[k][0] += 1; exc_k[k][1] += l["gross"]
    exclusions = [dict(kind=k, lines=n, gross_kzt=round(g)) for k, (n, g) in sorted(exc_k.items(), key=lambda x: -x[1][0])]

    s3cats = {"S3_C1_GOODS": "C1 Purchased goods & services", "S3_C2_CAPITAL": "C2 Capital goods",
              "S3_C4_TRANSPORT": "C4 Upstream transport", "S3_C5_WASTE": "C5 Waste", "S3_C6_TRAVEL": "C6 Business travel"}
    materiality = [dict(category=v, tco2e=round(by_cat.get(k, 0), 1), share_pct=round(100 * by_cat.get(k, 0) / total, 2),
                        material=by_cat.get(k, 0) / total >= MATERIAL_SHARE) for k, v in s3cats.items()]
    gaps = [("C3 Fuel- and energy-related (well-to-tank, grid T&D losses)", "likely material (typically 10-25% of S1+S2); "
             "no WTT/T&D factor in emission_factors.csv"),
            ("C7 Employee commuting (~850 staff)", "not in ledger (no purchase); survey needed"),
            ("C8 Upstream leased assets", "warehouse/office rent counted in C1 spend (SP_RENT); energy of leased "
             "space not metered separately"),
            ("C9 Downstream transport not paid by Dala", "no data in purchase ledger"),
            ("C10-C12 Processing / use / end-of-life of sold products", "outside ledger; C12 packaging end-of-life "
             "relevant for PET and cartons"),
            ("C13-C15 Downstream leased assets, franchises, investments", "not applicable")]

    # completeness cross-checks against the company profile
    milk_t = sum(l["activity_qty"] for l in inc if l["ef_id"] == "EF_MILK_KG") / 1000
    diesel_l = sum(l["activity_qty"] for l in inc if l["ef_id"] == "EF_DIESEL_L" and l["category"] == "S1_MOBILE")
    diesel_l += sum(l["activity_qty"] for l in inc if l["ef_id"] == "EF_DIESEL_T" and l["category"] == "S1_MOBILE") * 1000 / 0.84
    months = {}
    for ef, key in (("EF_GRID_KZ_KWH", "electricity"), ("EF_NG_M3", "natural gas"), ("EF_HEAT_GCAL", "district heat"),
                    ("EF_WATER_M3", "water"), ("EF_WASTEWATER_M3", "wastewater"), ("EF_WASTE_LANDFILL_T", "landfill waste")):
        ms = Counter()
        for l in inc:
            if l["ef_id"] == ef:
                m = re.search(r"\b(0[1-9]|1[0-2])[./]2025", l["desc"])
                site = "office" if re.search(r"кеңсе|офис", l["desc"].lower()) else "plant/other"
                if m:
                    ms[(site, m.group(1))] += 1
        months[key] = {site: sorted({mm for s, mm in ms if s == site}) for site in sorted({s for s, _ in ms})}
    returns = [l for l in inc if l["tco2e"] < 0]

    S = dict(
        company="Dala Foods LLP (fictional, synthetic data)", year=2025, standard="GHG Protocol Corporate Standard, "
        "operational control; Scope 2 location-based", unit="tCO2e",
        boundary=f"All {n_train + n_test} lines of the 2025 purchase ledger: {n_train} train lines use organiser labels, "
                 f"{n_test} test lines use this pipeline's predictions (out/audit_test.csv). Ledger is one year split in two.",
        totals=dict(total=round(total, 1), scope1=round(s1, 1), scope2_location=round(s2, 1),
                    scope2_market=None, scope3=round(s3, 1),
                    scope2_market_note="Not determinable: no supplier-specific factors, energy attribute certificates or "
                                       "residual mix in the data. Location-based value (0.8 kg/kWh grid) is the only "
                                       "defensible figure; market-based must be added once contracts/EACs are known.",
                    by_category={c: round(v, 1) for c, v in sorted(by_cat.items(), key=lambda x: -x[1])}),
        uncertainty=dict(total_pct=round(100 * unc_t / total, 1), total_range=[round(total - unc_t), round(total + unc_t)],
                         by_scope_pct={k: round(100 * v / max(1e-9, {"S1": s1, "S2": s2, "S3": s3}[k]), 1)
                                       for k, v in unc_scope.items()}, method=UNC_SOURCES),
        methods=dict(activity_lines=len(act), spend_lines=len(spd), excluded_lines=len(exc),
                     activity_share_lines_pct=round(100 * len(act) / len(inc), 1),
                     activity_share_tco2e_pct=round(100 * sum(abs(l["tco2e"]) for l in act) / abs_tot, 1),
                     spend_share_tco2e_pct=round(100 * sum(abs(l["tco2e"]) for l in spd) / abs_tot, 1)),
        data_quality=dict(weighted_score=round(dq_w, 2), scale="1 best .. 5 worst, weighted by |tCO2e|", tiers=dq_tiers),
        hotspots=hot, top_lines=top_lines,
        intensity=dict(t_per_t_product=round(total / PRODUCT_T, 3), kg_per_kg_product=round(total / PRODUCT_T, 3),
                       kg_per_litre_product_approx=round(total / PRODUCT_T * 1.03, 3),
                       scope12_t_per_t_product=round((s1 + s2) / PRODUCT_T, 4),
                       t_per_t_raw_milk=round(total / RAW_MILK_T, 3), product_t=PRODUCT_T, raw_milk_t=RAW_MILK_T,
                       note="Company-level cradle-to-gate intensity over the ledger boundary, no allocation between "
                            "products; litre figure assumes 1.03 kg/l. Not an ISO 14067 product footprint."),
        completeness=dict(raw_milk_in_ledger_t=round(milk_t), raw_milk_profile_t=RAW_MILK_T,
                          raw_milk_coverage_pct=round(100 * milk_t / RAW_MILK_T, 1),
                          own_fleet_diesel_l=round(diesel_l), trucks=TRUCKS,
                          diesel_l_per_truck=round(diesel_l / TRUCKS), utility_months=months,
                          refrigerant_lines=sum(1 for l in inc if l["category"] == "S1_FUGITIVE"),
                          flights_all_legs="yes (great-circle per leg; return trips = 2 legs)",
                          returns=dict(lines=len(returns), tco2e=round(sum(l["tco2e"] for l in returns), 2))),
        exclusions=exclusions, scope3_materiality=materiality,
        scope3_gaps=[dict(category=a, status=b) for a, b in gaps],
        inbox_requests=[dict(sender=a, ask=b) for a, b in INBOX])

    # ---------- self-checks ----------
    assert len(L) == n_train + n_test == len({l["line_id"] for l in L}), "every ledger line exactly once"
    assert abs(s1 + s2 + s3 - total) < 1e-6, "scopes must sum to total"
    assert abs(sum(by_cat.values()) - total) < 1e-6
    assert all(l["tco2e"] == 0 for l in exc), "EXCLUDE must carry 0 t"
    assert all(l["category"] in P.CATS for l in L)
    assert len(act) + len(spd) + len(exc) == len(L)
    assert sum(t["lines"] for t in dq_tiers.values()) == len(inc)

    os.makedirs(P.OUTD, exist_ok=True)
    with open(os.path.join(P.OUTD, "inventory_summary.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, ensure_ascii=False, indent=2)
    rows = [dict(scope=SCOPE.get(c, "Scope 3"), category=c, tco2e=round(v, 2), share_pct=round(100 * v / total, 2),
                 lines=sum(l["category"] == c for l in inc),
                 activity_lines=sum(l["category"] == c and l["method"] == "activity" for l in inc),
                 spend_lines=sum(l["category"] == c and l["method"] == "spend" for l in inc),
                 weighted_dq=round(sum(abs(l["tco2e"]) * l["dq"] for l in inc if l["category"] == c) /
                                   max(1e-9, sum(abs(l["tco2e"]) for l in inc if l["category"] == c)), 2))
            for c, v in sorted(by_cat.items(), key=lambda x: (SCOPE.get(x[0], "Scope 3"), -x[1]))]
    rows.append(dict(scope="TOTAL", category="", tco2e=round(total, 2), share_pct=100.0, lines=len(inc),
                     activity_lines=len(act), spend_lines=len(spd), weighted_dq=round(dq_w, 2)))
    P.write(os.path.join(P.OUTD, "inventory_summary.csv"), rows, list(rows[0]))
    P.write(os.path.join(P.OUTD, "inventory_lines.csv"), L, ["line_id", "source", "category", "ef_id", "activity_qty",
            "activity_unit", "method", "tco2e", "dq", "unc_pct", "reason"])
    with open(os.path.join(P.OUTD, "inventory_summary.md"), "w", encoding="utf-8") as f:
        f.write(md(S, rows))
    t = S["totals"]
    print(f"total {t['total']:.0f} t = S1 {t['scope1']:.0f} + S2 {t['scope2_location']:.0f} + S3 {t['scope3']:.0f}; "
          f"+/-{S['uncertainty']['total_pct']}%; {S['intensity']['t_per_t_product']} t/t product; DQ {dq_w:.2f}")
    print("INVENTORY SELF-CHECKS OK")


def md(S, rows):
    t, m, u, i, c = S["totals"], S["methods"], S["uncertainty"], S["intensity"], S["completeness"]
    o = [f"# Dala Foods LLP: 2025 GHG inventory (summary for Bank Alatau)",
         f"*Synthetic case data. {S['standard']}. {S['boundary']}*", "",
         f"**Total {t['total']:,.0f} tCO2e** (range {u['total_range'][0]:,}-{u['total_range'][1]:,}, +/-{u['total_pct']}%). "
         f"Scope 1 {t['scope1']:,.0f} | Scope 2 (location) {t['scope2_location']:,.0f} | Scope 3 {t['scope3']:,.0f}. "
         f"Intensity **{i['t_per_t_product']} tCO2e per t of product** ({i['product_t']:,} t), "
         f"{i['t_per_t_raw_milk']} per t raw milk; Scope 1+2 only {i['scope12_t_per_t_product']} t/t.", "",
         "| Scope | Category | tCO2e | Share | Lines (act/spend) | DQ |", "|---|---|---:|---:|---|---:|"]
    o += [f"| {r['scope']} | {r['category']} | {r['tco2e']:,.1f} | {r['share_pct']}% | {r['lines']} "
          f"({r['activity_lines']}/{r['spend_lines']}) | {r['weighted_dq']} |" for r in rows]
    o += ["", f"**Activity vs spend:** {m['activity_share_tco2e_pct']}% of tCO2e and {m['activity_share_lines_pct']}% of "
          f"lines are activity-based; spend-based (net of VAT) {m['spend_share_tco2e_pct']}% of tCO2e. "
          f"Market-based Scope 2: not determinable ({t['scope2_market_note'].split(':')[1].strip().split('.')[0]}).", "",
          "**Hotspots (by factor):** " + "; ".join(f"{h['rank']}. {h['factor']} {h['tco2e']:,.0f} t ({h['share_pct']}%)"
                                                     for h in S["hotspots"]), "",
          f"**Data quality** (1 best..5, |t|-weighted): **{S['data_quality']['weighted_score']}**. " +
          "; ".join(f"DQ{k} +/-{v['uncertainty_pct']:g}%: {v['lines']} lines, {v['tco2e']:,.0f} t"
                    for k, v in S["data_quality"]["tiers"].items()) + ".", "",
          f"**Uncertainty method:** {u['method']}", "",
          "**Scope 3 materiality** (material = >=1% of total): " +
          "; ".join(f"{x['category']} {x['share_pct']}%{' (material)' if x['material'] else ''}" for x in S["scope3_materiality"]) +
          ". **Not covered:** " + "; ".join(f"{g['category']}: {g['status']}" for g in S["scope3_gaps"][:2]) + ".", "",
          "**Excluded (0 t):** " + "; ".join(f"{e['kind']} {e['lines']} lines (KZT {e['gross_kzt'] / 1e6:,.1f} m)"
                                             for e in S["exclusions"]) +
          ". Advances (1610) are excluded because the delivery documents that close them are in the ledger.", "",
          f"**Completeness checks:** raw milk in ledger {c['raw_milk_in_ledger_t']:,} t vs ~{c['raw_milk_profile_t']:,} t "
          f"in profile ({c['raw_milk_coverage_pct']}%); own-fleet diesel {c['own_fleet_diesel_l']:,} l "
          f"= {c['diesel_l_per_truck']:,} l per truck ({c['trucks']} trucks); refrigerant lines {c['refrigerant_lines']}; "
          f"flights: all legs; credit notes {c['returns']['lines']} lines ({c['returns']['tco2e']} t).", "",
          "**Traceability:** every number rolls up from out/inventory_lines.csv (line_id, factor, quantity, method, DQ, "
          "reason) to the ledger document; test-line reasons in out/audit_test.csv.",
          "", "**Inbox asks covered:** bank (scopes, traceability, activity/spend, DQ), retailer (intensity per kg/l), "
          "CFO (one command, no paid API), IT (masked data only), chief accountant (duplicates, advances), "
          "sustainability officer (refrigerants in, flights all legs)."]
    return "\n".join(o) + "\n"


if __name__ == "__main__":
    main()
