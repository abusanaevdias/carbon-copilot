"""Carbon Copilot report: audit_all.csv, inventory.json, dashboard.html, bank_summary.html (+ .pdf if Edge/Chrome found).

Run:  python report.py                       (after pipeline.py; reads masked/ + out/audit_*.csv)
      python report.py --new-ledger masked_new.csv --new-audit audit_new.csv   (extra lines, split "new")
Python API: report.main(new_ledger=None, new_audit=None, pdf=True) -> inventory dict.
Only MASKED text reaches the outputs. Every number on the pages comes from inventory.json / audit_all.csv.
Exit 0 = self-checks passed (lines sum to total, categories sum to scopes, formulas recompute, 1-page bank PDF).
"""
import base64, glob, hashlib, html, json, os, re, shutil, subprocess, sys, time, tempfile
from collections import Counter, defaultdict
from string import Template
import pipeline as P
from inventory import PRODUCT_T, RAW_MILK_T

HERE = os.path.dirname(os.path.abspath(__file__))
TPL = os.path.join(HERE, "templates")
YEAR, MILK_SENS = 2025, 0.20
SCOPE_OF = lambda c: "-" if c == "EXCLUDE" else c[1]
CAT_NAME = {"S1_STATIONARY": "Stationary combustion", "S1_MOBILE": "Mobile combustion (own fleet)",
            "S1_FUGITIVE": "Refrigerant leaks", "S2_ELECTRICITY": "Purchased electricity", "S2_HEAT": "Purchased heat",
            "S3_C1_GOODS": "Cat 1 Purchased goods & services", "S3_C2_CAPITAL": "Cat 2 Capital goods",
            "S3_C4_TRANSPORT": "Cat 4 Upstream transport", "S3_C5_WASTE": "Cat 5 Waste & wastewater",
            "S3_C6_TRAVEL": "Cat 6 Business travel"}
FLAG_RULES = [("new_wording", r"new wording"), ("injection", r"instruction-like"), ("spend_fallback", r"fallback"),
              ("lexicon", r"^rule:")]
REVIEW = {"new_wording", "injection", "spend_fallback", "unit_alias", "outlier"}
TIER_B = re.compile(r"density|pcs x|great-circle|bulk density|from description|t x [\d.,]+ km|x 0\.84")
NOT_COVERED = [("Cat 3 Fuel- and energy-related", "well-to-tank and grid-loss factors; not in the factor file"),
               ("Cat 7 Employee commuting", "~850 staff; needs a commuting survey, no purchase in the ledger"),
               ("Cat 9 Downstream transport", "transport paid by customers; not in a purchase ledger"),
               ("Cat 12 End-of-life of sold packaging", "packaging tonnage sold and disposal routes")]
FIELDS = ["line_id", "split", "posting_date", "doc_no", "counterparty", "description", "gl_account", "quantity", "unit",
          "net_kzt", "category", "scope", "ef_id", "ef_value", "ef_unit", "activity_qty", "activity_unit", "method",
          "quality", "tco2e", "flags", "reason"]


def quality(a):
    if a.get("quality"):
        return a["quality"]
    if a["method"] == "excluded":
        return "-"
    if a["method"] == "spend":
        return "C"
    return "B" if TIER_B.search(a["reason"]) else "A"


def flags(a, raw_unit):
    if a.get("flags") is not None and "flags" in a:
        return a["flags"]
    f = [k for k, rx in FLAG_RULES if re.search(rx, a["reason"])]
    if raw_unit.strip() and raw_unit.strip() != P.canon_unit(raw_unit):
        f.append("unit_alias")
    return "|".join(f)


def build_lines(new_ledger=None, new_audit=None):
    parts = [("train", "ledger_train.csv", os.path.join(P.OUTD, "audit_train_cv.csv")),
             ("test", "ledger_test.csv", os.path.join(P.OUTD, "audit_test.csv"))]
    parts = [(s, os.path.join(P.MASKED, l), a) for s, l, a in parts]
    if new_ledger and new_audit:
        parts.append(("new", new_ledger, new_audit))
    out, labels, inputs = [], {}, []
    for split, lp, ap in parts:
        led = {r["line_id"]: r for r in P.load(lp)}
        inputs += [lp, ap]
        for a in P.load(ap):
            r = led[a["line_id"]]
            if "y_tco2e" in r:
                labels[a["line_id"]] = P.num(r["y_tco2e"])
            ef = P.EF.get(a["ef_id"], {})
            out.append(dict(line_id=a["line_id"], split=split, posting_date=r["posting_date"], doc_no=r["doc_no"],
                            counterparty=r["counterparty"], description=r["description"], gl_account=r["gl_account"],
                            quantity=P.num(r["quantity"]), unit=r["unit"],
                            net_kzt=round(P.num(r["amount_kzt_gross"]) - P.num(r["vat_kzt"]), 2),
                            category=a["category"], scope=SCOPE_OF(a["category"]), ef_id=a["ef_id"],
                            ef_value=float(ef["kgCO2e_per_unit"]) if ef else 0.0, ef_unit=ef.get("unit", ""),
                            activity_qty=P.num(a["activity_qty"]), activity_unit=a["activity_unit"], method=a["method"],
                            quality=quality(a), tco2e=P.num(a["tco2e"]), flags=flags(a, r["unit"]), reason=a["reason"]))
    mark_outliers(out)
    return out, labels, inputs


def mark_outliers(L):
    """t per KZT net outside 0.2x-5x of the train range of the same factor -> 'outlier' (review, never changed)."""
    rng = defaultdict(list)
    for l in L:
        if l["split"] == "train" and l["method"] == "activity" and l["net_kzt"] > 0 and l["tco2e"] > 0:
            rng[l["ef_id"]].append(l["tco2e"] / l["net_kzt"])
    for l in L:
        v = rng.get(l["ef_id"])
        if l["split"] != "train" and v and l["method"] == "activity" and l["net_kzt"] > 0 and l["tco2e"] > 0:
            x = l["tco2e"] / l["net_kzt"]
            if not 0.2 * min(v) <= x <= 5 * max(v) and "outlier" not in l["flags"]:
                l["flags"] = "|".join(filter(None, [l["flags"], "outlier"]))


def excl_reason(r):
    if r.startswith("duplicate"):
        return "duplicate"
    m = re.search(r"= ([a-z -]+?)(,|$| not)", r)
    return m.group(1) if m else "other non-purchase"


def summarise(L, labels, inputs, t0):
    inc = [l for l in L if l["method"] != "excluded"]
    exc = [l for l in L if l["method"] == "excluded"]
    total = sum(l["tco2e"] for l in inc)
    abs_tot = sum(abs(l["tco2e"]) for l in inc) or 1
    by_scope = {s: round(sum(l["tco2e"] for l in inc if l["scope"] == s), 4) for s in "123"}
    cats = []
    for c in [c for c in P.CATS if c != "EXCLUDE"]:
        ls = [l for l in inc if l["category"] == c]
        cats.append(dict(category=c, name=CAT_NAME.get(c, c), scope=SCOPE_OF(c), lines=len(ls),
                         tco2e=round(sum(l["tco2e"] for l in ls), 4),
                         activity_tco2e=round(sum(l["tco2e"] for l in ls if l["method"] == "activity"), 4),
                         spend_tco2e=round(sum(l["tco2e"] for l in ls if l["method"] == "spend"), 4),
                         tiers={q: dict(lines=sum(l["quality"] == q for l in ls),
                                        tco2e=round(sum(l["tco2e"] for l in ls if l["quality"] == q), 4)) for q in "ABC"}))
    fac = defaultdict(lambda: [0, 0.0, set()])
    for l in inc:
        f = fac[l["ef_id"]]; f[0] += 1; f[1] += l["tco2e"]; f[2].add(l["category"])
    by_factor = [dict(ef_id=e, name=P.EF[e]["name"], unit=P.EF[e]["unit"], value=float(P.EF[e]["kgCO2e_per_unit"]),
                      lines=n, tco2e=round(t, 4), share_pct=round(100 * t / total, 2), categories=sorted(cs))
                 for e, (n, t, cs) in sorted(fac.items(), key=lambda x: -x[1][1])]
    act = [l for l in inc if l["method"] == "activity"]; spd = [l for l in inc if l["method"] == "spend"]
    method_share = dict(activity_lines=len(act), spend_lines=len(spd), excluded_lines=len(exc),
                        activity_tco2e=round(sum(l["tco2e"] for l in act), 4), spend_tco2e=round(sum(l["tco2e"] for l in spd), 4),
                        activity_pct_lines=round(100 * len(act) / len(inc), 2),
                        activity_pct_tco2e=round(100 * sum(abs(l["tco2e"]) for l in act) / abs_tot, 2))
    quality_ = {q: dict(lines=sum(l["quality"] == q for l in inc), tco2e=round(sum(l["tco2e"] for l in inc if l["quality"] == q), 4),
                        pct_tco2e=round(100 * sum(abs(l["tco2e"]) for l in inc if l["quality"] == q) / abs_tot, 2))
                for q in "ABC"}
    quality_["definition"] = {"A": "physical quantity in the factor's unit or a pure unit scale",
                              "B": "physical quantity through an assumption (density, piece weight, bulk density, "
                                   "great-circle distance, quantity read from the description)",
                              "C": "spend-based, net of VAT"}
    queue = [l for l in L if REVIEW & set(l["flags"].split("|"))]
    milk = sum(l["tco2e"] for l in inc if l["ef_id"] == "EF_MILK_KG")
    tok = lambda kind: len({m for l in L for col in ("counterparty", "description")
                            for m in re.findall(kind + r"_[0-9a-f]{8}", l[col])})
    train_pred = sum(l["tco2e"] for l in L if l["split"] == "train")
    credit = [l for l in inc if l["tco2e"] < 0]
    excl = Counter("duplicate" if l["reason"].startswith("duplicate") else "non_purchase" for l in exc)
    excl_kind = Counter(excl_reason(l["reason"]) for l in exc)
    sha = hashlib.sha256()
    for p in inputs:
        with open(p, "rb") as f:
            sha.update(f.read())
    return dict(
        meta=dict(year=YEAR, entity="Dala Foods LLP (case data, synthetic)", generated_at=time.strftime("%Y-%m-%d %H:%M"),
                  rows=len(L), rows_by_split=dict(Counter(l["split"] for l in L)), product_t=PRODUCT_T,
                  input_sha256=sha.hexdigest(), runtime_s=round(time.time() - t0, 2),
                  boundary="Operational control: plant, warehouse, head office. Whole 2025 purchase ledger "
                           "(train lines = out-of-fold predictions, test lines = predictions).",
                  method="GHG Protocol Corporate Standard; Scope 2 location-based; factors = emission_factors.csv "
                         "(organiser values for this case)"),
        total=round(total, 4), by_scope=by_scope, by_category=cats, by_factor=by_factor, method_share=method_share,
        quality=quality_, intensity=dict(t_per_t_product=round(total / PRODUCT_T, 3), product_t=PRODUCT_T,
                                         note="Corporate-inventory intensity, not an ISO 14067 product footprint "
                                              "(no allocation between milk, kefir, yoghurt and cream)."),
        checks=dict(duplicates=excl["duplicate"], non_purchase=excl["non_purchase"],
                    exclusions=dict(sorted(excl_kind.items(), key=lambda x: -x[1])),
                    credit_notes=dict(lines=len(credit), tco2e=round(sum(l["tco2e"] for l in credit), 4)),
                    injection_flagged=sum("injection" in l["flags"] for l in L),
                    masked_cp=tok("CP"), masked_persons=tok("PERSON"),
                    review_queue=dict(lines=len(queue), tco2e=round(sum(l["tco2e"] for l in queue), 4),
                                      by_flag=dict(Counter(f for l in queue for f in l["flags"].split("|") if f in REVIEW))),
                    milk_t_ledger=round(sum(l["activity_qty"] for l in inc if l["ef_id"] == "EF_MILK_KG") / 1000),
                    milk_t_profile=RAW_MILK_T,
                    train_label_delta_t=round(train_pred - sum(labels.values()), 4) if labels else None),
        sensitivity=dict(factor="EF_MILK_KG", factor_value=float(P.EF["EF_MILK_KG"]["kgCO2e_per_unit"]),
                         factor_note=P.EF["EF_MILK_KG"]["note"], change_pct=MILK_SENS * 100,
                         milk_tco2e=round(milk, 4), milk_share_pct=round(100 * milk / total, 2),
                         low=round(total - MILK_SENS * milk, 1), high=round(total + MILK_SENS * milk, 1),
                         total_change_pct=round(100 * MILK_SENS * milk / total, 1),
                         note="Sensitivity to one factor, not an uncertainty range."),
        not_covered=[dict(category=a, data_needed=b) for a, b in NOT_COVERED])


# ---------- pages ----------
def font_css():
    faces, rng = [], {"latin": "U+0000-00FF, U+0131, U+0152-0153, U+02BB-02BC, U+02C6, U+02DA, U+02DC, U+0304, U+0308, "
                               "U+0329, U+2000-206F, U+20AC, U+2122, U+2191, U+2193, U+2212, U+2215, U+FEFF, U+FFFD",
                      "latin-ext": "U+0100-02BA, U+02BD-02C5, U+02C7-02CC, U+02CE-02D7, U+02DD-02FF, U+1E00-1E9F, "
                                   "U+2020, U+20A0-20AB, U+20AD-20C4, U+2113, U+2C60-2C7F, U+A720-A7FF",
                      "cyrillic": "U+0301, U+0400-045F, U+0490-0491, U+04B0-04B1, U+2116",
                      "cyrillic-ext": "U+0460-052F, U+1C80-1C8A, U+20B4, U+2DE0-2DFF, U+A640-A69F, U+FE2E-FE2F"}
    for sub, r in rng.items():
        with open(os.path.join(TPL, "fonts", f"onest-{sub}.woff2"), "rb") as f:
            b = base64.b64encode(f.read()).decode()
        faces.append(f"@font-face{{font-family:Onest;font-style:normal;font-weight:100 900;font-display:block;"
                     f"src:url(data:font/woff2;base64,{b}) format('woff2');unicode-range:{r}}}")
    return "\n".join(faces)


def fill(name, **kw):
    with open(os.path.join(TPL, name), encoding="utf-8") as f:
        return Template(f.read()).substitute(fonts=font_css(), **kw)


E = lambda s: html.escape(str(s))
T = lambda x: f"{x:,.0f}"
PCT = lambda x, tot: f"{100 * x / tot:.1f}%" if tot else "0%"


def bank_html(S):
    tot = S["total"]; bs = S["by_scope"]; ms = S["method_share"]; ck = S["checks"]; sn = S["sensitivity"]
    cat_rows = []
    for s in "123":
        cat_rows.append(f'<tr class="sc"><td>Scope {s}{" (location-based)" if s == "2" else ""}</td>'
                        f'<td class="n">{sum(c["lines"] for c in S["by_category"] if c["scope"] == s)}</td>'
                        f'<td class="n">{T(bs[s])}</td><td class="n">{PCT(bs[s], tot)}</td></tr>')
        for c in sorted((c for c in S["by_category"] if c["scope"] == s), key=lambda c: -c["tco2e"]):
            cat_rows.append(f'<tr><td class="in">{E(c["name"])}</td><td class="n">{c["lines"]}</td>'
                            f'<td class="n">{c["tco2e"]:,.0f}</td><td class="n">{PCT(c["tco2e"], tot)}</td></tr>')
    top = S["by_factor"][:5]
    mx = max(f["tco2e"] for f in top)
    hot = "".join(f'<tr><td>{i + 1}. {E(f["name"].split(" (")[0])}</td>'
                  f'<td class="bar"><i style="width:{100 * f["tco2e"] / mx:.1f}%"{" class=hot" if i == 0 else ""}></i></td>'
                  f'<td class="n">{T(f["tco2e"])}</td><td class="n">{f["share_pct"]:.1f}%</td></tr>'
                  for i, f in enumerate(top))
    q = S["quality"]
    short = {"A": "Physical quantity in the factor's unit", "B": "Physical quantity via an assumption (density, "
             "piece weight, distance)", "C": "Spend-based, net of VAT"}
    tiers = "".join(f'<tr><td><b>{k}</b> {short[k]}</td><td class="n">{q[k]["lines"]}</td>'
                    f'<td class="n">{T(q[k]["tco2e"])}</td><td class="n">{q[k]["pct_tco2e"]:.1f}%</td></tr>' for k in "ABC")
    ex = "; ".join(f"{E(k)} {n}" for k, n in ck["exclusions"].items())
    nc = "".join(f'<li><b>{E(x["category"])}</b>: {E(x["data_needed"])}</li>' for x in S["not_covered"])
    sp_pct = 100 - ms["activity_pct_tco2e"]
    return fill("bank_summary.html", year=S["meta"]["year"], entity=E(S["meta"]["entity"]),
                boundary=E(S["meta"]["boundary"]), method=E(S["meta"]["method"]), total=T(tot),
                s1=T(bs["1"]), s2=T(bs["2"]), s3=T(bs["3"]), p1=PCT(bs["1"], tot), p2=PCT(bs["2"], tot), p3=PCT(bs["3"], tot),
                sens_low=T(sn["low"]), sens_high=T(sn["high"]), sens_pct=f'{sn["change_pct"]:g}', sens_tot=sn["total_change_pct"],
                milk_share=f'{sn["milk_share_pct"]:.0f}', milk_ef=sn["factor_value"],
                intensity=f'{S["intensity"]["t_per_t_product"]:.2f}', product_t=T(S["intensity"]["product_t"]),
                intensity_note=E(S["intensity"]["note"]), cat_rows="".join(cat_rows),
                lines_in=ms["activity_lines"] + ms["spend_lines"],
                act_lines=ms["activity_lines"], spd_lines=ms["spend_lines"],
                act_pct=f'{ms["activity_pct_tco2e"]:.1f}', spd_pct=f"{sp_pct:.1f}",
                act_t=T(ms["activity_tco2e"]), spd_t=T(ms["spend_tco2e"]),
                act_lpct=f'{ms["activity_pct_lines"]:.0f}', spd_lpct=f'{100 - ms["activity_pct_lines"]:.0f}',
                tiers=tiers, hot=hot, excluded=ms["excluded_lines"], exclusions=ex,
                credit_n=ck["credit_notes"]["lines"], credit_t=f'{ck["credit_notes"]["tco2e"]:,.1f}',
                milk_ledger=T(ck["milk_t_ledger"]), milk_profile=T(ck["milk_t_profile"]), not_covered=nc,
                rows=S["meta"]["rows"], sha=S["meta"]["input_sha256"][:12], generated=E(S["meta"]["generated_at"]))


def dashboard_html(S, L):
    keep = ["line_id", "split", "posting_date", "doc_no", "counterparty", "description", "gl_account", "quantity", "unit",
            "net_kzt", "category", "scope", "ef_id", "ef_value", "ef_unit", "activity_qty", "activity_unit", "method",
            "quality", "tco2e", "flags", "reason"]
    data = dict(inv=S, cats={c["category"]: c["name"] for c in S["by_category"]},
                ef={e: P.EF[e]["name"] for e in P.EF}, cols=keep, lines=[[l[k] for k in keep] for l in L])
    js = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c").replace("\u2028", "\\u2028")
    return fill("dashboard.html", data=js)


def find_browser():
    pf = [os.environ.get(k, "") for k in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
    cands = [os.path.join(p, s) for p in pf if p for s in (r"Microsoft\Edge\Application\msedge.exe",
                                                             r"Google\Chrome\Application\chrome.exe")]
    cands += [shutil.which(n) for n in ("msedge", "chrome", "google-chrome", "chromium")]
    return next((c for c in cands if c and os.path.exists(c)), None)


def print_pdf(src, dst):
    b = find_browser()
    if not b:
        print("bank_summary.pdf skipped: no Edge/Chrome found (open out/bank_summary.html and print to PDF)")
        return None
    prof = tempfile.mkdtemp(prefix="cc_pdf_")
    try:
        subprocess.run([b, "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={prof}",
                        "--no-pdf-header-footer", f"--print-to-pdf={dst}", "file:///" + src.replace("\\", "/")],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
    finally:
        shutil.rmtree(prof, ignore_errors=True)
    with open(dst, "rb") as f:
        return len(re.findall(rb"/Type\s*/Page[^s]", f.read()))


SHOTS = [("bank_summary", "bank_summary.html"), ("dashboard", ""), ("dashboard_scope3", "#scope=3&tab=lines"),
         ("dashboard_milk", "#scope=3&cat=S3_C1_GOODS&ef=EF_MILK_KG&tab=lines"),
         ("dashboard_line", "#scope=3&cat=S3_C1_GOODS&ef=EF_MILK_KG&line=L00004&tab=lines"),
         ("dashboard_queue", "#tab=queue")]


def screens(size="1280,720"):
    """Deck screenshots: out/screens/<state>.png at 1280x720 from the generated dashboard (file://, no network)."""
    b, d = find_browser(), os.path.join(P.OUTD, "screens")
    os.makedirs(d, exist_ok=True)
    url = "file:///" + P.OUTD.replace("\\", "/") + "/"
    for name, h in SHOTS:
        h = h if h.endswith(".html") else "dashboard.html" + h
        prof = tempfile.mkdtemp(prefix="cc_shot_")
        subprocess.run([b, "--headless=new", "--disable-gpu", "--no-first-run", "--hide-scrollbars", f"--user-data-dir={prof}",
                        f"--window-size={size}", f"--screenshot={os.path.join(d, name + '.png')}", url + h],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60)
        shutil.rmtree(prof, ignore_errors=True)
        print("screen", os.path.join(d, name + ".png"))


def main(new_ledger=None, new_audit=None, pdf=True):
    t0 = time.time()
    L, labels, inputs = build_lines(new_ledger, new_audit)
    S = summarise(L, labels, inputs, t0)

    # ---------- self-checks ----------
    inc = [l for l in L if l["method"] != "excluded"]
    assert len({l["line_id"] for l in L}) == len(L), "line_id must be unique"
    assert abs(sum(l["tco2e"] for l in L) - S["total"]) < 0.1, "audit_all sum != total"
    assert abs(sum(S["by_scope"].values()) - S["total"]) < 0.1, "scopes != total"
    for s in "123":
        assert abs(sum(c["tco2e"] for c in S["by_category"] if c["scope"] == s) - S["by_scope"][s]) < 0.1, s
    assert all(l["tco2e"] == 0 for l in L if l["method"] == "excluded")
    bad = [l["line_id"] for l in inc if abs(l["activity_qty"] * l["ef_value"] / 1000 - l["tco2e"]) > 1e-3]
    assert not bad, f"formula does not recompute: {bad[:5]}"
    assert sum(S["quality"][q]["lines"] for q in "ABC") == len(inc)
    assert abs(sum(S["quality"][q]["pct_tco2e"] for q in "ABC") - 100) < 0.05
    assert all(re.search(r"CP_[0-9a-f]{8}", l["counterparty"]) for l in L), "unmasked counterparty in output"

    os.makedirs(P.OUTD, exist_ok=True)
    P.write(os.path.join(P.OUTD, "audit_all.csv"), L, FIELDS)
    S["meta"]["runtime_s"] = round(time.time() - t0, 2)
    with open(os.path.join(P.OUTD, "inventory.json"), "w", encoding="utf-8") as f:
        json.dump(S, f, ensure_ascii=False, indent=1)
    bank = os.path.join(P.OUTD, "bank_summary.html")
    for path, page in ((bank, bank_html(S)), (os.path.join(P.OUTD, "dashboard.html"), dashboard_html(S, L))):
        with open(path, "w", encoding="utf-8") as f:
            f.write(page)
    if pdf:
        pages = print_pdf(bank, os.path.join(P.OUTD, "bank_summary.pdf"))
        if pages is not None:
            assert pages == 1, f"bank_summary.pdf has {pages} pages, must be 1"
            print("bank_summary.pdf: 1 page")
    ck, bs = S["checks"], S["by_scope"]
    print(f"report: {len(L)} lines; total {S['total']:,.1f} t = S1 {bs['1']:,.1f} + S2 {bs['2']:,.1f} + S3 {bs['3']:,.1f}; "
          f"{S['intensity']['t_per_t_product']} t/t product; review queue {ck['review_queue']['lines']}; "
          f"train vs labels delta {ck['train_label_delta_t']} t")
    print("REPORT SELF-CHECKS OK")
    return S


if __name__ == "__main__":
    a = sys.argv[1:]
    get = lambda k: a[a.index(k) + 1] if k in a else None
    main(get("--new-ledger"), get("--new-audit"), pdf="--no-pdf" not in a)
    if "--screens" in a:
        screens()
