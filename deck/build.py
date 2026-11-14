"""Build the 8-slide submission deck v5 (editorial tokens, eight compositions via data-layout, PLAN_V5): numbers from out/*.json|csv -> deck.pdf + pages/pN.png.

Run:  python deck/build.py             generate slides into deck.html, print deck.pdf (headless Edge/Chrome),
                                       rasterise pages/*.png, print the lint table
      python deck/build.py --lint      same, exit 1 on any lint breach (word caps, exhibit size, fonts, banned terms)
      python deck/build.py --measure   first re-time `python pipeline.py` (temp copy, out/ untouched) and re-run
                                       tests/test_injection.py; cached in deck/measured.json
Manual values (leaderboard score, team) live in deck/manual.json; anything unknown prints as ЗАПОЛНИТЬ.
out/llm_eval.json (WP-B) is optional: absent -> the second-reader annotations are simply not drawn.
Every injected value and its source is written to deck/numbers_used.json for the proofread.
Charts are inline SVG generated here; the only bitmap is the bank one-pager thumbnail.
"""
import csv, html as H, json, os, re, shutil, subprocess, sys, tempfile, time

D = os.path.dirname(os.path.abspath(__file__))
CASE = os.path.dirname(D)
OUT = os.path.join(CASE, "out")
FILL = '<span class="fill">ЗАПОЛНИТЬ</span>'
USED = {}
BANNED = r"WAPE|macro-F1|naive Bayes|\bGL\b|lexicon|stdlib|k-fold|\bISO ?\d+"
FORBIDDEN = r"no AI model|zero network calls|\bverified\b|\bassured\b"


def jload(p):
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


def num(key, v, src, fmt="{:,.0f}"):
    """Format a sourced number and log it; None -> visible ЗАПОЛНИТЬ."""
    USED[key] = {"value": v, "source": src}
    return FILL if v is None else fmt.format(v)


def esc(s):
    return H.escape(str(s), quote=False)


# ---------- measured: runtime + injection suite ----------
def measure():
    tmp = tempfile.mkdtemp(prefix="cc_rt_")
    try:
        c = os.path.join(tmp, "case-bv2"); os.makedirs(c)
        shutil.copytree(os.path.join(CASE, "..", "cases"), os.path.join(tmp, "cases"))
        shutil.copy(os.path.join(CASE, "pipeline.py"), c)
        shutil.copytree(os.path.join(CASE, "masked"), os.path.join(c, "masked"))
        shutil.copytree(os.path.join(CASE, "tests"), os.path.join(c, "tests"))
        t = time.time()
        r = subprocess.run([sys.executable, "pipeline.py"], cwd=c, capture_output=True)
        rt = round(time.time() - t, 1)
        env = dict(os.environ, PYTHONIOENCODING="utf-8")
        inj = subprocess.run([sys.executable, os.path.join("tests", "test_injection.py")], cwd=c,
                             capture_output=True, text=True, encoding="utf-8", env=env)
        changed = re.findall(r"changed\s+(\d+)/(\d+)", inj.stdout)
        m = dict(measured_at=time.strftime("%Y-%m-%d %H:%M"), pipeline_runtime_s=rt, pipeline_exit=r.returncode,
                 injection_exit=inj.returncode, injection_forms=len(changed),
                 injection_lines=sum(int(b) for _, b in changed), injection_changed=sum(int(a) for a, _ in changed))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    json.dump(m, open(os.path.join(D, "measured.json"), "w"), indent=1)
    print("measured:", m)


# ---------- numbers ----------
def numbers():
    inv = jload(os.path.join(OUT, "inventory.json"))
    assert inv, "out/inventory.json missing: run python pipeline.py first"
    ms = jload(os.path.join(D, "measured.json")) or {}
    man = jload(os.path.join(D, "manual.json")) or {}
    ck, bs = inv["checks"], inv["by_scope"]
    fac = {f["ef_id"]: f for f in inv["by_factor"]}
    cat = {c["category"]: c for c in inv["by_category"]}
    aud = list(csv.DictReader(open(os.path.join(OUT, "audit_all.csv"), encoding="utf-8-sig")))
    inc = [a for a in aud if a["category"] != "EXCLUDE"]
    queue = [a for a in inc if re.search(r"new_wording|outlier|injection", a["flags"])]
    clf = [a for a in inc if a["reason"].startswith("classifier") and a not in queue]
    N = dict(total=inv["total"], s1=bs["1"], s2=bs["2"], s3=bs["3"], rows=inv["meta"]["rows"],
             excluded=inv["method_share"]["excluded_lines"], dup=ck["duplicates"], nonp=ck["non_purchase"],
             cp=ck["masked_cp"], persons=ck["masked_persons"], queue=ck["review_queue"]["lines"],
             milk=fac["EF_MILK_KG"]["tco2e"], milk_lines=fac["EF_MILK_KG"]["lines"], n_factors=len(inv["by_factor"]),
             grid=fac["EF_GRID_KZ_KWH"]["tco2e"], gas=fac["EF_NG_M3"]["tco2e"], heat=fac["EF_HEAT_GCAL"]["tco2e"],
             goods=cat["S3_C1_GOODS"]["tco2e"], sens=inv["sensitivity"]["total_change_pct"],
             act_pct=inv["method_share"]["activity_pct_tco2e"],
             n_queue=len(queue), n_clf=len(clf), n_rules=len(inc) - len(queue) - len(clf), n_inc=len(inc))
    assert N["n_queue"] == N["queue"], "review queue in audit_all.csv != inventory.json"
    N["runtime"], N["inj_lines"], N["inj_changed"] = (ms.get("pipeline_runtime_s"), ms.get("injection_lines"),
                                                     ms.get("injection_changed"))
    N["leaderboard"], N["team"] = man.get("leaderboard_score"), man.get("team")
    rd = lambda p: {r["scenario"]: r for r in csv.DictReader(open(p, encoding="utf-8"))} if os.path.exists(p) else {}
    N["eval"], N["eval_prev"] = rd(os.path.join(OUT, "eval_hard.csv")), rd(os.path.join(OUT, "eval_hard_prev.csv"))
    N["llm"] = jload(os.path.join(OUT, "llm_eval.json"))
    N["intensity"] = inv["intensity"]["t_per_t_product"]
    N["credit"] = ck["credit_notes"]["lines"]
    return N


# ---------- cost model (WP-1): every input in deck/cost_model.json, tagged sourced | brief | measured | ASSUMED ----------
CM_PATH = os.path.join(D, "cost_model.json")
CM = json.load(open(CM_PATH, encoding="utf-8"))["inputs"]
V_ = lambda k: CM[k]["value"]
FX, CAP, LOAN = V_("fx"), V_("cap"), V_("loan")


def _people(hours, queue_lines):
    h = hours["run"] + queue_lines * hours["queue_min_per_line"] / 60 + hours["factors"] + hours["maintenance"] + hours["verifier_support"]
    return h, h * V_("wage") * V_("load") / V_("hours_per_month")


def cost(N, log=True):
    """C0 contract (PLAN_V5): all KZT unless named. Reads cost_model.json, inventory (N) and h2h/results.json."""
    hrs = V_("hours")
    hours, people = _people(hrs, N["queue"])
    ver_y1 = V_("verifier_days_y1") * V_("day_rate_usd") * FX
    ver_y2 = ver_y1 * V_("year2_factor")
    sp = json.load(open(os.path.join(CASE, "h2h", "results.json"), encoding="utf-8"))["spend"]
    ai = N["rows"] * (sp["deepseek"]["usd_per_call"] + sp["jev"]["usd_per_call"]) * FX
    rate = V_("wage") * V_("load") / V_("hours_per_month")
    own = hours - V_("shared_hours")   # per-company hours outside the shared factor library
    C = dict(people=people, tool=people, ver_y1=ver_y1, ver_y2=ver_y2, y1_extra=ver_y1 - ver_y2,
             total_y1=people + ver_y1, total_y2=people + ver_y2, hours=hours, ai_ceiling_kzt=ai,
             per_t_y2=(people + ver_y2) / N["total"], loan_pct_y2=100 * (people + ver_y2) / LOAN,
             cap_pct_y1=100 * (people + ver_y1) / CAP, cap_pct_y2=100 * (people + ver_y2) / CAP, cap_pct_tool=100 * people / CAP,
             per_company={n: (V_("shared_hours") / n + own) * rate for n in (1, 5, 10)})
    assert C["tool"] + C["ver_y2"] + C["y1_extra"] - C["total_y1"] < 1e-6
    stress = people + V_("verifier_days_stress") * V_("day_rate_usd") * FX
    assert stress < CAP, f"stress case {stress:,.0f} >= cap"
    if log:
        tag = lambda *ks: " + ".join(sorted({CM[k]["tag"] for k in ks}))
        src = "deck/build.py cost() <- deck/cost_model.json"
        for k, ks in dict(people=("wage", "load", "hours_per_month", "hours"), tool=("wage", "load", "hours_per_month", "hours"),
                          ver_y1=("verifier_days_y1", "day_rate_usd", "fx"), ver_y2=("verifier_days_y1", "day_rate_usd", "fx", "year2_factor"),
                          y1_extra=("verifier_days_y1", "day_rate_usd", "fx", "year2_factor"),
                          total_y1=("hours", "verifier_days_y1", "day_rate_usd", "fx"), total_y2=("hours", "verifier_days_y1", "year2_factor", "fx"),
                          hours=("hours",), ai_ceiling_kzt=("ai_usd_per_call", "fx"), per_t_y2=("hours", "year2_factor"),
                          loan_pct_y2=("hours", "year2_factor", "loan"), cap_pct_y1=("hours", "verifier_days_y1", "cap"),
                          cap_pct_y2=("hours", "year2_factor", "cap"), cap_pct_tool=("hours", "cap"),
                          per_company=("hours", "shared_hours")).items():
            USED[f"cost_{k}"] = {"value": C[k] if k != "per_company" else {n: round(v) for n, v in C[k].items()},
                                 "source": src, "tag": tag(*ks)}
        USED["cost_stress_y1"] = {"value": round(stress), "source": src, "tag": tag("verifier_days_stress", "hours")}
    return C


def cost_check():
    """--cost-check: v4 inputs reproduce 8,241,497; C0 expected values to +-1 KZT; every sourced input has an http source."""
    N = numbers()
    v4 = json.load(open(CM_PATH, encoding="utf-8"))["v4_check"]
    v4_total = _people(v4["hours"], N["queue"])[1] + v4["ver_usd"] * FX
    assert abs(v4_total - v4["expected_total"]) <= 1, f"v4 {v4_total:,.0f}"
    C = cost(N, log=False)
    exp = dict(people=825938, ver_y1=7500000, ver_y2=5250000, total_y1=8325938, total_y2=6075938, ai_ceiling_kzt=332)
    bad = [f"{k} {C[k]:,.2f} != {v:,}" for k, v in exp.items() if abs(C[k] - v) > 1]
    bad += [f"per_company[{n}] {C['per_company'][n]:,.0f}" for n, v in {1: 825938, 5: 319292, 10: 255962}.items() if abs(C["per_company"][n] - v) > 1]
    bad += [f"per_t_y2 {C['per_t_y2']:.2f}"] * (round(C["per_t_y2"], 2) != 30.43) + [f"loan_pct_y2 {C['loan_pct_y2']:.4f}"] * (round(C["loan_pct_y2"], 4) != .1215)
    bad += [f"{k}: sourced without http source" for k, v in CM.items() if v["tag"] == "sourced" and not str(v["source"]).startswith("http")]
    assert C["tool"] == C["people"] and C["y1_extra"] == C["ver_y1"] - C["ver_y2"]
    print("COST CHECK", "OK" if not bad else "FAIL\n  " + "\n  ".join(bad), {k: round(C[k], 4) for k in exp | {"per_t_y2": 0, "loan_pct_y2": 0}})
    return not bad


# ---------- masked lines used on slides (masked text only: pseudonyms, no names) ----------
def lines():
    aud = {a["line_id"]: a for a in csv.DictReader(open(os.path.join(OUT, "audit_all.csv"), encoding="utf-8-sig"))}
    led = {r["line_id"]: r for r in csv.DictReader(open(os.path.join(CASE, "masked", "ledger_test.csv"), encoding="utf-8-sig"))}
    test = [a for a in aud.values() if a["split"] == "test"]
    milk = max((a for a in test if a["ef_id"] == "EF_MILK_KG" and a["unit"] == "т"), key=lambda a: float(a["tco2e"]))
    inj = next(a for a in test if "injection" in a["flags"])
    pick = lambda f: next(a for a in aud.values() if f(a))
    messy = [pick(lambda a: a["ef_id"] == "EF_MILK_KG" and re.search(r"[әіүұқғңөһ]", a["description"].lower())),
             pick(lambda a: a["description"] == "Raw ocw milk"),
             pick(lambda a: "third-party carrier" in a["reason"]),
             pick(lambda a: a["reason"].startswith("flight") and "-ALA" in a["description"] and "intl" in a["reason"])]
    return milk, led[milk["line_id"]], inj, messy


# ---------- extra sources: rebuilt typical approaches and the AI readers we tried ----------
def extra():
    sc = {(r["scenario"], r["approach"]): float(r["score"])
          for r in csv.DictReader(open(os.path.join(CASE, "compete", "out", "scores.csv"), encoding="utf-8"))}
    bl = jload(os.path.join(CASE, "compete", "out", "baselines.json")) or {}
    return dict(sc=sc, inj=bl.get("injection", {}))


def readers():
    """One row per AI reader: (name, accuracy on unseen accounts, its label, dangerous flips, attacked lines, USD per 1,000 lines).
    Same metric across rows where the source allows; differences are marked * and named in the slide footer."""
    laya = jload(os.path.join(CASE, "laya_spike", "out_results.json"))
    sd = jload(os.path.join(CASE, "scaledown_eval", "results.json"))
    h = jload(os.path.join(CASE, "h2h", "results.json"))
    R = []
    if laya:
        f1, inj = laya["splits"]["GL account"]["laya"][1], laya["injection"]
        R.append(("Laya", num("laya_gl_cat_f1", f1, "laya_spike/out_results.json GL split category macro-F1", "{:.2f}") + " F1*", f1,
                  num("laya_inj_changed", inj["changed"], "laya_spike/out_results.json injection.changed") + "*", inj["changed"], inj["n"],
                  num("laya_usd_1k", 0, "laya_spike/REPORT.md: local CPU, no API", "${:.2f}"), 0.0, "local"))
    if sd:
        e2, e3 = sd["E2"], sd["E3"]
        d = e3["picks_changed"] - e3["changed_to_none"]
        R.append(("ScaleDown", num("scaledown_cat_acc", 100 * e2["scaledown_cat_acc"], "scaledown_eval/results.json E2 (148 lines)", "{:.0f}") + "%*",
                  e2["scaledown_cat_acc"], num("scaledown_dangerous", d, "scaledown_eval/results.json E3 changed - to NONE"), d, e3["sent"],
                  num("scaledown_usd_1k", 1000 * 0.0000167, "scaledown_eval/RESULTS.md est. per call x 1000", "${:.3f}"), .0167, "est."))
    if h:
        for k, nm in (("jev", "Jev"), ("deepseek", "DeepSeek")):
            m = h[k]
            d = sum(m[s].get("dangerous", 0) for s in ("E3_raw", "E4_raw"))
            n = m["E3_raw"]["sent"] + m["E4_raw"]["sent"]
            u = 1000 * h["spend"][k]["usd_per_call"]
            R.append((nm, num(f"{k}_cat_acc", 100 * m["E2"]["cat_acc"], "h2h/results.json E2 cat_acc (295 lines)", "{:.0f}") + "%",
                      m["E2"]["cat_acc"], num(f"{k}_dangerous", d, "h2h/results.json E3_raw+E4_raw dangerous"), d, n,
                      num(f"{k}_usd_1k", u, "h2h/results.json spend usd_per_call x 1000", "${:.2f}"), u, ""))
    # consensus: DeepSeek + Jev must agree. Numbers only if a measured result exists; cost alone is the sum of both calls.
    rc = jload(os.path.join(CASE, "h2h", "results_consensus.json")) or {}
    e2 = (rc.get("E2") or {}).get("consensus") or {}
    e4 = (rc.get("E4_raw_consensus_only") or {})
    cons = dict(cat_acc=e2.get("cat_acc"), dangerous=(e4.get("consensus") or {}).get("dangerous", 0) if e4.get("sent") else None,
                sent=e4.get("sent")) if e2 else None
    if h:
        u = 1000 * (h["spend"]["jev"]["usd_per_call"] + h["spend"]["deepseek"]["usd_per_call"])
        cu = num("consensus_usd_1k", u, "h2h/results.json spend: DeepSeek + Jev per call x 1000", "${:.2f}")
        ca = (num("consensus_cat_acc", 100 * cons["cat_acc"], "h2h/results_consensus.json E2 consensus cat_acc", "{:.0f}") + "%"
              if cons and cons.get("cat_acc") is not None else "in test")
        if cons and cons.get("dangerous") is not None:
            R.append(("Both must agree", ca, cons["cat_acc"], num("consensus_dangerous", cons["dangerous"],
                      "h2h/results_consensus.json E4_raw_consensus_only consensus.dangerous"), cons["dangerous"], cons["sent"], cu, u, "acc"))
        else:
            R.append(("Both must agree", ca, cons and cons.get("cat_acc"), "in test", None, None, cu, u, "acc"))
    return R, cons


# ---------- drawing helpers (v5: one stage per slide, eight compositions, editorial tokens unchanged) ----------
W = 1088  # stage width in px (1280 - 2 x 96)


def page(i, kicker, ctx, title, body, foot, layout):
    """One slide: top band, title, body (holds exactly one .ex), footer. `layout` -> class + data-layout for the lint."""
    return (f'<section class="L-{layout}" data-layout="{layout}"><div class="top"><span class="k">{kicker}</span><span class="c">{ctx}</span></div>'
            f'<div class="stage"><h1>{title}</h1>{body}</div>'
            f'<div class="bot"><span class="foot">{foot}</span><span class="pg">{i} / 8</span></div></section>')


def exh(top, inner, h, style=""):
    return f'<div class="ex" style="top:{top}px;height:{h}px;{style}">{inner}</div>'


def at(top, inner, style=""):
    """Absolutely placed block inside the stage (not the exhibit)."""
    return f'<div style="position:absolute;left:0;right:0;top:{top}px;{style}">{inner}</div>'


def pct(v):
    return f"{max(v * 100, .4):.2f}%"


def row(label, val, frac, cls="", lab_cls=""):
    """Label | thin bar | value on one line."""
    return (f'<div class="row x"><span class="{lab_cls}">{label}</span><div><div class="t {cls}" style="width:{pct(frac)}"></div></div>'
            f'<span class="v {lab_cls}">{val}</span></div>')


def cap(text, style=""):
    return f'<div class="cap x" style="{style}">{text}</div>'


def big(v, label, size=46, c="", sub="", ncls=""):
    """Heavy numeral with a small label under it."""
    return (f'<div class="eb"><div class="num x {c} {ncls}" style="font-size:{size}px">{v}</div>'
            f'<div class="x lbl" style="margin-top:8px">{label}</div>'
            + (f'<div class="x sub">{sub}</div>' if sub else "") + '</div>')


def ASM(v):
    """An ASSUMED figure: the lint wants 'est.' after it in the same block."""
    return f'<span class="a">{v}</span>'


# ---------- the eight slides (PLAN_V5 §B3) ----------
def s1(N, V):
    tot = N["total"]
    parts = [("Raw milk from farms", N["milk"], "var(--acc)", "milk_t"), ("Bought power and heat", N["s2"], "var(--g1)", "s2_t"),
             ("Fuel burned on site", N["s1"], "var(--g3)", "s1_t"), ("Other suppliers", N["s3"] - N["milk"], "var(--rule)", "s3_other_t")]
    assert abs(sum(p[1] for p in parts) - tot) < 1, "slide 1 bars do not sum to the total"
    share = N["milk"] / tot
    assert .75 < share < .85, "'four-fifths' no longer true"
    num("milk_pct", 100 * share, "out/inventory.json", "{:.0f}")
    bar = "".join(f'<div style="width:{W * t / tot:.2f}px;height:28px;background:{c};border-right:2px solid var(--bg)"></div>'
                  for _, t, c, _ in parts)
    legend = "".join(
        f'<div style="border-top:1px solid var(--rule);padding-top:10px"><div style="display:flex;align-items:center;gap:8px">'
        f'<span style="width:10px;height:10px;background:{c};flex:none"></span>'
        f'<span class="num x {"acc" if k == 0 else ""}" style="font-size:22px">{num(key, t, "out/inventory.json")}</span>'
        f'<span class="x sub">t · {100 * t / tot:.0f}%</span></div>'
        f'<div class="x lbl" style="margin-top:4px">{l}</div></div>' for k, (l, t, c, key) in enumerate(parts))
    hero = (f'<div class="hero num x">{num("total", tot, "out/inventory.json")}</div>'
            f'<div class="x lbl" style="font-size:15px;margin-top:12px">tonnes CO2e in 2025, all gases</div>'
            f'<div style="display:flex;margin-top:40px">{bar}</div>'
            f'<div style="display:grid;grid-template-columns:repeat(4,1fr);column-gap:24px;margin-top:16px;text-align:left">{legend}</div>')
    loan = num("loan_scope_pct", 100 * (N["grid"] + N["gas"] + N["heat"]) / tot, "out/inventory.json by_factor grid+gas+heat", "{:.0f}")
    USED["loan_scope_t"] = {"value": N["grid"] + N["gas"] + N["heat"], "source": "out/inventory.json by_factor EF_GRID_KZ_KWH+EF_NG_M3+EF_HEAT_GCAL"}
    lead = f"The KZT {num('loan_bn', LOAN / 1e9, '[brief]')} bn green loan's solar and heat pumps touch only {loan}% of emissions."
    team = esc(N["team"]) if N["team"] else FILL
    lb = num("leaderboard", N["leaderboard"], "deck/manual.json", "{:.4f}")
    title = f"Four-fifths of the footprint sits on {num('farms', 25, '[brief] about 25 farms')} farms."
    body = (exh(100, hero, 346, "text-align:center")
            + at(464, f'<p class="lead" style="margin:0;text-align:center">{lead}</p>'))
    return page(1, "Carbon Copilot", f"Dala Foods · 2025 · Team {team} · Leaderboard {lb}", title, body,
                "Case data, synthetic. Supplier and staff names masked. Detail: bank one-pager, README.", "hero")


def s2(N, V, messy):
    rows_ = N["rows"]
    led = "".join(f'<div class="mono" style="font-size:14px;line-height:1.4;padding:8px 0;border-top:1px solid var(--rule);'
                  f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis"><span class="x">{esc(a["description"])}</span>'
                  f'<br><span class="x" style="color:var(--g2)">{float(a["quantity"]):,.6g} {esc(a["unit"])}</span></div>'
                  for a in messy[:4])
    left = (cap(f'{num("rows", rows_, "out/inventory.json")} lines in · names masked', "margin-bottom:8px") + led
            + '<div style="border-top:1px solid var(--rule)"></div>')
    mid = (f'<div style="text-align:center"><div class="x acc" style="font-size:15px;font-weight:600">1 command</div>'
           f'<svg width="140" height="24" style="display:block;margin:10px auto"><path d="M2 12H134M124 4l10 8-10 8" fill="none" stroke="var(--acc)" stroke-width="1.5"/></svg>'
           f'<div class="num x acc" style="font-size:56px">{V["rt"]} s</div><div class="x sub" style="margin-top:6px">full run, all checks</div></div>')
    right = (cap("1 page out · for the bank", "margin-bottom:8px")
             + '<div style="width:420px;height:262px;border:1px solid var(--rule);background:url(assets/bank.png) top left/420px auto no-repeat"></div>')
    ex = (f'<div style="display:grid;grid-template-columns:439px 162px 439px;column-gap:24px;align-items:center;height:100%">'
          f'<div>{left}</div>{mid}<div>{right}</div></div>')
    who = [("Officer runs it", "once a year, one command"), ("Bank reads it", "one page, all gases"),
           ("Verifier recomputes", "any line from its invoice"), ("CFO pays", f"within the KZT {CAP / 1e6:.0f} m cap")]
    strip = ('<div style="display:grid;grid-template-columns:repeat(4,1fr);column-gap:24px;border-top:1px solid var(--ink);padding-top:12px">'
             + "".join(f'<div><div class="x" style="font-size:15px;font-weight:600">{a}</div><div class="x sub" style="margin-top:2px">{b}</div></div>'
                       for a, b in who) + '</div>')
    for k, v in (("intensity", N["intensity"]), ("cp", N["cp"]), ("persons", N["persons"])):
        num(k, v, "out/inventory.json")
    num("cap_m", CAP / 1e6, "[brief]", "{:.0f}")
    return page(2, "Product", "Who uses it · the journey", "One command takes the ledger to the bank's page.",
                exh(104, ex, 300) + at(424, strip),
                "Inbox requests from the case brief; answers in README, Who asked for what.", "split")


def s3(N, V, cons_state):
    R = N["rows"]
    for k in ("rows", "cp", "persons", "excluded"):
        num(k, N[k], "out/inventory.json")
    st = [("Lines in", "1C export", R), ("Names masked", f"{N['cp']} suppliers, {N['persons']} people", R),
          ("Purchases only", f"−{N['excluded']} taxes, duplicates", N["n_inc"]), ("With a formula", "quantity × factor", N["n_inc"])]
    num("included", N["n_inc"], "out/audit_all.csv")
    chev = '<div class="x" style="color:var(--g3);font-size:28px;padding-top:6px;text-align:center">›</div>'
    cells = chev.join(f'<div><div class="num x" style="font-size:48px">{v:,}</div>'
                      f'<div style="height:6px;background:var(--g3);width:{100 * v / R:.1f}%;margin:14px 0 10px"></div>'
                      f'<div class="x" style="font-size:15px;font-weight:600">{a}</div>'
                      f'<div class="x sub" style="margin-top:2px">{b}</div></div>' for a, b, v in st)
    flow = (cap("Lines at each stage", "margin-bottom:12px")
            + f'<div style="display:grid;grid-template-columns:1fr 24px 1fr 24px 1fr 24px 1fr;column-gap:12px">{cells}</div>')
    seg = [("rules", N["n_rules"], "var(--ink)"), ("classifier", N["n_clf"], "var(--g3)"), ("person", N["n_queue"], "var(--acc)")]
    for k, (_, n, _) in enumerate(seg):
        num(f"dec_{k}", n, "out/audit_all.csv reasons/flags")
    w = [n / N["n_inc"] for _, n, _ in seg]
    stack = "".join(f'<div style="height:24px;width:{W * f:.2f}px;background:{c};border-right:2px solid var(--bg)"></div>'
                    for (_, _, c), f in zip(seg, w))
    lab = lambda v, l, c="": (f'<div class="num x {c}" style="font-size:30px">{v}</div><div class="x lbl" style="margin-top:6px">{l}</div>')
    who = (cap(f"Who decides each of the {N['n_inc']:,} lines", "margin:48px 0 10px")
           + f'<div style="display:flex">{stack}</div>'
           + f'<div style="position:relative;height:70px;margin-top:12px">'
           f'<div style="position:absolute;left:0">{lab(f"{N["n_rules"]:,}", "rules decide")}</div>'
           f'<div style="position:absolute;left:{W * w[0]:.0f}px">{lab(f"{N["n_clf"]:,}", "classifier decides")}</div>'
           f'<div style="position:absolute;right:0;text-align:right">{lab(N["n_queue"], "a person decides", "acc")}</div></div>'
           f'<div style="display:flex;justify-content:flex-end;align-items:center;gap:10px;margin-top:4px">'
           f'<span class="x sub">before a person</span>'
           f'<svg width="28" height="10"><path d="M0 5H26" stroke="var(--g3)" stroke-width="1"/></svg>'
           f'<span class="x" style="font-size:14px;font-weight:600;border:1px solid var(--ink);padding:5px 10px">2 AI readers must agree '
           f'<span class="acc" style="font-weight:400">· {cons_state}</span></span></div>')
    lead = f"{R:,} lines, four stages, {V['rt']} seconds with every self-check; re-runs are identical."
    head = (f'<div style="display:flex;justify-content:space-between;align-items:flex-end;gap:24px">'
            f'<h1 style="flex:0 0 810px">Code decides every number. AI only advises.</h1>'
            f'<p class="lead" style="flex:0 0 254px;margin:0;font-size:15px;text-align:right">{lead}</p></div>')
    body = exh(104, flow + who, 372)
    sec = page(3, "Architecture", "From ledger to inventory", "", body,
               "Python on a laptop; AI readers via OpenRouter, masked text only. Detail: README.", "flow")
    return sec.replace("<h1></h1>", head)


def s4(N, milk, mled):
    tot = N["total"]
    q_t, ef, t = float(mled["quantity"]), float(milk["ef_value"]), float(milk["tco2e"])
    assert abs(q_t * ef - t) < 0.001, "worked line does not recompute"
    num("worked_qty_t", q_t, "masked/ledger_test.csv", "{:.2f}"); num("worked_ef", ef, "out/audit_all.csv ef_value", "{:g}")
    num("worked_tco2e", t, "out/audit_all.csv", "{:,.1f}")
    u = '<span style="font-size:40px;font-weight:600;letter-spacing:0"> t</span>'
    terms = [(f"{q_t:,.2f}{u}", "raw milk bought", f"invoice {esc(mled['doc_no'])}", ""), ("×", "", "", "op"),
             (f"{ef:g}", "factor", "kg CO2e per kg", ""), ("=", "", "", "op"),
             (f"{t:,.1f}{u}", "tonnes CO2e", f"line {esc(milk['line_id'])}", "acc")]
    nums_ = "".join(f'<div class="{"acc" if c == "acc" else ""}" style="font-weight:{400 if c == "op" else 700};'
                    f'{"color:var(--g3);" if c == "op" else ""}letter-spacing:-.02em;line-height:1">{v}</div>' for v, _, _, c in terms)
    labs = "".join((f'<div style="font-size:15px;line-height:1.35;margin-top:14px;border-top:1px solid var(--rule);padding-top:8px;text-align:left">'
                    f'<span class="x">{a}</span><br><span class="x sub">{b}</span></div>') if a else "<div></div>" for _, a, b, _ in terms)
    eq = (f'<div style="display:grid;grid-template-columns:repeat(5,auto);justify-content:center;column-gap:30px;font-size:100px">'
          f'{nums_}{labs}</div>')
    lad = [("All emissions", tot, "{:,.0f}"), ("Scope 3", N["s3"], "{:,.0f}"), ("Purchased goods", N["goods"], "{:,.0f}"),
           ("Raw milk", N["milk"], "{:,.0f}"), ("this line", t, "{:,.1f}")]
    crumbs = ' <span style="color:var(--g3)">›</span> '.join(
        f'<span class="x {"acc" if k == 4 else ""}">{l} <b style="font-weight:700">'
        f'{num(f"lad_{k}", v, "out/inventory.json" if k < 4 else "out/audit_all.csv", fm)}</b></span>' for k, (l, v, fm) in enumerate(lad))
    drill = (f'<div style="border-top:1px solid var(--rule);padding-top:14px;text-align:center;font-size:16px">'
             f'{cap("Drill path, tonnes CO2e", "margin-bottom:8px")}{crumbs}</div>')
    facts = (f'<div style="text-align:center;font-size:15px;color:var(--g1)">'
             f'{num("act_pct", N["act_pct"], "out/inventory.json", "{:.1f}")}% of tonnes from physical quantities · '
             f'milk factor ±20% moves the total ±{num("sens", N["sens"], "out/inventory.json", "{:.0f}")}%</div>')
    num("milk_lines", N["milk_lines"], "out/inventory.json")
    return page(4, "Audit trail", "One worked line", "Every tonne traces to one line you can recompute.",
                exh(128, eq, 200) + at(358, drill) + at(440, facts),
                "1 t per t = 1 kg per kg. Detail: audit.csv, one row per ledger line.", "eq")


def s5(N, X):
    inj, sc, ev = X["inj"], X["sc"], N["eval"]
    n = N["inj_lines"] or 2400
    mini = "--lw:150px;--vw:44px;--rh:32px;font-size:14px;border-top:1px solid var(--rule);margin-top:20px"
    ir = [("Text classifier", inj.get("clf_starter"), ""), ("+ account, unit", inj.get("clf_units"), ""),
          ("Organisers' starter", inj.get("starter"), ""), ("Carbon Copilot", N["inj_changed"], "acc")]
    mx = max(v for _, v, _ in ir if v is not None)
    b1 = "".join(row(l, num(f"inj_{k}", v, "compete/out/baselines.json injection" if k < 3 else "deck/measured.json"),
                     (v or 0) / mx, "bg-acc" if c else "", c) for k, (l, v, c) in enumerate(ir))
    num("inj_lines", N["inj_lines"], "deck/measured.json")
    num("inj_forms", 12, "deck/measured.json injection_forms")
    h = jload(os.path.join(CASE, "h2h", "results.json")) or {}
    forms = sum(h.get("deepseek", {}).get(s, {}).get("forms", 0) for s in ("E3_raw", "E4_raw")) or None
    moved = sum(h[m][s]["prod_t_changed_vs_clean_row"] for m in ("deepseek", "jev") for s in ("E3_raw", "E4_raw", "E4_strip")) if h else None
    num("attack_t_moved", moved, "h2h/results.json prod_t_changed_vs_clean_row")   # kept for the proofread, no longer drawn
    rc = jload(os.path.join(CASE, "h2h", "results_consensus.json")) or {}
    sp = rc.get("E4_strip_production") or {}
    fix_d = sum(sp.get(m, {}).get("dangerous", 0) for m in ("deepseek", "jev", "consensus")) if sp else None
    assert not sp or sp["final"]["accepted"] == 0, "round 2: a flagged line took a pick"
    b2 = "".join(f'<div class="x" style="height:32px;display:flex;align-items:center;border-bottom:1px solid var(--rule)">{t}</div>' for t in (
        f'{num("cp", N["cp"], "out/inventory.json")} suppliers, {num("persons", N["persons"], "out/inventory.json")} people masked',
        f'{num("guards", 4, "mask.py self-check, make_zip.py, .git/hooks/pre-commit, llm_fallback.py check_payload")} guards: run, zip, commit, AI payload',
        f'{num("attack_forms", forms, "h2h/results.json E3+E4 forms")} attack forms → '
        f'{num("filter_gaps", 8, "h2h/RESULTS.md: strip_injection recognised 2 of 10 new forms, 8 missed")} gaps → fixed',
        f'after the fix: {num("fix_dangerous", fix_d, "h2h/results_consensus.json E4_strip_production dangerous (deepseek+jev+consensus)")} '
        f'dangerous picks of {num("fix_lines", sp.get("sent"), "h2h/results_consensus.json E4_strip_production sent")}'))
    num("names_sent_runtime", 0, "h2h/RESULTS.md: blocked_by_leak_check 0, 0 of 2,000 payloads held a name")
    for k, v in (("dev_gaps", 2), ("dev_raw_rows", 4), ("dev_carrier_names", 1)):
        num(k, v, "README_SUBMISSION.md Data rule log")
    r = [("Organisers' starter", sc.get(("random 5-fold", "starter")), "starter"),
         ("Text classifier only", sc.get(("random 5-fold", "clf_units")), "clf_units"),
         ("Everything by money", sc.get(("random 5-fold", "spend_only")), "spend_only"),
         ("Carbon Copilot", sc.get(("random 5-fold", "ours")), "ours")]
    b3 = "".join(row(l, num(f"cmp_{k}", v, "compete/out/scores.csv random 5-fold", "{:.2f}"), v or 0,
                     "bg-acc" if k == "ours" else "", "acc" if k == "ours" else "") for l, v, k in r)
    ua = float(ev["group k-fold: GL account"]["score"])
    num("eval_unseen_wording", float(ev["group k-fold: description template"]["score"]), "out/eval_hard.csv", "{:.4f}")
    pv = N["eval_prev"].get("group k-fold: GL account")
    num("eval_gl_before_rule", float(pv["score"]) if pv else None, "out/eval_hard_prev.csv", "{:.2f}")
    num("labelled", 1269, "compete/out/baselines.json rows")
    col = lambda v, lbl, rows_, extra="", first=False: (
        f'<div style="{"" if first else "border-left:1px solid var(--rule);padding-left:24px;"}">'
        f'<div class="num x acc" style="font-size:72px">{v}</div><div class="x" style="font-size:15px;margin-top:10px">{lbl}</div>'
        f'<div style="{mini}">{rows_}</div>{extra}</div>')
    ex = ('<div style="display:grid;grid-template-columns:1fr 1fr 1fr;column-gap:24px">'
          + col(num("inj_changed", N["inj_changed"], "deck/measured.json"), f"lines changed by hidden text, of {n:,}", b1, first=True)
          + col("0", "names sent out when the tool runs", b2)
          + col(f"{r[3][1]:.2f}", "score, same labelled lines", b3,
                f'<div class="x lbl" style="margin-top:12px"><b>{num("eval_unseen_accounts", ua, "out/eval_hard.csv", "{:.2f}")}</b> on accounts never seen</div>')
          + '</div>')
    lo = min(v for _, v, _ in r[:3])
    title = (f"Hidden instructions change nothing; shortcuts score "
             f"<span style=\"white-space:nowrap\">{lo:.2f}–{r[2][1]:.2f}</span>, ours {r[3][1]:.2f}.")
    foot = ("Development log: our review found 2 data-rule gaps (4 raw rows, 1 carrier name reached our coding assistant). "
            "Both closed; a leak fails the run.")
    assert len(foot.split()) <= 25, "slide 5 footer > 25 tokens"
    return page(5, "Safeguards · Accuracy", "Injection test · masking · organisers' formula", title, exh(112, ex, 360), foot, "trio")


def s6(N, RD):
    mx_d = max([r[4] for r in RD if r[4] is not None] + [1])
    mx_u = max([r[7] for r in RD if r[7] is not None] + [1e-9])

    def cell(txt, frac, sub, c):
        bar = (f'<div style="height:6px;margin-top:6px;background:var(--rule)"><div class="{"bg-acc" if c == "acc" else ""}" '
               f'style="height:6px;width:{pct(frac)};background:var(--g3)"></div></div>') if frac is not None else \
              '<div style="height:6px;margin-top:8px;border-top:1px dashed var(--g3)"></div>'
        return (f'<div><div style="display:flex;align-items:baseline;gap:8px"><span class="x num {c}" style="font-size:26px">{txt}</span>'
                + (f'<span class="x sub">{sub}</span>' if sub else "") + f'</div>{bar}</div>')
    T = "display:grid;grid-template-columns:161px 254px 254px 347px;column-gap:24px"
    head = (f'<div style="{T};align-items:end;padding-bottom:10px">' + "".join(f'<div class="cap x" style="letter-spacing:.06em">{h}</div>' for h in
            ("AI reader", "Right category, unseen accounts", "Dangerous flips under attack", "Cost per 1,000 lines")) + '</div>')
    rows_ = ""
    for nm, acc, accv, dn, dv, n, u, uv, tag in RD:
        c = "acc" if tag == "acc" else ""
        rows_ += (f'<div style="{T};height:{64 if c else 56}px;align-items:center;border-top:1px solid {"var(--ink)" if c else "var(--rule)"};'
                  f'{"margin-top:8px" if c else ""}">'
                  f'<div class="x {c}" style="font-size:16px;font-weight:600">{nm}</div>'
                  + cell(acc, accv, "", c)
                  + cell(dn, None if dv is None else dv / mx_d, f"of {n:,}" if n else "", c)
                  + cell(u, uv / mx_u, tag if tag in ("local", "est.") else ("both calls" if c else ""), c) + '</div>')
    grid = f'<div class="tbl">{head}{rows_}</div>'
    foot = ("* Laya: category F1, 1,269 lines; flips = any change, 50 lines. ScaleDown: 148-line sample, cost est. "
            "Others: 295 lines, 22 forms; both: 20 × 10.")
    return page(6, "Architecture · AI stack", "Same masked lines · unseen accounts · attacks",
                "Five AI readers tested. Two lead, and both must agree.", exh(88, grid, 380), foot, "table")


def s7(N, C):
    ok = all(C.get(k) is not None for k in ("tool", "ver_y1", "ver_y2", "y1_extra", "total_y1", "total_y2"))
    m = lambda v: None if v is None else v / 1e6
    r2 = lambda v: round(v / 1e6, 2)
    if ok:
        assert abs(r2(C["tool"]) + r2(C["ver_y2"]) + r2(C["y1_extra"]) - r2(C["total_y1"])) < .005, "slide 7: tool + check + extra != year 1"
        assert C["total_y1"] < CAP
    g = lambda k: C.get(k)
    f2 = "{:.2f}"
    tool = num("s7_tool_m", m(g("tool")), "cost() tool", f2)
    ver2 = num("s7_ver_y2_m", m(g("ver_y2")), "cost() ver_y2", f2)
    extra = num("s7_y1_extra_m", m(g("y1_extra")), "cost() y1_extra", f2)
    y1 = num("s7_total_y1_m", m(g("total_y1")), "cost() total_y1", f2)
    y2 = num("s7_total_y2_m", m(g("total_y2")), "cost() total_y2", f2)
    ptool = num("s7_cap_pct_tool", g("cap_pct_tool"), "cost() cap_pct_tool", "{:.0f}")
    pver = num("s7_ver_y1_cap_pct", None if g("ver_y1") is None else 100 * g("ver_y1") / CAP, "cost() ver_y1 / cap", "{:.0f}")
    pt = num("s7_per_t_y2", g("per_t_y2"), "cost() per_t_y2", "{:.0f}")
    lp = num("s7_loan_pct_y2", g("loan_pct_y2"), "cost() loan_pct_y2", "{:.2f}")
    ai = num("s7_ai_ceiling_kzt", g("ai_ceiling_kzt"), "cost() ai_ceiling_kzt", "{:,.0f}")
    capm = num("cost_cap_m", CAP / 1e6, "[brief]", "{:.0f}")
    px = lambda v: W * (v or 0) / CAP
    BT, BH = 120, 56          # bar top / height inside the exhibit
    x1, x2, x3 = px(g("tool")), px(g("tool")) + px(g("ver_y2")), px(g("total_y1"))
    seg = lambda x, w, cls, st="": f'<div class="{cls}" style="position:absolute;top:{BT}px;height:{BH}px;left:{x:.2f}px;width:{w:.2f}px;{st}"></div>'
    bar = (seg(0, W, "", "background:#E9E6DF") + seg(0, x1, "bg-ink") + seg(x1, x2 - x1, "", "background:var(--g3)")
           + seg(x2, x3 - x2, "hatch"))
    ann = lambda x, top, html: (f'<div class="x eb" style="position:absolute;left:{x:.2f}px;top:{top}px;height:{BT - top}px;'
                                f'border-left:1px solid var(--ink);padding-left:8px;font-size:15px;line-height:1.3;white-space:nowrap">{html}</div>')
    anns = (ann(0, 0, f'<b>Carbon Copilot {ASM(tool)}</b> · {ptool}% of cap, est.')
            + ann(x1, 56, f'<b>Independent check {ASM(ver2)}</b>, from year 2, est.')
            + ann(x2, 0, f'<b>+{ASM(extra)} in year 1</b> ({y1} all-in), est.')
            + f'<div style="position:absolute;right:0;top:0;height:{BT + BH + 10}px;border-right:2px solid var(--acc);padding-right:10px;display:flex;align-items:baseline;line-height:1">'
              f'<span class="x lbl" style="margin-right:8px">CFO cap, KZT m</span><span class="num x acc" style="font-size:44px">{capm}</span></div>')
    figs = (f'<div style="position:absolute;left:0;right:0;top:{BT + BH + 40}px;display:grid;grid-template-columns:repeat(3,1fr);column-gap:24px;'
            f'border-top:1px solid var(--ink);padding-top:16px">'
            + big(ASM(y2), "all-in from year 2, est.", 44)
            + big(ASM(f"KZT {pt}"), "per tonne CO2e from year 2, est.", 44)
            + big(ASM(f"{lp}%"), f"of the KZT {LOAN / 1e9:.0f} bn loan, est.", 44) + '</div>')
    ailine = (f'<div style="position:absolute;left:0;top:{BT + BH + 168}px;font-size:15px;color:var(--g1)">'
              f'AI: at most KZT {ai} a year, both readers on every line.</div>')
    title = f"The tool is {ptool}% of the cap; the bank's check is {pver}%."
    foot = ("est. = our assumption inside published ranges; days, rates and sources in README, Cost to run. "
            "Year-2 saving not yet quoted by a verifier.")
    return page(7, "Cost to run", f"KZT million a year · against the KZT {capm} m cap", title,
                exh(96, anns + bar + figs + ailine, 380), foot, "bar")


MK = json.load(open(os.path.join(D, "market.json"), encoding="utf-8"))["inputs"]


def mk(key, fmt="{:,.0f}"):
    """A market.json figure through num(); its source and tag go to numbers_used.json."""
    r = MK[key]
    s = num(f"mkt_{key}", r["value"], f"deck/market.json {key} <- {r['source']}", fmt)
    USED[f"mkt_{key}"]["tag"] = r["tag"]
    return s


def s8(N, C):
    inv = jload(os.path.join(OUT, "inventory.json"))
    fs = sorted((f["tco2e"] for f in inv["by_factor"]), reverse=True)
    top3 = num("top3_factor_pct", 100 * sum(fs[:3]) / inv["total"], "out/inventory.json by_factor top 3", "{:.0f}")
    gl = float(N["eval"]["group k-fold: GL account"]["score"]) if "group k-fold: GL account" in N["eval"] else None
    pc = C.get("per_company") or {}
    p1 = num("s8_per_company_1_m", pc.get(1) and pc[1] / 1e6, "cost() per_company[1]", "{:.2f}")
    p10 = num("s8_per_company_10_m", pc.get(10) and pc[10] / 1e6, "cost() per_company[10]", "{:.2f}")
    yr = lambda k, v: num(f"s8_year_{k}", v, "PLAN_V6 B2 horizons", "{}")
    T = lambda k: ASM(mk(k))
    steps = [
        (f"1 · {yr('h1', 2026)} · Dala Foods", "Limited assurance + farm milk data", T("target_h1"),
         "company · bank's check passed · target",
         f"Built: verifier pack; {num('top3_n', 3, 'out/inventory.json by_factor')} factors = {top3}%",
         f"{num('farms', 25, '[brief] about 25 farms')} farms' data moves ±{num('sens', N['sens'], 'out/inventory.json', '{:.0f}')}% of total", True),
        (f"2 · {yr('h2', '2026–27')} · any 1C ledger", "Connector, shared factors, bank portfolios", T("target_h2"),
         "companies · one bank's borrowers · target",
         f"Built: {num('day1', gl, 'out/eval_hard.csv GL account', '{:.2f}')} on accounts never seen",
         f'Tool per company {ASM(p1)} → {p10} m, est.', False),
        (f"3 · {yr('h3', 2027)} on · exporters", "CBAM-ready product declarations", T("target_h3"),
         f"companies by {yr('h3_kpi', 2029)} · target",
         "Built: every tonne recomputes per line",
         f"First CBAM declarations: {mk('cbam_first_declaration', '{}')}", False)]
    cols = "".join(
        f'<div class="eb" style="position:relative;padding-top:20px">'
        f'<div style="position:absolute;top:-5px;left:0;width:10px;height:10px;border-radius:50%;background:{"var(--acc)" if first else "var(--ink)"}"></div>'
        f'{cap(when)}<div class="x" style="font-size:22px;font-weight:700;line-height:1.2;letter-spacing:-.01em;margin-top:8px;min-height:53px;text-wrap:balance">{t}</div>'
        f'<div class="num x {"acc" if first else ""}" style="font-size:64px;margin-top:14px">{k}</div>'
        f'<div class="x lbl" style="margin-top:6px">{kc}</div>'
        f'<div class="x sub" style="margin-top:10px;color:var(--ink)">{proof}</div>'
        f'<div class="x sub" style="margin-top:2px">{small}</div></div>'
        for when, t, k, kc, proof, small, first in steps)
    G = "display:grid;grid-template-columns:repeat(3,347px);column-gap:24px;justify-content:space-between"
    mkt = [(mk("ets_companies"), "companies in the national ETS"), (f"€{mk('cbam_exports_eur_m')} m", "exports to the EU under CBAM")]
    market = (f'<div style="position:absolute;left:0;right:0;top:282px;border-top:1px solid var(--ink);padding-top:10px">'
              f'{cap("Market · Kazakhstan")}<div style="{G};margin-top:8px">'
              + "".join(f'<div><span class="num x" style="font-size:30px">{v}</span> <span class="x sub">{l}</span></div>' for v, l in mkt)
              + '</div></div>')
    ex = (f'<div style="position:absolute;left:0;right:0;top:12px;border-top:1px solid var(--ink)"></div>'
          f'<div style="position:absolute;left:0;right:0;top:12px;{G}">{cols}</div>{market}')
    foot = ("Targets are ours, not traction: no company beyond the case uses it yet. "
            "Market: ICAP 2024, CBAM Atlas 2025. Milk factor illustrative; est. = assumed hours.")
    return page(8, "Roadmap", "Targets, not traction · market figures sourced",
                "One dairy today. Any 1C ledger next. Then Central Asia's exporters.",
                exh(100, ex, 378), foot, "timeline")


# ---------- browser ----------
def browser():
    # Chrome first: Edge's --headless=new prints nothing for --dump-dom on this machine
    br = next((p for p in (r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                           r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
                           shutil.which("msedge") or "", shutil.which("chrome") or "") if p and os.path.exists(p)), None)
    assert br, "no Edge/Chrome found"
    return br


def run_browser(args, url):
    prof = tempfile.mkdtemp(prefix="cc_deck_")
    try:
        return subprocess.run([browser(), "--headless=new", "--disable-gpu", "--no-first-run", f"--user-data-dir={prof}",
                               *args, url], capture_output=True, timeout=120, text=True, encoding="utf-8", errors="replace")
    finally:
        time.sleep(.5); shutil.rmtree(prof, ignore_errors=True)


def whitespace(png, bg=(0xF7, 0xF5, 0xF0), cell=16, tol=30):
    """Share of 16 px cells with no pixel off the background: a whitespace proxy for the >=45% rule."""
    import numpy as np, pymupdf
    p = pymupdf.Pixmap(png)
    a = np.frombuffer(p.samples, np.uint8).reshape(p.height, p.width, p.n)[:, :, :3].astype(int)
    ink = (np.abs(a - np.array(bg)).max(axis=2) > tol)
    h, w = (p.height // cell) * cell, (p.width // cell) * cell
    occ = ink[:h, :w].reshape(h // cell, cell, w // cell, cell).any(axis=(1, 3))
    return 100 * (1 - occ.mean())


WS_FLOOR = [55, 45, 50, 60, 45, 50, 50, 50]   # PLAN_V5 §B3, per slide
LAYOUTS = {"hero", "split", "flow", "eq", "trio", "table", "bar", "timeline"}


def lint(url, pdf, ws):
    r = run_browser(["--dump-dom", "--virtual-time-budget=8000"], url)
    m = re.search(r'<pre id="lint">(.*?)</pre>', r.stdout, re.S)
    assert m and m.group(1).strip(), "lint: page script did not report (dump-dom empty)"
    S = json.loads(H.unescape(m.group(1)))
    bad = []
    print(f"\n{'#':>2} {'title w':>7} {'lines':>5} {'lead w':>6} {'body w':>6} {'foot w':>6} {'facts':>5} {'minpx':>5} {'white%':>6}  title")
    for s in S:
        i = s["slide"]
        chk = [(s["title_words"] <= 12, f"title {s['title_words']} words > 12"),
               (s["title_lines"] <= 4, f"title {s['title_lines']} lines > 4"),
               (s["ex_count"] == 1, f"{s['ex_count']} exhibits"),
               (s["lead_words"] <= 18, f"lead {s['lead_words']} words > 18"),
               (s["body_words"] <= 35, f"body {s['body_words']} words > 35"),
               (s["ann_max_words"] <= 10, f"an annotation has {s['ann_max_words']} words > 10"),
               (s["foot_words"] <= 30, f"footer {s['foot_words']} words > 30"),
               (s["ann"] <= 2, f"{s['ann']} annotations > 2"),
               (s["x_max_words"] <= 7, f"an exhibit label has {s['x_max_words']} words > 7"),
               (s["min_font"] >= 13, f"text {s['min_font']}px < 13: {s['min_font_el']!r}"),
               (not s["outside"], f"outside the slide: {s['outside']}"),
               (not s["spill"], f"spills out of its column: {s['spill']}"),
               (ws[i - 1] >= WS_FLOOR[i - 1], f"whitespace {ws[i - 1]:.0f}% < {WS_FLOOR[i - 1]}"),
               (s["ex_w"] >= 1000 or i not in (1, 3, 6, 7, 8), f"exhibit {s['ex_w']} px wide < 1000"),
               (s["hero_n"] == (1 if i in (1, 4) else 0), f"{s['hero_n']} elements at >= 100 px (want {1 if i in (1, 4) else 0})"),
               (not s["est_missing"] or i not in (7, 8), f"ASSUMED figure without 'est.': {s['est_missing']}"),
               (not re.search(BANNED, s["text_ban"]), f"banned term outside footer: {re.findall(BANNED, s['text_ban'])}"),
               (not re.search(FORBIDDEN, s["text_all"], re.I), f"forbidden claim: {re.findall(FORBIDDEN, s['text_all'], re.I)}"),
               (i != 8 or not re.search(r"customer|client|pilot|partner|\bTAM\b", s["text_all"], re.I), "slide 8: customer/client/pilot/partner/TAM"),
               (i != 5 or not re.search(r"left the laptop|Disclosed", s["text_all"]), "slide 5: 'left the laptop' or 'Disclosed'")]
        bad += [f"slide {i}: {msg}" for ok, msg in chk if not ok]
        print(f"{i:>2} {s['title_words']:>7} {s['title_lines']:>5} {s['lead_words']:>6} {s['body_words']:>6} "
              f"{s['foot_words']:>6} {s['ann']:>5} {s['min_font']:>5g} {ws[i - 1]:>6.0f}  {s['title']}")
    lay = [s["layout"] for s in S]
    if set(lay) != LAYOUTS or len(lay) != 8: bad.append(f"layouts not 8 distinct: {lay}")
    ks = [s["kicker"] for s in S]
    for w in ("Product", "Architecture", "Safeguards", "Cost to run", "Roadmap"):
        if not any(re.search(r"(?<!\w)" + re.escape(w) + r"(?!\w)", k) for k in ks): bad.append(f"no kicker with {w!r}")
    if not any(re.search(r"(?<!\w)Roadmap(?!\w)", s["kicker"]) and not s["has_table"] for s in S): bad.append("Roadmap kicker is on a slide with a table")
    print("layouts:", " ".join(lay))
    import pymupdf
    doc = pymupdf.open(pdf)
    fl = {f[1:5] for p in doc for f in p.get_fonts()}  # (ext, type, basefont, name); Chrome embeds the Onest variable font as Type3
    fonts = {f[2] or f"{f[1]} (embedded outlines)" for f in fl}
    not_emb = [f for f in fl if f[0] == "n/a" and f[1] != "Type3"]
    mb = os.path.getsize(pdf) / 1e6
    if doc.page_count != 8: bad.append(f"pdf has {doc.page_count} pages, need 8")
    if not_emb: bad.append(f"fonts not embedded: {not_emb}")
    if not any(f[1] == "Type3" for f in fl): bad.append("Onest (Type3 outlines) missing from the pdf")
    if mb >= 10: bad.append(f"pdf {mb:.1f} MB >= 10")
    print(f"pdf: {doc.page_count} pages, {mb:.2f} MB, fonts: {sorted(fonts)}")
    print("LINT OK" if not bad else "LINT FAIL\n  " + "\n  ".join(bad))
    return not bad


def build():
    N = numbers()
    milk, mled, inj, messy = lines()
    C = cost(N)   # PLAN_V5 C0 dict (WP-1)
    V = dict(rt=num("runtime_s", round((jload(os.path.join(D, "measured.json")) or {}).get("run_py_s") or N["runtime"] or 0) or None, "deck/measured.json run_py_s: full python run.py incl. self-checks"))
    X = extra()
    RD, cons = readers()
    cons_state = "measured" if cons and cons.get("cat_acc") is not None and cons.get("dangerous") is not None else "in test"
    slides = [s1(N, V), s2(N, V, messy), s3(N, V, cons_state), s4(N, milk, mled), s5(N, X), s6(N, RD), s7(N, C), s8(N, C)]
    html = open(os.path.join(D, "deck.html"), encoding="utf-8").read().replace("<!--SLIDES-->", "\n".join(slides))
    out_html = os.path.join(D, "_deck_render.html")
    open(out_html, "w", encoding="utf-8").write(html)
    json.dump(USED, open(os.path.join(D, "numbers_used.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    if os.path.exists(os.path.join(D, "assets", "bank.png")) is False and os.path.exists(os.path.join(OUT, "bank_summary.pdf")):
        import pymupdf
        pymupdf.open(os.path.join(OUT, "bank_summary.pdf"))[0].get_pixmap(matrix=pymupdf.Matrix(1.6, 1.6)).save(os.path.join(D, "assets", "bank.png"))

    pdf, url = os.path.join(D, "deck.pdf"), "file:///" + out_html.replace("\\", "/")
    if os.path.exists(pdf):
        os.remove(pdf)
    run_browser(["--no-pdf-header-footer", "--virtual-time-budget=8000", f"--print-to-pdf={pdf}"], url)
    for _ in range(60):  # Edge may return before the file is flushed
        if os.path.exists(pdf) and os.path.getsize(pdf) > 0:
            break
        time.sleep(.5)
    assert os.path.exists(pdf), "browser did not write deck.pdf"
    import pymupdf
    doc = pymupdf.open(pdf)
    pages = os.path.join(D, "pages"); shutil.rmtree(pages, ignore_errors=True); os.makedirs(pages)
    for i, p in enumerate(doc):
        p.get_pixmap(matrix=pymupdf.Matrix(1, 1)).save(os.path.join(pages, f"p{i + 1}.png"))
    ws = [whitespace(os.path.join(pages, f"p{i + 1}.png")) for i in range(doc.page_count)]
    print(f"deck.pdf: {doc.page_count} pages; second reader: {'on (out/llm_eval.json)' if N['llm'] else 'omitted (no out/llm_eval.json)'}; "
          f"ЗАПОЛНИТЬ left: {html.count('ЗАПОЛНИТЬ')}")
    doc.close()
    return lint(url, pdf, ws)


if __name__ == "__main__":
    if "--cost-check" in sys.argv:
        sys.exit(0 if cost_check() else 1)
    if "--measure" in sys.argv or not os.path.exists(os.path.join(D, "measured.json")):
        measure()
    ok = build()
    sys.exit(0 if ok or "--lint" not in sys.argv else 1)
