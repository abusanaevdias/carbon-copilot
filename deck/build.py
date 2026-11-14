"""Build the 8-slide submission deck v3 (editorial report layout): numbers from out/*.json|csv -> deck.pdf + pages/pN.png.

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


# ---------- cost model (every rate sourced or ASSUMED; detail in README "Cost to run") ----------
WAGE = 443315          # KZT/month, 2025 national average nominal wage, stat.gov.kz
LOAD = 2.0             # ASSUMED: specialist premium + employer charges over the national average
HOURS_M = 168          # ASSUMED: 40 h x 4.2 weeks
VER_USD = 15000        # ASSUMED point inside the sourced USD 5,000-50,000+ range (glocert)
FX = 500               # ASSUMED KZT per USD
CAP = 15_000_000
LOAN = 5_000_000_000   # [brief]


def cost(N):
    rate = WAGE * LOAD / HOURS_M
    hours = 16 + N["queue"] * .25 + 40 + 80   # yearly run, review queue x 15 min, factor/rules update, maintenance
    people, ver = hours * rate, VER_USD * FX
    total = people + ver
    USED.update(cost_total_kzt={"value": round(total), "source": "deck/build.py cost()"},
                cost_people_kzt={"value": round(people), "source": "deck/build.py cost()"},
                cost_check_kzt={"value": ver, "source": "deck/build.py cost() ASSUMED USD 15k x 500"},
                cost_hours={"value": hours, "source": "deck/build.py cost() ASSUMED hours"})
    return dict(people=people, ver=ver, compute=0, total=total, hours=hours)


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
    laya = jload(os.path.join(CASE, "laya_spike", "out_results.json"))
    sd = jload(os.path.join(CASE, "scaledown_eval", "results.json"))
    return dict(sc=sc, inj=bl.get("injection", {}),
                laya_gl=laya["splits"]["GL account"]["laya"][0] if laya else None,
                sd_cat=sd["E2"]["scaledown_cat_acc"] if sd else None, sd_new=None if not sd else
                round((sd["E2"]["deepseek_cat_acc"] - sd["E2"]["scaledown_cat_acc"]) * sd["E2"]["sample"]),
                jev_x=round(0.000220 / 0.0000667, 1))  # per-call cost, jev_eval/RESULTS.md table


# ---------- drawing helpers (v3: editorial report, HTML only, no boxes) ----------
def slide(i, kicker, ctx, title, lead, facts, ex, foot, cls="", tail=""):
    f = "".join(f'<div class="ann">{a}</div>' for a in facts)
    return (f'<section class="{cls}"><div class="top"><span class="k">{kicker}</span><span class="c">{ctx}</span></div>'
            f'<div class="grid"><div class="lc"><h1>{title}</h1><p class="lead">{lead}</p>'
            f'<div class="facts">{f}{tail}</div></div><div class="ex">{ex}</div></div>'
            f'<div class="bot"><span class="foot">{foot}</span><span class="pg">{i} / 8</span></div></section>')


def pct(v):
    return f"{max(v * 100, .4):.2f}%"


def brow(label, val, frac, cls="", lab_cls=""):
    """B1 bar: label and value on one line, thin bar below."""
    return (f'<div class="br"><div class="l x"><span class="{lab_cls}">{label}</span><span class="{lab_cls}">{val}</span></div>'
            f'<div class="b {cls}" style="width:{pct(frac)}"></div></div>')


def row(label, val, frac, cls="", lab_cls=""):
    """B2 bar: label | thin bar | value on one line."""
    return (f'<div class="row x"><span class="{lab_cls}">{label}</span><div><div class="t {cls}" style="width:{pct(frac)}"></div></div>'
            f'<span class="v {lab_cls}">{val}</span></div>')


def cap(text, style=""):
    return f'<div class="cap x" style="{style}">{text}</div>'


# ---------- the eight slides ----------
def s1(N, V):
    tot = N["total"]
    parts = [("Raw milk from farms", N["milk"], "bg-acc", "milk_t"), ("Bought power and heat", N["s2"], "", "s2_t"),
             ("Fuel burned on site", N["s1"], "", "s1_t"), ("Other suppliers", N["s3"] - N["milk"], "", "s3_other_t")]
    assert abs(sum(p[1] for p in parts) - tot) < 1, "slide 1 bars do not sum to the total"
    share = N["milk"] / tot
    assert .75 < share < .85, "'four-fifths' no longer true"
    bars = "".join(brow(l, f"{num(k, t, 'out/inventory.json')} t · {100 * t / tot:.0f}%", t / parts[0][1], c,
                        "acc" if c else "") for l, t, c, k in parts)
    team = esc(N["team"]) if N["team"] else FILL
    lb = num("leaderboard", N["leaderboard"], "deck/manual.json", "{:.4f}")
    chips = (f'<div class="cap x" style="display:flex;gap:40px;border-top:1px solid var(--rule);padding-top:12px">'
             f'<span>Team {team}</span><span>Leaderboard {lb}</span></div>')
    loan = num("loan_scope_pct", 100 * (N["grid"] + N["gas"] + N["heat"]) / tot, "out/inventory.json by_factor grid+gas+heat", "{:.0f}")
    USED["loan_scope_t"] = {"value": N["grid"] + N["gas"] + N["heat"], "source": "out/inventory.json by_factor EF_GRID_KZ_KWH+EF_NG_M3+EF_HEAT_GCAL"}
    lead = (f"The KZT {num('loan_bn', LOAN / 1e9, '[brief]')} bn green loan funds solar and heat pumps. "
            f"They touch {loan}% of emissions; milk is {num('milk_pct', 100 * share, 'out/inventory.json', '{:.0f}')}%.")
    facts = [f"Made by one command from {num('rows', N['rows'], 'out/inventory.json')} ledger lines in {V['rt']} seconds."]
    tail = (f'<div class="num x" style="font-size:88px;margin-top:26px">{num("total", tot, "out/inventory.json")}</div>'
            f'<div class="x" style="font-size:15px;color:var(--g1);margin-top:8px">tonnes CO2e in 2025, all gases as CO2</div>')
    ex = f'<div>{bars}</div>{chips}'
    title = f"Four-fifths of the footprint sits on {num('farms', 25, '[brief] about 25 farms')} farms."
    return slide(1, "Carbon Copilot", "Dala Foods · GHG inventory 2025", title, lead, facts, ex,
                 "Case data, synthetic. Supplier and staff names masked. Detail: bank one-pager, README.", tail=tail)


def s2(N, V, C, messy):
    led = "".join(f'<div class="mono" style="font-size:13px;line-height:1.35;padding:7px 0;border-top:1px solid var(--rule);'
                  f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis"><span class="x">{esc(a["description"])}</span>'
                  f'<br><span class="x" style="color:var(--g2)">{float(a["quantity"]):,.6g} {esc(a["unit"])}</span></div>' for a in messy)
    journey = f'''<div style="display:grid;grid-template-columns:250px 1fr 150px;column-gap:20px;align-items:center">
 <div>{cap("1C export · names hidden", "margin-bottom:6px")}{led}</div>
 <div style="text-align:center"><div class="x acc" style="font-size:15px;font-weight:500">1 command</div>
  <svg width="96" height="24" style="display:block;margin:6px auto"><path d="M2 12H90M80 4l10 8-10 8" fill="none" stroke="var(--acc)" stroke-width="1.5"/></svg>
  <div class="x acc num" style="font-size:32px">{V['rt']} s</div></div>
 <div>{cap("Bank page", "margin-bottom:6px")}<img src="assets/bank.png" style="width:150px;border:1px solid var(--rule)"></div>
</div>'''
    inbox = [("Bank", "Inventory, every number traceable", f"{num('included', N['n_inc'], 'out/audit_all.csv')} lines, each with a formula"),
             ("Retailer", "Footprint per litre", f"{num('intensity', N['intensity'], 'out/inventory.json intensity', '{:.1f}')} t per tonne; per litre not claimed"),
             ("CFO", f"Under KZT {num('cap_m', CAP / 1e6, '[brief]')} m a year", f"KZT {num('cost_total_m', C['total'] / 1e6, 'deck/build.py cost()', '{:.1f}')} m a year"),
             ("IT & Security", "No names to cloud AI", f"{num('cp', N['cp'], 'out/inventory.json')} suppliers, {num('persons', N['persons'], 'out/inventory.json')} people masked first")]
    rows = "".join(f'<div style="display:grid;grid-template-columns:104px 200px 1fr;column-gap:16px;font-size:14px;'
                   f'padding:7px 0;border-top:1px solid var(--rule)"><span class="x" style="font-weight:500">{w}</span>'
                   f'<span class="x" style="color:var(--g2)">{a}</span><span class="x">{b}</span></div>' for w, a, b in inbox)
    ex = f'{journey}<div style="margin-top:30px">{cap("Inbox request · what the run gives", "margin-bottom:6px")}{rows}</div>'
    lead = "Dala's sustainability officer runs it once a year. The same run answers all four Inbox requests."
    facts = ["<b>Sustainability officer</b> runs it and explains every number.",
             "<b>Bank loan officer</b> reads one page, drills to any line.",
             "<b>Verifier</b> re-performs a sample of lines by hand."]
    return slide(2, "Product", "Who uses it · the journey", "One command takes the ledger to the bank's page.", lead, facts, ex,
                 "Inbox requests from the case brief. Detail: README, Who asked for what.", "w7")


def s3(N, V):
    R = N["rows"]
    st = [("Ledger lines in · 1C export", R, R),
          (f"Names masked · {N['cp']} suppliers, {N['persons']} people", R, R),
          (f"Not purchases removed · −{N['excluded']}", N["n_inc"], N["n_inc"])]
    for k in ("rows", "cp", "persons", "excluded"):
        num(k, N[k], "out/inventory.json")
    bars = "".join(brow(l, f"{v:,}", f / R) for l, v, f in st)
    seg = [("Rules", N["n_rules"], "bg-ink"), ("Classifier", N["n_clf"], ""), ("Person", N["n_queue"], "bg-acc")]
    for k, (_, n, _) in enumerate(seg):
        num(f"dec_{k}", n, "out/audit_all.csv reasons/flags")
    w = [n / N["n_inc"] for _, n, _ in seg]
    stack = "".join(f'<div class="{c}" style="height:12px;width:{100 * f:.2f}%;background:var(--g3);margin-right:2px"></div>'
                    for (_, _, c), f in zip(seg, w))
    who = (f'<div class="br"><div class="l x"><span>Who decides each of the {N["n_inc"]:,} lines</span></div>'
           f'<div style="display:flex">{stack}</div>'
           f'<div class="x" style="position:relative;height:44px;font-size:15px;margin-top:8px">'
           f'<span style="position:absolute;left:0">Rules {N["n_rules"]:,}</span>'
           f'<span style="position:absolute;left:{100 * w[0]:.1f}%">Classifier {N["n_clf"]:,}</span>'
           f'<span class="acc" style="position:absolute;right:0;top:22px;font-weight:500">Person {N["n_queue"]}</span></div></div>')
    lead = (f"{N['rows']:,} ledger lines pass four stages on a laptop in {V['rt']} seconds. "
            "A re-run gives the identical file.")
    llm = N["llm"] or {}
    g = llm.get("gated_lines_year")
    facts = ["<b>Code does all arithmetic.</b> No model can exclude a line or write a number.",
             "<b>Second reader:</b> DeepSeek sees masked text only and picks a label from a fixed list.",
             f"Its answers are cached; it advises on {num('gated_lines_year', g, 'out/llm_eval.json')} lines a year."]
    return slide(3, "Architecture", "From ledger to inventory", "Code decides every number. AI only advises.", lead, facts,
                 bars + who, "Python on a laptop; second reader via OpenRouter, masked text only. Detail: README.")


def s4(N, milk, mled):
    tot = N["total"]
    q_t, ef, t = float(mled["quantity"]), float(milk["ef_value"]), float(milk["tco2e"])
    assert abs(q_t * ef - t) < 0.001, "worked line does not recompute"
    num("worked_qty_t", q_t, "masked/ledger_test.csv", "{:.2f}"); num("worked_ef", ef, "out/audit_all.csv ef_value", "{:g}")
    num("worked_tco2e", t, "out/audit_all.csv", "{:,.1f}")
    terms = [(f"{q_t:,.2f} t", "raw milk bought", f"invoice {esc(mled['doc_no'])}", ""),
             ("×", "", "", "g3"), (f"{ef:g}", "factor", "kg CO2e per kg", ""), ("=", "", "", "g3"),
             (f"{t:,.1f} t", "tonnes CO2e", f"line {milk['line_id']}", "acc")]
    f = "".join(f'<div><div class="num x {"acc" if c == "acc" else ""}" style="font-size:46px;'
                f'{"color:var(--g3)" if c == "g3" else ""}">{big}</div>'
                + (f'<div class="x" style="font-size:15px;margin-top:10px;border-top:1px solid var(--rule);padding-top:8px">{a}</div>'
                   f'<div class="x" style="font-size:13px;color:var(--g2)">{b}</div>' if a else "") + '</div>'
                for big, a, b, c in terms)
    formula = f'<div style="display:grid;grid-template-columns:auto auto auto auto auto;justify-content:space-between;align-items:start">{f}</div>'
    lad = [("All emissions", tot, "{:,.0f}"), ("Scope 3 · suppliers", N["s3"], "{:,.0f}"), ("Purchased goods", N["goods"], "{:,.0f}"),
           (f"Raw milk · {N['milk_lines']} lines", N["milk"], "{:,.0f}"), ("This line", t, "{:,.1f}")]
    drill = "".join(row(l, num(f"lad_{k}", v, "out/inventory.json" if k < 4 else "out/audit_all.csv", fm), v / tot,
                        "bg-acc" if k == 4 else "", "acc" if k == 4 else "") for k, (l, v, fm) in enumerate(lad))
    ex = (formula + f'<div style="margin-top:44px;--lw:190px;--vw:80px;--rh:40px">{cap("Drill path, tonnes CO2e", "margin-bottom:4px")}{drill}</div>')
    lead = "From the total, three clicks reach the invoice. A verifier repeats the sum on a calculator."
    facts = [f"<b>{num('act_pct', N['act_pct'], 'out/inventory.json', '{:.1f}')}%</b> of tonnes come from physical quantities; "
             f"{100 - N['act_pct']:.1f}% are estimated from money spent."]
    return slide(4, "Audit trail", "One worked line", "Every tonne traces to one line you can recompute.", lead, facts, ex,
                 "1 t per t = 1 kg per kg. Detail: audit.csv, one row per ledger line.", "w7")


def s5(N, X):
    inj = X["inj"]
    n = N["inj_lines"] or 2400
    rows_ = [("Text classifier, description only", inj.get("clf_starter"), ""),
             ("Classifier + account and unit", inj.get("clf_units"), ""),
             ("Organisers' starter", inj.get("starter"), ""),
             ("Carbon Copilot", N["inj_changed"], "acc")]
    mx = max(v for _, v, _ in rows_ if v is not None)
    bars = "".join(row(l, num(f"inj_{k}", v, "compete/out/baselines.json injection" if k < 3 else "deck/measured.json"),
                       (v or 0) / mx, "bg-acc" if c else "", c) for k, (l, v, c) in enumerate(rows_))
    ex = (cap(f"Lines changed by hidden text, of {n:,}", "margin-bottom:6px")
          + f'<div style="--lw:240px;--vw:56px;--rh:48px;border-top:1px solid var(--rule)">{bars}</div>'
          + cap("Rebuilt by us on masked data", "margin-top:14px;letter-spacing:.04em;text-transform:none"))
    lead = (f"We hid {num('inj_forms', 12, 'deck/measured.json injection_forms')} forms of instruction in "
            f"{num('inj_lines', N['inj_lines'], 'deck/measured.json')} test lines. Not one category, factor or tonne changed.")
    facts = [f"<b>0 names</b> leave the laptop: {N['cp']} suppliers, {N['persons']} people masked first.",
             "<b>×1000 guard:</b> a tonne typed as kilograms stops the run.",
             f"<b>{num('dup', N['dup'], 'out/inventory.json')} duplicates</b> dropped, VAT removed, "
             f"<b>{num('credit_notes', N['credit'], 'out/inventory.json checks.credit_notes')} credit notes</b> counted negative."]
    foot = ("Disclosed: in development 4 unmasked rows and 1 carrier name reached our coding assistant. "
            "Fixed; a self-check now stops the run.")
    return slide(5, "Safeguards", "Injection test · masking · unit guard",
                 "Hidden instructions, unit slips and leaked names stop before the result.", lead, facts, ex, foot)


def s6(N, X):
    sc, ev = X["sc"], N["eval"]
    r = [("Organisers' starter", sc.get(("random 5-fold", "starter")), "starter"),
         ("Text classifier only", sc.get(("random 5-fold", "clf_units")), "clf_units"),
         ("Everything by money", sc.get(("random 5-fold", "spend_only")), "spend_only"),
         ("Carbon Copilot", sc.get(("random 5-fold", "ours")), "ours")]
    bars = "".join(row(l, num(f"cmp_{k}", v, "compete/out/scores.csv random 5-fold", "{:.2f}"), v or 0,
                       "bg-acc" if k == "ours" else "", "acc" if k == "ours" else "") for l, v, k in r)
    lo = min(v for _, v, _ in r)
    uw = float(ev["group k-fold: description template"]["score"])
    ua = float(ev["group k-fold: GL account"]["score"])
    pv = N["eval_prev"].get("group k-fold: GL account")
    ex = (f'<div style="--lw:200px;--vw:56px;--rh:52px">{bars}</div>'
          + cap(f"Unseen accounts before the added rule: {num('eval_gl_before_rule', float(pv['score']) if pv else None, 'out/eval_hard_prev.csv', '{:.2f}')}",
                "margin-top:16px;border-top:1px solid var(--rule);padding-top:12px;letter-spacing:.04em;text-transform:none"))
    lead = (f"On wording it never saw, ours scores {num('eval_unseen_wording', uw, 'out/eval_hard.csv', '{:.4f}')}; "
            f"on accounts it never saw, {num('eval_unseen_accounts', ua, 'out/eval_hard.csv', '{:.2f}')} after one added rule.")
    readers = [f"Laya {num('laya_gl_ef_acc', 100 * X['laya_gl'] if X['laya_gl'] else None, 'laya_spike/out_results.json GL split ef acc', '{:.0f}')}% right factors"]
    if X["sd_cat"] is not None:
        readers.append(f"ScaleDown {num('scaledown_cat_acc', 100 * X['sd_cat'], 'scaledown_eval/results.json E2', '{:.0f}')}% right categories")
    readers.append(f"Jev = DeepSeek at {num('jev_cost_x', X['jev_x'], 'jev_eval/RESULTS.md per-call cost', '{:.1f}')}× the cost")
    k = len(readers) + 1
    facts = [f"<b>We tested {k} AI readers</b> on unseen accounts: {', '.join(readers)}. So AI only advises; DeepSeek kept."]
    title = f"Shortcuts score {lo:.2f}–{r[2][1]:.2f} on the same lines. Ours: {r[3][1]:.2f}."
    foot = (f"Score = 0.4 × macro-F1 + 0.6 × (1 − WAPE), organisers' formula; random split. Approaches rebuilt by us (compete/). "
            f"Leaderboard: {num('leaderboard', N['leaderboard'], 'deck/manual.json', '{:.4f}')}")
    return slide(6, "Accuracy", f"Same {num('labelled', 1269, 'compete/out/baselines.json rows')} labelled lines · organisers' formula",
                 title, lead, facts, ex, foot, "w7")


def s7(N, C):
    m = lambda v: v / 1e6
    tot = C["total"]
    assert abs(C["people"] + C["ver"] + C["compute"] - tot) < 1, "slide 7 parts do not sum to the total"
    llm = N["llm"] or {}
    ai = llm["usd_full_ledger"] * FX if llm.get("usd_full_ledger") is not None else None
    r = [("Independent check, est.", C["ver"], "", ""), (f"People, {C['hours']:.0f} h, est.", C["people"], "", ""),
         ("Computer and AI", ai or 0, "", ""), ("Total", tot, "bg-acc", "acc"), ("CFO's cap", CAP, "", "")]
    bars = "".join(row(l, num(f"cost_{k}", m(v), "deck/build.py cost()" if k != 2 else "out/llm_eval.json usd_full_ledger x 500", "{:.2f}"),
                       v / CAP, "lite" if k == 4 else c, lc) for k, (l, v, c, lc) in enumerate(r))
    ex = cap("KZT million a year", "margin-bottom:6px") + f'<div style="--lw:200px;--vw:56px;--rh:50px">{bars}</div>'
    share = 100 * tot / CAP
    title = f"Running it costs KZT {m(tot):.1f} m a year, {share:.0f}% of the cap."
    num("cost_pct", share, "deck/build.py cost()", "{:.0f}")
    lead = (f"Most of it is the independent check the bank expects next year. "
            f"People need about {C['hours']:.0f} hours a year.")
    facts = [f"<b>AI: about KZT {num('ai_full_ledger_kzt', ai, 'out/llm_eval.json usd_full_ledger x 500', '{:,.0f}')} a year</b> for the full ledger, masked text only.",
             f"<b>{num('loan_pct', 100 * tot / LOAN, 'deck/build.py cost() / [brief] 5 bn', '{:.2f}')}%</b> of the KZT {LOAN / 1e9:.0f} bn loan it supports."]
    return slide(7, "Cost to run", f"Against the KZT {CAP / 1e6:.0f} m cap", title, lead, facts, ex,
                 "est. = our assumption; rates and sources in README, Cost to run.")


def s8(N):
    gl = float(N["eval"]["group k-fold: GL account"]["score"]) if "group k-fold: GL account" in N["eval"] else None
    steps = [("1", "Built today", "2025 inventory, bank one-pager, audit file.", ""),
             ("2", "Next year", f"Independent check. Milk factors measured on Dala's {N_FARMS} farms.", "acc"),
             ("3", "Then", f"Next company: its accounts and a labelled sample. Day-1 score on unseen accounts "
                           f"{num('day1', gl, 'out/eval_hard.csv GL account', '{:.2f}')}.", "")]
    ex = "".join(f'<div style="display:grid;grid-template-columns:90px 1fr;column-gap:24px;border-top:1px solid var(--rule);padding:18px 0 22px">'
                 f'<div class="num x {c}" style="font-size:64px">{n}</div>'
                 f'<div><div class="x {c}" style="font-size:18px;font-weight:500">{h}</div>'
                 f'<div style="font-size:16px;line-height:1.45;color:var(--g1);margin-top:6px">{t}</div></div></div>'
                 for n, h, t, c in steps)
    lead = "The engine runs on any company's accounting ledger. Only its account list and a labelled sample change."
    facts = [f"<b>Biggest lever is the milk factor:</b> ±20% moves the total "
             f"±{num('sens', N['sens'], 'out/inventory.json', '{:.0f}')}%, more than any software error."]
    return slide(8, "Roadmap", "Three steps", "Next: an independent check for Dala, then other companies.", lead, facts, ex,
                 "Detail: README, Roadmap.")


N_FARMS = 25


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
    """Share of 16 px cells with no pixel off the background: a whitespace proxy for the >=35% rule."""
    import numpy as np, pymupdf
    p = pymupdf.Pixmap(png)
    a = np.frombuffer(p.samples, np.uint8).reshape(p.height, p.width, p.n)[:, :, :3].astype(int)
    ink = (np.abs(a - np.array(bg)).max(axis=2) > tol)
    h, w = (p.height // cell) * cell, (p.width // cell) * cell
    occ = ink[:h, :w].reshape(h // cell, cell, w // cell, cell).any(axis=(1, 3))
    return 100 * (1 - occ.mean())


def lint(url, pdf, ws):
    r = run_browser(["--dump-dom", "--virtual-time-budget=8000"], url)
    m = re.search(r'<pre id="lint">(.*?)</pre>', r.stdout, re.S)
    assert m and m.group(1).strip(), "lint: page script did not report (dump-dom empty)"
    S = json.loads(H.unescape(m.group(1)))
    bad = []
    print(f"\n{'#':>2} {'title w':>7} {'lines':>5} {'lead w':>6} {'body w':>6} {'foot w':>6} {'facts':>5} {'minpx':>5} {'white%':>6}  title")
    for s in S:
        i = s["slide"]
        chk = [(s["title_words"] <= 15, f"title {s['title_words']} words > 15"),
               (s["title_lines"] <= 4, f"title {s['title_lines']} lines > 4"),
               (s["ex_count"] == 1, f"{s['ex_count']} exhibits"),
               (s["lead_words"] <= 30, f"lead {s['lead_words']} words > 30"),
               (s["body_words"] <= 60, f"body {s['body_words']} words > 60"),
               (s["foot_words"] <= 30, f"footer {s['foot_words']} words > 30"),
               (s["ann"] <= 3, f"{s['ann']} facts > 3"),
               (s["x_max_words"] <= 7, f"an exhibit label has {s['x_max_words']} words > 7"),
               (s["min_font"] >= 13, f"text {s['min_font']}px < 13: {s['min_font_el']!r}"),
               (not s["outside"], f"outside the slide: {s['outside']}"),
               (not s["spill"], f"spills out of its column: {s['spill']}"),
               (ws[i - 1] >= 35, f"whitespace {ws[i - 1]:.0f}% < 35"),
               (not re.search(BANNED, s["text_ban"]), f"banned term outside footer: {re.findall(BANNED, s['text_ban'])}"),
               (not re.search(FORBIDDEN, s["text_all"], re.I), f"forbidden claim: {re.findall(FORBIDDEN, s['text_all'], re.I)}")]
        bad += [f"slide {i}: {msg}" for ok, msg in chk if not ok]
        print(f"{i:>2} {s['title_words']:>7} {s['title_lines']:>5} {s['lead_words']:>6} {s['body_words']:>6} "
              f"{s['foot_words']:>6} {s['ann']:>5} {s['min_font']:>5g} {ws[i - 1]:>6.0f}  {s['title']}")
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
    C = cost(N)
    V = dict(rt=num("runtime_s", round(N["runtime"]) if N["runtime"] else None, "deck/measured.json (6.9 s rounded)"))
    X = extra()
    slides = [s1(N, V), s2(N, V, C, messy), s3(N, V), s4(N, milk, mled), s5(N, X), s6(N, X), s7(N, C), s8(N)]
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
    if "--measure" in sys.argv or not os.path.exists(os.path.join(D, "measured.json")):
        measure()
    ok = build()
    sys.exit(0 if ok or "--lint" not in sys.argv else 1)
