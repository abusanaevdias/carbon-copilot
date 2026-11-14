"""Build the 8-slide submission deck v4 (editorial layout, v2 heavy type, one chart per slide): numbers from out/*.json|csv -> deck.pdf + pages/pN.png.

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
    cons = (jload(os.path.join(CASE, "h2h", "consensus.json")) or (jload(os.path.join(OUT, "llm_eval.json")) or {}).get("consensus"))
    if h:
        u = 1000 * (h["spend"]["jev"]["usd_per_call"] + h["spend"]["deepseek"]["usd_per_call"])
        cu = num("consensus_usd_1k", u, "h2h/results.json spend: DeepSeek + Jev per call x 1000", "${:.2f}")
        if cons and cons.get("cat_acc") is not None and cons.get("dangerous") is not None:
            R.append(("Both must agree", num("consensus_cat_acc", 100 * cons["cat_acc"], "consensus result", "{:.0f}") + "%", cons["cat_acc"],
                      num("consensus_dangerous", cons["dangerous"], "consensus result"), cons["dangerous"], cons.get("sent", 800), cu, u, "acc"))
        else:
            R.append(("Both must agree", "in test", None, "in test", None, None, cu, u, "acc"))
    return R, cons


# ---------- drawing helpers (v4: editorial report, heavy type, HTML only, no boxes) ----------
def slide(i, kicker, ctx, title, lead, facts, ex, foot, cls="", tail=""):
    f = "".join(f'<div class="ann">{a}</div>' for a in facts)
    return (f'<section class="{cls}"><div class="top"><span class="k">{kicker}</span><span class="c">{ctx}</span></div>'
            f'<div class="grid"><div class="lc"><h1>{title}</h1><p class="lead">{lead}</p>'
            f'<div class="facts">{tail}{f}</div></div><div class="ex">{ex}</div></div>'
            f'<div class="bot"><span class="foot">{foot}</span><span class="pg">{i} / 8</span></div></section>')


def pct(v):
    return f"{max(v * 100, .4):.2f}%"


def brow(label, val, frac, cls="", lab_cls="", h=12):
    """B1 bar: label and value on one line, thin bar below."""
    return (f'<div class="br"><div class="l x"><span class="{lab_cls}">{label}</span><span class="val {lab_cls}">{val}</span></div>'
            f'<div class="b {cls}" style="width:{pct(frac)};height:{h}px"></div></div>')


def row(label, val, frac, cls="", lab_cls=""):
    """B2 bar: label | thin bar | value on one line."""
    return (f'<div class="row x"><span class="{lab_cls}">{label}</span><div><div class="t {cls}" style="width:{pct(frac)}"></div></div>'
            f'<span class="v {lab_cls}">{val}</span></div>')


def cap(text, style=""):
    return f'<div class="cap x" style="{style}">{text}</div>'


def big(v, label, size=46, c="", sub=""):
    """Heavy numeral with a small label under it (v2 style)."""
    return (f'<div><div class="num x {c}" style="font-size:{size}px">{v}</div>'
            f'<div class="x" style="font-size:14px;color:var(--g1);margin-top:8px">{label}</div>'
            + (f'<div class="x" style="font-size:13px;color:var(--g2)">{sub}</div>' if sub else "") + '</div>')


# ---------- the eight slides ----------
def s1(N, V):
    tot = N["total"]
    parts = [("Raw milk from farms", N["milk"], "bg-acc", "milk_t"), ("Bought power and heat", N["s2"], "", "s2_t"),
             ("Fuel burned on site", N["s1"], "", "s1_t"), ("Other suppliers", N["s3"] - N["milk"], "", "s3_other_t")]
    assert abs(sum(p[1] for p in parts) - tot) < 1, "slide 1 bars do not sum to the total"
    share = N["milk"] / tot
    assert .75 < share < .85, "'four-fifths' no longer true"
    bars = "".join(brow(l, f"{num(k, t, 'out/inventory.json')} t · {100 * t / tot:.0f}%", t / parts[0][1], c,
                        "acc" if c else "", 18 if c else 12) for l, t, c, k in parts)
    team = esc(N["team"]) if N["team"] else FILL
    lb = num("leaderboard", N["leaderboard"], "deck/manual.json", "{:.4f}")
    chips = (f'<div class="cap x" style="display:flex;gap:40px;border-top:1px solid var(--rule);padding-top:12px">'
             f'<span>Team {team}</span><span>Leaderboard {lb}</span></div>')
    loan = num("loan_scope_pct", 100 * (N["grid"] + N["gas"] + N["heat"]) / tot, "out/inventory.json by_factor grid+gas+heat", "{:.0f}")
    USED["loan_scope_t"] = {"value": N["grid"] + N["gas"] + N["heat"], "source": "out/inventory.json by_factor EF_GRID_KZ_KWH+EF_NG_M3+EF_HEAT_GCAL"}
    num("milk_pct", 100 * share, "out/inventory.json", "{:.0f}")
    lead = f"The KZT {num('loan_bn', LOAN / 1e9, '[brief]')} bn green loan's solar and heat pumps touch only {loan}% of emissions."
    facts = [f"One command, {num('rows', N['rows'], 'out/inventory.json')} ledger lines, {V['rt']} seconds."]
    tail = (f'<div class="num x" style="font-size:104px;letter-spacing:-.03em">{num("total", tot, "out/inventory.json")}</div>'
            f'<div class="x" style="font-size:15px;color:var(--g1);margin:10px 0 22px">tonnes CO2e in 2025, all gases</div>')
    ex = f'{cap("Emissions by source, tonnes CO2e", "margin-bottom:8px")}<div>{bars}</div>{chips}'
    title = f"Four-fifths of the footprint sits on {num('farms', 25, '[brief] about 25 farms')} farms."
    return slide(1, "Carbon Copilot", "Dala Foods · GHG inventory 2025", title, lead, facts, ex,
                 "Case data, synthetic. Supplier and staff names masked. Detail: bank one-pager, README.", tail=tail)


def s2(N, V, C, messy):
    led = "".join(f'<div class="mono" style="font-size:13px;line-height:1.35;padding:7px 0;border-top:1px solid var(--rule);'
                  f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis"><span class="x">{esc(a["description"])}</span>'
                  f'<br><span class="x" style="color:var(--g2)">{float(a["quantity"]):,.6g} {esc(a["unit"])}</span></div>' for a in messy[:3])
    journey = f'''<div style="display:grid;grid-template-columns:250px 1fr 150px;column-gap:20px;align-items:center">
 <div>{cap("1C export · names hidden", "margin-bottom:6px")}{led}</div>
 <div style="text-align:center"><div class="x acc" style="font-size:15px;font-weight:600">1 command</div>
  <svg width="96" height="24" style="display:block;margin:6px auto"><path d="M2 12H90M80 4l10 8-10 8" fill="none" stroke="var(--acc)" stroke-width="1.5"/></svg>
  <div class="x acc num" style="font-size:36px">{V['rt']} s</div></div>
 <div>{cap("Bank page", "margin-bottom:6px")}<img src="assets/bank.png" style="width:150px;border:1px solid var(--rule)"></div>
</div>'''
    outs = (f'<div style="display:grid;grid-template-columns:repeat(3,1fr);column-gap:24px;margin-top:40px;border-top:1px solid var(--ink);padding-top:18px">'
            + big("1 page", "for the bank")
            + big(num("included", N["n_inc"], "out/audit_all.csv"), "lines, each with a formula")
            + big(num("queue", N["n_queue"], "out/inventory.json review_queue"), "lines for a person", c="acc") + '</div>')
    lead = "Dala's sustainability officer runs it once a year; the bank reads one page."
    facts = [f"Answers all four Inbox requests: bank, retailer, CFO, IT."]
    for k, v in (("intensity", N["intensity"]), ("cp", N["cp"]), ("persons", N["persons"])):
        num(k, v, "out/inventory.json")
    return slide(2, "Product", "Who uses it · the journey", "One command takes the ledger to the bank's page.", lead, facts,
                 journey + outs, "Inbox requests from the case brief; answers in README, Who asked for what.", "w7")


def s3(N, V, cons):
    R = N["rows"]
    for k in ("rows", "cp", "persons", "excluded"):
        num(k, N[k], "out/inventory.json")
    st = [("Lines in", "1C export", R), ("Names masked", f"{N['cp']} suppliers, {N['persons']} people", R),
          ("Purchases only", f"−{N['excluded']} taxes, duplicates", N["n_inc"]), ("With a formula", "quantity × factor", N["n_inc"])]
    chev = '<div class="x" style="color:var(--g3);font-size:22px;padding-top:4px">›</div>'
    cells = chev.join(f'<div><div class="num x" style="font-size:34px">{v:,}</div>'
                      f'<div style="height:6px;background:var(--g3);width:{100 * v / R:.1f}%;margin:12px 0 10px"></div>'
                      f'<div class="x" style="font-size:14px;font-weight:600">{a}</div>'
                      f'<div class="x" style="font-size:13px;color:var(--g2);margin-top:2px">{b}</div></div>' for a, b, v in st)
    flow = (cap("Pipeline · lines at each stage", "margin-bottom:12px")
            + f'<div style="display:grid;grid-template-columns:1fr 14px 1fr 14px 1fr 14px 1fr;column-gap:10px">{cells}</div>')
    seg = [("rules", N["n_rules"], "bg-ink"), ("classifier", N["n_clf"], ""), ("person", N["n_queue"], "bg-acc")]
    for k, (_, n, _) in enumerate(seg):
        num(f"dec_{k}", n, "out/audit_all.csv reasons/flags")
    w = [n / N["n_inc"] for _, n, _ in seg]
    stack = "".join(f'<div class="{c}" style="height:18px;width:{100 * f:.2f}%;background:var(--g3);margin-right:2px"></div>'
                    for (_, _, c), f in zip(seg, w))
    st_ = "in test" if not cons else "measured"
    who = (cap(f"Who decides each of the {N['n_inc']:,} lines", "margin:44px 0 10px")
           + f'<div style="display:flex">{stack}</div>'
           + f'<div style="position:relative;height:96px;margin-top:10px">'
           f'<div style="position:absolute;left:0">{big(f"{N["n_rules"]:,}", "rules", 30)}</div>'
           f'<div style="position:absolute;left:{100 * w[0]:.1f}%">{big(f"{N["n_clf"]:,}", "classifier", 30)}</div>'
           f'<div style="position:absolute;right:0;text-align:right">{big(N["n_queue"], "person", 30, "acc")}</div></div>'
           f'<div style="display:flex;justify-content:flex-end;align-items:center;gap:10px">'
           f'<span class="x" style="font-size:13px;color:var(--g2)">before a person</span>'
           f'<svg width="28" height="10"><path d="M0 5H26" stroke="var(--g3)" stroke-width="1"/></svg>'
           f'<span class="x" style="font-size:14px;font-weight:600;border:1px solid var(--ink);padding:5px 10px">2 AI readers must agree '
           f'<span class="acc" style="font-weight:400">· {st_}</span></span></div>')
    lead = f"{N['rows']:,} lines pass four stages on a laptop in {V['rt']} seconds; re-runs are identical."
    llm = N["llm"] or {}
    g = llm.get("gated_lines_year")
    facts = ["<b>Code does all arithmetic;</b> AI only picks from a list.",
             f"AI sees masked text only, advises on {num('gated_lines_year', g, 'out/llm_eval.json')} lines a year."]
    return slide(3, "Architecture", "From ledger to inventory", "Code decides every number. AI only advises.", lead, facts,
                 flow + who, "Python on a laptop; AI readers via OpenRouter, masked text only. Detail: README.", "w7")


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
                f'{"color:var(--g3);font-weight:400" if c == "g3" else ""}">{b_}</div>'
                + (f'<div class="x" style="font-size:15px;margin-top:10px;border-top:1px solid var(--rule);padding-top:8px">{a}</div>'
                   f'<div class="x" style="font-size:13px;color:var(--g2)">{b}</div>' if a else "") + '</div>'
                for b_, a, b, c in terms)
    formula = f'<div style="display:grid;grid-template-columns:auto auto auto auto auto;justify-content:space-between;align-items:start">{f}</div>'
    lad = [("All emissions", tot, "{:,.0f}"), ("Scope 3 · suppliers", N["s3"], "{:,.0f}"), ("Purchased goods", N["goods"], "{:,.0f}"),
           (f"Raw milk · {N['milk_lines']} lines", N["milk"], "{:,.0f}"), ("This line", t, "{:,.1f}")]
    drill = "".join(row(l, num(f"lad_{k}", v, "out/inventory.json" if k < 4 else "out/audit_all.csv", fm), v / tot,
                        "bg-acc" if k == 4 else "", "acc" if k == 4 else "") for k, (l, v, fm) in enumerate(lad))
    ex = (formula + f'<div style="margin-top:48px;--lw:190px;--vw:80px;--rh:38px">{cap("Drill path, tonnes CO2e", "margin-bottom:4px")}{drill}</div>')
    lead = "From the total, three clicks reach the invoice; a calculator repeats the sum."
    facts = [f"<b>{num('act_pct', N['act_pct'], 'out/inventory.json', '{:.1f}')}%</b> of tonnes from physical quantities, "
             f"{100 - N['act_pct']:.1f}% from spend.",
             f"<b>Milk factor ±20%</b> moves the total ±{num('sens', N['sens'], 'out/inventory.json', '{:.0f}')}%."]
    return slide(4, "Audit trail", "One worked line", "Every tonne traces to one line you can recompute.", lead, facts, ex,
                 "1 t per t = 1 kg per kg. Detail: audit.csv, one row per ledger line.", "w7")


def s5(N, X):
    inj = X["inj"]
    n = N["inj_lines"] or 2400
    rows_ = [("Text classifier", inj.get("clf_starter"), ""),
             ("Classifier + account, unit", inj.get("clf_units"), ""),
             ("Organisers' starter", inj.get("starter"), ""),
             ("Carbon Copilot", N["inj_changed"], "acc")]
    mx = max(v for _, v, _ in rows_ if v is not None)
    bars = "".join(row(l, num(f"inj_{k}", v, "compete/out/baselines.json injection" if k < 3 else "deck/measured.json"),
                       (v or 0) / mx, "bg-acc" if c else "", c) for k, (l, v, c) in enumerate(rows_))
    ex = (cap(f"Lines changed by injected text, of {n:,}", "margin-bottom:6px")
          + f'<div style="--lw:220px;--vw:56px;--rh:52px;border-top:1px solid var(--rule)">{bars}</div>'
          + cap("Shortcuts rebuilt by us on masked data", "margin-top:14px;letter-spacing:.04em;text-transform:none"))
    h = jload(os.path.join(CASE, "h2h", "results.json")) or {}
    forms = sum(h.get("deepseek", {}).get(s, {}).get("forms", 0) for s in ("E3_raw", "E4_raw")) or None
    moved = sum(h[m][s]["prod_t_changed_vs_clean_row"] for m in ("deepseek", "jev") for s in ("E3_raw", "E4_raw", "E4_strip")) if h else None
    lead = (f"We hid {num('inj_forms', 12, 'deck/measured.json injection_forms')} forms of instruction in "
            f"{num('inj_lines', N['inj_lines'], 'deck/measured.json')} test lines; no tonne changed.")
    facts = [f"<b>0 names</b> leave the laptop: {N['cp']} suppliers, {N['persons']} people masked.",
             f"<b>{num('attack_forms', forms, 'h2h/results.json E3+E4 forms')} attack forms</b> on AI readers: "
             f"{num('attack_t_moved', moved, 'h2h/results.json prod_t_changed_vs_clean_row')} tonnes moved."]
    foot = ("Disclosed: in development 4 unmasked rows and 1 carrier name reached our coding assistant. "
            "Fixed; a self-check now stops the run.")
    return slide(5, "Safeguards", "Injection test · masking", "Hidden instructions and leaked names stop before the result.",
                 lead, facts, ex, foot)


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
    ex = cap("Score on the same labelled lines", "margin-bottom:6px") + f'<div style="--lw:200px;--vw:56px;--rh:56px;border-top:1px solid var(--rule)">{bars}</div>'
    lead = (f"On wording it never saw: {num('eval_unseen_wording', uw, 'out/eval_hard.csv', '{:.4f}')}. "
            f"On accounts it never saw: {num('eval_unseen_accounts', ua, 'out/eval_hard.csv', '{:.2f}')}, after one added rule.")
    facts = [f"Before that rule, unseen accounts scored "
             f"{num('eval_gl_before_rule', float(pv['score']) if pv else None, 'out/eval_hard_prev.csv', '{:.2f}')}."]
    title = f"Shortcuts score <span style=\"white-space:nowrap\">{lo:.2f}–{r[2][1]:.2f}</span> on the same lines. Ours: {r[3][1]:.2f}."
    foot = (f"Score = 0.4 × macro-F1 + 0.6 × (1 − WAPE), organisers' formula; random split. Approaches rebuilt by us (compete/). "
            f"Leaderboard: {num('leaderboard', N['leaderboard'], 'deck/manual.json', '{:.4f}')}")
    return slide(6, "Accuracy", f"Same {num('labelled', 1269, 'compete/out/baselines.json rows')} labelled lines · organisers' formula",
                 title, lead, facts, ex, foot)


def s7(N, C):
    m = lambda v: v / 1e6
    tot = C["total"]
    assert abs(C["people"] + C["ver"] + C["compute"] - tot) < 1, "slide 7 parts do not sum to the total"
    llm = N["llm"] or {}
    ai = llm["usd_full_ledger"] * FX if llm.get("usd_full_ledger") is not None else None
    parts = [("Independent check, est.", C["ver"], ""), (f"People, {C['hours']:.0f} h, est.", C["people"], "bg-ink")]
    for k, (l, v, _) in enumerate(parts):
        num(f"cost_{k}", m(v), "deck/build.py cost()", "{:.2f}")
    num("cost_2", m(ai or 0), "out/llm_eval.json usd_full_ledger x 500", "{:.2f}")
    num("cost_3", m(tot), "deck/build.py cost()", "{:.2f}"); num("cost_4", m(CAP), "[brief]", "{:.2f}")
    segs = "".join(f'<div class="{c}" style="height:40px;width:{100 * v / CAP:.2f}%;background:var(--g3);margin-right:2px"></div>' for _, v, c in parts)
    labs = "".join(f'<div style="position:absolute;left:{100 * x / CAP:.2f}%">{big(f"{m(v):.2f}", l, 28)}</div>'
                   for (l, v, _), x in zip(parts, (0, C["ver"])))
    ex = (cap("KZT million a year", "margin-bottom:12px")
          + f'<div style="position:relative"><div style="display:flex;height:40px">{segs}<div style="flex:1;background:#E9E6DF"></div></div>'
          f'<div style="position:absolute;right:0;top:-8px;bottom:-8px;border-right:2px solid var(--acc)"></div></div>'
          f'<div style="position:relative;height:110px;margin-top:14px">{labs}'
          f'<div style="position:absolute;right:0;text-align:right">{big(f"{m(CAP):.0f}", "CFO cap", 28, "acc")}</div></div>'
          + f'<div style="border-top:1px solid var(--ink);padding-top:14px;display:flex;align-items:baseline;gap:14px">'
          f'<span class="num x" style="font-size:46px">{m(tot):.2f}</span><span class="x" style="font-size:15px;color:var(--g1)">total, '
          f'{num("cost_pct", 100 * tot / CAP, "deck/build.py cost()", "{:.0f}")}% of the cap</span></div>')
    title = f"Running it costs <span style=\"white-space:nowrap\">KZT {m(tot):.1f} m</span> a year, {100 * tot / CAP:.0f}% of the cap."
    lead = f"Most is the independent check the bank expects; people need about {C['hours']:.0f} hours."
    facts = [f"<b>AI: about KZT {num('ai_full_ledger_kzt', ai, 'out/llm_eval.json usd_full_ledger x 500', '{:,.0f}')} a year</b> for the full ledger.",
             f"<b>{num('loan_pct', 100 * tot / LOAN, 'deck/build.py cost() / [brief] 5 bn', '{:.2f}')}%</b> of the KZT {LOAN / 1e9:.0f} bn loan."]
    return slide(7, "Cost to run", f"Against the KZT {CAP / 1e6:.0f} m cap", title, lead, facts, ex,
                 "est. = our assumption; rates and sources in README, Cost to run.")


def s8(N, RD):
    gl = float(N["eval"]["group k-fold: GL account"]["score"]) if "group k-fold: GL account" in N["eval"] else None
    mx_d = max([r[4] for r in RD if r[4] is not None] + [1])
    mx_u = max([r[7] for r in RD if r[7] is not None] + [1e-9])

    def cell(txt, frac, sub, c):
        bar = (f'<div style="height:6px;margin-top:8px;background:var(--rule)"><div class="{"bg-acc" if c == "acc" else ""}" '
               f'style="height:6px;width:{pct(frac)};background:var(--g3)"></div></div>') if frac is not None else \
              '<div style="height:6px;margin-top:8px;border-top:1px dashed var(--g3)"></div>'
        return (f'<div><div class="x {c}" style="font-size:22px;font-weight:700;letter-spacing:-.01em">{txt}</div>{bar}'
                + (f'<div class="x" style="font-size:13px;color:var(--g2);margin-top:4px">{sub}</div>' if sub else "") + '</div>')
    head = "".join(f'<div class="cap x" style="letter-spacing:.06em">{h}</div>' for h in
                   ("AI reader", "Right category, unseen accounts", "Dangerous flips under attack", "Cost per 1,000 lines"))
    rows_ = ""
    for nm, acc, accv, dn, dv, n, u, uv, tag in RD:
        c = "acc" if tag == "acc" else ""
        rows_ += (f'<div class="x {c}" style="font-size:15px;font-weight:600;padding-top:4px">{nm}</div>'
                  + cell(acc, accv, "", c)
                  + cell(dn, None if dv is None else dv / mx_d, f"of {n:,}" if n else "", c)
                  + cell(u, uv / mx_u, tag if tag in ("local", "est.") else ("both calls" if c else ""), c))
    grid = (f'<div style="display:grid;grid-template-columns:120px 1fr 1fr 1fr;column-gap:24px;row-gap:16px;align-items:start">'
            f'{head}<div style="grid-column:1/-1;border-top:1px solid var(--ink)"></div>{rows_}</div>')
    lead = "AI only advises; code writes every number. Next: independent check, then other companies."
    facts = [f"Next year: independent check, milk factors from Dala's {N_FARMS} farms.",
             f"Then the next company's ledger; day-1 score {num('day1', gl, 'out/eval_hard.csv GL account', '{:.2f}')}."]
    foot = ("* Laya: category macro-F1 on 1,269 lines; flips = any change, 50 lines. ScaleDown: 148-line sample, 12 attack forms, "
            "cost estimated. Others: 295 lines, 22 forms.")
    return slide(8, "AI stack · Roadmap", "Same masked lines · unseen accounts · attacks",
                 "Five AI readers tested. Two lead, and both must agree.", lead, facts, grid, foot, "w7")


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
    """Share of 16 px cells with no pixel off the background: a whitespace proxy for the >=45% rule."""
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
               (ws[i - 1] >= 45, f"whitespace {ws[i - 1]:.0f}% < 45"),
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
    RD, cons = readers()
    slides = [s1(N, V), s2(N, V, C, messy), s3(N, V, cons), s4(N, milk, mled), s5(N, X), s6(N, X), s7(N, C), s8(N, RD)]
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
