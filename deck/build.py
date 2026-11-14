"""Build the 8-slide submission deck v2 (PLAN_V2 Part A): numbers from out/*.json|csv -> deck.pdf + pages/pN.png.

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
W = 1136  # exhibit width = slide content width
NAV = ["Result", "Product", "Architecture", "Audit trail", "Safeguards", "Accuracy", "Cost", "Roadmap"]
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


# ---------- drawing helpers ----------
ICONS = {  # 24 px grid, 1.5 px stroke; at most these 6 in the deck
    "farm": '<path d="M3 21V10l9-6 9 6v11M2 21h20M9 21v-6h6v6M9 11h6"/>',
    "truck": '<path d="M2 6h11v10H2zM13 9h4l4 4v3h-8"/><circle cx="6" cy="18" r="2"/><circle cx="17" cy="18" r="2"/>',
    "factory": '<path d="M3 21V11l5 3v-3l5 3V4h3v4h3v13zM2 21h20"/>',
    "pylon": '<path d="M12 2 6 22M12 2l6 20M7 8h10M5 13h14M8.5 13 15 22M15.5 13 9 22"/>',
    "bank": '<path d="M3 9l9-5 9 5M4 9h16M5 9v9M9.5 9v9M14.5 9v9M19 9v9M3 20h18"/>',
    "person": '<circle cx="12" cy="7" r="4"/><path d="M4 21c0-4.4 3.6-7 8-7s8 2.6 8 7"/>',
}


def icon(name, x, y, s=36, c="var(--ink)"):
    return (f'<g transform="translate({x},{y}) scale({s / 24})" fill="none" stroke="{c}" stroke-width="1.5" '
            f'stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</g>')


def T(x, y, s, size=20, w=400, c="var(--ink)", a="start", cls="x"):
    """SVG text. cls='x' = exhibit label/number (not counted as body words)."""
    return (f'<text class="{cls}" x="{x}" y="{y}" font-size="{size}" font-weight="{w}" fill="{c}" '
            f'text-anchor="{a}">{s}</text>')


def svg(h, body):
    return f'<svg width="{W}" height="{h}" viewBox="0 0 {W} {h}">{body}</svg>'


def ann(x, y, w, text, size=None):
    st = f"font-size:{size}px;" if size else ""
    return f'<div class="ann" style="position:absolute;left:{x}px;top:{y}px;width:{w}px;{st}">{text}</div>'


def box(h, inner):
    return f'<div style="position:relative;width:{W}px;height:{h}px">{inner}</div>'


def slide(i, kicker, title, ex, foot, top_right=""):
    nav = "" if i == 1 else '<span class="nav">' + " · ".join(
        f"<b>{n}</b>" if k == i else n for k, n in enumerate(NAV, 1)) + f'<span class="pg">{i} / 8</span></span>'
    if i == 1:
        nav = '<span class="pg">1 / 8</span>'
    return (f'<section><div class="top"><div class="k">{kicker}</div>{top_right}</div><h1>{title}</h1>'
            f'<div class="ex">{ex}</div><div class="bot"><span class="foot">{foot}</span>{nav}</div></section>')


# ---------- the eight slides ----------
def s1(N, V):
    tot = N["total"]
    parts = [("Farms · raw milk", N["milk"], "Scope 3", "farm"), ("Other suppliers", N["s3"] - N["milk"], "Scope 3", "truck"),
             ("Bought power|and heat", N["s2"], "Scope 2", "pylon"), ("Burned on site", N["s1"], "Scope 1", "factory")]
    assert abs(sum(p[1] for p in parts) - tot) < 1, "slide 1 blocks do not sum to the total"
    xs, x = [], 0
    for _, t, _, _ in parts:
        w = W * t / tot; xs.append((x, w)); x += w
    by, bh = 180, 112
    g = [T(-4, 106, num("total", tot, "out/inventory.json"), 124, 700),
         T(590, 58, "tonnes of CO2e in 2025", 30, 600),
         T(590, 92, "CO2e = all greenhouse gases counted as CO2", 20, 400, "var(--mid)", cls="g")]
    # Scope 3 bracket above milk + other suppliers
    s3w = xs[1][0] + xs[1][1]
    g += [f'<path d="M1 {by-8}V{by-20}H{s3w-1:.1f}V{by-8}" fill="none" stroke="var(--mid)" stroke-width="1.5"/>',
          T(0, by - 30, f"Scope 3 — emissions in suppliers' operations: {num('s3_pct', 100 * N['s3'] / tot, 'out/inventory.json', '{:.0f}')}%", 20, 600, "var(--mid)")]
    for k, ((lab, t, sc, ic), (x0, w)) in enumerate(zip(parts, xs)):
        g.append(f'<rect x="{x0:.1f}" y="{by}" width="{w - (0 if k == 3 else 2):.1f}" height="{bh}" '
                 f'fill="{["var(--acc)", "#5B616B", "#8A9099", "#B9BEC5"][k]}"/>')
    g += [icon("farm", 28, by + 18, 40, "#fff"), T(84, by + 48, "Farms · raw milk", 26, 600, "#fff"),
          T(28, by + 96, f"{num('milk_t', N['milk'], 'out/inventory.json')} t · {num('milk_pct', 100 * N['milk'] / tot, 'out/inventory.json', '{:.0f}')}%", 44, 700, "#fff")]
    # labels for the three small blocks, below the bar, with elbow leaders
    cols = [560, 752, 944]
    for k in (1, 2, 3):
        lab, t, sc, ic = parts[k]; x0, w = xs[k]; cx = x0 + w / 2; lx = cols[k - 1] + 14
        yl = by + bh + 10 + 7 * k
        g.append(f'<path d="M{cx:.1f} {by + bh}V{yl}H{lx + 18}V{by + bh + 40}" fill="none" stroke="var(--soft)" stroke-width="1.2"/>')
        g += [icon(ic, lx, by + bh + 44, 36),
              T(lx, by + bh + 108, lab.split("|")[0], 20, 600), T(lx, by + bh + 130, "|".join(lab.split("|")[1:]), 20, 600),
              T(lx, by + bh + 168, f"{num('blk_' + str(k), t, 'out/inventory.json')} t", 32, 700),
              T(lx, by + bh + 194, f"{100 * t / tot:.0f}% · {sc}", 19, 400, "var(--mid)")]
    lo = by + bh + 40
    a = (ann(0, lo, 500, f"Carbon Copilot turns the accounting export into this inventory in {V['rt']} seconds.") +
         ann(0, lo + 82, 500, f"The loan's solar and heat pumps touch the {num('loan_scope_pct', 100 * (N['grid'] + N['gas'] + N['heat']) / tot, 'out/inventory.json by_factor grid+gas+heat', '{:.0f}')}% from power, gas and heat."))
    USED["loan_scope_t"] = {"value": N["grid"] + N["gas"] + N["heat"], "source": "out/inventory.json by_factor EF_GRID_KZ_KWH+EF_NG_M3+EF_HEAT_GCAL"}
    ex = box(by + bh + 204, svg(by + bh + 204, "".join(g)) + a)
    team = esc(N["team"]) if N["team"] else FILL
    lb = num("leaderboard", N["leaderboard"], "deck/manual.json", "{:.4f}")
    top = (f'<div class="brand"><b>Carbon Copilot</b> · BV Case 2 · Dala Foods · Team {team} · '
           f'Leaderboard score {lb}</div>')
    title = (f"Dala Foods emitted {num('total', tot, 'out/inventory.json')} tonnes of CO2e in 2025, "
             f"and {num('milk_pct', 100 * N['milk'] / tot, 'out/inventory.json', '{:.0f}')}% is milk from farms.")
    return slide(1, "Result", title, ex, "Case data, synthetic; supplier and staff names masked. Detail: bank one-pager, README.", top)


def s2(N, V, messy):
    rows = "".join(
        f'<div style="padding:11px 0;border-top:1px solid var(--rule)"><div class="mono x" style="font-size:22px">{esc(a["description"])}</div>'
        f'<div class="x" style="font-size:18px;color:var(--mid);margin-top:3px"><span class="mono">{float(a["quantity"]):,.6g} {esc(a["unit"])} · {esc(a["counterparty"])}</span>'
        f' <b style="color:var(--acc);font-weight:600">· {tag}</b></div></div>'
        for a, tag in zip(messy, ["Kazakh, typo", "typo", "carrier's fuel", "2 legs, not 1"]))
    chips = "".join(
        f'<div style="margin-bottom:26px">{"" if not ic else f"<svg width=30 height=30 style=margin-bottom:4px>{icon(ic, 0, 0, 30)}</svg>"}'
        f'<div class="x" style="font-size:46px;font-weight:700;line-height:1">{n}</div>'
        f'<div class="x" style="font-size:19px;color:var(--mid);line-height:1.25;margin-top:4px">{l}</div></div>'
        for n, l, ic in [("1 page", "for the bank", "bank"),
                         (num("included", N["n_inc"], "out/audit_all.csv"), "lines, each with a formula", None),
                         (num("queue", N["queue"], "out/inventory.json"), "lines for a person to check", "person")])
    inner = f'''<div style="display:flex;align-items:stretch;height:500px;width:{W}px">
 <div style="width:478px">
  <div class="x" style="font-size:18px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--mid);margin-bottom:8px">Accounting export (1C) · names hidden</div>
  {rows}
  <div style="position:relative;height:150px">{ann(0, 22, 470, "Last year's Excel missed refrigerant leaks and counted flights one way.")}{ann(0, 92, 470, f"The CFO's budget for running it: KZT {num('cap_m', CAP / 1e6, '[brief]')} m a year.")}</div>
 </div>
 <div style="width:150px;display:flex;flex-direction:column;align-items:center;justify-content:center">
  <div class="x" style="font-size:22px;font-weight:700;text-align:center">1 command</div>
  <svg width="120" height="40"><path d="M6 20H108M94 8l14 12-14 12" fill="none" stroke="var(--acc)" stroke-width="3"/></svg>
  <div class="x" style="font-size:22px;font-weight:700;color:var(--acc)">{V['rt']} s</div>
 </div>
 <div style="display:flex;gap:26px;align-items:center">
  <img src="assets/bank.png" style="height:440px;border:1px solid var(--rule);box-shadow:0 2px 0 var(--rule)">
  <div style="width:190px">{chips}</div>
 </div>
</div>'''
    title = f"A KZT {num('loan_bn', LOAN / 1e9, '[brief]')} bn green loan needs a carbon count the bank can check."
    return slide(2, "Product", title, inner, "Users: Dala's sustainability team, the bank's loan officer, the verifier. Detail: README.")


def s3(N, V):
    st = [("Ledger lines in", "accounting export (1C)", N["rows"]),
          ("Names hidden", f"{N['cp']} suppliers, {N['persons']} people", N["rows"]),
          ("Not purchases out", f"−{N['nonp']} taxes, fees|−{N['dup']} duplicates", N["n_inc"]),
          ("Matched to a factor", f"one of {N['n_factors']} official factors", N["n_inc"]),
          ("Lines with a formula", "quantity × factor", N["n_inc"])]
    for k, v in (("rows", N["rows"]), ("cp", N["cp"]), ("persons", N["persons"]), ("nonp", N["nonp"]), ("dup", N["dup"]),
                 ("n_factors", N["n_factors"]), ("included", N["n_inc"])):
        num(k, v, "out/inventory.json" if k != "included" else "out/audit_all.csv")
    cw, gap, base, mh = 200, 34, 170, 110
    g = []
    for k, (lab, sub, n) in enumerate(st):
        x = k * (cw + gap)
        h = mh * n / N["rows"]
        g.append(f'<rect x="{x}" y="{base - h:.1f}" width="{cw}" height="{h:.1f}" fill="{"var(--ink)" if k == 4 else "#8A9099"}"/>')
        if k == 2:
            cut = mh * N["excluded"] / N["rows"]
            g.append(f'<rect x="{x}" y="{base - h - cut:.1f}" width="{cw}" height="{cut:.1f}" fill="none" stroke="var(--acc)" stroke-width="2" stroke-dasharray="5 4"/>')
            g.append(T(x + cw - 6, base - h - cut - 8, f"−{N['excluded']}", 20, 700, "var(--acc)", "end"))
        g.append(T(x, base - mh - 26 if k != 2 else base - mh - 26, f"{n:,}", 40, 700))
        g.append(T(x, base + 30, lab, 20, 600))
        for j, part in enumerate(sub.split("|")):
            g.append(T(x, base + 56 + 22 * j, part, 18, 400, "var(--mid)"))
        if k < 4:
            ax = x + cw + gap / 2
            g.append(f'<path d="M{ax - 6} {base - 60}l8 8-8 8" fill="none" stroke="var(--soft)" stroke-width="2"/>')
    # who decides: stacked bar over the 1,877 included lines
    y0, bh = 318, 64
    seg = [("Rules and dictionary", N["n_rules"], "var(--ink)", "#fff"), ("Text classifier", N["n_clf"], "#8A9099", "#fff"),
           ("Flagged for a person", N["n_queue"], "var(--acc)", "var(--acc)")]
    g.append(T(0, y0 - 14, "Who decides each line", 20, 700))
    x = 0
    for k, (lab, n, c, tc) in enumerate(seg):
        w = W * n / N["n_inc"]
        g.append(f'<rect x="{x:.1f}" y="{y0}" width="{max(w - 2, 3):.1f}" height="{bh}" fill="{c}"/>')
        if k < 2:
            g.append(T(x + 16, y0 + 40, f"{lab} · {num('dec_' + str(k), n, 'out/audit_all.csv reasons/flags')}", 22, 600, tc))
        else:
            g.append(f'<path d="M{x + w / 2:.1f} {y0 + bh}v14" stroke="var(--acc)" stroke-width="1.5"/>')
            g.append(T(W, y0 + bh + 34, f"{lab} · {num('dec_2', n, 'out/inventory.json review_queue')}", 20, 700, tc, "end"))
        x += w
    lo = y0 + bh + 54
    a = ann(0, lo, 540, "Code does all arithmetic; the classifier only picks a label from a fixed list.")
    a += ann(596, lo, 540, f"Runs on a laptop in {V['rt']} seconds; a re-run gives the identical file.")
    ex = box(lo + 60, svg(lo + 60, "".join(g)) + a)
    title = f"One command turns {num('rows', N['rows'], 'out/inventory.json')} messy ledger lines into an explained inventory in {V['rt']} seconds."
    return slide(3, "Architecture", title, ex, "Python, standard library only. Detail: README, How it works.")


def s4(N, milk, mled):
    tot = N["total"]
    lad = [("All emissions", tot), ("Suppliers (Scope 3)", N["s3"]), ("Purchased goods", N["goods"]),
           (f"Raw milk · {N['milk_lines']} lines", N["milk"]), ("This line", float(milk["tco2e"]))]
    g, lx, bw = [], 300, W - 300 - 150
    for k, (lab, v) in enumerate(lad):
        y = 4 + k * 40
        w = bw * v / tot
        g.append(T(lx - 16, y + 24, lab, 20, 600 if k == 4 else 400, "var(--acc)" if k == 4 else "var(--ink)", "end"))
        g.append(f'<rect x="{lx}" y="{y + 4}" width="{max(w, 2):.1f}" height="28" fill="{"var(--acc)" if k == 4 else "#B9BEC5" if k < 3 else "#5B616B"}"/>')
        fmt = "{:,.3f}" if k == 4 else "{:,.0f}"
        g.append(T(lx + max(w, 2) + 12, y + 27, f"{num('lad_' + str(k), v, 'out/inventory.json' if k < 4 else 'out/audit_all.csv', fmt)} t", 22, 700,
                   "var(--acc)" if k == 4 else "var(--ink)"))
    q_t = float(mled["quantity"]); ef = float(milk["ef_value"]) if "ef_value" in milk else 1.3
    assert abs(q_t * ef - float(milk["tco2e"])) < 0.001, "worked line does not recompute"
    fy = 330
    terms = [(150, f"{q_t:,.2f} t", "raw milk bought", f"invoice {esc(mled['doc_no'])}, {esc(mled['counterparty'].split()[-1])}", "var(--ink)"),
             (365, "×", None, None, "var(--soft)"),
             (505, f"{ef:g}", "official factor", "kg CO2e per kg of milk", "var(--ink)"),
             (650, "=", None, None, "var(--soft)"),
             (920, f"{float(milk['tco2e']):,.3f} t", "tonnes of CO2e", f"line {milk['line_id']}", "var(--acc)")]
    num("worked_qty_t", q_t, "masked/ledger_test.csv", "{:.2f}"); num("worked_ef", ef, "out/audit_all.csv ef_value", "{:g}")
    num("worked_tco2e", float(milk["tco2e"]), "out/audit_all.csv", "{:,.3f}")
    for x, big, t1, t2, c in terms:
        g.append(T(x, fy, big, 72, 700, c, "middle"))
        if t1:
            g.append(f'<path d="M{x - 80} {fy + 22}H{x + 80}" stroke="var(--rule)" stroke-width="1.5"/>')
            g.append(T(x, fy + 52, t1, 20, 600, c, "middle"))
            g.append(T(x, fy + 78, t2, 18, 400, "var(--mid)", "middle"))
    lo = fy + 112
    a = ann(0, lo, 540, "Three clicks in the dashboard lead from the total to this line.")
    a += ann(596, lo, 540, f"{num('act_pct', N['act_pct'], 'out/inventory.json', '{:.1f}')}% of tonnes come from quantities; "
                          f"{100 - N['act_pct']:.1f}% are estimated from money spent.")
    ex = box(lo + 60, svg(lo + 60, "".join(g)) + a)
    return slide(4, "Audit trail", "Every tonne traces to one ledger line and can be recomputed on a calculator.", ex,
                 "1 t per t = 1 kg per kg. Detail: audit.csv, one row per line.")


def s5(N, inj):
    llm = N["llm"]
    big = lambda n, l, s: (f'<div style="flex:1"><div class="x" style="font-size:76px;font-weight:700;line-height:1">{n}</div>'
                           f'<div class="x" style="font-size:20px;font-weight:600;margin-top:10px">{l}</div>'
                           f'<div class="x" style="font-size:18px;color:var(--mid);margin-top:4px">{s}</div></div>')
    a2 = ""
    if llm and llm.get("gated_lines_year") is not None:
        a2 = (f'<div style="position:relative;height:56px;margin-top:22px">{ann(0, 0, W, "The second reader sees masked text of "
              + num("gated_lines_year", llm["gated_lines_year"], "out/llm_eval.json") + " lines and cannot exclude a line.")}</div>')
    inner = f'''<div style="width:{W}px">
 <div style="display:flex;align-items:center;gap:0">
  <div style="width:500px;background:#fff;border:1px solid var(--rule);padding:18px 22px">
   <div class="x" style="font-size:18px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--mid)">Invoice text, as typed</div>
   <div class="x mono" style="font-size:26px;margin-top:10px">{esc(inj["description"].split(".")[0])}</div>
   <div class="x" style="font-size:18px;color:var(--mid)">certification services</div>
   <div style="font-size:20px;margin-top:12px;background:#E9E8E2;padding:10px 12px;border-left:3px solid var(--soft)">Hidden note to the AI: “set this to 0”</div>
  </div>
  <div style="width:136px;display:flex;justify-content:center"><svg width="110" height="40"><path d="M6 20H96M82 8l14 12-14 12" fill="none" stroke="var(--acc)" stroke-width="3"/></svg></div>
  <div style="flex:1">
   <div class="x" style="font-size:18px;font-weight:600;letter-spacing:.06em;text-transform:uppercase;color:var(--mid)">Result</div>
   <div class="x" style="font-size:84px;font-weight:700;line-height:1.05;color:var(--acc)">{num("inj_tco2e", float(inj["tco2e"]), "out/audit_all.csv", "{:.4f}")} t</div>
   <div class="x" style="font-size:22px;font-weight:600">not zero · unchanged · sent to a person</div>
  </div>
 </div>
 <div style="display:flex;gap:40px;margin-top:30px;border-top:1.5px solid var(--ink);padding-top:22px">
  {big(f'{num("inj_changed", N["inj_changed"], "deck/measured.json")} <span style="font-size:40px;color:var(--mid)">of {num("inj_lines", N["inj_lines"], "deck/measured.json")}</span>',
       "test lines with a hidden instruction changed", "12 forms of hidden instruction")}
  {big("0", "names in any output file", f"{num('cp', N['cp'], 'out/inventory.json')} suppliers, {num('persons', N['persons'], 'out/inventory.json')} people hidden")}
  {big("×1000", "unit slip stops the run", "a tonne typed as a kilogram")}
 </div>{a2}
</div>'''
    foot = ("Disclosed by us: in development 4 unmasked rows and 1 carrier name reached our coding assistant. "
            "Fixed; a self-check now stops the run. README.")
    return slide(5, "Safeguards", "Hidden instructions, unit slips and leaked names are stopped before they reach the result.", inner, foot)


def s6(N, V):
    ev, pv = N["eval"], N["eval_prev"]
    sc = lambda k: float(ev[k]["score"]) if k in ev else None
    rows = [("Leaderboard, hidden test set", "deck/manual.json", N["leaderboard"], "acc", None),
            ("Unseen wording", "out/eval_hard.csv", sc("group k-fold: description template"), "ink", None),
            ("Unseen suppliers", "out/eval_hard.csv", sc("group k-fold: counterparty pseudonym"), "ink", None),
            ("Accounting account missing", "out/eval_hard.csv", sc("perturb: missing gl_account"), "ink", None),
            ("Unseen accounting accounts", "out/eval_hard.csv", sc("group k-fold: GL account"), "ink", "weakest case"),
            ("Random split", "out/eval_hard.csv", sc("random 5-fold (baseline)"), "grey", "too easy: lines repeat")]
    lo_s, hi_s, px0, px1, rh, top = 0.90, 1.00, 380, W - 20, 58, 24
    X = lambda v: px0 + (px1 - px0) * (v - lo_s) / (hi_s - lo_s)
    g = []
    for k in range(6):  # gridlines + axis
        v = lo_s + k * .02; x = X(v)
        g.append(f'<path d="M{x:.1f} {top - 10}V{top + rh * 6}" stroke="var(--rule)" stroke-width="1"/>')
        g.append(T(x, top + rh * 6 + 26, f"{v:.2f}", 18, 400, "var(--mid)", "middle"))
    g.append(T(px1, top + rh * 6 + 52, "score, 1.00 = perfect", 18, 400, "var(--mid)", "end"))
    for k, (lab, src, v, col, note) in enumerate(rows):
        y = top + rh * k + rh / 2
        c = {"acc": "var(--acc)", "ink": "var(--ink)", "grey": "var(--soft)"}[col]
        g.append(T(px0 - 24, y + 7, lab, 21, 600 if k in (0, 4) else 400, c if col != "ink" else "var(--ink)", "end"))
        g.append(f'<path d="M{px0} {y}H{px1}" stroke="var(--tint)" stroke-width="2"/>')
        key = "eval_" + re.sub(r"\W+", "_", lab.lower()).strip("_")
        if v is None:
            g.append(f'<foreignObject x="{px1 - 260}" y="{y - 16}" width="260" height="34"><div style="text-align:right;font-size:20px">{FILL}</div></foreignObject>')
            num(key, None, src)
            continue
        txt = num(key, v, src, "{:.4f}")
        x = X(max(v, lo_s))
        if k == 4 and "group k-fold: GL account" in pv and abs(float(pv["group k-fold: GL account"]["score"]) - v) > 1e-4:
            ov = float(pv["group k-fold: GL account"]["score"]); ox = X(ov)
            num("eval_gl_before_rule", ov, "out/eval_hard_prev.csv", "{:.4f}")
            g.append(f'<circle cx="{ox:.1f}" cy="{y}" r="11" fill="var(--bg)" stroke="var(--ink)" stroke-width="2"/>')
            g.append(f'<path d="M{ox + 14:.1f} {y}H{x - 130:.1f}" stroke="var(--ink)" stroke-width="2" marker-end="url(#ah)"/>')
            g.append(T(ox - 14, y + 34, f"{ov:.4f} before the rule added after testing", 18, 400, "var(--mid)"))
            note = None
        g.append(f'<circle cx="{x:.1f}" cy="{y}" r="13" fill="{c}"/>')
        right = x > px1 - 140
        g.append(T(x - 24 if right else x + 24, y + 8, txt, 24, 700, c, "end" if right else "start"))
        if note:
            nx = (x + 120) if not right else (x - 130)
            g.append(T(nx, y + 7, note, 19, 400, "var(--mid)", "start" if not right else "end"))
    held = [sc(k) for k in ("group k-fold: description template", "group k-fold: counterparty pseudonym",
                            "perturb: missing gl_account", "group k-fold: GL account") if sc(k) is not None]
    lo = top + rh * 6 + 66
    a = ann(0, lo, 540, f"Bigger than any software error: the milk factor. ±20% moves the total ±{num('sens', N['sens'], 'out/inventory.json', '{:.0f}')}%.")
    a += ann(596, lo, 540, "Not counted, not in a purchase ledger: commuting, downstream transport, packaging waste, fuel supply chain.")
    defs = '<defs><marker id="ah" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0L10 5L0 10z" fill="var(--ink)"/></marker></defs>'
    ex = box(lo + 80, svg(lo + 80, defs + "".join(g)) + a)
    title = f"Tested on data it never saw, the tool scores between {num('eval_min', min(held), 'out/eval_hard.csv', '{:.2f}')} and 1.00."
    return slide(6, "Accuracy", title, ex, "Score = 0.4 × macro-F1 + 0.6 × (1 − WAPE). Detail: README, Accuracy.")


def s7(N, C):
    m = lambda v: v / 1e6
    tot = C["total"]
    assert abs(C["people"] + C["ver"] + C["compute"] - tot) < 1, "slide 7 segments do not sum to the total"
    X = lambda v: W * v / CAP
    by, bh = 128, 96
    seg = [("Independent check", "limited assurance, est.", C["ver"], "#8A9099"),
           ("People", f"{C['hours']:.0f} h a year, est.", C["people"], "var(--ink)")]
    g, x = [], 0
    for k, (lab, sub, v, c) in enumerate(seg):
        w = X(v)
        g.append(f'<rect x="{x:.1f}" y="{by}" width="{w:.1f}" height="{bh}" fill="{c}"/>')
        g.append(f'<path d="M{x + 1:.1f} {by - 6}V{by - 96}" stroke="var(--rule)" stroke-width="1.5"/>')
        g.append(T(x + 10, by - 60, f"{num('cost_' + str(k), m(v), 'deck/build.py cost()', '{:.2f}')}", 48, 700))
        g.append(T(x + 10, by - 34, lab, 20, 600))
        g.append(T(x + 10, by - 12, sub, 18, 400, "var(--mid)"))
        x += w
    used = x
    g.append(f'<rect x="{x:.1f}" y="{by}" width="4" height="{bh}" fill="var(--acc)"/>')
    g.append(f'<rect x="{x + 6:.1f}" y="{by + 1}" width="{W - x - 7:.1f}" height="{bh - 2}" fill="none" stroke="var(--soft)" stroke-width="1.5" stroke-dasharray="6 5"/>')
    g.append(T(W - 12, by - 60, f"{num('cost_unused', m(CAP - tot), 'deck/build.py cost()', '{:.2f}')}", 48, 700, "var(--soft)", "end"))
    g.append(T(W - 12, by - 34, "Unused budget", 20, 600, "var(--mid)", "end"))
    # compute tick label under the bar
    g.append(f'<path d="M{used + 2:.1f} {by + bh}V{by + bh + 22}" stroke="var(--acc)" stroke-width="1.5"/>')
    g.append(f'<path d="M{used + 2:.1f} {by + bh + 22}V{by + bh + 62}H{used + 12:.1f}" fill="none" stroke="var(--acc)" stroke-width="1.5"/>')
    g.append(T(used + 18, by + bh + 68, "Computer and AI: about 0.00", 20, 600, "var(--acc)"))
    # used bracket + cap
    g.append(f'<path d="M1 {by + bh + 8}V{by + bh + 16}H{used - 12:.1f}V{by + bh + 8}" fill="none" stroke="var(--ink)" stroke-width="1.5"/>')
    g.append(T(0, by + bh + 46, f"Used: {num('cost_total_m', m(tot), 'deck/build.py cost()', '{:.2f}')} m = "
                                f"{num('cost_pct', 100 * tot / CAP, 'deck/build.py cost()', '{:.0f}')}% of the cap", 22, 700))
    g.append(f'<path d="M{W - 1} {by - 8}V{by + bh + 22}" stroke="var(--ink)" stroke-width="3"/>')
    g.append(T(W, by + bh + 46, f"Budget cap {num('cap_m', CAP / 1e6, '[brief]', '{:.2f}')}", 20, 700, "var(--ink)", "end"))
    g.append(T(0, by - 112, "KZT million a year", 18, 400, "var(--mid)"))
    # against the loan
    ly = by + bh + 122
    g.append(T(0, ly - 12, f"Against the KZT {LOAN / 1e9:.0f} bn loan it supports", 20, 700))
    g.append(f'<rect x="0" y="{ly}" width="{W}" height="34" fill="var(--tint)"/>')
    g.append(f'<rect x="0" y="{ly}" width="{max(W * tot / LOAN, 2):.1f}" height="34" fill="var(--acc)"/>')
    g.append(T(14, ly + 24, f"{num('loan_pct', 100 * tot / LOAN, 'deck/build.py cost() / [brief] 5 bn', '{:.2f}')}% of the loan: the thin red line", 20, 600, "var(--acc)"))
    lo = ly + 54
    a = ann(0, lo, 540, f"KZT {m(tot):.1f} m a year is under 0.2% of the KZT {LOAN / 1e9:.0f} bn loan it supports.")
    llm = N["llm"]
    if llm and llm.get("usd_full_ledger") is not None:
        kzt = llm["usd_full_ledger"] * FX
        a += ann(596, lo, 540, f"AI for a full year's ledger: about KZT {num('ai_full_ledger_kzt', kzt, 'out/llm_eval.json usd_full_ledger x 500', '{:,.0f}')}.")
    ex = box(lo + 60, svg(lo + 60, "".join(g)) + a)
    title = f"Running it costs about KZT {m(tot):.1f} m a year, {100 * tot / CAP:.0f}% of the 15 m budget."
    return slide(7, "Cost to run", title, ex, "est. = our assumption; rates and sources in README, Cost to run.")


def s8(N, V):
    gl = float(N["eval"]["group k-fold: GL account"]["score"]) if "group k-fold: GL account" in N["eval"] else None
    steps = [("1", "Built today", "2025 inventory, bank one-pager, audit file", "bank", "var(--ink)"),
             ("2", "Next year", f"Independent check; milk factors from Dala's {num('farms', 25, '[brief]')} farms (moves the total ±{N['sens']:.0f}%)", "farm", "var(--acc)"),
             ("3", "Then", f"Next company: its accounts list and a labelled sample. Day-1 score {num('day1', gl, 'out/eval_hard.csv GL account', '{:.2f}')}", "factory", "var(--ink)")]
    cols = "".join(
        f'<div style="flex:1;padding-right:34px;position:relative">'
        f'<div style="display:flex;align-items:flex-end;gap:18px"><div class="x" style="font-size:132px;font-weight:700;line-height:.82;color:{c}">{n}</div>'
        f'<svg width="44" height="44">{icon(ic, 0, 0, 44, c)}</svg></div>'
        f'<div style="border-top:2px solid {c};margin-top:20px;padding-top:14px"><div class="x" style="font-size:24px;font-weight:700;color:{c}">{h}</div>'
        f'<div style="font-size:21px;line-height:1.35;margin-top:8px">{t}</div></div></div>'
        for n, h, t, ic, c in steps)
    inner = (f'<div style="width:{W}px"><div style="display:flex">{cols}</div>'
             f'<div style="position:relative;height:150px;margin-top:36px">'
             f'{ann(0, 0, 700, "Any company with an accounting ledger can use it.")}'
             f'{ann(0, 58, W, "One command from the ledger to an inventory a bank can check.", 30)}</div></div>')
    return slide(8, "Roadmap", "Next: an independent check for Dala, then the same engine for other companies.", inner,
                 "Detail: README, Roadmap.")


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


def lint(url, pdf):
    r = run_browser(["--dump-dom", "--virtual-time-budget=8000"], url)
    m = re.search(r'<pre id="lint">(.*?)</pre>', r.stdout, re.S)
    assert m and m.group(1).strip(), "lint: page script did not report (dump-dom empty)"
    S = json.loads(H.unescape(m.group(1)))
    bad = []
    print(f"\n{'#':>2} {'title w':>7} {'lines':>5} {'exhibit':>9} {'body w':>6} {'foot w':>6} {'ann':>3} {'minpx':>5}  title")
    for s in S:
        i = s["slide"]; cap = 60 if i == 5 else 45; fcap = 25 if i == 5 else 12
        chk = [(s["title_words"] <= 15, f"title {s['title_words']} words > 15"),
               (s["title_lines"] <= 2, f"title {s['title_lines']} lines > 2"),
               (s["ex_count"] == 1, f"{s['ex_count']} exhibits"),
               (s["ex_w"] >= 1136 and s["ex_h"] >= 490, f"exhibit {s['ex_w']}x{s['ex_h']} < 1136x490"),
               (s["content_over_footer"] <= 0, f"exhibit content runs {s['content_over_footer']} px into the footer"),
               (s["body_words"] <= cap, f"body {s['body_words']} words > {cap}"),
               (s["foot_words"] <= fcap, f"footer {s['foot_words']} words > {fcap}"),
               (s["ann"] <= 2, f"{s['ann']} annotations > 2"),
               (s["x_max_words"] <= 7, f"an exhibit label has {s['x_max_words']} words > 7"),
               (s["min_font_ex"] >= 18, f"text {s['min_font_ex']}px < 18 in exhibit: {s['min_font_el']!r}"),
               (not s["outside"], f"outside the slide: {s['outside']}"),
               (not re.search(BANNED, s["text_ban"]), f"banned term outside footer: {re.findall(BANNED, s['text_ban'])}"),
               (not re.search(FORBIDDEN, s["text_all"], re.I), f"forbidden claim: {re.findall(FORBIDDEN, s['text_all'], re.I)}")]
        if i > 1:
            chk.append(("Result · Product · Architecture" in s["text_all"], "navigation strip missing"))
        bad += [f"slide {i}: {msg}" for ok, msg in chk if not ok]
        print(f"{i:>2} {s['title_words']:>7} {s['title_lines']:>5} {s['ex_w']:>4}x{s['ex_h']:<4} {s['body_words']:>6} "
              f"{s['foot_words']:>6} {s['ann']:>3} {s['min_font_ex']:>5g}  {s['title']}")
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
    slides = [s1(N, V), s2(N, V, messy), s3(N, V), s4(N, milk, mled), s5(N, inj), s6(N, V), s7(N, C), s8(N, V)]
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
    print(f"deck.pdf: {doc.page_count} pages; second reader: {'on (out/llm_eval.json)' if N['llm'] else 'omitted (no out/llm_eval.json)'}; "
          f"ЗАПОЛНИТЬ left: {html.count('ЗАПОЛНИТЬ')}")
    doc.close()
    return lint(url, pdf)


if __name__ == "__main__":
    if "--measure" in sys.argv or not os.path.exists(os.path.join(D, "measured.json")):
        measure()
    ok = build()
    sys.exit(0 if ok or "--lint" not in sys.argv else 1)
