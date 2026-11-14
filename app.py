"""Carbon Copilot local web app: drop a ledger CSV in the browser, watch the real pipeline stages, get the inventory.

Run:  python app.py [--port 8765] [--no-open]        (--port 0 = random free port)
Binds 127.0.0.1 only. Same code path as `python run.py --new file.csv`; the submission files are never written.
Second reader pinned to DALA_LLM=cache (or off): no network path at all. Prints counts and error types only.
"""
import atexit, csv, html, io, json, os, re, secrets, shutil, sys, tempfile, threading, time, webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from string import Template
from urllib.parse import parse_qs, unquote, urlsplit

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "cases", "Case2_DalaFoods"))
DEMO = os.path.join(HERE, "demo", "new_lines.csv")
NEED = [os.path.join(DATA, f) for f in ("ledger_train.csv", "ledger_test.csv", "emission_factors.csv")] + \
       [os.path.join(HERE, f) for f in ("out/audit_train_cv.csv", "out/audit_test.csv", ".mask_salt",
                                        "masked/ledger_train.csv", "masked/ledger_test.csv")]
_missing = [p for p in NEED if not os.path.exists(p)]
if _missing:
    sys.exit(f"FAILED: {os.path.relpath(_missing[0], HERE)} not found; run `python run.py` first")
if os.environ.get("DALA_LLM", "").strip().lower() != "off":
    os.environ["DALA_LLM"] = "cache"                      # never openrouter/laya from the app

sys.path.insert(0, HERE)
import mask, pipeline as P, report, llm_fallback  # noqa: E402

REQ = ["line_id", "posting_date", "doc_no", "counterparty", "description", "gl_account", "cost_center", "quantity",
       "unit", "amount_kzt_gross", "vat_kzt"]
MAX_BYTES, MAX_ROWS = 5 * 1024 * 1024, 20_000
LINE_ID = re.compile(r"[A-Za-z0-9_.-]{1,40}")
JOB_ID = re.compile(r"[A-Za-z0-9_-]{11}")
SERVED = {"dashboard.html": "text/html; charset=utf-8", "bank_summary.html": "text/html; charset=utf-8",
          "bank_summary.pdf": "application/pdf", "audit.csv": "text/csv; charset=utf-8"}
STAGES = [("mask", "Mask names"), ("dups", "Duplicates and non-purchase lines"), ("rules", "Rules"),
          ("clf", "Classifier"), ("ef", "Emission factors"), ("checks", "Self-checks"), ("report", "Report")]
NOT_CSV = "This is not a CSV file. Export the ledger as CSV and drop it again."
CSP = ("default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; font-src data:; connect-src 'self'; "
       "frame-src 'self'; frame-ancestors 'self'; base-uri 'none'; form-action 'self'")
LOCK = threading.Lock()   # ponytail: global lock, one job at a time; subprocess per job if concurrency matters
JOBS = {}                 # id -> job dict, insertion order = age
FONTS = report.font_css()
E = html.escape


class Bad(Exception):
    def __init__(self, status, message, **extra):
        super().__init__(message)
        self.status, self.message, self.extra = status, message, extra


def base_ids():
    return {r["line_id"] for n in ("ledger_train.csv", "ledger_test.csv") for r in P.load(os.path.join(HERE, "masked", n))}


def validate(body, filename):
    """Checks the upload before anything else reads it. Returns (rows, normalised UTF-8 comma CSV). Counts only."""
    if not filename.lower().endswith(".csv") or b"\0" in body:
        raise Bad(415, NOT_CSV)
    for enc in ("utf-8-sig", "cp1251"):
        try:
            text = body.decode(enc)
            break
        except UnicodeDecodeError:
            pass
    else:
        raise Bad(415, NOT_CSV)
    first = text.split("\n", 1)[0]
    rd = csv.reader(io.StringIO(text), delimiter=";" if first.count(";") > first.count(",") else ",")
    header = [h.strip() for h in next(rd, [])]
    miss = [c for c in REQ if c not in header]
    if miss:
        raise Bad(422, f"Missing columns: {', '.join(miss)}. Expected: {', '.join(REQ)}. Found: {', '.join(header)[:300]}.")
    rows = []
    for cells in rd:
        if any(c.strip() for c in cells):
            rows.append({k: v.strip() for k, v in zip(header, cells)})
            if len(rows) > MAX_ROWS:
                raise Bad(413, "File is larger than 5 MB / 20,000 lines. Split it by month.")
    if not rows:
        raise Bad(422, "The file has a header but no lines.")
    ids = [r.get("line_id", "") for r in rows]
    bad = sum(not LINE_ID.fullmatch(i) for i in ids) + len(ids) - len(set(ids))
    if bad:
        raise Bad(422, f"{bad} lines have a missing, repeated or unusual line_id.")
    nocp = sum(not r.get("counterparty") for r in rows)
    if nocp:
        raise Bad(422, f"{nocp} lines have no counterparty. The masking step needs one.")
    clash = len(set(ids) & base_ids())
    if clash:
        raise Bad(422, f"{clash} line_ids are already in the base ledger, so these lines are already counted.",
                  base="/base/dashboard.html")
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(REQ)                                       # canonical order (mask.py expects counterparty 4th); extras dropped
    w.writerows([r.get(k, "") for k in REQ] for r in rows)
    return len(rows), out.getvalue()


def safe_cell(v):
    v = str(v)
    try:
        float(v)
        return v
    except ValueError:
        return "'" + v if v[:1] in ("=", "+", "-", "@") else v


def run_job(job, text):
    d, cur = job["dir"], 0
    masked, out = os.path.join(d, "masked"), os.path.join(d, "out")
    saved = mask.MAP_FILE, P.MASKED, P.OUTD
    st = job["stages"]

    def mark(idx, state, detail=None):
        for i in idx:
            st[i]["state"] = state
            if detail is not None:
                st[i]["detail"] = detail[i] if isinstance(detail, dict) else detail
    try:
        mark([0], "running")
        up, mp = os.path.join(d, "upload.csv"), os.path.join(d, "mask_map.csv")
        with open(up, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        for n in ("audit_train_cv.csv", "audit_test.csv"):
            shutil.copyfile(os.path.join(HERE, "out", n), os.path.join(out, n))
        mask.MAP_FILE = mp
        try:
            mask.main(DATA, up, out=masked)
            lk = mask.check(DATA, up, out=masked)         # re-reads the raw file, so delete only after it
        finally:
            for p in (up, mp):
                if os.path.exists(p):
                    os.remove(p)
        if any(lk.values()):
            raise RuntimeError("raw name left after masking")
        train, test, new = (P.load(os.path.join(masked, n)) for n in ("ledger_train.csv", "ledger_test.csv", "ledger_new.csv"))
        persons = set(re.findall(r"PERSON_[0-9a-f]{8}", " ".join(r["description"] + " " + r["cost_center"] for r in new)))
        mark([0], "done", f"{len({r['counterparty'] for r in new})} counterparties · {len(persons)} person spans · leaks 0")

        cur = 1
        mark([1, 2, 3, 4], "running")
        dups = P.dup_ids(train + test + new)              # same calls and order as run.predict_new
        model = P.train_model(train, dups)
        preds = [dict(line_id=r["line_id"], **P.predict(r, model, dups)) for r in new]
        preds = llm_fallback.apply(new, preds, model, dups, train)
        P.write(os.path.join(out, "audit_new.csv"), preds, ["line_id"] + [k for k in preds[0] if k != "line_id"])
        rs = [p["reason"] for p in preds]
        inc = [p for p in preds if p["method"] != "excluded"]
        by_clf = sum(p["reason"].startswith(("classifier", "new wording", "unknown account")) for p in inc)
        fl = lambda p: p["flags"].split("|")
        mark([1, 2, 3, 4], "done", {
            1: f"duplicates {sum('duplicate' in x for x in rs)} · non-purchase "
               f"{sum('not an emission' in x or 'non-purchase' in x for x in rs)}",
            2: f"{len(inc) - by_clf} lines decided by a rule",
            3: f"{by_clf} by the classifier · new wording {sum('new_wording' in fl(p) for p in preds)}",
            4: f"activity {sum(p['method'] == 'activity' for p in preds)} · spend {sum(p['method'] == 'spend' for p in preds)}"
               f" · second reader (cache) {sum(any(f.startswith('second_reader') for f in fl(p)) for p in preds)}"})

        cur = 5
        mark([5, 6], "running")
        bad = P.guard(new, preds, model[4])
        P.MASKED, P.OUTD = masked, out
        report.main(os.path.join(masked, "ledger_new.csv"), os.path.join(out, "audit_new.csv"), pdf=True)
        pdf = os.path.exists(os.path.join(out, "bank_summary.pdf"))
        mark([5, 6], "done", {5: f"off-range lines {len(bad)} · sums and formulas recompute",
                              6: "dashboard, bank page, " + ("PDF 1 page" if pdf else "PDF skipped (no Edge/Chrome)")})

        rows = [r for r in P.load(os.path.join(out, "audit_all.csv")) if r["split"] == "new"]
        for r in rows:
            r["tco2e"] = float(r["tco2e"])
        flg = lambda r: set(filter(None, r["flags"].split("|")))
        job["summary"] = dict(
            total=sum(r["tco2e"] for r in rows), lines=len(rows), pdf=pdf,
            by_scope={s: sum(r["tco2e"] for r in rows if r["scope"] == s) for s in "123"},
            counted=sum(r["method"] != "excluded" for r in rows),
            duplicates=sum(r["reason"].startswith("duplicate") for r in rows),
            nonpurchase=sum(r["method"] == "excluded" and not r["reason"].startswith("duplicate") for r in rows),
            review=sum(bool(flg(r) & report.REVIEW) for r in rows),
            injection=[dict(line_id=r["line_id"], description=r["description"], category=r["category"], tco2e=r["tco2e"])
                       for r in rows if "injection" in flg(r)])
        with open(os.path.join(out, "audit.csv"), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.writer(f)
            w.writerow(report.FIELDS)
            w.writerows([safe_cell(r[k]) for k in report.FIELDS] for r in rows)
        job["elapsed"] = time.time() - job["t0"]
        job["state"] = "done"
        print(f"job {job['id']}: {len(rows)} lines, {job['summary']['total']:.4f} tCO2e, {job['elapsed']:.1f} s")
    except Exception as e:                                # noqa: BLE001 - shown as stage + type only, never the text
        mark([cur], "error")
        job["error"] = (f"Stopped at stage '{STAGES[cur][1]}' ({type(e).__name__}). Nothing was kept. "
                        "Fallback: python run.py --new file.csv")
        job["elapsed"], job["state"] = time.time() - job["t0"], "error"
        print(f"job {job['id']}: stopped at stage '{STAGES[cur][1]}' ({type(e).__name__})")
        shutil.rmtree(d, ignore_errors=True)
    finally:
        mask.MAP_FILE, P.MASKED, P.OUTD = saved
        LOCK.release()


def new_job(name, n, text):
    jid = secrets.token_urlsafe(8)
    d = tempfile.mkdtemp(prefix="cc_job_")
    for sub in ("masked", "out"):
        os.makedirs(os.path.join(d, sub))
    JOBS[jid] = dict(id=jid, name=name, n=n, dir=d, t0=time.time(), state="running", error="", elapsed=0.0,
                     stages=[dict(key=k, label=l, state="pending", detail="") for k, l in STAGES])
    for old in list(JOBS)[:-3]:                           # keep the last 3 job folders
        shutil.rmtree(JOBS.pop(old)["dir"], ignore_errors=True)
    threading.Thread(target=run_job, args=(JOBS[jid], text), daemon=True).start()
    return jid


@atexit.register
def _cleanup():
    for j in JOBS.values():
        shutil.rmtree(j["dir"], ignore_errors=True)


# ---------- pages ----------
CSS = """:root{--bg:#F7F5F0;--ink:#15171A;--g2:#5F6166;--rule:#C9C6BE;--acc:#A3372B}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--ink)}
body{font:15px/1.45 Onest,system-ui,sans-serif;font-variant-numeric:tabular-nums;-webkit-font-smoothing:antialiased}
header{height:52px;border-bottom:1px solid var(--ink);display:flex;align-items:center;justify-content:space-between;
 padding:0 max(24px,calc((100% - 1088px)/2))}header b{font-size:16px}header span{color:var(--g2);font-size:13px}
main{width:1088px;margin:0 auto;padding:40px 0 64px}@media(max-width:1100px){main{width:calc(100% - 48px)}}
a{color:var(--ink);text-underline-offset:3px}:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
.k{font-size:12px;letter-spacing:.12em;text-transform:uppercase;color:var(--g2);margin:0 0 14px}
h1{font-size:56px;line-height:1;font-weight:800;letter-spacing:-.02em;margin:0 0 32px}
h2{font-size:12px;letter-spacing:.12em;text-transform:uppercase;font-weight:600;margin:48px 0 0;padding-bottom:10px;
 border-bottom:1px solid var(--ink)}.small{font-size:12px;color:var(--g2)}
table{width:100%;border-collapse:collapse}td{padding:11px 16px 11px 0;border-bottom:1px solid var(--rule);vertical-align:top}
td.n{text-align:right;padding-right:0;white-space:nowrap}
"""
SHELL = Template("""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Carbon Copilot</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>$fonts
$css
.zone{position:relative;display:flex;flex-direction:column;align-items:center;justify-content:center;height:220px;
 border:1px dashed var(--ink);cursor:pointer;transition:border-color .12s}.zone.over{border:1px solid var(--acc)}
.zone:focus-within{outline:2px solid var(--acc);outline-offset:2px}.zone input{position:absolute;opacity:0;width:1px;height:1px}
.zone b{font-size:20px;font-weight:700}.zone span{font-size:13px;color:var(--g2);margin-top:6px}
.below{display:flex;justify-content:space-between;gap:48px;margin-top:20px}.below p{margin:0;max-width:600px}
button.link{background:none;border:0;padding:0;font:inherit;font-weight:600;color:var(--ink);text-decoration:underline;
 text-underline-offset:3px;cursor:pointer;white-space:nowrap}
#err{color:var(--acc);border-top:1px solid var(--ink);padding-top:10px;margin-top:20px}#err:empty{display:none}
.cols{margin-top:28px}#stages{border-top:1px solid var(--ink)}#stages td:first-child{width:40px;color:var(--g2)}
#stages td:nth-child(3){width:110px;font-size:12px;letter-spacing:.08em;text-transform:uppercase}
#stages tr.pending{color:var(--g2)}#stages tr.running td:nth-child(3),#stages tr.error td:nth-child(3){color:var(--acc)}
#stages tr td{transition:color .15s}@media(prefers-reduced-motion:reduce){*{transition:none!important}}
</style></head><body><header><b>Carbon Copilot</b><span>127.0.0.1 · no network</span></header><main>
<section id="drop"><p class="k">Local · nothing leaves this laptop</p><h1>Drop a ledger. Get the carbon inventory.</h1>
<label class="zone" id="zone"><input type="file" id="file" accept=".csv,text/csv">
<b>Drop a ledger CSV here or choose a file</b><span>CSV, UTF-8 or cp1251, comma or semicolon, up to 5 MB</span></label>
<p id="err" aria-live="polite"></p>
<div class="below"><p>Runs on this laptop. Names are masked before anything else. No upload to any server.
The raw file is deleted after masking.</p><button class="link" id="demo" type="button">or use the demo file (12 lines)</button></div>
<p class="small cols">Expected columns: $cols</p></section>
<section id="prog" hidden><p class="k" id="pk"></p><h1>Processing the ledger.</h1>
<table aria-live="polite"><tbody id="stages"></tbody></table>
<p class="small">Stages 2 to 5 are one pass over the lines: one pass, four counts. Counts come from the audit rows.</p></section>
</main><script>
function q(i){return document.getElementById(i)}
function err(m,base){var e=q("err");e.textContent=m||"";if(base){var a=document.createElement("a");a.href=base;
 a.target="_blank";a.textContent=" Open the base dashboard.";e.appendChild(a)}}
function show(p){q("drop").hidden=p;q("prog").hidden=!p}
function start(url,opt){err("");fetch(url,opt).then(function(r){return r.json().then(function(j){return [r.ok,j]})})
 .then(function(a){if(!a[0]){err(a[1].error,a[1].base);return}history.pushState(null,"","/jobs/"+a[1].id);show(true);poll(a[1].id)})
 .catch(function(){err("The local server did not answer. Is python app.py still running?")})}
function send(f){if(f)start("/api/jobs",{method:"POST",headers:{"X-Filename":encodeURIComponent(f.name)},body:f})}
function render(j){var tb=q("stages");tb.textContent="";j.stages.forEach(function(s,i){var tr=document.createElement("tr");
 tr.className=s.state;[String(i+1),s.label,s.state==="pending"?"waiting":s.state,s.detail].forEach(function(t,k){
 var td=document.createElement("td");td.textContent=t;if(k===3)td.className="n";tr.appendChild(td)});tb.appendChild(tr)});
 q("pk").textContent=j.name+" · "+j.lines+" lines · "+j.elapsed_s.toFixed(1)+" s"}
function poll(id){fetch("/api/jobs/"+id).then(function(r){if(!r.ok)throw 0;return r.json()}).then(function(j){render(j);
 if(j.state==="done")location.href="/jobs/"+id;else if(j.state==="error"){show(false);err(j.error);history.replaceState(null,"","/")}
 else setTimeout(function(){poll(id)},300)}).catch(function(){show(false);err("This job is no longer available.")})}
var z=q("zone");["dragenter","dragover"].forEach(function(t){z.addEventListener(t,function(e){e.preventDefault();z.classList.add("over")})});
z.addEventListener("dragleave",function(){z.classList.remove("over")});
z.addEventListener("drop",function(e){e.preventDefault();z.classList.remove("over");send(e.dataTransfer.files[0])});
window.addEventListener("dragover",function(e){e.preventDefault()});window.addEventListener("drop",function(e){e.preventDefault()});
q("file").addEventListener("change",function(){send(this.files[0]);this.value=""});
q("demo").addEventListener("click",function(){start("/api/jobs?demo=1",{method:"POST"})});
var m=location.pathname.match(/^\\/jobs\\/([A-Za-z0-9_-]+)$$/);if(m){show(true);poll(m[1])}
</script></body></html>""")
RESULTS = Template("""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Carbon Copilot · results</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>$fonts
$css
.hero{font-size:96px;line-height:1;font-weight:800;letter-spacing:-.02em}.hero span{font-size:22px;font-weight:600;color:var(--g2);
 letter-spacing:0;margin-left:10px}.sub{color:var(--g2);margin:10px 0 0}
.grid{display:grid;gap:32px;margin-top:32px;border-top:1px solid var(--ink);padding-top:14px}.g3{grid-template-columns:repeat(3,1fr)}
.g4{grid-template-columns:repeat(4,1fr);border-top-color:var(--rule);margin-top:24px}.grid p{margin:0}
.big{font-size:32px;font-weight:700;line-height:1.1;margin-top:4px!important}.lab{font-size:13px;color:var(--g2)}
.acc{color:var(--acc)}.acts{display:flex;flex-wrap:wrap;gap:12px;align-items:center;margin-top:32px}
.btn{padding:10px 16px;border:1px solid var(--ink);text-decoration:none;font-weight:600;font-size:14px}.btn.p{background:var(--ink);color:var(--bg)}
iframe{display:block;width:100%;height:720px;border:0;border-bottom:1px solid var(--rule)}main+iframe{margin-top:-64px}td.d{word-break:break-word}
</style></head><body><header><b>Carbon Copilot</b><span>127.0.0.1 · no network</span></header><main>
<p class="k">$kicker</p><div class="hero">$total<span>tCO2e</span></div><p class="sub">the uploaded lines, Scope 1 + 2 + 3</p>
<div class="grid g3">$scopes</div><div class="grid g4">$facts</div>
<div class="acts"><a class="btn p" href="$base/dashboard.html#split=new" target="_blank">Open dashboard</a>
<a class="btn" href="$base/bank_summary.html" target="_blank">Bank one-pager</a>$pdf
<a class="btn" href="$base/audit.csv">Audit CSV</a><a href="/">New file</a></div>
<p class="small">Bank pages: whole company inventory with your lines added.</p>$inj
<h2>Dashboard · new lines</h2></main>
<iframe src="$base/dashboard.html#split=new" title="Dashboard, new lines"></iframe></body></html>""")


def results_page(job):
    S, b = job["summary"], f"/jobs/{job['id']}"
    tot = S["total"]
    pct = lambda x: f"{100 * x / tot:.1f}%" if tot else "0%"
    scopes = "".join(f'<div><p class="lab">Scope {s} · {pct(S["by_scope"][s])}</p><p class="big">{S["by_scope"][s]:,.1f} t</p></div>'
                     for s in "123")
    facts = [("Counted lines", S["counted"], ""), ("Excluded", S["duplicates"] + S["nonpurchase"],
              f"duplicates {S['duplicates']} · non-purchase {S['nonpurchase']}"),
             ("To review", S["review"], "acc"), ("Instruction-like text ignored", len(S["injection"]), "acc")]
    facts = "".join(f'<div><p class="lab">{l}</p><p class="big{" acc" if x == "acc" else ""}">{v}</p>'
                    f'{f"<p class=small>{x}</p>" if x and x != "acc" else ""}</div>' for l, v, x in facts)
    inj = ""
    if S["injection"]:
        inj = ("<h2>Instruction-like text found and ignored</h2><table>" + "".join(
            f'<tr><td>{E(r["line_id"])}</td><td class="d">{E(r["description"])}</td><td>{E(r["category"])}</td>'
            f'<td class="n">{r["tco2e"]:.4f} t</td></tr>' for r in S["injection"]) + "</table>")
    pdf = (f'<a class="btn" href="{b}/bank_summary.pdf">Bank PDF</a>' if S["pdf"]
           else '<span class="small">Bank PDF needs Edge or Chrome; open the one-pager and print it.</span>')
    return RESULTS.substitute(fonts=FONTS, css=CSS, base=b, total=f"{tot:,.1f}", scopes=scopes, facts=facts, pdf=pdf,
                              inj=inj, kicker=E(f"{job['name']} · {S['lines']} lines · {job['elapsed']:.1f} s"))


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):                            # no request lines (paths carry job ids only, but keep it quiet)
        pass

    def _send(self, code, body, ctype="application/json", extra=()):
        if isinstance(body, str):
            body = body.encode("utf-8")
        elif isinstance(body, dict):
            body = json.dumps(body).encode()
        self.send_response(code)
        for k, v in [("Content-Type", ctype), ("Content-Length", str(len(body))), ("Content-Security-Policy", CSP),
                     ("X-Content-Type-Options", "nosniff"), ("Cache-Control", "no-store"),
                     ("Referrer-Policy", "no-referrer"), *extra]:
            self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _host_ok(self):
        port = self.server.server_address[1]
        ok = {f"127.0.0.1:{port}", f"localhost:{port}"}
        origin = self.headers.get("Origin")
        if self.headers.get("Host", "") not in ok or (origin and origin.removeprefix("http://") not in ok):
            self._send(403, {"error": "Forbidden host."})
            return False
        return True

    def do_GET(self):
        if not self._host_ok():
            return
        path = urlsplit(self.path).path
        if path == "/":
            return self._send(200, SHELL_HTML, "text/html; charset=utf-8")
        if path == "/base/dashboard.html":
            with open(os.path.join(HERE, "out", "dashboard.html"), "rb") as f:
                return self._send(200, f.read(), "text/html; charset=utf-8")
        parts = path.split("/")
        job = JOBS.get(parts[2]) if parts[1] == "jobs" and len(parts) in (3, 4) and JOB_ID.fullmatch(parts[2]) else None
        if job and len(parts) == 3:
            done = job["state"] == "done"
            return self._send(200, results_page(job) if done else SHELL_HTML, "text/html; charset=utf-8")
        if job and len(parts) == 4 and parts[3] in SERVED and job["state"] == "done":
            with open(os.path.join(job["dir"], "out", parts[3]), "rb") as f:
                data = f.read()
            att = [("Content-Disposition", f'attachment; filename="{parts[3]}"')] if parts[3][-4:] in (".pdf", ".csv") else []
            return self._send(200, data, SERVED[parts[3]], att)
        if len(parts) == 4 and parts[1] == "api" and parts[2] == "jobs":
            job = JOBS.get(parts[3]) if JOB_ID.fullmatch(parts[3]) else None
            if job:
                el = job["elapsed"] if job["state"] != "running" else time.time() - job["t0"]
                return self._send(200, dict(state=job["state"], stages=job["stages"], error=job["error"], name=job["name"],
                                            lines=job["n"], elapsed_s=round(el, 2), summary=job.get("summary")))
        self.not_found()

    def not_found(self):
        self._send(404, {"error": "Not found."})

    def do_POST(self):
        if not self._host_ok():
            return
        u = urlsplit(self.path)
        if u.path != "/api/jobs":
            return self.not_found()
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BYTES:                                 # refuse before reading; drain so the client sees the 413
            left = min(n, 64 * MAX_BYTES)
            while left > 0 and (chunk := self.rfile.read(min(left, 65536))):
                left -= len(chunk)
            self.close_connection = True
            return self._send(413, {"error": "File is larger than 5 MB / 20,000 lines. Split it by month."})
        body = self.rfile.read(n) if n else b""
        if not body and parse_qs(u.query).get("demo") == ["1"]:
            name, body = "new_lines.csv", open(DEMO, "rb").read()
        else:
            name = os.path.basename(unquote(self.headers.get("X-Filename", "upload.csv")))[:120] or "upload.csv"
        try:
            rows, text = validate(body, name)
        except Bad as b:
            return self._send(b.status, {"error": b.message, **b.extra})
        if not LOCK.acquire(blocking=False):
            return self._send(409, {"error": "Another file is being processed. Try again in a few seconds."})
        self._send(202, {"id": new_job(name, rows, text)})


SHELL_HTML = SHELL.substitute(fonts=FONTS, css=CSS, cols=", ".join(REQ))


def make_server(port=8765):
    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main(argv):
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else 8765
    srv = make_server(port)
    url = f"http://127.0.0.1:{srv.server_address[1]}/"
    print(f"Carbon Copilot on {url} (127.0.0.1 only, second reader {os.environ['DALA_LLM']}); Ctrl+C to stop")
    if "--no-open" not in argv:
        webbrowser.open(url)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()


if __name__ == "__main__":
    main(sys.argv[1:])
