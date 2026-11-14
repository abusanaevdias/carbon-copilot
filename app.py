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
        job["counts"] = dict(cp=len({r["counterparty"] for r in new}), persons=len(persons))   # replaced whole, never mutated
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
        job["counts"] = {**job["counts"], "excluded": len(preds) - len(inc), "rule": len(inc) - by_clf, "clf": by_clf}
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
        job["counts"] = {**job["counts"], "offrange": len(bad)}
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
                       for r in rows if "injection" in flg(r)],
            review_top=[dict(line_id=r["line_id"], description=r["description"], flags=sorted(flg(r) & report.REVIEW),
                             tco2e=r["tco2e"]) for r in sorted((r for r in rows if flg(r) & report.REVIEW),
                                                               key=lambda r: -abs(r["tco2e"]))[:5]])
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
    JOBS[jid] = dict(id=jid, name=name, n=n, dir=d, t0=time.time(), state="running", error="", elapsed=0.0, counts={},
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
# "Run receipt" (APP_REDESIGN.md): one run, seven stages, the same box from drop to file card. Tokens in section 4.
CSS = """:root{--paper:#FFFFFF;--sheet:#FFFFFF;--well:#F1F4F8;--pass:#F8FAFC;--ink:#0F172A;--ink-2:#1E293B;--on-ink:#CBD5E1;
 --mute:#475569;--faint:#8391A5;--rule:#CBD5E1;--acc:#1D4ED8;--acc-wash:#E4ECFD;--bar2:#767B86;--bar3:#BEC0C5;
 --alert:#B42318;--alert-bg:#FEF0EE;--alert-line:#E0796B;--caution:#8A4B00;--caution-bg:#FDF3DC;--caution-line:#D99A2B;
--t-fast:150ms;--t-base:250ms;--t-slow:400ms;--ease:cubic-bezier(.22,1,.36,1);
 --lift:0 1px 0 rgba(21,23,26,.04),0 12px 32px -16px rgba(21,23,26,.18);--mono:ui-monospace,"Cascadia Mono",Consolas,monospace}
/* Palettes (PALETTES.md): :root is slate (default); data-theme on <html> picks another. --alert* = needs a human (red),
   --caution* = instruction-like text ignored (amber); never the accent. APCA: tools/apca_check.py --all */
:root[data-theme=warm]{--paper:#F7F5F0;--sheet:#FFFFFF;--well:#EFECE5;--pass:#F8F6F3;--ink:#15171A;--ink-2:#2A2D31;--on-ink:#D9D6CF;
 --mute:#4A4D52;--faint:#8A8C90;--rule:#C9C6BE;--acc:#A3372B;--acc-wash:#F3E4DF;--bar2:#727478;--bar3:#BBB8B0;
 --alert:#A3271C;--alert-bg:#F8E3DE;--alert-line:#D27565;--caution:#7E4A00;--caution-bg:#F6EBD0;--caution-line:#C99433}
:root[data-theme=cool]{--paper:#F4F5F6;--sheet:#FFFFFF;--well:#E8EAED;--pass:#F5F6F7;--ink:#111418;--ink-2:#272B31;--on-ink:#D0D4D9;
 --mute:#464B52;--faint:#868B92;--rule:#C6CAD0;--acc:#1E5E48;--acc-wash:#DFEDE6;--bar2:#75777A;--bar3:#B9BABC;
 --alert:#B42318;--alert-bg:#FCEDEB;--alert-line:#E0796B;--caution:#8A4B00;--caution-bg:#FBF1DA;--caution-line:#D99A2B}
:root[data-theme=mineral]{--paper:#ECF0F3;--sheet:#FAFBFC;--well:#DFE5EA;--pass:#EFF2F5;--ink:#12171C;--ink-2:#262D34;--on-ink:#CDD5DC;
 --mute:#424A53;--faint:#7F8892;--rule:#B9C2CB;--acc:#A3372B;--acc-wash:#F3E2DE;--bar2:#707478;--bar3:#B1B5B9;
 --alert:#A3271C;--alert-bg:#F7E2DE;--alert-line:#D27565;--caution:#7E4A00;--caution-bg:#F5EAD2;--caution-line:#C99433}
:root[data-theme=graphite]{--paper:#15171A;--sheet:#1D2024;--well:#24282D;--pass:#2A2E34;--ink:#ECEEF0;--ink-2:#D3D7DC;--on-ink:#454B53;
 --mute:#B8BEC5;--faint:#7E858E;--rule:#3A4047;--acc:#3CC6B2;--acc-wash:#163430;--bar2:#B8BABD;--bar3:#7C7E81;
 --alert:#FFA898;--alert-bg:#341B18;--alert-line:#D0675A;--caution:#F5C15E;--caution-bg:#30260F;--caution-line:#B88A2E;
 --lift:0 1px 0 rgba(0,0,0,.2),0 12px 32px -16px rgba(0,0,0,.6);color-scheme:dark}
*{box-sizing:border-box}html{background:var(--paper)}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 Onest,system-ui,sans-serif;font-variant-numeric:tabular-nums;
 -webkit-font-smoothing:antialiased}
header{height:56px;border-bottom:1px solid var(--rule);display:flex;align-items:center;justify-content:space-between;
 padding:0 max(48px,calc((100% - 1184px)/2))}header b{font-size:16px;font-weight:700;letter-spacing:-.01em}
.wm{color:inherit;text-decoration:none;border-radius:8px}
.pill{display:flex;align-items:center;gap:12px;height:32px;padding:0 14px;border:1px solid var(--rule);border-radius:999px;
 background:var(--sheet);font-size:13px;color:var(--mute)}.pill i{width:8px;height:8px;border-radius:50%;background:var(--ink)}
.pill span+span{border-left:1px solid var(--rule);padding-left:12px}
main{max-width:1280px;margin:0 auto;padding:40px 48px 96px}
.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));column-gap:24px;row-gap:32px}
.sg{display:grid;grid-template-columns:subgrid}.z{margin-top:32px}
@media(max-width:959px){.grid{grid-template-columns:minmax(0,1fr)}.grid>*,.sg>*{grid-column:1/-1!important}}
.zn{display:inline-block;width:28px;margin-left:-28px;font:400 12px/1 var(--mono);letter-spacing:0;color:var(--mute);vertical-align:baseline}
.zn.in{width:auto;margin:0 12px 0 0}
a{color:var(--ink);text-underline-offset:3px}:focus-visible{outline:2px solid var(--acc);outline-offset:2px}
h1{font-size:56px;line-height:1;font-weight:800;letter-spacing:-.025em;margin:0 0 32px;text-wrap:balance}
h2{font-size:24px;line-height:1.2;font-weight:700;letter-spacing:-.01em;margin:0 0 16px}
p{margin:0}.small{font-size:13px;line-height:1.4;color:var(--mute)}.mono{font-family:var(--mono);font-size:13px}
.sr{position:absolute;width:1px;height:1px;overflow:hidden;clip-path:inset(50%);white-space:nowrap}
.btn{display:inline-flex;align-items:center;gap:8px;height:44px;padding:0 18px;border:1px solid var(--ink);border-radius:8px;
 background:var(--sheet);color:var(--ink);font:600 15px/1 Onest,system-ui,sans-serif;text-decoration:none;cursor:pointer;
 transition:transform 100ms,background-color var(--t-fast)}.btn:hover{background:var(--well)}.btn:active{transform:scale(.98)}
.chip{display:inline-block;font-family:var(--mono);font-size:12px;line-height:1.5;padding:2px 8px;border:1px solid var(--rule);
 border-radius:8px;background:var(--sheet);color:var(--mute)}
.rail{list-style:none;margin:0;padding:0}
.rail li{position:relative;display:grid;grid-template-columns:18px minmax(0,1fr);column-gap:14px;min-height:48px;padding:4px 0 10px}
.rail li:not(:last-child)::before{content:"";position:absolute;left:8.5px;top:27px;bottom:-3px;width:1px;background:var(--rule)}
.rail .g{margin-top:2px}.rail .l{font-size:15px;line-height:1.45}.rail .rw{display:none}
.rail .d{grid-column:2;max-width:44ch;display:flex;flex-wrap:wrap;column-gap:16px;font-size:13px;line-height:1.4;color:var(--mute)}
.rail [data-st=pending] .l{color:var(--mute)}.rail .d .al{color:var(--alert);font-weight:600}.rail [data-st=running] .l{font-weight:600}
.rail [data-st=done] .d,.rail [data-st=error] .d{animation:up var(--t-base) var(--ease) both}
.rail .pass{margin:0 -12px;padding-left:12px;padding-right:12px;background:var(--pass)}
.rail li.pass::before{left:20.5px}.rail .pa{padding-top:12px;border-radius:12px 12px 0 0}.rail li.pa::before{top:35px}
.rail .pz{padding-bottom:14px;border-radius:0 0 12px 12px}.rail li:has(+.pa)::before{bottom:-11px}
.rail .pn{grid-column:2;margin-top:8px;max-width:44ch;font-size:13px;line-height:1.4;color:var(--mute)}
.g{position:relative;display:inline-block;width:18px;height:18px;border-radius:50%;flex:none;color:var(--ink)}
[data-st=pending]>.g{border:1.5px solid var(--faint)}
[data-st=running]>.g,.g.run{border:2px solid var(--rule);border-top-color:currentColor;border-right-color:currentColor;
 animation:spin 800ms linear infinite}
[data-st=done]>.g,.g.ok{background:currentColor;animation:pop var(--t-fast) var(--ease) both}
[data-st=done]>.g::after,.g.ok::after{content:"";position:absolute;left:6px;top:3px;width:4px;height:8px;border:solid var(--paper);
 border-width:0 2px 2px 0;transform:rotate(45deg)}
[data-st=error]>.g,.g.x{color:var(--acc);animation:pop var(--t-fast) var(--ease) both}
[data-st=error]>.g::before,[data-st=error]>.g::after,.g.x::before,.g.x::after{content:"";position:absolute;left:8px;top:1px;width:2px;
 height:16px;border-radius:1px;background:currentColor;transform:rotate(45deg)}[data-st=error]>.g::after,.g.x::after{transform:rotate(-45deg)}
@keyframes spin{to{transform:rotate(1turn)}}@keyframes pop{from{opacity:.3;transform:scale(.7)}}
@keyframes up{from{opacity:0;transform:translateY(4px)}}@keyframes up8{from{opacity:0;transform:translateY(8px)}}
@keyframes grow{from{transform:scaleX(0)}}
@view-transition{navigation:auto}::view-transition-group(*){animation-duration:400ms;animation-timing-function:cubic-bezier(.22,1,.36,1)}
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition-property:opacity!important;
 transition-duration:150ms!important}::view-transition-group(*),::view-transition-old(*),::view-transition-new(*){animation:none!important}[data-st=running]>.g,.g.run{border:1.5px solid currentColor;
 background:linear-gradient(90deg,currentColor 50%,transparent 50%)}.rail [data-st=running] .rw{display:inline;margin-left:8px;
 font-size:13px;font-weight:400;color:var(--mute)}}
@media(forced-colors:active){[data-st=done]>.g,.g.ok{background:CanvasText}.rail .pass{border:1px solid CanvasText}}
"""
HEADER = ('<header><a class="wm" href="/"><b>Carbon Copilot</b></a><p class="pill"><i aria-hidden="true"></i><span>Local</span><span>127.0.0.1</span>'
          '<span>no network</span></p></header>')
WORD = {"pending": "pending", "running": "running", "done": "done", "error": "stopped"}
DESC = ["Names replaced before anything else reads the file", "Repeated documents and non-purchase lines set aside",
        "Lexicon and account rules", "Classifier for wording the rules do not know", "Activity or spend factor",
        "Sums and formulas recomputed", "Dashboard, bank page, PDF"]


OFFRANGE = re.compile(r"off-range lines [1-9]")              # flagged guard count gets the alert colour


def rail(stages):
    """Seven rows keyed by stage; the client updates them in place. Detail parts were joined with ' · ' upstream."""
    out = []
    for i, s in enumerate(stages):
        cls = ("pass pa" if i == 1 else "pass pz" if i == 4 else "pass" if 1 < i < 4 else "")
        pn = '<span class="pn">Stages 2 to 5 are one pass over the lines: one pass, four counts.</span>' if i == 4 else ""
        det = s["detail"] if s["state"] == "done" and s["detail"] else DESC[i]
        d = "".join(f'<span{" class=al" if OFFRANGE.match(p) else ""}>{E(p)}</span>' for p in det.split(" · ") if p)
        cur = ' aria-current="step"' if s["state"] == "running" else ""
        out.append(f'<li class="{cls}" data-k="{E(s["key"])}" data-st="{E(s["state"])}"{cur}><span class="g" aria-hidden="true">'
                   f'</span><span class="l">{E(s["label"])}<span class="sr">, {WORD.get(s["state"], "")}</span>'
                   f'<span class="rw" aria-hidden="true">running</span></span><span class="d">{d}</span>{pn}</li>')
    return '<ol class="rail" id="rail">' + "".join(out) + "</ol>"


# CSV document glyph shared by the drop zone and the run file card: page + folded corner + 3x2 table hint, 48 units.
DOC = ('<path class="bg" d="M10 4h20l8 8v32H10z"/><path d="M30 4v8h8"/>'
       '<rect x="16" y="16" width="16" height="18" rx="1.5"/><path d="M16 22h16M16 28h16M22 16v18"/>')
SHELL = Template("""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Carbon Copilot</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>$fonts
$css
.intro{grid-column:1/8;display:flex;flex-direction:column}.intro h1{font-size:clamp(40px,4.2vw,56px);min-height:2em}
.side{grid-column:8/13;display:flex;flex-direction:column;padding:24px;border-radius:20px;background:var(--well)}
.side h2{margin-bottom:20px}main:has(.box[data-s=empty]) .rail [data-st=pending] .l{color:var(--ink)}
.box{position:relative;flex:1;min-height:320px;border:1px dashed var(--faint);border-radius:20px;background:var(--well);
 transition:background-color var(--t-slow) var(--ease),border-color var(--t-fast),transform var(--t-fast) var(--ease),
 box-shadow var(--t-fast) var(--ease)}
.box::before{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;transition:opacity var(--t-slow);
 background:repeating-linear-gradient(180deg,transparent 0 31px,color-mix(in srgb,var(--rule) 45%,transparent) 31px 32px);
 -webkit-mask-image:radial-gradient(ellipse 50% 46% at 50% 50%,transparent 62%,#000 100%);
 mask-image:radial-gradient(ellipse 50% 46% at 50% 50%,transparent 62%,#000 100%)}
.box:focus-within{outline:2px solid var(--acc);outline-offset:2px}.box input{position:absolute;opacity:0;width:1px;height:1px}
.box.over{background:var(--sheet);border:1px solid var(--ink);transform:translateY(-2px);box-shadow:var(--lift)}
.box[data-s=run],.box[data-s=stop]{background:var(--sheet);border:1px solid var(--rule);box-shadow:var(--lift)}
.box[data-s=run]{view-transition-name:run}
.box[data-s=err]{background:var(--sheet)}
.box[data-s=run]::before,.box[data-s=stop]::before,.box[data-s=err]::before,.box.over::before{opacity:0}
.pane{position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;padding:32px;
 text-align:center;opacity:0;visibility:hidden;transform:translateY(8px);
 transition:opacity 150ms,transform 150ms,visibility 0s linear 150ms}
.box[data-s=empty] .p-empty,.box[data-s=check] .p-check,.box[data-s=run] .p-run,.box[data-s=err] .p-err,.box[data-s=stop] .p-stop{
 opacity:1;visibility:visible;transform:none;transition:opacity var(--t-slow) var(--ease),transform var(--t-slow) var(--ease),visibility 0s}
.p-empty{cursor:pointer}.p-empty b{font-size:22px;line-height:1.25;font-weight:700;letter-spacing:-.01em;max-width:420px}
.p-empty b u{text-decoration-thickness:1px;text-underline-offset:4px}.p-empty .rel,.over .p-empty .idle{display:none}
.over .p-empty .rel{display:block}.p-empty span{font-size:13px;color:var(--mute)}
.di{display:block;width:56px;height:56px;margin-bottom:4px;color:var(--ink);overflow:visible}
.di .bg{fill:var(--well);transition:fill var(--t-slow) var(--ease)}.over .di .bg{fill:var(--sheet)}
.di .bd{fill:currentColor;stroke:var(--well);stroke-width:3;paint-order:stroke;transition:stroke var(--t-slow) var(--ease)}.over .di .bd{stroke:var(--sheet)}
.di .ar{stroke:var(--paper)}.di .up{transition:transform 200ms var(--ease)}.over .di .up{transform:translateY(-3px)}
.p-check .g{width:28px;height:28px}.p-check p{font-size:18px}
.p-run{align-items:stretch;justify-content:space-between;text-align:left;padding:28px 32px}
.fc{display:flex;align-items:center;gap:10px}.fd{flex:none;width:32px;height:32px;margin:-2px 0 0 -6px;color:var(--ink)}
.fc .mono{font-size:15px;color:var(--ink)}.fc .small{margin-left:auto}
.timer{font-size:40px;line-height:1;font-weight:700;letter-spacing:-.02em}.timer small{font-size:18px;font-weight:600;color:var(--mute);margin-left:6px}
#cstage{margin-top:12px}.segs{display:flex;gap:4px;margin-top:8px}.segs i{flex:1;height:8px;border-radius:4px;background:var(--bar3);overflow:hidden;position:relative}
.lg{margin:0;font-size:14px;line-height:20px}.lg div{display:flex;justify-content:space-between;gap:16px;padding:5px 0;
 border-bottom:1px solid var(--rule)}.lg dt{color:var(--mute)}.lg dd{margin:0;color:var(--faint)}
.lg dd[data-on]{color:var(--ink);font-weight:600;animation:up var(--t-base) var(--ease) both}
.lg .sum{border-top:3px double var(--ink);border-bottom:0;margin-top:2px}.lg .sum dt{color:var(--ink);font-weight:600}
.segs i::after{content:"";position:absolute;inset:0;background:var(--ink);transform:scaleX(0);transform-origin:left;
 transition:transform var(--t-base) var(--ease)}.segs i.on::after{transform:none}
.p-err .g,.p-stop .g{width:24px;height:24px}.p-err .g::before,.p-err .g::after,.p-stop .g::before,.p-stop .g::after{left:11px;top:2px;height:20px}
.p-err h2,.p-stop h2{margin:0}.p-err p,.p-stop p{max-width:520px;overflow-wrap:anywhere}
.p-err .row,.p-stop .row{display:flex;gap:16px;align-items:center;margin-top:8px}
pre.cmd{margin:0;padding:10px 14px;border-radius:8px;background:var(--well);font:13px/1.4 var(--mono);
 user-select:all;white-space:pre-wrap}
.after{margin-top:8px;display:flex;flex-wrap:wrap;align-items:center;justify-content:space-between;column-gap:24px}
.lnk{appearance:none;display:inline-flex;align-items:center;gap:6px;min-height:44px;padding:0;border:0;background:none;
 font:inherit;color:var(--mute);cursor:pointer;border-radius:6px}.lnk svg{flex:none;transition:transform var(--t-fast) var(--ease)}
#demo u{color:var(--ink);font-weight:600;text-decoration-thickness:1px;text-underline-offset:3px}#demo:hover svg{transform:translateX(2px)}
#colsb[aria-expanded=true] svg{transform:rotate(180deg)}#colsb:hover{color:var(--ink)}
.cl{flex-basis:100%;list-style:none;margin:0;padding:4px 0 0;display:grid;grid-template-columns:repeat(3,minmax(0,1fr));
 grid-template-rows:repeat(4,auto);grid-auto-flow:column;gap:4px 24px;font:13px/1.5 var(--mono);color:var(--mute)}.cl[hidden]{display:none}
body[data-s=check] .after,body[data-s=run] .after,body[data-s=stop] .after{display:none}
.foot{margin-top:auto;padding-top:24px}
#frame{position:fixed;inset:12px;border:2px solid var(--ink);border-radius:20px;pointer-events:none;opacity:0;z-index:9;
 transition:opacity 100ms}body.drag #frame{opacity:1;transition-duration:var(--t-fast)}
</style></head><body data-s="empty">$header<main class="grid">
<section class="intro"><h1><span class="zn" aria-hidden="true">01</span><span id="h1">Drop a ledger.<br>Get the carbon inventory.</span></h1>
<div class="box" id="box" data-s="empty" aria-busy="false"><input type="file" id="file" accept=".csv,text/csv">
<label class="pane p-empty" for="file"><svg class="di" viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.5"
 stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">$doc
<g class="up"><circle class="bd" cx="38" cy="40" r="7"/><path class="ar" d="M38 43.5v-7m-3 3 3-3 3 3"/></g></svg>
<b class="idle">Drop a ledger CSV here or <u>choose a file</u></b><b class="rel">Release to start</b>
<span>CSV, UTF-8 or cp1251, comma or semicolon, up to 5 MB</span></label>
<div class="pane p-check" aria-hidden="true"><span class="g run"></span><p class="mono" id="kname"></p><p>Checking the file</p></div>
<div class="pane p-run"><div class="fc"><svg class="fd" viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="2.25" stroke-linecap="round"
 stroke-linejoin="round" aria-hidden="true">$doc</svg><span class="mono" id="cname"></span>
<span class="small" id="clines"></span></div>$runlog<div><p class="timer"><span id="tn">0.0</span><small>s elapsed</small></p>
<p id="cstage">Starting</p><div class="segs" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div></div></div>
<div class="pane p-err" role="alert"><span class="g x" aria-hidden="true"></span><h2>Not processed</h2><p id="em"></p>
<p class="row"><label class="btn" for="file">Drop another file</label><a id="ebase" href="/base/dashboard.html" target="_blank" hidden>
Open the base dashboard</a></p></div>
<div class="pane p-stop" role="alert"><span class="g x" aria-hidden="true"></span><p class="mono" id="sname"></p><p id="sm"></p>
<pre class="cmd" id="scmd"></pre><p class="row"><label class="btn" for="file">Drop another file</label></p></div></div>
<div class="after"><button class="lnk" id="demo" type="button">or<u>try the demo file</u>(12 lines)<svg width="16"
 height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"
 aria-hidden="true"><path d="M3 8h10M9 4l4 4-4 4"/></svg></button>
<button class="lnk" id="colsb" type="button" aria-expanded="false" aria-controls="cols">Expected columns ($ncols)<svg width="16"
 height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"
 aria-hidden="true"><path d="m4 6 4 4 4-4"/></svg></button><ul class="cl" id="cols" hidden translate="no">$cols</ul></div></section>
<aside class="side"><h2><span class="zn in" aria-hidden="true">02</span>What happens to the file</h2>$rail
<p class="small foot">Runs on this laptop. No upload to any server. The raw file is deleted after masking.</p></aside>
</main><div id="frame" aria-hidden="true"></div><p class="sr" id="live" aria-live="polite"></p><script>
var $$=function(i){return document.getElementById(i)},box=$$("box"),inp=$$("file"),h1=$$("h1"),H1=[].slice.call(h1.childNodes),
 rows=[].slice.call(document.querySelectorAll("#rail li")),segs=[].slice.call(document.querySelectorAll(".segs i")),
 WORD={pending:"pending",running:"running",done:"done",error:"stopped"},T={b:0,a:0,on:false},dc=0;
rows.forEach(function(li){li._d=li.querySelector(".d").textContent;li._det=""});
function say(t){$$("live").textContent=t}
function title(t){document.title=t?t+", Carbon Copilot":"Carbon Copilot"}
function setS(s){box.dataset.s=s;document.body.dataset.s=s;var busy=s==="check"||s==="run";
 box.setAttribute("aria-busy",busy?"true":"false");inp.disabled=busy}
function busy(){return box.dataset.s==="check"||box.dataset.s==="run"}
function parts(el,txt){el.textContent="";txt.split(" \\u00b7 ").forEach(function(p){if(!p)return;
 var s=document.createElement("span");s.textContent=p;if(/^off-range lines [1-9]/.test(p))s.className="al";el.appendChild(s)})}
function setRow(li,s){if(li.dataset.st===s.state&&li._det===s.detail)return;li.dataset.st=s.state;li._det=s.detail;
 li.querySelector(".l .sr").textContent=", "+(WORD[s.state]||"");
 if(s.state==="running")li.setAttribute("aria-current","step");else li.removeAttribute("aria-current");
 var d=li.querySelector(".d");if(s.state==="done"&&s.detail)parts(d,s.detail);else if(s.state==="pending")d.textContent=li._d}
var lg=[].slice.call(document.querySelectorAll("#lg dd"));
function setLg(dd,v){var s=v==null?"\\u2014":String(v);if(dd.textContent===s)return;dd.textContent=s;
 if(v==null)dd.removeAttribute("data-on");else dd.setAttribute("data-on","")}
function resetRows(){rows.forEach(function(li,i){setRow(li,{state:"pending",detail:""})});segs.forEach(function(g){g.classList.remove("on")});
 lg.forEach(function(dd){setLg(dd,null)})}
function tick(){if(!T.on)return;$$("tn").textContent=(T.b+(performance.now()-T.a)/1000).toFixed(1);requestAnimationFrame(tick)}
function fail(m,base){T.on=false;setS("err");h1.replaceChildren.apply(h1,H1);$$("em").textContent=m||"";var b=$$("ebase");b.hidden=!base;
 if(base)b.href=base;title("Not processed")}
function stop(m,name){T.on=false;setS("stop");h1.textContent="Processing stopped.";$$("sname").textContent=name||"";var k=m.indexOf("Fallback: ");
 $$("sm").textContent=k<0?m:m.slice(0,k).trim();$$("scmd").textContent=k<0?"python run.py --new file.csv":m.slice(k+10);
 title("Stopped");history.replaceState(null,"","/")}
function render(j){var n=0,cur=-1;j.stages.forEach(function(s,i){var was=rows[i].dataset.st;setRow(rows[i],s);
 if(s.state==="done"){n++;if(was!=="done")cur=i}});
 if(cur>=0){var s=j.stages[cur];say("Stage "+(cur+1)+" of 7 done: "+s.label+(s.detail?", "+s.detail.split(" \\u00b7 ").join(", "):""))}
 segs.forEach(function(g,i){g.classList.toggle("on",i<n)});var at=Math.min(n+1,7),C=j.counts||{};
 lg.forEach(function(dd){var k=dd.dataset.c;setLg(dd,k==="total"?(j.summary?j.summary.total.toFixed(1):null):C[k])});
 $$("cname").textContent=j.name;$$("clines").textContent=j.lines+" lines";
 $$("cstage").textContent=n===7?"7 of 7 stages done":"Stage "+at+" of 7: "+j.stages[at-1].label;
 T.b=j.elapsed_s;T.a=performance.now();title(n===7?"Done":"Processing "+at+" of 7")}
function poll(id){fetch("/api/jobs/"+id).then(function(r){if(!r.ok)throw 0;return r.json()}).then(function(j){
 if(box.dataset.s!=="run"){setS("run");h1.textContent="Processing the ledger."}render(j);
 if(j.state==="done"){T.on=false;$$("tn").textContent=j.elapsed_s.toFixed(1);setTimeout(function(){location.href="/jobs/"+id},600)}
 else if(j.state==="error"){$$("tn").textContent=j.elapsed_s.toFixed(1);stop(j.error,j.name)}
 else{if(!T.on){T.on=true;requestAnimationFrame(tick)}setTimeout(function(){poll(id)},300)}})
 .catch(function(){fail("This job is no longer available.");history.replaceState(null,"","/")})}
function start(url,opt,name){if(busy())return;resetRows();$$("kname").textContent=name;setS("check");title("Checking the file");
 say("Checking "+name);fetch(url,opt).then(function(r){return r.json().then(function(j){return [r.ok,j]})}).then(function(a){
 if(!a[0]){fail(a[1].error,a[1].base);return}history.pushState(null,"","/jobs/"+a[1].id);poll(a[1].id)})
 .catch(function(){fail("The local server did not answer. Is python app.py still running?")})}
function send(f){if(f)start("/api/jobs",{method:"POST",headers:{"X-Filename":encodeURIComponent(f.name)},body:f},f.name)}
function off(){dc=0;document.body.classList.remove("drag");box.classList.remove("over")}
window.addEventListener("dragenter",function(e){e.preventDefault();if(busy())return;if(++dc===1){if(box.dataset.s!=="empty")setS("empty");
 document.body.classList.add("drag");box.classList.add("over")}});
window.addEventListener("dragleave",function(){if(dc&&--dc===0)off()});
window.addEventListener("dragover",function(e){e.preventDefault()});
window.addEventListener("drop",function(e){e.preventDefault();off();if(!busy())send(e.dataTransfer.files[0])});
inp.addEventListener("change",function(){send(this.files[0]);this.value=""});
$$("colsb").addEventListener("click",function(){var o=this.getAttribute("aria-expanded")!=="true";this.setAttribute("aria-expanded",o);$$("cols").hidden=!o});
$$("demo").addEventListener("click",function(){start("/api/jobs?demo=1",{method:"POST"},"new_lines.csv")});
var m=location.pathname.match(/^\\/jobs\\/([A-Za-z0-9_-]+)$$/);if(m){setS("run");h1.textContent="Processing the ledger.";poll(m[1])}
</script></body></html>""")

RESULTS = Template("""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>$title</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>$fonts
$css
.rise{animation:up8 var(--t-slow) var(--ease) both}.d1{animation-delay:60ms}.d2{animation-delay:120ms}.d3{animation-delay:180ms}
html{scroll-padding-top:88px}
.top{grid-column:1/-1;position:sticky;top:0;z-index:5;display:grid;margin:-8px 0;padding:8px 0;background:var(--paper)}
.top>*{grid-area:1/1}.top:has(.receipt[open]){position:static}@media(max-width:959px){.top{position:static}}
.receipt{background:var(--well);border-radius:12px}
.receipt summary{display:flex;align-items:center;gap:16px;min-height:56px;padding:0 var(--navw,408px) 0 20px;cursor:pointer;list-style:none}
.receipt summary::-webkit-details-marker{display:none}
.tr{position:relative;z-index:1;justify-self:end;align-self:start;display:flex;align-items:center;gap:20px;height:56px;margin-right:10px;cursor:default}
/* zone indicator: a read-out, not tabs. Dots + plain text, no underline, catches clicks so they never reach the summary below */
.zi{display:flex;align-items:center;gap:12px;height:24px;padding-left:20px;border-left:1px solid var(--rule);font-size:13px;color:var(--mute);user-select:none}
.zi ol{display:flex;align-items:center;gap:4px;margin:0;padding:0;list-style:none}
.zi li{width:6px;height:6px;border-radius:3px;background:var(--faint);transition:width var(--t-base) var(--ease),background-color var(--t-base)}
.zi li.on{width:16px;background:var(--ink)}.zi .zl{min-width:176px;white-space:nowrap}.zi .zl b{color:var(--ink);font-weight:600}
.zi .zl .mono{font-size:12px;color:var(--ink);margin-right:4px}
.nf{height:36px;padding:0 14px;font-size:14px}
/* --navw = .tr width + its 10px margin + 24px air, so the Stages button never sits under the indicator */
@media(max-width:1199px){.zi .pre{display:none}.zi .zl{min-width:128px}.tr{gap:16px}.zi{padding-left:16px}
 .rs .mono{max-width:18ch}.receipt summary{--navw:352px}}
@media(max-width:959px){.zi{display:none}.receipt summary{--navw:150px}}
.zone.pre{opacity:0;transform:translateY(10px)}.zone{transition:opacity var(--t-slow) var(--ease),transform var(--t-slow) var(--ease)}
.zone.pre.in{opacity:1;transform:none}.receipt{view-transition-name:run}
.ticks{display:flex;gap:4px}.ticks .g{width:14px;height:14px}.ticks .g::after{left:4.5px;top:2px;width:3px;height:6px;border-width:0 1.5px 1.5px 0}
.rs{display:flex;align-items:center;gap:16px;flex-wrap:wrap}.rs>span+span{border-left:1px solid var(--rule);padding-left:16px}
.rs .mono{font-size:14px;max-width:24ch;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}/* disclosure: a bordered button with a chevron, its own shape, a divider away from the indicator */
.receipt .tog{margin-left:auto;display:inline-flex;align-items:center;gap:6px;height:36px;padding:0 10px 0 14px;border:1px solid var(--rule);
 border-radius:8px;background:var(--sheet);font-size:14px;font-weight:600;transition:border-color var(--t-fast)}
.receipt summary:hover .tog{border-color:var(--ink)}.receipt .tog svg{transition:transform var(--t-fast) var(--ease)}
.receipt[open] .tog svg{transform:rotate(180deg)}
.receipt .body{padding:8px 32px 16px}
.receipt .rail{max-width:720px}
.zone{grid-column:1/-1;row-gap:32px}.zh{grid-column:1/-1;margin:0 0 -16px}.z03{margin-top:16px}
.heroc{grid-column:1/9}.heroc h1{font-size:18px;line-height:1.45;font-weight:400;letter-spacing:0;color:var(--mute);margin:0 0 12px}
.num{font-size:clamp(88px,10vw,128px);line-height:.9;font-weight:800;letter-spacing:-.04em;white-space:nowrap;font-variant-numeric:proportional-nums}
.num small{font-size:.4em;font-weight:700;letter-spacing:-.01em;color:var(--mute)}.num .u{margin-left:.35em}.num .fr{margin-left:.06em}
.heroc .sub{margin-top:20px;color:var(--mute)}
.led{grid-column:9/13}.led h2{font-size:15px;line-height:1.4;font-weight:600;letter-spacing:0;margin:0 0 4px}
.strip{grid-column:1/-1;padding:8px;border:1px solid var(--rule);border-radius:20px;background:var(--sheet)}
.out{grid-column:span 3;display:grid;grid-template-columns:auto minmax(0,1fr);column-gap:12px;row-gap:2px;align-items:center;
 min-height:56px;padding:10px 12px;border-radius:12px;text-decoration:none;transition:background-color var(--t-fast),transform 100ms}
.out:hover{background:var(--well)}.out:active{transform:scale(.98)}.out .chip{grid-row:1/3;min-width:44px;text-align:center}
.out b{font-weight:600}.out .small{grid-column:2}
.out.p{background:var(--ink);color:var(--paper)}.out.p .chip{background:transparent;color:var(--paper);border-color:color-mix(in srgb,var(--paper) 50%,transparent)}
.out.p .small{color:var(--on-ink)}.out.p:hover{background:var(--ink-2)}.nopdf{grid-column:span 3;align-self:center;padding:10px 12px}
.zh.vh{display:flex;align-items:baseline;justify-content:space-between}.vh .new{font-weight:600}
.scope{grid-column:1/-1;row-gap:12px}.bar{grid-column:1/-1;display:flex;gap:2px;height:16px;border-radius:8px;overflow:hidden}
.bar i{display:block;min-width:4px;transform-origin:left;animation:grow var(--t-slow) var(--ease) both}
.bar i:nth-child(2){animation-delay:60ms}.bar i:nth-child(3){animation-delay:120ms}
.s1{background:var(--ink)}.s2{background:var(--bar2)}.s3{background:var(--bar3)}
.slab{grid-column:span 4}
.slab p:first-child{font-size:13px;color:var(--mute);display:flex;align-items:center;gap:8px}
.slab p:first-child i{width:10px;height:10px;border-radius:3px}
.slab .t{font-size:24px;font-weight:700;letter-spacing:-.01em;line-height:1.3}.slab .small{margin-left:8px}
.ledger{width:100%;border-collapse:collapse}.ledger td{padding:12px 0;border-bottom:1px solid var(--rule)}
.ledger td.n{text-align:right;font-weight:600}.ledger tr.sub td{padding:8px 0 8px 24px;color:var(--mute);font-size:13px}
.ledger tr.sub td.n{font-weight:400}.ledger tr.sum td{border-top:3px double var(--ink);border-bottom:0;font-weight:700}
.ledger .ex td{border-bottom:0}.ledger tr.sub:not(.last) td{border-bottom:0}
.hp{grid-column:1/-1;row-gap:16px;padding:8px;border:1px solid var(--rule);border-radius:20px;background:var(--sheet)}
.ban{grid-column:1/-1;align-items:stretch;border-radius:12px}.ban>.rv{--a:var(--alert);--a-bg:var(--alert-bg);--a-line:var(--alert-line)}
.ban>.inj{--a:var(--caution);--a-bg:var(--caution-bg);--a-line:var(--caution-line)}
.ban>div{grid-column:1/9;display:flex;align-items:center;gap:12px;background:var(--a-bg);border:1px solid var(--a-line);border-radius:12px;padding:14px 16px}.ban>div+div{grid-column:9/13}.ban>.wide{grid-column:1/-1}
.qcol{grid-column:1/9;padding:0 8px 8px}.qcol.wide{grid-column:1/-1}.q .t{max-width:68ch}
.tiles{grid-column:9/13;display:flex;flex-direction:column;gap:8px}
.tiles.wide{grid-column:1/-1;display:grid;grid-template-columns:repeat(3,minmax(0,1fr))}
@media(max-width:959px){.tiles.wide{grid-template-columns:minmax(0,1fr)}}
.tile{background:var(--caution-bg);border-radius:12px;padding:16px;border:1px solid var(--caution-line)}
.tile blockquote{margin:6px 0 8px;max-width:40ch;overflow-wrap:anywhere}.tile blockquote::before{content:"\\201C"}
.tile blockquote::after{content:"\\201D"}.tile .o{font-size:13px;color:var(--mute)}.tile .o b{color:var(--ink);font-weight:600}
.ban .c{font-size:24px;line-height:1;font-weight:700;color:var(--a)}.ban .g.w{background:var(--a);color:var(--a-bg);
 font:700 13px/18px Onest,system-ui,sans-serif;text-align:center}
.q{list-style:none;margin:8px 0 0;padding:0}.q li{display:grid;grid-template-columns:72px minmax(0,1fr) auto;gap:16px;
 padding:12px 0 12px 12px;border-bottom:1px solid var(--rule);align-items:baseline;border-left:3px solid var(--alert-line)}
.q .chip{background:var(--alert-bg);border-color:var(--alert-line)}.q .chip.inj{background:var(--caution-bg);border-color:var(--caution-line)}
.q .t{overflow-wrap:anywhere}.q .fl{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}.q .n{text-align:right;font-weight:600;white-space:nowrap}
.more{display:inline-block;margin-top:14px;font-weight:600}.quiet{grid-column:1/-1;color:var(--mute)}
.vh h2{margin:0}.dh{display:flex;align-items:center;gap:16px;flex-wrap:wrap;margin:0 0 12px}
.seg{display:inline-flex;gap:4px;padding:4px;background:var(--paper);border-radius:12px}
.seg a{display:flex;align-items:center;height:36px;padding:0 16px;border-radius:8px;color:var(--mute);font-weight:600;text-decoration:none;
 transition:background-color var(--t-fast),color var(--t-fast),transform 100ms}.seg a:active{transform:scale(.98)}
.seg a{border:1px solid transparent}.seg a[aria-current=true]{background:var(--sheet);color:var(--ink);border-color:var(--rule)}
.desk{grid-column:1/-1;margin-top:-16px;padding:12px;background:var(--well);border-radius:20px}
.desk iframe{display:block;width:100%;height:760px;border:0;border-radius:12px;background:var(--sheet);transition:opacity var(--t-slow) var(--ease)}
.desk iframe.fade{opacity:0;transition-duration:100ms}
</style></head><body>$header<main class="grid">
<div class="top"><details class="receipt rise"><summary><span class="zn in" aria-hidden="true">02</span>
<span class="ticks" aria-hidden="true">$ticks</span><span class="rs"><span class="mono" dir="auto" translate="no">$name</span>
<span>$lines lines<span class="sr">, 7 of 7 stages</span></span><span>$elapsed s</span></span>
<span class="tog">Stages<svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"
 stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m4 6 4 4 4-4"/></svg></span></summary><div class="body">$rail</div></details>
<div class="tr"><div class="zi" aria-hidden="true"><ol><li data-z="result" data-n="03" data-l="Result" class="on"></li>
<li data-z="review" data-n="04" data-l="Needs a human"></li><li data-z="outputs" data-n="05" data-l="Outputs"></li></ol>
<span class="zl"><span class="pre">You're in: </span><span class="mono" id="zn">03</span><b id="zt">Result</b></span></div>
<a class="btn nf" href="/"><svg width="16" height="16" viewBox="0 0 16 16" fill="none" stroke="currentColor" stroke-width="1.5"
 stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M8 10.5V2.5M4.75 5.75 8 2.5l3.25 3.25M2.5 10v2.5a1 1 0 0 0 1 1h9a1 1 0 0 0 1-1V10"/></svg>New file</a></div></div>
<section class="zone sg z03" id="result" aria-labelledby="h03"><div class="heroc rise d1">
<h1 id="h03"><span class="zn" aria-hidden="true">03</span>Emissions in the uploaded lines</h1>
<p class="num"><span class="sr">$total tCO2e</span><span id="cnt" aria-hidden="true" data-v="$intpart"><span>$intpart</span><small class="fr">$frac</small><small class="u">tCO2e</small></span></p>
<p class="sub">Scope 1, 2 and 3 for $counted counted lines of $lines.</p></div>
<div class="led rise d2"><h2>The lines</h2><table class="ledger">$ledger</table></div>
<div class="scope sg rise d3" role="group" aria-label="Emissions by scope">$bar</div></section>
<section class="zone sg z" id="review" aria-labelledby="h04"><h2 class="zh" id="h04"><span class="zn" aria-hidden="true">04</span>Needs a human</h2>
$needs</section>
<section class="zone sg z" id="outputs" aria-labelledby="h05"><div class="zh vh"><h2 id="h05"><span class="zn" aria-hidden="true">05</span>Outputs</h2>
<a class="new" href="/">New file</a></div><div class="strip sg">
<a class="out p" href="$base/dashboard.html#split=new" target="_blank"><span class="chip">HTML</span><b>Dashboard</b><span class="small">new lines</span></a>
<a class="out" href="$base/bank_summary.html" target="_blank"><span class="chip">HTML</span><b>Bank one-pager</b><span class="small">whole company</span></a>$pdf
<a class="out" href="$base/audit.csv"><span class="chip">CSV</span><b>Audit CSV</b><span class="small">$lines rows</span></a></div>
<div class="desk"><div class="dh"><nav class="seg" aria-label="Viewer">
<a href="$base/dashboard.html#split=new" target="viewer" aria-current="true">Dashboard</a>
<a href="$base/bank_summary.html" target="viewer" aria-current="false">Bank one-pager</a></nav>
<p class="small">Bank pages: whole company inventory with your lines added.</p></div>
<iframe id="viewer" name="viewer" src="$base/dashboard.html#split=new" title="Outputs viewer"></iframe></div></section>
</main><script>
(function(){var rm=matchMedia("(prefers-reduced-motion: reduce)"),c=document.getElementById("cnt"),fr=document.getElementById("viewer");
 var v=+c.dataset.v.replace(/,/g,""),f=c.firstChild,t0=performance.now();
 if(!rm.matches&&isFinite(v))(function s(t){var k=rm.matches?1:Math.min(1,(t-t0)/400),e=1-Math.pow(1-k,3);
  f.textContent=(k<1?Math.round(v*e):v).toLocaleString("en-US");if(k<1)requestAnimationFrame(s)})(t0);
 var segs=[].slice.call(document.querySelectorAll(".seg a"));segs.forEach(function(a){a.addEventListener("click",function(){
  segs.forEach(function(b){b.setAttribute("aria-current",b===a?"true":"false")});if(!rm.matches)fr.classList.add("fade")})});
 fr.addEventListener("load",function(){fr.classList.remove("fade")});
 if(!("IntersectionObserver" in window))return;
 var zs=[].slice.call(document.querySelectorAll(".zone")),zi=[].slice.call(document.querySelectorAll(".zi li"));
 var zn=document.getElementById("zn"),zt=document.getElementById("zt");
 var at=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting)zi.forEach(function(l){
  var on=l.dataset.z===e.target.id;l.classList.toggle("on",on);if(on){zn.textContent=l.dataset.n;zt.textContent=l.dataset.l}})})},
  {rootMargin:"-30% 0px -60% 0px"});
 zs.forEach(function(z){at.observe(z)});
 if(rm.matches)return;                                  /* zones below the fold rise once on first sight */
 var io=new IntersectionObserver(function(es){es.forEach(function(e){if(e.isIntersecting){e.target.classList.add("in");io.unobserve(e.target)}})},
  {rootMargin:"0px 0px -6% 0px"});
 zs.forEach(function(z){if(z.getBoundingClientRect().top>innerHeight){z.classList.add("pre");io.observe(z)}})})();
</script></body></html>""")


def results_page(job):
    S, b = job["summary"], f"/jobs/{job['id']}"
    tot = S["total"]
    pct = lambda x: f"{100 * x / tot:.1f}%" if tot else "0%"
    whole, _, frac = f"{tot:,.1f}".partition(".")
    pos = {s: max(0.0, S["by_scope"][s]) for s in "123"}
    span = sum(pos.values()) or 1.0
    bar = ('<div class="bar" aria-hidden="true">' + "".join(f'<i class="s{s}" style="flex:{pos[s] / span:.4f} 1 0"></i>' for s in "123")
           + '</div>' + "".join(
               f'<div class="slab"><p><i class="s{s}" aria-hidden="true"></i>Scope {s}</p>'
               f'<p><span class="t">{S["by_scope"][s]:,.1f} t</span><span class="small">{pct(S["by_scope"][s])}</span>'
               f'<span class="sr"> of the total</span></p></div>' for s in "123"))
    exc = S["duplicates"] + S["nonpurchase"]
    ledger = (f'<tr><td>Counted</td><td class="n">{S["counted"]}</td></tr>'
              f'<tr class="ex"><td>Excluded</td><td class="n">{exc}</td></tr>'
              f'<tr class="sub"><td>Duplicate document</td><td class="n">{S["duplicates"]}</td></tr>'
              f'<tr class="sub last"><td>Not a purchase</td><td class="n">{S["nonpurchase"]}</td></tr>'
              f'<tr class="sum"><td>Lines in the file</td><td class="n">{S["lines"]}</td></tr>')
    top, inj, rv = S.get("review_top", []), S["injection"], S["review"]
    both = " wide" if not (rv and inj) else ""
    counts = (f'<div class="rv{both}"><span class="g w" aria-hidden="true">!</span><span class="c">{rv}</span>'
              f'<p>lines to review. Flags never change a result; a person confirms each line.</p></div>' if rv else "") + (
              f'<div class="inj{both}"><span class="g w" aria-hidden="true">!</span><span class="c">{len(inj)}</span>'
              f'<p>{"line" if len(inj) == 1 else "lines"} carried instruction-like text. It was ignored.</p></div>' if inj else "")
    queue = (f'<div class="qcol{" wide" if not inj else ""}"><ul class="q">' + "".join(
                 f'<li><span class="mono" translate="no">{E(r["line_id"])}</span><div><p class="t" dir="auto" translate="no">'
                 f'{E(r["description"])}</p><p class="fl">' + "".join(f'<span class="chip{" inj" if f == "injection" else ""}">{E(f)}</span>' for f in r["flags"])
                 + f'</p></div><span class="n">{r["tco2e"]:,.2f} t</span></li>' for r in top)
             + f'</ul><a class="more" href="{b}/dashboard.html#split=new&amp;tab=queue" target="_blank">'
               f'Open all {rv} in the dashboard</a></div>') if rv else ""
    tiles = (f'<div class="tiles{" wide" if not rv else ""}">' + "".join(
                 f'<article class="tile"><p class="mono" translate="no">{E(r["line_id"])}</p><blockquote class="t" dir="auto" '
                 f'translate="no">{E(r["description"])}</blockquote><p class="o">Ignored. Counted as <b class="mono">'
                 f'{E(r["category"])}</b>, {r["tco2e"]:,.2f} t</p></article>' for r in inj) + "</div>") if inj else ""
    needs = (f'<div class="hp sg"><div class="ban sg" role="note">{counts}</div>{queue}{tiles}</div>' if rv or inj
             else '<p class="quiet">No lines to review.</p>')
    pdf = (f'<a class="out" href="{b}/bank_summary.pdf"><span class="chip">PDF</span><b>Bank PDF</b><span class="small">1 page</span></a>'
           if S["pdf"] else '<p class="small nopdf">Bank PDF needs Edge or Chrome; open the one-pager and print it.</p>')
    ticks = "".join(f'<span class="g ok" style="animation-delay:{60 + 40 * i}ms"></span>' for i in range(len(job["stages"])))
    return RESULTS.substitute(fonts=FONTS, css=CSS, header=HEADER, base=b, total=f"{tot:,.1f}", intpart=whole, frac="." + frac,
                              title=E(f"{tot:,.1f} tCO2e, Carbon Copilot"), name=E(job["name"]), lines=S["lines"],
                              counted=S["counted"], elapsed=f"{job['elapsed']:.1f}", ticks=ticks, rail=rail(job["stages"]),
                              bar=bar, ledger=ledger, needs=needs, pdf=pdf)


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
                                            lines=job["n"], elapsed_s=round(el, 2), summary=job.get("summary"),
                                            counts=job["counts"]))
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


# Processing card ledger: real counts from job["counts"] (and summary.total at the end), filled in place as stages finish.
RUNLOG = [("cp", "Counterparties masked"), ("persons", "Person names masked"), ("excluded", "Set aside: duplicate or not a purchase"),
          ("rule", "Decided by a rule"), ("clf", "Decided by the classifier"), ("offrange", "Off-range lines"),
          ("total", "Emissions, tCO2e")]
SHELL_HTML = SHELL.substitute(doc=DOC, fonts=FONTS, css=CSS, header=HEADER, cols="".join(f'<li>{c}</li>' for c in REQ), ncols=len(REQ),
                             runlog='<dl class="lg" id="lg">' + "".join(
                                 f'<div{" class=sum" if k == "total" else ""}><dt>{l}</dt><dd data-c="{k}">—</dd></div>'
                                 for k, l in RUNLOG) + "</dl>",
                             rail=rail([dict(key=k, label=l, state="pending", detail="") for k, l in STAGES]))


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
