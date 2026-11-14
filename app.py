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
# "Run receipt" (APP_REDESIGN.md): one run, seven stages, the same box from drop to file card. Tokens in section 4.
CSS = """:root{--paper:#F7F5F0;--sheet:#FFFFFF;--well:#EFECE5;--ink:#15171A;--mute:#4A4D52;--faint:#8A8C90;--rule:#C9C6BE;
 --acc:#A3372B;--acc-wash:#F3E4DF;--t-fast:150ms;--t-base:250ms;--t-slow:400ms;--ease:cubic-bezier(.22,1,.36,1);
 --lift:0 1px 0 rgba(21,23,26,.04),0 12px 32px -16px rgba(21,23,26,.18);--mono:ui-monospace,"Cascadia Mono",Consolas,monospace}
*{box-sizing:border-box}html{background:var(--paper)}
body{margin:0;background:var(--paper);color:var(--ink);font:15px/1.45 Onest,system-ui,sans-serif;font-variant-numeric:tabular-nums;
 -webkit-font-smoothing:antialiased}
header{height:56px;border-bottom:1px solid var(--rule);display:flex;align-items:center;justify-content:space-between;
 padding:0 max(48px,calc((100% - 1184px)/2))}header b{font-size:16px;font-weight:700;letter-spacing:-.01em}
.pill{display:flex;align-items:center;gap:12px;height:32px;padding:0 14px;border:1px solid var(--rule);border-radius:999px;
 background:var(--sheet);font-size:13px;color:var(--mute)}.pill i{width:8px;height:8px;border-radius:50%;background:var(--ink)}
.pill span+span{border-left:1px solid var(--rule);padding-left:12px}
main{max-width:1280px;margin:0 auto;padding:40px 48px 96px}
.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:24px}
@media(max-width:959px){.grid{grid-template-columns:minmax(0,1fr)}.grid>*{grid-column:1/-1!important}}
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
.rail li{position:relative;display:grid;grid-template-columns:18px minmax(0,1fr);column-gap:14px;min-height:48px;padding:4px 24px 10px 0}
.rail li:not(:last-child)::before{content:"";position:absolute;left:8.5px;top:27px;bottom:-3px;width:1px;background:var(--rule)}
.rail .g{margin-top:2px}.rail .l{font-size:15px;line-height:1.45}.rail .rw{display:none}
.rail .d{grid-column:2;display:flex;flex-wrap:wrap;column-gap:16px;font-size:13px;line-height:1.4;color:var(--mute)}
.rail [data-st=pending] .l{color:var(--mute)}.rail [data-st=running] .l{font-weight:600}
.rail [data-st=done] .d,.rail [data-st=error] .d{animation:up var(--t-base) var(--ease) both}
.rail .pass::after{content:"";position:absolute;right:0;top:0;bottom:0;width:8px;border-right:1px solid var(--ink)}
.rail .pa::after{top:12px;border-top:1px solid var(--ink)}.rail .pz::after{bottom:12px;border-bottom:1px solid var(--ink)}
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
@media(prefers-reduced-motion:reduce){*,*::before,*::after{animation:none!important;transition-property:opacity!important;
 transition-duration:150ms!important}[data-st=running]>.g,.g.run{border:1.5px solid currentColor;
 background:linear-gradient(90deg,currentColor 50%,transparent 50%)}.rail [data-st=running] .rw{display:inline;margin-left:8px;
 font-size:13px;font-weight:400;color:var(--mute)}}
@media(forced-colors:active){[data-st=done]>.g,.g.ok{background:CanvasText}.rail .pass::after{border-color:CanvasText}}
"""
HEADER = ('<header><b>Carbon Copilot</b><p class="pill"><i aria-hidden="true"></i><span>Local</span><span>127.0.0.1</span>'
          '<span>no network</span></p></header>')
WORD = {"pending": "pending", "running": "running", "done": "done", "error": "stopped"}
DESC = ["Names replaced before anything else reads the file", "Repeated documents and non-purchase lines set aside",
        "Lexicon and account rules", "Classifier for wording the rules do not know", "Activity or spend factor",
        "Sums and formulas recomputed", "Dashboard, bank page, PDF"]


def rail(stages):
    """Seven rows keyed by stage; the client updates them in place. Detail parts were joined with ' · ' upstream."""
    out = []
    for i, s in enumerate(stages):
        cls = ("pass pa" if i == 1 else "pass pz" if i == 4 else "pass" if 1 < i < 4 else "")
        det = s["detail"] if s["state"] == "done" and s["detail"] else DESC[i]
        d = "".join(f"<span>{E(p)}</span>" for p in det.split(" · ") if p)
        cur = ' aria-current="step"' if s["state"] == "running" else ""
        out.append(f'<li class="{cls}" data-k="{E(s["key"])}" data-st="{E(s["state"])}"{cur}><span class="g" aria-hidden="true">'
                   f'</span><span class="l">{E(s["label"])}<span class="sr">, {WORD.get(s["state"], "")}</span>'
                   f'<span class="rw" aria-hidden="true">running</span></span><span class="d">{d}</span></li>')
    return '<ol class="rail" id="rail">' + "".join(out) + "</ol>"


SHELL = Template("""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>Carbon Copilot</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>$fonts
$css
.intro{grid-column:1/8}.side{grid-column:8/13;padding-top:6px}.side h2{margin-bottom:20px}
.box{position:relative;height:340px;border:1px dashed var(--faint);border-radius:20px;background:var(--well);
 transition:background-color var(--t-slow) var(--ease),border-color var(--t-fast),transform var(--t-fast) var(--ease),
 box-shadow var(--t-fast) var(--ease)}
.box::before{content:"";position:absolute;inset:0;border-radius:inherit;pointer-events:none;transition:opacity var(--t-slow);
 background:linear-gradient(90deg,transparent calc(100% - 129px),rgba(201,198,190,.7) calc(100% - 129px) calc(100% - 128px),
 transparent calc(100% - 128px) calc(100% - 65px),rgba(201,198,190,.7) calc(100% - 65px) calc(100% - 64px),transparent calc(100% - 64px)),
 repeating-linear-gradient(180deg,transparent 0 31px,rgba(201,198,190,.4) 31px 32px);
 -webkit-mask-image:radial-gradient(ellipse 50% 46% at 44% 50%,transparent 62%,#000 100%);
 mask-image:radial-gradient(ellipse 50% 46% at 44% 50%,transparent 62%,#000 100%)}
.box:focus-within{outline:2px solid var(--acc);outline-offset:2px}.box input{position:absolute;opacity:0;width:1px;height:1px}
.box.over{background:var(--sheet);border:1px solid var(--ink);transform:translateY(-2px);box-shadow:var(--lift)}
.box[data-s=run],.box[data-s=stop]{background:var(--sheet);border:1px solid var(--rule);box-shadow:var(--lift)}
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
.sheet{width:44px;height:54px;border:1.5px solid var(--ink);border-radius:6px;background:var(--sheet);margin-bottom:8px;
 background-image:repeating-linear-gradient(180deg,transparent 0 9px,var(--rule) 9px 10px);background-position:0 8px;
 background-size:100% calc(100% - 16px);background-repeat:no-repeat;transition:transform var(--t-fast) var(--ease)}
.over .sheet{transform:translateY(-4px)}
.p-check .g{width:28px;height:28px}.p-check p{font-size:18px}
.p-run{align-items:stretch;justify-content:space-between;text-align:left;padding:28px 32px}
.fc{display:flex;align-items:center;gap:12px}.fc .sheet{width:28px;height:34px;margin:0;border-radius:4px;background-size:100% 18px}
.fc .mono{font-size:15px;color:var(--ink)}.fc .small{margin-left:auto}
.timer{font-size:40px;line-height:1;font-weight:700;letter-spacing:-.02em}.timer small{font-size:18px;font-weight:600;color:var(--mute);margin-left:6px}
.segs{display:flex;gap:4px;margin-top:14px}.segs i{flex:1;height:8px;border-radius:4px;background:var(--well);overflow:hidden;position:relative}
.segs i::after{content:"";position:absolute;inset:0;background:var(--ink);transform:scaleX(0);transform-origin:left;
 transition:transform var(--t-base) var(--ease)}.segs i.on::after{transform:none}
.p-err .g,.p-stop .g{width:24px;height:24px}.p-err .g::before,.p-err .g::after,.p-stop .g::before,.p-stop .g::after{left:11px;top:2px;height:20px}
.p-err h2,.p-stop h2{margin:0}.p-err p,.p-stop p{max-width:520px;overflow-wrap:anywhere}
.p-err .row,.p-stop .row{display:flex;gap:16px;align-items:center;margin-top:8px}
pre.cmd{margin:0;padding:10px 14px;border:1px solid var(--rule);border-radius:8px;background:var(--well);font:13px/1.4 var(--mono);
 user-select:all;white-space:pre-wrap}
.after{margin-top:16px;display:flex;flex-direction:column;align-items:flex-start;gap:16px;transition:opacity var(--t-fast)}
.chips{display:flex;flex-wrap:wrap;gap:6px;align-items:center}.chips .small{margin-right:4px}
body[data-s=check] .after,body[data-s=run] .after,body[data-s=stop] .after{opacity:0;visibility:hidden}
.passnote{margin:4px 0 0 32px}.foot{margin-top:24px;padding-top:16px;border-top:1px solid var(--rule)}
#frame{position:fixed;inset:12px;border:2px solid var(--ink);border-radius:20px;pointer-events:none;opacity:0;z-index:9;
 transition:opacity 100ms}body.drag #frame{opacity:1;transition-duration:var(--t-fast)}
</style></head><body data-s="empty">$header<main class="grid">
<section class="intro"><h1 id="h1">Drop a ledger.<br>Get the carbon inventory.</h1>
<div class="box" id="box" data-s="empty" aria-busy="false"><input type="file" id="file" accept=".csv,text/csv">
<label class="pane p-empty" for="file"><span class="sheet" aria-hidden="true"></span>
<b class="idle">Drop a ledger CSV here or <u>choose a file</u></b><b class="rel">Release to start</b>
<span>CSV, UTF-8 or cp1251, comma or semicolon, up to 5 MB</span></label>
<div class="pane p-check" aria-hidden="true"><span class="g run"></span><p class="mono" id="kname"></p><p>Checking the file</p></div>
<div class="pane p-run"><div class="fc"><span class="sheet" aria-hidden="true"></span><span class="mono" id="cname"></span>
<span class="small" id="clines"></span></div><div><p class="timer"><span id="tn">0.0</span><small>s elapsed</small></p></div>
<div><p id="cstage">Starting</p><div class="segs" aria-hidden="true"><i></i><i></i><i></i><i></i><i></i><i></i><i></i></div></div></div>
<div class="pane p-err" role="alert"><span class="g x" aria-hidden="true"></span><h2>Not processed</h2><p id="em"></p>
<p class="row"><label class="btn" for="file">Drop another file</label><a id="ebase" href="/base/dashboard.html" target="_blank" hidden>
Open the base dashboard</a></p></div>
<div class="pane p-stop" role="alert"><span class="g x" aria-hidden="true"></span><p class="mono" id="sname"></p><p id="sm"></p>
<pre class="cmd" id="scmd"></pre><p class="row"><label class="btn" for="file">Drop another file</label></p></div></div>
<div class="after"><button class="btn" id="demo" type="button">Use the demo file (12 lines)</button>
<p class="chips"><span class="small">Expected columns</span>$cols</p></div></section>
<aside class="side"><h2>What happens to the file</h2>$rail
<p class="small passnote">Stages 2 to 5 are one pass over the lines: one pass, four counts.</p>
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
 var s=document.createElement("span");s.textContent=p;el.appendChild(s)})}
function setRow(li,s){if(li.dataset.st===s.state&&li._det===s.detail)return;li.dataset.st=s.state;li._det=s.detail;
 li.querySelector(".l .sr").textContent=", "+(WORD[s.state]||"");
 if(s.state==="running")li.setAttribute("aria-current","step");else li.removeAttribute("aria-current");
 var d=li.querySelector(".d");if(s.state==="done"&&s.detail)parts(d,s.detail);else if(s.state==="pending")d.textContent=li._d}
function resetRows(){rows.forEach(function(li,i){setRow(li,{state:"pending",detail:""})});segs.forEach(function(g){g.classList.remove("on")})}
function tick(){if(!T.on)return;$$("tn").textContent=(T.b+(performance.now()-T.a)/1000).toFixed(1);requestAnimationFrame(tick)}
function fail(m,base){T.on=false;setS("err");h1.replaceChildren.apply(h1,H1);$$("em").textContent=m||"";var b=$$("ebase");b.hidden=!base;
 if(base)b.href=base;title("Not processed")}
function stop(m,name){T.on=false;setS("stop");h1.textContent="Processing stopped.";$$("sname").textContent=name||"";var k=m.indexOf("Fallback: ");
 $$("sm").textContent=k<0?m:m.slice(0,k).trim();$$("scmd").textContent=k<0?"python run.py --new file.csv":m.slice(k+10);
 title("Stopped");history.replaceState(null,"","/")}
function render(j){var n=0,cur=-1;j.stages.forEach(function(s,i){var was=rows[i].dataset.st;setRow(rows[i],s);
 if(s.state==="done"){n++;if(was!=="done")cur=i}});
 if(cur>=0){var s=j.stages[cur];say("Stage "+(cur+1)+" of 7 done: "+s.label+(s.detail?", "+s.detail.split(" \\u00b7 ").join(", "):""))}
 segs.forEach(function(g,i){g.classList.toggle("on",i<n)});var at=Math.min(n+1,7);
 $$("cname").textContent=j.name;$$("clines").textContent=j.lines+" lines";
 $$("cstage").textContent=n===7?"7 of 7 stages done":"Stage "+at+" of 7: "+j.stages[at-1].label;
 T.b=j.elapsed_s;T.a=performance.now();title(n===7?"Done":"Processing "+at+" of 7")}
function poll(id){fetch("/api/jobs/"+id).then(function(r){if(!r.ok)throw 0;return r.json()}).then(function(j){
 if(box.dataset.s!=="run"){setS("run");h1.textContent="Processing the ledger."}render(j);
 if(j.state==="done"){T.on=false;$$("tn").textContent=j.elapsed_s.toFixed(1);setTimeout(function(){location.href="/jobs/"+id},300)}
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
$$("demo").addEventListener("click",function(){start("/api/jobs?demo=1",{method:"POST"},"new_lines.csv")});
var m=location.pathname.match(/^\\/jobs\\/([A-Za-z0-9_-]+)$$/);if(m){setS("run");h1.textContent="Processing the ledger.";poll(m[1])}
</script></body></html>""")

RESULTS = Template("""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>$title</title>
<meta name="viewport" content="width=device-width,initial-scale=1"><style>$fonts
$css
.rise{animation:up8 var(--t-slow) var(--ease) both}.d1{animation-delay:60ms}.d2{animation-delay:120ms}.d3{animation-delay:180ms}
.receipt{grid-column:1/-1}.receipt summary{display:flex;align-items:center;gap:16px;min-height:56px;padding:0 20px;
 background:var(--sheet);border:1px solid var(--rule);border-radius:14px;cursor:pointer;list-style:none}
.receipt summary::-webkit-details-marker{display:none}.receipt[open] summary{border-radius:14px 14px 0 0}
.ticks{display:flex;gap:4px}.ticks .g{width:14px;height:14px}.ticks .g::after{left:4.5px;top:2px;width:3px;height:6px;border-width:0 1.5px 1.5px 0}
.rs{display:flex;align-items:center;gap:16px;flex-wrap:wrap}.rs>span+span{border-left:1px solid var(--rule);padding-left:16px}
.rs .mono{font-size:14px}.receipt .tog{margin-left:auto;font-weight:600;text-decoration:underline;text-underline-offset:3px}
.receipt .tog .h,.receipt[open] .tog .s{display:none}.receipt[open] .tog .h{display:inline}
.receipt .body{border:1px solid var(--rule);border-top:0;border-radius:0 0 14px 14px;background:var(--sheet);padding:20px 20px 8px}
.receipt .rail{max-width:720px}
.heroc{grid-column:1/9;margin-top:40px}.heroc h1{font-size:18px;line-height:1.45;font-weight:400;letter-spacing:0;color:var(--mute);margin:0 0 12px}
.num{font-size:128px;line-height:.9;font-weight:800;letter-spacing:-.04em;white-space:nowrap;font-variant-numeric:proportional-nums}
.num small{font-size:.4em;font-weight:700;letter-spacing:-.01em;color:var(--mute)}.num .u{margin-left:.35em}.num .fr{margin-left:.06em}
.heroc .sub{margin-top:20px;color:var(--mute)}
.outs{grid-column:9/13;margin-top:40px;align-self:start;background:var(--sheet);border:1px solid var(--rule);border-radius:14px;padding:8px}
.outs h2{font-size:15px;line-height:1.4;font-weight:600;letter-spacing:0;margin:8px 12px 8px}
.out{display:flex;align-items:center;gap:12px;min-height:48px;padding:6px 12px;border-radius:8px;text-decoration:none;
 transition:background-color var(--t-fast),transform 100ms}.out:hover{background:var(--well)}.out:active{transform:scale(.98)}
.out .chip{min-width:44px;text-align:center}.out b{font-weight:600}.out .small{margin-left:auto}
.out.p{background:var(--ink);color:var(--paper)}.out.p .chip{background:transparent;color:var(--paper);border-color:rgba(247,245,240,.5)}
.out.p .small{color:#D9D6CF}.out.p:hover{background:#2A2D31}.nopdf{padding:8px 12px}.outs .new{display:block;padding:12px 12px 8px}
.scope{grid-column:1/-1;margin-top:40px}.bar{display:flex;gap:2px;height:16px;border-radius:8px;overflow:hidden}
.bar i{display:block;min-width:4px;transform-origin:left;animation:grow var(--t-slow) var(--ease) both}
.bar i:nth-child(2){animation-delay:60ms}.bar i:nth-child(3){animation-delay:120ms}
.s1{background:var(--ink)}.s2{background:#7A7C80}.s3{background:var(--rule)}
.slab{display:flex;gap:2px;margin-top:12px}.slab div{min-width:170px;padding-right:16px}
.slab p:first-child{font-size:13px;color:var(--mute);display:flex;align-items:center;gap:8px}
.slab p:first-child i{width:10px;height:10px;border-radius:3px}
.slab .t{font-size:24px;font-weight:700;letter-spacing:-.01em;line-height:1.3}
.sec{margin-top:64px}.left{grid-column:1/6}.right{grid-column:6/13}.full{grid-column:1/-1}
.ledger{width:100%;border-collapse:collapse}.ledger td{padding:12px 0;border-bottom:1px solid var(--rule)}
.ledger td.n{text-align:right;font-weight:600}.ledger tr.sub td{padding:8px 0 8px 24px;color:var(--mute);font-size:13px}
.ledger tr.sub td.n{font-weight:400}.ledger tr.sum td{border-top:3px double var(--ink);border-bottom:0;font-weight:700}
.ledger .ex td{border-bottom:0}.ledger tr.sub:not(.last) td{border-bottom:0}
.ban{display:flex;align-items:center;gap:14px;background:var(--acc-wash);border-radius:14px;padding:14px 18px}
.ban .c{font-size:24px;line-height:1;font-weight:700;color:var(--acc)}.ban .g.w{background:var(--acc);color:var(--paper);
 font:700 13px/18px Onest,system-ui,sans-serif;text-align:center}
.q{list-style:none;margin:8px 0 0;padding:0}.q li{display:grid;grid-template-columns:72px minmax(0,1fr) auto;gap:16px;
 padding:12px 0;border-bottom:1px solid var(--rule);align-items:baseline}
.q .t,.cards .t{overflow-wrap:anywhere}.q .fl{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}.q .n{text-align:right;font-weight:600;white-space:nowrap}
.more{display:inline-block;margin-top:14px;font-weight:600}.quiet{color:var(--mute);padding:14px 0;border-top:1px solid var(--rule)}
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;margin-top:16px}
@media(max-width:959px){.cards{grid-template-columns:minmax(0,1fr)}}
.card{background:var(--sheet);border:1px solid var(--rule);border-left:3px solid var(--acc);border-radius:14px;padding:16px 20px}
.card blockquote{margin:10px 0 12px;font-size:15px}.card blockquote::before{content:"\\201C"}.card blockquote::after{content:"\\201D"}
.card .o{font-size:13px;color:var(--mute);padding-top:10px;border-top:1px solid var(--rule)}.card .o b{color:var(--ink);font-weight:600}
.vh{display:flex;align-items:center;gap:24px;flex-wrap:wrap}.vh h2{margin:0}
.seg{display:inline-flex;gap:4px;padding:4px;background:var(--well);border-radius:12px}
.seg a{display:flex;align-items:center;height:36px;padding:0 16px;border-radius:8px;color:var(--mute);font-weight:600;text-decoration:none;
 transition:background-color var(--t-fast),color var(--t-fast),transform 100ms}.seg a:active{transform:scale(.98)}
.seg a[aria-current=true]{background:var(--sheet);color:var(--ink);box-shadow:0 1px 2px rgba(21,23,26,.12)}
.desk{margin-top:16px;padding:12px;background:var(--well);border-radius:20px;box-shadow:var(--lift)}
.desk iframe{display:block;width:100%;height:760px;border:0;border-radius:12px;background:var(--sheet);transition:opacity var(--t-slow) var(--ease)}
.desk iframe.fade{opacity:0;transition-duration:100ms}
</style></head><body>$header<main class="grid">
<details class="receipt rise"><summary><span class="ticks" aria-hidden="true">$ticks</span><span class="rs">
<span class="mono" dir="auto" translate="no">$name</span><span>$lines lines</span><span>7 of 7 stages</span><span>$elapsed s</span></span>
<span class="tog"><span class="s">Show stages</span><span class="h">Hide stages</span></span></summary><div class="body">$rail</div></details>
<section class="heroc rise d1"><h1>Emissions in the uploaded lines</h1>
<p class="num"><span class="sr">$total tCO2e</span><span id="cnt" aria-hidden="true" data-v="$intpart"><span>$intpart</span><small class="fr">$frac</small><small class="u">tCO2e</small></span></p>
<p class="sub">Scope 1, 2 and 3 for $counted counted lines of $lines.</p></section>
<section class="outs rise d2" aria-labelledby="oh"><h2 id="oh">Outputs</h2>
<a class="out p" href="$base/dashboard.html#split=new" target="_blank"><span class="chip">HTML</span><b>Dashboard</b><span class="small">new lines</span></a>
<a class="out" href="$base/bank_summary.html" target="_blank"><span class="chip">HTML</span><b>Bank one-pager</b></a>$pdf
<a class="out" href="$base/audit.csv"><span class="chip">CSV</span><b>Audit CSV</b><span class="small">$lines rows</span></a>
<a class="new" href="/">New file</a></section>
<section class="scope rise d3" aria-label="Emissions by scope">$bar</section>
<section class="sec left"><h2>The lines</h2><table class="ledger">$ledger</table></section>
<section class="sec right"><h2>Review queue</h2>$review</section>
$inj
<section class="sec full"><div class="vh"><h2>Outputs, opened here</h2><nav class="seg" aria-label="Viewer">
<a href="$base/dashboard.html#split=new" target="viewer" aria-current="true">Dashboard</a>
<a href="$base/bank_summary.html" target="viewer" aria-current="false">Bank one-pager</a></nav>
<p class="small">Bank pages: whole company inventory with your lines added.</p></div>
<div class="desk"><iframe id="viewer" name="viewer" src="$base/dashboard.html#split=new" title="Outputs viewer"></iframe></div></section>
</main><script>
(function(){var rm=matchMedia("(prefers-reduced-motion: reduce)"),c=document.getElementById("cnt"),fr=document.getElementById("viewer");
 var v=+c.dataset.v.replace(/,/g,""),f=c.firstChild,t0=performance.now();
 if(!rm.matches&&isFinite(v))(function s(t){var k=rm.matches?1:Math.min(1,(t-t0)/400),e=1-Math.pow(1-k,3);
  f.textContent=(k<1?Math.round(v*e):v).toLocaleString("en-US");if(k<1)requestAnimationFrame(s)})(t0);
 var segs=[].slice.call(document.querySelectorAll(".seg a"));segs.forEach(function(a){a.addEventListener("click",function(){
  segs.forEach(function(b){b.setAttribute("aria-current",b===a?"true":"false")});if(!rm.matches)fr.classList.add("fade")})});
 fr.addEventListener("load",function(){fr.classList.remove("fade")})})();
</script></body></html>""")


def results_page(job):
    S, b = job["summary"], f"/jobs/{job['id']}"
    tot = S["total"]
    pct = lambda x: f"{100 * x / tot:.1f}%" if tot else "0%"
    whole, _, frac = f"{tot:,.1f}".partition(".")
    pos = {s: max(0.0, S["by_scope"][s]) for s in "123"}
    span = sum(pos.values()) or 1.0
    bar = ('<div class="bar" aria-hidden="true">' + "".join(f'<i class="s{s}" style="flex:{pos[s] / span:.4f} 1 0"></i>' for s in "123")
           + '</div><div class="slab">' + "".join(
               f'<div style="flex:{pos[s] / span:.4f} 1 0"><p><i class="s{s}" aria-hidden="true"></i>Scope {s}</p>'
               f'<p class="t">{S["by_scope"][s]:,.1f} t</p><p class="small">{pct(S["by_scope"][s])} of the total</p></div>'
               for s in "123") + "</div>")
    exc = S["duplicates"] + S["nonpurchase"]
    ledger = (f'<tr><td>Counted</td><td class="n">{S["counted"]}</td></tr>'
              f'<tr class="ex"><td>Excluded</td><td class="n">{exc}</td></tr>'
              f'<tr class="sub"><td>Duplicate document</td><td class="n">{S["duplicates"]}</td></tr>'
              f'<tr class="sub last"><td>Not a purchase</td><td class="n">{S["nonpurchase"]}</td></tr>'
              f'<tr class="sum"><td>Lines in the file</td><td class="n">{S["lines"]}</td></tr>')
    top = S.get("review_top", [])
    if S["review"]:
        review = (f'<div class="ban" role="note"><span class="g w" aria-hidden="true">!</span><span class="c">{S["review"]}</span>'
                  f'<p>lines to review. Flags never change a result; a person confirms each line.</p></div><ul class="q">' + "".join(
                      f'<li><span class="mono" translate="no">{E(r["line_id"])}</span><div><p class="t" dir="auto" translate="no">'
                      f'{E(r["description"])}</p><p class="fl">' + "".join(f'<span class="chip">{E(f)}</span>' for f in r["flags"])
                      + f'</p></div><span class="n">{r["tco2e"]:,.2f} t</span></li>' for r in top)
                  + f'</ul><a class="more" href="{b}/dashboard.html#split=new&amp;tab=queue" target="_blank">'
                    f'Open all {S["review"]} in the dashboard</a>')
    else:
        review = '<p class="quiet">No lines to review.</p>'
    inj = ""
    if S["injection"]:
        n = len(S["injection"])
        inj = (f'<section class="sec full"><h2>Instruction-like text</h2><div class="ban" role="note"><span class="g w" aria-hidden="true">!'
               f'</span><span class="c">{n}</span><p>{"line" if n == 1 else "lines"} carried instruction-like text. It was ignored.</p>'
               '</div><div class="cards">' + "".join(
                   f'<article class="card"><p class="mono" translate="no">{E(r["line_id"])}</p><blockquote class="t" dir="auto" '
                   f'translate="no">{E(r["description"])}</blockquote><p class="o">Ignored. Counted as <b class="mono">'
                   f'{E(r["category"])}</b>, {r["tco2e"]:.4f} t</p></article>' for r in S["injection"]) + "</div></section>")
    pdf = (f'<a class="out" href="{b}/bank_summary.pdf"><span class="chip">PDF</span><b>Bank PDF</b><span class="small">1 page</span></a>'
           if S["pdf"] else '<p class="small nopdf">Bank PDF needs Edge or Chrome; open the one-pager and print it.</p>')
    ticks = '<span class="g ok"></span>' * len(job["stages"])
    return RESULTS.substitute(fonts=FONTS, css=CSS, header=HEADER, base=b, total=f"{tot:,.1f}", intpart=whole, frac="." + frac,
                              title=E(f"{tot:,.1f} tCO2e, Carbon Copilot"), name=E(job["name"]), lines=S["lines"],
                              counted=S["counted"], elapsed=f"{job['elapsed']:.1f}", ticks=ticks, rail=rail(job["stages"]),
                              bar=bar, ledger=ledger, review=review, inj=inj, pdf=pdf)


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


SHELL_HTML = SHELL.substitute(fonts=FONTS, css=CSS, header=HEADER, cols="".join(f'<span class="chip">{c}</span>' for c in REQ),
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
