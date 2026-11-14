"""TypeSafe Jev (typesafe/jev-router) as the second reader, on the SAME sets as llm_fallback --eval.

Reuses llm_fallback read-only: folds, gate, per-line allowlist, masked payload, check_payload (leak checks), parse,
run_split metrics and e3. Only the backend is swapped (L.read_many is replaced at runtime; the file is not edited).
Own cache (jev_cache.json) and spend log (spend.json, hard cap CAP_USD). Never touches llm_cache.json / llm_spend.json.

Run (from anywhere):
  python eval_jev.py --pilot 20          first 20 E1 gated lines, measures cost/latency per call
  python eval_jev.py --full [--e1step 3 --e2step 4 --e3n 5]   every k-th gated E1/E2 line + E3 (e3n lines x 12 forms);
                                         DeepSeek and pipeline scored on the SAME lines (out/llm_eval.csv); writes results.json
"""
import json, math, os, sys, time, urllib.request, hashlib, threading
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
os.environ["DALA_LLM"] = "cache"                       # llm_fallback must never call DeepSeek or save its files
import pipeline as P
import llm_fallback as L
import eval_hard as H

MODEL = "typesafe/jev-router"                          # ~typesafe/jev-latest: output "decisions", no endpoints/pricing
CAP_USD = 0.10
CACHE_PATH, SPEND_PATH = os.path.join(HERE, "jev_cache.json"), os.path.join(HERE, "spend.json")
CACHE = L._json(CACHE_PATH, {})
SPEND = L._json(SPEND_PATH, dict(model=MODEL, cap_usd=CAP_USD, usd=0.0, calls=0, tokens_in=0, tokens_out=0,
                                 reasoning_tokens=0, latency_s=[], routed_to={}, key_usage_log=[], stopped=False))
_lock = threading.Lock()


def key_usage(tag):
    k = os.environ["OPENROUTER_API_KEY"]
    r = urllib.request.Request("https://openrouter.ai/api/v1/key", headers={"Authorization": "Bearer " + k})
    u = json.load(urllib.request.urlopen(r, timeout=20))["data"]["usage"]
    SPEND["key_usage_log"].append(dict(tag=tag, usage_usd=u, at=time.strftime("%H:%M:%S")))
    return u


def save():
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(CACHE, f, indent=0, sort_keys=True)
    with open(SPEND_PATH, "w", encoding="utf-8") as f:
        json.dump(SPEND, f, indent=1)


def ckey(p):
    return hashlib.sha256(json.dumps([L.PROMPT_V, MODEL, p], ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def _conf(choice_obj):
    """Sequence probability of the answer tokens, if logprobs came back (else None)."""
    lp = ((choice_obj.get("logprobs") or {}).get("content")) or []
    return round(math.exp(sum(t["logprob"] for t in lp)), 4) if lp else None


def call(p, opts):
    """One masked payload -> dict(label, conf). label in opts | NONE | REJECTED | None (error / budget)."""
    k = ckey(p)
    if k in CACHE:
        return CACHE[k]
    if ALLOWED is not None and k not in ALLOWED:      # outside the sample: not sent, scored as "not asked"
        return dict(label=None, conf=None)
    L.check_payload(p)                                  # leak check before anything leaves (raises on leak)
    with _lock:
        per = SPEND["usd"] / SPEND["calls"] if SPEND["calls"] else 0.002
        if SPEND["usd"] + 2 * per > CAP_USD:           # ponytail: 2x mean cost as headroom; 8 threads may overshoot by ~8 calls
            SPEND["stopped"] = True
            return dict(label=None, conf=None)
    body = dict(model=MODEL, temperature=0, seed=42, max_tokens=400, usage=dict(include=True),
                logprobs=True, top_logprobs=3, reasoning=dict(effort="low", exclude=True),
                provider=dict(data_collection="deny"),
                messages=[dict(role="system", content=L.SYSTEM), dict(role="user", content=json.dumps(p, ensure_ascii=False))],
                response_format=dict(type="json_schema", json_schema=dict(name="pick", strict=True, schema=dict(
                    type="object", additionalProperties=False, required=["choice"],
                    properties=dict(choice=dict(type="string", enum=opts + ["NONE"]))))))
    req = urllib.request.Request("https://openrouter.ai/api/v1/chat/completions", json.dumps(body).encode(),
                                 {"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"], "Content-Type": "application/json"})
    out, t0 = None, time.time()
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                out = json.load(resp)
            break
        except Exception as e:
            msg = e.read().decode()[:200] if hasattr(e, "read") else str(e)[:120]
            print(f"  jev error: {type(e).__name__} {msg}", file=sys.stderr)
            time.sleep(1 + attempt)
    dt = time.time() - t0
    if out is None:
        return dict(label=None, conf=None)
    u = out.get("usage") or {}
    with _lock:
        SPEND["usd"] += float(u.get("cost") or 0)
        SPEND["calls"] += 1
        SPEND["tokens_in"] += u.get("prompt_tokens", 0); SPEND["tokens_out"] += u.get("completion_tokens", 0)
        SPEND["reasoning_tokens"] += (u.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0
        SPEND["latency_s"].append(round(dt, 2))
        m = out.get("model", "?"); SPEND["routed_to"][m] = SPEND["routed_to"].get(m, 0) + 1
        SPEND["cost_missing"] = SPEND.get("cost_missing", 0) + (u.get("cost") is None)
    try:
        ch = out["choices"][0]
        text = ch["message"]["content"] or ""
    except (KeyError, IndexError):
        return dict(label=None, conf=None)
    lab = L.parse(text.strip().removeprefix("```json").removesuffix("```").strip(), opts) or "REJECTED"
    res = dict(label=lab, conf=_conf(ch), model=out.get("model"))
    with _lock:
        CACHE[k] = res
    return res


def read_many(jobs, threads=8):
    try:
        with ThreadPoolExecutor(threads) as ex:
            res = list(ex.map(lambda j: call(*j), jobs))
    finally:
        save()
    for (p, _), r in zip(jobs, res):
        CONF[ckey(p)] = r.get("conf")
    return [r["label"] for r in res]


CONF = {}
ALLOWED = None                                           # set of payload keys that may be sent (None = all)
L.read_many = read_many                                  # backend swap for run_split / e3 (runtime only)


def splits():
    train = P.load(os.path.join(P.MASKED, "ledger_train.csv"))
    test = P.load(os.path.join(P.MASKED, "ledger_test.csv"))
    dups = P.dup_ids(train + test)
    tf, _ = H.group_folds(train, H.template)
    gf, _ = H.group_folds(train, lambda r: r["gl_account"])
    return train, L._fold_preds(train, dups, tf), L._fold_preds(train, dups, gf)


def calib(rows_out, sel_payloads):
    """Confidence vs correctness on answered lines (only if logprobs came back)."""
    pts = [(CONF.get(ckey(p)), r["sr_ok"]) for r, p in zip(rows_out, sel_payloads) if CONF.get(ckey(p)) is not None]
    if not pts:
        return None
    b = {}
    for c, ok in pts:
        k = "0.99+" if c >= 0.99 else "0.9-0.99" if c >= 0.9 else "<0.9"
        n, a = b.get(k, (0, 0)); b[k] = (n + 1, a + ok)
    return {k: dict(n=n, acc=round(a / n, 3)) for k, (n, a) in b.items()}


def main():
    a = sys.argv
    before = key_usage("before")
    train, r1, r2 = splits()
    if "--pilot" in a:
        n = int(a[a.index("--pilot") + 1])
        sel = [j for j, (r, p, m, _) in enumerate(r1) if L.gate(r, p, m)[0]][:n]
        jobs = [L.payload(r1[j][0], r1[j][2], r1[j][3]) for j in sel]
        labs = read_many(jobs)
        ok = sum(lab == r1[j][0]["y_category"] + "|" + r1[j][0]["y_ef_id"] for j, lab in zip(sel, labs))
        print(f"pilot: {n} lines, {ok} exact, labels {labs}")
    else:
        global ALLOWED
        e3n = int(a[a.index("--e3n") + 1]) if "--e3n" in a else 5
        step = dict(E1=int(a[a.index("--e1step") + 1]) if "--e1step" in a else 3,
                    E2=int(a[a.index("--e2step") + 1]) if "--e2step" in a else 4)
        ds = {}
        with open(os.path.join(P.OUTD, "llm_eval.csv"), encoding="utf-8") as f:
            for x in __import__("csv").DictReader(f):
                ds[(x["split"][:2], x["line_id"])] = x
        out = {}
        for name, res in (("E1 unseen wording", r1), ("E2 unseen accounts", r2)):
            sp = name[:2]
            sel = [j for j, (r, p, m, _) in enumerate(res) if L.gate(r, p, m)[0]]
            samp = sel[::step[sp]]
            ALLOWED = {ckey(L.payload(res[j][0], res[j][2], res[j][3])[0]) for j in samp}
            _, rows, _ = L.run_split(name + " (Jev, sample only scored)", train, res)
            ids = {res[j][0]["line_id"] for j in samp}
            rs = [x for x in rows if x["line_id"] in ids]
            asked = [x for x in rs if x["second_reader"]]
            def acc(rr, f):
                return round(sum(f(x) for x in rr) / max(len(rr), 1), 4)
            cat = lambda u: u.split("|")[0]
            d = [ds[(sp, x["line_id"])] for x in rs]
            o = dict(sample=len(rs), answered_by_jev=sum(x["second_reader"] not in ("", "NONE", "REJECTED") for x in rs),
                     none=sum(x["second_reader"] == "NONE" for x in rs), rejected=sum(x["second_reader"] == "REJECTED" for x in rs),
                     not_sent_budget=len(rs) - len(asked),
                     pipeline_ef_acc=acc(rs, lambda x: x["ours_ok"]), jev_ef_acc=acc(rs, lambda x: x["sr_ok"]),
                     deepseek_ef_acc=acc(d, lambda x: x["sr_ok"] == "True"),
                     pipeline_cat_acc=acc(rs, lambda x: cat(x["ours"]) == x["y_category"]),
                     jev_cat_acc=acc(rs, lambda x: cat(x["used"]) == x["y_category"]),
                     deepseek_cat_acc=acc(d, lambda x: cat(x["used"]) == x["y_category"]),
                     jev_vs_deepseek_same_pick=sum(x["used"] == y["used"] for x, y in zip(rs, d)),
                     pipeline_abs_t=round(sum(abs(float(x["ours_tco2e"]) - P.num(x["y_tco2e"])) for x in rs), 2),
                     jev_abs_t=round(sum(abs(float(x["sr_tco2e"]) - P.num(x["y_tco2e"])) for x in rs), 2),
                     deepseek_abs_t=round(sum(abs(float(y["sr_tco2e"]) - P.num(y["y_tco2e"])) for y in d), 2))
            print(sp, o)
            out[sp] = o
            with open(os.path.join(HERE, f"rows_{sp}.json"), "w", encoding="utf-8") as f:
                json.dump(rs, f, indent=0, default=str)
        ALLOWED = None
        if not SPEND["stopped"]:
            sel1 = [j for j, (r, p, m, _) in enumerate(r1) if L.gate(r, p, m)[0]]
            out["E3"] = L.e3(r1, sel1, n=e3n)
        out["stopped_at_cap"] = SPEND["stopped"]
        with open(os.path.join(HERE, "results.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
    time.sleep(3)                                        # key usage lags a few seconds
    after = key_usage("after")
    SPEND["key_delta_last_run_usd"] = round(after - before, 6)
    lat = sorted(SPEND["latency_s"])
    if lat:
        SPEND["latency_median_s"] = lat[len(lat) // 2]; SPEND["latency_p90_s"] = lat[int(len(lat) * 0.9)]
    SPEND["usd_per_call"] = round(SPEND["usd"] / max(SPEND["calls"], 1), 7)
    save()
    print(f"spend: response-sum USD {SPEND['usd']:.5f} over {SPEND['calls']} calls; key delta this run {after - before:.5f}; "
          f"routed {SPEND['routed_to']}; median latency {SPEND.get('latency_median_s')}s")


if __name__ == "__main__":
    main()
