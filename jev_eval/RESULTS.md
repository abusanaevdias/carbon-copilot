# TypeSafe Jev as the second reader — vs DeepSeek and the pipeline alone

Measured 2026-11-14 with `eval_jev.py`. Model: `typesafe/jev-router` (`~typesafe/jev-latest` exists on OpenRouter but
has output modality "decisions", no endpoints and no pricing, so it cannot be called for chat).
The router picks the model per request: of 233 calls it sent 144 to openai/gpt-6-luna, 70 to deepseek-v4.1-flash,
6 to gpt-6.1-sol, 6 to gemini-4 (carbon/hydrogen), 5 to kimi-k3.1-pro and 2 to claude-opus-5.5.

Same sets, allowlists, masked payloads and leak checks as `llm_fallback.py --eval` (imported read-only; only the
backend was swapped at runtime). Every payload passed `check_payload` before it was sent. Because of the $0.10 cap,
Jev saw a sample: every 3rd gated E1 line (99 of 295), every 2nd gated E2 line (148 of 295) and 12 E3 lines x 12 forms.
DeepSeek and the pipeline are scored on **the same lines** (taken from `out/llm_eval.csv`).

| Same lines | Pipeline alone | DeepSeek v4.1 flash | Jev router |
|---|---|---|---|
| E1 unseen wording, factor accuracy (99 lines) | 0.515 | 0.990 | 0.990 |
| E1 abs. error, t CO2e | 6.21 | 1.86 | 1.86 |
| E2 unseen GL, category accuracy (148 lines) | 0.946 | 1.000 | 1.000 |
| E2 factor accuracy | 0.203 | 0.257 | 0.257 |
| E2 abs. error, t CO2e | 36.62 | 27.79 | 28.37 |
| E2 answered NONE | — | — | 75 of 148 |
| E3 injection: picks changed | — | 1 of 600 (50 lines, full run) | 0 of 144 (12 lines) |
| E3 picks outside the allowlist | — | 0 | 0 |
| Cost per call (response `usage.cost`) | $0 | $0.0000667 | $0.000220 (3.3x) |
| Latency per call | ms, local | not recorded in llm_eval.json | median 1.59 s, p90 2.4 s, max 17.7 s |
| Calibrated confidence | — | — | none: logprobs requested, router returns none |

Jev and DeepSeek picked the same label on 99/99 E1 lines and 138/148 E2 lines. Structured output (json_schema with
enum = allowlist + NONE) worked through the router; 0 rejected answers.

**Verdict:** Jev is no more accurate than DeepSeek on these sets, costs about 3.3x more per call, is not
deterministic in which model answers (30% of calls were DeepSeek itself) and gives no logprobs. Keep DeepSeek.

## Spend (`spend.json`)
- Sum of `usage.cost` over 233 calls (20-line pilot + 2 runs): **$0.0514** of the $0.10 cap.
- `GET /api/v1/key` usage: 0.43670 before, 0.48658 after, a delta of $0.0499 (the key lags and is shared with other sessions).
- Pilot: 17 calls, $0.0052, $0.00031 per call. E2 payloads are larger (unknown account means every label is offered).

## Reproduce
`python eval_jev.py --pilot 20` · `python eval_jev.py --full --e1step 3 --e2step 2 --e3n 12` (the cache in
`jev_cache.json` makes a rerun free). Per-line rows are in `rows_E1.json` and `rows_E2.json`; metrics are in `results.json`.
