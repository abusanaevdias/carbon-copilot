# Carbon Copilot: Dala Foods 2025 GHG inventory from the purchase ledger

Carbon Copilot turns the raw 1C purchase-ledger export into a GHG inventory (Scope 1, 2, 3), a leaderboard
predictions file and a per-line audit file. Every tonne traces to one ledger line, one factor from
`emission_factors.csv` and a written reason.

## Setup and one command

Requirements: Python 3.9 or newer. Standard library only: no `pip install`, no network, no API key.

```
<any folder>/
  cases/Case2_DalaFoods/      the organiser data pack as given (ledger_train.csv, ledger_test.csv,
                              emission_factors.csv, categories.csv, airports.csv, packaging_specs.csv,
                              sample_submission.csv)
  Team<N>_DalaFoods/          this zip, unpacked
```

```
cd Team<N>_DalaFoods
python run.py                              # raw CSVs -> everything below; exit code 0 = all self-checks passed
python run.py --new demo/new_lines.csv     # also classify extra raw lines (same columns as ledger_test.csv)
```

`python run.py <DATA_DIR>` accepts another location of the data pack once `pipeline.py` reads the `DALA_DATA`
variable that `run.py` sets; otherwise `run.py` stops and says where to put the data.

What one run does, in order (it prints counts only, never names or rows):

1. **Mask** (`mask.py`): counterparty and employee names in the raw ledgers are replaced with salted
   pseudonyms on this machine; then the masked files are re-read from disk and scanned for every raw name.
   Any hit stops the run.
2. **Pipeline** (`pipeline.py`): rules (duplicates, non-purchase GL accounts, capital accounts, flights,
   refrigerants, lexicon) with a naive Bayes fallback restricted by the GL account; unit conversion into the
   factor's unit; spend-based fallback net of VAT. Self-checks: 5-fold CV, line ids equal
   `sample_submission.csv`, every category and factor is from the organiser files, injected instructions
   change nothing.
3. **Report** (`report.py`): inventory for train + test, bank one-pager, dashboard.

Outputs:

| File | Content |
|---|---|
| `predictions.csv` | `line_id,category,tco2e` for the 683 test lines |
| `audit.csv` | per test line: category, ef_id, activity quantity and unit, method, tCO2e, reason |
| `out/inventory.json`, `out/bank_summary.html`, `out/dashboard.html` | inventory, 1-page bank summary, drill-down dashboard (opens from `file://`, no network) |
| `out/audit_new.csv` | only with `--new`: the same audit columns for the extra lines |
| `masked/`, `mask_map.csv`, `.mask_salt` | local only, never packed or shared |

Other commands: `python mask.py --check` re-scans the masked files for raw names (exit 1 on a leak);
`python eval_hard.py` runs the hard hold-out scenarios; `python tests/<test>.py` runs the tests;
`python make_zip.py <N>` builds this zip and refuses if any packed file contains a raw name.

The run is deterministic: deleting `.mask_salt` changes every pseudonym but gives a byte-identical
`predictions.csv`, because no feature uses the pseudonym itself.

## Development log (masking): two gaps found by our review, both closed

The brief: unmasked counterparty or employee names must not be sent to an external cloud AI. The code was
written with a cloud AI coding assistant, so the rule applied to our development process, not only to the
runtime. Both gaps were found during development:

1. **Before masking existed.** While inspecting the file format, 4 raw training rows, including counterparty
   names, were displayed to the cloud AI assistant. After that the assistant wrote `mask.py` from the brief,
   the reference files and the header line only, and read ledger rows only after the masking self-check passed.
2. **Mask v1 missed a pattern.** The first regex set did not catch "ИП <surname>" inside a description, so
   one carrier name reached the AI when the masked data was first viewed. Fix: the `legal_form_name` pattern
   (legal form + name, e.g. ИП / ТОО + name), files re-masked, and a zero-leak self-check that fails the run.

Current safeguards:

- Every counterparty becomes `<legal form> CP_<8 hex>` (SHA-256 of a local salt + name). Free text
  (`description`, `cost_center`) is masked for `Surname I.O.`, `I.O. Surname`, full names with patronymic,
  legal form + name, names after trigger words (employee, passenger, carrier, driver ...) and counterparty
  names quoted in text. Last run: 1,269 + 683 lines, 67 counterparties, 8 person spans, leaks 0.
- The self-check runs twice: in memory before writing and on the files written to disk. It checks every raw
  counterparty name, its core without the legal form, every masked person span and any word with a
  patronymic ending. One hit = exit code 1.
- `make_zip.py` scans every file it packs for the same names.
- **By default the run makes no network calls: the two AI readers answer from a local cache of masked-text
  replies. Live AI calls are opt-in, send masked text only and return a flag, never a number.** The AI assistant was used only to write code, on masked data after incident 2.
- Ledger text is data, never instructions: instruction-like phrases are cut before features are built and
  the line is flagged; all numbers come from code and `emission_factors.csv`.

Residual risk we know of: a name in a format none of the patterns covers. The patronymic and surname-suffix
scans are the backstop; the last scan found 3 surname-like tokens, which the GL accounts show are product
words, not people.

## Cost to run

Inputs: `deck/cost_model.json` (each tagged brief, measured, sourced with URL, or ASSUMED). Computed by `cost()` in
`deck/build.py`; check with `python deck/build.py --cost-check`. "est." on a slide = our assumption inside a published range.

| Line (KZT m a year) | v4 | Year 1 | Year 2+ | Basis |
|---|---|---|---|---|
| Independent check | 7.50 | 7.50 = 8 days × USD 1,875 × 500 | 5.25 (× 0.70) | days, rate, FX ASSUMED inside sourced ranges; −30% = low end of a published 30–60% |
| People | 0.74 (140.5 h) | 0.83 (156.5 h) | 0.83 | hours ASSUMED; wage KZT 443,315 (stat.gov.kz) × 2.0 ÷ 168 h = KZT 5,278/h |
| AI | 0.00 | ≤ KZT 332 | ≤ KZT 332 | measured: both readers on all 1,952 lines (h2h/results.json) |
| **Total** | **8.24** | **8.33 (56% of cap)** | **6.08 (41%)** | cap KZT 15 m [brief] |
| Per tonne CO2e | 41.3 | 41.7 | 30.4 | 199,638 t measured |
| Share of the KZT 5 bn loan | 0.165% | 0.167% | 0.12% | [brief] |
| Tool only | 0.74 | 0.83 (6% of cap) | 0.83 | |

People hours: yearly run 16, review queue 18 lines × 15 min = 4.5, factor update 40, maintenance 80, verifier support 16.
Three-year average 6.83 m (46%). **Stress:** at 15 verifier days (top of the sourced range) year 1 is 14.89 m, 99% of the cap.
Tool cost per company with a shared factor library (120 h shared, 36.5 h each, est.): 0.83 m (1), 0.32 m (5), 0.26 m (10).

| Lever | Effect | In base case? |
|---|---|---|
| Tool vs the bank's check shown apart | tool 0.83 m = 6% of cap; check required by the bank either way | yes |
| Verifier pack (`python verifier_pack.py --seed N`) | fee year 2+: 7.50 → 5.25 m, est.; not quoted by any verifier | yes, as est. |
| External review every second year | would halve the average fee | no: a negotiation with the bank |
| People hours +16 h verifier support | +0.08 m | yes |
| Cut maintenance 80 → 40 h | −0.21 m | no |
| AI only on flagged lines | under KZT 1 instead of KZT 332 | ceiling shown |
| Local AI fallback | saves KZT 332, worse accuracy | no |

Verifier pack (`out/verifier_pack/`, masked data only): `factors.csv` (47 factors; top 3 = 90.4% of tonnes),
`reconcile.csv` (milk 121,153 t in the ledger vs about 120,000 t in the brief, +1.0%; other totals to be supplied by Dala),
`sample.csv` (18 review-queue lines + 23 lines ≥ 0.25% of the total + 25 random lines on the verifier's seed),
`recompute.csv` (an Excel formula per line that recomputes tCO2e next to ours; checked on all 1,877 included lines).

Sources, opened 2026-11-14:
Glocert fees and days (glocertinternational.com/resources/guides/ghg-verification-process-and-timeline/): mid-size 3–10 sites
Scope 1+2+3 limited USD 15–30k, 8–15 days · Glocert evidence pack (…/ghg-verification-readiness-checklist-evidence-pack/):
fixing the top deficiencies "reduces total verification time by 25-35%" · California Climate Action Registry workshop 2008,
slide 19: cost "Highest in Year 1; can decrease 30-60% in years 2 & 3" · Lincus proposal to a California city, RY2025
USD 8,640 = 40 h × USD 216 (URL in cost_model.json) · SGS Kazakhstan, CLATI, DQS: formula only, no price · EC MRV
evaluation 2016: verification EUR 6,789 per installation (2014) · Brightest: GBP 15–40k · CARB via ClimateAssured: USD 55,213 ·
LMA Guidance on Green Loan Principles, 26 March 2025.
Corrections after opening: the LMA "where appropriate ... should appoint an external review provider" is about review
**before origination** of the loan, not yearly GHG assurance; yearly is the use-of-proceeds update ("shall be updated at
least annually"), which lender and borrower "may agree" to review externally. Not found: any public Kazakh verification price.

## Real vs not built

Real and runnable: one command from raw CSVs, masking with a failing self-check, rule + naive Bayes
classification, unit conversion, VAT-net spend fallback, duplicate and non-purchase exclusion, credit notes,
injection robustness, per-line audit, `--new` for extra lines, the zip builder.

Not built: browser upload, logins, database, multi-company configuration, LLM explanations, market-based
Scope 2, product-level (ISO 14067) footprint, a statistical uncertainty range. Scope 3 categories a purchase
ledger cannot show (Cat 3, 7, 9, 12) are not assessed. Nothing here is verified or assured.

## AI use in development

Most of the code was written with a cloud AI coding assistant, working on masked data (see the log above).
The team can explain every rule and every check that fails the run.
