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

## Data rule log (masking): two incidents, disclosed by us

The brief: unmasked counterparty or employee names must not be sent to an external cloud AI. The code was
written with a cloud AI coding assistant, so the rule applied to our development process, not only to the
runtime. It was broken twice, both times during development:

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
- **Runtime makes zero network calls and uses no LLM**, so ledger data never leaves the machine when the
  tool runs. The AI assistant was used only to write code, on masked data after incident 2.
- Ledger text is data, never instructions: instruction-like phrases are cut before features are built and
  the line is flagged; all numbers come from code and `emission_factors.csv`.

Residual risk we know of: a name in a format none of the patterns covers. The patronymic and surname-suffix
scans are the backstop; the last scan found 3 surname-like tokens, which the GL accounts show are product
words, not people.

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
