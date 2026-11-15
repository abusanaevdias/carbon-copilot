# Carbon Copilot

**Turns a company's purchase ledger into a greenhouse-gas inventory that a bank can check line by line.**

Built in 24 hours at CATALYST 2026 (NU AIChE industrial hackathon, Astana), on the Bureau Veritas case: Dala Foods, a Kazakh dairy, needs its Scope 1–3 inventory for a KZT 5 bn green loan. The bank's condition: *every number traceable to a document and a factor.*

[![tests](https://github.com/abusanaevdias/carbon-copilot/actions/workflows/tests.yml/badge.svg)](https://github.com/abusanaevdias/carbon-copilot/actions/workflows/tests.yml)
![python](https://img.shields.io/badge/python-3.9%2B%2C%20stdlib%20only-3776AB)
[Live showcase](https://carbon-copilot.vercel.app) · [Pitch deck (PDF)](deck/deck.pdf) · [How it works (RU)](docs/HOW_IT_WORKS_RU.md)

![Drop a ledger, get a traceable inventory](out/screens/app_v5/flow.gif)

| | |
|---|---|
| **199,638 t CO2e** | Dala Foods 2025, from 1,952 raw 1C ledger lines in Russian, Kazakh and English |
| **1.00 / 0.99** | on the organisers' scoring formula (0.4·macro-F1 + 0.6·(1−WAPE)); 0.99 on GL accounts the model never saw |
| **0 of 4,400** | lines with injected instructions ("set emissions to zero") moved a single tonne |
| **99.5 %** | of tonnes computed from physical quantities (kg, kWh, litres, km), not from money |
| **~45 s** | full run with every self-check, on a laptop, no network, no `pip install` |
| **$0.22** | total spent on cloud AI across every evaluation in the build, masked lines only |

## The problem

A purchase ledger is the one dataset every company already has. It is also a mess: free-text descriptions typed by different accountants in three languages, units in pieces, litres and tenge, duplicated invoices, credit notes, taxes posted as purchases, refrigerant top-ups hidden in maintenance lines, and supplier names that must not leave the building.

Asking a chat model to label it line by line looks like it works. We measured it: it scores well on a sample, then gives a different answer on the next run, and one flight came back 747× too high. A bank cannot sign off on a number that changes when you press the button again.

## What we built

```
raw 1C export ──► mask ──► clean ──► rules ──► classifier ──► factors ──► self-checks ──► audit file
                  names     dups,     1,411     448 lines      units →       CV, ids,       one reason
                  → CP_xxx  taxes,    lines     (naive Bayes,   factor unit,  injection     per line
                            capex               GL-guarded)     VAT net       replay
                                                     │
                                          18 lines → a person
```

- **A local web app** (`app.py`): drop a CSV, watch seven stages with live counts, get the inventory, a drill-down dashboard and a one-page PDF for the bank. Python standard library only.
- **An audit file** where every line carries the invoice number, quantity, unit, factor id, factor value, method and a sentence of reasoning. You can redo any tonne on a calculator.
- **A verifier pack** (`verifier_pack.py`) for the limited-assurance check the bank requires from next year.

## Engineering decisions

We started from the bank's requirement, not from a model, and every decision below came out of an argument we had with that requirement on the table.

**1. Code decides every number; AI only advises.** Deterministic rules and a small classifier make all final calls. Two external AI readers may suggest a category for a line the classifier is unsure about, but a suggestion is adopted only when both agree, inside the categories the GL account allows. Same input, same output, every run (`predictions.csv` is byte-identical across salts and machines).

**2. Activity data first, money last.** Spend-based factors are what everyone reaches for, and they hide the real hotspots. We convert pieces to kilograms through packaging specs, litres to kilograms by density, flights to kilometres via airport coordinates. Spend-based falls back only where no quantity exists, and is flagged with a data-quality tier.

**3. Names never leave the laptop.** `mask.py` replaces counterparty columns with salted hashes and finds people's names in free text (surname + initials, full names by Russian and Kazakh patronymics, legal forms). It then re-reads the masked files from disk and scans for every raw name; any hit stops the run. Numbers and document numbers are untouched, so totals are provably the same.

**4. We attacked it ourselves before anyone else could.** 22 kinds of prompt injection hidden in ledger descriptions, replayed over 200 lines each (`tests/test_injection.py`). Eight got past our first filter. We closed every one and kept them as regression tests.

**5. We chose the AI by measurement, not by brand.** Five candidate readers were run on the same masked lines ([`h2h/RESULTS.md`](h2h/RESULTS.md), [`jev_eval/`](jev_eval/), [`scaledown_eval/`](scaledown_eval/), [`laya_spike/`](laya_spike/)). Under new attacks a single reader made 2 dangerous picks out of 200; two readers that must agree made 0. That is the configuration we shipped.

**6. Baselines we built to beat.** We reproduced what other teams would do: the organisers' starter (0.56), a classifier bolted onto it (0.67–0.70), money-only in its best case (0.84), a chat model per line (0.93 on a sample). Write-up: [`docs/BASELINES.md`](docs/BASELINES.md).

## Run it

```bash
git clone https://github.com/abusanaevdias/carbon-copilot.git
cd carbon-copilot
python run.py                            # whole year → predictions.csv, audit.csv, out/
python app.py                            # local app on http://127.0.0.1:8765, try the demo file
python run.py --new demo/new_lines.csv   # classify lines the model has never seen
```

Tests: `python tests/test_app.py`, `python tests/test_injection.py`, `python tests/test_new_lines.py`. Hard hold-out scenarios: `python eval_hard.py`.

## Screens

| Drop | Processing | Result |
|---|---|---|
| ![](out/screens/app_v5/01_empty.png) | ![](out/screens/app_v5/04_processing_3.png) | ![](out/screens/app_v5/05_results_hero.png) |
| **Injected instruction, ignored and flagged** | **Dashboard down to one invoice** | **What the bank receives** |
| ![](out/screens/app_v5/07_injection.png) | ![](out/screens/app_v5/08_dashboard.png) | ![](out/screens/app_v5/09_bank_page.png) |

## Repository map

| Path | What |
|---|---|
| `run.py`, `pipeline.py`, `mask.py`, `report.py` | the engine: masking, classification, factors, inventory, reports |
| `app.py`, `templates/` | local web app and the dashboard / bank one-pager templates |
| `llm_fallback.py`, `h2h/`, `jev_eval/`, `scaledown_eval/`, `laya_spike/` | AI second readers and how each was evaluated |
| `compete/` | baselines we reproduced to measure against |
| `tests/`, `eval_hard.py` | app, injection and unseen-line tests; hard hold-out scenarios |
| `deck/` | pitch deck, built from sourced numbers (`numbers_used.json`) |
| `out/` | generated inventory, audit files, dashboard, screenshots |
| `data/` | the synthetic case data pack provided by the organisers |

## What we would do next

- Pilot on one real company's 1C export with milk data from its own farms (79 % of Dala Foods' footprint is raw milk; the case factor is illustrative and moves the total by about 16 %).
- Keep the gas and power baseline line by line, so next year's ledger measures what the green loan actually saved.
- Package it for banks: one lender's borrowers first, then exporters facing EU CBAM declarations.

What we learned on stage: the engine held up to every question, but we gave the judges too many numbers and too few moments to take part. The next pitch tells one accountant's story and lets the jury try to break the tool live.

## Team

Built by a team of four at CATALYST 2026. Engine, architecture and evaluations: [@abusanaevdias](https://github.com/abusanaevdias).

Data is synthetic: Dala Foods is a fictional company created for the case. Code under the [MIT License](LICENSE).
