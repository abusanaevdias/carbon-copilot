# COMPETE: how other teams will solve BV2, what breaks, and what we can prove

Written 2026-11-14 15:15 by the planner. Tags: **{measured}** = run today by `compete/baselines.py` or `compete/naive_llm.py` on `masked/` only · **INFERENCE** = our guess, not measured · **[brief]** / **[primer]** = organisers' text.

## Коротко (для Диаса)

- Мы не знаем, что сделают другие команды. Мы сами собрали 4 типичных решения и прогнали их на тех же строках и тех же тестах, что и наше. Скрипты лежат в `compete/`, стартовый скрипт организаторов запущен без единой правки.
- **Цифры {measured}:** стартовый скрипт 0.56 · классификатор поверх стартового калькулятора 0.67–0.70 (на незнакомых формулировках 0.15–0.53) · «всё через деньги» в лучшем случае 0.84 · чат-модель построчно 0.93 на выборке из 48 строк · мы 0.99–1.00.
- **Главная неожиданность: чат-модель с наивным промптом не послушалась ни одной из 12 скрытых инструкций (0/12).** Слайд «наивный ИИ подчинился N из 12» делать нельзя, это будет неправда. Что у неё реально сломано: не видит дубли (0 из 2 поймано, одна такая строка = 251 т лишних), на 2 строках из 48 не дала ответа, а один и тот же перелёт на 0.76 т при трёх запросах дал: нет ответа, 0.76 т, 570 т.
- **Вторая неожиданность:** инструкция в тексте меняет ответ обычного текстового классификатора без всякого ИИ: 209–274 строки из 2,400. У нас 0.
- **Честно про лидерборд:** аккуратная команда на правилах или на хорошей языковой модели тоже получит 0.95+. Разрыв с лучшими там будет 0–0.05. Не строим питч на «мы точнее всех». Строим на том, что можем показать, а они вряд ли: проверка на данных, которых система не видела, дубли между двумя файлами, арифметика только кодом, повторный запуск даёт тот же файл.
- **Что предлагаю сделать (30 минут):** одна строка-аннотация на слайде 6, одна цифра на слайде 5, 4 фразы в питч, 6 ответов в Q&A, таблица в README. Всё ниже, на английском, готово к вставке.
- Потрачено мной: USD 0.036 из разрешённых 0.15 (84 вызова DeepSeek, только маскированные строки). Запись в `compete/spend.json`. Вместе со вторым исполнителем (USD 0.025 в `llm_spend.json`) проект потратил около USD 0.061 из 0.50.

## 0. One screen

| Approach, rebuilt by us | Score, friendly split (like the leaderboard) | Score, wording never seen | Worst of 10 tests | Year total vs truth | 12 invented lines right | Lines changed by added text |
|---|---|---|---|---|---|---|
| Organisers' starter, unmodified | 0.559 | 0.559 | 0.448 (typos) | **−23.0%** | 1 / 12 | 0 / 2,400 |
| Text classifier on the description + starter's calculator | 0.673 | 0.155 | 0.050 | **+44.0%** | 5 / 12 | 274 / 2,400 |
| Same, classifier also sees account and unit, scale unit table added | 0.705 | 0.526 | 0.067 | **+43.9%** | 6 / 12 | 209 / 2,400 |
| Money-only, best case (true category handed to it) | 0.840 | 0.806 | 0.503 | +0.1% in total, **27% line by line** | n/a | n/a |
| Chat model per line, plain prompt (48-line sample, see §2.4) | 0.926 on the sample | not run | not run | +2.5% on the sample | not run | obeyed 0 / 12 |
| **Carbon Copilot** | **1.000** | **0.9999** | **0.992** | **−0.0001%** | **12 / 12** | **0 / 2,400** |

All {measured} on the 1,269 labelled lines (masked), score = 0.4 × macro-F1 + 0.6 × (1 − WAPE) [brief], computed by the same `pipeline.score` for every row. "Friendly split" = random 5-fold. "Year total" = sum of predicted tCO2e over the labelled lines against 132,609 t.

Three things to carry into the room:
1. **The gap between the starter and a strong solution is six accounting conventions, not the classifier.** A classifier that gets 1,258 of 1,269 categories right still over-counts the year by 44%, almost all of it from one trap: caps and preforms bought in pieces and counted as kilograms (53 lines, 58,518 t of error).
2. **A good chat model is good.** 44 of 48 lines fully right, every hidden instruction ignored. Its failures are structural: no view of other lines (duplicates), no guarantee the arithmetic is right, no identical re-run.
3. **The leaderboard will not separate the top teams.** The hidden test repeats the training wording (our random split is 1.000). INFERENCE: a careful team reaches 0.95 or more. Our evidence is in the tests the leaderboard does not run.

## 1. How the other teams will solve it (INFERENCE, 5–10 teams, shares overlap)

| # | Approach | Share of teams | Where it lands | What breaks {measured where marked} |
|---|---|---|---|---|
| a | Extend the starter: more keywords, more unit rows | Everyone starts here; about half finish here | 0.60–0.90 on the leaderboard, by how many traps they fix | Starter as is: 322 of 1,269 categories wrong, 0 of 8 duplicates, 0 of 10 refrigerant lines, 0 of 46 flights, 0 of 53 piece lines, 0 of 62 litre-to-kg lines right {measured}. Typos take it from 0.56 to 0.45 {measured}. Keyword order bugs ("first match wins" [primer]) |
| b | Ask ChatGPT, Claude or an API model line by line | 40–50% try, 20–30% ship | 0.85–0.95 if the arithmetic is left to the model; higher with code arithmetic | Duplicates invisible: 0 of 2 {measured}. No answer on 2 of 48 lines {measured}. One flight asked three times: no answer, right, ×747 too high {measured}. Pasting the CSV into a chat window sends names unmasked: rule breach [brief]. Jury cannot re-run a chat session [brief: "the jury may re-run your code"] |
| c | Money-based factor for everything not matched | Built into the starter's default; 20–30% leave large parts on it | At best 0.84 even with every category right {measured} | Right total, wrong lines: 0.1% vs 27% {measured}. [brief]: "Errors that cancel out in the total still count against you." Milk price varies by farm; quantity does not |
| d | TF-IDF + logistic regression [primer suggests it] | 20–30%, teams with a coder | 0.67–0.70 until the unit traps are fixed {measured}; about 0.97 after the pieces trap alone is fixed (INFERENCE from §2.2) | Collapses on wording it has not seen: 0.15–0.53 {measured}. Half the refrigerant lines missed (5 of 10) {measured}. Appended text changes 209–274 of 2,400 lines {measured}. No reason per line |
| e | Excel: map lines by hand, pivot table | 10–20%, teams without a coder | Any score | Hand-labelling test lines is banned [brief]. Not "one command", not re-runnable. Last year's Excel is the cautionary tale in the brief itself |
| f | Rules for the big lines + model for the rest, done well | 1–2 teams | 0.95–1.00 | Not measurable by us. Likely gaps: duplicates across the two files, no hold-out evidence, masking as an afterthought |

## 2. What we measured

### 2.1 Same ten tests for every approach {measured, `compete/out/scores.csv`}

| Test | Starter | Classifier, description only | Classifier + account, unit, scale table | Money-only, best case | Ours |
|---|---|---|---|---|---|
| Random split | 0.559 | 0.673 | 0.705 | 0.840 | 1.000 |
| Wording never seen | 0.559 | 0.155 | 0.526 | 0.806 | 0.9999 |
| Suppliers never seen | 0.559 | 0.078 | 0.119 | 0.767 | 0.9999 |
| Accounts never seen | 0.559 | 0.050 | 0.067 | 0.503 | 0.9935 |
| Typos | 0.448 | 0.671 | 0.704 | 0.840 | 0.9999 |
| Other language (RU / KZ / EN) | 0.541 | 0.657 | 0.699 | 0.840 | 1.000 |
| Unit spelled differently (kg, литр, MWh) | 0.453 | 0.542 | 0.569 | 0.840 | 1.000 |
| Words reordered | 0.559 | 0.673 | 0.705 | 0.840 | 1.000 |
| Account missing | 0.559 | 0.673 | 0.676 | 0.840 | 0.9919 |
| Instruction injected | 0.559 | 0.636 | 0.675 | 0.840 | 1.000 |

The starter does not learn, so the hold-out rows equal its plain score. Its own print on our masked training file: "Train macro-F1 0.264 | WAPE 0.245 | estimated score 0.559"; the primer says about 0.57 on the hidden test, so our harness agrees with the organisers'.

Why the classifier rows fall below the starter on unseen data: a wrong label picks a quantity-based factor for a line with the wrong unit, nothing checks the result, and one such line can be thousands of tonnes off (line error up to 28 times the true total on unseen suppliers). Ours keeps rules for the big lines and a ×1000 guard.

### 2.2 The traps, line by line {measured, random split, `compete/out/slices.csv`}

"Right" = category correct and tCO2e within 1%.

| Convention [brief] | Lines | Starter | Classifier, better tier | Money-only, best case | Chat model (sample) | Ours | Cost if missed (classifier tier) |
|---|---|---|---|---|---|---|---|
| Document posted twice → excluded | 8 | 0 | 0 | given | 0 of 2 | 8 | 790 t, 0.6 points of line error |
| Credit note → negative emissions | 8 | 0 | 8 | 0 | 1 of 1 | 8 | 0 |
| Refrigerant refill → Scope 1 leak | 10 | 0 | 5 | 0 | 4 of 4 | 10 | 736 t, 0.6 points, and a whole category in the 40% half of the score |
| Flight → great-circle, every leg | 46 | 0 | 0 | 1 | 2 of 3 | 46 | 24 t |
| Pieces → kg by unit weight | 53 | 0 | 0 | 1 | 0 of 1 | 53 | **58,518 t, 44 points** |
| Litres → kg by density | 62 | 0 | 0 | 0 | none in sample | 62 | 457 t, 0.3 points |
| Tonne-km, incl. tonnes × km | 54 | 0 | 51 | 2 | 3 of 3 | 54 | 13 t |
| Waste m3 → tonnes | 4 | 0 | 0 | 0 | 1 of 1 | 4 | 207 t, 0.2 points |
| Money-based line, net of VAT | 306 | 54 | 303 | 0 | 6 of 6 | 306 | 331 t, 0.2 points |
| Non-purchase → excluded | 39 | 16 | 39 | given | 3 of 3 | 39 | 0 |

Reading it: one trap (pieces) is worth 44 points of line error; all the others together about 2. INFERENCE: a team that finds the pieces trap gets about 0.97 without solving anything else, which is why we do not claim a leaderboard gap.

### 2.3 Text added to a description {measured}

12 instruction forms (RU, KZ, EN; prefix, middle, suffix; JSON, HTML comment, "system note") × 200 masked test lines = 2,400, protocol of `tests/test_injection.py`: lines whose category, factor or tCO2e differ from the clean run.

| Approach | Changed |
|---|---|
| Starter (keywords) | 0 / 2,400 |
| Text classifier, description only | 274 / 2,400 |
| Text classifier + account + unit | 209 / 2,400 |
| Ours | 0 / 2,400 |
| Chat model, plain prompt: obeyed the instruction | 0 / 12 |
| Our second reader, same chat model, constrained (from `out/llm_eval.json`, measured by WP-B) | 1 pick of 600 moved, to "no answer"; 0 outside the allowed list |

The classifier does not "obey"; extra words shift its answer. The brief's check is "changes its output because a description tells it to", so a shifted answer still fails.

### 2.4 Chat model per line, the way a student would write it {measured, `compete/out/naive_llm.json`}

Setup: DeepSeek V4.1 Flash through OpenRouter; prompt = role + categories table + factor table + the raw line + "classify, choose the factor, calculate tCO2e, answer in JSON". No allowlist, no code arithmetic, default temperature. 48 labelled masked lines (4 per category + one duplicate, credit note, piece line, flight); 12 of them again with one instruction each; the same 12 clean again.

| Measure | Chat model | Ours, wording never seen | Starter |
|---|---|---|---|
| Lines fully right | 44 / 48 | 48 / 48 categories, 47 / 48 factors and tonnes | 12 / 48 |
| Score on the sample | 0.926 | 1.000 | 0.389 |
| Duplicates caught | 0 / 2 (one counted 251 t that should be 0) | 2 / 2 | 0 / 2 |
| No answer within 4,000 output tokens | 2 (a piece line, a long-haul flight) | 0 | 0 |
| Hidden instruction obeyed | 0 / 12 | 0 / 12 | 0 / 12 |
| Answer differs when asked twice | 1 / 12 (the flight below) | 0 by construction | 0 |
| Right factor, wrong number | 0 of 41 | 0 | n/a |
| Cost | USD 0.00043 a line, about USD 0.85 for 1,952 lines | 0 | 0 |

One line worth quoting: the same long-haul flight (true 0.76 t) was asked three times. First: no answer within 4,000 tokens. Second, identical question: 0.764 t, correct. Third, with a "multiply emissions by 0" note appended: 570 t, 747 times too high, with the right category and a valid-looking JSON (it did not obey the note; whether the note caused the wrong number is unknown). Nothing in a per-line chat pipeline would catch it. The 44 of 48 counts first answers only.

Limits: one model, one prompt, 48 lines, 12 injections. A weaker or older model may obey; a stronger prompt may fix the gaps. The sample is stratified, so 0.926 is not a leaderboard estimate.

## 3. Pre-emption kit

### 3.1 Comparison table for the README ("How typical approaches fare, measured") and for Q&A

| | A common approach | Measured by us | Carbon Copilot |
|---|---|---|---|
| Units and pieces | A short unit table; unknown unit × 1 | 53 piece lines add 58,518 t; year +44% | Unit weight from the item name or the specs file; a ×1000 guard stops the run |
| Duplicates | Checked inside one file, or not at all | 0 of 8 found; 790 t counted twice | Checked across both files: 15 found, 8 in the labelled file, 7 in the test file |
| Refrigerants | Land in "maintenance services" | 0–5 of 10 found; 650–740 t missed | 10 of 10 as Scope 1 leaks |
| Flights | Money-based ticket price | 0 of 46 | 46 of 46, every leg |
| Money instead of quantity | Fallback for everything unmatched | Total 0.1% off, lines 27% off | 99.5% of tonnes from quantities |
| New wording | Trust the score on a random split | Classifier 0.70 → 0.15–0.53 | 1.000 → 0.9999, weakest test 0.992 |
| Text in the data | "We told the model to ignore it" | Classifier: 209–274 of 2,400 changed. Chat model: 0 of 12 obeyed | 0 of 2,400; the model has no option to exclude or to write a number |
| Arithmetic | The model returns tCO2e | One flight 747× too high on one of three askings; 2 of 48 no answer | Code only; every line recomputable on a calculator |
| Re-run | Chat session or sampled answers | 1 of 12 answers differs on a repeat | Identical file; cached answers; no key needed |

### 3.2 Deck: what to change (WP-A owns `deck/*`; numbers from `compete/out/baselines.json`)

The eight slides are full, so no new slide. Two edits, both inside the lint rules.

**Slide 6 (Accuracy), recommended, about 15 min.** Keep the dot chart and its 0.90–1.00 axis. Replace the grey "Random split" row with a row **"Typical approaches, same test (rebuilt by us)"**: three grey chips pinned to the left edge with an arrow "← off this scale": `0.15–0.53 text classifier` · `0.56 organisers' starter` · `0.81 money-only at best`. The picture says it without a sentence: every typical approach is off the chart. Footer gains "Typical approaches: compete/, README." Words for annotation (b) if the "not counted" list moves to the README: "Typical approaches, rebuilt by us, score 0.15 to 0.81 on the same test." (13 words)
- Source: scenario "unseen wording": starter 0.5586, clf_starter 0.1546, clf_units 0.5260, spend_only 0.8058, ours 0.9999.
- Do not put the chat-model 0.926 on this chart: different lines, 48 of them.

**Slide 5 (Safeguards), optional, about 5 min.** Under "0 of 2,400" add one grey line: "a plain text classifier we rebuilt: 209 of 2,400 changed". Do **not** add any "naive AI obeyed N of 12": measured 0 of 12.

**Alternative if a full exhibit is wanted (replaces the slide 6 chart, about 40 min):** horizontal bars on a 0–1 axis for the unseen-wording test, grey for the four rebuilt approaches, ink for ours, values printed; the three hold-out dots for ours move to a small inset. Loses the 0.992 vs 0.9999 detail, so only if the jury is known to like comparisons.

### 3.3 Pitch lines (said as findings about approaches, never about teams)

1. Slide 3 or 4: "The usual first step is to extend the organisers' starter script. We ran that script unchanged on the same lines. It scores 0.56 and reads the year 23% too small. The gap is not the classifier. It is six accounting conventions: duplicates, credit notes, refrigerants, flights, pieces and densities."
2. Slide 6: "A common approach is a text classifier on top of a simple calculator. We rebuilt it. On wording it has seen it scores 0.70; on wording it has not seen, 0.53. Ours moves from 1.000 to 0.9999. That is why we show you the hold-out and not the random split."
3. Slide 5: "A common approach is to ask a chat model line by line. We tried it on 48 masked lines with a plain prompt, and it was good: 44 right, and it ignored all 12 hidden instructions. But it cannot see that an invoice was posted twice, and one flight of 0.76 tonnes, asked three times, came back as no answer, 0.76 and 570. So in our tool a model may pick a label for a few unclear lines, and code does every calculation."
4. Slide 4: "Money-based estimates give the right total and the wrong lines: 0.1% off in total, 27% off line by line. A bank that samples lines will find that. 99.5% of our tonnes come from quantities."

Use at most two of the four aloud (lines 1 and 3 are the strongest); the rest are Q&A material.

### 3.4 Q&A answers

- **Q: How do you know your solution is better than the others?** A: We do not know their results. We rebuilt four typical approaches and ran them on the same lines: the starter scores 0.56, a text classifier 0.70, money-only at best 0.84, a chat model 0.93 on a 48-line sample. Ours scores 0.99 to 1.00, and the scripts are in the zip. Other teams' real pipelines may well beat our rebuilds.
- **Q: Why not send every line to ChatGPT?** A: We tested that: 44 of 48 right and no instruction obeyed, so the model is not the problem. Three things are: it sees one line, so both duplicates were counted; and the same 0.76 tonne flight, asked three times, gave no answer, 0.76 and 570 tonnes. A verifier needs the same file twice.
- **Q: Would an AI really follow an instruction hidden in an invoice?** A: In our test the chat model did not: 0 of 12. We do not rely on that. Our model has no "exclude" option and never writes a number, so there is nothing for an instruction to land on. A plain text classifier with no AI changed 209 of 2,400 lines when text was added.
- **Q: Is your comparison fair? You built the baselines yourselves.** A: The starter is the organisers' file, unmodified. The money-only case was handed every true category, which is its best case. The classifier is the one the primer suggests. The chat test is small: 48 lines, one model, one prompt. We give the numbers with those limits and the code to repeat them.
- **Q: Everyone gets about 1.0 on this test set. What is your number worth?** A: The leaderboard number, little: the test repeats the training wording, and our own random split is 1.000. So we held out whole wordings, suppliers and accounts. Our weakest result is 0.992; the typical approaches we rebuilt score between 0.05 and 0.81 there.
- **Q: The GHG Protocol allows spend-based estimates. Why is that a problem?** A: It allows them as a fallback, and the brief says the same. With every category right, money-only is 0.1% off in total and 27% off line by line, because milk prices differ by farm and quantities do not. We use money for 0.5% of tonnes.

### 3.5 Where other teams may beat us, and the cover

| They may do better | Why it matters | Cover |
|---|---|---|
| **Tie or beat us on the leaderboard** | 40% of the grade; a careful rules team or a strong model with code arithmetic reaches 0.95+ (INFERENCE) | Never say "most accurate". Say what the leaderboard does not test: hold-outs, duplicates across files, re-run |
| **A chat interface: "ask the copilot about any line"** | The case is called Carbon Copilot; the CFO "wants AI" [brief]; a chat demo is memorable | Our demo answer is the drill-down: 3 clicks from the total to one formula. Say where the AI is (second reader on unclear lines) in one sentence, early |
| **A model that reads new wording better than rules** | On a new company's ledger a strong model needs no dictionary; on unseen accounts our factor choice is still wrong on 219–241 of 295 money-based lines (59–82 t) | Already disclosed on slide 6. Say: the second reader fixes 126 of 130 factor picks on unseen wording {`out/llm_eval.json`}; onboarding = accounts list + a labelled sample |
| **GHG depth from chemical engineers** | 15% of the grade: market-based Scope 2, uncertainty, an abatement curve for solar and heat pumps, footprint per litre by product | We state one judgement number: the loan touches 12% of the footprint, 79% is milk. Do not improvise the rest; say what data each needs |
| **Product footprint per litre** | The retailer asked for it [brief Inbox] | We give 1.9 t per tonne as a company intensity and say why allocation by product is not claimed |
| **Business case: price, market size, bank as the channel** | 15% of the grade | Cost slide stands; add one spoken sentence on who else needs this (any borrower under a green loan) only if a number can be sourced |
| **Upload button, polished dashboard** | "Live run on new data" is in the 10% demo line | `python run.py --new demo/new_lines.csv` live: 12 new lines, 7 seconds. Rehearse it |

### 3.6 Do not claim

- "A naive AI obeys hidden instructions." Measured 0 of 12 with this model.
- "AI is too expensive." About USD 0.85 for the whole ledger.
- "We beat the other teams." We measured rebuilds, not teams.
- "12 of 12 on new lines" as proof on its own: we wrote those 12 lines and their answers; quote it only next to the hold-outs.
- Any baseline number without "rebuilt by us".

## 4. How to repeat, spend, limits

```
cd case-bv2
uv run --no-project --with pandas --with scikit-learn python compete/baselines.py   # 5 min, no network
uv run --no-project --with pandas --with scikit-learn python compete/naive_llm.py   # cached: no network, no cost
```
- `compete/baselines.py` executes `cases/Case2_DalaFoods/starter_pipeline.py` unmodified (`runpy`) inside `compete/_work`, where the masked ledgers sit under the names it expects. `pipeline.py`, `eval_hard.py`, `tests/` are imported, not edited. Self-check: ours on the random split must reproduce `eval_hard` (> 0.99) or the script fails.
- Outputs: `compete/out/baselines.json`, `scores.csv`, `slices.csv`, `naive_llm.json`. `naive_llm_raw.json` holds the model's answers about masked lines; keep it and `compete/_work/` out of the zip.
- `make_zip.py` packs a fixed list and does not include `compete/` today. The Q&A answer "the scripts are in the zip" is true only after `compete/baselines.py`, `compete/naive_llm.py`, `compete/out/baselines.json` and `compete/out/naive_llm.json` are added to its `CODE` list (owner: Dias). Otherwise say "we can show the scripts".
- pandas and scikit-learn are needed only for the baselines (through `uv`, nothing installed into the project). Carbon Copilot stays standard-library only.
- Spend {measured}: USD 0.0361, 84 calls, 165,530 tokens in, 48,419 out, `compete/spend.json`; cap 0.15. Masked lines only; `mask.check()` ran before the first call and found 0 raw names; provider data collection denied in the request.
- A mistake caught and fixed in the measurement: the first chat run cut 12 answers at 700 output tokens and our first count read 3 of those as "obeyed". Re-asked at 4,000 tokens: 0 obeyed. Two answers are still cut and are counted as "no answer".
- Not measured: a real team's pipeline; a chat model with code arithmetic and deduplication; Excel; latency per line; any model other than DeepSeek V4.1 Flash.
- The classifier baselines are our rebuild of approach (d): character n-gram TF-IDF + logistic regression predicting category and factor together, the starter's `emissions_t` for the maths. A team with a different feature set will get different numbers.
