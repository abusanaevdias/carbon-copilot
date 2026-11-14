"""Local masking of Dala Foods ledgers before any cloud-AI use (stdlib only).

Run:  python mask.py [path/to/Case2_DalaFoods] [--new extra.csv]   mask raw ledgers -> masked/
      python mask.py --check [path/to/Case2_DalaFoods] [--new extra.csv]  re-scan masked/ on disk, exit 1 on any leak
Writes masked/ledger_train.csv, masked/ledger_test.csv (+ ledger_new.csv), mask_map.csv (LOCAL ONLY, never share).
Prints counts only. Never prints names or rows.
"""
import csv, hashlib, os, re, secrets, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SRC = os.path.join(HERE, "..", "cases", "Case2_DalaFoods")
OUT = os.path.join(HERE, "masked")
SALT_FILE = os.path.join(HERE, ".mask_salt")
MAP_FILE = os.path.join(HERE, "mask_map.csv")
FREE_TEXT = ["description", "cost_center"]

# legal-form tokens (kept in the pseudonym: non-identifying, useful signal)
LEGAL = ["ТОО", "ИП", "КХ", "ФХ", "АО", "АҚ", "ЖШС", "ЖК", "ШҚ", "ГКП", "РГП", "КГП", "ООО",
         "LLP", "LLC", "JSC", "Ltd", "GmbH", "IE", "PE"]
LEGAL_RE = re.compile(r"(?<![\w])(" + "|".join(LEGAL) + r")(?![\w])", re.I)

# keyword flags computed from the ORIGINAL counterparty, emitted as non-identifying features
CP_KW = [
    ("milk",      ["сүт", "сут", "молок", "milk", "dairy", "агро", "agro", "ферм", "farm", "шаруа"]),
    ("fuel",      ["petro", "петро", "ойл", "oil", "газ", "gas", "азс", "fuel", "мұнай", "kazmunay", "helios", "sinooil"]),
    ("energy",    ["energy", "энерг", "электр", "жылу", "тепло", "heat", "power", "сбыт", "aes", "алес"]),
    ("pack",      ["pack", "пак", "тара", "пласт", "plast", "pet", "картон", "carton", "tetra", "гофр"]),
    ("transport", ["trans", "транс", "logist", "логист", "cargo", "карго", "жол", "rail", "темір", "ктж", "express"]),
    ("travel",    ["air", "авиа", "эйр", "hotel", "отел", "қонақ", "taxi", "такси", "yandex", "яндекс", "travel", "тур"]),
    ("chem",      ["chem", "хим", "ecolab", "clean", "клин"]),
    ("it",        ["soft", "софт", "it ", "tech", "digital", "1с", "1c", "telecom", "телеком", "beeline", "kcell", "связь"]),
    ("finance",   ["bank", "банк", "insur", "страх", "kaspi", "halyk", "сақтандыру"]),
    ("gov",       ["налог", "казначей", "мемлекет", "государ", "комитет", "департамент", "гос", "акимат", "кгд", "салық"]),
    ("build",     ["строй", "құрылыс", "build", "construct", "монтаж"]),
    ("water",     ["вод", "су ", "water", "канал", "сток"]),
    ("waste",     ["waste", "отход", "мусор", "тбо", "эко", "eco", "қалдық"]),
    ("refrig",    ["холод", "frio", "cool", "refri", "фреон", "климат", "суық"]),
    ("auto",      ["авто", "auto", "motor", "моторс", "truck", "камаз", "toyota", "hyundai"]),
]

U = "А-ЯЁӘӨҮҰҚҒҢҺІA-Z"          # upper letters (Cyrillic incl. Kazakh + Latin)
L = "а-яёәөүұқғңһіa-z"          # lower letters
W = f"[{U}][{L}]+(?:-[{U}]?[{L}]+)?"   # capitalised word, optional hyphen part
INI = f"[{U}]\\.\\s?(?:[{U}]\\.)?"       # И. or И.О.
PATRO = r"(?:вич|вна|ұлы|улы|қызы|кызы|ovich|evich|ovna|evna|uly|kyzy)"
NAME_PATTERNS = [
    ("surname_initials", re.compile(f"(?<![\\w.]){W}\\s+{INI}")),                 # Иванов И.О.
    ("initials_surname", re.compile(f"(?<![\\w.]){INI}\\s?{W}")),                 # И.О. Иванов
    ("full_fio",         re.compile(f"(?<!\\w)(?:{W}\\s+){{1,2}}[{U}][{L}]+{PATRO}(?!\\w)")),  # Ф И Отчество / Имя Отчество
    ("fio_reverse",      re.compile(f"(?<!\\w)[{U}][{L}]+{PATRO}\\s+{W}(?!\\w)")),
    ("legal_form_name",  re.compile(                                          # ИП Фамилия / ТОО «Название»
        r"(?<![\w])(?:" + "|".join(LEGAL) + r")\s+((?:[\"«“]?" + W + r"[\"»”]?)(?:\s+(?:" + W + "|" + INI + r"))?)")),
    ("keyword_name",     re.compile(
        r"(?i:сотрудник[а-я]*|работник[а-я]*|employee|staff|traveller|traveler|passenger|пассажир[а-я]*|"
        r"подотчет[а-я ]*лиц[оа]|подотчётн[а-я ]*лиц[оа]|фио|ф\.и\.о\.|получател[ья]|командированн[а-я]+|"
        r"қызметкер[а-я]*|перевозчик[а-я]*|поставщик[а-я]*|contractor|carrier|жолаушы|водител[ья]|driver|for|для|на имя)"
        f"[:\\s]+((?:{W})(?:\\s+(?:{W}|{INI}))?(?:\\s+{W})?)")),
]
# words after a trigger that are NOT names (avoid masking "для Цеха", "for Plant")
NOT_NAME = re.compile(r"(?i)^(цех|склад|офис|завод|plant|office|warehouse|production|admin|отдел|"
                      r"филиал|производств|сырь|транспорт|грузовик|машин|авто|котельн|линии|линия|"
                      r"молок|упаков|alatau|алматы|астана|almaty|astana|шымкент|караганд|dala|дала)")


def pseudo(salt, name):
    h = hashlib.sha256((salt + "|" + name.strip().lower()).encode()).hexdigest()[:8]
    return h


def cp_features(cp):
    low = (cp or "").lower() + " "
    m = LEGAL_RE.search(cp or "")
    form = m.group(1).upper() if m else ""
    kws = [k for k, words in CP_KW if any(w in low for w in words)]
    return form, "|".join(kws)


def core_of(cp):
    s = LEGAL_RE.sub(" ", cp or "")
    s = re.sub(r"[\"'«»“”„()]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def mask_text(text, cores, counts, person_map, salt):
    if not text:
        return text
    for core, token in cores:                  # counterparty names quoted inside free text
        if core.lower() in text.lower():
            text = re.sub(re.escape(core), token, text, flags=re.I)
            counts["cp_in_text"] = counts.get("cp_in_text", 0) + 1
    for label, rx in NAME_PATTERNS:
        def rep(m, label=label):
            span = m.group(1) if label in ("keyword_name", "legal_form_name") else m.group(0)
            if label == "keyword_name" and NOT_NAME.match(span):
                return m.group(0)
            if span.startswith(("CP_", "PERSON_")):
                return m.group(0)
            tok = person_map.setdefault(span, "PERSON_" + pseudo(salt, "p:" + span))
            counts[label] = counts.get(label, 0) + 1
            return m.group(0).replace(span, tok)
        text = rx.sub(rep, text)
    return text


def mask_rows(data, salt):
    """data = {file name: (fields, raw rows)} -> masked data, maps, counts. Pure: reads and prints nothing."""
    cps = sorted({row["counterparty"] for _, rows in data.values() for row in rows if row["counterparty"]})
    cp_map = {}
    for cp in cps:
        form, _ = cp_features(cp)
        cp_map[cp] = (form + " " if form else "") + "CP_" + pseudo(salt, cp)
    cores = sorted({(core_of(cp), cp_map[cp]) for cp in cps if len(core_of(cp)) >= 4},
                   key=lambda t: -len(t[0]))
    counts, person_map, masked = {}, {}, {}
    for name, (fields, rows) in data.items():
        out_fields = fields[:3] + ["counterparty", "cp_legal_form", "cp_kw"] + fields[4:]
        out_rows = []
        for row in rows:
            o = dict(row)
            cp = row["counterparty"]
            o["counterparty"] = cp_map.get(cp, "")
            o["cp_legal_form"], o["cp_kw"] = cp_features(cp)
            for col in FREE_TEXT:
                if col in o:
                    o[col] = mask_text(o[col], cores, counts, person_map, salt)
            out_rows.append(o)
        masked[name] = (out_fields, out_rows)
    return masked, cp_map, person_map, counts, cores


def leaks(blob, cp_map, cores, person_map):
    """Count raw names left in a lower-cased text blob. Returns counts only."""
    return dict(counterparty=sum(1 for cp in cp_map if cp.lower() in blob),
                core=sum(1 for c, _ in cores if c.lower() in blob),
                person=sum(1 for p in person_map if p.lower() in blob),
                patronymic_left=len(re.findall(f"[{L}]+{PATRO}(?![{L}])", blob)))


def read_raw(src, new=None):
    data = {}
    files = [("ledger_train.csv", os.path.join(src, "ledger_train.csv")), ("ledger_test.csv", os.path.join(src, "ledger_test.csv"))]
    if new:
        files.append(("ledger_new.csv", new))
    for name, path in files:
        with open(path, encoding="utf-8-sig", newline="") as f:
            r = csv.DictReader(f)
            data[name] = (r.fieldnames, list(r))
    return data


def salt_value():
    if not os.path.exists(SALT_FILE):
        with open(SALT_FILE, "w") as f:
            f.write(secrets.token_hex(16))
    return open(SALT_FILE).read().strip()


def check(src=DEFAULT_SRC, new=None, out=OUT):
    """Re-scan the masked files ON DISK for any raw name. Returns the leak counts; caller decides the exit code."""
    _, cp_map, person_map, _, cores = mask_rows(read_raw(src, new), salt_value())
    names = ["ledger_train.csv", "ledger_test.csv"] + (["ledger_new.csv"] if new else [])
    blob = "\n".join(open(os.path.join(out, n), encoding="utf-8-sig").read() for n in names).lower()
    return leaks(blob, cp_map, cores, person_map)


def main(src=DEFAULT_SRC, new=None, out=OUT):
    salt = salt_value()
    os.makedirs(out, exist_ok=True)
    data = read_raw(src, new)
    masked, cp_map, person_map, counts, cores = mask_rows(data, salt)
    for name, (_, rows) in data.items():
        print(f"{name}: rows={len(rows)}")

    # ---- self-check (in memory, prints nothing sensitive) ----
    blob = "\n".join("\t".join(str(r.get(k, "")) for k in f) for f, rows in masked.values() for r in rows).lower()
    lk = leaks(blob, cp_map, cores, person_map)
    print(f"unique counterparties masked={len(cp_map)} | person spans masked={len(person_map)}")
    print("masked per pattern:", dict(sorted(counts.items())))
    print("self-check leaks:", " ".join(f"{k}={v}" for k, v in lk.items()))
    assert not any(lk.values()), "MASKING SELF-CHECK FAILED"

    stale = os.path.join(out, "ledger_new.csv")
    if not new and os.path.exists(stale):
        os.remove(stale)
    for name, (fields, rows) in masked.items():
        with open(os.path.join(out, name), "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    with open(MAP_FILE, "w", encoding="utf-8-sig", newline="") as f:   # LOCAL ONLY
        w = csv.writer(f)
        w.writerow(["kind", "original", "pseudonym"])
        w.writerows([("counterparty", k, v) for k, v in cp_map.items()])
        w.writerows([("person", k, v) for k, v in person_map.items()])
    print("SELF-CHECK OK; masked files written")
    return dict(counterparties=len(cp_map), persons=len(person_map), leaks=sum(lk.values()))


def _cli(argv):
    new = argv[argv.index("--new") + 1] if "--new" in argv else None
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] != "--new")]
    src = pos[0] if pos else DEFAULT_SRC
    if "--check" in argv:
        lk = check(src, new)
        print("disk leak check:", " ".join(f"{k}={v}" for k, v in lk.items()))
        sys.exit(1 if any(lk.values()) else 0)
    main(src, new)


if __name__ == "__main__":
    _cli(sys.argv[1:])
