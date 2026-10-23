"""
Phase 1 — Data collection.

Four public sources (Table 7.1 of the proposal):
  1. NHTSA customer complaints   -> api.nhtsa.gov  (symptom narratives)
  2. Automotive diagnostic Q&A   -> Kaggle CSV / HF dataset (query-solution pairs)
  3. NHTSA safety recalls        -> api.nhtsa.gov  (defect / consequence / remedy)
  4. Technical manuals           -> public PDFs / text (procedural knowledge)

Every loader returns a list[dict] with a common schema:
  {"id", "source", "text", "meta": {...}}
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

NHTSA_BASE = "https://api.nhtsa.gov"

# A spread of popular models so the corpus covers several makes/components.
DEFAULT_VEHICLES = [
    ("honda", "civic", 2016), ("honda", "accord", 2018), ("honda", "cr-v", 2019),
    ("toyota", "camry", 2017), ("toyota", "corolla", 2019), ("toyota", "rav4", 2019),
    ("ford", "f-150", 2018), ("ford", "escape", 2017), ("ford", "focus", 2014),
    ("chevrolet", "silverado", 2019), ("chevrolet", "equinox", 2018), ("chevrolet", "malibu", 2016),
    ("nissan", "altima", 2015), ("nissan", "rogue", 2018),
    ("hyundai", "sonata", 2015), ("hyundai", "elantra", 2017), ("kia", "optima", 2016),
    ("jeep", "cherokee", 2015), ("jeep", "wrangler", 2018), ("subaru", "outback", 2016),
    ("volkswagen", "jetta", 2015), ("bmw", "3-series", 2016), ("tesla", "model 3", 2019),
]


def _hid(*parts) -> str:
    return hashlib.md5("|".join(str(p) for p in parts).encode()).hexdigest()[:12]


def _get(url: str, params: dict | None = None, retries: int = 3, sleep: float = 0.4) -> dict:
    import requests  # local import so the module imports without network deps

    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(sleep * (attempt + 1))
    return {}


# ---------------------------------------------------------------------------
# 1. NHTSA complaints
# ---------------------------------------------------------------------------
def fetch_nhtsa_complaints(vehicles: Iterable[tuple[str, str, int]] = DEFAULT_VEHICLES,
                           max_per_vehicle: int = 150, cache_dir: str | Path = "data/raw") -> list[dict]:
    cache = Path(cache_dir) / "nhtsa_complaints.json"
    if cache.exists():
        return json.loads(cache.read_text())

    docs: list[dict] = []
    for make, model, year in vehicles:
        js = _get(f"{NHTSA_BASE}/complaints/complaintsByVehicle",
                  {"make": make, "model": model, "modelYear": year})
        for row in (js.get("results") or [])[:max_per_vehicle]:
            summary = (row.get("summary") or "").strip()
            if len(summary) < 40:
                continue
            comps = row.get("components") or "UNKNOWN"
            docs.append({
                "id": f"cmp_{row.get('odiNumber', _hid(summary))}",
                "source": "complaint",
                "text": (f"Vehicle: {year} {make.title()} {model.title()}. Component: {comps}. "
                         f"Crash: {row.get('crash')}. Fire: {row.get('fire')}. "
                         f"Complaint: {summary}"),
                "meta": {"make": make, "model": model, "year": year,
                         "component": comps.split(":")[0].strip().lower(),
                         "severity": "high" if (row.get("crash") or row.get("fire") or row.get("numberOfInjuries", 0)) else "normal",
                         "date": row.get("dateOfIncident")},
            })
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(docs))
    return docs


# ---------------------------------------------------------------------------
# 2. NHTSA recalls
# ---------------------------------------------------------------------------
def fetch_nhtsa_recalls(vehicles: Iterable[tuple[str, str, int]] = DEFAULT_VEHICLES,
                        cache_dir: str | Path = "data/raw") -> list[dict]:
    cache = Path(cache_dir) / "nhtsa_recalls.json"
    if cache.exists():
        return json.loads(cache.read_text())

    docs: list[dict] = []
    for make, model, year in vehicles:
        js = _get(f"{NHTSA_BASE}/recalls/recallsByVehicle",
                  {"make": make, "model": model, "modelYear": year})
        for row in js.get("results") or []:
            text = (f"Vehicle: {year} {make.title()} {model.title()}. Recall {row.get('NHTSACampaignNumber')}. "
                    f"Component: {row.get('Component')}. Defect: {row.get('Summary')} "
                    f"Consequence: {row.get('Consequence')} Remedy: {row.get('Remedy')}")
            docs.append({
                "id": f"rcl_{row.get('NHTSACampaignNumber', _hid(text))}_{_hid(make, model, year)}",
                "source": "recall",
                "text": text,
                "meta": {"make": make, "model": model, "year": year,
                         "component": (row.get("Component") or "unknown").split(":")[0].strip().lower(),
                         "severity": "high", "date": row.get("ReportReceivedDate")},
            })
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(docs))
    return docs


# ---------------------------------------------------------------------------
# 3. Diagnostic Q&A
# ---------------------------------------------------------------------------
def load_diagnostic_qa(path: str | Path | None = None, max_rows: int = 3000) -> list[dict]:
    """
    Loads a question/answer CSV. Works with the Kaggle "automotive diagnostic Q&A"
    style files — it auto-detects the question and answer columns. If no file is
    given it falls back to a Hugging Face dataset download (requires internet).
    """
    df: pd.DataFrame | None = None
    if path and Path(path).exists():
        df = pd.read_csv(path) if str(path).endswith(".csv") else pd.read_json(path, lines=str(path).endswith(".jsonl"))
    else:
        df = _load_public_qa()

    cols = {c.lower(): c for c in df.columns}
    qcol = next((cols[c] for c in cols if any(k in c for k in ("question", "instruction", "input", "query", "prompt"))), df.columns[0])
    acol = next((cols[c] for c in cols if any(k in c for k in ("answer", "response", "output", "solution"))), df.columns[-1])
    df = df[[qcol, acol]].dropna().drop_duplicates().head(max_rows)

    docs = []
    for i, (q, a) in enumerate(zip(df[qcol], df[acol])):
        q, a = str(q).strip(), str(a).strip()
        if len(a) < 30:
            continue
        docs.append({
            "id": f"qa_{_hid(q)}",
            "source": "qa",
            "text": f"Question: {q}\nAnswer: {a}",
            "meta": {"make": None, "model": None, "year": None,
                     "component": _guess_component(q + " " + a), "severity": "normal", "date": None},
        })
    return docs


# ---------------------------------------------------------------------------
# 4. Technical manuals
# ---------------------------------------------------------------------------
def load_manuals(folder: str | Path = "data/manuals") -> list[dict]:
    """
    Reads every .pdf / .txt / .md in `folder`. Put public service/owner manuals here
    (e.g. NHTSA-hosted manuals, manufacturer PDFs released publicly).
    """
    folder = Path(folder)
    docs = []
    if not folder.exists():
        return docs
    for f in sorted(folder.iterdir()):
        text = ""
        if f.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader
                text = "\n".join((p.extract_text() or "") for p in PdfReader(str(f)).pages)
            except Exception:
                continue
        elif f.suffix.lower() in (".txt", ".md"):
            text = f.read_text(errors="ignore")
        if len(text) < 200:
            continue
        docs.append({
            "id": f"man_{_hid(f.name)}",
            "source": "manual",
            "text": text,
            "meta": {"make": _guess_make(f.name), "model": None, "year": None,
                     "component": None, "severity": "normal", "date": None, "file": f.name},
        })
    return docs


def _load_public_qa() -> pd.DataFrame:
    """
    Diagnostic Q&A from two public Hugging Face datasets:
      * Ichlibitiche/mechanicdb-obd2-repair-sample (config dtc_fixes_joined): DTC code,
        technical explanation and ranked repair procedures
      * Epitech/obd-codes-fine-tune: instruction/output pairs interpreting OBD-II codes
    Manufacturer-specific codes with conflicting meanings are dropped.
    """
    from datasets import load_dataset
    rows = []
    try:
        ds = load_dataset("Ichlibitiche/mechanicdb-obd2-repair-sample", "dtc_fixes_joined", split="train")
        for r in ds:
            code = r.get("dtc_code") or ""
            desc = r.get("short_description") or ""
            expl = r.get("detailed_technical_explanation") or ""
            fix = r.get("fix_title") or ""
            steps = r.get("step_by_step") or r.get("step_by_step_procedure") or ""
            rows.append({"question": f"My scan tool shows {code} ({desc}). What does it mean and how do I fix it?",
                         "answer": f"{expl} Recommended repair: {fix}. {steps}".strip()})
    except Exception as e:
        print("mechanicdb unavailable:", e)
    try:
        ds = load_dataset("Epitech/obd-codes-fine-tune", split="train")
        ep = pd.DataFrame([{"question": ((r.get("instruction") or "") + (" " + r["input"] if r.get("input") else "")).strip(),
                            "answer": (r.get("output") or "").strip()} for r in ds])
        n_ans = ep.groupby("question")["answer"].nunique()
        ep = ep[ep["question"].isin(n_ans[n_ans == 1].index)].drop_duplicates("question")
        rows.extend(ep.to_dict("records"))
    except Exception as e:
        print("Epitech obd dataset unavailable:", e)
    if not rows:
        raise RuntimeError("No Q&A source available; pass path= to a CSV with question/answer columns.")
    return pd.DataFrame(rows)


def load_dtc_reference_manual() -> list[dict]:
    """OBD-II DTC reference (CodesList/obd2-codes) grouped by system into manual-style sections."""
    try:
        from datasets import load_dataset
        ds = load_dataset("CodesList/obd2-codes", split="train")
    except Exception as e:
        print("DTC reference unavailable:", e); return []
    import collections
    groups = collections.defaultdict(list)
    for r in ds:
        cat = (r.get("category") or "General").strip()
        groups[cat].append(f"{r.get('code')}: {r.get('description')} (indicator: {r.get('severity')})")
    docs = []
    for cat, lines in groups.items():
        for j in range(0, len(lines), 40):
            text = f"OBD-II Diagnostic Trouble Code Reference - {cat} system, section {j//40+1}. " + " | ".join(lines[j:j+40])
            docs.append({"id": f"man_dtc_{_hid(cat, j)}", "source": "manual", "text": text,
                         "meta": {"make": None, "model": None, "year": None, "component": _guess_component(cat),
                                  "severity": "normal", "date": None, "file": f"OBD2_DTC_reference_{cat}.txt"}})
    return docs


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
COMPONENT_KEYWORDS = {
    "engine": ["engine", "misfire", "stall", "oil", "cylinder", "timing", "piston"],
    "transmission": ["transmission", "gear", "clutch", "shift", "cvt", "torque converter"],
    "brakes": ["brake", "abs", "rotor", "caliper", "pad"],
    "electrical": ["battery", "alternator", "wiring", "fuse", "electrical", "starter", "sensor"],
    "steering": ["steering", "power steering", "rack", "tie rod"],
    "suspension": ["suspension", "strut", "shock", "control arm", "ball joint"],
    "fuel system": ["fuel", "injector", "pump", "throttle"],
    "cooling": ["coolant", "radiator", "overheat", "thermostat", "water pump"],
    "airbag": ["airbag", "air bag", "srs"],
    "exhaust": ["exhaust", "catalytic", "o2 sensor", "emission"],
}
MAKES = [m for m, _, _ in DEFAULT_VEHICLES] + ["mazda", "audi", "mercedes", "lexus", "dodge"]


def _guess_component(text: str) -> str:
    t = text.lower()
    best, score = "general", 0
    for comp, kws in COMPONENT_KEYWORDS.items():
        s = sum(t.count(k) for k in kws)
        if s > score:
            best, score = comp, s
    return best


def _guess_make(text: str) -> str | None:
    t = text.lower()
    return next((m for m in MAKES if m in t), None)


def collect_all(qa_path: str | None = None, manuals_dir: str = "data/manuals",
                cache_dir: str = "data/raw") -> list[dict]:
    """Run all four loaders and return the merged raw document list."""
    docs = []
    docs += fetch_nhtsa_complaints(cache_dir=cache_dir)
    docs += fetch_nhtsa_recalls(cache_dir=cache_dir)
    docs += load_diagnostic_qa(qa_path)
    docs += load_manuals(manuals_dir) or load_dtc_reference_manual()
    return docs
