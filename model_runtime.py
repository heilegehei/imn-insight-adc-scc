"""Validated UI adapter around the unchanged, saved Top-2 Stacking runtime.

No fitting, feature selection, calibration, clipping or threshold search occurs
here. Uploaded values are never written to disk or placed in a shared data cache.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import threading
import warnings
from functools import lru_cache
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from runtime import frozen_runtime

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "assets" / "top2_stacking_model_package.pkl"
MODEL_SHA256 = "8ae7aa19e9608cbacac38736361e7d523502e411e31575446e2e7af8afe6a276"
DIRECT_COLUMNS = tuple(frozen_runtime.DIRECT_COLUMNS)
LAYERS = tuple(frozen_runtime.LAYERS)
LAYER_LABELS = dict(zip(LAYERS, ("I", "I + M", "I + M + N")))
THRESHOLDS = dict(zip(LAYERS, (0.7005904515139433, 0.7349164121493799, 0.6194277055171358)))
MAIN_LAYER = LAYERS[-1]
COMPOSITES = ("NLR", "SIRI", "LMR", "PNI", "AGR", "GAR")
SELECTED_DIRECT = ("MONO", "FIB", "LYMPH_PCT", "GLB", "TG", "HDL_C", "UA")
COMPOSITE_INPUTS = ("NEUT", "MONO", "LYMPH", "ALB", "TP", "GLU")
GROUP_LABELS = {"I": "Inflammation", "M": "Immune", "N": "Nutrition / metabolism"}
MAX_ROWS = 2000
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
_LOCK = threading.RLock()

_FIELDS = (
    ("WBC", "White blood cells", "10⁹/L", "I", 6.4),
    ("NEUT", "Neutrophils", "10⁹/L", "I", 4.2),
    ("NEUT_PCT", "Neutrophils", "%", "I", 65.6),
    ("MONO", "Monocytes", "10⁹/L", "I", 0.43),
    ("MONO_PCT", "Monocytes", "%", "I", 6.7),
    ("PLT", "Platelets", "10⁹/L", "I", 229.0),
    ("RDW_CV", "Red cell distribution width (CV)", "%", "I", 13.2),
    ("CRP", "C-reactive protein", "mg/L", "I", 8.7),
    ("FIB", "Fibrinogen", "g/L", "I", 3.9),
    ("LDH", "Lactate dehydrogenase", "U/L", "I", 178.0),
    ("LYMPH", "Lymphocytes", "10⁹/L", "M", 1.4),
    ("LYMPH_PCT", "Lymphocytes", "%", "M", 21.9),
    ("EOS", "Eosinophils", "10⁹/L", "M", 0.13),
    ("EOS_PCT", "Eosinophils", "%", "M", 2.0),
    ("BASO", "Basophils", "10⁹/L", "M", 0.03),
    ("BASO_PCT", "Basophils", "%", "M", 0.5),
    ("HGB", "Hemoglobin", "g/L", "N", 122.0),
    ("ALB", "Albumin", "g/L", "N", 36.7),
    ("TP", "Total protein", "g/L", "N", 65.9),
    ("GLB", "Globulin", "g/L", "N", 29.2),
    ("PA", "Prealbumin", "mg/dL", "N", 19.6),
    ("GLU", "Glucose", "mmol/L", "N", 5.3),
    ("TG", "Triglycerides", "mmol/L", "N", 1.27),
    ("TC", "Total cholesterol", "mmol/L", "N", 4.5),
    ("HDL_C", "HDL cholesterol", "mmol/L", "N", 1.05),
    ("LDL_C", "LDL cholesterol", "mmol/L", "N", 2.87),
    ("UA", "Uric acid", "µmol/L", "N", 305.0),
)
INPUT_FIELDS = tuple(dict(zip(("name", "label", "unit", "group", "example"), row)) for row in _FIELDS)
ALIASES = {"CHOL": "TC", "HDL-C": "HDL_C", "LDL-C": "LDL_C", "RDW-CV": "RDW_CV",
           "NEUT%": "NEUT_PCT", "MONO%": "MONO_PCT", "LYMPH%": "LYMPH_PCT",
           "EOS%": "EOS_PCT", "BASO%": "BASO_PCT", "RECORD_ID": "analysis_row_id",
           "ANALYSIS_ROW_ID": "analysis_row_id"}
_MISSING = {"", "na", "n/a", "nan", "null", "none"}


class InputError(ValueError):
    """An input problem that is safe to explain without exposing patient data."""


@lru_cache(maxsize=1)
def load_package():
    # Only the trusted local artifact can be unpickled, never an uploaded file.
    if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != MODEL_SHA256:
        raise RuntimeError("The saved model failed its integrity check.")
    with _LOCK:
        package = frozen_runtime.load_model_package(MODEL_PATH)
    if tuple(package["direct_columns"]) != DIRECT_COLUMNS:
        raise RuntimeError("The model input schema has changed.")
    if package["imputer"].sample_posterior:
        raise RuntimeError("A deterministic frozen imputer is required.")
    for layer in LAYERS:
        if list(package["layers"][layer]["stacking_meta"].classes_) != [0, 1]:
            raise RuntimeError("Unexpected positive-class order.")
    return package


def example_frame() -> pd.DataFrame:
    """A constructed example, not an individual patient's record."""
    return pd.DataFrame([{f["name"]: f["example"] for f in INPUT_FIELDS}])


def input_template() -> bytes:
    return pd.DataFrame(columns=["analysis_row_id", *DIRECT_COLUMNS]).to_csv(index=False).encode("utf-8-sig")


def read_csv_bytes(raw: bytes) -> pd.DataFrame:
    if len(raw) > MAX_UPLOAD_BYTES:
        raise InputError("Upload a CSV smaller than 5 MB.")
    try:
        text = raw.decode("utf-8-sig")
        reader = csv.reader(io.StringIO(text), strict=True)
        header = next(reader)
        canonical = [ALIASES.get(c.strip().upper(), c.strip().upper()) for c in header]
        if len(canonical) != len(set(canonical)):
            raise InputError("Duplicate or conflicting columns. For example, use TC or CHOL, not both.")
        row_count = 0
        for csv_row in reader:
            if not csv_row:
                continue
            row_count += 1
            if row_count > MAX_ROWS:
                raise InputError(f"Provide no more than {MAX_ROWS:,} records per batch.")
            if len(csv_row) != len(header):
                raise InputError(f"CSV row {row_count} has a different number of fields from the header. Use commas for empty values.")
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False, nrows=MAX_ROWS + 1)
    except InputError:
        raise
    except (UnicodeError, ValueError, StopIteration, csv.Error, pd.errors.ParserError) as exc:
        raise InputError("The file must be a non-empty, comma-separated UTF-8 CSV with a header.") from exc
    return frame


def normalize_inputs(frame: pd.DataFrame):
    if not 1 <= len(frame) <= MAX_ROWS:
        raise InputError(f"Provide between 1 and {MAX_ROWS:,} records per batch.")
    columns = [ALIASES.get(str(c).strip().upper(), str(c).strip().upper()) for c in frame.columns]
    if len(columns) != len(set(columns)):
        raise InputError("Duplicate or conflicting columns. Use TC or CHOL, not both.")
    frame = frame.copy().reset_index(drop=True)
    frame.columns = columns
    absent = [c for c in DIRECT_COLUMNS if c not in frame]
    ignored = [c for c in frame if c not in (*DIRECT_COLUMNS, "analysis_row_id")]
    if len(absent) == len(DIRECT_COLUMNS):
        raise InputError("No laboratory columns were found. Start with the downloadable CSV template.")
    out = pd.DataFrame(index=frame.index, columns=DIRECT_COLUMNS, dtype=float)
    problems = []
    crp_converted = 0
    for c in DIRECT_COLUMNS:
        if c not in frame:
            continue
        for row, value in enumerate(frame[c]):
            if value is None or pd.isna(value) or str(value).strip().lower() in _MISSING:
                continue
            raw = str(value).strip()
            if c == "CRP" and (match := re.fullmatch(r"([<>])\s*(\d+(?:\.\d*)?|\.\d+)(?:([eE][+-]?\d+))?", raw)):
                boundary = float(match[2] + (match[3] or ""))
                value = boundary / np.sqrt(2) if match[1] == "<" else boundary
                crp_converted += 1
            try:
                number = float(value)
            except (TypeError, ValueError):
                problems.append(f"row {row + 1}, {c}: enter a number or leave blank")
                continue
            if not np.isfinite(number) or number < 0:
                problems.append(f"row {row + 1}, {c}: enter a finite, non-negative value")
            elif (c.endswith("_PCT") or c == "RDW_CV") and number > 100:
                problems.append(f"row {row + 1}, {c}: percentages must be within 0–100")
            elif c in ("LYMPH", "MONO", "ALB") and number == 0:
                problems.append(f"row {row + 1}, {c}: must be greater than zero for the composite indices")
            else:
                out.at[row, c] = number
    if problems:
        suffix = f"; and {len(problems) - 6} more" if len(problems) > 6 else ""
        raise InputError("Check " + "; ".join(problems[:6]) + suffix + ".")
    empty = out.isna().all(axis=1)
    if empty.any():
        rows = ", ".join(str(i + 1) for i in out.index[empty][:6])
        raise InputError(f"All laboratory values are blank in row(s) {rows}. Enter observed measurements first.")
    invalid_protein = out["TP"].notna() & out["ALB"].notna() & (out["TP"] <= out["ALB"])
    if invalid_protein.any():
        rows = ", ".join(str(i + 1) for i in out.index[invalid_protein][:6])
        raise InputError(f"Total protein must exceed albumin (both g/L); check row(s) {rows}.")
    ids = frame.get("analysis_row_id", pd.Series([f"record_{i + 1:04d}" for i in frame.index]))
    ids = ids.fillna("").astype(str)
    blank_ids = ids.str.strip().eq("")
    ids.loc[blank_ids] = [f"record_{i + 1:04d}" for i in ids.index[blank_ids]]
    return out, ids, {"absent_columns": absent, "ignored_columns": ignored, "crp_converted": crp_converted}


def score_frame(frame: pd.DataFrame) -> dict:
    direct, ids, notices = normalize_inputs(frame)
    package = load_package()
    with _LOCK, threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="'force_all_finite' was renamed", category=FutureWarning)
        completed = pd.DataFrame(package["imputer"].transform(direct), columns=DIRECT_COLUMNS)
        if not np.isfinite(completed.to_numpy()).all() or (completed["TP"] <= completed["ALB"]).any():
            raise InputError("The completed inputs contain an invalid protein balance. Supply observed TP and ALB values.")
        features = frozen_runtime._add_composites(completed)
        if not np.isfinite(features.to_numpy()).all():
            raise InputError("The composite indices are not finite. Check their component measurements.")
        # Preserve the authoritative saved prediction function; no fitting calls.
        predictions = frozen_runtime.predict_frame(package, direct)
    result = pd.DataFrame({"analysis_row_id": ids})
    result["missing_input_count"] = direct.isna().sum(axis=1)
    result["imputed_fields"] = direct.isna().apply(lambda row: ";".join(row.index[row]), axis=1)
    for layer in LAYERS:
        p = predictions[f"{layer}__Stacking"].to_numpy()
        if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
            raise RuntimeError("The model returned invalid probabilities.")
        result[f"{layer}__adenocarcinoma_probability"] = p
        result[f"{layer}__threshold"] = THRESHOLDS[layer]
        result[f"{layer}__class"] = np.where(p >= THRESHOLDS[layer], "Adenocarcinoma", "Squamous cell carcinoma")
    for name in COMPOSITES:
        result[name] = features[name].to_numpy()
    notices["out_of_training_range_count"] = int(((direct < package["imputer_minimum"]) | (direct > package["imputer_maximum"])).sum().sum())
    return {"results": result, "completed": completed, "input": direct, "notices": notices}


def safe_result_table(frame: pd.DataFrame) -> pd.DataFrame:
    safe = frame.copy()
    # Prevent user-supplied record IDs from becoming spreadsheet formulas.
    if "analysis_row_id" in safe:
        safe["analysis_row_id"] = safe["analysis_row_id"].map(
            lambda value: "'" + str(value) if str(value).lstrip().startswith(("=", "+", "-", "@", "\t", "\r")) else str(value)
        )
    return safe


def export_csv(frame: pd.DataFrame) -> bytes:
    return safe_result_table(frame).to_csv(index=False, float_format="%.12g").encode("utf-8-sig")


def metadata() -> dict:
    return json.loads((ROOT / "assets" / "model_manifest.json").read_text(encoding="utf-8"))
