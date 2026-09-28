"""Runtime helpers for the bundled Top-2 Stacking model package."""

from __future__ import annotations

import importlib.util
import os
import pickle
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# The native model factory enforces a deterministic single-thread contract at
# import time.  Set these defaults before loading it so the package behaves
# identically in a clean deployment process and in the training environment.
for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"


DIRECT_COLUMNS = (
    "WBC", "NEUT", "NEUT_PCT", "MONO", "MONO_PCT", "PLT", "RDW_CV", "CRP", "FIB", "LDH",
    "LYMPH", "LYMPH_PCT", "EOS", "EOS_PCT", "BASO", "BASO_PCT", "HGB", "ALB", "TP", "GLB",
    "PA", "GLU", "TG", "TC", "HDL_C", "LDL_C", "UA",
)
LAYERS = ("I", "I_plus_M", "I_plus_M_plus_N")


def _load_formal_factory() -> None:
    """Register the native model classes before unpickling the package."""

    package_dir = Path(__file__).resolve().parent
    path = package_dir / "runtime_support" / "formal_model_factory.py"
    if "_stage06_native_model_factory" in sys.modules:
        return
    spec = importlib.util.spec_from_file_location("stage12_runtime_formal_factory", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load model factory: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def load_model_package(path: str | Path) -> dict[str, Any]:
    _load_formal_factory()
    with Path(path).open("rb") as stream:
        package = pickle.load(stream)
    if package.get("package_schema") != "stage12.education_top2_stacking_package.v1":
        raise ValueError("Unexpected model package schema")
    return package


def _add_composites(completed: pd.DataFrame) -> pd.DataFrame:
    out = completed.loc[:, DIRECT_COLUMNS].copy()
    out["NLR"] = out["NEUT"] / out["LYMPH"]
    out["SIRI"] = out["NEUT"] * out["MONO"] / out["LYMPH"]
    out["LMR"] = out["LYMPH"] / out["MONO"]
    out["PNI"] = out["ALB"] + 5.0 * out["LYMPH"]
    out["AGR"] = out["ALB"] / (out["TP"] - out["ALB"])
    out["GAR"] = out["GLU"] / out["ALB"]
    return out


def predict_frame(package: dict[str, Any], frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in package["direct_columns"] if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required direct columns: {missing}")
    direct = frame.loc[:, package["direct_columns"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    completed = pd.DataFrame(package["imputer"].transform(direct), columns=package["direct_columns"], index=frame.index)
    features = _add_composites(completed)
    output = pd.DataFrame(index=frame.index)
    if "analysis_row_id" in frame.columns:
        output["analysis_row_id"] = frame["analysis_row_id"].astype(str).to_numpy()
    for layer in LAYERS:
        spec = package["layers"][layer]
        x = features.loc[:, spec["columns"]].to_numpy(dtype=float)
        member_matrix = np.column_stack([spec["base_members"][name].predict_proba(x)[:, 1] for name in spec["stacking_member_order"]])
        output[f"{layer}__Stacking"] = spec["stacking_meta"].predict_proba(member_matrix)[:, 1]
    return output


def predict_csv(package_path: str | Path, input_csv: str | Path, output_csv: str | Path) -> pd.DataFrame:
    package = load_model_package(package_path)
    frame = pd.read_csv(input_csv, encoding="utf-8-sig", low_memory=False)
    output = predict_frame(package, frame)
    output.to_csv(output_csv, index=False, encoding="utf-8-sig", float_format="%.12g")
    return output
