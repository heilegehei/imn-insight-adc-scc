"""Inference-only runtime shared by the rebuilt analysis and deployed tool."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

from .frozen_runtime import DIRECT_COLUMNS, LAYERS, _add_composites, _load_formal_factory


def load_model_package(path: str | Path) -> dict:
    _load_formal_factory()
    with Path(path).open("rb") as stream:
        package = pickle.load(stream)
    if package.get("package_schema") != "imn.weighted_voting.v2":
        raise ValueError("Unexpected Weighted voting model package")
    for layer in LAYERS:
        spec = package["layers"][layer]
        weights = np.asarray([spec["weights"][name] for name in spec["member_order"]])
        if np.any(weights < 0) or not np.isclose(weights.sum(), 1.0, atol=1e-8):
            raise ValueError("Invalid frozen voting weights")
    return package


def completed_features(package: dict, frame: pd.DataFrame) -> pd.DataFrame:
    direct = frame.loc[:, package["direct_columns"]].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    completed = pd.DataFrame(package["imputer"].transform(direct), columns=package["direct_columns"], index=frame.index)
    # Canonical numerical precision suppresses BLAS batch-size noise (~1e-13)
    # before tree thresholds are evaluated. This is not clinical winsorization.
    return _add_composites(completed).round(package.get("inference_precision_decimals", 10))


def calibrate(spec: dict, probability: np.ndarray) -> np.ndarray:
    calibration = spec["calibration"]
    epsilon = calibration["clip_epsilon"]
    clipped = np.clip(np.asarray(probability, dtype=float), epsilon, 1 - epsilon)
    return expit(calibration["intercept"] + calibration["slope"] * np.log(clipped / (1 - clipped)))


def predict_raw_features(package: dict, layer: str, features: pd.DataFrame | np.ndarray) -> np.ndarray:
    spec = package["layers"][layer]
    x = features.loc[:, spec["columns"]].to_numpy(float) if isinstance(features, pd.DataFrame) else np.asarray(features, dtype=float)
    x = np.round(x, package.get("inference_precision_decimals", 10))
    result = np.zeros(len(x), dtype=float)
    for name in spec["member_order"]:
        result += spec["weights"][name] * spec["base_members"][name].predict_proba(x)[:, 1]
    return result


def predict_frame(package: dict, frame: pd.DataFrame) -> pd.DataFrame:
    features = completed_features(package, frame)
    result = pd.DataFrame(index=frame.index)
    if "analysis_row_id" in frame:
        result["analysis_row_id"] = frame.analysis_row_id.astype(str).to_numpy()
    for layer in LAYERS:
        raw = predict_raw_features(package, layer, features)
        result[f"{layer}__Weighted_Voting"] = raw
        result[f"{layer}__Weighted_Voting_calibrated"] = calibrate(package["layers"][layer], raw)
    return result
