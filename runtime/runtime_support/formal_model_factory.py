"""Frozen stage-08 model, ensemble, calibration, threshold, and metric APIs.

This module deliberately builds on the untouched stage-06 factory.  It adds
the formal H<=8 configuration registry and execution interfaces needed by the
real A-centre outer-fold runner.  It never reads study data.
"""

from __future__ import annotations

import hashlib
import importlib.util
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pygam
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline


MASTER_SEED = 2431905072
MAX_CONFIGURATIONS = 8
EXPECTED_PYGAM_VERSION = "0.12.0"
EXPECTED_STAGE06_FACTORY_SHA256 = (
    "11ab536c6921320963928a2a5730d8db8ea3a620d5d7c619450f8dd5564e74c5"
)
INNER_AUROC_EQUIVALENCE = 0.005
PROBABILITY_CLIP_EPSILON = 1e-6
NUMERIC_ATOL = 1e-12

MODULE_DIR = Path(__file__).resolve().parent
STAGE06_FACTORY_PATH = MODULE_DIR / "integrated_model_factory.py"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


if pygam.__version__ != EXPECTED_PYGAM_VERSION:
    raise RuntimeError(
        f"Formal GAM requires pyGAM {EXPECTED_PYGAM_VERSION}; found {pygam.__version__}"
    )
if sha256_file(STAGE06_FACTORY_PATH) != EXPECTED_STAGE06_FACTORY_SHA256:
    raise RuntimeError("Untouched stage-06 native model factory hash changed")

_stage06_spec = importlib.util.spec_from_file_location(
    "_stage06_native_model_factory", STAGE06_FACTORY_PATH
)
if _stage06_spec is None or _stage06_spec.loader is None:
    raise RuntimeError("Cannot load the stage-06 native model factory")
_stage06 = importlib.util.module_from_spec(_stage06_spec)
sys.modules[_stage06_spec.name] = _stage06
_stage06_spec.loader.exec_module(_stage06)

BASE_FAMILIES = tuple(_stage06.EXPECTED_MEMBERS)
ENSEMBLE_FAMILIES = ("Soft Voting", "Weighted Voting", "Stacking")
ALL_CANDIDATES = BASE_FAMILIES + ENSEMBLE_FAMILIES


def _cartesian(**axes: Sequence[Any]) -> list[dict[str, Any]]:
    keys = tuple(axes)
    return [
        dict(zip(keys, values, strict=True))
        for values in itertools.product(*(axes[key] for key in keys))
    ]


def _model_spaces() -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]]]:
    spaces: dict[str, list[dict[str, Any]]] = {
        "Logistic": _cartesian(C=[0.01, 0.1, 1.0, 10.0, 100.0]),
        "GAM": _cartesian(n_splines=[4, 5, 6], lam=[0.1, 1.0, 10.0, 100.0]),
        "KNN": _cartesian(
            n_neighbors=[3, 5, 9, 15, 25, 35],
            weights=["uniform", "distance"],
            p=[1, 2],
        ),
        "RBF-SVM": _cartesian(C=[0.1, 1.0, 10.0, 100.0], gamma=["scale", 0.01, 0.1, 1.0]),
        "GaussianNB": _cartesian(
            var_smoothing=[1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4, 1e-2]
        ),
        "DecisionTree": _cartesian(
            criterion=["gini", "log_loss"],
            max_depth=[2, 3, 5, None],
            min_samples_leaf=[1, 5, 10, 20],
            ccp_alpha=[0.0, 0.001, 0.01],
        ),
        "RandomForest": _cartesian(
            max_features=["sqrt", 0.5, 1.0],
            max_depth=[None, 4, 8],
            min_samples_leaf=[1, 3, 8],
            max_samples=[None, 0.8],
        ),
        "ExtraTrees": _cartesian(
            max_features=["sqrt", 0.5, 1.0],
            max_depth=[None, 4, 8],
            min_samples_leaf=[1, 3, 8],
        ),
        "GBDT": _cartesian(
            learning_rate=[0.1, 0.05, 0.02],
            n_estimators=[100, 250, 500],
            max_depth=[1, 2, 3],
            min_samples_leaf=[5, 10, 20],
            subsample=[0.8, 1.0],
        ),
        "XGBoost": _cartesian(
            learning_rate=[0.1, 0.05, 0.02],
            n_estimators=[100, 250, 500],
            max_depth=[2, 3, 4],
            min_child_weight=[1, 5],
            subsample=[0.8, 1.0],
            colsample_bytree=[0.8, 1.0],
            reg_lambda=[1.0, 10.0],
        ),
        "LightGBM": _cartesian(
            learning_rate=[0.1, 0.05, 0.02],
            n_estimators=[100, 250, 500],
            num_leaves=[7, 15, 31],
            max_depth=[3, 4, 5],
            min_child_samples=[10, 20, 40],
            subsample=[0.8, 1.0],
            colsample_bytree=[0.8, 1.0],
            reg_lambda=[0.0, 1.0, 10.0],
        ),
        "AdaBoost": _cartesian(
            learning_rate=[1.0, 0.5, 0.1],
            n_estimators=[50, 100, 300],
            base_max_depth=[1, 2, 3],
            base_min_samples_leaf=[1, 5, 10],
        ),
        "RotationForest": _cartesian(
            n_estimators=[100, 200, 400], group_size=[2, 3, 5]
        ),
        "MLP": _cartesian(
            hidden_layer_sizes=[[8], [16], [32], [16, 8], [32, 16]],
            activation=["relu", "tanh"],
            alpha=[1e-4, 1e-3, 1e-2, 1e-1],
        ),
    }
    # Coupled axes in the protocol are pairs, not independent Cartesian axes.
    lr_n_pairs = {(0.1, 100), (0.05, 250), (0.02, 500)}
    for family in ("GBDT", "XGBoost", "LightGBM"):
        spaces[family] = [
            config
            for config in spaces[family]
            if (config["learning_rate"], config["n_estimators"]) in lr_n_pairs
        ]
    leaves_depth_pairs = {(7, 3), (15, 4), (31, 5)}
    spaces["LightGBM"] = [
        config
        for config in spaces["LightGBM"]
        if (config["num_leaves"], config["max_depth"]) in leaves_depth_pairs
    ]
    ada_pairs = {(1.0, 50), (0.5, 100), (0.1, 300)}
    spaces["AdaBoost"] = [
        config
        for config in spaces["AdaBoost"]
        if (config["learning_rate"], config["n_estimators"]) in ada_pairs
    ]

    defaults: dict[str, dict[str, Any]] = {
        "Logistic": {"C": 1.0},
        "GAM": {"n_splines": 5, "lam": 10.0},
        "KNN": {"n_neighbors": 9, "weights": "uniform", "p": 2},
        "RBF-SVM": {"C": 1.0, "gamma": "scale"},
        "GaussianNB": {"var_smoothing": 1e-9},
        "DecisionTree": {
            "criterion": "gini",
            "max_depth": 3,
            "min_samples_leaf": 5,
            "ccp_alpha": 0.0,
        },
        "RandomForest": {
            "max_features": "sqrt",
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_samples": None,
        },
        "ExtraTrees": {
            "max_features": "sqrt",
            "max_depth": None,
            "min_samples_leaf": 1,
        },
        "GBDT": {
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 2,
            "min_samples_leaf": 10,
            "subsample": 1.0,
        },
        "XGBoost": {
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 3,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
        },
        "LightGBM": {
            "learning_rate": 0.1,
            "n_estimators": 100,
            "num_leaves": 15,
            "max_depth": 4,
            "min_child_samples": 20,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
        },
        "AdaBoost": {
            "learning_rate": 1.0,
            "n_estimators": 50,
            "base_max_depth": 1,
            "base_min_samples_leaf": 5,
        },
        "RotationForest": {"n_estimators": 100, "group_size": 3},
        "MLP": {
            "hidden_layer_sizes": [16],
            "activation": "relu",
            "alpha": 1e-3,
        },
    }
    if tuple(spaces) != BASE_FAMILIES or tuple(defaults) != BASE_FAMILIES:
        raise AssertionError("Formal 14-family order changed")
    for family in BASE_FAMILIES:
        if defaults[family] not in spaces[family]:
            raise AssertionError(f"{family}: default configuration is absent from grid")
    return spaces, defaults


def _registry_seed(family: str, master_seed: int) -> tuple[int, dict[str, Any]]:
    key = {
        "master_seed": int(master_seed),
        "stage": "formal_model_configuration_registry",
        "outer_repeat": None,
        "outer_fold": None,
        "inner_fold": None,
        "layer": None,
        "algorithm": family,
        "imputation_id": None,
    }
    encoded = canonical_json(key).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:4], "big", signed=False), key


def _config_id(family: str, parameters: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        canonical_json({"family": family, "parameters": dict(parameters)}).encode("utf-8")
    ).hexdigest()
    return f"{family.replace('-', '_').replace(' ', '_')}_{digest[:12]}"


def generate_frozen_registry(
    master_seed: int = MASTER_SEED,
    maximum_configurations: int = MAX_CONFIGURATIONS,
) -> dict[str, Any]:
    if not (1 <= int(maximum_configurations) <= 8):
        raise ValueError("Formal H must be in [1, 8]")
    spaces, defaults = _model_spaces()
    families: dict[str, Any] = {}
    for family in BASE_FAMILIES:
        full_space = spaces[family]
        default = defaults[family]
        remaining = [config for config in full_space if config != default]
        sample_seed, seed_key = _registry_seed(family, master_seed)
        count = min(int(maximum_configurations) - 1, len(remaining))
        if len(full_space) <= int(maximum_configurations):
            selected = remaining
        else:
            rng = np.random.default_rng(sample_seed)
            selected_indices = sorted(
                int(index)
                for index in rng.choice(len(remaining), size=count, replace=False)
            )
            selected = [remaining[index] for index in selected_indices]
        selected = [default, *selected]
        configurations = [
            {
                "registry_index": index,
                "config_id": _config_id(family, config),
                "is_default": index == 0,
                "parameters": config,
            }
            for index, config in enumerate(selected)
        ]
        families[family] = {
            "search_space_size": len(full_space),
            "sample_seed_uint32": sample_seed,
            "sample_seed_key": seed_key,
            "default_config_id": configurations[0]["config_id"],
            "configuration_count": len(configurations),
            "configurations": configurations,
        }

    registry: dict[str, Any] = {
        "schema_version": "stage08.formal_model_registry.v1",
        "master_seed_uint32": int(master_seed),
        "maximum_configurations_per_base_family_H": int(maximum_configurations),
        "generation_rule": (
            "predefined_default_first_then_deterministic_without_replacement_sample; "
            "sampled_entries_preserve_full_grid_order"
        ),
        "reuse_scope": "all_outer_folds_all_three_layers_and_final_A",
        "base_family_order": list(BASE_FAMILIES),
        "ensemble_family_order": list(ENSEMBLE_FAMILIES),
        "stage06_native_factory": {
            "relative_path": "stage06_integrated_gate/integrated_model_factory.py",
            "sha256": EXPECTED_STAGE06_FACTORY_SHA256,
            "pygam_version": EXPECTED_PYGAM_VERSION,
            "solver": "upstream_native_undamped",
        },
        "families": families,
        "ensembles": {
            "Soft Voting": {"configurations": [{"config_id": "Soft_Voting_fixed", "parameters": {}}]},
            "Weighted Voting": {"configurations": [{"config_id": "Weighted_Voting_fixed", "parameters": {}}]},
            "Stacking": {
                "configurations": [
                    {"config_id": f"Stacking_C_{value:g}", "parameters": {"C": value}}
                    for value in (0.01, 0.1, 1.0, 10.0)
                ]
            },
        },
        "formal_failure_policy": {
            "configuration_valid_only_if_all_5_inner_folds_x_10_imputations_complete": True,
            "eligible_numerical_rescue_base_families": ["Logistic", "GAM", "MLP"],
            "maximum_numerical_rescues": 1,
            "rescue_action": "max_iter_x2_only_same_seed_and_statistical_parameters",
            "refit_failure_action": "try_next_pre_ranked_fully_valid_configuration",
            "family_eligibility": "complete_predictions_all_required_folds_and_layers",
        },
    }
    registry["registry_sha256"] = hashlib.sha256(
        canonical_json(registry).encode("utf-8")
    ).hexdigest()
    validate_registry(registry)
    return registry


def validate_registry(registry: Mapping[str, Any]) -> None:
    copied = dict(registry)
    recorded_hash = copied.pop("registry_sha256", None)
    actual_hash = hashlib.sha256(canonical_json(copied).encode("utf-8")).hexdigest()
    if recorded_hash != actual_hash:
        raise AssertionError("Frozen registry self-hash mismatch")
    if tuple(copied["base_family_order"]) != BASE_FAMILIES:
        raise AssertionError("Frozen registry family order mismatch")
    if tuple(copied["ensemble_family_order"]) != ENSEMBLE_FAMILIES:
        raise AssertionError("Frozen registry ensemble order mismatch")
    if copied["stage06_native_factory"]["sha256"] != EXPECTED_STAGE06_FACTORY_SHA256:
        raise AssertionError("Frozen registry references the wrong stage-06 factory")
    for family in BASE_FAMILIES:
        block = copied["families"][family]
        configurations = block["configurations"]
        if not (1 <= len(configurations) <= MAX_CONFIGURATIONS):
            raise AssertionError(f"{family}: H limit violated")
        if configurations[0]["is_default"] is not True:
            raise AssertionError(f"{family}: default is not first")
        if len({entry["config_id"] for entry in configurations}) != len(configurations):
            raise AssertionError(f"{family}: duplicate configurations")
        for index, entry in enumerate(configurations):
            if entry["registry_index"] != index:
                raise AssertionError(f"{family}: registry index is not contiguous")


def load_frozen_registry(path: Path | str) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        registry = json.load(stream)
    validate_registry(registry)
    regenerated = generate_frozen_registry(
        registry["master_seed_uint32"],
        registry["maximum_configurations_per_base_family_H"],
    )
    if canonical_json(registry) != canonical_json(regenerated):
        raise AssertionError("Registry does not reproduce from its frozen seed and rule")
    return registry


def _entry_parameters(config: Mapping[str, Any]) -> dict[str, Any]:
    if "parameters" in config:
        return dict(config["parameters"])
    return dict(config)


class ConfigurationInvalidError(ValueError):
    pass


class PredictionInvalidError(RuntimeError):
    """A fitted configuration produced unusable probabilities."""

    pass


def positive_probability(model: Any, X: Any, label: str) -> np.ndarray:
    """Validate prediction output and expose a narrowly catchable failure."""

    try:
        return _stage06.positive_probability(model, X, label)
    except AssertionError as error:
        raise PredictionInvalidError(f"{label}: invalid prediction contract: {error}") from error


def make_model(
    family: str,
    config: Mapping[str, Any],
    seed: int,
    *,
    n_features: int | None = None,
) -> Any:
    """Construct one formal base model from a frozen registry entry."""

    if family not in BASE_FAMILIES:
        raise ConfigurationInvalidError(f"Unknown base family: {family}")
    parameters = _entry_parameters(config)
    model = _stage06.make_models(int(seed))[family]

    if family == "Logistic":
        model.set_params(model__C=parameters["C"])
    elif family == "GAM":
        model.set_params(n_splines=parameters["n_splines"], lam=parameters["lam"])
    elif family == "KNN":
        model.set_params(
            model__n_neighbors=parameters["n_neighbors"],
            model__weights=parameters["weights"],
            model__p=parameters["p"],
        )
    elif family == "RBF-SVM":
        available = model.get_params(deep=True)
        prefix = "estimator" if "estimator__svc__C" in available else "base_estimator"
        model.set_params(
            **{
                f"{prefix}__svc__C": parameters["C"],
                f"{prefix}__svc__gamma": parameters["gamma"],
            }
        )
    elif family == "GaussianNB":
        model.set_params(model__var_smoothing=parameters["var_smoothing"])
    elif family == "DecisionTree":
        model.set_params(**parameters)
    elif family == "RandomForest":
        model.set_params(**parameters)
    elif family == "ExtraTrees":
        model.set_params(**parameters)
    elif family == "GBDT":
        model.set_params(**parameters)
    elif family == "XGBoost":
        model.set_params(**parameters)
    elif family == "LightGBM":
        # LightGBM ignores subsample<1 when subsample_freq remains zero.
        model.set_params(
            **parameters,
            subsample_freq=1 if float(parameters["subsample"]) < 1.0 else 0,
        )
    elif family == "AdaBoost":
        available = model.get_params(deep=True)
        prefix = "estimator" if "estimator__max_depth" in available else "base_estimator"
        model.set_params(
            learning_rate=parameters["learning_rate"],
            n_estimators=parameters["n_estimators"],
            **{
                f"{prefix}__max_depth": parameters["base_max_depth"],
                f"{prefix}__min_samples_leaf": parameters["base_min_samples_leaf"],
            },
        )
    elif family == "RotationForest":
        if n_features is None:
            raise ConfigurationInvalidError(
                "RotationForest requires n_features for the frozen group-size constraint"
            )
        group = int(parameters["group_size"])
        if group > int(n_features):
            raise ConfigurationInvalidError(
                f"RotationForest group_size={group} exceeds n_features={n_features}"
            )
        available = model.get_params(deep=True)
        group_parameters: dict[str, Any]
        if "min_group" in available and "max_group" in available:
            group_parameters = {"min_group": group, "max_group": group}
        elif "group" in available:
            group_parameters = {"group": group}
        else:
            raise RuntimeError("Unsupported RotationForest group-size API")
        model.set_params(n_estimators=parameters["n_estimators"], **group_parameters)
    elif family == "MLP":
        model.set_params(
            model__hidden_layer_sizes=tuple(parameters["hidden_layer_sizes"]),
            model__activation=parameters["activation"],
            model__alpha=parameters["alpha"],
        )
    return model


@dataclass(frozen=True)
class FittedModel:
    model: Any
    family: str
    audit: dict[str, Any]


def fit_model_fail_closed(model: Any, X: Any, y: Any, label: str) -> FittedModel:
    """Fit once, allowing only the frozen single max_iter x2 rescue."""

    family = _stage06._model_family(model)
    convergence, messages = _stage06._fit_attempt(model, X, y)
    audit: dict[str, Any] = {
        "family": family,
        "physical_attempts": 1,
        "rescue_used": False,
        "messages": sorted(set(messages)),
    }
    if not convergence:
        return FittedModel(model=model, family=family, audit=audit)

    rescue_family, parameter, original_limit = _stage06._rescue_family_and_parameter(model)
    if rescue_family is None:
        raise _stage06.FitAttemptError(
            f"{label}: convergence signal in non-rescuable family: {convergence[0]}",
            family=family,
            physical_attempts=1,
        )
    rescued = clone(model)
    rescued.set_params(**{parameter: int(original_limit) * 2})
    try:
        rescue_convergence, rescue_messages = _stage06._fit_attempt(rescued, X, y)
    except _stage06.FitAttemptError as error:
        raise _stage06.NumericalRescueError(
            f"{label}: {rescue_family} rescue attempt failed: {error}",
            family=rescue_family,
            physical_attempts=2,
            stdout_tail=error.stdout_tail,
            stderr_tail=error.stderr_tail,
        ) from error
    if rescue_convergence:
        raise _stage06.NumericalRescueError(
            f"{label}: {rescue_family} failed its single max_iter x2 rescue: "
            f"first={convergence[0]}; rescue={rescue_convergence[0]}",
            family=rescue_family,
            physical_attempts=2,
        )
    audit.update(
        physical_attempts=2,
        rescue_used=True,
        original_max_iter=int(original_limit),
        rescue_max_iter=int(original_limit) * 2,
        messages=sorted(
            set(
                [
                    *messages,
                    *rescue_messages,
                    f"NUMERIC_RESCUE: {rescue_family} max_iter x2 once",
                ]
            )
        ),
    )
    return FittedModel(model=rescued, family=rescue_family, audit=audit)


def fit_predict_fail_closed(
    model: Any, X_fit: Any, y_fit: Any, X_new: Any, label: str
) -> tuple[np.ndarray, FittedModel]:
    fitted = fit_model_fail_closed(model, X_fit, y_fit, label)
    probability = positive_probability(fitted.model, X_new, label)
    return probability, fitted


class FamilyRefitExhaustedError(RuntimeError):
    def __init__(self, family: str, failures: Sequence[Mapping[str, Any]]) -> None:
        super().__init__(f"{family}: all pre-ranked fully valid configurations failed refit")
        self.family = family
        self.failures = list(failures)


def refit_with_preranked_fallback(
    family: str,
    ranked_configurations: Sequence[Mapping[str, Any]],
    X_fit: Any,
    y_fit: Any,
    X_new: Any,
    seed: int,
    *,
    n_features: int,
    label: str,
) -> dict[str, Any]:
    """Refit in frozen pre-rank order; never edits a failed configuration."""

    failures: list[dict[str, Any]] = []
    for fallback_rank, entry in enumerate(ranked_configurations):
        config_id = str(entry["config_id"])
        try:
            model = make_model(
                family, entry, int(seed), n_features=int(n_features)
            )
            probability, fitted = fit_predict_fail_closed(
                model, X_fit, y_fit, X_new, f"{label}/{family}/{config_id}"
            )
        except (
            ConfigurationInvalidError,
            PredictionInvalidError,
            _stage06.FitAttemptError,
        ) as error:
            failures.append(
                {
                    "fallback_rank": fallback_rank,
                    "config_id": config_id,
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "physical_attempts": int(getattr(error, "physical_attempts", 0)),
                }
            )
            continue
        return {
            "family": family,
            "selected_config_id": config_id,
            "fallback_rank": fallback_rank,
            "probability": probability,
            "fitted_model": fitted.model,
            "fit_audit": fitted.audit,
            "earlier_refit_failures": failures,
        }
    raise FamilyRefitExhaustedError(family, failures)


def rank_fully_valid_configurations(
    family: str,
    inner_results: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Apply AUROC±0.005, then log loss, then frozen registry order.

    Only records explicitly declaring all 5 inner folds x 10 imputations
    complete are admitted.  Iterative selection yields a complete fallback
    order without altering any metrics.
    """

    entries = {
        entry["config_id"]: entry
        for entry in registry["families"][family]["configurations"]
    }
    result_ids = [str(result["config_id"]) for result in inner_results]
    if len(result_ids) != len(set(result_ids)):
        raise AssertionError(f"{family}: duplicate config_id results are forbidden")
    valid: list[dict[str, Any]] = []
    for result in inner_results:
        if result.get("status") != "VALID":
            continue
        if int(result.get("inner_folds_completed", -1)) != 5:
            continue
        if int(result.get("imputations_per_inner_fold", -1)) != 10:
            continue
        if result.get("complete_probabilities") is not True:
            continue
        fit_records = result.get("fit_records")
        if not isinstance(fit_records, list) or len(fit_records) != 50:
            continue
        expected_pairs = {(fold, imp) for fold in range(5) for imp in range(1, 11)}
        observed_pairs: set[tuple[int, int]] = set()
        records_valid = True
        for record in fit_records:
            pair = (
                int(record.get("inner_fold", -1)),
                int(record.get("imputation_id", -1)),
            )
            observed_pairs.add(pair)
            probability = np.asarray(record.get("probability", []), dtype=float).reshape(-1)
            expected_n = int(record.get("new_rows", -1))
            if (
                record.get("status") != "PASS"
                or int(record.get("positive_class", -1)) != 1
                or expected_n <= 0
                or len(probability) != expected_n
                or not np.isfinite(probability).all()
                or np.any(probability < 0)
                or np.any(probability > 1)
            ):
                records_valid = False
                break
        if observed_pairs != expected_pairs or not records_valid:
            continue
        fold_outputs = result.get("fold_outputs")
        if not isinstance(fold_outputs, list) or len(fold_outputs) != 5:
            continue
        aggregate_valid = True
        for fold_index, fold_output in enumerate(fold_outputs):
            aggregate = np.asarray(fold_output.get("probability", []), dtype=float).reshape(-1)
            outcomes = np.asarray(fold_output.get("outcome", []), dtype=int).reshape(-1)
            row_ids = [str(value) for value in fold_output.get("new_row_ids", [])]
            source = [
                np.asarray(record["probability"], dtype=float)
                for record in fit_records
                if int(record["inner_fold"]) == fold_index
            ]
            if (
                len(source) != 10
                or len(aggregate) == 0
                or len(aggregate) != len(outcomes)
                or len(aggregate) != len(row_ids)
                or len(row_ids) != len(set(row_ids))
                or not np.allclose(aggregate, np.vstack(source).mean(axis=0), atol=1e-15, rtol=0)
            ):
                aggregate_valid = False
                break
        if not aggregate_valid:
            continue
        config_id = str(result["config_id"])
        if config_id not in entries:
            raise AssertionError(f"{family}: unknown config_id in inner results")
        auc = float(result["mean_auroc"])
        loss = float(result["mean_log_loss"])
        if not (math.isfinite(auc) and math.isfinite(loss)):
            raise AssertionError(f"{family}/{config_id}: non-finite tuning metric")
        valid.append(
            {
                **dict(entries[config_id]),
                "mean_auroc": auc,
                "mean_log_loss": loss,
            }
        )
    if not valid:
        return []
    ranked: list[dict[str, Any]] = []
    remaining = valid.copy()
    while remaining:
        best_auc = max(item["mean_auroc"] for item in remaining)
        auc_equivalent = [
            item
            for item in remaining
            if best_auc - item["mean_auroc"] <= INNER_AUROC_EQUIVALENCE + NUMERIC_ATOL
        ]
        best_loss = min(item["mean_log_loss"] for item in auc_equivalent)
        loss_tied = [
            item
            for item in auc_equivalent
            if abs(item["mean_log_loss"] - best_loss) <= NUMERIC_ATOL
        ]
        chosen = min(loss_tied, key=lambda item: int(item["registry_index"]))
        ranked.append({**chosen, "pre_rank": len(ranked)})
        remaining.remove(chosen)
    return ranked


def validate_member_matrix(matrix: Any, member_names: Sequence[str], label: str) -> np.ndarray:
    return _stage06.validate_member_matrix(matrix, member_names, label)


def fit_weighted_voting(
    member_oof: Any, y: Any, member_names: Sequence[str] = BASE_FAMILIES
) -> tuple[np.ndarray, dict[str, Any]]:
    return _stage06.fit_weighted(member_oof, y, member_names)


def fit_stacking(
    member_oof: Any,
    y: Any,
    seed: int,
    C: float,
    member_names: Sequence[str] = BASE_FAMILIES,
) -> FittedModel:
    matrix = validate_member_matrix(member_oof, member_names, "Stacking OOF")
    stacker = Pipeline(
        [
            (
                "model",
                LogisticRegression(
                    penalty="l2",
                    C=float(C),
                    solver="lbfgs",
                    max_iter=5000,
                    tol=1e-6,
                    class_weight=None,
                    random_state=int(seed),
                ),
            )
        ]
    )
    # Stacking is an independent ensemble candidate, not the base Logistic
    # family.  It receives one fit attempt; any convergence signal fails the
    # candidate and must never inherit the base-family max_iter rescue.
    try:
        convergence, messages = _stage06._fit_attempt(
            stacker, matrix, np.asarray(y, dtype=int)
        )
    except _stage06.FitAttemptError as error:
        raise _stage06.FitAttemptError(
            f"Stacking: one-shot fit failed: {error}",
            family="Stacking",
            physical_attempts=1,
            stdout_tail=error.stdout_tail,
            stderr_tail=error.stderr_tail,
        ) from error
    if convergence:
        raise _stage06.FitAttemptError(
            f"Stacking: convergence warning fails closed without rescue: {convergence[0]}",
            family="Stacking",
            physical_attempts=1,
        )
    fitted = FittedModel(
        model=stacker,
        family="Stacking",
        audit={
            "family": "Stacking",
            "physical_attempts": 1,
            "rescue_used": False,
            "messages": sorted(set(messages)),
        },
    )
    if fitted.model.n_features_in_ != 14:
        raise AssertionError("Stacking did not receive exactly 14 probability columns")
    return fitted


def ensemble_probabilities(
    member_matrix: Any,
    *,
    weights: Any,
    stacker: Any,
    member_names: Sequence[str] = BASE_FAMILIES,
) -> dict[str, np.ndarray]:
    matrix = validate_member_matrix(member_matrix, member_names, "Ensemble prediction")
    checked_weight = _stage06.validate_weights(weights, "Weighted prediction weights")
    stacker_model = stacker.model if isinstance(stacker, FittedModel) else stacker
    soft = matrix.mean(axis=1)
    weighted = matrix @ checked_weight
    stacking = _stage06.positive_probability(
        stacker_model, matrix, "Stacking prediction"
    )
    if not np.allclose(soft, matrix.sum(axis=1) / 14.0, atol=1e-15, rtol=0):
        raise AssertionError("Soft Voting is not the exact 14-member mean")
    return {
        "Soft Voting": soft,
        "Weighted Voting": weighted,
        "Stacking": stacking,
    }


def fit_platt_calibrator(raw_cross_fitted_probability: Any, y: Any, seed: int) -> Any:
    """Fit the project-level two-parameter map on honest training OOF only."""

    return _stage06.fit_recalibrator(raw_cross_fitted_probability, y, int(seed))


def apply_platt_calibrator(calibrator: Any, raw_probability: Any) -> np.ndarray:
    return _stage06.apply_recalibrator(calibrator, raw_probability)


def _validate_binary_probability(y: Any, probability: Any) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(y, dtype=int).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    if target.shape != p.shape or target.size == 0:
        raise AssertionError("Outcome and probability lengths differ or are empty")
    if not np.array_equal(np.unique(target), np.asarray([0, 1])):
        raise AssertionError("Both outcome classes 0 and 1 are required")
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(p > 1):
        raise AssertionError("Probabilities must be finite and within [0,1]")
    return target, p


def threshold_metrics(y: Any, probability: Any, threshold: float) -> dict[str, Any]:
    target, p = _validate_binary_probability(y, probability)
    value = float(threshold)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise AssertionError("Threshold must be finite and in [0,1]")
    predicted = (p >= value).astype(int)
    tp = int(np.sum((target == 1) & (predicted == 1)))
    fn = int(np.sum((target == 1) & (predicted == 0)))
    tn = int(np.sum((target == 0) & (predicted == 0)))
    fp = int(np.sum((target == 0) & (predicted == 1)))

    def ratio(numerator: int, denominator: int) -> float | None:
        return numerator / denominator if denominator else None

    sensitivity = ratio(tp, tp + fn)
    specificity = ratio(tn, tn + fp)
    return {
        "threshold": value,
        "tp": tp,
        "fn": fn,
        "tn": tn,
        "fp": fp,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "youden_index": float(sensitivity + specificity - 1),
        "accuracy": (tp + tn) / len(target),
        "balanced_accuracy": float((sensitivity + specificity) / 2),
        "positive_predictive_value": ratio(tp, tp + fp),
        "negative_predictive_value": ratio(tn, tn + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def youden_threshold(y: Any, calibrated_probability: Any) -> dict[str, Any]:
    """Return an actual candidate threshold using all three frozen tie breaks."""

    target, p = _validate_binary_probability(y, calibrated_probability)
    candidates = np.unique(p)
    rows = [threshold_metrics(target, p, threshold) for threshold in candidates]
    best_youden = max(row["youden_index"] for row in rows)
    tied_1 = [
        row
        for row in rows
        if abs(row["youden_index"] - best_youden) <= NUMERIC_ATOL
    ]
    best_balance = min(
        abs(row["sensitivity"] - row["specificity"]) for row in tied_1
    )
    tied_2 = [
        row
        for row in tied_1
        if abs(
            abs(row["sensitivity"] - row["specificity"]) - best_balance
        )
        <= NUMERIC_ATOL
    ]
    median_threshold = float(np.median([row["threshold"] for row in tied_2]))
    chosen = min(
        tied_2,
        key=lambda row: (abs(row["threshold"] - median_threshold), row["threshold"]),
    )
    return {
        **chosen,
        "candidate_threshold_count": int(len(candidates)),
        "maximum_youden_tie_count": int(len(tied_1)),
        "balance_tie_count": int(len(tied_2)),
        "tie_break_median": median_threshold,
        "threshold_is_actual_candidate": bool(np.any(candidates == chosen["threshold"])),
        "descriptive_only": True,
    }


def probability_metrics(
    y: Any,
    probability: Any,
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Compute the frozen probability metrics and optional threshold metrics."""

    target, p = _validate_binary_probability(y, probability)
    calibrator = fit_platt_calibrator(p, target, seed=MASTER_SEED)
    intercept = float(calibrator.intercept_[0])
    slope = float(calibrator.coef_[0, 0])
    result: dict[str, Any] = {
        "n": int(len(target)),
        "outcome_1_n": int(target.sum()),
        "outcome_0_n": int(len(target) - target.sum()),
        "auroc": float(roc_auc_score(target, p)),
        "brier_score": float(brier_score_loss(target, p)),
        "log_loss": float(
            log_loss(
                target,
                np.clip(p, PROBABILITY_CLIP_EPSILON, 1 - PROBABILITY_CLIP_EPSILON),
                labels=[0, 1],
            )
        ),
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "calibration_distance": float(math.hypot(intercept, slope - 1.0)),
    }
    if threshold is not None:
        result["threshold_metrics"] = threshold_metrics(target, p, float(threshold))
    return result


__all__ = [
    "ALL_CANDIDATES",
    "BASE_FAMILIES",
    "ENSEMBLE_FAMILIES",
    "FamilyRefitExhaustedError",
    "FittedModel",
    "PredictionInvalidError",
    "apply_platt_calibrator",
    "ensemble_probabilities",
    "fit_model_fail_closed",
    "fit_platt_calibrator",
    "fit_predict_fail_closed",
    "fit_stacking",
    "fit_weighted_voting",
    "generate_frozen_registry",
    "load_frozen_registry",
    "make_model",
    "probability_metrics",
    "positive_probability",
    "rank_fully_valid_configurations",
    "refit_with_preranked_fallback",
    "threshold_metrics",
    "validate_member_matrix",
    "validate_registry",
    "youden_threshold",
]
