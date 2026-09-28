"""Frozen stage-04 model implementations for the integrated synthetic gate."""

from __future__ import annotations

import hashlib
import inspect
import io
import os
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Any

import numpy as np
from lightgbm import LGBMClassifier
from pygam import LogisticGAM, s
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sktime.classification.sklearn import RotationForest
from xgboost import XGBClassifier


for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    if os.environ.get(_thread_variable) != "1":
        raise RuntimeError(f"Single-thread contract violated: {_thread_variable} != 1")


EXPECTED_MEMBERS = (
    "Logistic",
    "GAM",
    "KNN",
    "RBF-SVM",
    "GaussianNB",
    "DecisionTree",
    "RandomForest",
    "ExtraTrees",
    "GBDT",
    "XGBoost",
    "LightGBM",
    "AdaBoost",
    "RotationForest",
    "MLP",
)
EXPECTED_CANDIDATES = EXPECTED_MEMBERS + (
    "Soft Voting",
    "Weighted Voting",
    "Stacking",
)
SIMPLEX_ATOL = 1e-6
LOG_LOSS_CLIP = 1e-6


def _subseed(seed: int, label: str) -> int:
    digest = hashlib.sha256(f"{seed}|{label}".encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big", signed=False)


class PyGAMBinaryClassifier(ClassifierMixin, BaseEstimator):
    def __init__(
        self,
        n_splines: int = 5,
        lam: float = 10.0,
        max_iter: int = 500,
        tol: float = 1e-5,
    ) -> None:
        self.n_splines = n_splines
        self.lam = lam
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X: Any, y: Any) -> "PyGAMBinaryClassifier":
        matrix = np.asarray(X, dtype=float)
        target = np.asarray(y, dtype=int)
        terms = s(0, n_splines=self.n_splines, spline_order=3)
        for feature_index in range(1, matrix.shape[1]):
            terms += s(feature_index, n_splines=self.n_splines, spline_order=3)
        self.model_ = LogisticGAM(
            terms=terms,
            lam=self.lam,
            max_iter=self.max_iter,
            tol=self.tol,
            verbose=False,
        ).fit(matrix, target)
        self.classes_ = np.asarray([0, 1], dtype=int)
        self.n_features_in_ = int(matrix.shape[1])
        return self

    def predict_proba(self, X: Any) -> Any:
        positive = np.asarray(
            self.model_.predict_proba(np.asarray(X, dtype=float)), dtype=float
        ).reshape(-1)
        return np.column_stack((1.0 - positive, positive))

    def predict(self, X: Any) -> Any:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _calibrated_svm(seed: int) -> Any:
    margin_model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "svc",
                SVC(
                    kernel="rbf",
                    C=1.0,
                    gamma="scale",
                    probability=False,
                    class_weight=None,
                    random_state=seed,
                ),
            ),
        ]
    )
    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=_subseed(seed, "svm_sigmoid_cv"),
    )
    parameters = inspect.signature(CalibratedClassifierCV).parameters
    kwargs: dict[str, Any] = {
        "method": "sigmoid",
        "cv": cv,
        "ensemble": False,
    }
    if "n_jobs" in parameters:
        kwargs["n_jobs"] = 1
    if "estimator" in parameters:
        kwargs["estimator"] = margin_model
    elif "base_estimator" in parameters:
        kwargs["base_estimator"] = margin_model
    else:
        raise RuntimeError("Unsupported CalibratedClassifierCV API")
    return CalibratedClassifierCV(**kwargs)


def _adaboost(seed: int) -> Any:
    tree = DecisionTreeClassifier(
        max_depth=1,
        min_samples_leaf=5,
        class_weight=None,
        random_state=_subseed(seed, "adaboost_tree"),
    )
    parameters = inspect.signature(AdaBoostClassifier).parameters
    kwargs: dict[str, Any] = {
        "n_estimators": 50,
        "learning_rate": 1.0,
        "random_state": seed,
    }
    if "estimator" in parameters:
        kwargs["estimator"] = tree
    elif "base_estimator" in parameters:
        kwargs["base_estimator"] = tree
    else:
        raise RuntimeError("Unsupported AdaBoostClassifier API")
    return AdaBoostClassifier(**kwargs)


def _rotation_forest(seed: int) -> Any:
    parameters = inspect.signature(RotationForest).parameters
    kwargs: dict[str, Any] = {
        "n_estimators": 100,
        "remove_proportion": 0.5,
        "n_jobs": 1,
        "random_state": seed,
    }
    if "min_group" in parameters and "max_group" in parameters:
        kwargs.update(min_group=3, max_group=3)
    elif "group" in parameters:
        kwargs["group"] = 3
    else:
        raise RuntimeError("Unsupported RotationForest group-size API")
    if "time_limit_in_minutes" in parameters:
        kwargs["time_limit_in_minutes"] = 0.0
    return RotationForest(**kwargs)


def make_models(seed: int) -> dict[str, Any]:
    models: dict[str, Any] = {
        "Logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        penalty="l2",
                        C=1.0,
                        solver="lbfgs",
                        max_iter=5000,
                        tol=1e-6,
                        class_weight=None,
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "GAM": PyGAMBinaryClassifier(),
        "KNN": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    KNeighborsClassifier(
                        n_neighbors=9,
                        weights="uniform",
                        p=2,
                        metric="minkowski",
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "RBF-SVM": _calibrated_svm(seed),
        "GaussianNB": Pipeline(
            [("scale", StandardScaler()), ("model", GaussianNB(var_smoothing=1e-9))]
        ),
        "DecisionTree": DecisionTreeClassifier(
            criterion="gini",
            splitter="best",
            max_depth=3,
            min_samples_leaf=5,
            ccp_alpha=0.0,
            class_weight=None,
            random_state=seed,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=500,
            max_features="sqrt",
            max_depth=None,
            min_samples_leaf=1,
            bootstrap=True,
            max_samples=None,
            class_weight=None,
            n_jobs=1,
            random_state=seed,
        ),
        "ExtraTrees": ExtraTreesClassifier(
            n_estimators=500,
            max_features="sqrt",
            max_depth=None,
            min_samples_leaf=1,
            bootstrap=False,
            class_weight=None,
            n_jobs=1,
            random_state=seed,
        ),
        "GBDT": GradientBoostingClassifier(
            loss="log_loss",
            learning_rate=0.1,
            n_estimators=100,
            max_depth=2,
            min_samples_leaf=10,
            subsample=1.0,
            random_state=seed,
        ),
        "XGBoost": XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            device="cpu",
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_child_weight=1,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            n_jobs=1,
            random_state=seed,
            verbosity=0,
        ),
        "LightGBM": LGBMClassifier(
            boosting_type="gbdt",
            device_type="cpu",
            objective="binary",
            n_estimators=100,
            learning_rate=0.1,
            num_leaves=15,
            max_depth=4,
            min_child_samples=20,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
            class_weight=None,
            deterministic=True,
            force_col_wise=True,
            n_jobs=1,
            random_state=seed,
            verbosity=-1,
        ),
        "AdaBoost": _adaboost(seed),
        "RotationForest": _rotation_forest(seed),
        "MLP": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(16,),
                        activation="relu",
                        solver="lbfgs",
                        alpha=1e-3,
                        max_iter=2000,
                        max_fun=50000,
                        tol=1e-5,
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
    if tuple(models) != EXPECTED_MEMBERS:
        raise AssertionError("Exact 14-member model order changed")
    return models


def positive_probability(model: Any, X: Any, label: str) -> np.ndarray:
    classes = np.asarray(getattr(model, "classes_", []))
    if classes.shape != (2,) or not np.array_equal(classes, np.asarray([0, 1])):
        raise AssertionError(f"{label}: classes_ is not [0, 1]")
    matrix = np.asarray(model.predict_proba(X), dtype=float)
    if matrix.shape != (len(X), 2):
        raise AssertionError(f"{label}: invalid predict_proba shape {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise AssertionError(f"{label}: non-finite probability")
    if np.any(matrix < -SIMPLEX_ATOL) or np.any(matrix > 1 + SIMPLEX_ATOL):
        raise AssertionError(f"{label}: probability outside [0,1]")
    if not np.allclose(matrix.sum(axis=1), 1.0, atol=SIMPLEX_ATOL, rtol=0):
        raise AssertionError(f"{label}: probability rows do not sum to one")
    return matrix[:, 1]


class FitAttemptError(AssertionError):
    def __init__(
        self,
        message: str,
        family: str,
        physical_attempts: int = 1,
        stdout_tail: str = "",
        stderr_tail: str = "",
    ) -> None:
        super().__init__(message)
        self.family = family
        self.physical_attempts = physical_attempts
        self.stdout_tail = stdout_tail[-500:]
        self.stderr_tail = stderr_tail[-500:]


class NumericalRescueError(FitAttemptError):
    pass


def _model_family(model: Any) -> str:
    if isinstance(model, PyGAMBinaryClassifier):
        return "GAM"
    if isinstance(model, Pipeline):
        final_step = list(model.named_steps.values())[-1]
        if isinstance(final_step, LogisticRegression):
            return "Logistic"
        if isinstance(final_step, KNeighborsClassifier):
            return "KNN"
        if isinstance(final_step, GaussianNB):
            return "GaussianNB"
        if isinstance(final_step, MLPClassifier):
            return "MLP"
    if isinstance(model, CalibratedClassifierCV):
        return "RBF-SVM"
    type_map = {
        DecisionTreeClassifier: "DecisionTree",
        RandomForestClassifier: "RandomForest",
        ExtraTreesClassifier: "ExtraTrees",
        GradientBoostingClassifier: "GBDT",
        XGBClassifier: "XGBoost",
        LGBMClassifier: "LightGBM",
        AdaBoostClassifier: "AdaBoost",
        RotationForest: "RotationForest",
    }
    for model_type, family in type_map.items():
        if isinstance(model, model_type):
            return family
    return type(model).__name__


def _rescue_family_and_parameter(model: Any):
    if isinstance(model, PyGAMBinaryClassifier):
        return "GAM", "max_iter", int(model.max_iter)
    if isinstance(model, Pipeline):
        final_step_name, final_step = list(model.named_steps.items())[-1]
        if isinstance(final_step, LogisticRegression):
            return "Logistic", f"{final_step_name}__max_iter", int(final_step.max_iter)
        if isinstance(final_step, MLPClassifier):
            return "MLP", f"{final_step_name}__max_iter", int(final_step.max_iter)
    return None, None, None


def _fit_attempt(model: Any, X_fit: Any, y_fit: Any):
    family = _model_family(model)
    stdout_capture = io.StringIO()
    stderr_capture = io.StringIO()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                model.fit(X_fit, y_fit)
    except Exception as error:
        raise FitAttemptError(
            f"{family}: fit raised {type(error).__name__}: {error}",
            family=family,
            physical_attempts=1,
            stdout_tail=stdout_capture.getvalue(),
            stderr_tail=stderr_capture.getvalue(),
        ) from error
    stdout_text = stdout_capture.getvalue().strip()
    stderr_text = stderr_capture.getvalue().strip()
    convergence_messages = [
        f"{warning.category.__name__}: {warning.message}"
        for warning in caught
        if issubclass(warning.category, ConvergenceWarning)
    ]
    for stream_name, stream_text in (("stdout", stdout_text), ("stderr", stderr_text)):
        if not stream_text:
            continue
        if family == "GAM" and stream_text.lower() == "did not converge":
            convergence_messages.append(f"{stream_name.upper()}: did not converge")
            continue
        raise FitAttemptError(
            f"{family}: unexpected non-empty captured {stream_name}: {stream_text[-500:]}",
            family=family,
            physical_attempts=1,
            stdout_tail=stdout_text,
            stderr_tail=stderr_text,
        )
    other_warnings = [
        f"{warning.category.__name__}: {warning.message}"
        for warning in caught
        if not issubclass(warning.category, ConvergenceWarning)
    ]
    return convergence_messages, other_warnings


def fit_predict(model: Any, X_fit: Any, y_fit: Any, X_new: Any, label: str):
    convergence, messages = _fit_attempt(model, X_fit, y_fit)
    physical_attempts = 1
    rescue_family = None
    rescue_used = False
    if convergence:
        family, parameter, original_limit = _rescue_family_and_parameter(model)
        if family is None:
            raise FitAttemptError(
                f"{label}: convergence signal in non-rescuable family: {convergence[0]}",
                family=_model_family(model),
                physical_attempts=1,
            )
        rescued = clone(model)
        rescued.set_params(**{parameter: original_limit * 2})
        try:
            rescue_convergence, rescue_messages = _fit_attempt(rescued, X_fit, y_fit)
        except FitAttemptError as error:
            raise NumericalRescueError(
                f"{label}: {family} rescue attempt failed: {error}",
                family=family,
                physical_attempts=2,
                stdout_tail=error.stdout_tail,
                stderr_tail=error.stderr_tail,
            ) from error
        physical_attempts = 2
        rescue_family = family
        rescue_used = True
        messages.extend(rescue_messages)
        if rescue_convergence:
            raise NumericalRescueError(
                f"{label}: {family} failed its single max_iter x2 rescue: "
                f"first={convergence[0]}; rescue={rescue_convergence[0]}",
                family=family,
                physical_attempts=2,
            )
        model = rescued
        messages.append(
            f"NUMERIC_RESCUE: {family} max_iter doubled once "
            f"from {original_limit} to {original_limit * 2}"
        )
    probability = positive_probability(model, X_new, label)
    audit = {
        "physical_attempts": physical_attempts,
        "rescue_used": rescue_used,
        "rescue_family": rescue_family,
    }
    return probability, sorted(set(messages)), audit


def validate_member_matrix(matrix: Any, member_names: Any, label: str) -> np.ndarray:
    names = tuple(member_names)
    if names != EXPECTED_MEMBERS or len(set(names)) != 14:
        raise AssertionError(f"{label}: exact ordered 14-member names are required")
    array = np.asarray(matrix, dtype=float)
    if array.ndim != 2 or array.shape[1] != 14:
        raise AssertionError(f"{label}: expected 14 member probability columns")
    if not np.isfinite(array).all() or np.any(array < 0) or np.any(array > 1):
        raise AssertionError(f"{label}: invalid member probabilities")
    return array


def validate_weights(weights: Any, label: str, tolerance: float = 1e-7) -> np.ndarray:
    weight = np.asarray(weights, dtype=float)
    if weight.shape != (14,):
        raise AssertionError(f"{label}: weight vector shape is not (14,)")
    if not np.isfinite(weight).all():
        raise AssertionError(f"{label}: non-finite weight")
    if np.any(weight < -tolerance):
        raise AssertionError(f"{label}: negative weight")
    if not np.isclose(weight.sum(), 1.0, atol=tolerance, rtol=0):
        raise AssertionError(f"{label}: weights do not sum to one")
    return weight


def fit_weighted(member_oof: Any, y: Any, member_names: Any):
    matrix = validate_member_matrix(member_oof, member_names, "Weighted OOF")
    target = np.asarray(y, dtype=int)

    def objective(weight: np.ndarray) -> float:
        p = np.clip(matrix @ weight, LOG_LOSS_CLIP, 1 - LOG_LOSS_CLIP)
        return float(log_loss(target, p, labels=[0, 1]))

    initial = np.full(14, 1 / 14, dtype=float)
    result = minimize(
        objective,
        initial,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * 14,
        constraints=[{"type": "eq", "fun": lambda weight: weight.sum() - 1.0}],
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    if not result.success:
        raise AssertionError(f"Weighted SLSQP failed: {result.message}")
    raw_weight = validate_weights(result.x, "Raw SLSQP solution")
    weight = np.asarray(raw_weight, dtype=float).copy()
    weight[np.abs(weight) <= 1e-7] = 0.0
    weight /= weight.sum()
    validate_weights(weight, "Normalized SLSQP solution", tolerance=1e-12)
    if objective(weight) > objective(initial) + 1e-10:
        raise AssertionError("Weighted solution is worse than uniform")
    return weight, {"iterations": int(result.nit), "message": str(result.message)}


def fit_stacking(member_oof: Any, y: Any, seed: int, member_names: Any) -> Any:
    matrix = validate_member_matrix(member_oof, member_names, "Stacking OOF")
    model = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-6,
        class_weight=None,
        random_state=seed,
    ).fit(matrix, np.asarray(y, dtype=int))
    if model.n_features_in_ != 14:
        raise AssertionError("Stacking did not receive exactly 14 probability columns")
    return model


def ensemble_probabilities(
    member_matrix: Any, member_names: Any, weight: Any, stacker: Any
):
    matrix = validate_member_matrix(member_matrix, member_names, "Ensemble prediction")
    checked_weight = validate_weights(weight, "Weighted prediction weights")
    soft = matrix.mean(axis=1)
    weighted = matrix @ checked_weight
    stacking = positive_probability(stacker, matrix, "Stacking prediction")
    if not np.allclose(soft, matrix.sum(axis=1) / 14.0, atol=1e-15, rtol=0):
        raise AssertionError("Soft Voting is not the exact 14-member mean")
    if not np.isfinite(weighted).all() or np.any(weighted < 0) or np.any(weighted > 1):
        raise AssertionError("Weighted Voting prediction outside [0,1]")
    return soft, weighted, stacking


def fit_recalibrator(raw_oof: Any, y: Any, seed: int) -> Any:
    raw = np.asarray(raw_oof, dtype=float).reshape(-1)
    if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1):
        raise AssertionError("Calibrator received invalid raw OOF probabilities")
    logit = np.log(
        np.clip(raw, LOG_LOSS_CLIP, 1 - LOG_LOSS_CLIP)
        / (1 - np.clip(raw, LOG_LOSS_CLIP, 1 - LOG_LOSS_CLIP))
    ).reshape(-1, 1)
    calibrator = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=5000,
        tol=1e-8,
        random_state=seed,
    ).fit(logit, np.asarray(y, dtype=int))
    if calibrator.n_features_in_ != 1:
        raise AssertionError("Project calibrator is not a two-parameter logistic map")
    return calibrator


def apply_recalibrator(calibrator: Any, raw_probability: Any) -> np.ndarray:
    raw = np.asarray(raw_probability, dtype=float).reshape(-1)
    if not np.isfinite(raw).all() or np.any(raw < 0) or np.any(raw > 1):
        raise AssertionError("Project recalibrator received illegal raw probability")
    clipped = np.clip(raw, LOG_LOSS_CLIP, 1 - LOG_LOSS_CLIP)
    logit = np.log(clipped / (1 - clipped)).reshape(-1, 1)
    calibrated = positive_probability(calibrator, logit, "Project calibration")
    return calibrated


def convergence_fail_closed_negative_tests() -> list[dict[str, Any]]:
    rng = np.random.default_rng(20260902)
    X = rng.normal(size=(120, 18))
    y = (2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.8 * X[:, 2] > 0).astype(int)
    cases = {
        "Logistic": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=100.0,
                        solver="lbfgs",
                        max_iter=1,
                        tol=1e-15,
                        random_state=20260902,
                    ),
                ),
            ]
        ),
        "GAM": PyGAMBinaryClassifier(
            n_splines=5,
            lam=0.1,
            max_iter=1,
            tol=1e-15,
        ),
        "MLP": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    MLPClassifier(
                        hidden_layer_sizes=(16,),
                        solver="lbfgs",
                        alpha=1e-4,
                        max_iter=1,
                        max_fun=50000,
                        tol=1e-15,
                        random_state=20260902,
                    ),
                ),
            ]
        ),
    }
    results: list[dict[str, Any]] = []
    for family, model in cases.items():
        try:
            fit_predict(model, X, y, X[:12], f"negative_low_max_iter/{family}")
        except NumericalRescueError as error:
            if error.family != family or error.physical_attempts != 2:
                raise AssertionError(f"{family}: rescue failure audit is incorrect")
            results.append(
                {
                    "family": family,
                    "initial_max_iter": 1,
                    "rescue_max_iter": 2,
                    "physical_attempts": error.physical_attempts,
                    "failed_closed_after_single_rescue": True,
                }
            )
        else:
            raise AssertionError(
                f"{family}: deliberately under-iterated negative test did not fail closed"
            )
    return results
