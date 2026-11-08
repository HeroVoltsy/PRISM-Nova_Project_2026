from pathlib import Path
from itertools import product
import json

import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

# ============================================================
# PRISM STRESS TESTING — LOSO / ORIGINAL CONNECTIVITY
# Reproduces the four model configurations in NOVAProject_Ablation.
# Stress is applied to held-out raw connectivity only.
# ============================================================

ROOT = Path.home() / "Projects" / "NOVAProject"
DATA = ROOT / "data" / "ABIDE_I" / "connectivity"
OUTPUT = ROOT / "outputs" / "stress_testing" / "loso_original"

X_FILE = DATA / "X_connectivity.npy"
METADATA_FILE = DATA / "metadata.csv"

OUTPUT.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_REPEATS = 5

NOISE_LEVELS = [0.01, 0.05, 0.10, 0.20]
DROPOUT_LEVELS = [0.05, 0.10, 0.20, 0.40]
CORRUPTION_LEVELS = [0.05, 0.10, 0.20, 0.40]
SHIFT_LEVELS = [0.25, 0.50, 1.00, 2.00]

# Systematic shift is applied to this fraction of connectivity edges.
SHIFT_FEATURE_FRACTION = 0.20

MODELS = {
    "LogisticRegression": Pipeline([
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(
            C=1.0,
            penalty="l2",
            solver="lbfgs",
            max_iter=3000,
            random_state=42,
        )),
    ]),
    "LassoClassifier": Pipeline([
        ("scaler", RobustScaler()),
        ("model", LogisticRegression(
            C=0.1,
            penalty="l1",
            solver="saga",
            max_iter=5000,
            tol=1e-3,
            random_state=42,
        )),
    ]),
    "RidgeClassifier": Pipeline([
        ("scaler", RobustScaler()),
        ("model", RidgeClassifier(
            alpha=1.0,
            class_weight=None,
        )),
    ]),
    "SupportVectorClassifier": Pipeline([
        ("scaler", RobustScaler()),
        ("model", SVC(
            C=1.0,
            kernel="rbf",
            gamma="scale",
            class_weight=None,
            probability=False,
            random_state=42,
        )),
    ]),
}


def get_score(fitted_model, X):
    if hasattr(fitted_model, "predict_proba"):
        return fitted_model.predict_proba(X)[:, 1]
    return fitted_model.decision_function(X)


def calculate_metrics(y_true, y_pred, y_score, baseline_pred=None):
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    result = {
        "N_Test": len(y_true),
        "Accuracy": accuracy_score(y_true, y_pred),
        "BalancedAccuracy": balanced_accuracy_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "Sensitivity": tp / (tp + fn) if tp + fn else np.nan,
        "Specificity": tn / (tn + fp) if tn + fp else np.nan,
        "ROC_AUC": (
            roc_auc_score(y_true, y_score)
            if len(np.unique(y_true)) == 2 else np.nan
        ),
        "PR_AUC": (
            average_precision_score(y_true, y_score)
            if len(np.unique(y_true)) == 2 else np.nan
        ),
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }

    if baseline_pred is None:
        result["PredictionFlipRate"] = 0.0
    else:
        result["PredictionFlipRate"] = np.mean(y_pred != baseline_pred)

    return result


def make_stress_conditions(X_train, X_test, seed):
    """
    Return deterministic stressed copies of X_test.
    All perturbations are defined in raw connectivity space.

    Gaussian noise:
        sigma * training feature SD, independently per cell.

    Feature dropout:
        selected edges are replaced by the training median,
        which maps to approximately zero after RobustScaler.

    Random feature corruption:
        selected participant-edge entries are replaced by values
        sampled from the corresponding edge's training distribution.

    Distribution shift:
        a fixed random 20% of edges receives a signed shift equal
        to severity * training SD for every held-out participant.
    """
    rng = np.random.default_rng(seed)
    n_test, n_features = X_test.shape
    train_sd = np.std(X_train, axis=0, ddof=0)
    train_median = np.median(X_train, axis=0)

    train_sd = np.where(np.isfinite(train_sd), train_sd, 0.0)
    train_median = np.where(np.isfinite(train_median), train_median, 0.0)

    conditions = []

    # Gaussian noise
    for severity in NOISE_LEVELS:
        X_stress = X_test.astype(np.float32, copy=True)
        noise = rng.normal(
            loc=0.0,
            scale=(severity * train_sd),
            size=X_stress.shape,
        ).astype(np.float32)
        X_stress += noise
        conditions.append(("gaussian_noise", severity, X_stress))

    # Feature dropout: same edge mask across all held-out participants.
    for severity in DROPOUT_LEVELS:
        X_stress = X_test.astype(np.float32, copy=True)
        n_drop = max(1, int(round(severity * n_features)))
        dropped = rng.choice(n_features, size=n_drop, replace=False)
        X_stress[:, dropped] = train_median[dropped].astype(np.float32)
        conditions.append(("feature_dropout", severity, X_stress))

    # Entry-wise corruption: replace selected participant-edge values
    # with empirical training values from the same edge.
    for severity in CORRUPTION_LEVELS:
        X_stress = X_test.astype(np.float32, copy=True)
        mask = rng.random(X_stress.shape) < severity
        row_idx, col_idx = np.where(mask)

        if len(row_idx):
            donor_rows = rng.integers(0, len(X_train), size=len(row_idx))
            X_stress[row_idx, col_idx] = X_train[donor_rows, col_idx]

        conditions.append(("random_feature_corruption", severity, X_stress))

    # Systematic distribution shift on a fixed random subset of edges.
    n_shift = max(1, int(round(SHIFT_FEATURE_FRACTION * n_features)))
    shift_features = rng.choice(
        n_features, size=n_shift, replace=False
    )
    shift_directions = rng.choice([-1.0, 1.0], size=n_shift)

    for severity in SHIFT_LEVELS:
        X_stress = X_test.astype(np.float32, copy=True)
        offsets = (
            severity
            * train_sd[shift_features]
            * shift_directions
        ).astype(np.float32)

        X_stress[:, shift_features] += offsets
        conditions.append(("distribution_shift", severity, X_stress))

    return conditions


def main():
    if not X_FILE.exists():
        raise FileNotFoundError(X_FILE)
    if not METADATA_FILE.exists():
        raise FileNotFoundError(METADATA_FILE)

    X = np.load(X_FILE, mmap_mode="r")
    metadata = pd.read_csv(METADATA_FILE)

    required = {"FILE_ID", "SITE_ID", "DX_GROUP"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"Missing metadata columns: {sorted(missing)}")

    if len(X) != len(metadata):
        raise ValueError(
            f"X has {len(X)} rows but metadata has {len(metadata)} rows."
        )

    if X.shape[1] != 19900:
        raise ValueError(
            f"Expected 19,900 connectivity features; found {X.shape[1]}."
        )

    if metadata[["FILE_ID", "SITE_ID", "DX_GROUP"]].isna().any().any():
        raise ValueError("Missing participant, site, or diagnosis metadata.")

    y = (metadata["DX_GROUP"].to_numpy() == 1).astype(np.int8)
    sites = metadata["SITE_ID"].to_numpy()
    file_ids = metadata["FILE_ID"].astype(str).to_numpy()

    if set(np.unique(y)) != {0, 1}:
        raise ValueError("Both ASD and control participants are required.")

    metric_rows = []
    prediction_frames = []

    unique_sites = np.unique(sites)

    print("Participants:", len(metadata))
    print("Features:", X.shape[1])
    print("Sites:", len(unique_sites))
    print("Models:", list(MODELS))
    print("Perturbation repeats:", N_REPEATS)

    for site_idx, site in enumerate(unique_sites, start=1):
        test_mask = sites == site
        train_idx = np.flatnonzero(~test_mask)
        test_idx = np.flatnonzero(test_mask)

        X_train = np.asarray(X[train_idx], dtype=np.float32)
        X_test = np.asarray(X[test_idx], dtype=np.float32)
        y_train = y[train_idx]
        y_test = y[test_idx]

        if len(np.unique(y_train)) != 2:
            raise ValueError(f"Training data for site {site} lacks a class.")
        if len(np.unique(y_test)) != 2:
            raise ValueError(f"Held-out site {site} lacks a class.")

        print(
            f"\n[{site_idx}/{len(unique_sites)}] {site} "
            f"| train={len(train_idx)} | test={len(test_idx)}"
        )

        for model_name, model in MODELS.items():
            # Fit only once per model and held-out site. The same fitted
            # estimator is reused for every perturbation replicate.
            fitted_model = clone(model)
            fitted_model.fit(X_train, y_train)

            baseline_pred = fitted_model.predict(X_test)
            baseline_score = get_score(fitted_model, X_test)

            baseline_metrics = calculate_metrics(
                y_test,
                baseline_pred,
                baseline_score,
            )

            for repeat in range(N_REPEATS):
                stress_seed = RANDOM_STATE + site_idx * 1000 + repeat

                metric_rows.append({
                    "Model": model_name,
                    "Site": site,
                    "StressType": "baseline",
                    "Severity": 0.0,
                    "Repeat": repeat,
                    "Seed": stress_seed,
                    **baseline_metrics,
                })

                prediction_frames.append(pd.DataFrame({
                    "Model": model_name,
                    "Site": site,
                    "StressType": "baseline",
                    "Severity": 0.0,
                    "Repeat": repeat,
                    "FILE_ID": file_ids[test_idx],
                    "y_true": y_test,
                    "y_pred": baseline_pred,
                    "y_score": baseline_score,
                }))

                conditions = make_stress_conditions(
                    X_train, X_test, stress_seed
                )

                for stress_type, severity, X_stress in conditions:
                    stressed_pred = fitted_model.predict(X_stress)
                    stressed_score = get_score(fitted_model, X_stress)

                    stressed_metrics = calculate_metrics(
                        y_test,
                        stressed_pred,
                        stressed_score,
                        baseline_pred=baseline_pred,
                    )

                    metric_rows.append({
                        "Model": model_name,
                        "Site": site,
                        "StressType": stress_type,
                        "Severity": severity,
                        "Repeat": repeat,
                        "Seed": stress_seed,
                        **stressed_metrics,
                    })

                    prediction_frames.append(pd.DataFrame({
                        "Model": model_name,
                        "Site": site,
                        "StressType": stress_type,
                        "Severity": severity,
                        "Repeat": repeat,
                        "FILE_ID": file_ids[test_idx],
                        "y_true": y_test,
                        "y_pred": stressed_pred,
                        "y_score": stressed_score,
                        "baseline_y_pred": baseline_pred,
                    }))

    metrics_df = pd.DataFrame(metric_rows)
    predictions_df = pd.concat(prediction_frames, ignore_index=True)

    # Add within-site/model/repeat degradation relative to the matching baseline.
    baseline = (
        metrics_df[metrics_df["StressType"] == "baseline"]
        [["Model", "Site", "Repeat", "Accuracy",
          "BalancedAccuracy", "ROC_AUC", "PR_AUC", "F1", "MCC"]]
        .rename(columns={
            "Accuracy": "Baseline_Accuracy",
            "BalancedAccuracy": "Baseline_BalancedAccuracy",
            "ROC_AUC": "Baseline_ROC_AUC",
            "PR_AUC": "Baseline_PR_AUC",
            "F1": "Baseline_F1",
            "MCC": "Baseline_MCC",
        })
    )

    metrics_df = metrics_df.merge(
        baseline,
        on=["Model", "Site", "Repeat"],
        how="left",
        validate="many_to_one",
    )

    for metric in [
        "Accuracy", "BalancedAccuracy", "ROC_AUC",
        "PR_AUC", "F1", "MCC",
    ]:
        metrics_df[f"{metric}_Delta"] = (
            metrics_df[metric] - metrics_df[f"Baseline_{metric}"]
        )

    # Average repeated perturbations within each site/model/condition.
    summary = (
        metrics_df.groupby(
            ["Model", "Site", "StressType", "Severity"],
            dropna=False
        )
        .agg(
            N_Test=("N_Test", "first"),
            Accuracy=("Accuracy", "mean"),
            BalancedAccuracy=("BalancedAccuracy", "mean"),
            ROC_AUC=("ROC_AUC", "mean"),
            PR_AUC=("PR_AUC", "mean"),
            F1=("F1", "mean"),
            MCC=("MCC", "mean"),
            Sensitivity=("Sensitivity", "mean"),
            Specificity=("Specificity", "mean"),
            PredictionFlipRate=("PredictionFlipRate", "mean"),
            Accuracy_Delta=("Accuracy_Delta", "mean"),
            BalancedAccuracy_Delta=("BalancedAccuracy_Delta", "mean"),
            ROC_AUC_Delta=("ROC_AUC_Delta", "mean"),
            PR_AUC_Delta=("PR_AUC_Delta", "mean"),
            F1_Delta=("F1_Delta", "mean"),
            MCC_Delta=("MCC_Delta", "mean"),
            Repeat_SD_BalancedAccuracy=("BalancedAccuracy", "std"),
        )
        .reset_index()
    )

    metrics_df.to_csv(OUTPUT / "loso_stress_metrics_by_repeat.csv", index=False)
    summary.to_csv(OUTPUT / "loso_stress_site_summary.csv", index=False)
    predictions_df.to_csv(OUTPUT / "loso_stress_predictions.csv", index=False)

    overall = (
        summary.groupby(["Model", "StressType", "Severity"])
        .agg(
            Sites=("Site", "nunique"),
            Mean_BalancedAccuracy=("BalancedAccuracy", "mean"),
            SD_BalancedAccuracy=("BalancedAccuracy", "std"),
            Mean_ROC_AUC=("ROC_AUC", "mean"),
            Mean_Accuracy_Delta=("Accuracy_Delta", "mean"),
            Mean_BalancedAccuracy_Delta=("BalancedAccuracy_Delta", "mean"),
            Mean_ROC_AUC_Delta=("ROC_AUC_Delta", "mean"),
            Mean_PredictionFlipRate=("PredictionFlipRate", "mean"),
        )
        .reset_index()
    )

    overall.to_csv(OUTPUT / "loso_stress_overall_summary.csv", index=False)

    audit = {
        "evaluation": "Leave-one-site-out",
        "representation": "Original 19,900 Fisher-z connectivity features",
        "participants": int(len(metadata)),
        "features": int(X.shape[1]),
        "sites": int(len(unique_sites)),
        "models": list(MODELS),
        "repeats": N_REPEATS,
        "random_state": RANDOM_STATE,
        "gaussian_noise": NOISE_LEVELS,
        "feature_dropout": DROPOUT_LEVELS,
        "random_feature_corruption": CORRUPTION_LEVELS,
        "distribution_shift_sd": SHIFT_LEVELS,
        "distribution_shift_feature_fraction": SHIFT_FEATURE_FRACTION,
        "training_policy": "Fit once per model and held-out site on unperturbed training data; never fit on perturbed test data.",
        "pca": "Not included: no PCA configuration is defined in the supplied training notebook.",
    }

    with open(OUTPUT / "stress_test_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    print("\nStress testing complete.")
    print("Output:", OUTPUT)
    print("\nOverall summary:")
    print(overall.to_string(index=False))


if __name__ == "__main__":
    main()
