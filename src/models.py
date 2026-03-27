"""
Models – logistic regression (physics-guided & attribute-only) and
Random Forest baseline.

Physics-guided model:
    p_ign(t) = σ(a · R_phys(t) + b)
    where σ is the sigmoid function.

Trained via binary cross-entropy (maximum likelihood).
"""

import os
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV


# ══════════════════════════════════════════════════════════════════
# Model training
# ══════════════════════════════════════════════════════════════════

def train_physics_logistic(X_train, y_train, C=1.0):
    """
    Logistic regression on the single physics-guided feature R_phys.

    p_ign = σ(a · R_phys + b)

    Parameters
    ----------
    X_train : ndarray (n, 1) – R_phys values
    y_train : ndarray (n,)   – binary labels

    Returns
    -------
    model : fitted LogisticRegression
    """
    model = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)

    a = model.coef_[0, 0]
    b = model.intercept_[0]
    print(f"  Physics logistic: a={a:.4f}, b={b:.4f}")
    return model


def train_attribute_logistic(X_train, y_train, C=1.0):
    """
    Logistic regression on the full normalised feature vector
    (T̃, RH̃, Ũ, SM̃, NDVI~, NDWI~, θ~, FRP~_hist, Count~_hist).

    No physics composition — used as a baseline.
    """
    model = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    print(f"  Attribute logistic: {X_train.shape[1]} features, "
          f"coefs summing to {model.coef_.sum():.4f}")
    return model


def train_random_forest(X_train, y_train, n_estimators=200, max_depth=10):
    """
    Random Forest baseline on full feature vector.

    Parameters
    ----------
    n_estimators : int
    max_depth : int – limit depth to avoid overfitting

    Returns
    -------
    model : fitted RandomForestClassifier
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"  Random Forest: {n_estimators} trees, depth={max_depth}")
    return model


def train_xgboost(X_train, y_train, n_estimators=200, max_depth=6,
                  learning_rate=0.1):
    """
    XGBoost gradient boosting baseline on full feature vector.

    Parameters
    ----------
    n_estimators : int
    max_depth : int
    learning_rate : float

    Returns
    -------
    model : fitted XGBClassifier
    """
    from xgboost import XGBClassifier

    # Compute scale_pos_weight for class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / max(n_pos, 1)

    model = XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        scale_pos_weight=scale,
        random_state=42,
        use_label_encoder=False,
        eval_metric="logloss",
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_train, y_train)
    print(f"  XGBoost: {n_estimators} trees, depth={max_depth}, lr={learning_rate}")
    return model


def train_lightgbm(X_train, y_train, n_estimators=200, max_depth=6,
                   learning_rate=0.1, num_leaves=31):
    """
    LightGBM gradient boosting on full feature vector.

    LightGBM is typically faster than XGBoost for large datasets and
    handles categorical features natively.

    Parameters
    ----------
    n_estimators : int
    max_depth : int
    learning_rate : float
    num_leaves : int

    Returns
    -------
    model : fitted LGBMClassifier
    """
    from lightgbm import LGBMClassifier

    # Compute scale_pos_weight for class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale = n_neg / max(n_pos, 1)

    model = LGBMClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        num_leaves=num_leaves,
        scale_pos_weight=scale,
        random_state=42,
        verbose=-1,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print(f"  LightGBM: {n_estimators} trees, depth={max_depth}, "
          f"leaves={num_leaves}, lr={learning_rate}")
    return model


def train_weather_logistic(X_train, y_train, C=1.0):
    """
    Weather-only logistic regression baseline.
    Uses only T̃, RH̃, Ũ (temperature, humidity, wind) — no vegetation or history.

    This isolates the contribution of weather vs vegetation/fire-history features.
    """
    model = LogisticRegression(
        C=C,
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
    )
    model.fit(X_train, y_train)
    print(f"  Weather-only logistic: {X_train.shape[1]} features, "
          f"coefs={model.coef_[0].round(4).tolist()}")
    return model


# ══════════════════════════════════════════════════════════════════
# Model persistence
# ══════════════════════════════════════════════════════════════════

def save_model(model, path, name="model"):
    """Save a trained model to disk via joblib (faster & safer than pickle)."""
    os.makedirs(path, exist_ok=True)
    filepath = os.path.join(path, f"{name}.joblib")
    joblib.dump(model, filepath, compress=3)
    print(f"  Saved model -> {filepath}")


def load_model(path, name="model"):
    """Load a trained model from disk."""
    # Try joblib first, fall back to pickle for legacy models
    filepath_joblib = os.path.join(path, f"{name}.joblib")
    filepath_pkl = os.path.join(path, f"{name}.pkl")

    if os.path.exists(filepath_joblib):
        return joblib.load(filepath_joblib)
    elif os.path.exists(filepath_pkl):
        import pickle
        with open(filepath_pkl, "rb") as f:
            return pickle.load(f)
    else:
        raise FileNotFoundError(f"No model found at {filepath_joblib} or {filepath_pkl}")


# ══════════════════════════════════════════════════════════════════
# Prediction
# ══════════════════════════════════════════════════════════════════

def predict_proba(model, X):
    """
    Return ignition probability p_ign for new data.

    Parameters
    ----------
    model : fitted sklearn classifier
    X : ndarray

    Returns
    -------
    probabilities : ndarray (n,) – P(ignition=1)
    """
    return model.predict_proba(X)[:, 1]


# ── Self-test ─────────────────────────────────────────────────────
if __name__ == "__main__":
    np.random.seed(0)
    n = 500
    X = np.random.rand(n, 1)
    y = (X[:, 0] > 0.6).astype(int)

    m = train_physics_logistic(X, y)
    p = predict_proba(m, X)
    assert p.shape == (n,)
    assert 0 <= p.min() and p.max() <= 1
    print("Models self-test passed ✓")
