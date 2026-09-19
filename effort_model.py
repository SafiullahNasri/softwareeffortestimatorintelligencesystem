"""Model loading, prediction, uncertainty quantification and explanation."""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from effort_schema import BASE_KEYS, BASELINE, FEATURE_BY_KEY, MODEL_COLUMNS


# --------------------------------------------------------------------------
# Feature engineering (must mirror the training pipeline exactly)
# --------------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["AdjustmentImpact"] = out["PointsNonAdjust"] - out["PointsAjust"]
    out["ExperienceScore"] = out["TeamExp"] + out["ManagerExp"]
    return out[MODEL_COLUMNS]


@dataclass
class Prediction:
    point: float
    low: float
    high: float
    sigma: float
    method: str
    samples: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def relative_spread(self) -> float:
        return (self.high - self.low) / max(self.point, 1e-9)


class ModelService:
    """Wraps the pickled regressor with the extras a production app needs."""

    DEFAULT_CV = 0.22  # fallback coefficient of variation when the model exposes no ensemble

    def __init__(self, model):
        self.model = model
        self.members = self._ensemble_members(model)

    # ---------------------------------------------------------------- load
    @classmethod
    def from_path(cls, path: str = "model.pkl") -> "ModelService":
        with open(path, "rb") as fh:
            return cls(pickle.load(fh))

    @staticmethod
    def _ensemble_members(model):
        """Return per-estimator predictors for bagged ensembles, else None.

        Only bagging-style ensembles (forest, extra trees, bagging) give
        meaningful spread: their members are independent estimates of the same
        target. Boosted ensembles are additive, so their trees are not usable
        this way and we fall back to a residual-based interval.
        """
        name = type(model).__name__.lower()
        if not any(tag in name for tag in ("forest", "extratrees", "bagging")):
            return None
        members = getattr(model, "estimators_", None)
        if not members or not hasattr(members[0], "predict"):
            return None
        return members

    # ------------------------------------------------------------- predict
    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return np.maximum(0.0, np.asarray(self.model.predict(engineer(df)), dtype=float))

    def predict_one(self, values: dict) -> float:
        return float(self.predict(pd.DataFrame([values]))[0])

    def predict_with_interval(self, values: dict, confidence: float = 0.80) -> Prediction:
        row = pd.DataFrame([values])
        point = float(self.predict(row)[0])
        alpha = (1 - confidence) / 2

        if self.members is not None:
            # Sub-estimators were fitted without feature names, so pass raw values.
            X = engineer(row).to_numpy(dtype=float)
            try:
                preds = np.array([float(m.predict(X)[0]) for m in self.members])
                preds = np.maximum(preds, 0.0)
                low, high = np.quantile(preds, [alpha, 1 - alpha])
                sigma = float(preds.std(ddof=1)) if len(preds) > 1 else 0.0
                return Prediction(point, float(low), float(high), sigma,
                                  f"ensemble spread over {len(preds)} estimators", preds)
            except Exception:
                pass

        # Fallback: lognormal band around the point estimate. Unknown experience
        # values (-1) widen the band, mirroring real estimation uncertainty.
        cv = self.DEFAULT_CV
        if values.get("TeamExp", 0) < 0:
            cv += 0.05
        if values.get("ManagerExp", 0) < 0:
            cv += 0.05
        sigma_log = np.sqrt(np.log(1 + cv ** 2))
        z = 1.2816 if confidence >= 0.79 else 1.6449
        low = point * np.exp(-z * sigma_log)
        high = point * np.exp(z * sigma_log)
        return Prediction(point, float(low), float(high), point * cv,
                          f"lognormal band (CV {cv:.0%})")

    # --------------------------------------------------------- explanation
    def shapley(self, values: dict, n_permutations: int = 80, seed: int = 7) -> pd.DataFrame:
        """Monte-Carlo Shapley attribution against the mid-range baseline.

        Every permutation walks from the baseline vector to the actual project,
        adding one feature at a time; a feature's contribution is the average
        jump in predicted effort caused by switching it on. Contributions sum to
        prediction - baseline_prediction, so the chart is additive and honest.
        All intermediate states are scored in a single batched predict call.
        """
        rng = np.random.default_rng(seed)
        keys = BASE_KEYS
        rows, index = [], []

        for p in range(n_permutations):
            order = rng.permutation(len(keys))
            state = dict(BASELINE)
            rows.append(dict(state))
            index.append((p, -1))
            for pos in order:
                state[keys[pos]] = values[keys[pos]]
                rows.append(dict(state))
                index.append((p, int(pos)))

        preds = self.predict(pd.DataFrame(rows))
        contrib = np.zeros(len(keys))
        prev = 0.0
        for i, (_, pos) in enumerate(index):
            if pos == -1:
                prev = preds[i]
                continue
            contrib[pos] += preds[i] - prev
            prev = preds[i]
        contrib /= n_permutations

        return (
            pd.DataFrame({
                "Feature": [FEATURE_BY_KEY[k].label for k in keys],
                "Key": keys,
                "Value": [values[k] for k in keys],
                "Baseline": [BASELINE[k] for k in keys],
                "Contribution": contrib,
            })
            .sort_values("Contribution", key=np.abs, ascending=False)
            .reset_index(drop=True)
        )

    def baseline_prediction(self) -> float:
        return self.predict_one(dict(BASELINE))

    def sweep(self, values: dict, key: str, points: int = 45) -> pd.DataFrame:
        f = FEATURE_BY_KEY[key]
        grid = np.unique(np.linspace(f.lo, f.hi, points).round().astype(int))
        frame = pd.DataFrame([{**values, key: int(v)} for v in grid])
        frame["Effort"] = self.predict(frame)
        return frame[[key, "Effort"]]

    def interaction_grid(self, values: dict, key_x: str, key_y: str, n: int = 22) -> tuple:
        fx, fy = FEATURE_BY_KEY[key_x], FEATURE_BY_KEY[key_y]
        xs = np.unique(np.linspace(fx.lo, fx.hi, n).round().astype(int))
        ys = np.unique(np.linspace(fy.lo, fy.hi, n).round().astype(int))
        rows = [{**values, key_x: int(x), key_y: int(y)} for y in ys for x in xs]
        z = self.predict(pd.DataFrame(rows)).reshape(len(ys), len(xs))
        return xs, ys, z

    # ------------------------------------------------------------- introspect
    def info(self) -> dict:
        m = self.model
        data = {
            "Model class": type(m).__name__,
            "Module": type(m).__module__,
            "Expects features": len(MODEL_COLUMNS),
            "Uncertainty source": "ensemble members" if self.members is not None else "residual band",
        }
        if self.members is not None:
            data["Ensemble size"] = len(self.members)
        for attr in ("n_estimators", "max_depth", "learning_rate", "alpha", "n_features_in_"):
            if hasattr(m, attr):
                data[attr] = getattr(m, attr)
        return data

    def global_importance(self) -> pd.DataFrame | None:
        imp = getattr(self.model, "feature_importances_", None)
        if imp is None:
            coef = getattr(self.model, "coef_", None)
            if coef is None:
                return None
            imp = np.abs(np.ravel(coef))
        imp = np.ravel(imp)
        if len(imp) != len(MODEL_COLUMNS):
            return None
        return (
            pd.DataFrame({"Feature": MODEL_COLUMNS, "Importance": imp})
            .sort_values("Importance", ascending=False)
            .reset_index(drop=True)
        )
