"""Planning analytics built on top of the raw effort prediction."""

from __future__ import annotations

import numpy as np
import pandas as pd

from effort_schema import BASE_KEYS, FEATURE_BY_KEY, PHASES, clamp


# --------------------------------------------------------------------------
# Monte Carlo risk simulation
# --------------------------------------------------------------------------
def monte_carlo(service, values: dict, n: int = 1500, input_noise: float = 0.15,
                model_cv: float = 0.18, seed: int = 11) -> np.ndarray:
    """Propagate input uncertainty + model noise into an effort distribution.

    Scope-like inputs are perturbed with a truncated normal (estimates of size
    are themselves estimates), then a lognormal model-error multiplier is
    applied so the result never goes negative and stays right-skewed - which is
    how software overruns actually behave.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        row = dict(values)
        for key in BASE_KEYS:
            f = FEATURE_BY_KEY[key]
            if key in ("YearEnd", "Language"):
                continue
            span = max(abs(values[key]) * input_noise, 1.0)
            row[key] = clamp(key, round(rng.normal(values[key], span)))
        rows.append(row)

    base = service.predict(pd.DataFrame(rows))
    sigma_log = np.sqrt(np.log(1 + model_cv ** 2))
    multiplier = rng.lognormal(-0.5 * sigma_log ** 2, sigma_log, size=n)  # mean 1.0
    return np.maximum(base * multiplier, 0.0)


def percentiles(samples: np.ndarray) -> dict:
    qs = [10, 50, 80, 90, 95]
    return {f"P{q}": float(np.percentile(samples, q)) for q in qs}


def probability_within(samples: np.ndarray, budget_hours: float) -> float:
    if budget_hours <= 0:
        return 0.0
    return float((samples <= budget_hours).mean())


# --------------------------------------------------------------------------
# Schedule and staffing
# --------------------------------------------------------------------------
def schedule(effort_hours: float, team_size: int, hours_per_month: int = 160,
             communication_penalty: float = 0.02) -> dict:
    """Duration with a Brooks-style communication overhead.

    Each extra pair of people adds coordination cost, so effective capacity
    grows slower than head-count: capacity = N / (1 + k*N*(N-1)/2).
    """
    n = max(int(team_size), 1)
    pairs = n * (n - 1) / 2
    efficiency = 1.0 / (1.0 + communication_penalty * pairs)
    effective = n * efficiency
    person_months = effort_hours / hours_per_month
    months = person_months / max(effective, 1e-9)
    # Empirical compression floor: schedules below ~75% of nominal rarely hold.
    nominal = 2.5 * (person_months ** (1 / 3)) if person_months > 0 else 0.0
    return dict(
        person_months=person_months,
        effective_team=effective,
        efficiency=efficiency,
        duration_months=months,
        nominal_months=nominal,
        compression=(months / nominal) if nominal > 0 else 1.0,
        feasible=months >= 0.75 * nominal,
    )


def staffing_curve(effort_hours: float, duration_months: float, steps: int = 60) -> pd.DataFrame:
    """Rayleigh (Putnam/Norden) staffing profile over the project timeline."""
    if duration_months <= 0:
        return pd.DataFrame(columns=["Month", "Staff"])
    td = duration_months / 1.8  # peak occurs before the end
    t = np.linspace(0.01, duration_months, steps)
    shape = (t / td ** 2) * np.exp(-(t ** 2) / (2 * td ** 2))
    area = np.trapezoid(shape, t) if hasattr(np, "trapezoid") else np.trapz(shape, t)
    staff = shape / max(area, 1e-9) * (effort_hours / 160) / max(duration_months, 1e-9) * duration_months
    return pd.DataFrame({"Month": t, "Staff": staff})


def phase_plan(effort_hours: float, duration_months: float, rate: float) -> pd.DataFrame:
    rows, start = [], 0.0
    for name, share in PHASES:
        hours = effort_hours * share
        length = duration_months * share
        rows.append(dict(Phase=name, Share=f"{share:.0%}", Hours=round(hours),
                         Months=round(length, 2), Start=round(start, 2),
                         End=round(start + length, 2), Cost=round(hours * rate)))
        start += length
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Risk scoring
# --------------------------------------------------------------------------
def risk_profile(values: dict, prediction, sched: dict) -> pd.DataFrame:
    """Transparent, rule-based risk register scored 0-100 (higher = riskier)."""
    exp_score = values["TeamExp"] + values["ManagerExp"]
    scope = values["PointsAjust"]
    rows = [
        ("Team & manager experience", _scale(exp_score, 11, -2),
         "Low combined experience raises rework and ramp-up cost."),
        ("Scope size", _scale(scope, 60, 1100),
         "Larger function-point counts grow non-linearly in integration effort."),
        ("Estimate uncertainty", _scale(prediction.relative_spread, 0.15, 1.2),
         "A wide prediction interval means the model has seen little like this."),
        ("Schedule compression", _scale(sched["compression"], 1.3, 0.6),
         "Compressing below the nominal schedule inflates effort sharply."),
        ("Coordination overhead", _scale(sched["efficiency"], 1.0, 0.5),
         "Large teams lose capacity to communication links."),
        ("Technical complexity", _scale(values["Adjustment"], 5, 52),
         "A high adjustment factor signals demanding technical requirements."),
    ]
    df = pd.DataFrame(rows, columns=["Factor", "Score", "Why it matters"])
    df["Level"] = pd.cut(df["Score"], [-0.1, 33, 66, 100.1], labels=["Low", "Medium", "High"])
    return df


def _scale(value, good, bad) -> float:
    """Map a value onto 0-100 risk, where `good` scores 0 and `bad` scores 100."""
    if good == bad:
        return 50.0
    x = (value - good) / (bad - good)
    return float(np.clip(x, 0, 1) * 100)


def overall_risk(df: pd.DataFrame) -> tuple:
    score = float(df["Score"].mean())
    if score < 33:
        return score, "Low", "#34b26b"
    if score < 66:
        return score, "Medium", "#d9a520"
    return score, "High", "#d1495b"


def classify_size(hours: float) -> tuple:
    for limit, name, color in ((2000, "Small", "#8fdcaa"), (5000, "Medium", "#5fc98a"),
                               (10000, "Large", "#34b26b"), (float("inf"), "Very Large", "#1b7449")):
        if hours < limit:
            return name, color
    return "Very Large", "#1b7449"
