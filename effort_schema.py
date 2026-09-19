"""Single source of truth for the model's feature contract."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    key: str
    label: str
    lo: int
    hi: int
    default: int
    help: str = ""


BASE_FEATURES = [
    Feature("TeamExp", "Team Experience", -1, 4, 2, "Years of team experience (-1 = unknown)."),
    Feature("ManagerExp", "Manager Experience", -1, 7, 3, "Years of project-manager experience (-1 = unknown)."),
    Feature("YearEnd", "Year End", 82, 88, 86, "Year the project ended (2-digit, dataset era)."),
    Feature("Length", "Project Length (months)", 1, 39, 10, "Calendar duration of the project."),
    Feature("Transactions", "Transactions", 9, 886, 150, "Count of basic logical transactions."),
    Feature("Entities", "Entities", 7, 387, 100, "Number of entities in the data model."),
    Feature("PointsNonAdjust", "Points Non Adjusted", 73, 1127, 270, "Unadjusted function points."),
    Feature("Adjustment", "Adjustment Factor", 5, 52, 28, "Technical complexity adjustment value."),
    Feature("PointsAjust", "Points Adjusted", 62, 1116, 255, "Adjusted function points."),
    Feature("Language", "Language", 1, 3, 1, "Programming-language generation code (1-3)."),
]

BASE_KEYS = [f.key for f in BASE_FEATURES]
ENGINEERED_KEYS = ["AdjustmentImpact", "ExperienceScore"]
MODEL_COLUMNS = BASE_KEYS + ENGINEERED_KEYS
FEATURE_BY_KEY = {f.key: f for f in BASE_FEATURES}

# Baseline reference vector used for Shapley attribution (mid-range of each feature).
BASELINE = {f.key: int(round((f.lo + f.hi) / 2)) for f in BASE_FEATURES}

PRESETS = {
    "Small project": dict(TeamExp=1, ManagerExp=2, YearEnd=84, Length=5, Transactions=60,
                          Entities=40, PointsNonAdjust=120, Adjustment=20, PointsAjust=110, Language=1),
    "Medium project": dict(TeamExp=2, ManagerExp=3, YearEnd=86, Length=10, Transactions=150,
                           Entities=100, PointsNonAdjust=270, Adjustment=28, PointsAjust=255, Language=1),
    "Large project": dict(TeamExp=3, ManagerExp=5, YearEnd=87, Length=24, Transactions=500,
                          Entities=250, PointsNonAdjust=800, Adjustment=40, PointsAjust=780, Language=2),
    "Inexperienced team": dict(TeamExp=-1, ManagerExp=0, YearEnd=86, Length=14, Transactions=300,
                               Entities=160, PointsNonAdjust=500, Adjustment=35, PointsAjust=480, Language=1),
}

# Phase distribution used for the delivery plan (classic waterfall-style split).
PHASES = [
    ("Requirements", 0.12),
    ("Design", 0.18),
    ("Implementation", 0.37),
    ("Testing & QA", 0.24),
    ("Deployment & Handover", 0.09),
]


def clamp(key: str, value) -> int:
    f = FEATURE_BY_KEY[key]
    return int(min(max(int(value), f.lo), f.hi))


def validate(values: dict) -> list:
    """Return a list of (severity, message) consistency warnings."""
    issues = []
    if values["PointsAjust"] > values["PointsNonAdjust"] * 1.4:
        issues.append(("warning", "Adjusted points far exceed non-adjusted points - verify the inputs."))
    if values["PointsNonAdjust"] > values["PointsAjust"] * 2:
        issues.append(("warning", "Adjustment is removing more than half of the function points."))
    if values["TeamExp"] < 0 or values["ManagerExp"] < 0:
        issues.append(("info", "An experience value of -1 means 'unknown' and widens the uncertainty band."))
    if values["Length"] <= 2 and values["PointsAjust"] > 500:
        issues.append(("warning", "A very large scope in under three months is an aggressive schedule."))
    if values["Transactions"] > 0 and values["Entities"] / max(values["Transactions"], 1) > 3:
        issues.append(("info", "Entities heavily outnumber transactions - the data model may be over-specified."))
    return issues
