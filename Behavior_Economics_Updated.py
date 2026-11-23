"""
Behavioral economics: how well can simple data models predict STUDENT
purchasing choices from a short survey?

HOW TO RUN (VS Code)
--------------------
1. Put this file, app.py, and the Google Forms Excel export in one folder.
   Any .xlsx in the folder whose name contains "Responses" is found
   automatically, so the exact file name does not matter.
2. Open the folder in VS Code, open a terminal (Ctrl+`), and run:
       pip install -r requirements.txt
       python Behavior_Economics_Updated.py
   Optional arguments:
       python Behavior_Economics_Updated.py --data "path/to/file.xlsx"
       python Behavior_Economics_Updated.py --sample all        # everyone
       python Behavior_Economics_Updated.py --sample student_age # under 26
       python Behavior_Economics_Updated.py --quick              # fast test run
3. Results go to outputs/<sample>/ . The Streamlit dashboard reads
   outputs/students/ by default:  streamlit run app.py

SAMPLES
-------
The survey did not ask whether a respondent is a student, so age is used as
a proxy. The form was distributed through a high school, so respondents
under 18 are treated as the student sample (PRIMARY analysis).
    students     Under 18 only                         (primary)
    student_age  Under 18 + 18-25                       (sensitivity)
    all          Every valid response, age as a feature (sensitivity / comparison)

WHAT CHANGED FROM THE PREVIOUS VERSION (summary; details in the paper)
---------------------------------------------------------------------
* Default data path no longer breaks when the file name differs.
* Analysis restricted to students by default; non-student rows are kept in
  the audit files instead of silently mixed in.
* Logistic regression now uses class_weight="balanced" like the random forest
  and a fixed default penalty (no inner tuning loop).
  Before, it almost always predicted the majority class (balanced accuracy
  about 0.50), which is why the model looked like it "didn't work".
* New "single-question" model per outcome: one theory-matched survey item.
  This tests whether the full model adds anything over the simplest model.
* Paired, corrected confidence intervals compare models directly.
* Odds ratios now come from an UNPENALISED logistic regression, per 1 point
  on the 1-5 scale (easier to interpret than per standard deviation), with
  Wald and bootstrap 95% intervals. The old version used a penalised model,
  which shrinks odds ratios toward 1.
* 1,000 permutations by default (100 limited the smallest p-value to 0.0099).
* Write-in answers to the scenario questions are coded as "Conditional /
  depends" and reported, instead of disappearing.
* A JSON summary is saved so the paper and dashboard use the same numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    RepeatedStratifiedKFold, StratifiedKFold,
    cross_validate, learning_curve, permutation_test_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
HERE = Path(__file__).resolve().parent

SEED = 42
N_SPLITS = 5
N_REPEATS = 10
N_PERMUTATIONS = 1000
N_BOOT = 1000
N_JOBS = -1
MIN_CLASS_COUNT = 20

CLEAR_SPAM_PHRASES = [
    "surveyswap.io",
    "start drop shipping prodects",
    "what yall know about drop shipping",
]

FREQ_ORDER = {"rarely": 1, "occasionally": 2, "regularly": 3, "frequently": 4}
KNOWN_AGES = {"under 18", "18-25", "26-40", "40+"}
LIKERT = ["price", "loyal", "novel", "cheaper", "rectend"]

SAMPLES = {
    "students": {"under 18"},
    "student_age": {"under 18", "18-25"},
    "all": KNOWN_AGES,
}

FEATURE_LABELS = {
    "age_18plus": "Age 18+ (vs under 18)",
    "freq_ord": "Non-essential purchase frequency (1-4)",
    "price": "Price importance (1-5)",
    "loyal": "Brand loyalty (1-5)",
    "novel": "Willingness to try unfamiliar products (1-5)",
    "cheaper": "Tendency to choose cheaper alternatives (1-5)",
    "rectend": "Likelihood of following recommendations (1-5)",
}

# (target column, positive label, negative label, primary?, single-item predictor)
# The single item is the survey question whose wording most directly matches
# the scenario. It was chosen on wording alone (face validity).
TARGETS = {
    "brand_switch": ("t_brand_switch", "Switch to cheaper Brand B",
                     "Stay with Brand A", True, "cheaper"),
    "snack_reaction": ("t_snack_reaction", "Cut back / switch / stop",
                       "Still buy at $3", True, "price"),
    "rec_follow": ("t_rec_follow", "Follow cheaper recommendation",
                   "Ignore it", True, "rectend"),
    "quality_top": ("t_quality_top", "Quality is the #1 factor",
                    "Another factor is #1", False, "price"),
}

TARGET_LABELS = {
    "brand_switch": "Brand switching",
    "snack_reaction": "Snack price reaction",
    "rec_follow": "Recommendation following",
    "quality_top": "Quality as top factor",
}

QUESTION_KEYS = {
    "age": "what is your age group",
    "freq": "how often do you make non-essential purchases",
    "price": "how important is price when making a purchase",
    "loyal": "how loyal are you to specific brands",
    "novel": "how likely are you to try a new or unfamiliar product",
    "cheaper": "how often do you choose cheaper alternatives",
    "rectend": "how likely are you to follow a product recommendation",
    "snack": "you usually buy a snack",
    "brand": "you usually buy brand a",
    "rec": "a system recommends a cheaper product",
    "top": "what is the most important factor",
}

BASELINE = "Baseline (class prior)"
SINGLE = "Single-question logistic"
LOGISTIC = "Logistic regression (all items)"
FOREST = "Random forest"


# ---------------------------------------------------------------------
# 1. LOAD
# ---------------------------------------------------------------------
def find_data_file() -> Path:
    candidates = sorted(
        p for p in HERE.glob("*.xlsx")
        if "responses" in p.name.lower() and not p.name.startswith("~$")
    )
    if not candidates:
        candidates = sorted(p for p in HERE.glob("*.xlsx") if not p.name.startswith("~$"))
    if not candidates:
        raise FileNotFoundError(
            f"No .xlsx survey export found in {HERE}.\n"
            "Put the Google Forms export beside this script, or run:\n"
            '    python Behavior_Economics_Updated.py --data "path/to/file.xlsx"'
        )
    if len(candidates) > 1:
        print(f"Several .xlsx files found; using {candidates[0].name}")
    return candidates[0]


def load_raw(path: Path) -> pd.DataFrame:
    path = Path(path).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Could not find the survey file:\n{path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Input must be an .xlsx, .xls, or .csv file.")

    df.columns = df.columns.astype(str).str.strip()
    rename = {}
    for short, fragment in QUESTION_KEYS.items():
        hits = [c for c in df.columns if fragment in c.lower()]
        if len(hits) != 1:
            raise ValueError(
                f"Expected one column matching '{fragment}' for '{short}', found {hits}"
            )
        rename[hits[0]] = short
    ts = [c for c in df.columns if "timestamp" in c.lower()]
    df = df.rename(columns=rename)
    keep = list(QUESTION_KEYS)
    if ts:
        df = df.rename(columns={ts[0]: "timestamp"})
        keep = ["timestamp"] + keep
    return df[keep].copy()


# ---------------------------------------------------------------------
# 2. CLEANING AND AUDIT TRAIL
# ---------------------------------------------------------------------
def clean(df: pd.DataFrame, sample: str):
    d = df.copy()
    for col in LIKERT:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    for col in ["age", "freq", "snack", "brand", "rec", "top"]:
        d[col] = d[col].fillna("").astype(str).str.strip()

    removal = {i: [] for i in d.index}
    review = {i: [] for i in d.index}

    def flag(mask, dest, reason):
        for i in d.index[mask.fillna(False).to_numpy()]:
            dest[i].append(reason)

    flag(d[LIKERT].isna().any(axis=1) | (d[["age", "freq"]] == "").any(axis=1),
         removal, "Missing core predictor")
    flag(((d[LIKERT] < 1) | (d[LIKERT] > 5)).any(axis=1),
         removal, "Likert response outside 1-5")
    flag(~d["age"].str.lower().isin(KNOWN_AGES), removal, "Unrecognised age category")
    flag(~d["freq"].str.lower().isin(FREQ_ORDER), removal,
         "Unrecognised purchase-frequency category")
    text = (d["snack"] + " | " + d["brand"] + " | " + d["rec"] + " | " + d["top"]).str.lower()
    flag(text.apply(lambda v: any(p in v for p in CLEAR_SPAM_PHRASES)),
         removal, "Clear promotional/spam response")
    flag(~d["age"].str.lower().isin(SAMPLES[sample]), removal,
         f"Outside '{sample}' sample (age group)")

    flag(d[LIKERT].nunique(axis=1) == 1, review,
         "Review: identical answer on all five Likert items")

    removed_ids = [i for i, r in removal.items() if r]
    kept = d.drop(index=removed_ids).reset_index(drop=True)
    removed = d.loc[removed_ids].copy()
    removed["removal_reason"] = ["; ".join(removal[i]) for i in removed_ids]
    review_ids = [i for i, r in review.items() if r and i not in removed_ids]
    flagged = d.loc[review_ids].copy()
    flagged["review_reason"] = ["; ".join(review[i]) for i in review_ids]
    return kept, removed.reset_index(drop=True), flagged.reset_index(drop=True)


# ---------------------------------------------------------------------
# 3. FEATURES AND OUTCOMES
# ---------------------------------------------------------------------
def norm(s: pd.Series) -> pd.Series:
    return s.str.lower().str.strip().str.rstrip(". ")


def add_features_and_targets(d: pd.DataFrame) -> pd.DataFrame:
    d = d.copy()
    d["age_18plus"] = (d["age"].str.lower() != "under 18").astype(int)
    d["freq_ord"] = d["freq"].str.lower().map(FREQ_ORDER)

    snack = norm(d["snack"])
    d["snack_coded"] = np.select(
        [snack == "still buy it", snack == "buy it less often",
         snack.str.startswith("switch"), snack == "stop buying it"],
        ["Still buy", "Buy less often", "Switch to cheaper", "Stop buying"],
        default="Conditional / write-in")
    d["t_snack_reaction"] = np.select(
        [d["snack_coded"] == "Still buy",
         d["snack_coded"].isin(["Buy less often", "Switch to cheaper", "Stop buying"])],
        [0, 1], default=np.nan)

    brand = norm(d["brand"])
    d["brand_coded"] = np.select(
        [brand == "switch to brand b", brand == "stay with brand a"],
        ["Switch to Brand B", "Stay with Brand A"], default="Conditional / write-in")
    d["t_brand_switch"] = np.select(
        [brand == "switch to brand b", brand == "stay with brand a"], [1, 0], default=np.nan)

    rec = norm(d["rec"])
    d["rec_coded"] = np.select(
        [rec.str.startswith("follow the recommendation"), rec == "ignore it"],
        ["Follow", "Ignore"], default="Conditional / write-in")
    d["t_rec_follow"] = np.select(
        [rec.str.startswith("follow the recommendation"), rec == "ignore it"],
        [1, 0], default=np.nan)

    d["t_quality_top"] = (norm(d["top"]) == "quality").astype(int)
    return d


def feature_list(sample: str) -> list[str]:
    # Age is constant in the student-only sample, so it is not a feature there.
    base = ["freq_ord"] + LIKERT
    return base if sample == "students" else ["age_18plus"] + base


# ---------------------------------------------------------------------
# 4. MODELS
# ---------------------------------------------------------------------
def tuned_logistic() -> Pipeline:
    # Fixed, pre-specified mild L2 penalty (sklearn default C=1 on standardised
    # inputs). With ~100 rows and 6 predictors, tuning C added run time and
    # instability (inner folds with a handful of minority cases) but no
    # measurable accuracy, so the simplest defensible setting is used.
    return Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced")),
    ])


def make_models(features: list[str], single_item: str) -> dict:
    single = Pipeline([
        ("pick", ColumnTransformer([("one", "passthrough", [single_item])])),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=5000, class_weight="balanced")),
    ])
    forest = RandomForestClassifier(
        n_estimators=500, max_depth=3, min_samples_leaf=5,
        class_weight="balanced", random_state=SEED)
    return {
        BASELINE: DummyClassifier(strategy="prior"),
        SINGLE: single,
        LOGISTIC: tuned_logistic(),
        FOREST: forest,
    }


SCORING = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "roc_auc": "roc_auc",
    "neg_brier": "neg_brier_score",
}


def corrected_ci(diff: np.ndarray, n: int):
    """Nadeau & Bengio (2003) corrected t-interval for repeated k-fold CV."""
    k = len(diff)
    n_test = n / N_SPLITS
    se = np.sqrt((1 / k + n_test / (n - n_test)) * diff.var(ddof=1))
    t = stats.t.ppf(0.975, k - 1)
    m = diff.mean()
    p = 2 * stats.t.sf(abs(m / se), k - 1) if se > 0 else np.nan
    return m, m - t * se, m + t * se, p


def evaluate_target(name, X, y, features, single_item):
    cv = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    scores = {
        m: cross_validate(est, X, y, cv=cv, scoring=SCORING, n_jobs=N_JOBS,
                          error_score="raise")
        for m, est in make_models(features, single_item).items()
    }
    rows = []
    for m, s in scores.items():
        row = {
            "target": name, "n": len(y), "positives": int(y.sum()), "model": m,
            "accuracy": s["test_accuracy"].mean(),
            "balanced_accuracy": s["test_balanced_accuracy"].mean(),
            "roc_auc": s["test_roc_auc"].mean(),
            "brier_score": -s["test_neg_brier"].mean(),
        }
        if m != BASELINE:
            mean, lo, hi, _ = corrected_ci(s["test_roc_auc"], len(y))
            row.update(auc_ci_low=lo, auc_ci_high=hi)
            mean, lo, hi, _ = corrected_ci(
                s["test_balanced_accuracy"] - 0.5, len(y))
            row.update(bal_acc_gain=mean, bal_acc_gain_ci_low=lo, bal_acc_gain_ci_high=hi)
        rows.append(row)

    comparisons = []
    for a, b in [(LOGISTIC, SINGLE), (FOREST, LOGISTIC)]:
        mean, lo, hi, p = corrected_ci(
            scores[a]["test_roc_auc"] - scores[b]["test_roc_auc"], len(y))
        comparisons.append({
            "target": name, "comparison": f"{a} minus {b}", "auc_difference": mean,
            "ci_low": lo, "ci_high": hi, "corrected_p": p,
        })
    return rows, comparisons


def holm_adjust(p_values: dict) -> dict:
    ordered = sorted(p_values.items(), key=lambda kv: kv[1])
    out, running = {}, 0.0
    for i, (k, p) in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - i) * p))
        out[k] = running
    return out


def permutation_p(X, y, n_perm):
    score, _, p = permutation_test_score(
        tuned_logistic(), X, y, scoring="roc_auc",
        cv=StratifiedKFold(N_SPLITS, shuffle=True, random_state=SEED),
        n_permutations=n_perm, random_state=SEED, n_jobs=N_JOBS)
    return score, p


# ---------------------------------------------------------------------
# 5. ODDS RATIOS (unpenalised logistic regression, per 1 scale point)
# ---------------------------------------------------------------------
def fit_unpenalised(X: np.ndarray, y: np.ndarray):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = LogisticRegression(penalty=None, max_iter=10000)
        m.fit(X, y)
    return m


def odds_ratios(X: pd.DataFrame, y: pd.Series, n_boot: int) -> pd.DataFrame:
    Xv, yv = X.to_numpy(float), y.to_numpy()
    model = fit_unpenalised(Xv, yv)
    beta = np.r_[model.intercept_, model.coef_[0]]

    # Wald standard errors from the observed information matrix.
    design = np.column_stack([np.ones(len(Xv)), Xv])
    p_hat = model.predict_proba(Xv)[:, 1]
    info = design.T @ (design * (p_hat * (1 - p_hat))[:, None])
    se = np.sqrt(np.diag(np.linalg.pinv(info)))[1:]
    z = model.coef_[0] / se
    wald_p = 2 * stats.norm.sf(np.abs(z))

    rng = np.random.default_rng(SEED)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(yv), len(yv))
        if len(np.unique(yv[idx])) < 2:
            continue
        boot.append(fit_unpenalised(Xv[idx], yv[idx]).coef_[0])
    boot = np.asarray(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5], axis=0)

    out = pd.DataFrame({
        "feature_code": X.columns,
        "feature": [FEATURE_LABELS[c] for c in X.columns],
        "odds_ratio": np.exp(model.coef_[0]),
        "wald_ci_low": np.exp(model.coef_[0] - 1.96 * se),
        "wald_ci_high": np.exp(model.coef_[0] + 1.96 * se),
        "wald_p": wald_p,
        "boot_ci_low": np.exp(lo),
        "boot_ci_high": np.exp(hi),
    })
    out["both_intervals_exclude_1"] = (
        ((out.wald_ci_low > 1) | (out.wald_ci_high < 1))
        & ((out.boot_ci_low > 1) | (out.boot_ci_high < 1))
    )
    out["events_per_predictor"] = min(y.sum(), len(y) - y.sum()) / X.shape[1]
    return out


# ---------------------------------------------------------------------
# 6. LEARNING CURVES
# ---------------------------------------------------------------------
def plot_learning_curves(samples: dict, path: Path, title_suffix: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping learning curves.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, (name, (X, y)) in zip(axes.ravel(), samples.items()):
        sizes, _, val = learning_curve(
            tuned_logistic(), X, y, scoring="roc_auc",
            train_sizes=np.linspace(0.3, 1.0, 6), shuffle=True, random_state=SEED,
            cv=RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=4, random_state=SEED),
            n_jobs=N_JOBS)
        m, s = val.mean(axis=1), val.std(axis=1)
        ax.plot(sizes, m, marker="o", color="#2a6f97", label="Cross-validated AUC")
        ax.fill_between(sizes, m - s, m + s, alpha=0.2, color="#2a6f97",
                        label="±1 SD across folds")
        ax.axhline(0.5, ls="--", color="#888", label="Chance (0.5)")
        ax.set_title(f"{TARGET_LABELS[name]} (n={len(y)})")
        ax.set_xlabel("Training rows")
        ax.set_ylabel("ROC-AUC")
        ax.set_ylim(0.2, 1.0)
        ax.legend(loc="lower right", fontsize=8)
    for ax in axes.ravel()[len(samples):]:
        ax.axis("off")
    fig.suptitle(f"Learning curves, logistic regression ({title_suffix})")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------
# 7. PREDICT A NEW PROFILE (used by app.py too)
# ---------------------------------------------------------------------
def profile_row(features, age_group, purchase_frequency, price, loyal, novel,
                cheaper, rectend) -> pd.DataFrame:
    age_key, freq_key = age_group.strip().lower(), purchase_frequency.strip().lower()
    if age_key not in KNOWN_AGES:
        raise ValueError(f"Age group must be one of {sorted(KNOWN_AGES)}")
    if freq_key not in FREQ_ORDER:
        raise ValueError(f"Purchase frequency must be one of {list(FREQ_ORDER)}")
    vals = [price, loyal, novel, cheaper, rectend]
    if any(not isinstance(v, (int, np.integer)) or not 1 <= v <= 5 for v in vals):
        raise ValueError("Likert answers must be integers from 1 to 5.")
    row = {"age_18plus": int(age_key != "under 18"), "freq_ord": FREQ_ORDER[freq_key],
           "price": price, "loyal": loyal, "novel": novel,
           "cheaper": cheaper, "rectend": rectend}
    return pd.DataFrame([row])[features]


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------
def main(data_path: Path, sample: str, quick: bool) -> None:
    n_perm = 199 if quick else N_PERMUTATIONS
    n_boot = 200 if quick else N_BOOT
    out = HERE / "outputs" / sample
    out.mkdir(parents=True, exist_ok=True)
    features = feature_list(sample)

    raw = load_raw(data_path)
    kept, removed, review = clean(raw, sample)
    data = add_features_and_targets(kept)
    removed.to_csv(out / "removed_rows.csv", index=False)
    review.to_csv(out / "rows_flagged_for_review.csv", index=False)
    data.to_csv(out / "cleaned_data.csv", index=False)

    print(f"\nSAMPLE: {sample}   (data file: {Path(data_path).name})")
    print("-" * 70)
    print(f"Original responses:      {len(raw)}")
    outside = removed.removal_reason.str.contains("Outside '", regex=False)
    invalid = removed.removal_reason.str.replace(r"Outside '[^']*' sample \(age group\)", "", regex=True).str.strip("; ") != ""
    print(f"Removed (spam/invalid):  {int(invalid.sum())}")
    print(f"Removed (outside sample, valid otherwise): {int((outside & ~invalid).sum())}")
    print(f"Retained:                {len(data)}")
    print(f"Flagged for review:      {len(review)}")

    coding = {
        col: data[col].value_counts().to_dict()
        for col in ["brand_coded", "snack_coded", "rec_coded", "top"]
    }
    for col, counts in coding.items():
        print(f"  {col}: {counts}")

    rows, comps, perm, samples_xy = [], [], {}, {}
    for name, (col, pos, neg, primary, single) in TARGETS.items():
        sub = data.dropna(subset=[col])
        X, y = sub[features], sub[col].astype(int)
        counts = y.value_counts()
        print(f"\n{name} [{'PRIMARY' if primary else 'EXPLORATORY'}]: n={len(y)}, "
              f"'{pos}'={counts.get(1, 0)}, '{neg}'={counts.get(0, 0)}")
        if len(counts) < 2 or counts.min() < MIN_CLASS_COUNT:
            print(f"  SKIPPED: smaller class < {MIN_CLASS_COUNT}")
            continue
        samples_xy[name] = (X, y)
        r, c = evaluate_target(name, X, y, features, single)
        rows += r
        comps += c
        _, perm[name] = permutation_p(X, y, n_perm)

    results = pd.DataFrame(rows)
    adj = holm_adjust({k: v for k, v in perm.items() if TARGETS[k][3]})
    is_lr = results.model == LOGISTIC
    results["permutation_p"] = np.where(is_lr, results.target.map(perm), np.nan)
    results["holm_p_primary"] = np.where(is_lr, results.target.map(adj), np.nan)
    results.to_csv(out / "cv_results.csv", index=False)
    comparisons = pd.DataFrame(comps)
    comparisons.to_csv(out / "model_comparisons.csv", index=False)

    pd.set_option("display.width", 200)
    print("\nCROSS-VALIDATED RESULTS (5-fold x 10 repeats)")
    print(results[["target", "n", "model", "accuracy", "balanced_accuracy",
                   "roc_auc", "auc_ci_low", "auc_ci_high", "brier_score",
                   "permutation_p", "holm_p_primary"]].round(3).to_string(index=False))
    print("\nPAIRED MODEL COMPARISONS (AUC difference, corrected CI)")
    print(comparisons.round(3).to_string(index=False))

    odds_tables, artifacts = [], {}
    for name, (X, y) in samples_xy.items():
        t = odds_ratios(X, y, n_boot)
        t.insert(0, "target", name)
        odds_tables.append(t)
        lr = results[(results.target == name) & is_lr].iloc[0]
        artifacts[name] = {
            "model": clone(tuned_logistic()).fit(X, y),
            "features": features,
            "label": TARGET_LABELS[name],
            "positive_label": TARGETS[name][1],
            "base_rate": float(y.mean()),
            "cv_auc": float(lr.roc_auc),
            "auc_ci_low": float(lr.auc_ci_low),
            "auc_ci_high": float(lr.auc_ci_high),
        }
    odds = pd.concat(odds_tables, ignore_index=True)
    odds.to_csv(out / "odds_ratios.csv", index=False)
    print("\nODDS RATIOS (unpenalised logistic, per +1 scale point)")
    print(odds.drop(columns=["feature_code"]).round(3).to_string(index=False))

    plot_learning_curves(samples_xy, out / "learning_curves.png",
                         {"students": "students under 18",
                          "student_age": "respondents under 26",
                          "all": "all respondents"}[sample])
    joblib.dump(artifacts, out / "final_models.joblib")

    summary = {
        "sample": sample, "features": features, "n_raw": len(raw),
        "n_retained": len(data), "n_flagged": len(review),
        "removed_reasons": removed.removal_reason.value_counts().to_dict(),
        "coding": coding, "permutation_p": perm, "holm_p_primary": adj,
        "n_permutations": n_perm, "n_bootstrap": n_boot,
        "single_items": {k: v[4] for k, v in TARGETS.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\nEXAMPLE PREDICTION (occasional buyer, price-focused, follows recs)")
    row = profile_row(features, "Under 18", "Occasionally", 5, 2, 3, 4, 4)
    for name, a in artifacts.items():
        print(f"  {a['label']:<28} P({a['positive_label']}) = "
              f"{a['model'].predict_proba(row)[0, 1]:.2f}  (CV AUC {a['cv_auc']:.2f})")
    print(f"\nFiles saved in: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--data", type=Path, default=None, help="Path to the survey .xlsx/.csv")
    parser.add_argument("--sample", choices=list(SAMPLES), default="students")
    parser.add_argument("--quick", action="store_true",
                        help="Fewer permutations/bootstraps for a fast test run")
    args = parser.parse_args()
    try:
        main(args.data or find_data_file(), args.sample, args.quick)
    except (FileNotFoundError, ValueError) as err:
        sys.exit(f"\nERROR: {err}")