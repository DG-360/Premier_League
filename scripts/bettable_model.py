#!/usr/bin/env python3
"""Premier League Betable model: train, validate, test, and publish current probabilities.

Design goals
------------
* Leakage-safe: every feature is built from matches available before kickoff.
* Minimal inputs: PL results only for M0-M2.
* Chronological evaluation:
    train      2017/18-2022/23
    validation 2023/24
    test 1     2024/25
    test 2     2025/26 (untouched second test)
* M0 = Elo only
* M1 = M0 + recent form + home/away form
* M2 = M1 + compact attack/defence form
* M3 = a bounded live context layer (injuries + manager/tactical/formation edge).
  Until dated historical context exists and is backtested, M3 is veto-only:
  it may reduce/remove a recommendation but may never create one.

Hyperparameters are tuned only inside the training seasons with rolling
chronological validation. 2023/24 selects M0/M1/M2. 2024/25 and 2025/26 are
report-only holdouts and cannot alter the selected production model.

The final selected base model is then refit on all completed historical seasons
for 2026/27 deployment and publishes current-matchweek P(Home/Draw/Away) to
Firebase. The website applies the conservative veto-only M3 layer immediately
from Firebase modelContext.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import sys
import urllib.request
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = "pl2627"
SOURCE = (
    "https://raw.githubusercontent.com/datasets/football-datasets/"
    "main/datasets/premier-league/season-{code}.csv"
)

TRAIN_CODES = ["1718", "1819", "1920", "2021", "2122", "2223"]
VAL_CODE = "2324"
TEST1_CODE = "2425"
TEST2_CODE = "2526"
ALL_CODES = TRAIN_CODES + [VAL_CODE, TEST1_CODE, TEST2_CODE]

SEASON_LABEL = {
    "1718": "2017-18", "1819": "2018-19", "1920": "2019-20",
    "2021": "2020-21", "2122": "2021-22", "2223": "2022-23",
    "2324": "2023-24", "2425": "2024-25", "2526": "2025-26",
}

# Current site codes -> football-data naming convention/canonical team name.
SITE_TEAM = {
    "ARS": "Arsenal", "AVL": "Aston Villa", "BOU": "Bournemouth",
    "BRE": "Brentford", "BHA": "Brighton", "CHE": "Chelsea",
    "COV": "Coventry", "CRY": "Crystal Palace", "EVE": "Everton",
    "FUL": "Fulham", "HUL": "Hull", "IPS": "Ipswich", "LEE": "Leeds",
    "LIV": "Liverpool", "MCI": "Man City", "MUN": "Man United",
    "NEW": "Newcastle", "NFO": "Nott'm Forest", "SUN": "Sunderland",
    "TOT": "Tottenham",
}

NAME_ALIASES = {
    "Manchester City": "Man City", "Manchester United": "Man United",
    "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Tottenham",
    "Brighton & Hove Albion": "Brighton", "Newcastle United": "Newcastle",
    "Leeds United": "Leeds", "Sunderland AFC": "Sunderland",
    "Hull City": "Hull", "Coventry City": "Coventry",
    "Ipswich Town": "Ipswich", "AFC Bournemouth": "Bournemouth",
}

FEATURES = {
    "M0": ["elo_diff"],
    "M1": ["elo_diff", "form5_diff", "venue5_diff"],
    "M2": ["elo_diff", "form5_diff", "venue5_diff", "gd5_diff", "attack_edge"],
}

C_GRID = [0.10, 0.30, 0.70, 1.00, 2.00]
MIN_SEASON_MATCHES = 350
MAX_SEASON_MATCHES = 390

# Conservative live M3 coefficients. These are not presented as historically
# learned until a dated context dataset is available.
LIVE_CONTEXT_DEFAULTS = {
    # away_absence - home_absence; each level is 0:none, 1:important, 2:major
    "injury_logit_per_level": 0.075,
    # tacticalEdge is -2..+2, positive means home tactical/formation edge
    "tactical_logit_per_level": 0.055,
    # uncertainty 0/1 flattens probabilities by increasing temperature
    "uncertainty_temperature": 0.10,
    "max_abs_logit_shift": 0.22,
}


def canon(name: str) -> str:
    x = str(name or "").strip()
    return NAME_ALIASES.get(x, x)


def fetch_text(url: str, timeout: int = 30) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "betable-model/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8-sig")


def load_seasons(cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    parts = []
    needed = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"]
    for code in ALL_CODES:
        fp = cache_dir / f"season-{code}.csv"
        if not fp.exists():
            fp.write_text(fetch_text(SOURCE.format(code=code)), encoding="utf-8")
        df = pd.read_csv(fp)
        missing = [c for c in needed if c not in df.columns]
        if missing:
            raise RuntimeError(f"{fp.name} missing columns: {missing}")
        df = df[needed].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.dropna(subset=["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG", "FTR"])
        df["HomeTeam"] = df["HomeTeam"].map(canon)
        df["AwayTeam"] = df["AwayTeam"].map(canon)
        df["FTHG"] = df["FTHG"].astype(int)
        df["FTAG"] = df["FTAG"].astype(int)

        # A normal Premier League season has 380 matches. Refuse to train on
        # truncated downloads/caches; attractive metrics from partial seasons
        # are worse than a hard failure.
        n_matches = int(len(df))
        if not (MIN_SEASON_MATCHES <= n_matches <= MAX_SEASON_MATCHES):
            raise RuntimeError(
                f"{fp.name} contains {n_matches} usable matches; expected roughly "
                f"a full Premier League season ({MIN_SEASON_MATCHES}-{MAX_SEASON_MATCHES}). "
                "Delete the cached file and rerun."
            )

        df["SeasonCode"] = code
        parts.append(df)
    return pd.concat(parts, ignore_index=True).sort_values(["Date", "SeasonCode"]).reset_index(drop=True)


def avg(q: Sequence[float], default: float) -> float:
    return float(np.mean(q)) if len(q) else float(default)


@dataclass
class TeamState:
    elo: Dict[str, float] = field(default_factory=dict)
    last_seen_season: Dict[str, int] = field(default_factory=dict)
    form: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=5)))
    gd: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=5)))
    gf: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=5)))
    ga: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=5)))
    home_form: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=5)))
    away_form: Dict[str, deque] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=5)))
    season_index: int = -1

    # Fixed, intentionally modest parameters. The logistic model learns how much
    # the resulting Elo difference matters.
    k: float = 24.0
    home_elo: float = 55.0
    season_carry: float = 0.78
    new_team_elo: float = 1440.0

    def _elo(self, team: str) -> float:
        return self.elo.get(team, self.new_team_elo)

    def start_season(self, season_index: int, teams: Iterable[str]) -> None:
        if self.season_index == season_index:
            return
        teams = set(map(canon, teams))
        for t in teams:
            if t in self.elo:
                gap = max(1, season_index - self.last_seen_season.get(t, season_index - 1))
                carry = self.season_carry ** gap
                self.elo[t] = 1500.0 + carry * (self.elo[t] - 1500.0)
            else:
                self.elo[t] = self.new_team_elo
            self.last_seen_season[t] = season_index
        # Form is intentionally season-local. Elo carries long-run strength.
        self.form = defaultdict(lambda: deque(maxlen=5))
        self.gd = defaultdict(lambda: deque(maxlen=5))
        self.gf = defaultdict(lambda: deque(maxlen=5))
        self.ga = defaultdict(lambda: deque(maxlen=5))
        self.home_form = defaultdict(lambda: deque(maxlen=5))
        self.away_form = defaultdict(lambda: deque(maxlen=5))
        self.season_index = season_index

    def features(self, home: str, away: str) -> Dict[str, float]:
        h, a = canon(home), canon(away)
        h_form = avg(self.form[h], 0.50)
        a_form = avg(self.form[a], 0.50)
        h_venue = avg(self.home_form[h], 0.50)
        a_venue = avg(self.away_form[a], 0.50)
        h_gd = avg(self.gd[h], 0.0)
        a_gd = avg(self.gd[a], 0.0)
        h_gf, a_gf = avg(self.gf[h], 1.35), avg(self.gf[a], 1.35)
        h_ga, a_ga = avg(self.ga[h], 1.35), avg(self.ga[a], 1.35)
        # Positive attack_edge means the home scoring-vs-conceding matchup is
        # stronger than the away scoring-vs-conceding matchup.
        attack_edge = (h_gf + a_ga) - (a_gf + h_ga)
        return {
            "elo_diff": self._elo(h) + self.home_elo - self._elo(a),
            "form5_diff": h_form - a_form,
            "venue5_diff": h_venue - a_venue,
            "gd5_diff": h_gd - a_gd,
            "attack_edge": attack_edge,
        }

    def update(self, home: str, away: str, hg: int, ag: int) -> None:
        h, a = canon(home), canon(away)
        hp = 3 if hg > ag else 1 if hg == ag else 0
        ap = 3 if ag > hg else 1 if hg == ag else 0
        self.form[h].append(hp / 3.0)
        self.form[a].append(ap / 3.0)
        self.gd[h].append(hg - ag)
        self.gd[a].append(ag - hg)
        self.gf[h].append(hg); self.ga[h].append(ag)
        self.gf[a].append(ag); self.ga[a].append(hg)
        self.home_form[h].append(hp / 3.0)
        self.away_form[a].append(ap / 3.0)

        eh = 1.0 / (1.0 + 10.0 ** ((self._elo(a) - (self._elo(h) + self.home_elo)) / 400.0))
        sh = 1.0 if hg > ag else 0.5 if hg == ag else 0.0
        gd = abs(hg - ag)
        margin_mult = 1.0 if gd <= 1 else min(1.45, 1.0 + 0.12 * (gd - 1))
        delta = self.k * margin_mult * (sh - eh)
        self.elo[h] = self._elo(h) + delta
        self.elo[a] = self._elo(a) - delta
        self.last_seen_season[h] = self.season_index
        self.last_seen_season[a] = self.season_index


def season_order(code: str) -> int:
    return ALL_CODES.index(code)


def build_feature_table(matches: pd.DataFrame) -> Tuple[pd.DataFrame, TeamState]:
    state = TeamState()
    rows: List[dict] = []
    current_code = None
    for code in ALL_CODES:
        s = matches[matches.SeasonCode == code].sort_values("Date")
        teams = set(s.HomeTeam).union(set(s.AwayTeam))
        state.start_season(season_order(code), teams)
        for r in s.itertuples(index=False):
            f = state.features(r.HomeTeam, r.AwayTeam)
            rows.append({
                "Date": r.Date, "SeasonCode": code,
                "HomeTeam": r.HomeTeam, "AwayTeam": r.AwayTeam,
                "Result": str(r.FTR), **f,
            })
            state.update(r.HomeTeam, r.AwayTeam, int(r.FTHG), int(r.FTAG))
    return pd.DataFrame(rows), state


def make_model(features: List[str], C: float) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("model", LogisticRegression(max_iter=4000, C=C, solver="lbfgs")),
    ])


def apply_temperature(prob: np.ndarray, T: float) -> np.ndarray:
    p = np.clip(np.asarray(prob, float), 1e-9, 1.0)
    z = np.log(p) / max(float(T), 1e-6)
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z)
    return q / q.sum(axis=1, keepdims=True)




def multiclass_log_loss(y: Sequence[str], prob: np.ndarray, classes: Sequence[str]) -> float:
    """Cross-entropy with an explicit probability-column order.

    sklearn sorts class labels internally, which is easy to misuse when we keep
    the human-readable H/D/A column order. This implementation makes the
    mapping explicit and avoids that ambiguity.
    """
    cls = list(classes)
    idx = {c: i for i, c in enumerate(cls)}
    p = np.clip(np.asarray(prob, float), 1e-15, 1.0)
    rows = np.arange(len(y))
    cols = np.asarray([idx[str(v)] for v in y], dtype=int)
    return float(-np.mean(np.log(p[rows, cols])))

def choose_temperature(y: Sequence[str], prob: np.ndarray, classes: Sequence[str]) -> float:
    grid = np.linspace(0.65, 1.65, 101)
    best_t, best_loss = 1.0, float("inf")
    for t in grid:
        q = apply_temperature(prob, float(t))
        loss = multiclass_log_loss(y, q, classes)
        if loss < best_loss:
            best_t, best_loss = float(t), float(loss)
    return best_t


def multiclass_brier(y: Sequence[str], prob: np.ndarray, classes: Sequence[str]) -> float:
    cls = list(classes)
    idx = {c: i for i, c in enumerate(cls)}
    one = np.zeros_like(prob, dtype=float)
    for j, label in enumerate(y):
        one[j, idx[str(label)]] = 1.0
    # Common multiclass Brier: mean summed squared probability error.
    return float(np.mean(np.sum((prob - one) ** 2, axis=1)))


def top_ece(y: Sequence[str], prob: np.ndarray, classes: Sequence[str], bins: int = 10) -> float:
    cls = np.asarray(list(classes))
    pred_i = np.argmax(prob, axis=1)
    pred = cls[pred_i]
    conf = np.max(prob, axis=1)
    correct = (pred == np.asarray(y)).astype(float)
    out = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (conf >= lo) & ((conf < hi) if i < bins - 1 else (conf <= hi))
        n = int(mask.sum())
        if not n:
            continue
        out += (n / len(y)) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(out)


def metrics(y: Sequence[str], prob: np.ndarray, classes: Sequence[str]) -> dict:
    cls = np.asarray(list(classes))
    pred = cls[np.argmax(prob, axis=1)]
    return {
        "n": int(len(y)),
        "log_loss": multiclass_log_loss(y, prob, classes),
        "brier": multiclass_brier(y, prob, classes),
        "accuracy": float(accuracy_score(y, pred)),
        "top_ece": top_ece(y, prob, classes),
        "mean_confidence": float(np.max(prob, axis=1).mean()),
    }


def predict_in_class_order(model: Pipeline, X: pd.DataFrame, wanted=("H", "D", "A")) -> np.ndarray:
    raw = model.predict_proba(X)
    got = list(model.classes_)
    out = np.zeros((len(X), len(wanted)), float)
    for j, c in enumerate(wanted):
        if c in got:
            out[:, j] = raw[:, got.index(c)]
    out /= np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)
    return out


def threshold_from_validation(y: Sequence[str], prob: np.ndarray, classes=("H", "D", "A")) -> dict:
    cls = np.asarray(classes)
    order = np.argsort(prob, axis=1)
    best_i = order[:, -1]
    second_i = order[:, -2]
    pred = cls[best_i]
    conf = prob[np.arange(len(prob)), best_i]
    margin = conf - prob[np.arange(len(prob)), second_i]
    correct = pred == np.asarray(y)

    chosen = None
    for t in np.arange(0.50, 0.76, 0.01):
        mask = (conf >= t) & (margin >= 0.10)
        n = int(mask.sum())
        if n >= 25 and float(correct[mask].mean()) >= 0.65:
            chosen = float(round(t, 2)); break
    if chosen is None:
        # Do not claim historical 65% reliability if the validation season did
        # not support it. Use a conservative fallback and expose the warning.
        chosen = 0.62
        mask = (conf >= chosen) & (margin >= 0.10)
        observed = float(correct[mask].mean()) if mask.any() else None
        n = int(mask.sum())
        supported = False
    else:
        mask = (conf >= chosen) & (margin >= 0.10)
        observed = float(correct[mask].mean())
        n = int(mask.sum())
        supported = True
    return {
        "min_prob": chosen,
        "min_margin": 0.10,
        "validation_n": n,
        "validation_accuracy_at_gate": observed,
        "gate_supported_65pct": supported,
    }



def tune_inside_training(
    feat: pd.DataFrame,
    model_name: str,
    cols: List[str],
    class_order: Sequence[str] = ("H", "D", "A"),
) -> dict:
    """Tune C and calibration temperature without touching 2023/24 or tests."""
    candidates = []
    for C in C_GRID:
        y_parts = []
        p_parts = []
        for i in range(1, len(TRAIN_CODES)):
            fit_codes = TRAIN_CODES[:i]
            fold_code = TRAIN_CODES[i]
            tr = feat[feat.SeasonCode.isin(fit_codes)]
            va = feat[feat.SeasonCode == fold_code]
            if tr.empty or va.empty:
                continue
            mdl = make_model(cols, C)
            mdl.fit(tr[cols], tr.Result)
            p0 = predict_in_class_order(mdl, va[cols], class_order)
            y_parts.extend(va.Result.astype(str).tolist())
            p_parts.append(p0)
        if not p_parts:
            raise RuntimeError(f"No rolling training folds available for {model_name}")
        oof0 = np.vstack(p_parts)
        T = choose_temperature(y_parts, oof0, class_order)
        oof = apply_temperature(oof0, T)
        met = metrics(y_parts, oof, class_order)
        candidates.append((met["log_loss"], C, T, met, y_parts, oof))
    candidates.sort(key=lambda x: (x[0], x[1]))
    _, C, T, met, y_oof, p_oof = candidates[0]
    return {
        "C": float(C), "T": float(T), "metrics": met,
        "y_oof": y_oof, "p_oof": p_oof,
    }


def evaluate_models(feat: pd.DataFrame) -> Tuple[dict, str, dict]:
    train = feat[feat.SeasonCode.isin(TRAIN_CODES)].copy()
    val = feat[feat.SeasonCode == VAL_CODE].copy()
    test1 = feat[feat.SeasonCode == TEST1_CODE].copy()
    test2 = feat[feat.SeasonCode == TEST2_CODE].copy()

    report = {"models": {}, "baselines": {}}
    class_order = ("H", "D", "A")

    prior = train.Result.value_counts(normalize=True).reindex(class_order, fill_value=0).values
    for name, split in [("validation", val), ("test_2425", test1), ("test_2526", test2)]:
        p = np.tile(prior, (len(split), 1))
        report["baselines"][name] = {
            "empirical_prior_log_loss": multiclass_log_loss(split.Result, p, class_order),
            "always_home_accuracy": float((split.Result == "H").mean()),
        }

    val_candidates = []
    fitted_info = {}
    for model_name, cols in FEATURES.items():
        tuned = tune_inside_training(feat, model_name, cols, class_order)
        C, T = tuned["C"], tuned["T"]

        mdl_train = make_model(cols, C)
        mdl_train.fit(train[cols], train.Result)
        pv = apply_temperature(predict_in_class_order(mdl_train, val[cols], class_order), T)
        val_met = metrics(val.Result, pv, class_order)

        trv = pd.concat([train, val], ignore_index=True)
        frozen = make_model(cols, C)
        frozen.fit(trv[cols], trv.Result)
        p1 = apply_temperature(predict_in_class_order(frozen, test1[cols], class_order), T)
        p2 = apply_temperature(predict_in_class_order(frozen, test2[cols], class_order), T)

        report["models"][model_name] = {
            "features": cols,
            "C": C,
            "temperature": T,
            "internal_rolling_cv": tuned["metrics"],
            "validation": val_met,
            "test_2425": metrics(test1.Result, p1, class_order),
            "test_2526": metrics(test2.Result, p2, class_order),
        }
        fitted_info[model_name] = {
            "C": C, "T": T, "cols": cols,
            "internal_y": tuned["y_oof"], "internal_prob": tuned["p_oof"],
        }
        val_candidates.append((val_met["log_loss"], len(cols), model_name))

    val_candidates.sort()
    best_loss = val_candidates[0][0]
    near = [x for x in val_candidates if x[0] <= best_loss + 0.005]
    selected_by_val = sorted(near, key=lambda x: (x[1], x[0]))[0][2]
    production = selected_by_val

    threshold = threshold_from_validation(
        fitted_info[production]["internal_y"],
        fitted_info[production]["internal_prob"],
        class_order,
    )
    threshold["calibration_source"] = "rolling_training_cv_2017_18_to_2022_23"

    report["selection"] = {
        "selected_by_validation": selected_by_val,
        "production_model": production,
        "reason": (
            "C and temperature tuned only by rolling CV within 2017/18-2022/23; "
            "M0/M1/M2 chosen only by 2023/24 log loss with a simplicity tie-break. "
            "2024/25 and 2025/26 are report-only holdouts and cannot change selection."
        ),
    }
    report["bettable_gate"] = threshold
    return report, production, fitted_info[production]

def parse_site_fixtures(index_path: Path) -> List[dict]:
    text = index_path.read_text(encoding="utf-8")
    m = re.search(r"const\s+FIXTURES\s*=\s*(\[[\s\S]*?\])\s*;", text)
    if not m:
        raise RuntimeError("Could not find const FIXTURES=[...] in index.html")
    weeks = json.loads(m.group(1))
    out = []
    for wi, week in enumerate(weeks, 1):
        for h, a, k in week:
            out.append({"id": f"{wi}_{h}_{a}", "mw": wi, "h": h, "a": a, "k": int(k)})
    return out


def http_json(url: str, method="GET", payload=None, timeout=30):
    data = None
    headers = {"User-Agent": "betable-model/1.0"}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = r.read().decode("utf-8")
        return json.loads(body) if body else None


def current_state_from_history(hist_state: TeamState, fixtures: List[dict], results: dict) -> TeamState:
    state = hist_state
    teams = [SITE_TEAM[f["h"]] for f in fixtures] + [SITE_TEAM[f["a"]] for f in fixtures]
    state.start_season(len(ALL_CODES), teams)
    for f in sorted(fixtures, key=lambda x: x["k"]):
        r = (results or {}).get(f["id"])
        if not isinstance(r, dict) or "h" not in r or "a" not in r:
            continue
        state.update(SITE_TEAM[f["h"]], SITE_TEAM[f["a"]], int(r["h"]), int(r["a"]))
    return state


def current_matchweek(fixtures: List[dict], results: dict) -> int:
    max_mw = max(f["mw"] for f in fixtures)
    for mw in range(1, max_mw + 1):
        week = [f for f in fixtures if f["mw"] == mw]
        if any(f["id"] not in (results or {}) for f in week):
            return mw
    return max_mw


def fit_final_model(feat: pd.DataFrame, model_name: str, info: dict) -> Pipeline:
    cols = FEATURES[model_name]
    mdl = make_model(cols, float(info["C"]))
    mdl.fit(feat[cols], feat.Result)
    return mdl


def current_predictions(
    final_model: Pipeline,
    model_name: str,
    temperature: float,
    state: TeamState,
    fixtures: List[dict],
    results: dict,
) -> Tuple[int, dict]:
    week = current_matchweek(fixtures, results)
    current = [f for f in fixtures if f["mw"] == week and f["id"] not in (results or {})]
    if not current:
        return week, {}
    rows = []
    for f in current:
        feats = state.features(SITE_TEAM[f["h"]], SITE_TEAM[f["a"]])
        rows.append({**f, **feats})
    frame = pd.DataFrame(rows)
    cols = FEATURES[model_name]
    prob = predict_in_class_order(final_model, frame[cols], ("H", "D", "A"))
    prob = apply_temperature(prob, temperature)
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    out = {}
    for i, r in frame.iterrows():
        pH, pD, pA = map(float, prob[i])
        vals = {"H": pH, "D": pD, "A": pA}
        ranked = sorted(vals, key=vals.get, reverse=True)
        out[r["id"]] = {
            "pH": round(pH, 6), "pD": round(pD, 6), "pA": round(pA, 6),
            "pick": ranked[0],
            "confidence": round(vals[ranked[0]], 6),
            "margin": round(vals[ranked[0]] - vals[ranked[1]], 6),
            "model": model_name,
            "mw": int(r["mw"]),
            "generatedAt": now_ms,
        }
    return week, out


def maybe_load_context_csv(path: Optional[str]) -> dict:
    """Validate optional historical context file.

    We intentionally do NOT fabricate M3 backtest data. A future context CSV
    can contain dated pre-match manager/injury/formation signals and this
    function records that the file is present. A learned M3 can then be added
    without changing the base model contract.
    """
    if not path:
        return {"available": False, "status": "veto_only_live_adjustment"}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(p)
    required = {
        "Date", "HomeTeam", "AwayTeam", "homeAbsence", "awayAbsence",
        "tacticalEdge", "uncertainty",
    }
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Historical context CSV missing columns: {missing}")
    return {
        "available": True,
        "status": "historical_context_present_but_not_promoted_automatically",
        "rows": int(len(df)),
        "note": "Context is stored for a future learned M3; until validated, live M3 is veto-only and cannot create a Betable qualification.",
    }


def write_outputs(out_dir: Path, report: dict, meta: dict, predictions: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out_dir / "model_predictions.json").write_text(json.dumps(predictions, indent=2), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--cache-dir", default=".cache/betable")
    ap.add_argument("--out-dir", default="model_output")
    ap.add_argument("--firebase-url", default=os.environ.get("FIREBASE_DB_URL", ""))
    ap.add_argument("--publish", action="store_true")
    ap.add_argument("--context-csv", default=None)
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    cache = (repo / args.cache_dir).resolve()
    out_dir = (repo / args.out_dir).resolve()

    matches = load_seasons(cache)
    feat, hist_state = build_feature_table(matches)
    report, production_model, info = evaluate_models(feat)
    m3_status = maybe_load_context_csv(args.context_csv)

    fixtures = parse_site_fixtures(repo / "index.html")
    db_url = (args.firebase_url or "").rstrip("/")
    results = {}
    if db_url:
        try:
            results = http_json(f"{db_url}/{ROOT}/results.json") or {}
        except Exception as e:
            print(f"[betable-model] Firebase read warning: {e}", file=sys.stderr)

    state = current_state_from_history(hist_state, fixtures, results)
    final_model = fit_final_model(feat, production_model, info)
    week, predictions = current_predictions(
        final_model, production_model, float(info["T"]), state, fixtures, results
    )

    generated = datetime.now(timezone.utc).isoformat()
    meta = {
        "generatedAt": generated,
        "season": "2026-27",
        "matchweek": week,
        "selectedModel": production_model,
        "features": FEATURES[production_model],
        "temperature": float(info["T"]),
        "bettable": report["bettable_gate"],
        "evaluation": {
            "validation": report["models"][production_model]["validation"],
            "test_2425": report["models"][production_model]["test_2425"],
            "test_2526": report["models"][production_model]["test_2526"],
        },
        "m3": {
            **m3_status,
            "liveAdjustment": LIVE_CONTEXT_DEFAULTS,
            "principle": (
                "M3 is a conservative veto-only layer for important absences and "
                "manager/tactical/formation changes. Until a dated leakage-safe backtest "
                "exists, M3 may reduce/remove a recommendation but cannot create one."
            ),
        },
        "data": {
            "train": [SEASON_LABEL[x] for x in TRAIN_CODES],
            "validation": SEASON_LABEL[VAL_CODE],
            "test1": SEASON_LABEL[TEST1_CODE],
            "test2": SEASON_LABEL[TEST2_CODE],
        },
    }
    report["generatedAt"] = generated
    report["m3"] = meta["m3"]
    write_outputs(out_dir, report, meta, predictions)

    print(json.dumps({
        "selection": report["selection"],
        "bettable_gate": report["bettable_gate"],
        "production_metrics": meta["evaluation"],
        "m3": m3_status,
        "current_matchweek": week,
        "current_predictions": len(predictions),
    }, indent=2))

    if args.publish:
        if not db_url:
            raise SystemExit("--publish requires FIREBASE_DB_URL or --firebase-url")
        patch = {
            "modelMeta": meta,
            "modelPredictions": predictions,
        }
        http_json(f"{db_url}/{ROOT}.json", method="PATCH", payload=patch)
        print(f"[betable-model] published {len(predictions)} base prediction(s) for MW{week}")


if __name__ == "__main__":
    main()
