from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

OUTCOME_VAR = "py_thoughts_wanting_to_die"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reproduce Table 1 descriptive statistics and P values."
    )
    parser.add_argument("--input", type=Path, default=Path("primary_cc.csv"))
    parser.add_argument("--output", type=Path, default=Path("table1_statistics.csv"))
    parser.add_argument("--expected-n", type=int, default=6605)
    parser.add_argument("--expected-events", type=int, default=1850)
    return parser.parse_args()


def read_numeric_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if value in ("", None):
                    parsed[key] = None
                else:
                    try:
                        number = float(value)
                        parsed[key] = int(number) if number.is_integer() else number
                    except ValueError:
                        parsed[key] = value
            rows.append(parsed)
    return rows


def format_p(p: float) -> str:
    if p < 0.001:
        return "<.001"
    return f"{p:.3f}".replace("0.", ".")


def main() -> None:
    args = parse_args()
    data = read_numeric_csv(args.input)
    negative = [r for r in data if r[OUTCOME_VAR] == 0]
    positive = [r for r in data if r[OUTCOME_VAR] == 1]

    if len(data) != args.expected_n:
        raise ValueError(f"Expected N={args.expected_n}; found {len(data)}")
    if len(positive) != args.expected_events:
        raise ValueError(f"Expected events={args.expected_events}; found {len(positive)}")
    if len(negative) + len(positive) != len(data):
        raise ValueError("Outcome is not binary 0/1 for every row")

    def mean_sd(rows: list[dict[str, Any]], variable: str) -> str:
        values = np.array([float(r[variable]) for r in rows], dtype=float)
        return f"{values.mean():.2f} ({values.std(ddof=1):.2f})"

    def n_pct(rows: list[dict[str, Any]], variable: str, value: Any) -> str:
        count = sum(r[variable] == value for r in rows)
        return f"{count:,} ({count / len(rows) * 100:.1f})"

    def welch_p(variable: str) -> float:
        x0 = np.array([float(r[variable]) for r in negative], dtype=float)
        x1 = np.array([float(r[variable]) for r in positive], dtype=float)
        return float(stats.ttest_ind(x0, x1, equal_var=False).pvalue)

    def chi_square_p(variable: str, categories: list[Any]) -> float:
        table = np.array(
            [
                [sum(r[variable] == category for r in negative) for category in categories],
                [sum(r[variable] == category for r in positive) for category in categories],
            ],
            dtype=int,
        )
        return float(stats.chi2_contingency(table, correction=False).pvalue)

    rows: list[list[str]] = []

    def add_continuous(label: str, variable: str) -> None:
        rows.append([
            label,
            mean_sd(data, variable),
            mean_sd(negative, variable),
            mean_sd(positive, variable),
            format_p(welch_p(variable)),
        ])

    def add_categorical(group_label: str, variable: str, categories: list[tuple[Any, str]]) -> None:
        p_value = format_p(chi_square_p(variable, [code for code, _ in categories]))
        rows.append([group_label, "", "", "", p_value])
        for code, label in categories:
            rows.append([
                f"    {label}",
                n_pct(data, variable, code),
                n_pct(negative, variable, code),
                n_pct(positive, variable, code),
                "",
            ])

    add_continuous("Age, mean (SD), y", "age_years")
    add_categorical("Sex, No. (%)", "sex_female", [(0, "Male"), (1, "Female")])
    add_categorical("Marital status, No. (%)", "married_current", [(0, "Not currently married"), (1, "Currently married")])
    add_categorical("Employment status, No. (%)", "employed_corrected", [(0, "Not currently working"), (1, "Currently working")])
    add_categorical("Living arrangement, No. (%)", "living_alone", [(0, "Living with others"), (1, "Living alone")])
    add_categorical("Subjective physical health, No. (%)", "poor_subjective_physical_health", [(1, "Very good"), (2, "Good"), (3, "Moderate"), (4, "Poor")])
    add_categorical("Subjective mental health, No. (%)", "poor_subjective_mental_health", [(1, "Very good"), (2, "Good"), (3, "Moderate"), (4, "Poor")])
    add_continuous("GAD-7 score, mean (SD)", "gad7_total")
    add_continuous("PSS-14 score, mean (SD)", "pss14_total")
    add_continuous("Low self-esteem score, mean (SD)", "low_self_esteem_score")
    add_categorical("Sense of belonging, No. (%)", "low_sense_of_belonging", [(1, "Strongly connected"), (2, "Somewhat connected"), (3, "Somewhat disconnected"), (4, "Strongly disconnected")])
    add_categorical("Perceived social equality, No. (%)", "low_perceived_social_equality", [(1, "Very high equality"), (2, "High equality"), (3, "Moderate equality"), (4, "Low equality")])
    add_categorical("Social trust, No. (%)", "low_social_trust", [(1, "Very high trust"), (2, "High trust"), (3, "Moderate trust"), (4, "Low trust")])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "Characteristic",
            f"Overall (N = {len(data):,})",
            f"No Thoughts of Wanting to Die (n = {len(negative):,})",
            f"Thoughts of Wanting to Die (n = {len(positive):,})",
            "P Value",
        ])
        writer.writerows(rows)

    print(f"Created: {args.output}")
    print(f"N={len(data):,}; no outcome={len(negative):,}; outcome={len(positive):,}")


if __name__ == "__main__":
    main()
