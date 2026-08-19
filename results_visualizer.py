#!/usr/bin/env python3
"""
USAGE
    python3 results_visualizer.py eval_results.csv
    python3 results_visualizer.py eval_results.csv --policy vision
    python3 results_visualizer.py eval_results.csv --csv-out summary.csv
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


COL_POLICY = "policy"
COL_RESULT = "result"
COL_STEPS = "steps"
COL_POSITION = "position_code"
COL_NOTES = "notes"
COL_GRASP_RESULT = "grasp_result"

SUCCESS_VALUE = "success"
FAIL_VALUE = "fail"

FAILURE_CATEGORIES: list[tuple[str, list[str]]] = [
    ("out of bounds", ["out of bounds"]),
    ("max steps", ["max steps"]),
    ("bad grasp", ["bad grasp"]),
    ("missed drop", ["missed drop"]),
]
OTHER_CATEGORY = "other"


def categorize_failure(note) -> str:
    """Map a free-text failure note to one of FAILURE_CATEGORIES, or OTHER_CATEGORY."""
    if note is None or (isinstance(note, float) and pd.isna(note)) or str(note).strip() == "":
        return OTHER_CATEGORY
    text = str(note).strip().lower()
    for category, patterns in FAILURE_CATEGORIES:
        for pattern in patterns:
            if pattern in text:
                return category
    return OTHER_CATEGORY


def load_data(csv_path: str | Path) -> pd.DataFrame:
    """Load and lightly validate the eval-results CSV."""
    df = pd.read_csv(csv_path)

    required = [COL_POLICY, COL_RESULT, COL_STEPS, COL_POSITION]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV is missing expected column(s): {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df[COL_RESULT] = df[COL_RESULT].astype(str).str.strip().str.lower()
    df[COL_GRASP_RESULT] = df[COL_GRASP_RESULT].astype(str).str.strip().str.lower()
    if COL_NOTES not in df.columns:
        df[COL_NOTES] = ""

    # Only failures should carry a failure category; successes get "n/a".
    df["failure_category"] = "n/a"
    fail_mask = df[COL_RESULT] == FAIL_VALUE
    df.loc[fail_mask, "failure_category"] = df.loc[fail_mask, COL_NOTES].apply(categorize_failure)

    df["is_success"] = df[COL_RESULT] == SUCCESS_VALUE
    return df


class EvalAnalyzer:
    """Convenience wrapper around an eval-results dataframe for repeated analysis."""

    def __init__(self, csv_path: str | Path):
        self.csv_path = Path(csv_path)
        self.df = load_data(csv_path)

    # -- basic slicing -------------------------------------------------
    def for_policy(self, policy: str) -> pd.DataFrame:
        return self.df[self.df[COL_POLICY] == policy]

    # -- accuracy --------------------------------------------------------
    def accuracy(self, by="policy", df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Success rate (accuracy), grouped by `by` (a column name or list of
        column names). Defaults to grouping by policy alone.
        """
        data = self.df if df is None else df
        group_cols = [by] if isinstance(by, str) else list(by)
        g = data.groupby(group_cols)
        out = g["is_success"].agg(n_runs="count", n_success="sum")
        out["accuracy"] = (out["n_success"] / out["n_runs"]).round(4)
        return out.reset_index().sort_values(group_cols)

    def grasp_accuracy(self, by="policy", df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Grasp success rate, grouped by `by` (a column name or list of column
        names). Uses the grasp_result column and defaults to grouping by policy.
        """
        data = self.df if df is None else df
        group_cols = [by] if isinstance(by, str) else list(by)
        is_grasp_success = data[COL_GRASP_RESULT] == SUCCESS_VALUE
        out = is_grasp_success.groupby([data[c] for c in group_cols]).agg(
            n_runs="count", n_grasp_success="sum"
        )
        out["grasp_accuracy"] = (out["n_grasp_success"] / out["n_runs"]).round(4)
        return out.reset_index().sort_values(group_cols)

    # -- steps -------------------------------------------------------------
    def avg_steps(
        self, by="policy", only_result: str | None = None, df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Average (and median/count) steps, grouped by `by`. Set
        only_result="success" or "fail" to restrict to just those runs;
        default uses all runs regardless of outcome.
        """
        data = self.df if df is None else df
        if only_result is not None:
            data = data[data[COL_RESULT] == only_result]
        group_cols = [by] if isinstance(by, str) else list(by)
        out = data.groupby(group_cols)[COL_STEPS].agg(
            avg_steps="mean", median_steps="median", n_runs="count"
        )
        out["avg_steps"] = out["avg_steps"].round(2)
        return out.reset_index().sort_values(group_cols)

    def avg_steps_by_position(
        self, only_result: str | None = None, df: pd.DataFrame | None = None
    ) -> pd.DataFrame:
        """
        Pivot table: rows = position_code, columns = policy, values = average
        steps -- lets you compare policies against each other at each
        individual position code.
        """
        data = self.df if df is None else df
        if only_result is not None:
            data = data[data[COL_RESULT] == only_result]
        pivot = data.pivot_table(
            index=COL_POSITION, columns=COL_POLICY, values=COL_STEPS, aggfunc="mean"
        ).round(2)
        return pivot

    def accuracy_by_position(self, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Pivot table: rows = position_code, columns = policy, values = accuracy."""
        data = self.df if df is None else df
        pivot = data.pivot_table(
            index=COL_POSITION, columns=COL_POLICY, values="is_success", aggfunc="mean"
        ).round(4)
        return pivot

    # -- failure categories ------------------------------------------------
    def failure_breakdown(self, normalize: bool = True, df: pd.DataFrame | None = None) -> pd.DataFrame:
        """
        Cross-tab of policy x failure_category, counting only failed runs.
        normalize=True (default) shows each cell as a % of that policy's
        total failures; normalize=False shows raw counts.
        """
        data = self.df if df is None else df
        fails = data[data[COL_RESULT] == FAIL_VALUE]
        table = pd.crosstab(fails[COL_POLICY], fails["failure_category"])
        # ensure all known categories appear even if a policy had zero of them
        for cat, _ in FAILURE_CATEGORIES:
            if cat not in table.columns:
                table[cat] = 0
        if OTHER_CATEGORY not in table.columns:
            table[OTHER_CATEGORY] = 0
        ordered_cols = [c for c, _ in FAILURE_CATEGORIES] + [OTHER_CATEGORY]
        table = table[ordered_cols]
        if normalize:
            table = (table.div(table.sum(axis=1), axis=0) * 100).round(1)
        return table

    def uncategorized_notes(self, df: pd.DataFrame | None = None) -> pd.Series:
        """Show raw notes text that landed in the 'other' failure bucket, with counts."""
        data = self.df if df is None else df
        fails = data[data[COL_RESULT] == FAIL_VALUE]
        other = fails[fails["failure_category"] == OTHER_CATEGORY]
        return other[COL_NOTES].value_counts()

    # -- top-level report ----------------------------------------------
    def report(self, policy: str | None = None) -> None:
        df = self.df if policy is None else self.for_policy(policy)
        title = f"EVAL REPORT ({self.csv_path.name})" + (f" -- policy={policy}" if policy else "")
        _print_header(title)

        _print_header("Success / fail counts and accuracy by policy", level=2)
        acc = (df.groupby(COL_POLICY)["is_success"].agg(n_runs="count", n_success="sum")
               .assign(n_fail=lambda d: d["n_runs"] - d["n_success"]))
        acc["accuracy"] = (acc["n_success"] / acc["n_runs"]).round(4)
        acc = acc[["n_runs", "n_success", "n_fail", "accuracy"]]
        print(acc.to_string())

        # _print_header("Average steps by policy (all runs)", level=2)
        # print(self.avg_steps(df=df).to_string(index=False))

        _print_header("Grasp accuracy by policy", level=2)
        print(self.grasp_accuracy(df=df).to_string(index=False))

        _print_header("Average steps by policy, successes only", level=2)
        print(self.avg_steps(only_result=SUCCESS_VALUE, df=df).to_string(index=False))

        # _print_header("Average steps by policy, failures only", level=2)
        # print(self.avg_steps(only_result=FAIL_VALUE, df=df).to_string(index=False))

        # _print_header("Average steps by policy x position_code (all runs)", level=2)
        # print(self.avg_steps_by_position(df=df).to_string())

        _print_header("Accuracy by policy x position_code", level=2)
        print(self.accuracy_by_position(df=df).to_string())

        _print_header("Failure category breakdown by policy (% of that policy's failures)", level=2)
        print(self.failure_breakdown(normalize=True, df=df).to_string())

        # _print_header("Failure category breakdown by policy (raw counts)", level=2)
        # print(self.failure_breakdown(normalize=False, df=df).to_string())

        uncategorized = self.uncategorized_notes(df=df)
        if not uncategorized.empty:
            _print_header("Notes that didn't match a known category ('other') -- review these", level=2)
            print(uncategorized.to_string())


def _print_header(text: str, level: int = 1) -> None:
    print()
    if level == 1:
        print("=" * len(text))
        print(text)
        print("=" * len(text))
    else:
        print(f"--- {text} ---")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze policy eval-results CSV.")
    parser.add_argument("csv_path", help="Path to eval_results.csv")
    parser.add_argument("--policy", default=None, help="Restrict the report to a single policy")
    parser.add_argument(
        "--csv-out",
        default=None,
        help="Optional path to write the per-policy summary table (accuracy + avg steps) as CSV",
    )
    args = parser.parse_args()

    if not Path(args.csv_path).exists():
        print(f"File not found: {args.csv_path}", file=sys.stderr)
        sys.exit(1)

    ea = EvalAnalyzer(args.csv_path)
    ea.report(policy=args.policy)

    if args.csv_out:
        acc = ea.accuracy()
        steps = ea.avg_steps()
        summary = acc.merge(steps[[COL_POLICY, "avg_steps", "median_steps"]], on=COL_POLICY)
        summary.to_csv(args.csv_out, index=False)
        print(f"\nSaved per-policy summary to {args.csv_out}")


if __name__ == "__main__":
    main()