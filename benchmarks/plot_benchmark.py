#!/usr/bin/env python3
"""AetherMoE — benchmarks/plot_benchmark.py

Milestone 4, Step 3: turns bench_cli's summary CSV into the "latency/
throughput charts" the spec asks for, and (with --consistency) checks
run-to-run reproducibility across repeated executions -- the M4 testing
plan's explicit ask: "multi-hour automated runs producing reproducible
P50/P95/P99 charts across repeated executions, checked for run-to-run
consistency."

Usage:
    python3 plot_benchmark.py bench_run_summary.csv
    python3 plot_benchmark.py bench_run_summary.csv --consistency
    python3 plot_benchmark.py bench_run_summary.csv --output charts/

Requires: pandas, matplotlib (same dependency shape as Milestone 2's own
Python benchmark scripts -- see requirements-m2.txt).
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def load_summary(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} has no data rows -- did bench_cli actually run?")
    return df


def plot_latency_percentiles(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in (
        (axes[0], "ttft", "Time-to-First-Token"),
        (axes[1], "inter_token", "Inter-token latency"),
    ):
        x = df["run_index"]
        # Plot in microseconds -- these numbers come out of telemetry.hpp
        # in nanoseconds (see MetricStats), and nanosecond y-axes are
        # unreadable at this scale.
        ax.plot(x, df[f"{metric}_p50_ns"] / 1000.0, marker="o", label="P50")
        ax.plot(x, df[f"{metric}_p95_ns"] / 1000.0, marker="o", label="P95")
        ax.plot(x, df[f"{metric}_p99_ns"] / 1000.0, marker="o", label="P99")
        ax.set_title(title)
        ax.set_xlabel("run index")
        ax.set_ylabel("latency (µs)")
        ax.legend()
        ax.grid(True, alpha=0.3)

    fig.suptitle("AetherMoE benchmark: latency percentiles across runs")
    fig.tight_layout()
    out_path = out_dir / "latency_percentiles.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_throughput(df: pd.DataFrame, out_dir: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].bar(df["run_index"], df["requests_per_second"], color="steelblue")
    axes[0].set_title("Completed requests/sec")
    axes[0].set_xlabel("run index")
    axes[0].set_ylabel("req/s")
    axes[0].grid(True, axis="y", alpha=0.3)

    axes[1].bar(df["run_index"], df["tokens_per_second"], color="darkorange")
    axes[1].set_title("Generated tokens/sec")
    axes[1].set_xlabel("run index")
    axes[1].set_ylabel("tokens/s")
    axes[1].grid(True, axis="y", alpha=0.3)

    fig.suptitle("AetherMoE benchmark: throughput across runs")
    fig.tight_layout()
    out_path = out_dir / "throughput.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_backpressure_and_drops(df: pd.DataFrame, out_dir: Path) -> Path:
    # Not required by the spec's charts, but exactly the kind of thing a
    # multi-hour run needs surfaced rather than buried in a log --
    # rejected requests (real backpressure) and dropped telemetry samples
    # (ring buffer saturation, see telemetry.hpp) are both "accounted for,
    # not silently lost" numbers worth a glance, not just a raw count.
    fig, ax = plt.subplots(figsize=(8, 5))
    width = 0.35
    x = df["run_index"]
    ax.bar(x - width / 2, df["rejected_backpressure"], width, label="rejected (backpressure)")
    ax.bar(x + width / 2, df["dropped_telemetry_samples"], width, label="dropped telemetry samples")
    ax.set_title("Backpressure and telemetry-drop accounting")
    ax.set_xlabel("run index")
    ax.set_ylabel("count")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = out_dir / "backpressure_and_drops.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def check_consistency(df: pd.DataFrame) -> int:
    """Reports run-to-run consistency of the P50/P95/P99 columns -- the
    M4 testing plan's explicit ask. Returns a process exit code (0 if
    consistent, 1 if any metric's coefficient of variation exceeds a
    generous 25% threshold across runs -- generous deliberately, since
    the point is to catch a run that's wildly different, e.g. from a
    resource leak building up over repeats, not to demand
    lab-instrument-grade repeatability from a benchmark tool).
    """
    if len(df) < 2:
        print("only one run in this summary -- nothing to compare for consistency.")
        return 0

    metrics = [
        "ttft_p50_ns", "ttft_p95_ns", "ttft_p99_ns",
        "inter_token_p50_ns", "inter_token_p95_ns", "inter_token_p99_ns",
        "requests_per_second", "tokens_per_second",
    ]
    print(f"{'metric':<24}{'mean':>14}{'stddev':>14}{'cv%':>10}{'verdict':>12}")
    threshold_pct = 25.0
    any_inconsistent = False
    for m in metrics:
        mean = df[m].mean()
        std = df[m].std()
        cv_pct = (std / mean * 100.0) if mean else 0.0
        verdict = "OK" if cv_pct <= threshold_pct else "INCONSISTENT"
        if verdict == "INCONSISTENT":
            any_inconsistent = True
        print(f"{m:<24}{mean:>14.1f}{std:>14.1f}{cv_pct:>9.1f}%{verdict:>12}")

    if any_inconsistent:
        print(f"\nOne or more metrics varied more than {threshold_pct:.0f}% (coefficient of "
              "variation) across runs -- worth investigating before trusting these numbers "
              "as reproducible (e.g. a resource leak building up across repeats, or "
              "background load on the machine during the run).")
        return 1
    print(f"\nAll metrics within {threshold_pct:.0f}% run-to-run variation -- consistent.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", help="path to bench_cli's *_summary.csv")
    parser.add_argument("--output", default=".", help="directory to write chart PNGs into")
    parser.add_argument("--consistency", action="store_true",
                         help="also print a run-to-run consistency report and exit nonzero if inconsistent")
    args = parser.parse_args()

    df = load_summary(args.csv_path)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    latency_path = plot_latency_percentiles(df, out_dir)
    throughput_path = plot_throughput(df, out_dir)
    drops_path = plot_backpressure_and_drops(df, out_dir)
    print(f"wrote {latency_path}")
    print(f"wrote {throughput_path}")
    print(f"wrote {drops_path}")

    if args.consistency:
        print()
        return check_consistency(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
