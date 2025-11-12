#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def ema(series: np.ndarray, alpha: float) -> np.ndarray:
    if len(series) == 0:
        return series
    out = np.empty_like(series, dtype=float)
    out[0] = float(series[0])
    for i in range(1, len(series)):
        out[i] = alpha * float(series[i]) + (1.0 - alpha) * out[i - 1]
    return out


def slope_and_cv(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    n = len(y)
    if n < 2:
        return np.nan, np.nan, np.nan, False

    x = np.arange(n, dtype=float)
    m, b = np.polyfit(x, y, 1)  # slope, intercept
    mu = np.nanmean(y)
    sd = np.nanstd(y)
    cv = (sd / (abs(mu) + 1e-8)) if np.isfinite(sd) and abs(mu) > 0 else np.nan
    norm_slope = (m / (abs(mu) + 1e-8)) if abs(mu) > 0 else np.nan
    converged = (np.isfinite(norm_slope) and np.isfinite(cv) and abs(norm_slope) < 0.01 and cv < 0.1)
    return float(m), float(norm_slope), float(cv), bool(converged)


def plot_series(x, y, out_base: Path, title: str, ylabel: str, xlabel: str = "Episode", ema_alpha: float = 0.2):
    ensure_dir(out_base.parent)
    fig = plt.figure(figsize=(8, 4.8))
    plt.plot(x, y, label="raw")
    if ema_alpha and ema_alpha > 0.0:
        plt.plot(x, ema(np.asarray(y), ema_alpha), linestyle="--", label=f"EMA α={ema_alpha}")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend(loc="best")
    plt.tight_layout()
    fig.savefig(str(out_base.with_suffix(".png")), dpi=300)
    fig.savefig(str(out_base.with_suffix(".pdf")))
    plt.close(fig)


def clean_numeric(series):
    return pd.to_numeric(series, errors="coerce")


def dedup_last(df: pd.DataFrame, key: str) -> pd.DataFrame:
    return df.sort_index().groupby(key, as_index=False).tail(1).sort_values(key).reset_index(drop=True)


def load_high_level_metrics(logs_dir: Path, drop_incomplete_tail: bool = False, min_step_ratio: float = 0.8):
    ep = logs_dir / "episode_metrics.csv"
    step = logs_dir / "high_level_step_rewards.csv"
    df_ep = None
    df_step_agg = None
    keep_eps = None

    # === High-level step (only scaled) ===
    if step.exists():
        df_step = pd.read_csv(step)
        for c in ["episode", "scaled_reward"]:
            if c in df_step.columns:
                df_step[c] = clean_numeric(df_step[c])

        if drop_incomplete_tail and "episode" in df_step.columns:
            cnt = df_step.groupby("episode").size().sort_index()
            if len(cnt) >= 1:
                base = cnt.iloc[:-1] if len(cnt) > 1 else cnt
                typical = int(np.median(base.values)) if len(base) > 0 else int(cnt.max())
                cutoff = int(np.ceil(typical * min_step_ratio))
                keep_eps = cnt[cnt >= cutoff].index
                before, after = len(cnt), len(keep_eps)
                print(f"[filter] 典型步數={typical}, 門檻={cutoff} ⇒ 保留 {after}/{before} 個 episodes")
                df_step = df_step[df_step["episode"].isin(keep_eps)].copy()

        if "episode" in df_step.columns:
            df_step_agg = df_step.groupby("episode", as_index=False).agg(
                hl_step_avg_reward=("scaled_reward", "mean"),  # rename to 'reward'
                count=("scaled_reward", "size"),
            )

    # === High-level episode metrics ===
    if ep.exists():
        df_ep = pd.read_csv(ep)
        for c in ["episode", "avg_reward", "avg_loss", "epsilon"]:
            if c in df_ep.columns:
                df_ep[c] = clean_numeric(df_ep[c])
        if "episode" in df_ep.columns:
            df_ep = dedup_last(df_ep, "episode")
            if drop_incomplete_tail and keep_eps is not None:
                df_ep = df_ep[df_ep["episode"].isin(keep_eps)].copy()

    return df_ep, df_step_agg


def load_low_level_metrics(logs_dir: Path):
    p = logs_dir / "low_level_episode_metrics.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    rename_map = {}
    if "avg_low_reward" not in df.columns and "avg_reward" in df.columns:
        rename_map["avg_reward"] = "avg_low_reward"
    if "avg_low_loss" not in df.columns and "avg_loss" in df.columns:
        rename_map["avg_loss"] = "avg_low_loss"
    if rename_map:
        df = df.rename(columns=rename_map)
    for c in ["episode", "avg_low_loss", "avg_low_reward", "epsilon"]:
        if c in df.columns:
            df[c] = clean_numeric(df[c])
    if "episode" in df.columns:
        df = dedup_last(df, "episode")
    return df


def load_eval_metrics(eval_dir: Path):
    p = eval_dir / "summary.csv"
    if not p.exists():
        return None
    df = pd.read_csv(p)
    if "avg_global_loss" in df.columns:
        df["avg_global_loss"] = clean_numeric(df["avg_global_loss"])
    if "tag" in df.columns:
        df["eval_index"] = np.arange(1, len(df) + 1)
        ep_num = df["tag"].astype(str).str.extract(r"ep(\d+)", expand=False)
        df["episode_from_tag"] = pd.to_numeric(ep_num, errors="coerce")
    return df


def append_summary_row(rows, name, y):
    if y is None or len(y) < 2:
        rows.append({
            "metric": name, "count": 0, "first": np.nan, "last": np.nan,
            "slope": np.nan, "norm_slope": np.nan, "cv": np.nan, "converged": False
        })
        return
    m, nm, cv, ok = slope_and_cv(y)
    rows.append({
        "metric": name,
        "count": int(len(y)),
        "first": float(np.nan_to_num(y[0])),
        "last": float(np.nan_to_num(y[-1])),
        "slope": float(m),
        "norm_slope": float(nm),
        "cv": float(cv),
        "converged": bool(ok)
    })


def main():
    parser = argparse.ArgumentParser(description="Make thesis-ready RL training plots (HL scaled-only reward)")
    parser.add_argument("--logs_dir", type=str, default="logs", help="directory containing logs/*.csv")
    parser.add_argument("--eval_dir", type=str, default="evaluation_logs", help="directory containing evaluation logs")
    parser.add_argument("--out_dir", type=str, default="figs", help="output directory for figures")
    parser.add_argument("--ema", type=float, default=0.2, help="EMA alpha (0 to disable)")
    parser.add_argument("--drop_incomplete_tail", action="store_true",
                        help="自動丟掉尾端未完成的 episode（依 step 計數）")
    parser.add_argument("--min_step_ratio", type=float, default=0.8,
                        help="不完整門檻：相對典型步數的比例，預設 0.8")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # --- High level ---
    df_hl, df_hl_step = load_high_level_metrics(
        logs_dir,
        drop_incomplete_tail=args.drop_incomplete_tail,
        min_step_ratio=args.min_step_ratio
    )

    if df_hl is not None and "episode" in df_hl.columns:
        x = df_hl["episode"].to_numpy()
        if "avg_reward" in df_hl.columns and df_hl["avg_reward"].notna().any():
            y = df_hl["avg_reward"].to_numpy(dtype=float)
            plot_series(x, y, out_dir / "HL_avg_reward", "High-Level Average Reward", "Avg Reward", ema_alpha=args.ema)
        if "avg_loss" in df_hl.columns and df_hl["avg_loss"].notna().any():
            y = df_hl["avg_loss"].to_numpy(dtype=float)
            plot_series(x, y, out_dir / "HL_avg_loss", "High-Level Average Loss", "Avg Loss", ema_alpha=args.ema)
        if "epsilon" in df_hl.columns and df_hl["epsilon"].notna().any():
            y = df_hl["epsilon"].to_numpy(dtype=float)
            plot_series(x, y, out_dir / "HL_epsilon", "High-Level Epsilon (exploration)", "Epsilon", ema_alpha=0.0)

    # --- High-level step rewards aggregated (SCALED ONLY, named as Reward) ---
    if df_hl_step is not None and "episode" in df_hl_step.columns:
        x = df_hl_step["episode"].to_numpy()
        if "hl_step_avg_reward" in df_hl_step.columns and df_hl_step["hl_step_avg_reward"].notna().any():
            y_reward = df_hl_step["hl_step_avg_reward"].to_numpy(dtype=float)
            plot_series(
                x, y_reward, out_dir / "HL_reward_per_ep",
                "High-Level Reward per Episode (aggregated)", "Avg Reward",
                ema_alpha=args.ema
            )

    # --- Low level ---
    df_ll = load_low_level_metrics(logs_dir)
    if df_ll is not None and "episode" in df_ll.columns:
        x = df_ll["episode"].to_numpy()
        if "avg_low_reward" in df_ll.columns and df_ll["avg_low_reward"].notna().any():
            y = df_ll["avg_low_reward"].to_numpy(dtype=float)
            plot_series(x, y, out_dir / "LL_avg_reward", "Low-Level Average Reward", "Avg Reward", ema_alpha=args.ema)
        if "avg_low_loss" in df_ll.columns and df_ll["avg_low_loss"].notna().any():
            y = df_ll["avg_low_loss"].to_numpy(dtype=float)
            plot_series(x, y, out_dir / "LL_avg_loss", "Low-Level Average Loss", "Avg Loss", ema_alpha=args.ema)
        if "epsilon" in df_ll.columns and df_ll["epsilon"].notna().any():
            y = df_ll["epsilon"].to_numpy(dtype=float)
            plot_series(x, y, out_dir / "LL_epsilon", "Low-Level Epsilon (exploration)", "Epsilon", ema_alpha=0.0)

    # --- Eval ---
    df_eval = load_eval_metrics(eval_dir)
    if df_eval is not None:
        if "avg_global_loss" in df_eval.columns and df_eval["avg_global_loss"].notna().any():
            y = df_eval["avg_global_loss"].to_numpy(dtype=float)
            if "episode_from_tag" in df_eval.columns and df_eval["episode_from_tag"].notna().any():
                x = df_eval["episode_from_tag"].fillna(df_eval["eval_index"]).to_numpy()
                xlabel = "Episode (from tag)"
            else:
                x = df_eval["eval_index"].to_numpy()
                xlabel = "Evaluation Index"
            plot_series(x, y, out_dir / "EVAL_avg_global_loss", "Evaluation: Avg Global Loss", "Avg Global Loss", xlabel=xlabel, ema_alpha=args.ema)

    # --- Summary table ---
    rows = []
    if df_hl is not None:
        if "avg_reward" in df_hl.columns and df_hl["avg_reward"].notna().any():
            append_summary_row(rows, "HL_avg_reward", df_hl["avg_reward"].to_numpy(dtype=float))
        if "avg_loss" in df_hl.columns and df_hl["avg_loss"].notna().any():
            append_summary_row(rows, "HL_avg_loss", df_hl["avg_loss"].to_numpy(dtype=float))
        if "epsilon" in df_hl.columns and df_hl["epsilon"].notna().any():
            append_summary_row(rows, "HL_epsilon", df_hl["epsilon"].to_numpy(dtype=float))

    if df_hl_step is not None and "hl_step_avg_reward" in df_hl_step.columns and df_hl_step["hl_step_avg_reward"].notna().any():
        append_summary_row(rows, "HL_reward_per_ep", df_hl_step["hl_step_avg_reward"].to_numpy(dtype=float))

    if df_ll is not None:
        if "avg_low_reward" in df_ll.columns and df_ll["avg_low_reward"].notna().any():
            append_summary_row(rows, "LL_avg_reward", df_ll["avg_low_reward"].to_numpy(dtype=float))
        if "avg_low_loss" in df_ll.columns and df_ll["avg_low_loss"].notna().any():
            append_summary_row(rows, "LL_avg_loss", df_ll["avg_low_loss"].to_numpy(dtype=float))
        if "epsilon" in df_ll.columns and df_ll["epsilon"].notna().any():
            append_summary_row(rows, "LL_epsilon", df_ll["epsilon"].to_numpy(dtype=float))

    if df_eval is not None and "avg_global_loss" in df_eval.columns:
        append_summary_row(rows, "EVAL_avg_global_loss", df_eval["avg_global_loss"].to_numpy(dtype=float))

    if rows:
        df_sum = pd.DataFrame(rows)
        df_sum.to_csv(out_dir / "metrics_summary.csv", index=False)

    # --- README ---
    readme = out_dir / "README_figs.txt"
    with open(readme, "w", encoding="utf-8") as f:
        f.write(
            "Figures generated by converge_hl_reward_only.py (High-Level raw removed; scaled renamed to Reward)\n"
            "Each figure includes raw curve and optional EMA smoothing.\n"
            "metrics_summary.csv lists slope, normalized slope (slope / mean), and CV (std/mean) for convergence interpretation.\n"
        )

    print(f"[OK] Plots saved to: {out_dir.resolve()}")
    if rows:
        print(f"[OK] Summary saved to: {(out_dir / 'metrics_summary.csv').resolve()}")


if __name__ == "__main__":
    main()
