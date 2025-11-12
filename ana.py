
# -*- coding: utf-8 -*-
"""
make_thesis_plots.py
--------------------
把目前專案中的 logs/ 與 evaluation_logs/ 讀進來，輸出「論文用」圖表與收斂指標：

輸入（若存在就讀，不存在就跳過）
- logs/episode_metrics.csv                # HL: episode, avg_reward, avg_loss, epsilon
- logs/low_level_episode_metrics.csv      # LL: episode, avg_low_loss, avg_low_reward, epsilon
- evaluation_logs/summary.csv             # Eval 摘要：avg_global_loss（尾 5 次平均）
- evaluation_logs/eval_episode_*_full_rounds.csv  # Eval 單次所有 round 權重，可用於補繪趨勢

- logs/high_reward_*.log                  # 高層每回合 reward 與選到的 slot 數（action_slot）
                                           # 欄位：round,scaled_reward,raw_reward,normalized_round,action_slot

輸出（到 figs/）
- HL_avg_reward.(png|pdf)
- HL_avg_loss.(png|pdf)
- HL_epsilon.(png|pdf)
- LL_avg_reward.(png|pdf)
- LL_avg_loss.(png|pdf)
- LL_epsilon.(png|pdf)
- EVAL_avg_global_loss.(png|pdf)
- HighLevel_action_dist.(png|pdf)         # 動作分佈堆疊圖（各 agent 的 slot 數分佈）
- metrics_summary.csv                     # 每條序列的 slope / norm_slope / CV / converged 布林

風格規範：
- 僅使用 matplotlib，不用 seaborn
- 每張圖單獨一個 figure（無 subplot）
- 不指定顏色（維持 matplotlib 預設）
"""

from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def ema(series: pd.Series, alpha: float) -> pd.Series:
    if alpha is None or alpha <= 0:
        return pd.Series([np.nan] * len(series), index=series.index)
    return series.ewm(alpha=alpha, adjust=False).mean()

def slope_and_cv(y: np.ndarray):
    """線性回歸斜率 + CV（最後 tail_n 片段）；回傳 (slope, norm_slope, CV, ok)"""
    y = np.asarray(y, dtype=float)
    y = y[~np.isnan(y)]
    if len(y) < 5:
        return np.nan, np.nan, np.nan, False
    tail_n = max(10, int(len(y) * 0.3))  # 使用最後 30%（至少 10 個點）
    tail = y[-tail_n:]
    x = np.arange(len(tail), dtype=float)
    X = np.vstack([x, np.ones_like(x)]).T
    try:
        # 最小平方法
        (a, b), *_ = np.linalg.lstsq(X, tail, rcond=None)
        slope = float(a)
    except Exception:
        slope = np.nan
    mean = float(np.mean(tail)) if len(tail) > 0 else np.nan
    std = float(np.std(tail)) if len(tail) > 0 else np.nan
    cv = (std / (mean + 1e-12)) if np.isfinite(mean) and abs(mean) > 1e-12 else np.nan
    norm_slope = (slope / (mean + 1e-12)) if np.isfinite(mean) and abs(mean) > 1e-12 else np.nan
    ok = np.isfinite(slope) and np.isfinite(cv) and np.isfinite(norm_slope)
    return slope, norm_slope, cv, ok

def plot_series(x, y, title, xlabel, ylabel, out_prefix, ema_alpha=0.2):
    """畫 raw 與 EMA，並回傳指標"""
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    if len(x) == 0 or len(y) == 0:
        return None

    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()
    ax.plot(x, y, marker='o', linestyle='--', linewidth=1.5, label='Raw')
    if ema_alpha and ema_alpha > 0:
        y_ema = ema(pd.Series(y), ema_alpha).to_numpy()
        ax.plot(x, y_ema, linestyle='-', linewidth=2.0, label=f'EMA α={ema_alpha}')
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()
    out_png = f"{out_prefix}.png"
    out_pdf = f"{out_prefix}.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)

    slope, ns, cv, ok = slope_and_cv(y)
    converged = (abs(ns) < 0.01 and cv < 0.10) if ok else False
    return {
        "title": title,
        "points": len(y),
        "slope": slope,
        "norm_slope": ns,
        "cv": cv,
        "converged": converged,
        "out_png": out_png,
        "out_pdf": out_pdf
    }

def safe_read_csv(path, **kwargs):
    try:
        return pd.read_csv(path, **kwargs)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding='utf-16-le', **kwargs)

def make_action_dist_plot(high_reward_logs, max_slots, out_prefix):
    """
    讀取 logs/high_reward_*.log 檔，欄位：round,scaled_reward,raw_reward,normalized_round,action_slot
    將 action_slot(0-based) 轉為 slot_count=action_slot+1，
    繪製「每個 slot 選擇次數」之堆疊柱狀圖（按照 agent 堆疊）。
    """
    # 收集每個 agent 的分佈
    per_agent = {}
    for p in high_reward_logs:
        name = p.stem  # high_reward_0
        agent_id = name.split('_')[-1]
        try:
            df = safe_read_csv(p)
        except Exception:
            continue
        if 'action_slot' not in df.columns:
            continue
        slots = (df['action_slot'].astype(int) + 1).clip(1, max_slots)  # 1..max_slots
        counts = slots.value_counts().reindex(range(1, max_slots+1), fill_value=0).sort_index()
        per_agent[agent_id] = counts.to_numpy()

    if not per_agent:
        return None

    # 組合為 (slot=1..max_slots) x (#agents)
    agents = sorted(per_agent.keys(), key=lambda s: int(s) if s.isdigit() else s)
    mat = np.column_stack([per_agent[a] for a in agents])  # shape: (max_slots, num_agents)

    x = np.arange(1, max_slots+1)
    fig = plt.figure(figsize=(10, 5))
    ax = plt.gca()

    bottom = np.zeros_like(x, dtype=float)
    for j, a in enumerate(agents):
        ax.bar(x, mat[:, j], bottom=bottom, label=f'Agent {a}')
        bottom = bottom + mat[:, j]

    ax.set_title("高層動作分佈（slot 數）")
    ax.set_xlabel("選到的 slot 數（1 = 最少）")
    ax.set_ylabel("出現次數")
    ax.grid(True, axis='y')
    ax.legend(ncol=max(1, len(agents)//2))
    fig.tight_layout()
    out_png = f"{out_prefix}.png"
    out_pdf = f"{out_prefix}.pdf"
    fig.savefig(out_png, dpi=200)
    fig.savefig(out_pdf)
    plt.close(fig)
    return {"out_png": out_png, "out_pdf": out_pdf}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs_dir", default="logs")
    ap.add_argument("--eval_dir", default="evaluation_logs")
    ap.add_argument("--out_dir",  default="figs")
    ap.add_argument("--ema", type=float, default=0.2)
    ap.add_argument("--max_slots", type=int, default=10)
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    eval_dir = Path(args.eval_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = []

    # === 1) High-level episode metrics ===
    hl_csv = logs_dir / "episode_metrics.csv"
    if hl_csv.exists():
        df = safe_read_csv(hl_csv)
        # 欄位名容錯（常見拼法）
        col_reward = "avg_reward" if "avg_reward" in df.columns else (df.columns[1] if len(df.columns) > 1 else None)
        col_loss   = "avg_loss"   if "avg_loss"   in df.columns else (df.columns[2] if len(df.columns) > 2 else None)
        col_eps    = "epsilon"    if "epsilon"    in df.columns else None
        if "episode" not in df.columns:
            df["episode"] = np.arange(1, len(df)+1)

        if col_reward in df:
            m = plot_series(df["episode"], df[col_reward], "高層：每集平均 Reward", "Episode", "Avg Reward",
                            str(out_dir / "HL_avg_reward"), ema_alpha=args.ema)
            if m: 
                m["series"] = "HL_avg_reward"; metrics.append(m)
        if col_loss in df:
            m = plot_series(df["episode"], df[col_loss], "高層：每集平均 Loss", "Episode", "Avg Loss",
                            str(out_dir / "HL_avg_loss"), ema_alpha=args.ema)
            if m: 
                m["series"] = "HL_avg_loss"; metrics.append(m)
        if col_eps in df:
            m = plot_series(df["episode"], df[col_eps], "高層：ε 探索率", "Episode", "Epsilon",
                            str(out_dir / "HL_epsilon"), ema_alpha=0.0)
            if m: 
                m["series"] = "HL_epsilon"; metrics.append(m)

    # === 2) Low-level episode metrics ===
    ll_csv = logs_dir / "low_level_episode_metrics.csv"
    if ll_csv.exists():
        df = safe_read_csv(ll_csv)
        if "episode" not in df.columns:
            df["episode"] = np.arange(1, len(df)+1)
        col_ll_loss   = "avg_low_loss"    if "avg_low_loss" in df.columns else None
        col_ll_reward = "avg_low_reward"  if "avg_low_reward" in df.columns else None
        col_ll_eps    = "epsilon"         if "epsilon" in df.columns else None

        if col_ll_reward in df:
            m = plot_series(df["episode"], df[col_ll_reward], "低層：每集平均 Reward", "Episode", "Avg Reward",
                            str(out_dir / "LL_avg_reward"), ema_alpha=args.ema)
            if m: 
                m["series"] = "LL_avg_reward"; metrics.append(m)
        if col_ll_loss in df:
            m = plot_series(df["episode"], df[col_ll_loss], "低層：每集平均 Loss", "Episode", "Avg Loss",
                            str(out_dir / "LL_avg_loss"), ema_alpha=args.ema)
            if m: 
                m["series"] = "LL_avg_loss"; metrics.append(m)
        if col_ll_eps in df:
            m = plot_series(df["episode"], df[col_ll_eps], "低層：ε 探索率", "Episode", "Epsilon",
                            str(out_dir / "LL_epsilon"), ema_alpha=0.0)
            if m: 
                m["series"] = "LL_epsilon"; metrics.append(m)

    # === 3) Eval summary ===
    eval_sum_csv = eval_dir / "summary.csv"
    if eval_sum_csv.exists():
        df = safe_read_csv(eval_sum_csv)
        # 容錯欄位名：可能是 avg_global_loss
        col = "avg_global_loss" if "avg_global_loss" in df.columns else None
        if col:
            # x 軸用累積索引（或 timestamp 排序）
            if "timestamp" in df.columns:
                df = df.sort_values("timestamp").reset_index(drop=True)
            x = np.arange(1, len(df)+1)
            m = plot_series(x, df[col], "EVAL：Avg Global Loss（各次評估）", "評估序號", "Avg Global Loss",
                            str(out_dir / "EVAL_avg_global_loss"), ema_alpha=args.ema)
            if m: 
                m["series"] = "EVAL_avg_global_loss"; metrics.append(m)

    # === 4) High-level action distribution (stacked) ===
    high_reward_logs = sorted((logs_dir).glob("high_reward_*.log"))
    if high_reward_logs:
        res = make_action_dist_plot(high_reward_logs, args.max_slots, str(out_dir / "HighLevel_action_dist"))

    # === 5) 寫出 metrics_summary.csv ===
    if metrics:
        out_csv = out_dir / "metrics_summary.csv"
        pd.DataFrame(metrics).to_csv(out_csv, index=False)

    # === 6) 小字說明 ===
    readme = out_dir / "README_figs.txt"
    readme.write_text(
        "本資料夾為自動產生的論文繪圖與指標彙整：\n"
        "1) 每張圖包含 Raw（點線）與 EMA 平滑（實線），EMA 係數可用 --ema 指定。\n"
        "2) metrics_summary.csv 的 slope 與 CV 是用最後 30%（至少 10 個點）的資料估計；"
        "   收斂判準為 |norm_slope| < 0.01 且 CV < 0.10（可自行調整程式碼）。\n"
        "3) HighLevel_action_dist 以 logs/high_reward_*.log 的 action_slot 欄位計數後，"
        "   堆疊柱狀（每個 slot 的次數，按 agent 堆疊）。\n",
        encoding="utf-8"
    )

if __name__ == "__main__":
    main()
