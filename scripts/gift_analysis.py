"""
GIFT-Eval benchmark analysis.

Downloads all model results from the HuggingFace leaderboard space, builds a
flat dataframe, then produces a set of plots saved under plots/gift/.

Usage:
    uv run python scripts/gift_analysis.py
    uv run python scripts/gift_analysis.py --out-dir my_plots
"""

import argparse
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_URL = "https://huggingface.co/spaces/Salesforce/GIFT-Eval/raw/main/results"

MODEL_DIRS = [
    "Chronos_small", "CleanTS-65M", "Credence", "DLinear",
    "DeOSAlphaTimeGPTPredictor-2025", "FFM", "FlowState-9.1M",
    "Kairos_10m", "Kairos_23m", "Kairos_50m",
    "Lag-Llama", "Lingjiang", "Migas-1.0",
    "Moirai2", "MoiraiAgent", "Moirai_base", "Moirai_large", "Moirai_small",
    "N-BEATS", "PatchTST-FM-r1", "PatchTST",
    "Reverso-Nano", "Reverso-Small", "Reverso",
    "Samay", "Synapse", "TSOrchestra",
    "TTM-R1-Pretrained", "TTM-R2-Finetuned", "TTM-R2-Pretrained",
    "TempoPFN", "TiRex", "TimeCopilot", "TimesFM-2.5",
    "Toto_Open_Base_1.0",
    "Xihe-max", "Xihe-ultra",
    "YingLong_110m", "YingLong_300m", "YingLong_50m", "YingLong_6m",
    "auto_arima", "auto_ets", "auto_theta",
    "chronos-2", "chronos-2-synth", "chronos_base", "chronos_bolt_base",
]

METRIC = "MASE[0.5]"

PAL = {
    "statistical":   "#e69f00",
    "foundation":    "#0072b2",
    "deep-learning": "#009e73",
    "other":         "#cc79a7",
}

FREQ_PERIOD_HOURS = {
    "1T": 1/60, "5T": 5/60, "10T": 10/60, "15T": 15/60, "30T": 0.5,
    "H": 1, "4H": 4, "D": 24, "W": 168,
    "M": 720, "ME": 720, "Q": 2160, "QE": 2160, "A": 8760, "YE": 8760,
}

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def fetch_config(model_dir: str) -> dict:
    url = f"{BASE_URL}/{model_dir}/config.json"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {}


def fetch_results(model_dir: str) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{model_dir}/all_results.csv"
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return pd.read_csv(io.StringIO(r.text))
    except Exception:
        pass
    return None


def load_all() -> pd.DataFrame:
    frames = []
    skipped = []
    for model_dir in MODEL_DIRS:
        cfg = fetch_config(model_dir)
        df = fetch_results(model_dir)
        if df is None or df.empty:
            skipped.append(model_dir)
            continue
        for k, v in cfg.items():
            df[f"cfg_{k}"] = v
        df["model_dir"] = model_dir
        frames.append(df)

    if skipped:
        print(f"Skipped (no data): {skipped}")
    print(f"Loaded {len(frames)} models")
    return pd.concat(frames, ignore_index=True)


def build_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    raw = raw.copy()
    raw.columns = [c.replace("eval_metrics/", "") for c in raw.columns]

    # Parse dataset → name / freq / horizon
    parts = raw["dataset"].str.split("/", expand=True, n=2)
    raw["ds_name"] = parts[0]
    raw["ds_freq"] = parts[1]
    raw["ds_horizon"] = parts[2]

    # Canonical model name
    raw["model_name"] = (
        raw["cfg_model"].fillna(raw["model_dir"]) if "cfg_model" in raw.columns
        else raw["model_dir"]
    )

    # Model type
    raw["model_type"] = raw.get("cfg_model_type", pd.Series(dtype=str))

    def categorise(row):
        mt = str(row.get("model_type") or "").lower()
        if "statistical" in mt:
            return "statistical"
        if mt in ("pretrained", "zero-shot", "foundation"):
            return "foundation"
        if "deep" in mt or "learning" in mt:
            return "deep-learning"
        return "other"

    raw["category"] = raw.apply(categorise, axis=1)
    raw["org"] = raw.get("cfg_org", pd.Series(dtype=str))

    df = raw.dropna(subset=[METRIC]).copy()
    print(
        f"Rows: {len(df)}, Models: {df['model_name'].nunique()}, "
        f"Datasets: {df['dataset'].nunique()}"
    )
    return df


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def map_freq_hours(f: str) -> float:
    f = str(f).upper()
    for k, v in sorted(FREQ_PERIOD_HOURS.items(), key=lambda x: -len(x[0])):
        if f.endswith(k):
            prefix = f[: -len(k)]
            mult = int(prefix) if prefix.isdigit() else 1
            return v * mult
    return float("nan")


def freq_bucket(f: str) -> str:
    h = map_freq_hours(f)
    if np.isnan(h):
        return "unknown"
    if h < 1:
        return "sub-hourly"
    if h <= 4:
        return "hourly"
    if h <= 48:
        return "daily"
    return "weekly+"


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------

def build_head2head(df: pd.DataFrame) -> pd.DataFrame:
    stat = df[df["category"] == "statistical"]
    fnd = df[df["category"] == "foundation"]

    stat_best = stat.groupby("dataset")[METRIC].min().rename("stat_best")
    found_best = fnd.groupby("dataset")[METRIC].min().rename("found_best")
    found_med  = fnd.groupby("dataset")[METRIC].median().rename("found_median")
    found_std  = fnd.groupby("dataset")[METRIC].std().rename("found_std")

    ds_meta = (
        df[["dataset", "ds_name", "ds_freq", "ds_horizon", "domain", "num_variates"]]
        .drop_duplicates("dataset")
        .set_index("dataset")
    )

    h2h = pd.concat([stat_best, found_best, found_med, found_std], axis=1).dropna(subset=["stat_best", "found_best"])
    h2h = h2h.join(ds_meta)

    h2h["log_ratio"] = np.log(h2h["found_best"] / h2h["stat_best"])
    h2h["stat_wins"] = h2h["log_ratio"] > 0
    h2h["winner"] = np.where(h2h["stat_wins"], "statistical", "foundation")
    h2h["freq_bucket"] = h2h["ds_freq"].apply(freq_bucket)
    h2h["horizon"] = pd.Categorical(h2h["ds_horizon"], ["short", "medium", "long"], ordered=True)
    h2h["cv"] = h2h["found_std"] / h2h["found_median"]
    h2h["log_freq_hours"] = h2h["ds_freq"].apply(map_freq_hours).pipe(np.log1p)
    h2h["horizon_ord"] = h2h["horizon"].cat.codes
    h2h["log_variates"] = np.log1p(h2h["num_variates"].astype(float))
    return h2h


def build_rank_pivot(df: pd.DataFrame) -> pd.DataFrame:
    pivot = df.pivot_table(index="dataset", columns="model_name", values=METRIC, aggfunc="median")
    keep = pivot[pivot.notna().sum(axis=1) >= pivot.shape[1] * 0.4]
    ranks = keep.rank(axis=1, method="average", na_option="keep")
    mean_rank = ranks.mean().sort_values()
    return ranks[mean_rank.index]


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save(fig: plt.Figure, path: Path, name: str):
    out = path / name
    fig.savefig(out, bbox_inches="tight", dpi=140)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_overall_ranking(df: pd.DataFrame, out: Path):
    model_avg = (
        df.groupby(["model_name", "category"])[METRIC]
        .median().reset_index()
        .sort_values(METRIC)
    )
    colors = model_avg["category"].map(PAL)

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(model_avg["model_name"], model_avg[METRIC], color=colors)
    ax.set_xticklabels(model_avg["model_name"], rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Median MASE across datasets")
    ax.set_title("Overall benchmark ranking (lower = better)")
    ax.legend(
        handles=[mpatches.Patch(color=v, label=k) for k, v in PAL.items()],
        loc="upper left",
    )
    save(fig, out, "01_overall_ranking.png")


def plot_mase_by_domain(df: pd.DataFrame, out: Path):
    domain_cat = (
        df.groupby(["domain", "category"])[METRIC]
        .median().unstack("category")
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    domain_cat.plot(
        kind="bar", ax=ax,
        color=[PAL.get(c, "grey") for c in domain_cat.columns],
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.set_ylabel("Median MASE")
    ax.set_title("Median MASE by domain and model category")
    ax.legend(title="category")
    save(fig, out, "02_mase_by_domain.png")


def plot_waterfall(h2h: pd.DataFrame, out: Path):
    srt = h2h.sort_values("log_ratio", ascending=False)
    colors = ["#e69f00" if v > 0 else "#0072b2" for v in srt["log_ratio"]]

    fig, ax = plt.subplots(figsize=(16, 5))
    ax.bar(range(len(srt)), srt["log_ratio"], color=colors, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(range(len(srt)))
    ax.set_xticklabels(srt.index, rotation=90, fontsize=5)
    ax.set_ylabel("log(found_best / stat_best)\n(+) = statistical wins,  (−) = foundation wins")
    ax.set_title("Per-dataset advantage: statistical (orange) vs foundation (blue)")
    ax.legend(
        handles=[
            mpatches.Patch(color="#e69f00", label="statistical wins"),
            mpatches.Patch(color="#0072b2", label="foundation wins"),
        ]
    )
    save(fig, out, "03_waterfall.png")


def plot_scatter_drivers(h2h: pd.DataFrame, out: Path):
    domains = h2h["domain"].dropna().unique()
    domain_pal = dict(zip(domains, plt.cm.tab10.colors[: len(domains)]))

    plot_cfg = [
        ("log_freq_hours", "Log frequency period (hours)"),
        ("horizon_ord",    "Horizon  (0=short, 1=med, 2=long)"),
        ("log_variates",   "Log num variates"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for (xvar, xlabel), ax in zip(plot_cfg, axes):
        sub = h2h[[xvar, "log_ratio", "domain"]].dropna()
        for dom, grp in sub.groupby("domain"):
            ax.scatter(grp[xvar], grp["log_ratio"], label=dom, alpha=0.7, s=30,
                       color=domain_pal.get(dom, "grey"))
        m, b = np.polyfit(sub[xvar], sub["log_ratio"], 1)
        xs = np.linspace(sub[xvar].min(), sub[xvar].max(), 100)
        ax.plot(xs, m * xs + b, "k--", linewidth=1.5)
        ax.axhline(0, color="grey", linewidth=0.8)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("log(found/stat)  +ve = stat wins")
        r, p = stats.pearsonr(sub[xvar], sub["log_ratio"])
        ax.set_title(f"r={r:+.2f}  p={p:.3f}")

    axes[0].legend(fontsize=7, loc="upper right")
    fig.suptitle("Drivers of statistical vs foundation advantage", y=1.01)
    fig.tight_layout()
    save(fig, out, "04_scatter_drivers.png")


def plot_boxplots(h2h: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    order_dom = h2h.groupby("domain")["log_ratio"].median().sort_values(ascending=False).index
    sns.boxplot(data=h2h, x="domain", y="log_ratio", order=order_dom,
                ax=axes[0], palette="tab10")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")
    axes[0].set_title("Statistical advantage by domain")
    axes[0].set_ylabel("log(found/stat)  +ve = stat wins")

    sns.boxplot(data=h2h, x="ds_horizon", y="log_ratio",
                order=["short", "medium", "long"], ax=axes[1], palette="Set2")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Statistical advantage by horizon")
    axes[1].set_ylabel("")

    fig.tight_layout()
    save(fig, out, "05_boxplots_domain_horizon.png")


def plot_freq_stacked_bar(h2h: pd.DataFrame, out: Path):
    ct = (
        pd.crosstab(h2h["freq_bucket"], h2h["winner"], normalize="index") * 100
    ).reindex(["sub-hourly", "hourly", "daily", "weekly+"])
    # ensure both columns exist
    for col in ("statistical", "foundation"):
        if col not in ct.columns:
            ct[col] = 0.0

    fig, ax = plt.subplots(figsize=(7, 4))
    ct[["foundation", "statistical"]].plot(
        kind="bar", stacked=True, ax=ax,
        color=["#0072b2", "#e69f00"],
    )
    ax.set_ylabel("% of datasets")
    ax.set_title("Who wins by data frequency?")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.legend(title="winner", loc="lower right")
    fig.tight_layout()
    save(fig, out, "06_freq_stacked_bar.png")


def plot_rank_heatmap(df: pd.DataFrame, out: Path):
    ranks = build_rank_pivot(df)

    ds_meta = (
        df[["dataset", "domain"]]
        .drop_duplicates("dataset")
        .set_index("dataset")
    )
    domain_order = ds_meta.reindex(ranks.index)["domain"].sort_values()
    ranks = ranks.loc[domain_order.index]

    n_ds = len(ranks)
    fig, ax = plt.subplots(figsize=(22, max(8, n_ds * 0.22)))
    sns.heatmap(
        ranks.T,
        cmap="RdYlGn_r",
        ax=ax,
        linewidths=0,
        cbar_kws={"label": "Rank (1=best)"},
        xticklabels=True,
        yticklabels=True,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90, fontsize=5)
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=7)
    ax.set_title("Per-dataset rank heatmap (models sorted by mean rank, datasets sorted by domain)", pad=10)

    # Domain separator lines
    prev, x = None, 0
    for ds in domain_order.index:
        dom = domain_order[ds]
        if dom != prev and prev is not None:
            ax.axvline(x, color="white", linewidth=2)
        prev = dom
        x += 1

    fig.tight_layout()
    save(fig, out, "07_rank_heatmap.png")


def plot_clustermap(df: pd.DataFrame, out: Path):
    ranks = build_rank_pivot(df)
    norm = ranks.div(ranks.max(axis=1), axis=0)
    clean = norm.dropna(thresh=int(norm.shape[1] * 0.5), axis=0)
    clean = clean.dropna(thresh=int(clean.shape[0] * 0.5), axis=1)
    clean = clean.fillna(clean.median())

    g = sns.clustermap(
        clean.T,
        cmap="RdYlGn_r",
        figsize=(22, max(7, clean.shape[1] * 0.3)),
        linewidths=0,
        xticklabels=True,
        yticklabels=True,
        dendrogram_ratio=(0.1, 0.15),
        cbar_pos=(0.02, 0.8, 0.02, 0.15),
    )
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=90, fontsize=5)
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), fontsize=7)
    g.figure.suptitle("Clustered performance profile (normalised ranks)", y=1.01)
    save(g.figure, out, "08_clustermap.png")


def plot_foundation_spread(h2h: pd.DataFrame, out: Path):
    domains = h2h["domain"].dropna().unique()
    domain_pal = dict(zip(domains, plt.cm.tab10.colors[: len(domains)]))

    sub = h2h.dropna(subset=["cv"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))

    ax = axes[0]
    for dom, grp in sub.groupby("domain"):
        ax.scatter(grp["stat_best"], grp["found_median"], label=dom, alpha=0.7, s=35,
                   color=domain_pal.get(dom, "grey"))
    lims = [
        min(sub["stat_best"].min(), sub["found_median"].min()),
        max(sub["stat_best"].max(), sub["found_median"].max()),
    ]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("Best statistical MASE")
    ax.set_ylabel("Median foundation MASE")
    ax.set_title("Below diagonal = foundation wins")
    ax.legend(fontsize=7)

    ax = axes[1]
    sc = ax.scatter(sub["cv"], sub["log_ratio"], alpha=0.6, c=sub["log_ratio"],
                    cmap="RdBu_r", vmin=-1.5, vmax=1.5, s=35)
    ax.axhline(0, color="black", linewidth=0.8)
    m, b = np.polyfit(sub["cv"], sub["log_ratio"], 1)
    xs = np.linspace(sub["cv"].min(), sub["cv"].max(), 100)
    ax.plot(xs, m * xs + b, "k--", linewidth=1.5)
    ax.set_xlabel("CV of foundation models (spread)")
    ax.set_ylabel("log(found/stat)  +ve = stat wins")
    ax.set_title("Foundation spread vs statistical advantage")
    plt.colorbar(sc, ax=ax, label="log_ratio")

    r, p = stats.pearsonr(sub["cv"], sub["log_ratio"])
    ax.set_title(f"Foundation spread vs advantage  (r={r:+.2f}, p={p:.3f})")

    fig.tight_layout()
    save(fig, out, "09_foundation_spread.png")


# ---------------------------------------------------------------------------
# Summary tables
# ---------------------------------------------------------------------------

def print_summary(h2h: pd.DataFrame):
    print("\n" + "=" * 70)
    n = len(h2h)
    sw = h2h["stat_wins"].sum()
    print(f"Statistical baseline beats best foundation model on {sw}/{n} datasets ({100*sw/n:.0f}%)\n")

    print("--- By domain ---")
    dom = (
        h2h.groupby("domain")
        .agg(
            n=("log_ratio", "size"),
            stat_wins_pct=("stat_wins", lambda x: f"{100*x.mean():.0f}%"),
            median_log_ratio=("log_ratio", "median"),
        )
        .sort_values("median_log_ratio", ascending=False)
    )
    print(dom.to_string())

    print("\n--- By frequency bucket ---")
    freq = (
        h2h.groupby("freq_bucket")
        .agg(
            n=("log_ratio", "size"),
            stat_wins_pct=("stat_wins", lambda x: f"{100*x.mean():.0f}%"),
            median_log_ratio=("log_ratio", "median"),
        )
        .reindex(["sub-hourly", "hourly", "daily", "weekly+"])
    )
    print(freq.to_string())

    print("\n--- By horizon ---")
    hor = (
        h2h.groupby("ds_horizon")
        .agg(
            n=("log_ratio", "size"),
            stat_wins_pct=("stat_wins", lambda x: f"{100*x.mean():.0f}%"),
            median_log_ratio=("log_ratio", "median"),
        )
        .reindex(["short", "medium", "long"])
    )
    print(hor.to_string())

    print("\n--- Datasets where statistical wins most (top 10) ---")
    cols = ["ds_name", "ds_freq", "ds_horizon", "domain", "num_variates", "stat_best", "found_best", "log_ratio"]
    print(h2h.sort_values("log_ratio", ascending=False)[cols].head(10).to_string(index=False))

    print("\n--- Datasets where foundation wins most (top 10) ---")
    print(h2h.sort_values("log_ratio")[cols].head(10).to_string(index=False))

    print("\n--- Correlation: dataset features vs statistical advantage ---")
    for feat in ["log_freq_hours", "horizon_ord", "log_variates"]:
        sub = h2h[[feat, "log_ratio"]].dropna()
        r, p = stats.pearsonr(sub[feat], sub["log_ratio"])
        print(f"  {feat:25s}: r={r:+.3f}  p={p:.3f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GIFT-Eval benchmark analysis")
    parser.add_argument("--out-dir", default="plots/gift", help="Output directory for plots")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("Loading data from HuggingFace…")
    raw = load_all()
    df = build_dataframe(raw)

    print("\nBuilding head-to-head comparison…")
    h2h = build_head2head(df)

    print_summary(h2h)

    print("\nGenerating plots…")
    plot_overall_ranking(df, out)
    plot_mase_by_domain(df, out)
    plot_waterfall(h2h, out)
    plot_scatter_drivers(h2h, out)
    plot_boxplots(h2h, out)
    plot_freq_stacked_bar(h2h, out)
    plot_rank_heatmap(df, out)
    plot_clustermap(df, out)
    plot_foundation_spread(h2h, out)

    print(f"\nDone. All plots saved to {out}/")


if __name__ == "__main__":
    main()
