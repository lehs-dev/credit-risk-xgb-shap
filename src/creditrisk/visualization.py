"""Artifact-only Chapter 3 figures; no model fitting or held-out inference.

Plot functions return matplotlib Figures for reuse in notebooks. Error bars are
recorded fold standard deviations, not standard errors or confidence intervals.
ROC/PR curves require individual scores and cannot be inferred from AUC/AP.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

CANDIDATES = ("xgb_default", "tpe_best", "pareto_auc", "pareto_balanced", "pareto_stability")
LABELS = {
    "xgb_default": "Default XGB", "tpe_best": "TPE best", "pareto_auc": "Pareto AUC",
    "pareto_balanced": "Pareto balanced", "pareto_stability": "Pareto stability",
    "catboost": "CatBoost", "random_forest": "Random forest", "gradient_boosting": "GBM",
    "lightgbm": "LightGBM", "xgboost_default": "Default XGB", "logistic_regression": "Logistic regression",
}
COLORS = dict(zip(CANDIDATES, ("#777777", "#D55E00", "#0072B2", "#009E73", "#CC79A7")))
PARAMETER_BOUNDS = {
    "max_depth": (2, 10, False), "learning_rate": (0.01, 0.30, True),
    "n_estimators": (100, 1000, False), "subsample": (0.5, 1.0, False),
    "colsample_bytree": (0.5, 1.0, False), "min_child_weight": (1, 15, False),
    "gamma": (0.0, 10.0, False), "reg_alpha": (0.0001, 10.0, True),
    "reg_lambda": (0.001, 100.0, True),
}
ROLE_TO_CANDIDATE = {
    "default": "xgb_default", "auc_best_tpe": "tpe_best", "auc": "pareto_auc",
    "balanced": "pareto_balanced", "stability": "pareto_stability",
}
FIGURE_STEMS = (
    "fig01_baseline_benchmark", "fig02_hpo_objective_space_pareto",
    "fig03_hyperparameter_tradeoff", "fig04_global_shap_importance",
    "fig05_sensitivity_analysis_boxplots", "fig06_final_test_performance",
    "fig07_summary_tradeoff_landscape", "fig08_shap_rank_variation",
)


def set_scientific_theme() -> None:
    """Use an available font and editable vector text; safe in headless kernels."""
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 10, "legend.fontsize": 9, "axes.spines.top": False,
        "axes.spines.right": False, "axes.axisbelow": True, "grid.alpha": 0.2,
        "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _require(table: pd.DataFrame, columns: list[str]) -> None:
    missing = set(columns) - set(table.columns)
    if missing:
        raise ValueError(f"Artifact is missing columns: {sorted(missing)}")
    if table.empty:
        raise ValueError("Artifact has no observations.")


def _finite(table: pd.DataFrame, columns: list[str]) -> None:
    _require(table, columns)
    if not np.isfinite(table[columns].to_numpy(dtype=float)).all():
        raise ValueError(f"Non-finite values in artifact columns: {columns}")


def _row(value: pd.DataFrame | pd.Series | Mapping) -> pd.Series:
    if isinstance(value, pd.DataFrame):
        if len(value) != 1:
            raise ValueError("Expected exactly one configuration row.")
        return value.iloc[0]
    return pd.Series(value)


def _finish(fig: Figure, title: str, note: str = "") -> Figure:
    fig.suptitle(title, fontsize=15, fontweight="bold", y=0.99)
    if note:
        fig.text(0.02, 0.015, note, fontsize=9, color="#444444", va="bottom")
    fig.subplots_adjust(top=0.87, bottom=0.20, wspace=0.30)
    return fig


def normalize_hyperparameters(
    table: pd.DataFrame,
    bounds: Mapping[str, tuple[float, float, bool]] = PARAMETER_BOUNDS,
) -> pd.DataFrame:
    """Scale each parameter using the locked search range, with log scales stated.

    Equal values across candidates stay equal; normalization never uses just the
    selected configurations, which would exaggerate small differences.
    """
    result = pd.DataFrame(index=table.index)
    for name, (low, high, logarithmic) in bounds.items():
        column = f"param_{name}"
        _finite(table, [column])
        values = table[column].astype(float)
        if low >= high or (logarithmic and low <= 0):
            raise ValueError(f"Invalid search bounds for {name}.")
        if ((values < low - 1e-12) | (values > high + 1e-12)).any():
            raise ValueError(f"{name} is outside its locked search range.")
        if logarithmic:
            values, low, high = np.log10(values), np.log10(low), np.log10(high)
        result[name] = (values - low) / (high - low)
    return result


def summarize_shap_importance(importance_by_fold: pd.DataFrame) -> pd.DataFrame:
    """Summarize observed global mean(|SHAP|), retaining population fold SD."""
    importance_by_fold = _clean_shap_columns(importance_by_fold)
    _require(importance_by_fold, ["fold"])
    if importance_by_fold["fold"].duplicated().any():
        raise ValueError("SHAP artifact contains duplicate folds.")
    features = [column for column in importance_by_fold if column != "fold"]
    if not features:
        raise ValueError("SHAP artifact has no feature columns.")
    _finite(importance_by_fold, features)
    if (importance_by_fold[features].to_numpy() < 0).any():
        raise ValueError("Global absolute SHAP importance must be non-negative.")
    return pd.DataFrame({
        "feature": features, "mean": importance_by_fold[features].mean().to_numpy(),
        "std": importance_by_fold[features].std(ddof=0).to_numpy(),
    }).sort_values("mean", ascending=False, kind="stable").reset_index(drop=True)


def shap_rank_matrix(importance_by_fold: pd.DataFrame) -> pd.DataFrame:
    """Rank within each fold using average ranks for ties, matching Spearman."""
    importance_by_fold = _clean_shap_columns(importance_by_fold)
    ordered = summarize_shap_importance(importance_by_fold)["feature"].tolist()
    values = importance_by_fold.set_index("fold")[ordered]
    return values.rank(axis=1, ascending=False, method="average").T


def _clean_shap_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Tolerate CSV column padding without changing the saved source artifact."""
    normalized = table.rename(columns=lambda name: name.strip() if isinstance(name, str) else name)
    if normalized.columns.duplicated().any():
        raise ValueError("SHAP feature names must remain unique after trimming whitespace.")
    return normalized


def paired_metric_table(repeats: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Return one complete, aligned candidate-by-seed table; reject false pairing."""
    _finite(repeats, [metric])
    _require(repeats, ["candidate", "seed"])
    if repeats.duplicated(["candidate", "seed"]).any():
        raise ValueError("Duplicate candidate/seed results cannot be paired.")
    wide = repeats.pivot(index="seed", columns="candidate", values=metric).sort_index()
    if wide.isna().any().any():
        raise ValueError("Candidates do not share a complete set of paired seeds.")
    order = [candidate for candidate in CANDIDATES if candidate in wide.columns]
    order += [candidate for candidate in wide.columns if candidate not in order]
    return wide[order]


def confusion_matrix_from_row(row: pd.Series | Mapping) -> np.ndarray:
    """Read persisted counts in sklearn order: [[TN, FP], [FN, TP]]."""
    values = np.asarray([row[name] for name in ("tn", "fp", "fn", "tp")], dtype=float)
    if not np.isfinite(values).all() or (values < 0).any() or (values != np.floor(values)).any():
        raise ValueError("Confusion counts must be finite, non-negative integers.")
    matrix = values.astype(np.int64).reshape(2, 2)
    if "n_test" in row and int(row["n_test"]) != int(matrix.sum()):
        raise ValueError("Confusion counts do not sum to n_test.")
    for name, total in (("n_negative", matrix[0].sum()), ("n_positive", matrix[1].sum())):
        if name in row and int(row[name]) != int(total):
            raise ValueError(f"Confusion counts disagree with {name}.")
    return matrix


def prepare_tradeoff_landscape(comparison: pd.DataFrame, final_summary: pd.DataFrame) -> pd.DataFrame:
    """Join locked Development stability to Test performance, never Test stability."""
    _finite(comparison, ["roc_auc_cv", "shap_stability", "top_5_jaccard"])
    _require(comparison, ["selection_role"])
    _finite(final_summary, ["roc_auc"])
    _require(final_summary, ["candidate"])
    development = comparison.copy()
    development["candidate"] = development["selection_role"].map(ROLE_TO_CANDIDATE)
    if development["candidate"].isna().any():
        raise ValueError("Unknown Development selection role.")
    merged = development.merge(
        final_summary[["candidate", "roc_auc"]].rename(columns={"roc_auc": "roc_auc_test"}),
        on="candidate", how="outer", validate="one_to_one", indicator=True,
    )
    if (merged["_merge"] != "both").any():
        raise ValueError("Development and Test configurations do not match.")
    return merged.drop(columns="_merge").set_index("candidate").loc[list(CANDIDATES)].reset_index()


def plot_baseline_benchmark(table: pd.DataFrame) -> Figure:
    metrics = (("roc_auc", "ROC-AUC ↑", 0.9), ("average_precision", "Average precision ↑", 0.65),
               ("brier_score", "Brier score ↓", 0.17))
    _require(table, ["model"])
    _finite(table, [f"{metric}_{suffix}" for metric, _, _ in metrics for suffix in ("mean", "std")])
    if (table[[f"{metric}_std" for metric, _, _ in metrics]] < 0).any().any():
        raise ValueError("Fold standard deviations cannot be negative.")
    table = table.sort_values("roc_auc_mean", ascending=False)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, (metric, label, maximum) in zip(axes, metrics):
        bars = ax.bar(np.arange(len(table)), table[f"{metric}_mean"],
                      yerr=table[f"{metric}_std"], capsize=4, color="#0072B2", alpha=0.85)
        ax.set_xticks(np.arange(len(table)), [LABELS.get(name, name) for name in table["model"]],
                      rotation=38, ha="right", fontsize=9)
        ax.set_ylim(0, maximum)
        ax.set_title(label)
        ax.set_ylabel("Core CV score")
        ax.grid(axis="y")
        for bar, value, deviation in zip(bars, table[f"{metric}_mean"], table[f"{metric}_std"]):
            ax.text(bar.get_x() + bar.get_width()/2, value + deviation + maximum*0.015,
                    f"{value:.3f}", ha="center", fontsize=8)
    _finish(fig, "Baseline benchmark on Development / Core CV",
            "Recorded five-fold mean ± fold SD (not a confidence interval). All bar axes start at zero.")
    fig.subplots_adjust(bottom=0.28)
    return fig


def plot_objective_space_pareto(trials: pd.DataFrame, pareto: pd.DataFrame,
                               selected: pd.DataFrame, single_best, default) -> Figure:
    if "state" in trials:
        trials = trials.loc[trials["state"] == "COMPLETE"]
    for table in (trials, pareto, selected):
        _finite(table, ["roc_auc_cv", "shap_stability"])
    single_best, default = _row(single_best), _row(default)
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.scatter(trials["roc_auc_cv"], trials["shap_stability"], color="#888888", alpha=0.5,
               s=35, label=f"NSGA-II completed trials (n={len(trials)})")
    front = pareto.sort_values("roc_auc_cv")
    ax.plot(front["roc_auc_cv"], front["shap_stability"], "o--", color="#222222", linewidth=1.3,
            markersize=5, label=f"Non-dominated trials (n={len(front)})")
    ax.scatter(single_best["roc_auc_cv"], single_best["shap_stability"], marker="D",
               s=85, color=COLORS["tpe_best"], label="TPE best")
    ax.scatter(default["roc_auc_cv"], default["shap_stability"], marker="s", s=85,
               color=COLORS["xgb_default"], label="Default XGB")
    offsets = {"auc": (-16, -26), "balanced": (-95, 14), "stability": (10, 10)}
    for _, row in selected.iterrows():
        candidate = ROLE_TO_CANDIDATE[row["selection_role"]]
        ax.scatter(row["roc_auc_cv"], row["shap_stability"], color=COLORS[candidate],
                   edgecolor="white", linewidth=1.0, marker="*", s=220, zorder=5,
                   label=LABELS[candidate])
        ax.annotate(f"trial {int(row['trial'])}", (row["roc_auc_cv"], row["shap_stability"]),
                    xytext=offsets[row["selection_role"]], textcoords="offset points", fontsize=9,
                    arrowprops={"arrowstyle": "-", "color": COLORS[candidate]})
    ax.set_xlabel("ROC-AUC on five-fold Core CV ↑")
    ax.set_ylabel("SHAP rank stability (Spearman) on fixed Reference ↑")
    ax.grid()
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    inset = ax.inset_axes([0.035, 0.63, 0.22, 0.29])
    inset.scatter(trials["roc_auc_cv"], trials["shap_stability"], s=3, color="#888888")
    inset.scatter([1], [1], marker="*", s=60, color="#E69F00", clip_on=False)
    inset.annotate("Ideal (1, 1)", (1, 1), xytext=(-72, -18), textcoords="offset points", fontsize=8)
    inset.set(xlim=(0, 1.05), ylim=(0, 1.05), title="Full objective scale")
    inset.tick_params(labelsize=7)
    _finish(fig, "HPO objective space and audited Pareto front",
            "Dashed segments connect observed front points only; intermediate configurations were not evaluated. Ideal point shown in inset.")
    fig.subplots_adjust(bottom=0.15)
    return fig


def plot_hyperparameter_tradeoff(single_best, selected: pd.DataFrame,
                                trials: pd.DataFrame | None = None) -> Figure:
    """Compare four locked configurations; optional trials retained for API reuse."""
    single = _row(single_best).to_dict()
    rows = [{"candidate": "tpe_best", **single}]
    for _, row in selected.iterrows():
        rows.append({"candidate": ROLE_TO_CANDIDATE[row["selection_role"]], **row.to_dict()})
    table = pd.DataFrame(rows).set_index("candidate")
    scaled = normalize_hyperparameters(table)
    fig, ax = plt.subplots(figsize=(13, 5.5))
    x = np.arange(len(scaled.columns))
    for candidate, values in scaled.iterrows():
        ax.plot(x, values, "o-", linewidth=2, markersize=5,
                color=COLORS[candidate], label=LABELS[candidate])
    ticks = [name.replace("_", "\n") + ("\n(log)" if PARAMETER_BOUNDS[name][2] else "")
             for name in scaled.columns]
    ax.set_xticks(x, ticks)
    ax.set_ylim(-0.04, 1.04)
    ax.set_ylabel("Position within locked search range [0, 1]")
    ax.grid()
    ax.legend(loc="upper center", ncol=4)
    _finish(fig, "Hyperparameter profiles of locked HPO representatives",
            "Each axis uses the full pre-specified search range; learning_rate, reg_alpha and reg_lambda use log10 scaling.\n"
            "Profiles describe configurations; adaptive trial associations do not establish causal parameter effects.")
    fig.subplots_adjust(bottom=0.27)
    return fig


def plot_global_shap_importance(importance_by_fold: pd.DataFrame) -> Figure:
    importance_by_fold = _clean_shap_columns(importance_by_fold)
    summary = summarize_shap_importance(importance_by_fold).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 9))
    y = np.arange(len(summary))
    ax.barh(y, summary["mean"], xerr=summary["std"], capsize=3, color="#0072B2", alpha=0.75,
            label="Five-fold mean ± SD")
    for offset, (_, fold) in zip(np.linspace(-0.17, 0.17, len(importance_by_fold)), importance_by_fold.iterrows()):
        ax.scatter(fold[summary["feature"]].to_numpy(dtype=float), y + offset,
                   s=12, color="#222222", alpha=0.5)
    ax.set_yticks(y, summary["feature"])
    ax.set_xlabel("Global mean(|SHAP|), raw log-odds margin")
    ax.grid(axis="x")
    ax.legend(loc="lower right")
    _finish(fig, "Default XGBoost: global SHAP importance on fixed Reference",
            "Only the saved Default XGB artifact is available: 1,000 Reference observations; five Core-fold models.\n"
            "Dots are observed fold importances; categorical one-hot SHAP values were grouped into original features. No Test SHAP was computed.")
    fig.subplots_adjust(left=0.17, top=0.90, bottom=0.13)
    return fig


def plot_shap_rank_variation(importance_by_fold: pd.DataFrame) -> Figure:
    ranks = shap_rank_matrix(importance_by_fold)
    fig, ax = plt.subplots(figsize=(8, 9))
    img = ax.imshow(ranks, aspect="auto", cmap="viridis_r", vmin=1, vmax=len(ranks))
    ax.set_xticks(np.arange(len(ranks.columns)), [f"Fold {fold}" for fold in ranks.columns])
    ax.set_yticks(np.arange(len(ranks)), ranks.index)
    for i in range(len(ranks)):
        for j in range(len(ranks.columns)):
            value = ranks.iloc[i, j]
            ax.text(j, i, f"{value:g}", ha="center", va="center", fontsize=8,
                    color="white" if value > len(ranks)*0.55 else "black")
    fig.colorbar(img, ax=ax, label="Within-fold feature rank (1 = most important)", pad=0.03)
    _finish(fig, "Default XGBoost: SHAP feature ranks across Core folds",
            "Ranks use saved global Reference importance; average ranks handle ties. Rows are ordered by mean importance, not by Test outcomes.")
    fig.subplots_adjust(left=0.20, top=0.91, bottom=0.09)
    return fig


def plot_sensitivity_analysis(repeats: pd.DataFrame) -> Figure:
    metrics = (("roc_auc_cv", "Core CV ROC-AUC ↑"), ("shap_stability", "Reference SHAP Spearman ↑"),
               ("top_5_jaccard", "Top-5 SHAP Jaccard ↑"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
    for ax, (metric, title) in zip(axes, metrics):
        wide = paired_metric_table(repeats, metric)
        x = np.arange(1, len(wide.columns) + 1)
        boxes = ax.boxplot([wide[c].to_numpy() for c in wide], positions=x,
                           patch_artist=True, widths=0.55, showfliers=False,
                           medianprops={"color": "black", "linewidth": 1.5})
        for box, candidate in zip(boxes["boxes"], wide.columns):
            box.set(facecolor=COLORS.get(candidate, "#56B4E9"), alpha=0.35)
        for _, row in wide.iterrows():
            ax.plot(x, row.to_numpy(), color="#555555", linewidth=0.6, alpha=0.22, zorder=1)
        for i, candidate in enumerate(wide.columns):
            ax.scatter(np.full(len(wide), x[i]), wide[candidate], s=18,
                       color=COLORS.get(candidate, "#56B4E9"), zorder=3)
        ax.set_xticks(x, [LABELS.get(c, c).replace(" ", "\n", 1) for c in wide.columns], fontsize=8)
        ax.set_title(title)
        ax.yaxis.set_major_locator(MaxNLocator(6))
        ax.grid(axis="y")
    _finish(fig, "Sensitivity to paired Core CV partitions",
            f"{repeats['seed'].nunique()} shared seeds; lines connect results from the same partition. Boxes show median/IQR; whiskers use 1.5 × IQR.\n"
            "The fixed 1,000-row Reference set is reused; partitions overlap and are not independent new datasets.")
    fig.subplots_adjust(bottom=0.24)
    return fig


def plot_final_test_performance(summary: pd.DataFrame) -> Figure:
    _require(summary, ["candidate", "tn", "fp", "fn", "tp", "threshold"])
    _finite(summary, ["roc_auc", "average_precision"])
    if summary["candidate"].duplicated().any():
        raise ValueError("Final Test artifact repeats a candidate.")
    table = summary.set_index("candidate").loc[list(CANDIDATES)]
    matrices = [confusion_matrix_from_row(row) for _, row in table.iterrows()]
    fig = plt.figure(figsize=(15, 8))
    grid = fig.add_gridspec(2, 10, height_ratios=(1.1, 1.0), hspace=0.7, wspace=1.2)
    top_axes = [fig.add_subplot(grid[0, :5]), fig.add_subplot(grid[0, 5:])]
    for ax, metric, title in zip(top_axes, ("roc_auc", "average_precision"),
                                 ("Held-out ROC-AUC", "Held-out average precision")):
        ax.bar(np.arange(len(table)), table[metric], color=[COLORS[c] for c in table.index], alpha=0.85)
        for x, value in enumerate(table[metric]):
            ax.text(x, value + 0.02, f"{value:.4f}", ha="center", fontsize=9)
        ax.set_xticks(np.arange(len(table)), [LABELS[c].replace(" ", "\n", 1) for c in table.index], fontsize=9)
        ax.set_ylim(0, 1)
        ax.set_title(title)
        ax.grid(axis="y")
    maximum = max(int(matrix.max()) for matrix in matrices)
    for i, ((candidate, row), matrix) in enumerate(zip(table.iterrows(), matrices)):
        ax = fig.add_subplot(grid[1, 2*i:2*i+2])
        ax.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for (r, c), count in np.ndenumerate(matrix):
            ax.text(c, r, f"{count:,}", ha="center", va="center", fontsize=11,
                    color="white" if count > maximum*0.55 else "black")
        ax.set_xticks([0, 1], ["Non-default", "Default"], rotation=28, ha="right", fontsize=8)
        ax.set_yticks([0, 1], ["Non-default", "Default"], fontsize=8)
        ax.set_xlabel("Predicted")
        if i == 0:
            ax.set_ylabel("Actual")
        ax.set_title(f"{LABELS[candidate]}\nthreshold = {row['threshold']:g}", fontsize=10)
    _finish(fig, "Final Test: persisted performance and confusion counts",
            "ROC/PR curves unavailable: individual held-out probabilities were not saved; aggregate AUC/AP cannot reconstruct curves.\n"
            "Matrices contain actual saved counts from the single locked evaluation (n = 6,000; 1,327 defaults). No refit or Test re-evaluation.")
    fig.subplots_adjust(left=0.06, right=0.98, top=0.88, bottom=0.17, wspace=1.2)
    return fig


def plot_summary_tradeoff_landscape(comparison: pd.DataFrame, final_summary: pd.DataFrame) -> Figure:
    table = prepare_tradeoff_landscape(comparison, final_summary)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for _, row in table.iterrows():
        candidate, color = row["candidate"], COLORS[row["candidate"]]
        axes[0].annotate("", (row["roc_auc_test"], row["shap_stability"]),
                         (row["roc_auc_cv"], row["shap_stability"]),
                         arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.5})
        axes[0].scatter(row["roc_auc_cv"], row["shap_stability"], s=55, color=color)
        axes[0].scatter(row["roc_auc_test"], row["shap_stability"], s=55,
                        facecolor="white", edgecolor=color, linewidth=1.8)
        axes[0].annotate(LABELS[candidate], (row["roc_auc_cv"], row["shap_stability"]),
                         xytext=(5, 7), textcoords="offset points", fontsize=8, color=color)
        axes[1].scatter(row["shap_stability"], row["top_5_jaccard"], s=80, color=color)
        axes[1].annotate(LABELS[candidate], (row["shap_stability"], row["top_5_jaccard"]),
                         xytext=(5, 8), textcoords="offset points", fontsize=8, color=color)
    axes[0].set(xlabel="ROC-AUC ↑", ylabel="Locked Development Reference Spearman ↑",
                title="CV → Test AUC with Development stability retained")
    axes[0].legend(handles=[Line2D([0], [0], marker="o", color="black", linestyle="", label="Core CV"),
                            Line2D([0], [0], marker="o", color="black", markerfacecolor="white",
                                   linestyle="", label="Final Test AUC")], loc="lower right")
    axes[0].margins(x=0.2, y=0.15)
    axes[1].set(xlabel="Reference SHAP Spearman ↑", ylabel="Reference top-5 SHAP Jaccard ↑",
                title="Whole-ranking and top-5 stability are distinct")
    axes[1].margins(x=0.27, y=0.2)
    for ax in axes:
        ax.grid()
    _finish(fig, "Performance–explanation trade-off across locked configurations",
            "Horizontal arrows change only the AUC measurement (Core CV to held-out Test); the y coordinate remains Development Reference stability.\n"
            "Test SHAP stability was not measured. High whole-ranking Spearman does not imply high top-5 overlap.")
    fig.subplots_adjust(bottom=0.20)
    return fig


def save_figure(fig: Figure, output_stem: Path | str) -> list[Path]:
    """Write matching 300-DPI PNG and vector PDF; caller owns figure lifetime."""
    stem = Path(output_stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ("png", "pdf"):
        path = stem.with_suffix(f".{extension}")
        fig.savefig(path, dpi=300, facecolor="white", bbox_inches="tight")
        outputs.append(path)
    return outputs


def generate_chapter3_figures(root: Path | str, output_dir: Path | str | None = None) -> dict:
    """Generate all figures from committed summaries and available Reference SHAP.

    This function only reads artifacts; it never imports training/evaluation code
    or loads raw/Core/Test observations. Missing SHAP is visibly disclosed.
    """
    root = Path(root).resolve()
    output = Path(output_dir) if output_dir is not None else root / "artifacts/figures"
    output.mkdir(parents=True, exist_ok=True)
    set_scientific_theme()
    tables = root / "artifacts/tables"
    input_files = []

    def read(name: str) -> pd.DataFrame:
        path = tables / name
        input_files.append(path)
        return pd.read_csv(path)

    baseline = read("baseline_benchmark.csv")
    trials = read("xgb_multi_hpo_trials.csv")
    pareto = read("xgb_multi_hpo_pareto.csv")
    selected = read("xgb_pareto_selected.csv")
    single = read("xgb_single_hpo_best.csv")
    default = read("xgb_default_objectives.csv")
    repeats = read("xgb_sensitivity_repeats.csv")
    final = read("xgb_final_test_summary.csv")
    comparison = read("xgb_pareto_comparison.csv")
    shap_path = root / "artifacts/shap/xgb_default_importance_by_fold.csv"
    shap = pd.read_csv(shap_path) if shap_path.is_file() else None
    if shap is not None:
        input_files.append(shap_path)

    def missing_shap() -> Figure:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.axis("off")
        ax.text(0.5, 0.55, "Reference SHAP artifact unavailable", fontsize=18, ha="center", transform=ax.transAxes)
        ax.text(0.5, 0.4, "Expected: artifacts/shap/xgb_default_importance_by_fold.csv\n"
                "No importance or rank values can be recovered from aggregate stability.\n"
                "No models were refitted to fill this gap.", ha="center", transform=ax.transAxes)
        return _finish(fig, "Saved SHAP data required for this figure")

    plotters = (
        lambda: plot_baseline_benchmark(baseline),
        lambda: plot_objective_space_pareto(trials, pareto, selected, single, default),
        lambda: plot_hyperparameter_tradeoff(single, selected),
        lambda: plot_global_shap_importance(shap) if shap is not None else missing_shap(),
        lambda: plot_sensitivity_analysis(repeats),
        lambda: plot_final_test_performance(final),
        lambda: plot_summary_tradeoff_landscape(comparison, final),
        lambda: plot_shap_rank_variation(shap) if shap is not None else missing_shap(),
    )
    outputs = []
    for stem, plotter in zip(FIGURE_STEMS, plotters):
        fig = plotter()
        try:
            for path in save_figure(fig, output / stem):
                outputs.append({"file": path.name, "bytes": path.stat().st_size,
                                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        finally:
            plt.close(fig)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "artifact_only_no_fitting_no_test_inference", "png_dpi": 300,
        "pdf_vector_text": True, "test_re_evaluated": False,
        "sources": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in input_files},
        "figures": outputs,
        "limitations": {
            "baseline_roc_pr": "Unavailable: per-fold probabilities were not saved. Means/SD are shown instead.",
            "final_test_roc_pr": "Unavailable: held-out probabilities were not saved. AUC/AP and actual confusion counts are shown instead.",
            "reference_shap": "Default XGBoost only; five Core-fold models evaluated on the fixed 1,000-row Reference set."
                              if shap is not None else "Unavailable: saved Default XGBoost Reference importance artifact missing.",
            "final_test_shap": "Not computed; Figure 07 retains Development Reference stability on its y axis.",
            "hyperparameters": "Locked search-range normalization; parameter associations are exploratory, not causal effects.",
            "uncertainty": "Error bars use recorded fold SD. Overlapping CV partitions do not constitute independent datasets.",
        },
        "availability": {
            "fig04_global_shap_importance": "observed_reference_importance" if shap is not None else "missing_artifact_notice",
            "fig06_final_test_performance": "aggregate_metrics_and_saved_confusion_counts_no_curves",
            "fig08_shap_rank_variation": "observed_reference_feature_ranks" if shap is not None else "missing_artifact_notice",
        },
    }
    (output / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest
