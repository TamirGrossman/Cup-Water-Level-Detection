from matplotlib.patches import Patch
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

from Model_Validation import kfold_cv_multiple

def plot_anova_f_scores(
    scores_df: pd.DataFrame,
    top_n: Optional[int] = None,
    figsize: tuple = (10, 6),
    f_threshold: Optional[float] = None,
    palette: str = "viridis",
):
    """
    Plot ANOVA F-scores from a DataFrame produced by `anova_f_scores`.
    Expects columns: 'feature', 'F', 'p'.

    Parameters:
      - scores_df: DataFrame with columns 'feature', 'F', 'p'.
      - top_n: if set, plot only the top_n features by F (descending).
      - figsize: figure size.
      - show_p_values: annotate bars with p-values.
      - p_value_fmt: format string for p-value annotation.
      - f_threshold: optional horizontal line at this F value (e.g., significance cutoff).
      - palette: seaborn color palette name.

    Returns:
      - matplotlib Figure object (caller can save or show it).
    """
    if not {"feature", "F"}.issubset(scores_df.columns):
        raise ValueError("scores_df must contain 'feature', 'F' column")

    df = scores_df.copy().sort_values("F", ascending=False)
    if top_n is not None:
        df = df.head(top_n)

    plt_style = sns.color_palette(palette, n_colors=len(df))
    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.barh(df["feature"][::-1], df["F"][::-1], color=plt_style[::-1])
    for bar, f in zip(bars, df["F"][::-1]):
            width = bar.get_width()
            ax.text(width + max(df["F"].max() * 0.01, 1e-6), bar.get_y() + bar.get_height() / 2,
                    '{:.0f}'.format(f), va="center", fontsize=9)

    ax.set_xlabel("ANOVA F-score")
    ax.set_ylabel("Feature")
    ax.set_title("ANOVA F-scores by Feature")
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    if f_threshold is not None:
        ax.axvline(f_threshold, color="red", linestyle="--", linewidth=1)
        ax.text(f_threshold, -0.4, f"F = {f_threshold}", color="red", va="bottom", ha="right", fontsize=9)

    # if show_p_values:
    #     for bar, p in zip(bars, df["p"][::-1]):
    #         width = bar.get_width()
    #         ax.text(width + max(df["F"].max() * 0.01, 1e-6), bar.get_y() + bar.get_height() / 2,
    #                 p_value_fmt.format(p), va="center", fontsize=9)

    plt.tight_layout()
    plt.show()

def plot_feature_impact(X, y, model, top_features, feature_names):
    # graph the impact of adding feature on model accuracy

    print("\nCalculating feature impact on accuracy (using mean kfold result)")
    print(f"Checking impact of {len(top_features)} ")

    features_idx = []
    k_results = []
    for feat in top_features:
        features_idx.append(feature_names.index(feat))
        X_curr = X.iloc[:, features_idx]
        temp, _ = kfold_cv_multiple(model, X_curr, y, k= 4)
        k_results.append(temp['accuracy']['mean'])
    
    plt.figure()
    plt.plot([i for i in range(len(feature_names))], k_results, linestyle='-')   # line with markers
    #plt.scatter(a, b, color='red')              # optional: scatter points
    plt.xlabel('number of features')
    plt.ylabel('accuracy')
    plt.title('Plot of mean accuracy vs number of features')
    plt.grid(True)
    plt.show()


def graph_flow_rate(flow_results: dict, figsize: tuple = (13, 5)):
    """
    Plot flow-rate results from `compute_flow_rates`, two views side by side:

      (1) Grouped bars: mean true vs predicted flow rate (mL/s) per fill class
          (20%, 40%, ...), pooled across all cups.
      (2) Scatter: predicted vs true flow rate per test sample, with a y = x
          reference line (points on the line = perfect agreement).

    Parameters:
      - flow_results: dict returned by compute_flow_rates, expected to contain
        'per_class' (DataFrame with columns 'fill_pct', 'true_flow_rate_mL_s',
        'pred_flow_rate_mL_s') and 'per_sample' (DataFrame with per-row
        'true_flow_rate_mL_s' and 'pred_flow_rate_mL_s').
    """
    per_class = flow_results["per_class"]
    per_sample = flow_results["per_sample"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # (1) Grouped bars: true vs predicted by fill class
    classes = per_class["fill_pct"].tolist()
    x = range(len(classes))
    width = 0.38
    ax1.bar([i - width / 2 for i in x], per_class["true_flow_rate_mL_s"],
            width=width, label="True", color="#4C72B0")
    ax1.bar([i + width / 2 for i in x], per_class["pred_flow_rate_mL_s"],
            width=width, label="Predicted", color="#DD8452")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels([f"{c:.0f}%" for c in classes])
    ax1.set_xlabel("Fill class")
    ax1.set_ylabel("Mean flow rate (mL/s)")
    ax1.set_title("Mean Flow Rate by Fill Class: True vs Predicted")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.4)
    for i, (t, p) in enumerate(zip(per_class["true_flow_rate_mL_s"],
                                   per_class["pred_flow_rate_mL_s"])):
        ax1.text(i - width / 2, t, f"{t:.1f}", ha="center", va="bottom", fontsize=8)
        ax1.text(i + width / 2, p, f"{p:.1f}", ha="center", va="bottom", fontsize=8)

    # (2) Scatter: predicted vs true per sample, with y = x diagonal
    true_v = per_sample["true_flow_rate_mL_s"]
    pred_v = per_sample["pred_flow_rate_mL_s"]
    cups = sorted(per_sample["cup"].unique())
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(vmin=min(cups), vmax=max(cups))
    sc = ax2.scatter(true_v, pred_v, c=per_sample["cup"], cmap=cmap, norm=norm,
                     alpha=0.8, edgecolor="k", linewidth=0.3)
    lo = float(min(true_v.min(), pred_v.min()))
    hi = float(max(true_v.max(), pred_v.max()))
    pad = (hi - lo) * 0.05 if hi > lo else 1.0
    ax2.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
             "r--", linewidth=1, label="y = x (perfect)")
    ax2.set_xlabel("True flow rate (mL/s)")
    ax2.set_ylabel("Predicted flow rate (mL/s)")
    ax2.set_title("Predicted vs True Flow Rate (per sample)")
    ax2.grid(linestyle="--", alpha=0.4)

    # Legend: y = x line plus one swatch per cup (same style as left legend)
    cup_handles = [
        Patch(facecolor=cmap(norm(c)), edgecolor="k", linewidth=0.3,
              label=f"Cup {c}")
        for c in cups
    ]
    handles, _ = ax2.get_legend_handles_labels()
    ax2.legend(handles=handles + cup_handles)

    plt.tight_layout()
    plt.show()


def graph_cv_comparison(
    cv_results_by_model: dict[str, dict],
    figsize: tuple = (13, 5),
    cv_label: str = "Stratified 4-Fold CV",
):
    """
    Compare cross-validation results across models, two views side by side:

      (1) Grouped bars: mean score per metric for each model, with std error
          bars (improvement comparison across models, per metric).
      (2) Per-fold lines: each model's score across the individual folds for a
          chosen metric (default 'accuracy'), showing fold-to-fold variation.

    Parameters:
      - cv_results_by_model: {model_name: cv_results}, where each cv_results is
        the dict returned by cross_validate_model, i.e.
        {metric: {'scores': np.ndarray, 'mean': float, 'std': float}}.
      - cv_label: title prefix (e.g. 'Stratified 4-Fold CV').
    """
    # Drop any models with no CV results (e.g. run_cross_validation=False).
    models = {name: res for name, res in cv_results_by_model.items() if res}
    if not models:
        print("\ngraph_cv_comparison: no CV results to plot.")
        return

    # Metrics common to all models, preserving the order of the first model.
    first = next(iter(models.values()))
    metrics = [m for m in first.keys()
               if all(m in res for res in models.values())]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    model_names = list(models.keys())
    colors = sns.color_palette("Set2", n_colors=len(model_names))

    # (1) Grouped bars: mean +/- std per metric, one group of bars per metric
    n_models = len(model_names)
    group_width = 0.8
    bar_width = group_width / max(n_models, 1)
    x = range(len(metrics))
    for mi, name in enumerate(model_names):
        means = [models[name][m]["mean"] for m in metrics]
        stds = [models[name][m]["std"] for m in metrics]
        offsets = [i - group_width / 2 + bar_width * (mi + 0.5) for i in x]
        ax1.bar(offsets, means, width=bar_width,
                label=name, color=colors[mi])
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(metrics, rotation=20, ha="right")
    ax1.set_ylabel("Mean CV score")
    ax1.set_ylim(0.9, 1.00)
    ax1.set_title(f"{cv_label}: Mean Score by Metric")
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    # (2) Per-fold lines for a single metric (prefer accuracy)
    fold_metric = "accuracy" if "accuracy" in metrics else metrics[0]
    for mi, name in enumerate(model_names):
        scores = models[name][fold_metric]["scores"]
        folds = range(1, len(scores) + 1)
        ax2.plot(folds, scores, marker="o", label=name, color=colors[mi])
    ax2.set_xlabel("Fold")
    ax2.set_ylabel(f"{fold_metric} score")
    ax2.set_title(f"{cv_label}: Per-Fold {fold_metric}")
    n_folds = len(first[fold_metric]["scores"])
    ax2.set_xticks(range(1, n_folds + 1))
    ax2.legend()
    ax2.grid(linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()


def graph_CM(cm, class_names, model_type):
    # graph confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title(f'{model_type} Confusion Matrix - KFold', fontsize=14)
    plt.tight_layout()
    plt.show()

# Example usage:
# scores = anova_f_scores(X, y)
# fig = plot_anova_f_scores(scores, top_n=20, f_threshold=4.0)
# fig.show()  # or fig.savefig("anova_f_scores.png")

