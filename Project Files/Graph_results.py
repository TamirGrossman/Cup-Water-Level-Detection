import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import Optional

def plot_anova_f_scores(
    scores_df: pd.DataFrame,
    top_n: Optional[int] = None,
    figsize: tuple = (10, 6),
    show_p_values: bool = True,
    p_value_fmt: str = "{:.2e}",
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


def graph_CM(cm, class_names):
    # graph confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix - KFold', fontsize=14)
    plt.tight_layout()
    plt.show()

# Example usage:
# scores = anova_f_scores(X, y)
# fig = plot_anova_f_scores(scores, top_n=20, f_threshold=4.0)
# fig.show()  # or fig.savefig("anova_f_scores.png")

