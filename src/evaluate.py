import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, f1_score, log_loss,)

from config import CLASS_NAMES, N_CLASSES


def compute_metrics(y_true, prob):
    y_pred = prob.argmax(axis=1)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "logloss": log_loss(y_true, prob, labels=range(N_CLASSES)),
    }


def plot_confusion_matrix(y_true, prob, save_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    ConfusionMatrixDisplay.from_predictions(
        y_true,
        prob.argmax(axis=1),
        display_labels=CLASS_NAMES,
        normalize="true",
        values_format=".2f",
        xticks_rotation=45,
        cmap="Reds",
        ax=ax,
    )
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_feature_importance(models, feature_names, save_path):
    importances = np.array([m.booster_.feature_importance(importance_type="gain") for m in models])
    imp_df = pd.DataFrame({"feature": feature_names, "mean": importances.mean(axis=0)})
    top = imp_df.sort_values("mean", ascending=False).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.barh(top["feature"], top["mean"])
    ax.set_xlabel("gain (mean over folds)")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_learning_curve(models, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric in zip(axes, ["multi_error", "multi_logloss"]):
        for data_name in ["train", "valid"]:
            curves = [m.evals_result_[data_name][metric] for m in models]
            stacked = np.full((len(curves), max(len(c) for c in curves)), np.nan)
            for i, c in enumerate(curves):
                stacked[i, : len(c)] = c
            ax.plot(np.nanmean(stacked, axis=0), label=data_name)
        ax.set_xlabel("iteration")
        ax.set_ylabel(metric)
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)