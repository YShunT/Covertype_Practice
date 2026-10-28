import json
import logging

import numpy as np
from sklearn.metrics import classification_report

from config import OUTPUT_DIR, CLASS_NAMES
from data import load_data
from evaluate import (
    compute_metrics,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_learning_curve,
)
from train import crossvalidation

logger = logging.getLogger(__name__)

def setup_logger(log_path):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
        ],
    )

def main():
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_logger(out_dir / ".log")

    X_train, X_test, y_train, y_test = load_data()
    logger.info(f"train_size: {X_train.shape}, test_size: {X_test.shape}")

    oof_prob, test_prob, models, fold_scores = crossvalidation(X_train, y_train, X_test)

    cv_metrics = compute_metrics(y_train, oof_prob)
    test_metrics = compute_metrics(y_test, test_prob)
    logger.info(f"classification report\n{classification_report(y_test, test_prob.argmax(axis=1), target_names=CLASS_NAMES, digits=4)}")

    metrics = {
        "fold_accuracy": [float(s) for s in fold_scores],
        "fold_accuracy_mean": float(np.mean(fold_scores)),
        "fold_accuracy_std": float(np.std(fold_scores)),
        "cv_oof": {k: float(v) for k, v in cv_metrics.items()},
        "test": {k: float(v) for k, v in test_metrics.items()},
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    plot_confusion_matrix(y_test, test_prob, out_dir / "confusion_matrix_test.png")
    plot_feature_importance(models, X_train.columns, out_dir / "feature_importance.png")
    plot_learning_curve(models, out_dir / "learning_curve.png")

    print("Done!!!")

if __name__ == "__main__":
    main()
