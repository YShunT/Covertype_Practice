import logging

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold

from config import N_SPLITS, SEED, N_CLASSES
from models import train_model

logger = logging.getLogger(__name__)


def crossvalidation(X_train, y_train, X_test):
    skfold = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    oof_prob = np.zeros((len(X_train), N_CLASSES))
    test_prob = np.zeros((len(X_test), N_CLASSES))
    models = []
    fold_scores = []

    logger.info(f"Start Cross Validation ({N_SPLITS} folds)")
    logger.info(f"oof_prob: {oof_prob.shape}, test_prob: {test_prob.shape}")

    for fold, (train_idx, val_idx) in enumerate(skfold.split(X_train, y_train)):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val, y_val = X_train.iloc[val_idx], y_train.iloc[val_idx]
        logger.info(f"Fold {fold + 1}/{N_SPLITS} - X_tr: {X_tr.shape}, X_val: {X_val.shape}")

        model = train_model(X_tr, y_tr, X_val, y_val)

        val_prob = model.predict_proba(X_val)
        fold_test_prob = model.predict_proba(X_test)
        logger.info(f"Fold {fold + 1}/{N_SPLITS} - val_prob: {val_prob.shape}, fold_test_prob: {fold_test_prob.shape}")

        oof_prob[val_idx] = val_prob
        test_prob += fold_test_prob / N_SPLITS

        score = accuracy_score(y_val, val_prob.argmax(axis=1))
        logger.info(f"Fold {fold + 1}/{N_SPLITS} - Accuracy: {score:.4f} (best_iter={model.best_iteration_})")

        models.append(model)
        fold_scores.append(score)

    return oof_prob, test_prob, models, fold_scores
