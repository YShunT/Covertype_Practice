import warnings

import lightgbm as lgb
from lightgbm import LGBMClassifier

from config import EARLY_STOPPING_ROUNDS, LGBM_PARAMS

# lightgbm 4.7 の LGBMClassifier は eval_X/eval_y で渡した学習データを学習データと認識できず、
# early stopping の判定に train が混ざる。そのため非推奨の eval_set を使い、その警告だけ消す
warnings.filterwarnings("ignore", message="The argument 'eval_set' is deprecated")


def train_model(X_tr, y_tr, X_val, y_val):
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_tr, y_tr), (X_val, y_val)],
        eval_names=["train", "valid"],
        eval_metric="multi_error",
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, first_metric_only=True, verbose=False)],
    )
    return model
