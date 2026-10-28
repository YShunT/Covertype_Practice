from lightgbm import LGBMClassifier

from config import LGBM_PARAMS


def train_model(X_tr, y_tr, X_val, y_val):
    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_X=(X_tr, X_val),
        eval_y=(y_tr, y_val),
        eval_names=["train", "valid"],
        eval_metric="multi_error",
    )
    return model
