import pandas as pd
from sklearn.model_selection import train_test_split

from config import COLUMNS, DATA_PATH, SEED, TEST_SIZE


def load_data():
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
    df["Cover_Type"] = df["Cover_Type"] - 1 # {1...7} ->{0...6}へ
    X = df.drop(columns="Cover_Type")
    y = df["Cover_Type"]

    X_train,X_test,y_train,y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    return X_train,X_test,y_train,y_test
