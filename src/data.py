import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import COLUMNS, DATA_PATH, SEED, TEST_SIZE


def add_features(df):
    # 最寄りの水面の標高
    df["Hydrology_Elevation"] = df["Elevation"] - df["Vertical_Distance_To_Hydrology"]
    # 水面までの直線距離
    df["Distance_To_Hydrology"] = np.sqrt(df["Horizontal_Distance_To_Hydrology"] ** 2 + df["Vertical_Distance_To_Hydrology"] ** 2)
    # 水と火災地点のどちらに近いか
    df["Hydrology_Fire_Diff"] = df["Horizontal_Distance_To_Hydrology"] - df["Horizontal_Distance_To_Fire_Points"]
    return df


def load_data():
    df = pd.read_csv(DATA_PATH, header=None, names=COLUMNS)
    df["Cover_Type"] = df["Cover_Type"] - 1 # {1...7} ->{0...6}へ
    df = add_features(df)
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
