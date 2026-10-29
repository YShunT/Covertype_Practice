from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "covertype" / "covtype.data.gz"
OUTPUT_DIR = ROOT / "outputs" / "LGBM_RegES"

SEED = 42
N_SPLITS = 5
TEST_SIZE = 0.2
N_CLASSES = 7

NUMERIC_COLS = [
    "Elevation",
    "Aspect",
    "Slope",
    "Horizontal_Distance_To_Hydrology",
    "Vertical_Distance_To_Hydrology",
    "Horizontal_Distance_To_Roadways",
    "Hillshade_9am",
    "Hillshade_Noon",
    "Hillshade_3pm",
    "Horizontal_Distance_To_Fire_Points",
]

WILDERNESS_COLS = [f"Wilderness_Area_{i}" for i in range(1, 5)]
SOIL_COLS = [f"Soil_Type_{i}" for i in range(1, 41)]
TAR_COLS = ["Cover_Type"]
COLUMNS = NUMERIC_COLS + WILDERNESS_COLS + SOIL_COLS + TAR_COLS
CLASS_NAMES = [
    "Spruce/Fir",
    "Lodgepole Pine",
    "Ponderosa Pine",
    "Cottonwood/Willow",
    "Aspen",
    "Douglas-fir",
    "Krummholz",
]

LGBM_PARAMS = dict(
    objective="multiclass",
    num_class=N_CLASSES,
    learning_rate=0.1,
    num_leaves=31,
    n_estimators=5000,  
    min_child_weight=1,
    reg_lambda=1,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1
)
EARLY_STOPPING_ROUNDS = 100