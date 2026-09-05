import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
POSSIBLE_PATHS = [
    os.path.join(BASE_DIR, "Data", "placement_predict_50k Dataset.csv"),
    r"C:\Users\swarn\Documents\sem_4\ML\Placement_Prediction\Data\placement_predict_50k Dataset.csv",
    r"C:\Users\swarn\Documents\sem_4\ML\Programs\placement_predict_50k Dataset.csv",
    r"C:\Users\swarn\Documents\sem_4\ML\Placement_Prediction_predict_50k Dataset.csv"
]

DATA_PATH = next((p for p in POSSIBLE_PATHS if os.path.exists(p)), POSSIBLE_PATHS[0])

def load_data(path: str = None) -> pd.DataFrame:
    if path is None:
        path = DATA_PATH
    if not os.path.exists(path):
        # Retry finding among possible paths
        for p in POSSIBLE_PATHS:
            if os.path.exists(p):
                path = p
                break
        else:
            raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    return df

def get_column_role(col: str) -> str:
    if col == "StudentID":
        return "Identifier"
    elif col in ["PlacementStatus", "Salary Package"]:
        return "Target (Label)"
    else:
        return "Feature"

def get_data_summary(path: str = DATA_PATH) -> dict:
    df = load_data(path)
    summary = {
        "n_rows": df.shape[0],
        "n_cols": df.shape[1],
        "columns": list(df.columns),
        "dtypes": {
            col: str(dtype)
            for col, dtype in df.dtypes.items()
        },
        "missing_counts": {
            col: int(df[col].isna().sum())
            for col in df.columns
        },
        "duplicate_counts": {
            col: int(df[col].duplicated().sum())
            for col in df.columns
        },
        "roles": {
            col: get_column_role(col)
            for col in df.columns
        },
        "preview": df.head(10).to_dict(orient="records"),
    }
    return summary


def get_duplicate_count(path: str = DATA_PATH) -> int:
    df = load_data(path)
    duplicate_count = df.duplicated().sum()
    return int(duplicate_count)

if __name__ == "__main__":
    data = load_data()
    print(get_data_summary(), "\n", get_duplicate_count())
