import logging
from pathlib import Path
from typing import Tuple
from sktime.forecasting.model_selection import temporal_train_test_split
import pandas as pd

import wandb
from dotenv import load_dotenv
import os

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REPO_PATH = Path(__file__).resolve().parent.parent
FEATURE_REPO_PATH = REPO_PATH / "feature_pipeline"
DATA_PATH = FEATURE_REPO_PATH / "data" / "eia_region_data.parquet"


def load_data_from_feast(
    data_path: Path = DATA_PATH,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load data from Feast.

    Args:
        data_path (Path): The path to the feast data file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: The training and testing data.
    """

    with wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        name="load_training_data",
        job_type="data_loading",
    ) as run:
        df = pd.read_parquet(
            data_path, columns=["respondent", "type", "event_timestamp", "value"]
        )
        df = df.drop_duplicates().reset_index(drop=True)
        logger.info(f"Built entity dataframe with {len(df)} rows")

        artifact = wandb.Artifact(name = "training_data", type = "dataset")
        artifact.add_file(local_path=DATA_PATH, name = "training_dataset")
        run.log_artifact(artifact)
        logger.info("Logged training dataset artifact")

    with wandb.init(
        project=os.getenv("WANDB_PROJECT"),
        name="train_test_split",
        job_type="prepare_dataset",
    ):
        
        df["event_timestamp"] = pd.PeriodIndex(df["event_timestamp"], freq = "H")
        df = df.set_index(["respondent", "type", "event_timestamp"])
        X = df.drop(columns=["value"])
        y = df["value"]
        X_train, X_test, y_train, y_test = temporal_train_test_split(
            X, y, test_size=0.2
        )



    return X_train, X_test, y_train, y_test


# Testing
if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data_from_feast(DATA_PATH)
    print(X_train.head())
    print(X_train.shape, X_test.shape)
    print(y_train.head())
    print(y_train.shape)
