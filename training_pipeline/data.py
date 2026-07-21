import logging
from pathlib import Path
from typing import List

import pandas as pd
from feast import FeatureStore
from feast.infra.offline_stores.file_source import SavedDatasetFileStorage

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REPO_PATH = Path(__file__).resolve().parent.parent  # root/
FEATURE_REPO_PATH = REPO_PATH / "feature_pipeline"  # has feature_store.yaml
DATA_PATH = FEATURE_REPO_PATH / "data" / "eia_region_data.parquet"
DATASET_PATH = REPO_PATH / "training_pipeline" / "data" / "eia_training_dataset.parquet"

# feature refs, "<feature_view_name>:<field_name>" - add more here as your
# feature views grow (e.g. "eia_region_data:value", "other_view:other_field")
FEATURES = ["eia_region_data:value"]


def build_entity_df(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """
    Rows to fetch features "as of": one per respondent/type/event_timestamp
    already sitting in the offline store (written by load.py). Feast uses
    event_timestamp to do a point-in-time join, so whatever timestamps you
    put here are the "as of" times for the returned feature values - this
    is what prevents future data from leaking into a training row.
    """
    df = pd.read_parquet(data_path, columns=["respondent", "type", "event_timestamp"])
    df = df.drop_duplicates().reset_index(drop=True)
    logger.info(f"Built entity dataframe with {len(df)} rows")
    return df


def get_historical_features(
    entity_df: pd.DataFrame,
    features: List[str] = FEATURES,
    repo_path: Path = FEATURE_REPO_PATH,
) -> pd.DataFrame:
    """
    Ad-hoc retrieval - good for quick exploration/notebooks. Doesn't persist
    anything; re-running this re-runs the join against current source data.
    """
    store = FeatureStore(repo_path=str(repo_path))
    retrieval_job = store.get_historical_features(
        entity_df=entity_df, features=features
    )
    df = retrieval_job.to_df()
    logger.info(
        f"Retrieved {len(df)} rows x {len(df.columns)} cols of historical features"
    )
    return df


def save_dataset(
    entity_df: pd.DataFrame,
    name: str,
    features: List[str] = FEATURES,
    repo_path: Path = FEATURE_REPO_PATH,
    output_path: Path = DATASET_PATH,
) -> pd.DataFrame:
    """
    Persist the retrieval as a Feast SavedDataset: the exact entity rows,
    timestamps, and feature values used get written to output_path and
    registered in Feast under `name`, so you (or a teammate) can pull the
    identical training set back later via:

        store.get_saved_dataset(name).to_df()

    instead of hoping the offline store hasn't drifted since you trained.
    """
    store = FeatureStore(repo_path=str(repo_path))
    retrieval_job = store.get_historical_features(
        entity_df=entity_df, features=features
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = store.create_saved_dataset(
        from_=retrieval_job,
        name=name,
        storage=SavedDatasetFileStorage(path=str(output_path)),
        allow_overwrite=True,
    )
    logger.info(f"Saved dataset '{name}' to {output_path}")
    return dataset.to_df()


# Testing
if __name__ == "__main__":
    entity_df = build_entity_df()

    df = save_dataset(entity_df, name="eia_region_training_set")
    print(df.head())
    print(df.shape)
