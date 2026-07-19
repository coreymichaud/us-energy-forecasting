import logging
from datetime import timedelta
from pathlib import Path

import pandas as pd
from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float32
from feast.value_type import ValueType

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

REPO_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = REPO_PATH / "data" / "eia_region_data.parquet"


def save_to_offline_store(df: pd.DataFrame, path: Path = DATA_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)

    df = df.copy()
    # Feast needs an explicit event timestamp column for the offline source
    df["event_timestamp"] = pd.to_datetime(df["period"])

    df.to_parquet(path, index=False)
    logger.info(f"Wrote {len(df)} rows to {path}")
    return path


def define_feature_store_objects(data_path: Path):
    """
    Defines the entities, source, and feature view for the EIA region data,
    with `description` documenting what each field is and what validation
    was run on it in validate.py.

    Design note: a single row is uniquely identified by
    (respondent, type, period) - e.g. "PJM Interconnection, LLC" has
    separate rows for demand ("D"), net generation ("NG"), and total
    interchange ("TI") at the same timestamp. So `respondent` AND `type`
    are both modeled as entities (a composite key). If only `respondent`
    were used as the entity, materializing to the online store would let
    each type overwrite the others for the same timestamp.
    """

    respondent = Entity(
        name="respondent",
        join_keys=["respondent"],
        value_type=ValueType.STRING,
        description=(
            "Balancing authority / respondent region reporting electricity "
            "data to EIA (e.g. 'PJM Interconnection, LLC'). Sourced from the "
            "API's 'respondent-name' field, renamed to 'respondent' in "
            "transform.py."
        ),
    )

    measurement_type = Entity(
        name="type",
        join_keys=["type"],
        value_type=ValueType.STRING,
        description=(
            "Short EIA code for what is being measured: 'D' = demand, "
            "'NG' = net generation, 'TI' = total interchange. Kept as the "
            "code (not the descriptive 'type-name') since it's compact and "
            "the mapping is stable."
        ),
    )

    source = FileSource(
        name="eia_region_data_source",
        path=str(data_path),
        timestamp_field="event_timestamp",
        description=(
            "Hourly electricity data pulled from the EIA v2 API "
            "(electricity/rto/region-data), cleaned by transform.py."
        ),
    )

    eia_feature_view = FeatureView(
        name="eia_region_data",
        entities=[respondent, measurement_type],
        ttl=timedelta(days=1),
        schema=[
            Field(
                name="value",
                dtype=Float32,
                description=(
                    "Measured value in megawatthours (MWh) for this "
                    "respondent/type/period. "
                    "Validation (validate.py, great_expectations): "
                    "ExpectColumnValuesToBeBetween(min_value=0, "
                    "max_value=100_000_000) at 'warning' severity, i.e. "
                    "flags negative or implausibly large readings without "
                    "hard-failing the pipeline."
                ),
                tags={
                    "validation_expectation": "ExpectColumnValuesToBeBetween",
                    "validation_min": "0",
                    "validation_max": "100000000",
                    "validation_severity": "warning",
                },
            ),
        ],
        source=source,
        online=True,
        description=(
            "Hourly EIA electricity data (demand, net generation, total "
            "interchange) by respondent region and measurement type."
        ),
    )

    return respondent, measurement_type, eia_feature_view


def load_to_feast(df: pd.DataFrame, repo_path: Path = REPO_PATH) -> FeatureStore:
    """
    Registers the entities/feature view with Feast (`apply`) and
    materializes the current data into the online store.
    """
    data_path = save_to_offline_store(df)
    respondent, measurement_type, feature_view = define_feature_store_objects(data_path)

    store = FeatureStore(repo_path=str(repo_path))
    store.apply([respondent, measurement_type, feature_view])
    logger.info("Applied entity and feature view definitions to Feast")

    store.materialize_incremental(end_date=pd.Timestamp.now(tz="UTC"))
    logger.info("Materialized features into the online store")

    return store


# Testing
if __name__ == "__main__":
    import os

    from dotenv import load_dotenv

    from extract import get_api_data
    from transform import transform_data
    from validate import build_expectation_suite

    load_dotenv()
    EIA_API_KEY = os.getenv("EIA_API_KEY")

    df = get_api_data(
        URL="https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly&data[0]=value&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000",
        START_DATE="2026-07-19",
        API_KEY=EIA_API_KEY,
    )
    df = transform_data(df)

    validation_result = build_expectation_suite(df)
    if not validation_result.success:
        logger.warning("One or more expectations failed - review before trusting these features")

    store = load_to_feast(df)
    print(store.list_feature_views())