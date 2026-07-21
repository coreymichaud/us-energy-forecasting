import pandas as pd
from datetime import datetime
from feast import FeatureStore

entity_df = pd.DataFrame(
    {"event_timestamp": [pd.Timestamp(datetime.now(), tz="UTC")], "driver_id": [1001]}
)

feature_refs = [
    "online_store:period",
    "online_store:respondent",
    "online_store:type",
    "online_store:value",
]

fs = FeatureStore(repo_path="feature_pipeline")

training_df = fs.get_historical_features(
    features=[
        "online_store:period",
        "online_store:respondent",
        "online_store:type",
        "online_store:value",
    ],
    entity_df=entity_df,
).to_df()
