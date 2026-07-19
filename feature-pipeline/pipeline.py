import argparse
import datetime
import logging
import os
import sys

from dotenv import load_dotenv

from etl.extract import get_api_data
from etl.transform import transform_data
from etl.validate import build_expectation_suite
from etl.load import load_to_feast

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

EIA_URL = (
    "https://api.eia.gov/v2/electricity/rto/region-data/data/"
    "?frequency=hourly&data[0]=value&sort[0][column]=period"
    "&sort[0][direction]=desc&offset=0&length=5000"
)


def run_pipeline(start_date: str):
    load_dotenv()
    api_key = os.getenv("EIA_API_KEY")
    if not api_key:
        logger.error("EIA_API_KEY not set - check your .env file")
        return None
    logger.info(f"Extracting data starting from {start_date}")
    df = get_api_data(URL=EIA_URL, START_DATE=start_date, API_KEY=api_key)
    if df.empty:
        logger.error("Extract returned no data - aborting pipeline")
        return None
    logger.info(f"Extracted {len(df)} rows")

    logger.info("Transforming data")
    df = transform_data(df)
    logger.info(f"Transformed data - columns: {list(df.columns)}")

    logger.info("Validating data")
    validation_result = build_expectation_suite(df)
    if not validation_result.success:
        logger.warning(
            "One or more expectations failed - inspect validation_result "
            "before trusting these features"
        )
    else:
        logger.info("All expectations passed")

    logger.info("Loading data into Feast")
    store = load_to_feast(df)
    feature_view_names = [fv.name for fv in store.list_feature_views()]
    logger.info(f"Loaded into Feast - feature views: {feature_view_names}")

    return store


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EIA electricity data ETL pipeline")
    parser.add_argument(
        "--start-date",
        default=datetime.date.today().isoformat(),
        help="Start date for the EIA API pull, format YYYY-MM-DD (default: today)",
    )
    args = parser.parse_args()

    store = run_pipeline(start_date=args.start_date)
    if store is None:
        sys.exit(1)
