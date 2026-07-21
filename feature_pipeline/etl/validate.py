import great_expectations as gx
import pandas as pd


def build_expectation_suite(df: pd.DataFrame) -> dict:
    """
    Build a Great Expectations suite for the given DataFrame.

    Args:
        df (pd.DataFrame): The DataFrame to validate.

    Returns:
        dict: The validation results.
    """

    # Creating context
    context = gx.get_context()

    # Creating the batch
    data_source = context.data_sources.add_pandas("pandas")
    data_asset = data_source.add_dataframe_asset(name="pd dataframe asset")

    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        "batch definition"
    )
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    # Creating expectation suite
    suite = gx.ExpectationSuite("EIA_suite")
    suite = context.suites.add(suite)

    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="value", min_value=0, max_value=100000000, severity="warning"
        )
    )

    # Result
    result = batch.validate(suite)

    return result


if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()
    from feature_pipeline.etl.extract import get_api_data
    from feature_pipeline.etl.transform import transform_data

    EIA_API_KEY = os.getenv("EIA_API_KEY")
    df = get_api_data(
        URL="https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly&data[0]=value&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000",
        START_DATE="2026-07-19",
        API_KEY=EIA_API_KEY,
    )
    df = transform_data(df)
    print(build_expectation_suite(df))
