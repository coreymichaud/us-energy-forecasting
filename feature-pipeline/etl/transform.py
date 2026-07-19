import pandas as pd

def transform_data(df: pd.DataFrame) -> pd.DataFrame:

    # Changing period column from object to datetime
    df["period"] = pd.to_datetime(df["period"])

    # Changing value to numeric
    df["value"] = pd.to_numeric(df["value"])

    # Dropping respondent column in favor of respondent-name, value-units because they're all MWh, and type-name because we can just use the shortened version
    df.drop(columns = ["respondent", "value-units", "type-name"], inplace = True)

    # Renaming respondent-name to just respondent
    df.rename(columns = {"respondent-name": "respondent"}, inplace = True)

    return df


# Testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    from extract import get_api_data
    load_dotenv()
    EIA_API_KEY = os.getenv("EIA_API_KEY")
    df = get_api_data(
        URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly&data[0]=value&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000",
        START_DATE = "2026-07-19",
        API_KEY = EIA_API_KEY
    )
    df = transform_data(df)
    print(df.head())
    print(df.dtypes)