import requests
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level = logging.INFO)

def get_api_data(URL: str, START_DATE: str, API_KEY: str) -> pd.DataFrame:

    try:
        logger.info(" Fetching data from EIA API")
        response = requests.get(f"{URL}&start={START_DATE}&api_key={API_KEY}")

        data = response.json()
        df = pd.DataFrame(data['response']['data'])

    except Exception as e:
        logger.error(f"Error fetching API data: {e}")
        df = pd.DataFrame()

    return df

# Testing
if __name__ == "__main__":
    from dotenv import load_dotenv
    import os
    load_dotenv()
    EIA_API_KEY = os.getenv("EIA_API_KEY")
    data = get_api_data(
        URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly&data[0]=value&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000",
        START_DATE = "2026-07-19",
        API_KEY = EIA_API_KEY
    )
    print(data.head())