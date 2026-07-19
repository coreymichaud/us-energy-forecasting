import requests
import json
from dotenv import load_dotenv
import os

load_dotenv()

EIA_API_KEY = os.getenv("EIA_API_KEY")

START_DATE = "2026-07-18T00"

response = requests.get(
    f"https://api.eia.gov/v2/electricity/rto/region-data/data/?frequency=hourly&data[0]=value&start={START_DATE}&sort[0][column]=period&sort[0][direction]=desc&offset=0&length=5000&api_key={EIA_API_KEY}"
)

data = response.json()

print(json.dumps(data, indent=4, sort_keys=True))
