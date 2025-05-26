import os
import sys
import json
import uvicorn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional
from fastapi import Query
from pathlib import Path

# Set up credentials
home = Path.home()
openbb_dir = home / ".openbb_platform"
openbb_dir.mkdir(exist_ok=True)

credentials = {}
credential_mappings = {
    "OPENBB_FRED_KEY": "fred_api_key",
    "OPENBB_FMP_KEY": "fmp_api_key",
    "OPENBB_POLYGON_KEY": "polygon_api_key",
    "OPENBB_BLS_KEY": "bls_api_key",
    "OPENBB_BENZINGA_KEY": "benzinga_api_key",
    "OPENBB_TIINGO_KEY": "tiingo_api_key",
    "OPENBB_EIA_KEY": "eia_api_key",
}

for env_key, cred_key in credential_mappings.items():
    if env_key in os.environ and os.environ[env_key]:
        credentials[cred_key] = os.environ[env_key]

if "OPENBB_HUB_TOKEN" in os.environ:
    os.environ["OPENBB_PAT"] = os.environ["OPENBB_HUB_TOKEN"]

user_settings = {
    "credentials": credentials,
    "preferences": {},
    "defaults": {"provider": "fred"}
}

settings_file = openbb_dir / "user_settings.json"
with open(settings_file, "w") as f:
    json.dump(user_settings, f, indent=2)

from openbb_core.api.rest_api import app
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openbb import obb

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add the widget URLs endpoint
@app.get("/widget-urls")
async def get_widget_urls():
    """Simple list of all widget URLs for easy copy-paste"""
    base_url = "https://ped8gafyuz.us-east-1.awsapprunner.com"
    
    urls = {
        "instructions": "Copy each URL and add as endpoint in OpenBB Workspace",
        "economic_data": {
            "GDP": f"{base_url}/api/v1/economy/fred_series?symbol=GDP&provider=fred",
            "10Y_Treasury": f"{base_url}/api/v1/economy/fred_series?symbol=DGS10&provider=fred",
            "2Y_Treasury": f"{base_url}/api/v1/economy/fred_series?symbol=DGS2&provider=fred",
            "VIX": f"{base_url}/api/v1/economy/fred_series?symbol=VIXCLS&provider=fred",
            "Unemployment": f"{base_url}/api/v1/economy/fred_series?symbol=UNRATE&provider=fred",
            "CPI": f"{base_url}/api/v1/economy/fred_series?symbol=CPIAUCSL&provider=fred",
            "M2_Money": f"{base_url}/api/v1/economy/fred_series?symbol=M2SL&provider=fred",
            "Fed_Funds": f"{base_url}/api/v1/economy/fred_series?symbol=FEDFUNDS&provider=fred"
        },
        "stocks": {
            "Apple": f"{base_url}/api/v1/equity/price/quote?symbol=AAPL&provider=fmp",
            "Microsoft": f"{base_url}/api/v1/equity/price/quote?symbol=MSFT&provider=fmp",
            "Google": f"{base_url}/api/v1/equity/price/quote?symbol=GOOGL&provider=fmp",
            "SPY_ETF": f"{base_url}/api/v1/equity/price/quote?symbol=SPY&provider=polygon"
        },
        "analytics": {
            "Treasury_Trend": f"{base_url}/api/v1/analytics/trend?symbol=DGS10&provider=fred&window=7",
            "Yield_Curve": f"{base_url}/api/v1/analytics/composite?symbols=DGS10&symbols=DGS2&provider=fred",
            "Market_Stress": f"{base_url}/api/v1/analytics/liquidity_stress",
            "GDP_M2_Ratio": f"{base_url}/api/v1/analytics/composite?symbols=GDP&symbols=M2SL&formula=GDP/M2SL&provider=fred"
        }
    }
    
    return urls

# Keep all your existing endpoints below...
# (Copy the rest of your working start_api.py content here)

if __name__ == "__main__":
    print("=== Starting OpenBB API ===")
    uvicorn.run(app, host="0.0.0.0", port=8000)
