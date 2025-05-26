import os
import sys
import json
import uvicorn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import Query, HTTPException
from pathlib import Path

# Create OpenBB configuration directory and set up credentials
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
    "OPENBB_BEA_KEY": "bea_api_key",
    "OPENBB_CENSUS_KEY": "census_api_key",
    "OPENBB_ECONDB_KEY": "econdb_api_key",
    "OPENBB_NASDAQ_KEY": "nasdaq_api_key",
}

for env_key, cred_key in credential_mappings.items():
    if env_key in os.environ and os.environ[env_key]:
        credentials[cred_key] = os.environ[env_key]

# Set OpenBB Hub token if available
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

print(f"Created settings with {len(credentials)} credentials")

# Import after setting up credentials
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

# Helper to clean JSON data
def clean_json_data(data):
    """Replace NaN, Inf values with None for JSON compatibility"""
    if isinstance(data, dict):
        return {k: clean_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(v) for v in data]
    elif isinstance(data, float):
        if np.isnan(data) or np.isinf(data):
            return None
        return data
    return data

# Debug endpoint to understand data structure
@app.get("/api/v1/debug/data_structure")
async def debug_data_structure(symbol: str = "DGS10", provider: str = "fred"):
    """Debug endpoint to see data structure"""
    try:
        if provider == "fred":
            data = obb.economy.fred_series(symbol=symbol, provider=provider)
        else:
            data = obb.equity.price.historical(symbol=symbol, provider=provider)
        
        # Try different ways to access data
        result = {
            "type": str(type(data)),
            "attributes": dir(data)[:20],  # First 20 attributes
        }
        
        # Try to get dataframe
        try:
            df = data.to_dataframe()
            result["dataframe_shape"] = df.shape
            result["dataframe_columns"] = list(df.columns)
            result["dataframe_index"] = str(df.index.name)
            result["first_row"] = df.head(1).to_dict()
        except Exception as e:
            result["dataframe_error"] = str(e)
        
        # Try to get results
        try:
            if hasattr(data, 'results'):
                result["has_results"] = True
                result["results_length"] = len(data.results) if data.results else 0
                if data.results and len(data.results) > 0:
                    result["first_result"] = str(data.results[0])[:200]
        except Exception as e:
            result["results_error"] = str(e)
        
        return result
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

# Simple data fetch that works with OpenBB structure
async def fetch_fred_data(symbol: str, limit: int = 252):
    """Fetch FRED data and return as DataFrame"""
    try:
        data = obb.economy.fred_series(symbol=symbol, provider="fred")
        
        # Convert to list of dicts first
        if hasattr(data, 'results') and data.results:
            records = []
            for item in data.results[-limit:]:  # Last N records
                if hasattr(item, 'date') and hasattr(item, symbol):
                    records.append({
                        'date': item.date,
                        'value': getattr(item, symbol)
                    })
            
            if records:
                df = pd.DataFrame(records)
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                return df
        
        # If above doesn't work, try to_dataframe
        df = data.to_dataframe()
        if symbol in df.columns:
            return df[[symbol]].rename(columns={symbol: 'value'})
        
        return None
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

# Working trend analysis
@app.get("/api/v1/analytics/trend")
async def calculate_trend(
    symbol: str,
    provider: str = "fred",
    window: int = 7
):
    """Calculate trend with moving average"""
    try:
        if provider == "fred":
            df = await fetch_fred_data(symbol)
            if df is None:
                return {"error": "Could not fetch data"}
        else:
            # For non-FRED providers, use the API directly
            data = obb.equity.price.historical(symbol=symbol, provider=provider)
            df = data.to_dataframe()
            if 'close' in df.columns:
                df = df[['close']].rename(columns={'close': 'value'})
        
        # Calculate trend metrics
        df['ma'] = df['value'].rolling(window=window).mean()
        df['roc'] = df['value'].pct_change(window) * 100
        
        # Simple slope calculation
        df['slope'] = df['value'].diff(window) / window
        
        # Return last 100 points
        result = df.tail(100).reset_index().to_dict('records')
        
        return {"data": clean_json_data(result)}
    except Exception as e:
        return {"error": str(e)}

# Working composite analysis
@app.get("/api/v1/analytics/composite")
async def composite_analysis(
    symbols: List[str] = Query(...),
    provider: str = "fred"
):
    """Fetch and combine multiple indicators"""
    try:
        all_data = {}
        
        for symbol in symbols:
            if provider == "fred":
                df = await fetch_fred_data(symbol)
                if df is not None:
                    all_data[symbol] = df['value']
            else:
                try:
                    data = obb.equity.price.historical(symbol=symbol, provider=provider)
                    df = data.to_dataframe()
                    if 'close' in df.columns:
                        all_data[symbol] = df['close']
                except Exception as e:
                    print(f"Error with {symbol}: {e}")
        
        if not all_data:
            return {"error": "No data fetched", "symbols_requested": symbols}
        
        # Combine data
        combined_df = pd.DataFrame(all_data)
        combined_df = combined_df.dropna()  # Remove NaN rows
        
        # Calculate spread if 2 series
        if len(combined_df.columns) == 2:
            cols = list(combined_df.columns)
            combined_df['spread'] = combined_df[cols[0]] - combined_df[cols[1]]
        
        # Return last 100 points
        result = combined_df.tail(100).reset_index().to_dict('records')
        
        return {
            "data": clean_json_data(result),
            "symbols_fetched": list(all_data.keys())
        }
    except Exception as e:
        return {"error": str(e)}

# Working liquidity stress
@app.get("/api/v1/analytics/liquidity_stress")
async def liquidity_stress_model():
    """Simple stress indicator using VIX"""
    try:
        # Try to get VIX data
        vix_df = await fetch_fred_data("VIXCLS", limit=252)
        
        if vix_df is None:
            return {"error": "Could not fetch VIX data"}
        
        # Calculate stress metrics
        vix_df['vix_ma20'] = vix_df['value'].rolling(20).mean()
        vix_df['vix_zscore'] = (vix_df['value'] - vix_df['vix_ma20']) / vix_df['value'].rolling(20).std()
        
        # Normalize to 0-100
        min_vix = vix_df['value'].min()
        max_vix = vix_df['value'].max()
        vix_df['stress_level'] = ((vix_df['value'] - min_vix) / (max_vix - min_vix)) * 100
        
        # Categories
        vix_df['stress_category'] = pd.cut(
            vix_df['stress_level'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['Very Low', 'Low', 'Normal', 'Elevated', 'High']
        )
        
        # Return last 100 points
        result = vix_df.tail(100).reset_index().to_dict('records')
        
        return {"data": clean_json_data(result)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/widgets.json")
async def widgets():
    return {
        # Debug widget
        "debug_data": {
            "name": "Debug Data Structure",
            "description": "See how data is structured",
            "category": "Debug",
            "endpoint": "/api/v1/debug/data_structure",
            "params": {"symbol": "DGS10", "provider": "fred"}
        },
        
        # Basic data
        "fred_series": {
            "name": "FRED Data",
            "description": "Any FRED series",
            "category": "Economy",
            "endpoint": "/api/v1/economy/fred_series",
            "params": {"symbol": "DGS10", "provider": "fred"}
        },
        
        # Working analytics
        "trend_simple": {
            "name": "Trend Analysis",
            "description": "Moving average and momentum",
            "category": "Analytics",
            "endpoint": "/api/v1/analytics/trend",
            "params": {"symbol": "DGS10", "provider": "fred", "window": 7}
        },
        "composite_simple": {
            "name": "Multi-Series Chart",
            "description": "Compare multiple series",
            "category": "Analytics",
            "endpoint": "/api/v1/analytics/composite",
            "params": {"symbols": ["DGS10", "DGS2"], "provider": "fred"}
        },
        "vix_stress": {
            "name": "VIX Stress Monitor",
            "description": "Market stress from VIX",
            "category": "Analytics",
            "endpoint": "/api/v1/analytics/liquidity_stress",
            "params": {}
        }
    }

if __name__ == "__main__":
    print("=== Starting OpenBB API ===")
    print(f"Loaded {len(credentials)} API keys")
    uvicorn.run(app, host="0.0.0.0", port=8000)

# Add this to your existing start_api.py file

@app.get("/openbb-workspace-config")
async def workspace_config():
    """Generate OpenBB Workspace compatible configuration"""
    base_url = "https://ped8gafyuz.us-east-1.awsapprunner.com"
    
    widgets_config = {
        "version": "1.0",
        "widgets": [
            # Economic Data
            {
                "id": "fred_gdp",
                "name": "GDP - Gross Domestic Product",
                "type": "api",
                "config": {
                    "url": f"{base_url}/api/v1/economy/fred_series?symbol=GDP&provider=fred",
                    "method": "GET",
                    "refresh": 300
                }
            },
            {
                "id": "fred_10y",
                "name": "10Y Treasury Rate",
                "type": "api",
                "config": {
                    "url": f"{base_url}/api/v1/economy/fred_series?symbol=DGS10&provider=fred",
                    "method": "GET",
                    "refresh": 60
                }
            },
            {
                "id": "fred_2y",
                "name": "2Y Treasury Rate",
                "type": "api",
                "config": {
                    "url": f"{base_url}/api/v1/economy/fred_series?symbol=DGS2&provider=fred",
                    "method": "GET",
                    "refresh": 60
                }
            },
            {
                "id": "fred_vix",
                "name": "VIX Volatility Index",
                "type": "api",
                "config": {
                    "url": f"{base_url}/api/v1/economy/fred_series?symbol=VIXCLS&provider=fred",
                    "method": "GET",
                    "refresh": 60
                }
            },
            {
                "id": "fred_unemployment",
                "name": "Unemployment Rate",
                "type": "api",
                "config": {
                    "url": f"{base_url}/api/v1/economy/fred_series?symbol=UNRATE&provider=fred",
                    "method": "GET",
                    "refresh": 300
                }
            },
            {
                "id": "fred_cpi",
                "name": "CPI - Inflation",
                "type": "api",
                "config": {
                    "url": f"{base_url}/api/v1/economy/fred_series?symbol=CPIAUCSL&provider=fred",
                    "method": "GET",
                    "refresh": 300
                }
            },
            # Stocks
            {
                "id": "stock_aapl",
                "name": "Apple (AAPL)",
                "type": "api",
                "config": {
                    "url": f"{base_url}/api/v1/equity/price/quote?symbol=AAPL&provider=fmp",
                    "method": "GET",
                    "refresh": 30
                }
            },
            {
                "id": "stock_spy",
                "name": "S&P 500 (SPY)",
                "type": "api",
                "config": {
                    "url": f"{base_url}/api/v1/equity/price/quote?symbol=SPY&provider=polygon",
                    "method": "GET",
                    "refresh": 30
                }
            },
            # Analytics
            {
                "id": "trend_10y",
                "name": "10Y Treasury Trend Analysis",
                "type": "api",
                "config": {
                    "url": f"{base_url}/api/v1/analytics/trend?symbol=DGS10&provider=fred&window=7",
                    "method": "GET",
                    "refresh": 300
                }
            },
            {
                "id": "yield_curve",
                "name": "Yield Curve (10Y-2Y)",
                "type": "api",
                "config": {
                    "url": f"{base_url}/api/v1/analytics/composite?symbols=DGS10&symbols=DGS2&provider=fred",
                    "method": "GET",
                    "refresh": 60
                }
            },
            {
                "id": "market_stress",
                "name": "Market Stress Monitor",
                "type": "api",
                "config": {
                    "url": f"{base_url}/api/v1/analytics/liquidity_stress",
                    "method": "GET",
                    "refresh": 60
                }
            }
        ],
        "dashboards": [
            {
                "id": "main",
                "name": "Market Overview",
                "widgets": ["fred_10y", "stock_spy", "market_stress", "yield_curve"]
            },
            {
                "id": "economy",
                "name": "Economic Indicators",
                "widgets": ["fred_gdp", "fred_unemployment", "fred_cpi", "trend_10y"]
            }
        ]
    }
    
    return widgets_config

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
