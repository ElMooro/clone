# Enhanced OpenBB API with Liquidity Intelligence Layer
import os
import sys
import json
import uvicorn
import numpy as np
import pandas as pd
import redis
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Union
from fastapi import Query, HTTPException, Response
from pathlib import Path
import hashlib
import pickle
from functools import wraps
import httpx
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
import xml.etree.ElementTree as ET
import traceback
import tempfile

warnings.filterwarnings('ignore')

# Redis connection setup with environment variable
REDIS_URL = os.getenv("REDIS_URL", "redis://default:RNSMN5OzpJSEMoAiG570jQIkRxII3uMY@redis-15406.c241.us-east-1-4.ec2.redns.redis-cloud.com:15406")
try:
    if REDIS_URL:
        redis_client = redis.from_url(REDIS_URL, decode_responses=False)
        redis_client.ping()
        print("✅ Redis connected successfully!")
    else:
        redis_client = None
        print("⚠️ Redis URL not configured")
except Exception as e:
    print(f"❌ Redis connection failed: {e}")
    redis_client = None

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
    "OPENBB_BEA_KEY": "bea_api_key",
    "OPENBB_CENSUS_KEY": "census_api_key",
    "OPENBB_ECONDB_KEY": "econdb_api_key",
    "OPENBB_NASDAQ_KEY": "nasdaq_api_key",
    "OPENBB_INTRINIO_KEY": "intrinio_api_key",
    "OPENBB_ECB_KEY": "ecb_api_key",
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
    expose_headers=["*"]
)

# ============= CACHING UTILITIES =============

def generate_cache_key(prefix: str, **kwargs) -> str:
    """Generate a unique cache key based on function name and parameters"""
    filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
    params_str = json.dumps(filtered_kwargs, sort_keys=True)
    hash_digest = hashlib.md5(params_str.encode()).hexdigest()[:8]
    return f"{prefix}:{hash_digest}"

def cache_result(expiration: int = 3600):
    """Decorator to cache function results in Redis"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not redis_client:
                return await func(*args, **kwargs)
            
            # Simple cache key generation
            cache_key = generate_cache_key(func.__name__, **kwargs)
            
            try:
                cached_data = redis_client.get(cache_key)
                if cached_data:
                    print(f"🎯 Cache hit: {cache_key}")
                    return pickle.loads(cached_data)

                print(f"🔄 Cache miss: {cache_key}")
                result = await func(*args, **kwargs)
                
                try:
                    pickled_result = pickle.dumps(result)
                    redis_client.setex(cache_key, expiration, pickled_result)
                except Exception as e:
                    print(f"⚠️ Error setting cache: {e}")
                
                return result
            
            except Exception as e:
                print(f"❌ Cache error: {e}")
                return await func(*args, **kwargs)
        return wrapper
    return decorator

# ============= ENHANCED FRED API INTEGRATION =============

class FREDAPIClient:
    """Enhanced FRED API client following official guidelines"""
    BASE_URL = "https://api.stlouisfed.org/fred"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.default_file_type = "json"

    async def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Make a request to FRED API"""
        if not self.api_key:
            print("FRED API key not configured.")
            return None
        url = f"{self.BASE_URL}{endpoint}"
        params['api_key'] = self.api_key
        params['file_type'] = params.get('file_type', self.default_file_type)

        async with httpx.AsyncClient(timeout=20.0) as client:
            try:
                response = await client.get(url, params=params)
                response.raise_for_status()
                if params['file_type'] == 'json':
                    return response.json()
                else:
                    return {"xml": response.text}
            except Exception as e:
                print(f"FRED API error: {e}")
            return None

    async def series_search(self, search_text: str, **kwargs) -> Optional[Dict]:
        params = {
            'search_text': search_text,
            'limit': kwargs.get('limit', 1000),
            'offset': kwargs.get('offset', 0),
            'order_by': kwargs.get('order_by', 'search_rank'),
            'sort_order': kwargs.get('sort_order', 'desc'),
        }
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request('/series/search', params)

    async def series_observations(self, series_id: str, **kwargs) -> Optional[Dict]:
        params = {
            'series_id': series_id,
            'limit': kwargs.get('limit', 100000),
            'offset': kwargs.get('offset', 0),
            'sort_order': kwargs.get('sort_order', 'asc'),
            'observation_start': kwargs.get('observation_start'),
            'observation_end': kwargs.get('observation_end'),
            'units': kwargs.get('units', 'lin'),
        }
        params = {k: v for k, v in params.items() if v is not None}
        return await self._make_request('/series/observations', params)

    async def series_info(self, series_id: str) -> Optional[Dict]:
        params = {'series_id': series_id}
        return await self._make_request('/series', params)

fred_client = FREDAPIClient(credentials.get('fred_api_key', ''))

# ============= DIRECT API INTEGRATIONS =============

async def ecb_api_direct(dataset: str, series_key: str, start_period: Optional[str] = None, end_period: Optional[str] = None):
    base_url = "https://data-api.ecb.europa.eu/service/data"
    url = f"{base_url}/{dataset}/{series_key}"
    params = {'format': 'jsondata'}
    if start_period: params['startPeriod'] = start_period
    if end_period: params['endPeriod'] = end_period

    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"ECB API error: {e}")
        return None

async def ny_fed_api_direct(endpoint: str, params: Optional[Dict] = None):
    base_url = "https://markets.newyorkfed.org/api"
    url = f"{base_url}/{endpoint.strip('/')}"
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.get(url, params=params or {})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"NY Fed API error: {e}")
        return None

async def polygon_api_direct(endpoint: str, params: Optional[Dict] = None):
    base_url = "https://api.polygon.io"
    url = f"{base_url}/{endpoint.strip('/')}"
    if not params: params = {}
    params['apiKey'] = credentials.get('polygon_api_key', '')
    if not params['apiKey']:
        print("Polygon API key not configured.")
        return None
    async with httpx.AsyncClient(timeout=20.0) as client:
        try:
            response = await client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            print(f"Polygon API error: {e}")
        return None

# ============= LIQUIDITY INTELLIGENCE LAYER =============

async def get_ofr_fsi_data():
    """Get OFR Financial Stress Index data"""
    try:
        csv_url = "https://www.financialresearch.gov/financial-stress-index/files/financial-stress-index-data.csv"
        async with httpx.AsyncClient(timeout=30.0) as client:
            csv_response = await client.get(csv_url)
            csv_response.raise_for_status()
            
        # Parse CSV data
        from io import StringIO
        df = pd.read_csv(StringIO(csv_response.text))
        df.columns = df.columns.str.strip().str.lower()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['value'] = pd.to_numeric(df['ofr_fsi'], errors='coerce')
        df.dropna(subset=['date', 'value'], inplace=True)
        df = df.sort_values('date')
        
        # Calculate basic metrics
        if len(df) > 1:
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            weekly_change = ((latest['value'] - prev['value']) / prev['value'] * 100) if prev['value'] != 0 else 0
            
            return {
                "latest_value": float(latest['value']),
                "date": latest['date'].strftime('%Y-%m-%d'),
                "weekly_change": float(weekly_change),
                "data_points": len(df),
                "source": "OFR"
            }
    except Exception as e:
        print(f"OFR FSI error: {e}")
    return None

async def get_treasury_auction_data():
    """Get Treasury auction monitoring data"""
    try:
        # Simple treasury rate check using FRED
        fred_data = await fred_client.series_observations(series_id="DGS10", limit=5, sort_order="desc")
        if fred_data and "observations" in fred_data:
            latest_obs = fred_data["observations"][0]
            if latest_obs["value"] != ".":
                return {
                    "latest_10y_rate": float(latest_obs["value"]),
                    "date": latest_obs["date"],
                    "source": "FRED_DGS10"
                }
    except Exception as e:
        print(f"Treasury data error: {e}")
    return None

async def get_ny_fed_fails_data():
    """Get NY Fed primary dealer fails data"""
    try:
        # Try to get fails data from NY Fed
        url = "https://www.newyorkfed.org/markets/primarydealer-fails-data"
        async with httpx.AsyncClient(timeout=30.0) as client:
            html = await client.get(url)
            html.raise_for_status()
            
            # Look for Excel link in HTML
            import re
            html_content = html.text
            excel_links = re.findall(r'href="([^"]*\.xls[^"]*)"', html_content, re.IGNORECASE)
            
            fail_link = None
            for link in excel_links:
                if "fails" in link.lower():
                    if link.startswith('/'):
                        fail_link = "https://www.newyorkfed.org" + link
                    else:
                        fail_link = link
                    break
            
            if fail_link:
                excel = await client.get(fail_link)
                
                # Use temporary file for Excel parsing
                with tempfile.NamedTemporaryFile(suffix='.xls', delete=False) as temp_file:
                    temp_file.write(excel.content)
                    temp_path = temp_file.name
                
                try:
                    df = pd.read_excel(temp_path, skiprows=5)
                    os.unlink(temp_path)  # Clean up
                    
                    df.columns = df.columns.str.strip().str.lower()
                    df = df.rename(columns={
                        "week ending": "date",
                        "total fails to deliver": "fails_deliver",
                        "total fails to receive": "fails_receive"
                    })
                    
                    df = df[['date', 'fails_deliver', 'fails_receive']].dropna()
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    df['fails_deliver'] = pd.to_numeric(df['fails_deliver'], errors='coerce')
                    df['fails_receive'] = pd.to_numeric(df['fails_receive'], errors='coerce')
                    df["fails_total"] = df["fails_deliver"] + df["fails_receive"]
                    df = df.sort_values('date')
                    
                    if not df.empty:
                        latest = df.iloc[-1]
                        return {
                            "latest_total_fails": float(latest['fails_total']),
                            "date": latest['date'].strftime('%Y-%m-%d'),
                            "fails_deliver": float(latest['fails_deliver']),
                            "fails_receive": float(latest['fails_receive']),
                            "source": "NYFED"
                        }
                except Exception as e:
                    if temp_path and os.path.exists(temp_path):
                        os.unlink(temp_path)
                    print(f"Excel processing error: {e}")
    except Exception as e:
        print(f"NY Fed fails error: {e}")
    return None

def calculate_liquidity_stress_score(ofr_data, treasury_data, fails_data):
    """Calculate composite liquidity stress score"""
    stress_components = []
    
    # OFR FSI component (normalized)
    if ofr_data:
        fsi_value = ofr_data.get('latest_value', 0)
        # Normalize FSI (typical range -1 to 5)
        fsi_normalized = min(max(fsi_value / 5.0, 0), 1)
        stress_components.append(fsi_normalized * 0.4)  # 40% weight
    
    # Treasury rate component (simplified)
    if treasury_data:
        rate = treasury_data.get('latest_10y_rate', 4.0)
        # Higher rates can indicate stress, normalize around 4%
        rate_stress = min(max((rate - 2.0) / 8.0, 0), 1)
        stress_components.append(rate_stress * 0.3)  # 30% weight
    
    # Fails component (simplified)
    if fails_data:
        fails = fails_data.get('latest_total_fails', 0)
        # Normalize fails (typical range 0 to 500B)
        fails_normalized = min(fails / 500_000_000_000, 1)
        stress_components.append(fails_normalized * 0.3)  # 30% weight
    
    if stress_components:
        overall_stress = sum(stress_components)
        return min(overall_stress, 1.0)
    return 0.0

# ============= ENHANCED PANDAS CALCULATIONS =============

def calculate_percent_changes(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    """Enhanced calculate_percent_changes that handles FRED '.' missing values"""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    # Handle FRED's '.' missing value indicator
    if value_column in df.columns:
        if df[value_column].dtype == 'object':
            df[value_column] = df[value_column].replace('.', np.nan)
            df[value_column] = df[value_column].replace('', np.nan)
        df[value_column] = pd.to_numeric(df[value_column], errors='coerce')

    if value_column not in df.columns or not pd.api.types.is_numeric_dtype(df[value_column]):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if not numeric_cols.empty:
            value_column = numeric_cols[0]
        else:
            return df

    # Ensure datetime index
    if not isinstance(df.index, pd.DatetimeIndex):
        date_cols = ['date', 'Date', 'DATE', 'timestamp']
        for col_name in date_cols:
            if col_name in df.columns:
                try:
                    df[col_name] = pd.to_datetime(df[col_name], errors='coerce')
                    if not df[col_name].isnull().all():
                        df.set_index(col_name, inplace=True)
                        break
                except:
                    continue

    df.sort_index(inplace=True)

    # Calculate percent changes
    if value_column in df.columns and pd.api.types.is_numeric_dtype(df[value_column]):
        df['daily_pct_change'] = df[value_column].pct_change(1) * 100
        df['weekly_pct_change'] = df[value_column].pct_change(5) * 100
        df['monthly_pct_change'] = df[value_column].pct_change(21) * 100
        df['yearly_pct_change'] = df[value_column].pct_change(252) * 100

        # Add unit changes
        df['daily_change_units'] = df[value_column].diff(1)
        df['weekly_change_units'] = df[value_column].diff(5)
        df['monthly_change_units'] = df[value_column].diff(21)
        df['yearly_change_units'] = df[value_column].diff(252)

        # Moving averages
        df['ma_7'] = df[value_column].rolling(window=7, min_periods=1).mean()
        df['ma_30'] = df[value_column].rolling(window=30, min_periods=1).mean()

    return df

def clean_json_data(data):
    """Clean data for JSON serialization"""
    if isinstance(data, dict):
        return {k: clean_json_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [clean_json_data(v) for v in data]
    elif isinstance(data, (float, np.floating)):
        if np.isnan(data) or np.isinf(data):
            return None
        return float(data)
    elif isinstance(data, (int, np.integer)):
        return int(data)
    elif isinstance(data, pd.Timestamp):
        return data.isoformat()
    elif pd.isna(data):
        return None
    return data

# ============= API ENDPOINTS =============

@app.get("/api/v1/test/cors")
async def test_cors():
    return {"status": "ok", "cors": "enabled", "timestamp": datetime.now().isoformat()}

@app.get("/api/v1/health")
async def health_check():
    return {
        "status": "healthy",
        "redis": "connected" if redis_client else "disconnected",
        "timestamp": datetime.now().isoformat()
    }

# ============= LIQUIDITY INTELLIGENCE ENDPOINTS =============

@app.get("/api/v1/ofr/fsi")
@cache_result(expiration=3600)
async def get_ofr_fsi():
    """Get OFR Financial Stress Index"""
    data = await get_ofr_fsi_data()
    if data:
        return {"symbol": "OFR_FSI", "provider": "ofr_direct", "metrics": data}
    return JSONResponse(status_code=404, content={"error": "Failed to fetch OFR FSI data"})

@app.get("/api/v1/treasury/monitor")
@cache_result(expiration=3600)
async def treasury_monitor():
    """Treasury auction monitoring"""
    data = await get_treasury_auction_data()
    if data:
        return {"symbol": "TREASURY_10Y", "provider": "fred_direct", "metrics": data}
    return JSONResponse(status_code=404, content={"error": "Failed to fetch Treasury data"})

@app.get("/api/v1/nyfed/fails")
@cache_result(expiration=86400)
async def nyfed_fails():
    """NY Fed primary dealer fails"""
    data = await get_ny_fed_fails_data()
    if data:
        return {"symbol": "NYFED_FAILS", "provider": "nyfed_direct", "metrics": data}
    return JSONResponse(status_code=404, content={"error": "Failed to fetch NY Fed fails data"})

@app.get("/api/v1/liquidity/dashboard")
@cache_result(expiration=1800)
async def liquidity_dashboard():
    """Unified liquidity stress dashboard"""
    try:
        # Gather all liquidity data
        ofr_data, treasury_data, fails_data = await asyncio.gather(
            get_ofr_fsi_data(),
            get_treasury_auction_data(), 
            get_ny_fed_fails_data(),
            return_exceptions=True
        )
        
        # Handle exceptions
        if isinstance(ofr_data, Exception): ofr_data = None
        if isinstance(treasury_data, Exception): treasury_data = None
        if isinstance(fails_data, Exception): fails_data = None
        
        # Calculate composite stress score
        overall_stress = calculate_liquidity_stress_score(ofr_data, treasury_data, fails_data)
        
        stress_level = (
            "EXTREME" if overall_stress > 0.8 else
            "HIGH" if overall_stress > 0.6 else  
            "MODERATE" if overall_stress > 0.3 else
            "LOW"
        )
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_stress_score": round(float(overall_stress), 3),
            "stress_level": stress_level,
            "components": {
                "ofr_fsi": ofr_data,
                "treasury_monitor": treasury_data,
                "nyfed_fails": fails_data
            },
            "alert_flags": {
                "high_stress": overall_stress > 0.7,
                "ofr_available": ofr_data is not None,
                "treasury_available": treasury_data is not None,
                "fails_available": fails_data is not None
            }
        }
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Liquidity dashboard error: {str(e)}"})

# ============= FRED API ENDPOINTS =============

@app.get("/api/v1/fred/search")
@cache_result(expiration=7200)
async def fred_search_endpoint(
    search_text: str = Query(..., description="Search text"),
    limit: int = Query(1000, description="Maximum results")
):
    results = await fred_client.series_search(search_text=search_text, limit=limit)
    if results: 
        return results
    return JSONResponse(status_code=404, content={"error": "Failed to search FRED series"})

@app.get("/api/v1/fred/series/observations")
@cache_result(expiration=1800)
async def fred_series_observations(
    series_id: str = Query(..., description="FRED series ID"),
    observation_start: Optional[str] = Query(None, description="Start date (YYYY-MM-DD)"),
    observation_end: Optional[str] = Query(None, description="End date (YYYY-MM-DD)"),
    limit: int = Query(100000, description="Maximum observations"),
    include_pandas: bool = Query(True, description="Include pandas calculations")
):
    data = await fred_client.series_observations(
        series_id=series_id, 
        observation_start=observation_start, 
        observation_end=observation_end,
        limit=limit
    )

    if not data or 'observations' not in data:
        return JSONResponse(status_code=404, content={"error": f"No FRED data found for {series_id}"})

    if include_pandas:
        try:
            df = pd.DataFrame(data['observations'])
            if 'date' in df.columns and 'value' in df.columns:
                # Handle FRED missing values
                df['value'] = df['value'].replace('.', np.nan)
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df.dropna(subset=['date'], inplace=True)
                
                if not df.empty:
                    df_calculated = calculate_percent_changes(df.copy(), 'value')
                    
                    # Get latest non-NaN value
                    valid_rows = df_calculated[df_calculated['value'].notna()]
                    if not valid_rows.empty:
                        latest_row = valid_rows.iloc[-1]
                        latest_metrics = {
                            "latest_value": latest_row.get('value'),
                            "daily_change": latest_row.get('daily_pct_change'),
                            "weekly_change": latest_row.get('weekly_pct_change'),
                            "monthly_change": latest_row.get('monthly_pct_change'),
                            "yearly_change": latest_row.get('yearly_pct_change'),
                            "daily_change_units": latest_row.get('daily_change_units'),
                            "weekly_change_units": latest_row.get('weekly_change_units'),
                            "monthly_change_units": latest_row.get('monthly_change_units'),
                            "yearly_change_units": latest_row.get('yearly_change_units'),
                        }
                        
                        return {
                            "series_id": series_id,
                            "data": clean_json_data(df_calculated.reset_index().tail(100).to_dict('records')),
                            "metrics": clean_json_data(latest_metrics),
                            "count": len(df_calculated)
                        }
        except Exception as e:
            print(f"Pandas processing error: {e}")
    
    return clean_json_data(data)

@app.get("/api/v1/fred/series/info")
@cache_result(expiration=86400)
async def fred_series_info(series_id: str): 
    info = await fred_client.series_info(series_id)
    if info: 
        return info
    return JSONResponse(status_code=404, content={"error": f"Failed to retrieve info for {series_id}"})

# ============= UNIVERSAL DATA ENDPOINT =============

@app.get("/api/v1/universal/search")
@cache_result(expiration=7200)
async def universal_search(query: str, provider: str = "all", limit: int = 50): 
    results = {"query": query, "provider": provider, "results": {}}
    
    if provider in ["all", "fred_direct"]:
        fred_results = await fred_client.series_search(query, limit=limit)
        if fred_results and 'seriess' in fred_results:
            results["results"]["fred_direct"] = fred_results['seriess'][:limit]
    
    return results

@app.get("/api/v1/universal/data")
@cache_result(expiration=1800)
async def universal_data(
    symbol: str,
    provider: str,
    data_type: str = "auto",
    include_changes: bool = True,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    if not symbol or not provider:
        return JSONResponse(status_code=400, content={"error": "Symbol and provider required"})

    try:
        raw_data_payload = None
        value_col_name = 'value'

        # FRED Direct API
        if provider == "fred_direct":
            fred_data = await fred_client.series_observations(
                series_id=symbol, 
                observation_start=start_date, 
                observation_end=end_date
            )
            if fred_data and 'observations' in fred_data:
                raw_data_payload = pd.DataFrame(fred_data['observations'])
                raw_data_payload['value'] = raw_data_payload['value'].replace('.', np.nan)
                raw_data_payload['date'] = pd.to_datetime(raw_data_payload['date'], errors='coerce')
                raw_data_payload['value'] = pd.to_numeric(raw_data_payload['value'], errors='coerce')
                raw_data_payload.dropna(subset=['date'], inplace=True)
                value_col_name = 'value'

        # NY Fed Direct API
        elif provider == "ny_fed_direct":
            endpoint_path = f"rates/{symbol.lower()}"
            ny_params = {}
            if start_date: ny_params['startDate'] = start_date
            if end_date: ny_params['endDate'] = end_date
            
            ny_data = await ny_fed_api_direct(endpoint_path, params=ny_params if ny_params else None)
            if ny_data:
                data_list = ny_data.get('refRates', ny_data if isinstance(ny_data, list) else [])
                if data_list:
                    raw_data_payload = pd.DataFrame(data_list)
                    # Standardize column names
                    if 'effectiveDate' in raw_data_payload.columns and 'percentRate' in raw_data_payload.columns:
                        raw_data_payload.rename(columns={'effectiveDate': 'date', 'percentRate': 'value'}, inplace=True)
                    raw_data_payload['date'] = pd.to_datetime(raw_data_payload['date'], errors='coerce')
                    raw_data_payload['value'] = pd.to_numeric(raw_data_payload['value'], errors='coerce')
                    raw_data_payload.dropna(subset=['date', 'value'], inplace=True)
                    value_col_name = 'value'

        # Polygon Direct API
        elif provider == "polygon_direct":
            if data_type == "historical" and start_date and end_date:
                endpoint = f"v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
            else:
                endpoint = f"v2/aggs/ticker/{symbol}/prev"
            
            poly_data = await polygon_api_direct(endpoint, params={'adjusted': 'true'})
            if poly_data and 'results' in poly_data:
                raw_data_payload = pd.DataFrame(poly_data['results'])
                raw_data_payload.rename(columns={'t': 'date', 'c': 'close'}, inplace=True)
                raw_data_payload['date'] = pd.to_datetime(raw_data_payload['date'], unit='ms', errors='coerce')
                raw_data_payload['close'] = pd.to_numeric(raw_data_payload['close'], errors='coerce')
                raw_data_payload.dropna(subset=['date', 'close'], inplace=True)
                value_col_name = 'close'

        # ECB Direct API
        elif provider == "ecb_direct":
            parts = symbol.split(".", 1)
            if len(parts) == 2:
                dataset, series_key = parts
                ecb_data = await ecb_api_direct(dataset, series_key, start_date, end_date)
                if ecb_data:
                    # Simplified ECB parsing
                    try:
                        series_data = ecb_data.get('dataSets', [{}])[0].get('series', {})
                        if series_data:
                            # Get first series
                            first_series = list(series_data.values())[0]
                            observations = first_series.get('observations', {})
                            if observations:
                                # Create simple dataframe
                                data_points = []
                                for idx, values in observations.items():
                                    if values:
                                        data_points.append({'date': f"2024-01-{int(idx)+1:02d}", 'value': values[0]})
                                
                                if data_points:
                                    raw_data_payload = pd.DataFrame(data_points)
                                    raw_data_payload['date'] = pd.to_datetime(raw_data_payload['date'], errors='coerce')
                                    raw_data_payload['value'] = pd.to_numeric(raw_data_payload['value'], errors='coerce')
                                    value_col_name = 'value'
                    except Exception as e:
                        print(f"ECB parsing error: {e}")

        if raw_data_payload is None or raw_data_payload.empty:
            return JSONResponse(status_code=404, content={"error": f"No data retrieved for {symbol} from {provider}"})

        # Sort by date
        raw_data_payload.sort_values(by='date', inplace=True)

        final_response = {
            "symbol": symbol,
            "provider": provider,
            "data": clean_json_data(raw_data_payload.tail(100).to_dict('records')),
            "metrics": {"value_column_used": value_col_name}
        }

        if include_changes and len(raw_data_payload) > 1:
            # Set date as index for calculations
            calc_df = raw_data_payload.set_index('date') if 'date' in raw_data_payload.columns else raw_data_payload
            df_calculated = calculate_percent_changes(calc_df, value_col_name)
            
            if not df_calculated.empty:
                # Get last valid row
                if provider == "fred_direct":
                    valid_rows = df_calculated[df_calculated[value_col_name].notna()]
                    latest_row = valid_rows.iloc[-1] if not valid_rows.empty else df_calculated.iloc[-1]
                else:
                    latest_row = df_calculated.iloc[-1]
                
                metrics = {
                    "latest_value": latest_row.get(value_col_name),
                    "daily_change": latest_row.get('daily_pct_change'),
                    "weekly_change": latest_row.get('weekly_pct_change'),
                    "monthly_change": latest_row.get('monthly_pct_change'),
                    "yearly_change": latest_row.get('yearly_pct_change'),
                    "daily_change_units": latest_row.get('daily_change_units'),
                    "weekly_change_units": latest_row.get('weekly_change_units'),
                    "monthly_change_units": latest_row.get('monthly_change_units'),
                    "yearly_change_units": latest_row.get('yearly_change_units'),
                    "value_column_used": value_col_name
                }
                final_response["metrics"] = clean_json_data(metrics)
                final_response["data"] = clean_json_data(df_calculated.reset_index().tail(100).to_dict('records'))

        return final_response

    except Exception as e:
        print(f"Universal data error: {e}")
        return JSONResponse(status_code=500, content={"error": f"Error retrieving data: {str(e)}"})

# ============= MAIN ENTRY POINT =============

if __name__ == "__main__":
    print("=== Starting Enhanced OpenBB API with Liquidity Intelligence ===")
    print(f"=== Redis Status: {'Connected ✅' if redis_client else 'Not Connected ❌'} ===")
    print("=== FRED API Client: Initialized ===")
    print("=== Liquidity Intelligence Layer: Added ===")
    print("=== New endpoints: /api/v1/ofr/fsi, /api/v1/treasury/monitor, /api/v1/nyfed/fails, /api/v1/liquidity/dashboard ===")
    
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)), log_level=log_level)