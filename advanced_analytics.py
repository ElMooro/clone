import os
import uvicorn
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from fastapi import Query
from scipy import stats

# Set environment variables
env_mapping = {
    "OPENBB_FRED_KEY": ["fred_api_key", "FRED_API_KEY"],
    "OPENBB_FMP_KEY": ["fmp_api_key", "FMP_API_KEY"],
    "OPENBB_POLYGON_KEY": ["polygon_api_key", "POLYGON_API_KEY"],
    "OPENBB_BLS_KEY": ["bls_api_key", "BLS_API_KEY"],
    "OPENBB_BENZINGA_KEY": ["benzinga_api_key", "BENZINGA_API_KEY"],
    "OPENBB_TIINGO_KEY": ["tiingo_api_key", "TIINGO_API_KEY"],
    "OPENBB_EIA_KEY": ["eia_api_key", "EIA_API_KEY"],
}

for source_key, target_keys in env_mapping.items():
    if source_key in os.environ:
        for target_key in target_keys:
            os.environ[target_key] = os.environ[source_key]

from openbb_core.api.rest_api import app
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Advanced Analytics Endpoints

@app.get("/api/v1/analytics/trend")
async def calculate_trend(
    symbol: str,
    provider: str = "fred",
    window: int = 7,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Calculate trend line slope using rolling linear regression"""
    try:
        # Fetch data using OpenBB
        from openbb import obb
        if provider == "fred":
            data = obb.economy.fred_series(symbol=symbol, start_date=start_date, end_date=end_date, provider=provider)
        else:
            data = obb.equity.price.historical(symbol=symbol, start_date=start_date, end_date=end_date, provider=provider)
        
        df = data.to_dataframe()
        df['trend_slope'] = df['value'].rolling(window=window).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else np.nan
        )
        
        return {"data": df.reset_index().to_dict('records')}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/analytics/changes")
async def calculate_changes(
    symbol: str,
    provider: str = "fred",
    periods: List[int] = Query(default=[1, 7, 30, 90]),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Calculate period-to-period changes and percent changes"""
    try:
        from openbb import obb
        if provider == "fred":
            data = obb.economy.fred_series(symbol=symbol, start_date=start_date, end_date=end_date, provider=provider)
        else:
            data = obb.equity.price.historical(symbol=symbol, start_date=start_date, end_date=end_date, provider=provider)
        
        df = data.to_dataframe()
        
        # Calculate changes for each period
        for period in periods:
            df[f'change_{period}d'] = df['value'].diff(period)
            df[f'pct_change_{period}d'] = df['value'].pct_change(period) * 100
        
        # Quarter over quarter growth
        df['qoq_growth'] = df['value'].pct_change(90) * 100
        
        return {"data": df.reset_index().to_dict('records')}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/analytics/moving_averages")
async def calculate_moving_averages(
    symbol: str,
    provider: str = "fred",
    windows: List[int] = Query(default=[7, 30, 90]),
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Calculate moving averages for multiple windows"""
    try:
        from openbb import obb
        if provider == "fred":
            data = obb.economy.fred_series(symbol=symbol, start_date=start_date, end_date=end_date, provider=provider)
        else:
            data = obb.equity.price.historical(symbol=symbol, start_date=start_date, end_date=end_date, provider=provider)
        
        df = data.to_dataframe()
        
        for window in windows:
            df[f'ma_{window}d'] = df['value'].rolling(window=window).mean()
        
        # Calculate delta (momentum)
        df['delta'] = df['value'].diff()
        
        return {"data": df.reset_index().to_dict('records')}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/analytics/z_score")
async def calculate_z_score(
    symbol: str,
    provider: str = "fred",
    lookback: int = 252,  # 1 year of trading days
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Calculate Z-score deviation from long-term mean"""
    try:
        from openbb import obb
        if provider == "fred":
            data = obb.economy.fred_series(symbol=symbol, start_date=start_date, end_date=end_date, provider=provider)
        else:
            data = obb.equity.price.historical(symbol=symbol, start_date=start_date, end_date=end_date, provider=provider)
        
        df = data.to_dataframe()
        
        # Calculate rolling mean and std
        df['rolling_mean'] = df['value'].rolling(window=lookback).mean()
        df['rolling_std'] = df['value'].rolling(window=lookback).std()
        
        # Calculate Z-score
        df['z_score'] = (df['value'] - df['rolling_mean']) / df['rolling_std']
        
        # Add alert levels
        df['extreme_high'] = df['z_score'] > 2
        df['extreme_low'] = df['z_score'] < -2
        
        return {"data": df.reset_index().to_dict('records')}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/analytics/composite")
async def composite_analysis(
    symbols: List[str] = Query(..., description="List of symbols to analyze"),
    formula: Optional[str] = Query(None, description="Custom formula e.g., 'GDP/M2'"),
    provider: str = "fred",
    timeframe: str = "1Y",
    overlay_yoy: bool = False
):
    """Composite analysis with multiple indicators and custom formulas"""
    try:
        from openbb import obb
        import pandas as pd
        
        # Timeframe mapping
        timeframe_map = {
            "1D": 1, "1W": 7, "1M": 30, "1Y": 365,
            "5Y": 1825, "10Y": 3650, "Max": None
        }
        
        days = timeframe_map.get(timeframe, 365)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days) if days else None
        
        # Fetch data for all symbols
        all_data = {}
        for symbol in symbols:
            if provider == "fred":
                data = obb.economy.fred_series(
                    symbol=symbol,
                    start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
                    end_date=end_date.strftime("%Y-%m-%d"),
                    provider=provider
                )
            else:
                data = obb.equity.price.historical(
                    symbol=symbol,
                    start_date=start_date.strftime("%Y-%m-%d") if start_date else None,
                    end_date=end_date.strftime("%Y-%m-%d"),
                    provider=provider
                )
            
            df = data.to_dataframe()
            all_data[symbol] = df['value']
        
        # Combine all data
        combined_df = pd.DataFrame(all_data)
        
        # Apply custom formula if provided
        if formula:
            # Simple formula parser (e.g., "GDP/M2")
            if "/" in formula:
                num, den = formula.split("/")
                if num in combined_df.columns and den in combined_df.columns:
                    combined_df['formula_result'] = combined_df[num] / combined_df[den]
            elif "-" in formula:
                left, right = formula.split("-")
                if left in combined_df.columns and right in combined_df.columns:
                    combined_df['formula_result'] = combined_df[left] - combined_df[right]
        
        # Add YoY % change if requested
        if overlay_yoy:
            for col in combined_df.columns:
                combined_df[f'{col}_yoy_pct'] = combined_df[col].pct_change(252) * 100
        
        return {"data": combined_df.reset_index().to_dict('records')}
    except Exception as e:
        return {"error": str(e)}

@app.get("/api/v1/analytics/liquidity_stress")
async def liquidity_stress_model(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
):
    """Composite Liquidity Stress Model using multiple indicators"""
    try:
        from openbb import obb
        
        # Key liquidity indicators
        indicators = {
            "DGS10": "10Y Treasury Yield",
            "DGS2": "2Y Treasury Yield",
            "BAMLH0A0HYM2": "High Yield Spread",
            "VIXCLS": "VIX",
            "DFF": "Fed Funds Rate",
            "DEXUSEU": "USD/EUR Exchange Rate"
        }
        
        stress_data = {}
        for symbol, name in indicators.items():
            try:
                data = obb.economy.fred_series(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    provider="fred"
                )
                df = data.to_dataframe()
                stress_data[name] = df['value']
            except:
                continue
        
        # Combine and calculate stress index
        stress_df = pd.DataFrame(stress_data)
        
        # Normalize each indicator (Z-score)
        for col in stress_df.columns:
            stress_df[f'{col}_zscore'] = stats.zscore(stress_df[col].dropna())
        
        # Calculate composite stress index (average of Z-scores)
        zscore_cols = [col for col in stress_df.columns if '_zscore' in col]
        stress_df['liquidity_stress_index'] = stress_df[zscore_cols].mean(axis=1)
        
        # Add stress levels
        stress_df['stress_level'] = pd.cut(
            stress_df['liquidity_stress_index'],
            bins=[-np.inf, -1, 0, 1, np.inf],
            labels=['Low', 'Normal', 'Elevated', 'High']
        )
        
        return {"data": stress_df.reset_index().to_dict('records')}
    except Exception as e:
        return {"error": str(e)}

@app.get("/widgets.json")
async def widgets():
    return {
        # Basic data widgets
        "fred_data": {
            "name": "FRED Economic Data",
            "description": "Federal Reserve data with analytics",
            "category": "Economy",
            "endpoint": "/api/v1/economy/fred_series",
            "params": {"symbol": "GDP", "provider": "fred"}
        },
        "stock_data": {
            "name": "Stock Price Data",
            "description": "Equity prices with analytics",
            "category": "Equity",
            "endpoint": "/api/v1/equity/price/historical",
            "params": {"symbol": "AAPL", "provider": "polygon"}
        },
        
        # Advanced analytics widgets
        "trend_analysis": {
            "name": "Trend Analysis",
            "description": "7-day rolling regression slope",
            "category": "Analytics",
            "endpoint": "/api/v1/analytics/trend",
            "params": {"symbol": "GDP", "provider": "fred", "window": 7}
        },
        "change_metrics": {
            "name": "Change Metrics",
            "description": "Period changes and % changes",
            "category": "Analytics",
            "endpoint": "/api/v1/analytics/changes",
            "params": {"symbol": "GDP", "provider": "fred", "periods": [1, 7, 30, 90]}
        },
        "moving_averages": {
            "name": "Moving Averages",
            "description": "Multiple period averages",
            "category": "Analytics",
            "endpoint": "/api/v1/analytics/moving_averages",
            "params": {"symbol": "SPY", "provider": "polygon", "windows": [7, 30, 90]}
        },
        "z_score_analysis": {
            "name": "Z-Score Analysis",
            "description": "Statistical deviation alerts",
            "category": "Analytics",
            "endpoint": "/api/v1/analytics/z_score",
            "params": {"symbol": "VIX", "provider": "fred", "lookback": 252}
        },
        "composite_chart": {
            "name": "Composite Analysis",
            "description": "Multiple indicators with formulas",
            "category": "Analytics",
            "endpoint": "/api/v1/analytics/composite",
            "params": {
                "symbols": ["GDP", "M2SL"],
                "formula": "GDP/M2SL",
                "provider": "fred",
                "timeframe": "5Y",
                "overlay_yoy": true
            }
        },
        "liquidity_stress": {
            "name": "Liquidity Stress Model",
            "description": "Multi-factor stress indicator",
            "category": "Analytics",
            "endpoint": "/api/v1/analytics/liquidity_stress",
            "params": {}
        }
    }

@app.get("/apps.json")
async def apps():
    return {
        "advanced_analytics": {
            "name": "Advanced Analytics Suite",
            "description": "Complete analytical toolkit",
            "widgets": ["trend_analysis", "change_metrics", "moving_averages", "z_score_analysis"]
        },
        "macro_dashboard": {
            "name": "Macro Analytics Dashboard",
            "description": "Economic indicators with advanced metrics",
            "widgets": ["composite_chart", "liquidity_stress", "z_score_analysis"]
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
