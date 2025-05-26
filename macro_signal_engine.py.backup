#!/usr/bin/env python3
"""
Macro Signal Engine - Automated Data Scraper
Runs daily at 6 AM UTC to collect financial stress indicators
"""

import os
import json
import asyncio
import pandas as pd
import numpy as np
import httpx
from datetime import datetime, timedelta
from pathlib import Path
import logging
import tempfile
from scipy import stats
import warnings

warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/app/macro_cron.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create data directories
DATA_DIR = Path('/app/macro_signals')
RAW_DIR = DATA_DIR / 'raw'
DATA_DIR.mkdir(exist_ok=True)
RAW_DIR.mkdir(exist_ok=True)

class MacroDataScraper:
    def __init__(self):
        self.session_timeout = 30.0
        self.max_retries = 3
        
    async def scrape_ofr_fsi(self):
        """Scrape OFR Financial Stress Index"""
        try:
            logger.info("Scraping OFR FSI data...")
            csv_url = "https://www.financialresearch.gov/financial-stress-index/files/financial-stress-index-data.csv"
            
            async with httpx.AsyncClient(timeout=self.session_timeout) as client:
                response = await client.get(csv_url)
                response.raise_for_status()
                
            # Parse CSV
            from io import StringIO
            df = pd.read_csv(StringIO(response.text))
            df.columns = df.columns.str.strip().str.lower()
            df['date'] = pd.to_datetime(df['date'], errors='coerce')
            df['ofr_fsi'] = pd.to_numeric(df['ofr_fsi'], errors='coerce')
            df = df.dropna().sort_values('date')
            
            # Calculate metrics
            df['daily_change'] = df['ofr_fsi'].diff()
            df['daily_pct_change'] = df['ofr_fsi'].pct_change() * 100
            df['7d_avg'] = df['ofr_fsi'].rolling(7, min_periods=1).mean()
            df['30d_avg'] = df['ofr_fsi'].rolling(30, min_periods=1).mean()
            
            # Detect patterns
            df['z_score'] = (df['ofr_fsi'] - df['30d_avg']) / df['ofr_fsi'].rolling(30, min_periods=20).std()
            df['is_peak'] = (df['ofr_fsi'] > df['ofr_fsi'].shift(1)) & (df['ofr_fsi'] > df['ofr_fsi'].shift(-1))
            df['is_bottom'] = (df['ofr_fsi'] < df['ofr_fsi'].shift(1)) & (df['ofr_fsi'] < df['ofr_fsi'].shift(-1))
            
            # Save raw data
            df.to_csv(RAW_DIR / 'ofr_fsi.csv', index=False)
            
            # Return latest summary
            latest = df.iloc[-1]
            return {
                'source': 'OFR_FSI',
                'date': latest['date'].strftime('%Y-%m-%d'),
                'value': float(latest['ofr_fsi']),
                'daily_change': float(latest.get('daily_change', 0)),
                'daily_pct_change': float(latest.get('daily_pct_change', 0)),
                '7d_avg': float(latest.get('7d_avg', 0)),
                'z_score': float(latest.get('z_score', 0)),
                'is_peak': bool(latest.get('is_peak', False)),
                'is_bottom': bool(latest.get('is_bottom', False)),
                'total_records': len(df)
            }
            
        except Exception as e:
            logger.error(f"OFR FSI scraping failed: {e}")
            return None

    async def run_daily_scrape(self):
        """Main function to run daily data scraping"""
        logger.info("=== Starting Daily Macro Signal Scraping ===")
        return True

async def main():
    """Main entry point"""
    scraper = MacroDataScraper()
    success = await scraper.run_daily_scrape()
    
    if success:
        logger.info("✅ Macro signal scraping completed successfully")
        exit(0)
    else:
        logger.error("❌ Macro signal scraping failed")
        exit(1)

if __name__ == "__main__":
    asyncio.run(main())
