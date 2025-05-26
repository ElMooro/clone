FROM python:3.10-slim

# Install system dependencies including cron
RUN apt-get update && apt-get install -y \
    gcc g++ \
    cron \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy OpenBB platform
COPY openbb_platform /app/openbb_platform

# Copy API files
COPY start_api.py /app/start_api.py
COPY macro_signal_engine.py /app/macro_signal_engine.py

# Install OpenBB platform
WORKDIR /app/openbb_platform
RUN pip install --upgrade pip poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi

WORKDIR /app/openbb_platform/core
RUN pip install -e .

WORKDIR /app/openbb_platform/extensions/platform_api
RUN pip install -e .

# Install additional dependencies
WORKDIR /app
RUN pip install pandas numpy scipy redis httpx statsmodels openpyxl xlrd

# Set up cron job for daily data scraping
RUN echo "0 6 * * * cd /app && /usr/local/bin/python /app/macro_signal_engine.py >> /app/macro_cron.log 2>&1" > /etc/cron.d/macro-scraper
RUN chmod 0644 /etc/cron.d/macro-scraper
RUN crontab /etc/cron.d/macro-scraper

# Create macro_signals directory
RUN mkdir -p /app/macro_signals/raw

# Set environment variables
ENV PYTHONPATH=/app/openbb_platform:$PYTHONPATH

# Expose port
EXPOSE 8000

# Start both cron and the API
CMD ./start.sh
