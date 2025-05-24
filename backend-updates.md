# Backend Updates Needed

## Add Root Endpoint to start_api.py

Add this code to ~/Desktop/OpenBBTerminal/start_api.py on your local machine:

```python
@app.get("/")
async def root():
    return {
        "message": "Welcome to OpenBB API",
        "status": "online",
        "endpoints": {
            "providers": "/api/v1/universal/providers",
            "data": "/api/v1/universal/data?symbol=SYMBOL&provider=PROVIDER",
            "search": "/api/v1/universal/search?query=QUERY",
            "dashboard": "/api/v1/dashboards/economic"
        }
    }
