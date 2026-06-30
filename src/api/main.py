"""
main.py

Retail Customer Feedback Intelligence Platform
"""

from fastapi import FastAPI

# from src.api.routes.insights import router as analytics_router
from src.api.routes.analyze import router as analyze_router
from src.api.routes.dashboard import router as dashboard_router

app = FastAPI(

    title="Retail Customer Feedback Intelligence Platform",

    version="1.0.0",

)

app.include_router(analyze_router)
app.include_router(dashboard_router)


@app.get("/")
def home():

    return {

        "application":
            "Retail Customer Feedback Intelligence Platform",

        "version":
            "1.0.0",

        "status":
            "running",
    }


@app.get("/health")
def health():

    return {

        "status": "healthy"

    }