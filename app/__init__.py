from fastapi import FastAPI
from app.routes import router

app = FastAPI(
    title="Data Analytics ChatBot",
    description="Upload CSV, ask questions, get SQL results + charts",
    version="1.0.0",
)

app.include_router(router, prefix="/api")
