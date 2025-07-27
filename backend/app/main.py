from fastapi import FastAPI
from .api import interview
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="AI Interview Assistant API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(interview.router, prefix="/api/v1")

@app.get("/")
def read_root():
    return {"message": "Welcome to the AI Interview Assistant API"}

# @app.get("/status")
# def get_status():
#     return {"status": "ok", "message": "Backend is running"}
