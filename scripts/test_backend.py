#!/usr/bin/env python3
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def read_root():
    return {"Hello": "World"}

@app.get("/api/tokens/balance")
async def get_balance():
    return {
        "total_tokens": 0,
        "provisional_tokens": 0,
        "combined_balance": 0,
        "next_training_threshold": 100,
        "remaining_for_training": 100
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
