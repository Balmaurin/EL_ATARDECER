#!/usr/bin/env python3
"""
Simple Backend Server - Sheily AI Tokens API
===========================================

Servidor FastAPI simplificado solo para endpoints de tokens.
"""

import sqlite3
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Base de datos
DATABASE_PATH = Path("../data/sheily_dashboard.db")

app = FastAPI(title="Sheily Tokens API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Base de datos helper
def get_connection():
    return sqlite3.connect(DATABASE_PATH)

def init_db():
    """Inicializar base de datos básica"""
    try:
        with get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    tokens INTEGER DEFAULT 0,
                    provisional_tokens INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """
            )
            # Usuario por defecto
            conn.execute(
                """
                INSERT OR IGNORE INTO users (username, tokens) VALUES (?, ?)
            """,
                ("default_user", 0),
            )
            conn.commit()
        print("✅ Database initialized!")
    except Exception as e:
        print(f"❌ Database init error: {e}")

def get_token_balance_split(user_id: int = 1):
    """Obtener balance dividido de tokens"""
    try:
        with get_connection() as conn:
            result = conn.execute(
                "SELECT tokens, provisional_tokens FROM users WHERE id = ?", (user_id,)
            ).fetchone()
            if result:
                return {
                    "total_tokens": result[0],
                    "provisional_tokens": result[1],
                    "combined_balance": result[0] + result[1]
                }
            return {"total_tokens": 0, "provisional_tokens": 0, "combined_balance": 0}
    except Exception as e:
        print(f"❌ Database query error: {e}")
        return {"total_tokens": 0, "provisional_tokens": 0, "combined_balance": 0}

def update_provisional_tokens(user_id: int, amount: int):
    """Update provisional tokens for user"""
    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE users SET provisional_tokens = provisional_tokens + ? WHERE id = ?", (amount, user_id)
            )
            conn.commit()
            print(f"User {user_id} provisional tokens updated by {amount}")
    except Exception as e:
        print(f"❌ Provisional tokens update error: {e}")

def update_user_tokens(user_id: int, tokens: int, reason: str):
    """Actualizar tokens del usuario"""
    try:
        with get_connection() as conn:
            conn.execute(
                "UPDATE users SET tokens = tokens + ? WHERE id = ?", (tokens, user_id)
            )
            conn.commit()
            print(f"User {user_id} tokens updated by {tokens}: {reason}")
    except Exception as e:
        print(f"❌ Tokens update error: {e}")

@app.on_event("startup")
async def startup_event():
    """Inicializar BD al iniciar"""
    init_db()
    print("✅ Simple backend tokens iniciado!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "Sheily Simple Tokens Backend", "version": "1.0.0"}

@app.get("/api/health")
async def health_check():
    """Verificar estado del servidor"""
    balance = get_token_balance_split()
    return {
        "status": "healthy",
        "message": "Simple tokens backend running",
        "endpoints": ["/api/tokens/balance", "/api/tokens", "/api/tokens/update"],
        "balance": balance
    }

@app.get("/api/tokens/balance")
async def get_token_balance():
    """Obtener balance dividido de tokens"""
    balance = get_token_balance_split()
    return {
        "total_tokens": balance["total_tokens"],
        "provisional_tokens": balance["provisional_tokens"],
        "combined_balance": balance["combined_balance"],
        "next_training_threshold": 100,
        "remaining_for_training": max(0, 100 - balance["provisional_tokens"])
    }

@app.get("/api/tokens")
async def get_tokens():
    """Obtener tokens del usuario"""
    balance = get_token_balance_split()
    return {"tokens": balance["total_tokens"]}

@app.post("/api/tokens/update")
async def update_tokens(token_update: dict):
    """Actualizar tokens del usuario"""
    try:
        tokens = token_update.get("tokens", 0)
        reason = token_update.get("reason", "Manual update")

        update_user_tokens(1, tokens, reason)
        new_balance = get_token_balance_split()

        return {
            "tokens": new_balance["total_tokens"],
            "change": tokens,
            "provisional_tokens": new_balance["provisional_tokens"],
            "combined_balance": new_balance["combined_balance"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error actualizando tokens: {e}")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting server on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
