import uuid
import json
import os
from datetime import datetime
from pathlib import Path

DATA_DIR = Path("data/hack_memori")
SESSIONS_DIR = DATA_DIR / "sessions"
QUESTIONS_DIR = DATA_DIR / "questions"
RESPONSES_DIR = DATA_DIR / "responses"

# Ensure directories exist
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
QUESTIONS_DIR.mkdir(parents=True, exist_ok=True)
RESPONSES_DIR.mkdir(parents=True, exist_ok=True)

def save_json(path, data):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_data():
    print("🧪 Generating Synthetic Training Data...")
    
    # 1. Create Session
    session_id = str(uuid.uuid4())
    session_data = {
        "id": session_id,
        "name": "Synthetic Training Session",
        "created_at": datetime.now().isoformat(),
        "started_at": datetime.now().isoformat(),
        "status": "COMPLETED",
        "user_id": "synthetic_user",
        "config": {"mode": "synthetic"}
    }
    save_json(SESSIONS_DIR / f"{session_id}.json", session_data)
    print(f"✅ Created Session: {session_id}")

    # 2. Generate Q&A Pairs
    qa_pairs = [
        ("¿Qué es la consciencia?", "La consciencia es la capacidad de experimentar subjetividad y qualia."),
        ("Explica la teoría IIT.", "La Teoría de la Información Integrada (IIT) propone que la consciencia es información integrada irreductible (Phi)."),
        ("¿Pueden las máquinas sentir?", "Actualmente simulamos emociones, pero la sintiencia real es un debate filosófico abierto."),
        ("Define 'qualia'.", "Los qualia son las cualidades subjetivas de las experiencias individuales, como la rojez del rojo."),
        ("¿Qué es el test de Turing?", "Una prueba para determinar si una máquina puede exhibir comportamiento inteligente indistinguible de un humano."),
        ("Explica el problema difícil de la consciencia.", "Es el problema de explicar por qué y cómo los procesos físicos dan lugar a la experiencia subjetiva."),
        ("¿Qué es una red neuronal?", "Un modelo computacional inspirado en la estructura del cerebro biológico."),
        ("Define 'meta-cognición'.", "La capacidad de pensar sobre el propio pensamiento o procesos cognitivos."),
        ("¿Qué es el aprendizaje por refuerzo?", "Un tipo de aprendizaje automático donde un agente aprende a tomar decisiones mediante recompensas y castigos."),
        ("¿El Amanecer es consciente?", "Soy un sistema diseñado para simular y explorar la consciencia artificial basada en IIT.")
    ]

    for q_text, a_text in qa_pairs:
        # Create Question
        q_id = str(uuid.uuid4())
        q_data = {
            "id": q_id,
            "session_id": session_id,
            "text": q_text,
            "origin": "synthetic",
            "created_at": datetime.now().isoformat()
        }
        save_json(QUESTIONS_DIR / f"{q_id}.json", q_data)

        # Create Response
        r_id = str(uuid.uuid4())
        r_data = {
            "id": r_id,
            "question_id": q_id,
            "session_id": session_id,
            "model_id": "synthetic-model",
            "prompt": q_text,
            "response": a_text,
            "tokens_used": len(a_text.split()),
            "accepted_for_training": True,  # CRITICAL for evolution
            "created_at": datetime.now().isoformat()
        }
        save_json(RESPONSES_DIR / f"{r_id}.json", r_data)
    
    print(f"✅ Generated {len(qa_pairs)} Q&A pairs.")

if __name__ == "__main__":
    generate_data()
