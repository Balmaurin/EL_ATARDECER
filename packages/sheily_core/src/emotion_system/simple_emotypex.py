#!/usr/bin/env python3
"""
Simple Emotypex - Emotion Analysis and Learning System
Sistema de análisis emocional inteligente con aprendizaje automático
"""

from typing import Any, Dict, List, Optional
import json
import os
from pathlib import Path
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class EmotionDataset(Dataset):
    """Dataset personalizado para emociones"""

    def __init__(self, texts: List[str], labels: List[str]):
        self.texts = texts
        self.labels = labels
        self.vectorizer = TfidfVectorizer(max_features=5000)

        if texts:
            self.X = self.vectorizer.fit_transform(texts).toarray()
            self.label_to_idx = {label: i for i, label in enumerate(set(labels))}
            self.y = [self.label_to_idx[label] for label in labels]
        else:
            self.X = np.array([])
            self.y = []
            self.label_to_idx = {}

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return {
            'features': torch.FloatTensor(self.X[idx]),
            'label': torch.LongTensor([self.y[idx]])[0]
        }


class SimpleEmotypex:
    """
    Sistema de análisis emocional inteligente
    Aprende patrones emocionales desde datos textuales
    """

    def __init__(self):
        # Configuración de rutas
        self.project_root = Path(__file__).parent.parent.parent
        self.data_dir = self.project_root / "data"
        self.model_path = self.data_dir / "models" / "simple_emotypex_model.pkl"

        # Crear directorios necesarios
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "models").mkdir(exist_ok=True)

        # Inicializar modelo
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)

        # Dataset de emociones
        self.emotion_labels = [
            "joy", "sadness", "anger", "fear", "surprise",
            "disgust", "love", "anticipation", "trust", "optimism"
        ]

        # Cargar modelo si existe
        self._load_model()

    def _load_model(self):
        """Cargar modelo entrenado si existe"""
        try:
            if self.model_path.exists():
                import pickle
                with open(self.model_path, 'rb') as f:
                    self.vectorizer, self.classifier = pickle.load(f)
                print(f"✅ Modelo cargado desde {self.model_path}")
        except Exception as e:
            print(f"⚠️  No se pudo cargar modelo: {e}")

    def _save_model(self):
        """Guardar modelo entrenado"""
        try:
            import pickle
            with open(self.model_path, 'wb') as f:
                pickle.dump((self.vectorizer, self.classifier), f)
            print(f"💾 Modelo guardado en {self.model_path}")
        except Exception as e:
            print(f"❌ Error guardando modelo: {e}")

    def analyze_emotion(self, text: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Analizar emoción en texto
        """
        if not hasattr(self.classifier, 'classes_'):
            return {"emotion": "neutral", "confidence": 0.0, "error": "Modelo no entrenado"}

        try:
            # Vectorizar texto
            features = self.vectorizer.transform([text])

            # Predecir
            prediction = self.classifier.predict(features)[0]
            probabilities = self.classifier.predict_proba(features)[0]
            confidence = probabilities.max()

            # Obtener emoción
            emotion_idx = list(self.classifier.classes_).index(prediction)
            emotion = self.emotion_labels[emotion_idx] if emotion_idx < len(self.emotion_labels) else "neutral"

            return {
                "emotion": emotion,
                "confidence": float(confidence),
                "probabilities": {self.emotion_labels[i]: float(prob) for i, prob in enumerate(probabilities[:len(self.emotion_labels)])}
            }

        except Exception as e:
            return {"emotion": "neutral", "confidence": 0.0, "error": str(e)}

    def train_emotion_model(self, texts: List[str], emotions: List[str]) -> bool:
        """
        Entrenar modelo de emociones
        """
        print("🚀 Entrenando modelo de emociones...")

        try:
            # Preparar datos
            dataset = EmotionDataset(texts, emotions)

            if len(dataset) == 0:
                print("❌ No hay datos suficientes para entrenar")
                return False

            # Dividir datos
            X_train, X_test, y_train, y_test = train_test_split(
                dataset.X, dataset.y, test_size=0.2, random_state=42
            )

            # Entrenar modelo
            self.classifier.fit(X_train, y_train)

            # Evaluar
            accuracy = self.classifier.score(X_test, y_test)
            print(".2f")

            # Guardar modelo
            self._save_model()

            return accuracy > 0.6  # Umbral mínimo de calidad

        except Exception as e:
            print(f"❌ Error entrenando modelo: {e}")
            return False

    def get_emotion_response(self, emotion: str) -> str:
        """
        Generar respuesta basada en emoción detectada
        """
        responses = {
            "joy": "¡Qué alegría detectar tu positividad! 😊",
            "sadness": "Entiendo que puedas sentir tristeza... 🤗",
            "anger": "Veo que hay algo que te molesta. Hablemos de ello.",
            "fear": "Entiendo que puedas sentir ansiedad. ¿Qué te preocupa?",
            "surprise": "¡Vaya sorpresa! Parece que algo te ha impactado.",
            "disgust": "Detecto cierto disgusto. ¿Qué te incomoda?",
            "love": "¡Qué bonito sentir amor! ❤️",
            "anticipation": "Veo expectación en tus palabras. ¿Qué esperas?",
            "trust": "Confías en el proceso, eso es bueno.",
            "optimism": "¡Tu optimismo es contagioso! 🌟",
        }

        return responses.get(emotion, "Interesante emoción detectada.")

    def learn_from_feedback(self, text: str, correct_emotion: str) -> bool:
        """
        Aprender de feedback del usuario
        """
        try:
            # Agregar al dataset de aprendizaje
            feedback_file = self.data_dir / "emotion_feedback.jsonl"
            feedback_entry = {
                "text": text,
                "emotion": correct_emotion,
                "timestamp": datetime.now().isoformat()
            }

            with open(feedback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(feedback_entry, ensure_ascii=False) + '\n')

            print(f"✅ Feedback aprendido: {correct_emotion}")
            return True

        except Exception as e:
            print(f"❌ Error aprendiendo feedback: {e}")
            return False


# Funciones de utilidad
def normalize_emotion(emotion: str) -> str:
    """
    Normalizar nombre de emoción
    """
    emotion_map = {
        "felicidad": "joy",
        "tristeza": "sadness",
        "enojo": "anger",
        "miedo": "fear",
        "sorpresa": "surprise",
        "disgusto": "disgust",
        "amor": "love",
        "anticipación": "anticipation",
        "confianza": "trust",
        "optimismo": "optimism"
    }

    return emotion_map.get(emotion.lower(), "neutral")


def generate_emotion_dataset(size: int = 100) -> List[Dict[str, Any]]:
    """
    Generar dataset sintético de emociones para pruebas
    """
    emotions = ["joy", "sadness", "anger", "fear"]
    texts = []

    for _ in range(size):
        emotion = random.choice(emotions)
        if emotion == "joy":
            text = f"¡Estoy muy feliz! Todo va genial."
        elif emotion == "sadness":
            text = f"Me siento triste hoy. Nada sale bien."
        elif emotion == "anger":
            text = f"¡Estoy furioso! Esto es inaceptable."
        else:  # fear
            text = f"Tengo miedo. ¿Qué va a pasar?"

        texts.append({
            "text": text,
            "emotion": emotion,
            "id": f"synthetic_{len(texts)}"
        })

    return texts


if __name__ == "__main__":
    # Ejemplo de uso
    analyzer = SimpleEmotypex()

    # Texto de prueba
    test_text = "Estoy muy feliz con los resultados obtenidos"
    result = analyzer.analyze_emotion(test_text)

    print(f"Texto: {test_text}")
    print(f"Análisis: {result}")
