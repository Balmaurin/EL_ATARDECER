#!/usr/bin/env python3
"""
SISTEMA DE CHAT INTERACTIVO CON SHEILY V1 ENTRENADO
==================================================
Chat funcional con el modelo realmente entrenado
"""

import torch
import json
import pickle
import numpy as np
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class SheliyChatSystem:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        
        # Cargar modelos
        self.load_models()
    
    def load_models(self):
        """Cargar todos los modelos entrenados"""
        print("🔄 Cargando modelos entrenados...")
        
        # Cargar modelo GPT2
        try:
            model_path = self.models_dir / "sheily_simple"
            self.tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
            self.model = GPT2LMHeadModel.from_pretrained(str(model_path))
            self.model.eval()  # Modo evaluación
            print("✅ Modelo GPT2 cargado")
        except Exception as e:
            print(f"❌ Error cargando GPT2: {e}")
            self.tokenizer = None
            self.model = None
        
        # Cargar embeddings
        try:
            with open(self.models_dir / "vectorizer.pkl", "rb") as f:
                self.vectorizer = pickle.load(f)
            
            with open(self.models_dir / "svd.pkl", "rb") as f:
                self.svd = pickle.load(f)
            
            self.embeddings = np.load(self.models_dir / "embeddings.npy")
            print("✅ Embeddings cargados")
        except Exception as e:
            print(f"❌ Error cargando embeddings: {e}")
            self.vectorizer = None
            self.svd = None
            self.embeddings = None
        
        # Cargar dataset original para respuestas similares
        try:
            with open(self.project_root / "data" / "training_data.json", "r", encoding="utf-8") as f:
                self.training_data = json.load(f)
            print("✅ Dataset de entrenamiento cargado")
        except:
            self.training_data = []
    
    def get_embedding(self, text):
        """Obtener embedding de un texto"""
        if self.vectorizer and self.svd:
            tfidf_vector = self.vectorizer.transform([text])
            embedding = self.svd.transform(tfidf_vector)
            return embedding[0]
        return None
    
    def find_similar_response(self, user_input):
        """Encontrar respuesta similar basada en embeddings"""
        if not self.vectorizer or not self.svd or not self.training_data:
            return None
        
        try:
            user_embedding = self.get_embedding(user_input)
            
            best_similarity = -1
            best_response = None
            
            for item in self.training_data:
                item_embedding = self.get_embedding(item["input"])
                
                # Calcular similitud coseno
                similarity = np.dot(user_embedding, item_embedding) / (
                    np.linalg.norm(user_embedding) * np.linalg.norm(item_embedding)
                )
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_response = item["output"]
            
            if best_similarity > 0.5:  # Umbral de similitud
                return f"[Embedding] {best_response}"
            
        except Exception as e:
            print(f"Error en similitud: {e}")
        
        return None
    
    def generate_gpt2_response(self, user_input):
        """Generar respuesta con GPT2"""
        if not self.tokenizer or not self.model:
            return None
        
        try:
            # Formatear entrada como en el entrenamiento
            prompt = f"Pregunta: {user_input} Respuesta:"
            
            # Tokenizar
            inputs = self.tokenizer.encode(prompt, return_tensors="pt")
            
            # Generar respuesta
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=inputs.shape[1] + 50,
                    num_return_sequences=1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=0.8,
                    top_p=0.9,
                    repetition_penalty=1.1
                )
            
            # Decodificar y limpiar respuesta
            full_response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extraer solo la parte de la respuesta
            if "Respuesta:" in full_response:
                response = full_response.split("Respuesta:")[-1].strip()
                
                # Limpiar respuesta
                if "Pregunta:" in response:
                    response = response.split("Pregunta:")[0].strip()
                
                return f"[GPT2] {response}" if response else None
            
        except Exception as e:
            print(f"Error generando respuesta GPT2: {e}")
        
        return None
    
    def get_fallback_response(self, user_input):
        """Respuestas de fallback basadas en palabras clave"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ["hola", "hi", "hey"]):
            return "[Fallback] ¡Hola! Soy Sheily, tu asistente AI entrenado. ¿En qué puedo ayudarte?"
        
        elif any(word in user_lower for word in ["como", "estas", "tal"]):
            return "[Fallback] ¡Estoy funcionando perfectamente! Gracias por preguntar. ¿Y tú cómo estás?"
        
        elif any(word in user_lower for word in ["que", "eres", "quien"]):
            return "[Fallback] Soy Sheily V1, un modelo de IA entrenado específicamente para este proyecto. Puedo ayudarte con conversaciones y preguntas básicas."
        
        elif any(word in user_lower for word in ["ayuda", "help"]):
            return "[Fallback] ¡Por supuesto! Puedo conversar contigo, responder preguntas básicas y ayudarte con información general. ¿Qué necesitas?"
        
        elif any(word in user_lower for word in ["gracias", "thanks"]):
            return "[Fallback] ¡De nada! Es un placer ayudarte. ¿Hay algo más en lo que pueda asistirte?"
        
        elif any(word in user_lower for word in ["adios", "bye", "chao"]):
            return "[Fallback] ¡Hasta luego! Ha sido un placer conversar contigo. ¡Que tengas un excelente día!"
        
        else:
            return "[Fallback] Entiendo tu mensaje, pero necesito más contexto para darte una respuesta específica. ¿Podrías reformular tu pregunta?"
    
    def process_message(self, user_input):
        """Procesar mensaje del usuario con múltiples estrategias"""
        if not user_input.strip():
            return "Por favor, escribe algo para que pueda ayudarte."
        
        # 1. Intentar respuesta basada en similitud de embeddings
        similarity_response = self.find_similar_response(user_input)
        if similarity_response:
            return similarity_response
        
        # 2. Intentar generación con GPT2
        gpt2_response = self.generate_gpt2_response(user_input)
        if gpt2_response:
            return gpt2_response
        
        # 3. Respuesta de fallback
        return self.get_fallback_response(user_input)
    
    def start_chat(self):
        """Iniciar chat interactivo"""
        print("\n" + "="*60)
        print("🤖 SHEILY V1 - CHAT INTERACTIVO")
        print("Modelo realmente entrenado y funcional")
        print("="*60)
        print("Comandos especiales:")
        print("- 'salir' o 'exit' para terminar")
        print("- 'info' para ver información del modelo")
        print("- 'test' para ejecutar pruebas rápidas")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n👤 Tú: ").strip()
                
                if user_input.lower() in ['salir', 'exit', 'quit']:
                    print("🤖 Sheily: ¡Hasta luego! Gracias por probar mi sistema entrenado.")
                    break
                
                elif user_input.lower() == 'info':
                    self.show_model_info()
                    continue
                
                elif user_input.lower() == 'test':
                    self.run_quick_tests()
                    continue
                
                elif not user_input:
                    continue
                
                # Procesar y responder
                response = self.process_message(user_input)
                print(f"🤖 Sheily: {response}")
                
            except KeyboardInterrupt:
                print("\n🤖 Sheily: Chat interrumpido. ¡Hasta luego!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def show_model_info(self):
        """Mostrar información del modelo"""
        print("\n📊 INFORMACIÓN DEL MODELO SHEILY V1")
        print("-" * 40)
        
        # Cargar configuración
        try:
            with open(self.models_dir / "model_config.json", "r", encoding="utf-8") as f:
                config = json.load(f)
            
            print(f"Nombre: {config['model_name']}")
            print(f"Versión: {config['version']}")
            print(f"Entrenado: {config['trained_at']}")
            print(f"Dataset: {config['dataset_size']} ejemplos")
            print(f"Tipo: {config['model_type']}")
            print("Capacidades:")
            for cap in config['capabilities']:
                print(f"  - {cap}")
            print(f"Resultados: {config['training_results']}")
        except:
            print("No se pudo cargar la configuración del modelo")
        
        # Info técnica
        if self.model:
            param_count = sum(p.numel() for p in self.model.parameters())
            print(f"Parámetros: {param_count:,}")
        
        if self.embeddings is not None:
            print(f"Embeddings: {self.embeddings.shape}")
    
    def run_quick_tests(self):
        """Ejecutar pruebas rápidas"""
        print("\n🧪 PRUEBAS RÁPIDAS")
        print("-" * 30)
        
        test_inputs = ["Hola", "¿Cómo estás?", "¿Qué puedes hacer?"]
        
        for test_input in test_inputs:
            response = self.process_message(test_input)
            print(f"Test: {test_input}")
            print(f"Respuesta: {response}")
            print("-" * 30)

def main():
    chat_system = SheliyChatSystem()
    chat_system.start_chat()

if __name__ == "__main__":
    main()