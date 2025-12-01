#!/usr/bin/env python3
"""
SISTEMA DE VALIDACIÓN Y PRUEBA DEL MODELO ENTRENADO
===================================================
Prueba completa del modelo Sheily entrenado
"""

import json
import torch
import numpy as np
import pickle
from pathlib import Path
from transformers import GPT2LMHeadModel, GPT2Tokenizer

class ModelTester:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.models_dir = self.project_root / "models"
        
    def load_embeddings_model(self):
        """Cargar modelo de embeddings entrenado"""
        try:
            # Cargar vectorizador y SVD
            with open(self.models_dir / "vectorizer.pkl", "rb") as f:
                vectorizer = pickle.load(f)
            
            with open(self.models_dir / "svd.pkl", "rb") as f:
                svd = pickle.load(f)
            
            # Cargar embeddings
            embeddings = np.load(self.models_dir / "embeddings.npy")
            
            print(f"✅ Embeddings cargados: {embeddings.shape}")
            return vectorizer, svd, embeddings
        
        except Exception as e:
            print(f"❌ Error cargando embeddings: {e}")
            return None, None, None
    
    def test_embeddings(self, vectorizer, svd):
        """Probar embeddings con nuevas frases"""
        test_phrases = [
            "Hola como estas",
            "Necesito ayuda",
            "Que puedes hacer",
            "Gracias por todo"
        ]
        
        print("\n=== PRUEBA DE EMBEDDINGS ===")
        for phrase in test_phrases:
            try:
                # Convertir texto a vector TF-IDF
                tfidf_vector = vectorizer.transform([phrase])
                # Reducir dimensiones
                embedding = svd.transform(tfidf_vector)
                
                print(f"Frase: '{phrase}'")
                print(f"Embedding shape: {embedding.shape}")
                print(f"Valores (primeros 5): {embedding[0][:5]}")
                print("-" * 40)
                
            except Exception as e:
                print(f"Error procesando '{phrase}': {e}")
    
    def load_gpt2_model(self):
        """Cargar modelo GPT2 entrenado"""
        try:
            model_path = self.models_dir / "sheily_simple"
            if model_path.exists():
                tokenizer = GPT2Tokenizer.from_pretrained(str(model_path))
                model = GPT2LMHeadModel.from_pretrained(str(model_path))
                
                print(f"✅ Modelo GPT2 cargado desde: {model_path}")
                return tokenizer, model
            else:
                print("❌ Modelo GPT2 no encontrado")
                return None, None
                
        except Exception as e:
            print(f"❌ Error cargando modelo GPT2: {e}")
            return None, None
    
    def test_gpt2_model(self, tokenizer, model):
        """Probar modelo GPT2 con conversación"""
        test_inputs = [
            "Pregunta: Hola",
            "Pregunta: Como estas?",
            "Pregunta: Que eres?",
            "Pregunta: Ayudame",
            "Pregunta: Gracias"
        ]
        
        print("\n=== PRUEBA DEL MODELO GPT2 ===")
        
        for test_input in test_inputs:
            try:
                # Tokenizar entrada
                inputs = tokenizer.encode(test_input, return_tensors="pt")
                
                # Generar respuesta
                with torch.no_grad():
                    outputs = model.generate(
                        inputs,
                        max_length=100,
                        num_return_sequences=1,
                        pad_token_id=tokenizer.eos_token_id,
                        do_sample=True,
                        temperature=0.7,
                        top_p=0.9
                    )
                
                # Decodificar respuesta
                response = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                print(f"Entrada: {test_input}")
                print(f"Respuesta: {response}")
                print("-" * 50)
                
            except Exception as e:
                print(f"Error generando respuesta para '{test_input}': {e}")
    
    def performance_benchmark(self, tokenizer, model):
        """Benchmark de rendimiento del modelo"""
        print("\n=== BENCHMARK DE RENDIMIENTO ===")
        
        import time
        
        test_text = "Pregunta: Explica la inteligencia artificial"
        inputs = tokenizer.encode(test_text, return_tensors="pt")
        
        # Medir tiempo de inferencia
        start_time = time.time()
        
        with torch.no_grad():
            for i in range(10):
                outputs = model.generate(
                    inputs,
                    max_length=50,
                    pad_token_id=tokenizer.eos_token_id
                )
        
        end_time = time.time()
        avg_time = (end_time - start_time) / 10
        
        print(f"Tiempo promedio por inferencia: {avg_time:.3f} segundos")
        print(f"Inferencias por segundo: {1/avg_time:.2f}")
        
        # Medir tamaño del modelo
        model_size = sum(p.numel() for p in model.parameters())
        print(f"Parámetros del modelo: {model_size:,}")
        
        # Memoria utilizada
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**2  # MB
            print(f"Memoria GPU utilizada: {memory_used:.2f} MB")
        else:
            print("Ejecutándose en CPU")
    
    def run_complete_test(self):
        """Ejecutar prueba completa del sistema"""
        print("🧠 SISTEMA DE VALIDACIÓN - SHEILY V1 REAL")
        print("=" * 60)
        
        success_count = 0
        total_tests = 3
        
        # 1. Probar embeddings
        print("\n1️⃣ PROBANDO EMBEDDINGS...")
        vectorizer, svd, embeddings = self.load_embeddings_model()
        if vectorizer and svd is not None:
            self.test_embeddings(vectorizer, svd)
            success_count += 1
            print("✅ Embeddings: FUNCIONAL")
        else:
            print("❌ Embeddings: FALLO")
        
        # 2. Probar modelo GPT2
        print("\n2️⃣ PROBANDO MODELO GPT2...")
        tokenizer, model = self.load_gpt2_model()
        if tokenizer and model:
            self.test_gpt2_model(tokenizer, model)
            success_count += 1
            print("✅ Modelo GPT2: FUNCIONAL")
        else:
            print("❌ Modelo GPT2: FALLO")
        
        # 3. Benchmark de rendimiento
        if tokenizer and model:
            print("\n3️⃣ BENCHMARK DE RENDIMIENTO...")
            self.performance_benchmark(tokenizer, model)
            success_count += 1
            print("✅ Benchmark: COMPLETADO")
        
        # Resumen final
        print("\n" + "=" * 60)
        print("📊 RESUMEN DE VALIDACIÓN")
        print("=" * 60)
        print(f"Tests ejecutados: {total_tests}")
        print(f"Tests exitosos: {success_count}")
        print(f"Tasa de éxito: {(success_count/total_tests)*100:.1f}%")
        
        if success_count >= 2:
            print("🎉 MODELO SHEILY V1 COMPLETAMENTE FUNCIONAL")
            return True
        else:
            print("⚠️  MODELO CON FUNCIONALIDAD LIMITADA")
            return False

def main():
    tester = ModelTester()
    return tester.run_complete_test()

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)