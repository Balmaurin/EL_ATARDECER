#!/usr/bin/env python3
"""
Test script para verificar que el sistema completo token-provisional + training real funcione
"""

import asyncio
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_real_training_integration():
    """Prueba completa del sistema: tokens provisorios -> training real"""

    logger.info("🧪 Iniciando TEST COMPLETO DE INTEGRACIÓN REAL...")

    try:
        # 1. Importar servicios
        from apps.backend.hack_memori_service import HackMemoriService
        from scripts.core.execute_real_evolution_cycle import EvolutionOrchestrator

        # 2. Crear servicio HACK-MEMORI
        service = HackMemoriService()
        logger.info("✅ HackMemoriService creado")

        # 3. Crear sesión de prueba
        session = service.create_session(
            name="Test Real Training Integration",
            user_id="1",
            config={
                "auto_generate": False,  # Desactivar auto-generación
                "max_questions": 100,
                "test_mode": True
            }
        )
        session_id = session["id"]
        logger.info(f"✅ Sesión de prueba creada: {session_id}")

        # 4. Simular 100 Q&A pairs directamente (en lugar de esperar auto-generación)
        logger.info("⚡ Simulando 100 Q&A pairs para llegar al threshold...")

        for i in range(100):
            # Agregar pregunta
            question_text = f"Pregunta de prueba #{i}: ¿Cómo se implanta consciencia real en IA?"
            question = service.add_question(session_id, question_text)
            q_id = question["id"]

            # Agregar respuesta aceptada (que da tokens)
            response_text = f"Respuesta detallada a la pregunta {i} sobre consciencia en IA..."
            service.add_response(
                q_id, session_id, "test-model-v1",
                question_text, response_text, 50  # 50 tokens usados
            )

            if i % 20 == 0:
                logger.info(f"📊 Progreso: {i+1}/100 Q&A generados")

        # 5. Verificar que se alcanzaron 100 Q&A y se activó training
        qa_count = service._get_session_qa_count(session_id)
        logger.info(f"✅ Total Q&A en sesión {session_id}: {qa_count}")

        # El training debería activarse automáticamente cuando llegue a 100 Q&A
        # (O ocurrio en el paso anterior)

        # 6. Esperar un poco por si hay delays asincrónicos
        await asyncio.sleep(2)

        # 7. Verificar estado de tokens después del training
        try:
            import sqlite3
            db_path = Path("data/sheily_dashboard.db")
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()

            cursor.execute("SELECT provisional_tokens, tokens FROM users WHERE id = 1")
            result = cursor.fetchone()
            conn.close()

            provisional_tokens = result[0] if result else 0
            total_tokens = result[1] if result else 0

            logger.info(f"💰 Estado de tokens usuario 1:")
            logger.info(f"   - Tokens provisorios: {provisional_tokens}")
            logger.info(f"   - Tokens totales: {total_tokens}")

            # Trasferidos = training completado exitosamente
            total_transferred = total_tokens - provisional_tokens
            logger.info(f"   - Tokens transferidos: {total_transferred}")

        except Exception as e:
            logger.error(f"❌ Error verificando tokens: {e}")

        # 8. Verificar que el EvolutionOrchestrator ejecutó algo
        evolution_state_path = Path("data/evolution_state/evolution_state.json")
        if evolution_state_path.exists():
            logger.info("✅ Evolution state encontrado - Training real ejecutado")
        else:
            logger.warning("⚠️ Evolution state no encontrado")

        # 9. Verificar logs de evolución
        evolution_log_path = Path("evolution_cycle.log")
        if evolution_log_path.exists():
            logger.info("✅ Logs de evolución encontrados")
            # Leer últimas líneas
            with open(evolution_log_path, 'r') as f:
                lines = f.readlines()[-10:]  # Últimas 10 líneas
                logger.info("📝 Últimas líneas del log de evolución:")
                for line in lines:
                    logger.info(f"   {line.strip()}")
        else:
            logger.warning("⚠️ Logs de evolución no encontrados")

        logger.info("🎉 TEST COMPLETO FINALIZADO")

        return {
            "session_id": session_id,
            "qa_count": qa_count,
            "provisional_tokens": provisional_tokens,
            "total_tokens": total_tokens,
            "transferred": total_transferred
        }

    except Exception as e:
        logger.error(f"❌ TEST FALLÓ: {e}")
        raise

if __name__ == "__main__":
    result = asyncio.run(test_real_training_integration())
    print("\n" + "="*50)
    print("RESULTADOS DEL TEST:")
    print(f"  Session ID: {result.get('session_id', 'ERROR')}")
    print(f"  Q&A Count: {result.get('qa_count', 'ERROR')}")
    print(f"  Provisional Tokens: {result.get('provisional_tokens', 'ERROR')}")
    print(f"  Total Tokens: {result.get('total_tokens', 'ERROR')}")
    print(f"  Tokens Transferidos: {result.get('transferred', 'ERROR')}")
    print("="*50)
