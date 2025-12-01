# Config Directory

Configuración centralizada del proyecto EL-AMANECERV3.

## Estructura

```
config/
├── ai/                         # Configuración de IA
│   ├── rag/                   # Configuración RAG (embeddings, chunks)
│   └── training/              # Configuración de entrenamiento y fine-tuning
│
├── core/                       # Configuración Core
│   ├── settings.py            # Configuración unificada (env vars, flags)
│   ├── sheily_constitution.yml # Constitución y reglas base de Sheily
│   └── universal.yaml         # Constantes universales
│
├── development/                # Configuración de desarrollo
│   ├── git/                   # Git hooks, branches
│   └── python/                # Requirements, linters, makefile
│
├── blockchain/                 # Configuración Blockchain (Solana)
├── database/                   # Configuración de Base de Datos
├── enterprise/                 # Configuración Enterprise
├── infrastructure/             # Infraestructura (Docker, Cloud)
└── security/                   # Políticas de seguridad
```

## Archivos Clave

### `core/settings.py`
La fuente de verdad para la configuración del sistema. Define:
- Variables de entorno
- Feature flags (Chat, RAG, Consciousness, etc.)
- Rutas de modelos LLM
- Conexiones a base de datos y Redis

### `ai/training/finetuning_config.json`
Configuración para el entrenamiento de modelos, incluyendo rutas de datasets y parámetros de LoRA.

### `development/python/requirements-*.txt`
Dependencias del proyecto divididas por área (dev, ci, rag).

## Uso

Para modificar la configuración del sistema, edite `config/core/settings.py` o use variables de entorno en un archivo `.env` en la raíz del proyecto.

**Última actualización**: 2025-11-27
