# Tools Directory - Estructura Completa

Herramientas organizadas por categoría para desarrollo, análisis, auditoría y mantenimiento.

## Estructura Detallada

```
tools/
├── ai/                         # Herramientas de IA
├── analysis/                   # Análisis de contenido y conocimiento
│   ├── analizar_conocimiento.py
│   ├── analyze_pdfs.py
│   └── ...
│
├── audit/                      # Auditoría de proyecto
│   ├── audit_enterprise_project.py  # ✅ Auditoría de Calidad/QA
│   ├── audit_project_structure.py   # ✅ Auditoría Estructural (antes en maintenance)
│   └── ...
│
├── automation/                 # Automatización
├── backup/                     # Backup y restauración
├── common/                     # Utilidades comunes
│
├── consciousness/              # Consciencia
│   ├── check_self_awareness.py
│   └── __init__.py
│
├── correctors/                 # Correctores de código
├── dependency_manager/         # Gestión de dependencias
├── deployment/                 # Despliegue
│   ├── deployment_manager.py
│   └── quick_start.py
│
├── development/                # Desarrollo
├── generators/                 # Generadores de código
│
├── llama_cpp/                  # Binarios llama.cpp
│
├── maintenance/                # Mantenimiento
│   ├── analyze_scripts_utility.py
│   ├── compute_project_metrics.py
│   └── ...
│
├── monitoring/                 # Monitorización
├── patches/                    # Parches del sistema
├── precommit/                  # Hooks pre-commit
│
├── rewards/                    # Sistema de Recompensas Sheily (antes tools/sheily)
│   ├── sheily_rewards.py
│   └── __init__.py
│
├── security/                   # Seguridad
├── solvers/                    # Solucionadores
│
├── testing/                    # Testing
│   ├── run_all_enterprise_tests.py
│   ├── fix_test_files.py
│   └── ...
│
├── utils/                      # Utilidades generales
└── validators/                 # Validadores
```

## Categorización Funcional

### 📊 Análisis y Auditoría
- `analysis/` - Análisis de contenido (PDFs, conocimiento, pesos neuronales)
- `audit/` - Auditoría completa (Estructural y QA)
- `maintenance/` - Métricas de código y limpieza

### 🧪 Testing y Validación
- `testing/` - Suite de tests enterprise
- `validators/` - Validadores de código/datos
- `solvers/` - Solucionadores de problemas

### 🚀 Deployment y Automatización
- `deployment/` - Gestión de despliegues
- `automation/` - Scripts de automatización
- `precommit/` - Hooks de Git

### 🧠 Específicos de Sheily
- `consciousness/` - Verificación de consciencia
- `rewards/` - Sistema de recompensas y gamificación
- `ai/` - Herramientas de IA
- `llama_cpp/` - Binarios del modelo LLM

### 🔒 Infraestructura
- `security/` - Seguridad y encriptación
- `backup/` - Backup y restauración
- `monitoring/` - Monitorización del sistema
- `dependency_manager/` - Gestión de dependencias

## Uso Rápido

### Auditoría
```bash
# Auditoría de Calidad (QA, Tests, Seguridad)
python tools/audit/audit_enterprise_project.py

# Auditoría Estructural (Archivos, Tamaños, Ramas)
python tools/audit/audit_project_structure.py
```

### Testing
```bash
# Todos los tests enterprise
python tools/testing/run_all_enterprise_tests.py

# Reparar tests
python tools/testing/fix_test_files.py
```

### Recompensas
```bash
# Demo interactiva de recompensas
python tools/rewards/sheily_rewards.py
```

## Cambios Recientes

- **Movido**: `tools/launchers` → `scripts/launchers`
- **Renombrado**: `tools/sheily` → `tools/rewards`
- **Consolidado**: `tools/maintenance/audit_complete_project.py` → `tools/audit/audit_project_structure.py`
- **Eliminado**: `tools/data` (vacío), `tools/n8n` (vacío)

**Última actualización**: 2025-11-27
