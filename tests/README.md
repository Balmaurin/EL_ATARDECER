# Tests Directory

Suite completa de tests para EL-AMANECERV3.

## Estructura

```
tests/
├── enterprise/              # Tests enterprise de alto nivel
│   ├── test_api_enterprise_suites.py
│   ├── test_blockchain_enterprise.py
│   ├── test_chaos_engineering.py
│   ├── test_consciousness_integration.py
│   ├── test_core_consciousness_suites.py
│   ├── test_database_operations_enterprise.py
│   ├── test_deployment_migration_enterprise.py
│   ├── test_frontend_ui_enterprise.py
│   ├── test_infrastructure_enterprise.py
│   ├── test_integration_end_to_end_enterprise.py
│   ├── test_mcp_integration_enterprise.py
│   ├── test_performance_benchmarks_enterprise.py
│   ├── test_rag_enterprise_system.py
│   ├── test_rag_system_enterprise.py
│   ├── test_security_enterprise.py
│   ├── test_training_system_enterprise.py
│   └── test_auto_improvement_enterprise.py
│
├── integration/            # Tests de integración entre componentes
│   └── test_integrations.py
│
├── diagnostics/            # Tests diagnósticos del sistema
│   └── diagnostico_testing.py
│
├── data/                   # Datos de test
│
└── results/                # Resultados de tests
```

## Ejecutar Tests

### Todos los Tests Enterprise
```bash
python tools/testing/run_all_enterprise_tests.py
```

### Tests Específicos
```bash
# Tests de blockchain
pytest tests/enterprise/test_blockchain_enterprise.py -v

# Tests de consciousness
pytest tests/enterprise/test_consciousness_integration.py -v

# Tests de RAG
pytest tests/enterprise/test_rag_system_enterprise.py -v

# Tests de integración
pytest tests/integration/test_integrations.py -v
```

### Tests por Categoría
```bash
# Solo tests enterprise
pytest tests/enterprise/ -v

# Solo tests de integración
pytest tests/integration/ -v

# Todos los tests
pytest tests/ -v
```

### Con Reportes
```bash
pytest tests/ -v --html=tests/results/report.html --self-contained-html
```

## Diagnósticos

**Test Diagnóstico Rápido**:
```bash
python tests/diagnostics/diagnostico_testing.py
```

## Configuración

- `conftest.py`: Fixtures y configuración global de pytest
- `__init__.py`: Inicialización del paquete de tests

## Notas

- Los tests requieren el entorno virtual activado
- Algunos tests enterprise requieren configuración previa (ver documentación)
- Los tests de integración verifican conexiones entre sistemas críticos

**Última actualización**: 2025-11-27
