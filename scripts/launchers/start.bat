@echo off
REM Script para iniciar Sheily Backend correctamente
echo.
echo ========================================
echo SHEILY BACKEND - INICIO
echo ========================================
echo.
echo Puerto: 8001
echo Docs: http://localhost:8001/docs
echo Health: http://localhost:8001/health
echo.

REM Iniciar usando uvicorn CLI (no uvicorn.run)
py -m uvicorn backend.main_api:app --host 0.0.0.0 --port 8001 --workers 1
