#!/usr/bin/env python3
"""
Simple Monitoring System - Sheily AI
====================================

Sistema de monitoreo funcional para métricas críticas, sin datos simulados.
"""

import json
import logging
import os
import sys
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional

import psutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ROOT_DIR = Path(__file__).resolve().parents[2]
for candidate in {
    ROOT_DIR,
    ROOT_DIR / "apps",
    ROOT_DIR / "packages" / "sheily_core" / "src",
}:
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

try:  # noqa: SIM105 - dependencia opcional
    from backend.src.core.config.settings import settings as backend_settings  # type: ignore
except Exception:  # pragma: no cover - backend opcional
    backend_settings = None  # type: ignore

try:  # noqa: SIM105 - dependencia opcional
    from sheily_core.monitoring.real_enterprise_monitor import (  # type: ignore
        get_real_enterprise_monitor,
    )
except Exception:  # pragma: no cover - monitor opcional
    get_real_enterprise_monitor = None  # type: ignore


class SimpleMonitor:
    """Sistema de monitoreo simple pero efectivo, utilizando datos reales."""

    def __init__(
        self,
        metrics_dir: str = "monitoring/metrics",
        request_window_seconds: int = 300,
    ):
        self.metrics_dir = Path(metrics_dir)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.metrics_history: List[Dict[str, Any]] = []
        self.request_window_seconds = max(30, request_window_seconds)

        self.log_file = self._resolve_log_file()
        self._log_position = 0
        self._request_events: Deque[datetime] = deque()
        self._error_events: Deque[datetime] = deque()
        self._response_times: Deque[float] = deque(maxlen=1000)
        self.backend_port = self._detect_backend_port()
        self.backend_host = self._detect_backend_host()
        self._monitor_start = datetime.now(timezone.utc)

        self.enterprise_monitor = (
            get_real_enterprise_monitor(str(self.log_file))
            if self.log_file and get_real_enterprise_monitor
            else None
        )
        self._ensure_log_file()

    def start_monitoring(self, interval_seconds: int = 60):
        """Iniciar monitoreo continuo."""
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return

        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, args=(interval_seconds,), daemon=True
        )
        self.monitor_thread.start()
        logger.info("✅ Monitoring started (interval: %ss)", interval_seconds)

    def stop_monitoring(self):
        """Detener monitoreo."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("🛑 Monitoring stopped")

    def get_current_metrics(self) -> Dict[str, Any]:
        """Obtener métricas actuales con datos reales."""
        system_metrics = self._get_system_metrics()
        application_metrics = self._get_application_metrics()
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "system": system_metrics,
            "application": application_metrics,
            "alerts": self._check_alerts(system_metrics),
        }

    def _monitoring_loop(self, interval: int):
        """Loop principal de monitoreo."""
        while self.is_monitoring:
            try:
                metrics = self.get_current_metrics()
                self.metrics_history.append(metrics)

                if len(self.metrics_history) % 5 == 0:
                    self._save_metrics_snapshot()

                alerts = metrics.get("alerts", [])
                if alerts:
                    self._handle_alerts(alerts)

                time.sleep(interval)

            except Exception as exc:  # pragma: no cover - resiliencia
                logger.error("Monitoring error: %s", exc)
                time.sleep(interval)

    def _get_system_metrics(self) -> Dict[str, Any]:
        """Obtener métricas del sistema usando psutil."""
        return {
            "cpu_percent": psutil.cpu_percent(interval=1),
            "memory_percent": psutil.virtual_memory().percent,
            "memory_used_gb": psutil.virtual_memory().used / (1024**3),
            "memory_total_gb": psutil.virtual_memory().total / (1024**3),
            "disk_usage_percent": psutil.disk_usage("/").percent,
            "disk_free_gb": psutil.disk_usage("/").free / (1024**3),
            "network_connections": len(psutil.net_connections(kind="inet")),
            "load_average": (
                psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
            ),
        }

    def _get_application_metrics(self) -> Dict[str, Any]:
        """Obtener métricas de aplicación a partir de logs y sistema."""
        self._process_log_updates()

        active_connections = self._count_active_connections()
        requests_per_second = self._calculate_requests_per_second()
        error_rate_percent = self._calculate_error_rate_percent()
        response_time_avg_ms = self._calculate_avg_response_time()

        try:
            process_memory_mb = psutil.Process().memory_info().rss / (1024**2)
        except Exception as exc:  # pragma: no cover - psutil edge case
            logger.debug("No se pudo obtener memoria del proceso: %s", exc)
            process_memory_mb = 0.0

        if self.enterprise_monitor is not None:
            try:
                enterprise_error_rate = (
                    self.enterprise_monitor.calculate_error_rate(
                        time_window_seconds=self.request_window_seconds
                    )
                    * 100.0
                )
                error_rate_percent = max(
                    error_rate_percent, round(enterprise_error_rate, 3)
                )
            except Exception as exc:  # pragma: no cover - dependencia opcional
                logger.debug(
                    "No se pudo consultar el monitor empresarial: %s", exc
                )

        return {
            "active_connections": active_connections,
            "requests_per_second": requests_per_second,
            "error_rate_percent": error_rate_percent,
            "response_time_avg_ms": response_time_avg_ms,
            "memory_usage_mb": round(process_memory_mb, 3),
        }

    def _check_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Verificar condiciones de alerta con métricas reales."""
        alerts: List[Dict[str, Any]] = []

        if metrics["cpu_percent"] > 80:
            alerts.append(
                {
                    "level": "WARNING",
                    "type": "HIGH_CPU",
                    "message": f"CPU usage is {metrics['cpu_percent']:.1f}%",
                    "value": metrics["cpu_percent"],
                    "threshold": 80,
                }
            )

        if metrics["memory_percent"] > 85:
            alerts.append(
                {
                    "level": "CRITICAL",
                    "type": "HIGH_MEMORY",
                    "message": f"Memory usage is {metrics['memory_percent']:.1f}%",
                    "value": metrics["memory_percent"],
                    "threshold": 85,
                }
            )

        if metrics["disk_usage_percent"] > 90:
            alerts.append(
                {
                    "level": "WARNING",
                    "type": "LOW_DISK_SPACE",
                    "message": f"Disk usage is {metrics['disk_usage_percent']:.1f}%",
                    "value": metrics["disk_usage_percent"],
                    "threshold": 90,
                }
            )

        return alerts

    def _handle_alerts(self, alerts: List[Dict[str, Any]]):
        """Manejar alertas detectadas."""
        for alert in alerts:
            level = alert["level"]
            message = alert["message"]

            if level == "CRITICAL":
                logger.critical("🚨 CRITICAL ALERT: %s", message)
            elif level == "WARNING":
                logger.warning("⚠️ WARNING ALERT: %s", message)
            else:
                logger.info("ℹ️ INFO ALERT: %s", message)

    def _resolve_log_file(self) -> Optional[Path]:
        """Determinar el archivo de log a monitorear."""
        candidates: List[Path] = []
        env_log = os.getenv("SHEILY_APP_LOG")
        if env_log:
            candidates.append(Path(env_log))
        candidates.extend(
            [
                Path("logs/application.log"),
                Path("logs/sheily_ai.log"),
                Path("logs/monitoring.log"),
            ]
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else None

    def _detect_backend_port(self) -> Optional[int]:
        """Obtener el puerto real configurado para el backend."""
        try:
            if backend_settings is None:
                return None
            port = getattr(backend_settings, "server_port", None)
            if isinstance(port, int):
                return port
            if isinstance(port, str) and port.isdigit():
                return int(port)
        except Exception as exc:  # pragma: no cover - opcional
            logger.debug("No se pudo detectar el puerto del backend: %s", exc)
        return None

    def _detect_backend_host(self) -> Optional[str]:
        """Obtener el host real configurado para el backend."""
        try:
            if backend_settings is None:
                return None
            host = getattr(backend_settings, "server_host", None)
            return host
        except Exception as exc:  # pragma: no cover - opcional
            logger.debug("No se pudo detectar el host del backend: %s", exc)
        return None

    def _ensure_log_file(self):
        """Garantizar que el archivo de logs exista."""
        if not self.log_file:
            return
        try:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            if not self.log_file.exists():
                self.log_file.touch()
        except Exception as exc:  # pragma: no cover - permisos
            logger.debug("No se pudo preparar el archivo de log: %s", exc)

    def _process_log_updates(self):
        """Procesar entradas nuevas del archivo de logs."""
        if not self.log_file:
            return

        try:
            with self.log_file.open("r", encoding="utf-8", errors="ignore") as log_file:
                log_file.seek(self._log_position)
                for line in log_file:
                    self._log_position = log_file.tell()
                    entry = self._parse_log_line(line)
                    if entry is None:
                        continue

                    timestamp = self._parse_timestamp(entry.get("timestamp"))
                    if self._should_count_as_request(entry):
                        self._request_events.append(timestamp)

                    if self._is_error_entry(entry):
                        self._error_events.append(timestamp)

                    response_time = self._extract_response_time(entry)
                    if response_time is not None:
                        self._response_times.append(response_time)
        except FileNotFoundError:
            self._ensure_log_file()
        except Exception as exc:  # pragma: no cover - variación de encoding
            logger.debug("No se pudieron procesar los logs: %s", exc)
        finally:
            self._prune_windows()

    def _parse_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Interpretar una línea de log."""
        line = line.strip()
        if not line:
            return None

        try:
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

        return {"timestamp": datetime.now(timezone.utc).isoformat(), "message": line}

    def _parse_timestamp(self, raw_timestamp: Any) -> datetime:
        """Convertir diferentes formatos de timestamp a UTC."""
        if isinstance(raw_timestamp, datetime):
            return raw_timestamp.astimezone(timezone.utc)
        if isinstance(raw_timestamp, (int, float)):
            return datetime.fromtimestamp(float(raw_timestamp), tz=timezone.utc)
        if isinstance(raw_timestamp, str) and raw_timestamp:
            try:
                normalized = raw_timestamp.replace("Z", "+00:00")
                return datetime.fromisoformat(normalized).astimezone(timezone.utc)
            except ValueError:
                logger.debug("Formato de timestamp no reconocido: %s", raw_timestamp)
        return datetime.now(timezone.utc)

    def _should_count_as_request(self, entry: Dict[str, Any]) -> bool:
        """Determinar si una entrada de log corresponde a una petición."""
        message = str(entry.get("message", "")).lower()
        if "request" in message or "graphql" in message or "endpoint" in message:
            return True

        context = entry.get("context") or {}
        if isinstance(context, dict):
            component = str(context.get("component", "")).lower()
            operation = str(context.get("operation", "")).lower()
            if component in {"api", "backend", "graphql", "llm"}:
                return True
            if operation in {"http_request", "graphql_query", "api_call"}:
                return True

        for key, value in entry.items():
            if key.startswith("extra_") and isinstance(value, str):
                content = value.lower()
                if "request" in content or "graphql" in content or "endpoint" in content:
                    return True

        return False

    def _is_error_entry(self, entry: Dict[str, Any]) -> bool:
        """Determinar si una entrada representa un error."""
        level = str(entry.get("level", "")).upper()
        if level in {"ERROR", "CRITICAL"}:
            return True
        message = str(entry.get("message", "")).lower()
        if "error" in message or "exception" in message or "traceback" in message:
            return True
        return False

    def _extract_response_time(self, entry: Dict[str, Any]) -> Optional[float]:
        """Extraer el tiempo de respuesta reportado en los logs."""
        keys_to_check = [
            "response_time_ms",
            "extra_response_time_ms",
            "latency_ms",
            "duration_ms",
            "response_time",
        ]
        for key in keys_to_check:
            value = entry.get(key)
            if value is None and key.startswith("extra_"):
                value = entry.get(key.replace("extra_", ""))
            if value is None:
                continue
            try:
                value_float = float(value)
                if value_float < 1 and key in {"response_time", "duration_ms"}:
                    value_float *= 1000
                return round(value_float, 3)
            except (TypeError, ValueError):
                continue
        return None

    def _prune_windows(self):
        """Mantener los eventos dentro de la ventana configurada."""
        cutoff = datetime.now(timezone.utc) - timedelta(seconds=self.request_window_seconds)
        while self._request_events and self._request_events[0] < cutoff:
            self._request_events.popleft()
        while self._error_events and self._error_events[0] < cutoff:
            self._error_events.popleft()

    def _calculate_requests_per_second(self) -> float:
        """Calcular solicitudes por segundo utilizando la ventana temporal."""
        if not self._request_events:
            return 0.0
        time_span = (
            self._request_events[-1] - self._request_events[0]
        ).total_seconds()
        effective_window = max(time_span, float(self.request_window_seconds))
        rps = len(self._request_events) / effective_window if effective_window else 0.0
        return round(rps, 3)

    def _calculate_error_rate_percent(self) -> float:
        """Calcular porcentaje de errores sobre solicitudes."""
        total_requests = len(self._request_events)
        if total_requests == 0:
            return 0.0
        error_rate = (len(self._error_events) / total_requests) * 100.0
        return round(error_rate, 3)

    def _calculate_avg_response_time(self) -> float:
        """Calcular promedio de tiempo de respuesta en ms."""
        if not self._response_times:
            return 0.0
        avg_response = sum(self._response_times) / len(self._response_times)
        return round(avg_response, 3)

    def _count_active_connections(self) -> int:
        """Contar conexiones activas hacia los servicios monitoreados."""
        ports_to_watch = {port for port in [self.backend_port, 8000, 8003, 8080, 9000] if port}
        if not ports_to_watch:
            return 0

        try:
            connections = psutil.net_connections(kind="inet")
        except Exception as exc:  # pragma: no cover - permisos
            logger.debug("No se pudieron leer las conexiones de red: %s", exc)
            return 0

        active_states = {"ESTABLISHED", "SYN_SENT", "SYN_RECV"}
        active_count = 0
        for connection in connections:
            laddr = getattr(connection, "laddr", None)
            if not laddr:
                continue
            if laddr.port in ports_to_watch and (
                not connection.status or connection.status in active_states
            ):
                active_count += 1
        return active_count

    def _save_metrics_snapshot(self):
        """Guardar snapshot de métricas."""
        if not self.metrics_history:
            return

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = f"metrics_snapshot_{timestamp}.json"
        filepath = self.metrics_dir / filename

        recent_metrics = (
            self.metrics_history[-10:]
            if len(self.metrics_history) > 10
            else self.metrics_history
        )

        with filepath.open("w", encoding="utf-8") as snapshot_file:
            json.dump(
                {
                    "snapshot_time": datetime.now(timezone.utc).isoformat(),
                    "metrics_count": len(recent_metrics),
                    "metrics": recent_metrics,
                },
                snapshot_file,
                indent=2,
                default=str,
            )

        self._cleanup_old_snapshots()

    def _cleanup_old_snapshots(self, keep_last: int = 10):
        """Limpiar snapshots antiguos dejando solo los más recientes."""
        snapshots = sorted(self.metrics_dir.glob("metrics_snapshot_*.json"))
        if len(snapshots) > keep_last:
            for old_snapshot in snapshots[:-keep_last]:
                old_snapshot.unlink(missing_ok=True)

    def get_metrics_report(self) -> Dict[str, Any]:
        """Generar reporte completo de métricas."""
        current = self.get_current_metrics()

        if self.metrics_history:
            recent_history = self.metrics_history[-60:]
            cpu_values = [m["system"]["cpu_percent"] for m in recent_history]
            memory_values = [m["system"]["memory_percent"] for m in recent_history]

            stats = {
                "cpu_avg": sum(cpu_values) / len(cpu_values) if cpu_values else 0.0,
                "cpu_max": max(cpu_values) if cpu_values else 0.0,
                "memory_avg": (
                    sum(memory_values) / len(memory_values) if memory_values else 0.0
                ),
                "memory_max": max(memory_values) if memory_values else 0.0,
                "total_measurements": len(self.metrics_history),
                "uptime_seconds": (
                    datetime.now(timezone.utc) - self._monitor_start
                ).total_seconds(),
            }
        else:
            stats = {"message": "No historical data available"}

        return {
            "current_metrics": current,
            "historical_stats": stats,
            "alerts_active": len(current.get("alerts", [])),
            "monitoring_status": "active" if self.is_monitoring else "inactive",
        }


monitor = SimpleMonitor()


def start_monitoring(interval_seconds: int = 60):
    """Función de conveniencia para iniciar monitoreo."""
    monitor.start_monitoring(interval_seconds)


def stop_monitoring():
    """Función de conveniencia para detener monitoreo."""
    monitor.stop_monitoring()


def get_metrics():
    """Función de conveniencia para obtener métricas."""
    return monitor.get_metrics_report()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Simple Monitoring System - Sheily AI"
    )
    parser.add_argument("--start", action="store_true", help="Start monitoring")
    parser.add_argument("--stop", action="store_true", help="Stop monitoring")
    parser.add_argument("--status", action="store_true", help="Show monitoring status")
    parser.add_argument(
        "--interval", type=int, default=60, help="Monitoring interval in seconds"
    )

    args = parser.parse_args()

    if args.start:
        print(f"🚀 Starting monitoring (interval: {args.interval}s)...")
        start_monitoring(args.interval)
        print("✅ Monitoring started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitoring...")
            stop_monitoring()
            print("✅ Monitoring stopped.")

    elif args.stop:
        print("🛑 Stopping monitoring...")
        stop_monitoring()
        print("✅ Monitoring stopped.")

    elif args.status:
        report = get_metrics()
        print("📊 MONITORING STATUS")
        print("=" * 30)
        print(f"Status: {report['monitoring_status']}")
        print(f"Active Alerts: {report['alerts_active']}")
        print(
            f"Total Measurements: {report['historical_stats'].get('total_measurements', 0)}"
        )

        current = report["current_metrics"]["system"]
        print("\n📈 CURRENT METRICS:")
        print(f"CPU: {current['cpu_percent']:.1f}%")
        print(f"Memory: {current['memory_percent']:.1f}%")
        print(f"Disk: {current['disk_usage_percent']:.1f}%")

    else:
        parser.print_help()
