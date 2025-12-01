#!/usr/bin/env python3
"""
Advanced Logging ELK Stack Enterprise Integration
=================================================

Sistema avanzado de logging con integración completa ELK Stack:
- Elasticsearch para indexación y búsqueda de logs
- Logstash para procesamiento y transformación de logs
- Kibana para dashboards y visualización
- Análisis automático de logs con ML/AI
- Correlación de eventos de seguridad
- Alertas inteligentes y notificaciones
- Métricas de rendimiento del sistema
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import socket
import threading
import time
import uuid
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

# Configuración de logging base
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("elk-logger")

try:
    import elasticsearch
    from elasticsearch import Elasticsearch, helpers

    ELASTICSEARCH_AVAILABLE = True
except ImportError:
    ELASTICSEARCH_AVAILABLE = False
    Elasticsearch = None
    helpers = None

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    aiohttp = None


@dataclass
class LogEntry:
    """Entrada de log estructurada para ELK"""

    timestamp: str
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    thread_name: str
    process_id: int
    hostname: str
    service_name: str
    service_version: str
    environment: str
    correlation_id: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    request_id: Optional[str] = None
    duration_ms: Optional[float] = None
    response_code: Optional[int] = None
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    security_event: Optional[str] = None
    anomaly_score: Optional[float] = None
    custom_fields: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertir a diccionario para Elasticsearch"""
        data = asdict(self)
        # Convertir campos None a vacío para Elasticsearch
        for key, value in data.items():
            if value is None:
                data[key] = ""
            elif isinstance(value, datetime):
                data[key] = value.isoformat()
        return data

    def to_json(self) -> str:
        """Convertir a JSON"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class ELKStackManager:
    """Gestor completo de integración ELK Stack"""

    def __init__(
        self,
        elasticsearch_hosts: List[str] = None,
        logstash_endpoint: str = None,
        kibana_url: str = None,
        service_name: str = "sheily_ai",
        service_version: str = "2.0",
    ):

        # Configuración ELK
        self.elasticsearch_hosts = elasticsearch_hosts or ["http://localhost:9200"]
        self.logstash_endpoint = logstash_endpoint or "http://localhost:5044"
        self.kibana_url = kibana_url or "http://localhost:5601"

        # Configuración del servicio
        self.service_name = service_name
        self.service_version = service_version
        self.hostname = socket.gethostname()
        self.environment = os.getenv("ENVIRONMENT", "development")

        # Conexiones
        self.es_client = None
        self.logstash_session = None

        # Buffers y caches
        self.log_buffer = deque(maxlen=10000)  # Buffer circular para logs
        self.error_patterns = defaultdict(int)
        self.performance_metrics = defaultdict(list)
        self.security_events = deque(maxlen=1000)

        # Análisis ML
        self.anomaly_detector = None
        self.baseline_metrics = {}
        self.correlation_rules = self._load_correlation_rules()

        # Inicialización
        self._initialize_connections()

        print("🔍 ELK Stack Manager inicializado")

    def _initialize_connections(self):
        """Inicializar conexiones a ELK Stack"""
        # Elasticsearch
        if ELASTICSEARCH_AVAILABLE:
            try:
                self.es_client = Elasticsearch(self.elasticsearch_hosts)
                if self.es_client.ping():
                    print(f"✅ Conectado a Elasticsearch: {self.elasticsearch_hosts}")
                    self._create_indices()
                else:
                    print("⚠️ Elasticsearch no responde")
            except Exception as e:
                print(f"⚠️ Error conectando a Elasticsearch: {e}")

        # Logstash (HTTP input)
        if AIOHTTP_AVAILABLE:
            try:
                # Nota: En producción, configurar Logstash con HTTP input plugin
                print("✅ Logstash HTTP endpoint configurado")
            except Exception as e:
                print(f"⚠️ Error configurando Logstash: {e}")

    def _create_indices(self):
        """Crear índices en Elasticsearch"""
        if not self.es_client:
            return

        indices_config = {
            "logs-application": {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "level": {"type": "keyword"},
                        "logger_name": {"type": "keyword"},
                        "message": {"type": "text"},
                        "module": {"type": "keyword"},
                        "correlation_id": {"type": "keyword"},
                        "service_name": {"type": "keyword"},
                        "anomaly_score": {"type": "float"},
                        "duration_ms": {"type": "float"},
                        "response_code": {"type": "integer"},
                    }
                }
            },
            "logs-security": {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "security_event": {"type": "keyword"},
                        "anomaly_score": {"type": "float"},
                        "error_type": {"type": "keyword"},
                        "user_id": {"type": "keyword"},
                    }
                }
            },
            "logs-performance": {
                "mappings": {
                    "properties": {
                        "timestamp": {"type": "date"},
                        "metric_name": {"type": "keyword"},
                        "value": {"type": "float"},
                        "service_name": {"type": "keyword"},
                        "correlation_id": {"type": "keyword"},
                    }
                }
            },
        }

        for index_name, config in indices_config.items():
            try:
                if not self.es_client.indices.exists(index=index_name):
                    self.es_client.indices.create(index=index_name, body=config)
                    print(f"✅ Índice creado: {index_name}")
            except Exception as e:
                print(f"⚠️ Error creando índice {index_name}: {e}")

    def _load_correlation_rules(self) -> List[Dict[str, Any]]:
        """Cargar reglas de correlación de eventos"""
        return [
            {
                "name": "multiple_auth_failures",
                "pattern": {"error_type": "authentication_failed"},
                "time_window": 300,  # 5 minutos
                "threshold": 5,
                "severity": "high",
            },
            {
                "name": "slow_performance_spike",
                "pattern": {"duration_ms": ">1000"},
                "time_window": 60,  # 1 minuto
                "threshold": 10,
                "severity": "medium",
            },
            {
                "name": "security_violations_burst",
                "pattern": {"security_event": "csp_violation"},
                "time_window": 600,  # 10 minutos
                "threshold": 20,
                "severity": "critical",
            },
        ]

    def log(self, level: str, message: str, **kwargs) -> None:
        """Registrar entrada de log con contexto estructurado"""
        log_entry = self._create_log_entry(level, message, **kwargs)
        self.log_buffer.append(log_entry)

        # Análisis inmediato para alertas
        self._analyze_log_entry(log_entry)

        # Enviar a ELK de forma asíncrona
        asyncio.create_task(self._send_to_elk(log_entry))

        # Mantener buffer en límites
        if len(self.log_buffer) >= 1000:
            asyncio.create_task(self._flush_buffer())

    def _create_log_entry(self, level: str, message: str, **kwargs) -> LogEntry:
        """Crear entrada de log estructurada"""
        # Obtener información de contexto (frame inspection)
        frame = None
        try:
            import inspect

            frame = inspect.currentframe().f_back.f_back
        except:
            pass

        module = kwargs.get("module", "")
        function = kwargs.get("function", "")
        line_number = kwargs.get("line_number", 0)

        if frame:
            module = module or frame.f_globals.get("__name__", "")
            function = function or frame.f_code.co_name
            line_number = line_number or frame.f_lineno

        # Generar correlation ID si no existe
        correlation_id = kwargs.get("correlation_id", str(uuid.uuid4()))

        return LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level.upper(),
            logger_name=kwargs.get("logger_name", "elk-logger"),
            message=message,
            module=module,
            function=function,
            line_number=line_number,
            thread_id=threading.get_ident(),
            thread_name=threading.current_thread().name,
            process_id=os.getpid(),
            hostname=self.hostname,
            service_name=self.service_name,
            service_version=self.service_version,
            environment=self.environment,
            correlation_id=correlation_id,
            session_id=kwargs.get("session_id"),
            user_id=kwargs.get("user_id"),
            request_id=kwargs.get("request_id"),
            duration_ms=kwargs.get("duration_ms"),
            response_code=kwargs.get("response_code"),
            error_type=kwargs.get("error_type"),
            error_message=kwargs.get("error_message"),
            stack_trace=kwargs.get("stack_trace"),
            security_event=kwargs.get("security_event"),
            anomaly_score=kwargs.get("anomaly_score"),
            custom_fields=kwargs.get("custom_fields", {}),
        )

    def _analyze_log_entry(self, log_entry: LogEntry):
        """Análisis inmediato de entrada de log para alertas"""
        # Contar patrones de error
        if log_entry.level in ["ERROR", "CRITICAL"]:
            error_key = f"{log_entry.error_type}:{log_entry.function}"
            self.error_patterns[error_key] += 1

        # Detectar anomalías
        if log_entry.level == "ERROR":
            anomaly_score = self._calculate_anomaly_score(log_entry)
            if anomaly_score > 0.8:  # Umbral alto
                log_entry.anomaly_score = anomaly_score
                asyncio.create_task(self._trigger_security_alert(log_entry))

        # Métricas de rendimiento
        if log_entry.duration_ms:
            self.performance_metrics[log_entry.function].append(log_entry.duration_ms)

            # Mantener solo últimas 100 mediciones por función
            if len(self.performance_metrics[log_entry.function]) > 100:
                self.performance_metrics[log_entry.function] = self.performance_metrics[
                    log_entry.function
                ][-100:]

    def _calculate_anomaly_score(self, log_entry: LogEntry) -> float:
        """Calcular score de anomalía basado en patrones normales"""
        # Versión simplificada - en producción usar ML
        anomaly_factors = 0
        total_factors = 5

        # Factor 1: Frecuencia de error inusual
        error_rate = self.error_patterns.get(
            f"{log_entry.error_type}:{log_entry.function}", 0
        )
        if error_rate > 10:  # Más de 10 errores del mismo tipo/función
            anomaly_factors += 1

        # Factor 2: Hora inusual del log
        hour = datetime.fromisoformat(log_entry.timestamp).hour
        if hour in [2, 3, 4, 5]:  # Horas de madrugada
            anomaly_factors += 0.5

        # Factor 3: Mensaje inusual
        suspicious_keywords = ["hack", "exploit", "breach", "attack", "unauthorized"]
        if any(keyword in log_entry.message.lower() for keyword in suspicious_keywords):
            anomaly_factors += 1

        # Factor 4: Error de seguridad
        if log_entry.security_event:
            anomaly_factors += 0.8

        # Factor 5: Duración inusual
        if log_entry.duration_ms and log_entry.duration_ms > 5000:  # Más de 5 segundos
            anomaly_factors += 0.5

        return min(anomaly_factors / total_factors, 1.0)

    async def _trigger_security_alert(self, log_entry: LogEntry):
        """Disparar alerta de seguridad"""
        alert = {
            "alert_type": "anomaly_detected",
            "severity": "high" if log_entry.anomaly_score > 0.9 else "medium",
            "timestamp": log_entry.timestamp,
            "service": log_entry.service_name,
            "message": f"Anomalía detectada: {log_entry.message}",
            "anomaly_score": log_entry.anomaly_score,
            "details": log_entry.to_dict(),
        }

        # Agregar a eventos de seguridad
        self.security_events.append(alert)

        logger.warning(
            f"🚨 ALERTA DE SEGURIDAD: {alert['message']} (Score: {alert['anomaly_score']:.2f})"
        )

        # Enviar notificación (webhook, email, etc.)
        await self._send_alert_notification(alert)

    async def _send_to_elk(self, log_entry: LogEntry):
        """Enviar log a ELK Stack"""
        try:
            # Elasticsearch
            if self.es_client:
                index_name = self._get_index_name(log_entry)
                doc = log_entry.to_dict()

                # Añadir @timestamp para Kibana
                doc["@timestamp"] = doc["timestamp"]

                response = self.es_client.index(index=index_name, document=doc)
                if response.get("result") != "created":
                    print(f"⚠️ Error indexando en Elasticsearch: {response}")

            # Logstash (via HTTP)
            if AIOHTTP_AVAILABLE and self.logstash_endpoint:
                try:
                    async with aiohttp.ClientSession() as session:
                        await session.post(
                            self.logstash_endpoint,
                            json=log_entry.to_dict(),
                            headers={"Content-Type": "application/json"},
                        )
                except Exception as e:
                    print(f"⚠️ Error enviando a Logstash: {e}")

        except Exception as e:
            print(f"❌ Error enviando a ELK: {e}")

    def _get_index_name(self, log_entry: LogEntry) -> str:
        """Determinar nombre de índice basado en tipo de log"""
        base_name = f"logs-{self.service_name}-{datetime.now().strftime('%Y.%m.%d')}"

        if log_entry.security_event or log_entry.anomaly_score:
            return f"{base_name}-security"
        elif log_entry.duration_ms:
            return f"{base_name}-performance"
        else:
            return f"{base_name}-application"

    async def _flush_buffer(self):
        """Enviar buffer completo a Elasticsearch usando bulk"""
        if not self.es_client or not self.log_buffer:
            return

        try:
            actions = []
            for log_entry in self.log_buffer:
                index_name = self._get_index_name(log_entry)
                actions.append({"_index": index_name, "_source": log_entry.to_dict()})

            if actions:
                success, failed = helpers.bulk(self.es_client, actions, stats_only=True)
                if failed > 0:
                    print(f"⚠️ {failed} logs fallaron en bulk insert")

            # Limpiar buffer
            self.log_buffer.clear()

        except Exception as e:
            print(f"❌ Error en bulk flush: {e}")

    def correlate_events(self) -> List[Dict[str, Any]]:
        """Correlacionar eventos basado en reglas definidas"""
        correlations = []

        for rule in self.correlation_rules:
            matches = self._find_correlation_matches(rule)
            if len(matches) >= rule["threshold"]:
                correlation = {
                    "rule_name": rule["name"],
                    "severity": rule["severity"],
                    "matches_count": len(matches),
                    "time_window": f"{rule['time_window']}s",
                    "affected_items": matches[:10],  # Top 10
                    "timestamp": datetime.now().isoformat(),
                }
                correlations.append(correlation)

        return correlations

    def _find_correlation_matches(self, rule: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Encontrar coincidencias para una regla de correlación"""
        matches = []
        cutoff_time = datetime.now() - timedelta(seconds=rule["time_window"])

        pattern = rule["pattern"]

        # Buscar en log buffer
        for log_entry in self.log_buffer:
            if datetime.fromisoformat(log_entry.timestamp) < cutoff_time:
                continue

            match = True
            for key, condition in pattern.items():
                value = getattr(log_entry, key, None)

                if isinstance(condition, str) and condition.startswith(">"):
                    if not value or value <= float(condition[1:]):
                        match = False
                        break
                elif value != condition:
                    match = False
                    break

            if match:
                matches.append(log_entry.to_dict())

        return matches

    def get_performance_report(self) -> Dict[str, Any]:
        """Generar reporte de rendimiento del sistema"""
        report = {
            "generated_at": datetime.now().isoformat(),
            "service_name": self.service_name,
            "environment": self.environment,
            "time_range": "last_24h",
        }

        # Métricas de rendimiento por función
        performance_data = {}
        for function_name, durations in self.performance_metrics.items():
            if durations:
                performance_data[function_name] = {
                    "count": len(durations),
                    "avg_duration": sum(durations) / len(durations),
                    "min_duration": min(durations),
                    "max_duration": max(durations),
                    "p95_duration": sorted(durations)[int(len(durations) * 0.95)],
                }

        report["performance_data"] = performance_data

        # Patrones de error
        report["error_patterns"] = dict(
            sorted(self.error_patterns.items(), key=lambda x: x[1], reverse=True)[:20]
        )  # Top 20

        # Correlaciones detectadas
        report["correlations"] = self.correlate_events()

        # Eventos de seguridad
        report["security_events"] = list(self.security_events)[-10:]  # Últimos 10

        return report

    async def _send_alert_notification(self, alert: Dict[str, Any]):
        """Enviar notificación de alerta a múltiples canales"""
        alert_sent = False

        # 1. Webhook HTTP (Slack, Discord, Teams, etc.)
        webhook_url = os.getenv("ALERT_WEBHOOK_URL")
        if webhook_url and AIOHTTP_AVAILABLE:
            try:
                # Formatear mensaje para diferentes plataformas
                webhook_payload = self._format_webhook_payload(alert, "slack")  # Default Slack format

                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook_url,
                        json=webhook_payload,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            logger.info(f"✅ Alerta enviada a webhook: {alert['alert_type']}")
                            alert_sent = True
                        else:
                            logger.error(f"❌ Error webhook ({response.status}): {await response.text()}")
            except Exception as e:
                logger.error(f"Error enviando alerta a webhook: {e}")

        # 2. Email notification (SMTP)
        email_config = self._get_email_config()
        if email_config and not alert_sent:  # Fallback si webhook falla
            try:
                await self._send_email_alert(alert, email_config)
                alert_sent = True
            except Exception as e:
                logger.error(f"Error enviando alerta por email: {e}")

        # 3. PagerDuty (para alertas críticas)
        if alert.get('severity') == 'critical':
            pagerduty_key = os.getenv("PAGERDUTY_INTEGRATION_KEY")
            if pagerduty_key and AIOHTTP_AVAILABLE:
                try:
                    await self._send_pagerduty_alert(alert, pagerduty_key)
                    alert_sent = True
                except Exception as e:
                    logger.error(f"Error enviando alerta a PagerDuty: {e}")

        # 4. Log local como último recurso
        if not alert_sent:
            logger.warning(f"📢 ALERTA LOCAL: {alert['message']} (Severity: {alert.get('severity', 'unknown')})")
            print(f"🚨 ALERTA NO ENVIADA - REVISAR CONFIGURACIÓN: {alert['message']}")

    def _format_webhook_payload(self, alert: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Formatear payload para diferentes plataformas de webhook"""
        base_message = f"🚨 *ALERTA*: {alert['message']}\n"
        base_message += f"• Servicio: {alert.get('service', 'unknown')}\n"
        base_message += f"• Severidad: {alert.get('severity', 'unknown')}\n"
        base_message += f"• Timestamp: {alert.get('timestamp', datetime.now().isoformat())}\n"

        if alert.get('anomaly_score'):
            base_message += f"• Score de anomalía: {alert['anomaly_score']:.2f}\n"

        if platform.lower() == "slack":
            return {
                "text": base_message,
                "attachments": [{
                    "color": "danger" if alert.get('severity') == 'critical' else "warning",
                    "fields": [
                        {"title": "Tipo", "value": alert.get('alert_type', 'unknown'), "short": True},
                        {"title": "Detalles", "value": str(alert.get('details', {}))[:500], "short": False}
                    ]
                }]
            }
        elif platform.lower() == "discord":
            return {
                "content": base_message,
                "embeds": [{
                    "title": "Alerta de Sistema",
                    "description": alert.get('message', ''),
                    "color": 15158332 if alert.get('severity') == 'critical' else 16776960,  # Red or Orange
                    "fields": [
                        {"name": "Tipo", "value": alert.get('alert_type', 'unknown'), "inline": True},
                        {"name": "Severidad", "value": alert.get('severity', 'unknown'), "inline": True},
                        {"name": "Servicio", "value": alert.get('service', 'unknown'), "inline": True}
                    ]
                }]
            }
        else:
            # Generic webhook format
            return {
                "alert": alert,
                "formatted_message": base_message
            }

    def _get_email_config(self) -> Optional[Dict[str, Any]]:
        """Obtener configuración de email desde variables de entorno"""
        smtp_server = os.getenv("SMTP_SERVER")
        smtp_port = os.getenv("SMTP_PORT", "587")
        smtp_user = os.getenv("SMTP_USER")
        smtp_password = os.getenv("SMTP_PASSWORD")
        alert_emails = os.getenv("ALERT_EMAILS", "").split(",") if os.getenv("ALERT_EMAILS") else []

        if not all([smtp_server, smtp_user, smtp_password, alert_emails]):
            return None

        return {
            "smtp_server": smtp_server,
            "smtp_port": int(smtp_port),
            "smtp_user": smtp_user,
            "smtp_password": smtp_password,
            "alert_emails": [email.strip() for email in alert_emails if email.strip()],
            "from_email": smtp_user
        }

    async def _send_email_alert(self, alert: Dict[str, Any], config: Dict[str, Any]):
        """Enviar alerta por email SMTP"""
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart

            # Crear mensaje
            msg = MIMEMultipart()
            msg['From'] = config['from_email']
            msg['To'] = ", ".join(config['alert_emails'])
            msg['Subject'] = f"🚨 ALERTA: {alert.get('alert_type', 'Sistema')} - {alert.get('service', 'Unknown')}"

            # Cuerpo del email
            body = f"""
ALERTA DE SISTEMA

Tipo: {alert.get('alert_type', 'Unknown')}
Servicio: {alert.get('service', 'Unknown')}
Severidad: {alert.get('severity', 'Unknown').upper()}
Timestamp: {alert.get('timestamp', datetime.now().isoformat())}

Mensaje:
{alert.get('message', 'Sin mensaje')}

Detalles:
{json.dumps(alert.get('details', {}), indent=2, ensure_ascii=False)}

--
Sistema de Monitoreo ELK
{self.service_name} v{self.service_version}
            """.strip()

            msg.attach(MIMEText(body, 'plain', 'utf-8'))

            # Enviar email
            server = smtplib.SMTP(config['smtp_server'], config['smtp_port'])
            server.starttls()
            server.login(config['smtp_user'], config['smtp_password'])
            text = msg.as_string()
            server.sendmail(config['from_email'], config['alert_emails'], text)
            server.quit()

            logger.info(f"✅ Alerta enviada por email a {len(config['alert_emails'])} destinatarios")

        except Exception as e:
            logger.error(f"Error enviando email: {e}")
            raise

    async def _send_pagerduty_alert(self, alert: Dict[str, Any], routing_key: str):
        """Enviar alerta crítica a PagerDuty"""
        if not AIOHTTP_AVAILABLE:
            return

        try:
            payload = {
                "routing_key": routing_key,
                "event_action": "trigger",
                "payload": {
                    "summary": alert.get('message', 'Alerta crítica del sistema'),
                    "severity": "critical" if alert.get('severity') == 'critical' else "error",
                    "source": alert.get('service', 'unknown'),
                    "component": "monitoring_system",
                    "group": "infrastructure",
                    "class": "alert",
                    "custom_details": alert
                }
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 202:
                        logger.info("✅ Alerta enviada a PagerDuty")
                    else:
                        logger.error(f"❌ Error PagerDuty ({response.status}): {await response.text()}")

        except Exception as e:
            logger.error(f"Error enviando a PagerDuty: {e}")
            raise


class ELKLogger(logging.Handler):
    """Logging Handler personalizado para ELKStackManager"""

    def __init__(self, elk_manager: ELKStackManager):
        super().__init__()
        self.elk_manager = elk_manager
        self.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )

    def emit(self, record):
        """Emitir log entry a través del ELK manager"""
        try:
            # Formatear mensaje
            message = self.format(record)

            # Extraer información adicional del record
            extra_kwargs = {
                "logger_name": record.name,
                "module": record.module if hasattr(record, "module") else "",
                "function": record.funcName if hasattr(record, "funcName") else "",
                "line_number": record.lineno if hasattr(record, "lineno") else 0,
            }

            # Agregar información estructurada si existe
            if hasattr(record, "correlation_id"):
                extra_kwargs["correlation_id"] = record.correlation_id
            if hasattr(record, "user_id"):
                extra_kwargs["user_id"] = record.user_id
            if hasattr(record, "request_id"):
                extra_kwargs["request_id"] = record.request_id
            if hasattr(record, "duration_ms"):
                extra_kwargs["duration_ms"] = record.duration_ms

            # Enviar a ELK
            self.elk_manager.log(record.levelname, message, **extra_kwargs)

        except Exception:
            self.handleError(record)


class LogAnalyzer:
    """Analizador automático de logs con capacidades ML"""

    def __init__(self, elk_manager: ELKStackManager):
        self.elk_manager = elk_manager
        self.analysis_patterns = self._load_analysis_patterns()

    def _load_analysis_patterns(self) -> List[Dict[str, Any]]:
        """Cargar patrones de análisis de logs"""
        return [
            {
                "name": "error_burst_detection",
                "query": {"level": "ERROR"},
                "threshold": 10,
                "time_window": 300,
                "action": "alert",
            },
            {
                "name": "performance_degradation",
                "query": {"duration_ms": ">2000"},
                "threshold": 5,
                "time_window": 60,
                "action": "monitor",
            },
            {
                "name": "security_threat_pattern",
                "query": {"message": "*unauthorized* OR *failed login*"},
                "threshold": 3,
                "time_window": 600,
                "action": "alert",
            },
        ]

    def analyze_logs(self, logs: List[LogEntry]) -> List[Dict[str, Any]]:
        """Analizar logs para detectar patrones y anomalías"""
        findings = []

        # Análisis de frecuencia
        error_counts = defaultdict(int)
        for log in logs:
            if log.level == "ERROR":
                key = f"{log.module}:{log.function}"
                error_counts[key] += 1

        # Detectar funciones con alta frecuencia de errores
        for func_key, count in error_counts.items():
            if count > 5:  # Más de 5 errores
                findings.append(
                    {
                        "type": "error_hotspot",
                        "severity": "high",
                        "description": f"Función {func_key} tiene {count} errores",
                        "recommendation": "Revisar lógica de error handling",
                    }
                )

        # Análisis de rendimiento
        duration_data = [log.duration_ms for log in logs if log.duration_ms]
        if duration_data:
            avg_duration = sum(duration_data) / len(duration_data)
            max_duration = max(duration_data)

            if max_duration > 10000:  # Más de 10 segundos
                findings.append(
                    {
                        "type": "performance_issue",
                        "severity": "medium",
                        "description": f"Operación muy lenta detectada: {max_duration}ms",
                        "recommendation": "Optimizar consulta de base de datos",
                    }
                )

        return findings


# ================================
# DEMO Y EJEMPLOS DE USO
# ================================


async def demo_elk_integration():
    """Demo del sistema de logging ELK avanzado"""
    print("🔍 DEMO: Advanced ELK Logging Integration")
    print("=" * 60)

    # Inicializar sistema ELK
    elk_manager = ELKStackManager()

    # Configurar handler personalizado para el logger estándar
    elk_handler = ELKLogger(elk_manager)
    logging.getLogger().addHandler(elk_handler)
    logging.getLogger().setLevel(logging.INFO)

    print("✅ ELK Logger configurado")

    # Simular diferentes tipos de logs
    logger.info(
        "🚀 Aplicación iniciada", correlation_id="demo-001", service_name="sheily_ai"
    )
    logger.info(
        "🔄 Procesando solicitud de usuario", user_id="user123", request_id="req-456"
    )
    logger.warning(
        "⚠️ Conexión lenta a base de datos", duration_ms=1500.5, module="database"
    )
    logger.error(
        "❌ Error de autenticación", error_type="auth_failed", user_id="user123"
    )

    # Logs de seguridad
    elk_manager.log(
        "INFO",
        "Intento de login fallido",
        security_event="failed_login",
        user_id="hacker123",
    )
    elk_manager.log(
        "ERROR",
        "Violación CSP detectada",
        security_event="csp_violation",
        blocked_uri="evil.com/script.js",
    )

    # Logs de rendimiento
    elk_manager.log(
        "INFO",
        "Consulta RAG completada",
        duration_ms=2500.0,
        response_code=200,
        correlation_id="rag-789",
    )

    await asyncio.sleep(1)  # Esperar procesamiento async

    # Generar reportes
    print("\n📊 Reporte de Rendimiento:")
    perf_report = elk_manager.get_performance_report()
    print(f"  • Funciones analizadas: {len(perf_report['performance_data'])}")
    print(f"  • Patrones de error: {len(perf_report['error_patterns'])}")
    print(f"  • Eventos de seguridad: {len(perf_report['security_events'])}")

    # Mostrar correlaciones
    correlations = perf_report["correlations"]
    if correlations:
        print(f"  • Correlaciones detectadas: {len(correlations)}")
        for corr in correlations[:2]:
            print(f"    - {corr['rule_name']}: {corr['matches_count']} coincidencias")

    # Análisis automático
    analyzer = LogAnalyzer(elk_manager)
    log_entries = list(elk_manager.log_buffer)
    findings = analyzer.analyze_logs(log_entries)

    if findings:
        print("\n🔍 Hallazgos de Análisis Automático:")
        for finding in findings:
            print(f"  • [{finding['severity'].upper()}] {finding['description']}")

    # Flush final
    await elk_manager._flush_buffer()

    print("\n✅ Demo de logging ELK completada exitosamente")
    print(f"📄 Total logs procesados: {len(log_entries)}")

    return perf_report


if __name__ == "__main__":
    # Ejecutar demo
    asyncio.run(demo_elk_integration())
