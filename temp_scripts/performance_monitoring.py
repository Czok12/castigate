"""
📊 PERFORMANCE-MONITORING UND AUTOMATISCHE OPTIMIERUNG
======================================================

Real-Time-Monitoring, Anomalie-Erkennung und selbstlernende Optimierungen
"""

import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any, Dict, List

import numpy as np


@dataclass
class PerformanceMetrics:
    """Performance-Metriken für Monitoring"""

    timestamp: float
    query_processing_time: float
    retrieval_time: float
    llm_generation_time: float
    total_response_time: float
    cache_hit_rate: float
    memory_usage_mb: float
    retrieval_quality_score: float
    answer_confidence: float
    user_satisfaction: float
    tokens_processed: int
    documents_retrieved: int


@dataclass
class SystemAlert:
    """System-Alert für Anomalien"""

    alert_id: str
    timestamp: float
    severity: str  # "low", "medium", "high", "critical"
    category: str  # "performance", "quality", "error", "resource"
    message: str
    metrics: Dict[str, Any]
    suggested_actions: List[str]


class PerformanceMonitor:
    """Real-Time Performance-Monitoring"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics_history: deque[PerformanceMetrics] = deque(maxlen=window_size)
        self.alerts: List[SystemAlert] = []

        # Rolling Statistics
        self.rolling_stats: Dict[str, deque] = {
            "response_times": deque(maxlen=window_size),
            "cache_hit_rates": deque(maxlen=window_size),
            "quality_scores": deque(maxlen=window_size),
            "confidence_scores": deque(maxlen=window_size),
        }

        # Thresholds für Alerts
        self.thresholds = {
            "max_response_time": 10.0,  # Sekunden
            "min_cache_hit_rate": 0.3,  # 30%
            "min_quality_score": 0.6,  # 60%
            "min_confidence": 0.5,  # 50%
            "max_memory_usage": 1000.0,  # MB
        }

        # Performance-Baseline
        self.baseline_metrics: Dict[str, Any] = {}
        self.baseline_calculated = False

        # Logger
        self.logger = logging.getLogger(__name__)

    def record_metrics(self, metrics: PerformanceMetrics):
        """Zeichnet Performance-Metriken auf"""

        # Zu Historie hinzufügen
        self.metrics_history.append(metrics)

        # Rolling Statistics aktualisieren
        self.rolling_stats["response_times"].append(metrics.total_response_time)
        self.rolling_stats["cache_hit_rates"].append(metrics.cache_hit_rate)
        self.rolling_stats["quality_scores"].append(metrics.retrieval_quality_score)
        self.rolling_stats["confidence_scores"].append(metrics.answer_confidence)

        # Baseline berechnen (nach ersten 20 Messungen)
        if len(self.metrics_history) == 20 and not self.baseline_calculated:
            self._calculate_baseline()

        # Anomalie-Erkennung
        if self.baseline_calculated:
            self._detect_anomalies(metrics)

    def _calculate_baseline(self):
        """Berechnet Performance-Baseline"""

        recent_metrics = list(self.metrics_history)[-20:]

        self.baseline_metrics = {
            "avg_response_time": np.mean(
                [m.total_response_time for m in recent_metrics]
            ),
            "avg_cache_hit_rate": np.mean([m.cache_hit_rate for m in recent_metrics]),
            "avg_quality_score": np.mean(
                [m.retrieval_quality_score for m in recent_metrics]
            ),
            "avg_confidence": np.mean([m.answer_confidence for m in recent_metrics]),
            "std_response_time": np.std(
                [m.total_response_time for m in recent_metrics]
            ),
        }

        self.baseline_calculated = True
        self.logger.info(f"Performance-Baseline berechnet: {self.baseline_metrics}")

    def _detect_anomalies(self, metrics: PerformanceMetrics):
        """Erkennt Performance-Anomalien"""

        alerts = []

        # 1. Überschreitung von Absolut-Thresholds
        if metrics.total_response_time > self.thresholds["max_response_time"]:
            alerts.append(
                self._create_alert(
                    "high",
                    "performance",
                    f"Antwortzeit zu hoch: {metrics.total_response_time:.2f}s",
                    {"response_time": metrics.total_response_time},
                    ["Cache-Strategie überprüfen", "Retrieval-Parameter optimieren"],
                )
            )

        if metrics.cache_hit_rate < self.thresholds["min_cache_hit_rate"]:
            alerts.append(
                self._create_alert(
                    "medium",
                    "performance",
                    f"Cache-Hit-Rate zu niedrig: {metrics.cache_hit_rate:.1%}",
                    {"cache_hit_rate": metrics.cache_hit_rate},
                    ["Cache-TTL erhöhen", "Query-Preprocessing verbessern"],
                )
            )

        if metrics.retrieval_quality_score < self.thresholds["min_quality_score"]:
            alerts.append(
                self._create_alert(
                    "high",
                    "quality",
                    f"Retrieval-Qualität zu niedrig: {metrics.retrieval_quality_score:.1%}",
                    {"quality_score": metrics.retrieval_quality_score},
                    ["Embedding-Modell überprüfen", "Chunking-Strategie anpassen"],
                )
            )

        # 2. Statistisch signifikante Abweichungen von Baseline
        if self.baseline_calculated:
            response_time_zscore = (
                metrics.total_response_time - self.baseline_metrics["avg_response_time"]
            ) / max(float(self.baseline_metrics["std_response_time"]), 0.1)

            if abs(response_time_zscore) > 2.5:  # 2.5 Standardabweichungen
                alerts.append(
                    self._create_alert(
                        "medium",
                        "performance",
                        f"Antwortzeit-Anomalie: {response_time_zscore:.1f} Standardabweichungen",
                        {
                            "zscore": response_time_zscore,
                            "response_time": metrics.total_response_time,
                        },
                        ["System-Load prüfen", "Parallele Verarbeitung optimieren"],
                    )
                )

        # 3. Trend-basierte Alerts
        trend_alerts = self._detect_trends()
        alerts.extend(trend_alerts)

        # Alerts speichern
        for alert in alerts:
            self.alerts.append(alert)
            self.logger.warning(f"Performance-Alert: {alert.message}")

    def _detect_trends(self) -> List[SystemAlert]:
        """Erkennt negative Trends in den Metriken"""

        if len(self.rolling_stats["response_times"]) < 10:
            return []

        alerts = []

        # Response-Time-Trend
        recent_times = list(self.rolling_stats["response_times"])[-10:]
        older_times = (
            list(self.rolling_stats["response_times"])[-20:-10]
            if len(self.rolling_stats["response_times"]) >= 20
            else []
        )

        if older_times:
            recent_avg = np.mean(recent_times)
            older_avg = np.mean(older_times)

            if recent_avg > older_avg * 1.5:  # 50% Verschlechterung
                alerts.append(
                    self._create_alert(
                        "medium",
                        "performance",
                        f"Verschlechterung der Antwortzeit: {recent_avg:.2f}s vs {older_avg:.2f}s",
                        {"recent_avg": recent_avg, "older_avg": older_avg},
                        ["Cache-Effizienz prüfen", "Retrieval-Strategie optimieren"],
                    )
                )

        # Cache-Hit-Rate-Trend
        recent_cache = list(self.rolling_stats["cache_hit_rates"])[-10:]
        older_cache = (
            list(self.rolling_stats["cache_hit_rates"])[-20:-10]
            if len(self.rolling_stats["cache_hit_rates"]) >= 20
            else []
        )

        if older_cache:
            recent_cache_avg = np.mean(recent_cache)
            older_cache_avg = np.mean(older_cache)

            if recent_cache_avg < older_cache_avg * 0.8:  # 20% Verschlechterung
                alerts.append(
                    self._create_alert(
                        "low",
                        "performance",
                        f"Verschlechterung der Cache-Effizienz: {recent_cache_avg:.1%} vs {older_cache_avg:.1%}",
                        {
                            "recent_cache": recent_cache_avg,
                            "older_cache": older_cache_avg,
                        },
                        ["Cache-Strategie überdenken", "Query-Patterns analysieren"],
                    )
                )

        return alerts

    def _create_alert(
        self,
        severity: str,
        category: str,
        message: str,
        metrics: Dict[str, Any],
        actions: List[str],
    ) -> SystemAlert:
        """Erstellt System-Alert"""

        alert_id = f"{category}_{int(time.time())}_{hash(message) % 10000}"

        return SystemAlert(
            alert_id=alert_id,
            timestamp=time.time(),
            severity=severity,
            category=category,
            message=message,
            metrics=metrics,
            suggested_actions=actions,
        )

    def get_current_performance_summary(self) -> Dict[str, Any]:
        """Gibt aktuelle Performance-Zusammenfassung zurück"""

        if not self.metrics_history:
            return {"status": "no_data"}

        recent_metrics = list(self.metrics_history)[-10:]

        return {
            "status": "healthy" if len(self.get_active_alerts()) == 0 else "issues",
            "current_metrics": {
                "avg_response_time": float(
                    np.mean([m.total_response_time for m in recent_metrics])
                ),
                "avg_cache_hit_rate": float(
                    np.mean([m.cache_hit_rate for m in recent_metrics])
                ),
                "avg_quality_score": float(
                    np.mean([m.retrieval_quality_score for m in recent_metrics])
                ),
                "avg_confidence": float(
                    np.mean([m.answer_confidence for m in recent_metrics])
                ),
            },
            "active_alerts": len(self.get_active_alerts()),
            "total_queries": len(self.metrics_history),
            "baseline_status": (
                "calculated" if self.baseline_calculated else "calculating"
            ),
        }

    def get_active_alerts(self, max_age_hours: int = 24) -> List[SystemAlert]:
        """Gibt aktive Alerts zurück"""

        cutoff_time = time.time() - (max_age_hours * 3600)
        return [alert for alert in self.alerts if alert.timestamp > cutoff_time]

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generiert detaillierten Performance-Report"""

        cutoff_time = time.time() - (hours * 3600)
        recent_metrics = [m for m in self.metrics_history if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {"error": "Keine Daten verfügbar"}

        # Statistiken berechnen
        response_times = [m.total_response_time for m in recent_metrics]
        cache_rates = [m.cache_hit_rate for m in recent_metrics]
        quality_scores = [m.retrieval_quality_score for m in recent_metrics]

        report = {
            "period": f"Letzte {hours} Stunden",
            "total_queries": len(recent_metrics),
            "performance_stats": {
                "response_time": {
                    "avg": float(np.mean(response_times)),
                    "min": float(np.min(response_times)),
                    "max": float(np.max(response_times)),
                    "p95": float(np.percentile(response_times, 95)),
                },
                "cache_efficiency": {
                    "avg_hit_rate": float(np.mean(cache_rates)),
                    "best_hit_rate": float(np.max(cache_rates)),
                    "worst_hit_rate": float(np.min(cache_rates)),
                },
                "quality_metrics": {
                    "avg_quality": float(np.mean(quality_scores)),
                    "quality_trend": self._calculate_trend(quality_scores),
                },
            },
            "alerts_summary": self._summarize_alerts(hours),
            "recommendations": self._generate_recommendations(recent_metrics),
        }

        return report

    def _calculate_trend(self, values: List[float]) -> str:
        """Berechnet Trend-Richtung"""

        if len(values) < 5:
            return "insufficient_data"

        recent = np.mean(values[-5:])
        older = np.mean(values[:5])

        if recent > older * 1.1:
            return "improving"
        elif recent < older * 0.9:
            return "declining"
        else:
            return "stable"

    def _summarize_alerts(self, hours: int) -> Dict[str, Any]:
        """Zusammenfassung der Alerts"""

        recent_alerts = self.get_active_alerts(hours)

        by_severity: defaultdict[str, int] = defaultdict(int)
        by_category: defaultdict[str, int] = defaultdict(int)

        for alert in recent_alerts:
            by_severity[alert.severity] += 1
            by_category[alert.category] += 1

        return {
            "total_alerts": len(recent_alerts),
            "by_severity": dict(by_severity),
            "by_category": dict(by_category),
            "most_recent": recent_alerts[-1].message if recent_alerts else None,
        }

    def _generate_recommendations(self, metrics: List[PerformanceMetrics]) -> List[str]:
        """Generiert Performance-Empfehlungen"""

        recommendations = []

        avg_response_time = np.mean([m.total_response_time for m in metrics])
        avg_cache_rate = np.mean([m.cache_hit_rate for m in metrics])
        avg_quality = np.mean([m.retrieval_quality_score for m in metrics])

        if avg_response_time > 5.0:
            recommendations.append(
                "Antwortzeit optimieren: Parallele Verarbeitung oder kleinere Chunk-Größen"
            )

        if avg_cache_rate < 0.4:
            recommendations.append(
                "Cache-Effizienz verbessern: TTL-Werte anpassen oder Query-Preprocessing"
            )

        if avg_quality < 0.7:
            recommendations.append(
                "Retrieval-Qualität steigern: Embedding-Modell oder Ranking-Algorithmus überprüfen"
            )

        return recommendations


class AutoOptimizer:
    """Automatische System-Optimierung basierend auf Performance-Daten"""

    def __init__(self, monitor: PerformanceMonitor):
        self.monitor = monitor
        self.optimization_history: List[Dict[str, Any]] = []

        # Optimierungsstrategien
        self.optimization_strategies = {
            "cache_ttl": self._optimize_cache_ttl,
            "retrieval_params": self._optimize_retrieval_params,
            "chunk_size": self._optimize_chunk_size,
            "batch_size": self._optimize_batch_size,
        }

        # Aktuelle Konfiguration
        self.current_config = {
            "cache_ttl": 86400,  # 24h
            "retrieval_k": 4,
            "chunk_size": 1000,
            "batch_size": 10,
        }

        self.logger = logging.getLogger(__name__)

    def run_optimization_cycle(self) -> Dict[str, Any]:
        """Führt einen Optimierungszyklus aus"""

        if not self.monitor.baseline_calculated:
            return {"status": "waiting_for_baseline"}

        # Analysiere aktuelle Performance
        performance_summary = self.monitor.get_current_performance_summary()

        if performance_summary["status"] == "healthy":
            return {"status": "no_optimization_needed"}

        # Identifiziere Optimierungsmöglichkeiten
        optimization_targets = self._identify_optimization_targets()

        if not optimization_targets:
            return {"status": "no_targets_identified"}

        # Führe Optimierungen durch
        optimizations_applied = []

        for target in optimization_targets:
            if target in self.optimization_strategies:
                result = self.optimization_strategies[target]()
                if result["success"]:
                    optimizations_applied.append(result)
                    self.logger.info(f"Optimierung angewendet: {target} - {result}")

        # Dokumentiere Optimierung
        optimization_record = {
            "timestamp": time.time(),
            "targets": optimization_targets,
            "optimizations_applied": optimizations_applied,
            "config_before": self.current_config.copy(),
            "performance_before": performance_summary,
        }

        self.optimization_history.append(optimization_record)

        return {
            "status": "optimizations_applied",
            "optimizations": optimizations_applied,
            "new_config": self.current_config,
        }

    def _identify_optimization_targets(self) -> List[str]:
        """Identifiziert Optimierungsziele basierend auf Performance-Daten"""

        active_alerts = self.monitor.get_active_alerts()
        targets = []

        for alert in active_alerts:
            if alert.category == "performance":
                if "Antwortzeit" in alert.message:
                    targets.extend(["retrieval_params", "batch_size"])
                elif "Cache" in alert.message:
                    targets.append("cache_ttl")
            elif alert.category == "quality":
                targets.extend(["retrieval_params", "chunk_size"])

        return list(set(targets))  # Deduplizierung

    def _optimize_cache_ttl(self) -> Dict[str, Any]:
        """Optimiert Cache-TTL basierend auf Hit-Rates"""

        recent_metrics = list(self.monitor.metrics_history)[-20:]
        avg_hit_rate = np.mean([m.cache_hit_rate for m in recent_metrics])

        if avg_hit_rate < 0.3:
            # Erhöhe TTL für bessere Hit-Rate
            new_ttl = min(
                self.current_config["cache_ttl"] * 1.5, 7 * 24 * 3600
            )  # Max 1 Woche
            self.current_config["cache_ttl"] = int(new_ttl)

            return {
                "success": True,
                "strategy": "increase_ttl",
                "old_value": self.current_config["cache_ttl"] / 1.5,
                "new_value": new_ttl,
                "reason": f"Hit-Rate zu niedrig: {avg_hit_rate:.1%}",
            }

        return {"success": False, "reason": "Cache-TTL bereits optimal"}

    def _optimize_retrieval_params(self) -> Dict[str, Any]:
        """Optimiert Retrieval-Parameter"""

        recent_metrics = list(self.monitor.metrics_history)[-10:]
        avg_quality = np.mean([m.retrieval_quality_score for m in recent_metrics])
        avg_response_time = np.mean([m.total_response_time for m in recent_metrics])

        if avg_quality < 0.6 and avg_response_time < 3.0:
            # Erhöhe k für bessere Qualität (wenn Performance erlaubt)
            new_k = min(self.current_config["retrieval_k"] + 1, 8)
            self.current_config["retrieval_k"] = new_k

            return {
                "success": True,
                "strategy": "increase_k",
                "old_value": self.current_config["retrieval_k"] - 1,
                "new_value": new_k,
                "reason": f"Qualität zu niedrig: {avg_quality:.1%}, Performance erlaubt höheres k",
            }
        elif avg_response_time > 5.0:
            # Reduziere k für bessere Performance
            new_k = max(self.current_config["retrieval_k"] - 1, 2)
            self.current_config["retrieval_k"] = new_k

            return {
                "success": True,
                "strategy": "decrease_k",
                "old_value": self.current_config["retrieval_k"] + 1,
                "new_value": new_k,
                "reason": f"Antwortzeit zu hoch: {avg_response_time:.2f}s",
            }

        return {"success": False, "reason": "Retrieval-Parameter bereits optimal"}

    def _optimize_chunk_size(self) -> Dict[str, Any]:
        """Optimiert Chunk-Größe basierend auf Qualitäts-Metriken"""

        recent_metrics = list(self.monitor.metrics_history)[-10:]
        avg_quality = np.mean([m.retrieval_quality_score for m in recent_metrics])

        if avg_quality < 0.6:
            # Experimentiere mit verschiedenen Chunk-Größen
            if self.current_config["chunk_size"] == 1000:
                new_size = 800  # Kleinere Chunks für bessere Präzision
            elif self.current_config["chunk_size"] == 800:
                new_size = 1200  # Größere Chunks für mehr Kontext
            else:
                new_size = 1000  # Zurück zum Standard

            self.current_config["chunk_size"] = new_size

            return {
                "success": True,
                "strategy": "adjust_chunk_size",
                "old_value": self.current_config["chunk_size"],
                "new_value": new_size,
                "reason": f"Qualität zu niedrig: {avg_quality:.1%}",
            }

        return {"success": False, "reason": "Chunk-Größe bereits optimal"}

    def _optimize_batch_size(self) -> Dict[str, Any]:
        """Optimiert Batch-Größe für parallele Verarbeitung"""

        recent_metrics = list(self.monitor.metrics_history)[-10:]
        avg_response_time = np.mean([m.total_response_time for m in recent_metrics])

        if avg_response_time > 7.0:
            # Erhöhe Batch-Größe für bessere Parallelisierung
            new_batch_size = min(self.current_config["batch_size"] + 5, 50)
            self.current_config["batch_size"] = new_batch_size

            return {
                "success": True,
                "strategy": "increase_batch_size",
                "old_value": self.current_config["batch_size"] - 5,
                "new_value": new_batch_size,
                "reason": f"Antwortzeit zu hoch: {avg_response_time:.2f}s",
            }

        return {"success": False, "reason": "Batch-Größe bereits optimal"}

    def get_optimization_report(self) -> Dict[str, Any]:
        """Gibt Optimierungs-Report zurück"""

        if not self.optimization_history:
            return {"status": "no_optimizations_yet"}

        recent_optimizations = self.optimization_history[-5:]  # Letzte 5

        return {
            "total_optimizations": len(self.optimization_history),
            "recent_optimizations": len(recent_optimizations),
            "current_config": self.current_config,
            "most_recent_optimization": self.optimization_history[-1],
            "optimization_effectiveness": self._calculate_optimization_effectiveness(),
        }

    def _calculate_optimization_effectiveness(self) -> Dict[str, float]:
        """Berechnet Effektivität der Optimierungen"""

        if len(self.optimization_history) < 2:
            return {"insufficient_data": True}

        # Vergleiche Performance vor und nach letzter Optimierung
        last_optimization = self.optimization_history[-1]

        # Hole Metriken vor und nach Optimierung
        optimization_time = last_optimization["timestamp"]

        before_metrics = [
            m for m in self.monitor.metrics_history if m.timestamp < optimization_time
        ][-10:]
        after_metrics = [
            m for m in self.monitor.metrics_history if m.timestamp > optimization_time
        ][-10:]

        if not before_metrics or not after_metrics:
            return {"insufficient_data": True}

        before_avg_response = np.mean([m.total_response_time for m in before_metrics])
        after_avg_response = np.mean([m.total_response_time for m in after_metrics])

        before_avg_quality = np.mean(
            [m.retrieval_quality_score for m in before_metrics]
        )
        after_avg_quality = np.mean([m.retrieval_quality_score for m in after_metrics])

        return {
            "response_time_improvement": float(
                (before_avg_response - after_avg_response) / before_avg_response
            ),
            "quality_improvement": float(
                (after_avg_quality - before_avg_quality) / before_avg_quality
            ),
            "optimization_effective": bool(
                after_avg_response < before_avg_response
                or after_avg_quality > before_avg_quality
            ),
        }


class PerformanceReporter:
    """Erstellt detaillierte Performance-Reports"""

    def __init__(self, monitor: PerformanceMonitor, optimizer: AutoOptimizer):
        self.monitor = monitor
        self.optimizer = optimizer

    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generiert umfassenden Performance-Report"""

        return {
            "timestamp": datetime.now().isoformat(),
            "system_status": self.monitor.get_current_performance_summary(),
            "performance_analysis": self.monitor.get_performance_report(24),
            "optimization_status": self.optimizer.get_optimization_report(),
            "active_alerts": [
                asdict(alert) for alert in self.monitor.get_active_alerts()
            ],
            "recommendations": self._generate_executive_recommendations(),
        }

    def _generate_executive_recommendations(self) -> List[str]:
        """Generiert High-Level-Empfehlungen"""

        recommendations = []

        system_status = self.monitor.get_current_performance_summary()
        active_alerts = self.monitor.get_active_alerts()

        if system_status["status"] != "healthy":
            recommendations.append(
                "System benötigt Aufmerksamkeit - kritische Performance-Issues identifiziert"
            )

        critical_alerts = [a for a in active_alerts if a.severity == "critical"]
        if critical_alerts:
            recommendations.append(
                f"{len(critical_alerts)} kritische Alerts erfordern sofortige Maßnahmen"
            )

        optimization_report = self.optimizer.get_optimization_report()
        if optimization_report.get("total_optimizations", 0) == 0:
            recommendations.append(
                "Automatische Optimierung noch nicht aktiv - Baseline-Phase läuft"
            )

        return recommendations

    def export_report_json(self, filepath: str) -> bool:
        """Exportiert Report als JSON"""

        try:
            report = self.generate_comprehensive_report()
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            self.monitor.logger.error(f"Fehler beim Export: {e}")
            return False
