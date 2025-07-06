"""
🎯 AUTO-OPTIMIZING SYSTEM MANAGER
=================================

Ultra-Advanced selbst-optimierendes System mit KI-gestützter Performance-Optimierung
"""

import sqlite3
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class SystemMetric:
    """Einzelne System-Metrik"""

    name: str
    value: float
    timestamp: float
    category: str  # performance, quality, user_satisfaction, resource_usage
    component: str  # retrieval, caching, neural_fusion, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationAction:
    """Einzelne Optimierungsmaßnahme"""

    action_id: str
    action_type: str  # parameter_tuning, cache_optimization, model_adjustment
    component: str
    parameters: Dict[str, Any]
    expected_impact: float
    confidence: float
    execution_time: float
    rollback_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceProfile:
    """Performance-Profil des Systems"""

    profile_name: str
    optimal_parameters: Dict[str, Any]
    average_response_time: float
    average_accuracy: float
    resource_efficiency: float
    user_satisfaction_score: float
    usage_patterns: Dict[str, float]
    last_updated: float


class MetricsCollector:
    """Sammelt und aggregiert System-Metriken"""

    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.metrics_buffer: deque = deque(maxlen=10000)
        self.aggregated_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=1000)
        )
        self.metric_callbacks: Dict[str, List[Callable]] = defaultdict(list)

        # Threading für kontinuierliche Sammlung
        self.is_collecting = False
        self.collection_thread: Optional[threading.Thread] = None

        # Metrik-Kategorien
        self.metric_categories = {
            "performance": ["response_time", "throughput", "latency", "queue_depth"],
            "quality": ["confidence_score", "accuracy", "relevance", "completeness"],
            "resource": ["memory_usage", "cpu_usage", "cache_hit_rate", "disk_io"],
            "user": [
                "satisfaction_score",
                "engagement_time",
                "query_refinements",
                "abandonment_rate",
            ],
        }

        # Baseline-Werte für Anomalie-Erkennung
        self.baselines: Dict[str, Dict[str, float]] = {}
        self._initialize_baselines()

    def _initialize_baselines(self):
        """Initialisiert Baseline-Werte für Metriken"""

        baseline_values = {
            "response_time": {"mean": 0.5, "std": 0.1, "min": 0.1, "max": 2.0},
            "confidence_score": {"mean": 0.75, "std": 0.15, "min": 0.3, "max": 0.95},
            "cache_hit_rate": {"mean": 0.8, "std": 0.1, "min": 0.5, "max": 0.95},
            "user_satisfaction": {"mean": 0.8, "std": 0.1, "min": 0.5, "max": 1.0},
            "accuracy": {"mean": 0.85, "std": 0.08, "min": 0.6, "max": 0.98},
        }

        for metric, values in baseline_values.items():
            self.baselines[metric] = values

    def start_collection(self):
        """Startet kontinuierliche Metrik-Sammlung"""

        if not self.is_collecting:
            self.is_collecting = True
            self.collection_thread = threading.Thread(
                target=self._collection_loop, daemon=True
            )
            self.collection_thread.start()
            print("✅ Metrik-Sammlung gestartet")

    def stop_collection(self):
        """Stoppt Metrik-Sammlung"""

        self.is_collecting = False
        if self.collection_thread:
            self.collection_thread.join(timeout=2.0)
        print("🛑 Metrik-Sammlung gestoppt")

    def _collection_loop(self):
        """Haupt-Loop für Metrik-Sammlung"""

        while self.is_collecting:
            try:
                # Sammle aktuelle System-Metriken
                current_metrics = self._collect_current_metrics()

                # Speichere in Buffer
                for metric in current_metrics:
                    self.metrics_buffer.append(metric)
                    self.aggregated_metrics[metric.name].append(metric.value)

                    # Callback-Ausführung
                    for callback in self.metric_callbacks.get(metric.name, []):
                        try:
                            callback(metric)
                        except Exception as e:
                            print(f"Fehler bei Metrik-Callback: {e}")

                time.sleep(self.collection_interval)

            except Exception as e:
                print(f"Fehler bei Metrik-Sammlung: {e}")
                time.sleep(self.collection_interval)

    def _collect_current_metrics(self) -> List[SystemMetric]:
        """Sammelt aktuelle System-Metriken"""

        current_time = time.time()
        metrics = []

        # Simuliere verschiedene System-Metriken
        # (In echter Implementierung würden diese von tatsächlichen Komponenten kommen)

        # Performance-Metriken
        metrics.extend(
            [
                SystemMetric(
                    name="response_time",
                    value=np.random.normal(0.5, 0.1),
                    timestamp=current_time,
                    category="performance",
                    component="retrieval_engine",
                ),
                SystemMetric(
                    name="throughput",
                    value=np.random.normal(10.0, 2.0),
                    timestamp=current_time,
                    category="performance",
                    component="query_processor",
                ),
            ]
        )

        # Quality-Metriken
        metrics.extend(
            [
                SystemMetric(
                    name="confidence_score",
                    value=np.random.normal(0.75, 0.1),
                    timestamp=current_time,
                    category="quality",
                    component="answer_generator",
                ),
                SystemMetric(
                    name="relevance_score",
                    value=np.random.normal(0.8, 0.08),
                    timestamp=current_time,
                    category="quality",
                    component="retrieval_engine",
                ),
            ]
        )

        # Resource-Metriken
        metrics.extend(
            [
                SystemMetric(
                    name="cache_hit_rate",
                    value=np.random.normal(0.8, 0.05),
                    timestamp=current_time,
                    category="resource",
                    component="caching_system",
                ),
                SystemMetric(
                    name="memory_usage",
                    value=np.random.normal(0.6, 0.1),
                    timestamp=current_time,
                    category="resource",
                    component="system",
                ),
            ]
        )

        return metrics

    def register_callback(
        self, metric_name: str, callback: Callable[[SystemMetric], None]
    ):
        """Registriert Callback für spezifische Metrik"""

        self.metric_callbacks[metric_name].append(callback)

    def get_metric_statistics(
        self, metric_name: str, window_minutes: int = 10
    ) -> Dict[str, float]:
        """Berechnet Statistiken für eine Metrik"""

        if metric_name not in self.aggregated_metrics:
            return {}

        cutoff_time = time.time() - (window_minutes * 60)
        recent_values = [
            m.value
            for m in self.metrics_buffer
            if m.name == metric_name and m.timestamp >= cutoff_time
        ]

        if not recent_values:
            return {}

        return {
            "mean": statistics.mean(recent_values),
            "median": statistics.median(recent_values),
            "std": statistics.stdev(recent_values) if len(recent_values) > 1 else 0.0,
            "min": min(recent_values),
            "max": max(recent_values),
            "count": len(recent_values),
            "trend": self._calculate_trend(recent_values),
        }

    def _calculate_trend(self, values: List[float]) -> str:
        """Berechnet Trend für Metrik-Werte"""

        if len(values) < 5:
            return "insufficient_data"

        # Einfache Trend-Berechnung
        mid_point = len(values) // 2
        first_half = statistics.mean(values[:mid_point])
        second_half = statistics.mean(values[mid_point:])

        diff_ratio = (second_half - first_half) / first_half if first_half != 0 else 0

        if diff_ratio > 0.05:
            return "increasing"
        elif diff_ratio < -0.05:
            return "decreasing"
        else:
            return "stable"


class AnomalyDetector:
    """Erkennt Anomalien in System-Metriken"""

    def __init__(self, sensitivity: float = 2.0):
        self.sensitivity = sensitivity  # Standard-Abweichungen für Anomalie-Schwellwert
        self.anomaly_history: deque = deque(maxlen=1000)
        self.detection_algorithms = {
            "statistical": self._statistical_anomaly_detection,
            "trend": self._trend_anomaly_detection,
            "pattern": self._pattern_anomaly_detection,
        }

    def detect_anomalies(
        self, metrics_collector: MetricsCollector
    ) -> List[Dict[str, Any]]:
        """Hauptfunktion für Anomalie-Erkennung"""

        anomalies = []

        # Prüfe alle wichtigen Metriken
        important_metrics = [
            "response_time",
            "confidence_score",
            "cache_hit_rate",
            "accuracy",
        ]

        for metric_name in important_metrics:
            stats = metrics_collector.get_metric_statistics(
                metric_name, window_minutes=5
            )

            if not stats:
                continue

            baseline = metrics_collector.baselines.get(metric_name, {})

            # Verschiedene Anomalie-Erkennungsalgorithmen anwenden
            for algorithm_name, algorithm in self.detection_algorithms.items():
                anomaly = algorithm(metric_name, stats, baseline)
                if anomaly:
                    anomaly["detection_algorithm"] = algorithm_name
                    anomaly["timestamp"] = time.time()
                    anomalies.append(anomaly)
                    self.anomaly_history.append(anomaly)

        return anomalies

    def _statistical_anomaly_detection(
        self, metric_name: str, stats: Dict[str, float], baseline: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Statistische Anomalie-Erkennung"""

        if not baseline or "mean" not in baseline or "std" not in baseline:
            return None

        current_mean = stats.get("mean", 0)
        baseline_mean = baseline["mean"]
        baseline_std = baseline["std"]

        # Z-Score berechnen
        z_score = (
            abs(current_mean - baseline_mean) / baseline_std if baseline_std > 0 else 0
        )

        if z_score > self.sensitivity:
            return {
                "type": "statistical_anomaly",
                "metric": metric_name,
                "severity": min(z_score / self.sensitivity, 3.0),  # Cap bei 3.0
                "current_value": current_mean,
                "expected_value": baseline_mean,
                "z_score": z_score,
                "description": f"{metric_name} weicht {z_score:.2f} Standardabweichungen vom Baseline ab",
            }

        return None

    def _trend_anomaly_detection(
        self, metric_name: str, stats: Dict[str, float], baseline: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Trend-basierte Anomalie-Erkennung"""

        trend = stats.get("trend", "stable")

        # Definiere unerwünschte Trends für verschiedene Metriken
        bad_trends = {
            "response_time": ["increasing"],
            "confidence_score": ["decreasing"],
            "cache_hit_rate": ["decreasing"],
            "accuracy": ["decreasing"],
        }

        if metric_name in bad_trends and trend in bad_trends[metric_name]:
            return {
                "type": "trend_anomaly",
                "metric": metric_name,
                "severity": 1.5,
                "trend": trend,
                "description": f"{metric_name} zeigt unerwünschten Trend: {trend}",
            }

        return None

    def _pattern_anomaly_detection(
        self, metric_name: str, stats: Dict[str, float], baseline: Dict[str, float]
    ) -> Optional[Dict[str, Any]]:
        """Pattern-basierte Anomalie-Erkennung"""

        # Extrem hohe Variabilität
        if stats.get("std", 0) > baseline.get("std", 0) * 2:
            return {
                "type": "pattern_anomaly",
                "metric": metric_name,
                "severity": 1.0,
                "pattern": "high_variability",
                "current_std": stats.get("std", 0),
                "expected_std": baseline.get("std", 0),
                "description": f"{metric_name} zeigt ungewöhnlich hohe Variabilität",
            }

        return None


class OptimizationEngine:
    """KI-gestütztes Optimierungs-Engine"""

    def __init__(self):
        self.optimization_history: deque = deque(maxlen=1000)
        self.active_optimizations: Dict[str, OptimizationAction] = {}
        self.optimization_strategies = {
            "response_time": self._optimize_response_time,
            "cache_performance": self._optimize_cache_performance,
            "quality_scores": self._optimize_quality_scores,
            "resource_usage": self._optimize_resource_usage,
        }

        # Performance-Profile für verschiedene Szenarien
        self.performance_profiles: Dict[str, PerformanceProfile] = {}
        self._initialize_performance_profiles()

        # Aktuelle System-Parameter
        self.current_parameters = {
            "retrieval_k": 5,
            "cache_ttl_multiplier": 1.0,
            "confidence_threshold": 0.6,
            "max_query_length": 500,
            "neural_fusion_weight": 0.3,
            "semantic_similarity_threshold": 0.85,
        }

    def _initialize_performance_profiles(self):
        """Initialisiert vordefinierte Performance-Profile"""

        profiles = {
            "speed_optimized": PerformanceProfile(
                profile_name="speed_optimized",
                optimal_parameters={
                    "retrieval_k": 3,
                    "cache_ttl_multiplier": 2.0,
                    "confidence_threshold": 0.5,
                    "neural_fusion_weight": 0.1,
                },
                average_response_time=0.3,
                average_accuracy=0.78,
                resource_efficiency=0.9,
                user_satisfaction_score=0.75,
                usage_patterns={"quick_queries": 0.8, "simple_questions": 0.7},
                last_updated=time.time(),
            ),
            "quality_optimized": PerformanceProfile(
                profile_name="quality_optimized",
                optimal_parameters={
                    "retrieval_k": 8,
                    "cache_ttl_multiplier": 0.8,
                    "confidence_threshold": 0.75,
                    "neural_fusion_weight": 0.5,
                },
                average_response_time=0.8,
                average_accuracy=0.92,
                resource_efficiency=0.7,
                user_satisfaction_score=0.88,
                usage_patterns={"complex_queries": 0.9, "research_tasks": 0.85},
                last_updated=time.time(),
            ),
            "balanced": PerformanceProfile(
                profile_name="balanced",
                optimal_parameters={
                    "retrieval_k": 5,
                    "cache_ttl_multiplier": 1.2,
                    "confidence_threshold": 0.65,
                    "neural_fusion_weight": 0.3,
                },
                average_response_time=0.5,
                average_accuracy=0.85,
                resource_efficiency=0.8,
                user_satisfaction_score=0.82,
                usage_patterns={"mixed_queries": 0.8},
                last_updated=time.time(),
            ),
        }

        self.performance_profiles.update(profiles)

    def generate_optimizations(
        self, anomalies: List[Dict[str, Any]], metrics_collector: MetricsCollector
    ) -> List[OptimizationAction]:
        """Generiert Optimierungsmaßnahmen basierend auf Anomalien"""

        optimizations = []

        # Analysiere Anomalien und generiere entsprechende Optimierungen
        for anomaly in anomalies:
            metric_name = anomaly.get("metric", "")
            severity = anomaly.get("severity", 1.0)

            # Wähle passende Optimierungsstrategie
            if metric_name in ["response_time", "latency"]:
                optimization = self._optimize_response_time(anomaly, metrics_collector)
            elif metric_name in ["cache_hit_rate"]:
                optimization = self._optimize_cache_performance(
                    anomaly, metrics_collector
                )
            elif metric_name in ["confidence_score", "accuracy", "relevance_score"]:
                optimization = self._optimize_quality_scores(anomaly, metrics_collector)
            elif metric_name in ["memory_usage", "cpu_usage"]:
                optimization = self._optimize_resource_usage(anomaly, metrics_collector)
            else:
                continue

            if optimization:
                optimization.confidence *= (
                    1.0 + severity * 0.2
                )  # Höhere Confidence bei höherer Severity
                optimizations.append(optimization)

        # Priorisiere und filtere Optimierungen
        optimizations = self._prioritize_optimizations(optimizations)

        return optimizations

    def _optimize_response_time(
        self, anomaly: Dict[str, Any], metrics_collector: MetricsCollector
    ) -> Optional[OptimizationAction]:
        """Optimiert Response-Zeit"""

        current_stats = metrics_collector.get_metric_statistics(
            "response_time", window_minutes=5
        )
        current_time = current_stats.get("mean", 0.5)

        if current_time > 1.0:  # Langsam
            # Aggressive Cache-Optimierung
            new_cache_multiplier = self.current_parameters["cache_ttl_multiplier"] * 1.5
            new_retrieval_k = max(3, self.current_parameters["retrieval_k"] - 1)

            return OptimizationAction(
                action_id=f"response_time_opt_{int(time.time())}",
                action_type="parameter_tuning",
                component="retrieval_engine",
                parameters={
                    "cache_ttl_multiplier": new_cache_multiplier,
                    "retrieval_k": new_retrieval_k,
                    "neural_fusion_weight": 0.2,  # Reduziere aufwändige Berechnungen
                },
                expected_impact=0.3,  # 30% Verbesserung erwartet
                confidence=0.75,
                execution_time=0.1,
                rollback_data=dict(self.current_parameters),
            )

        return None

    def _optimize_cache_performance(
        self, anomaly: Dict[str, Any], metrics_collector: MetricsCollector
    ) -> Optional[OptimizationAction]:
        """Optimiert Cache-Performance"""

        current_stats = metrics_collector.get_metric_statistics(
            "cache_hit_rate", window_minutes=5
        )
        hit_rate = current_stats.get("mean", 0.8)

        if hit_rate < 0.7:  # Niedrige Hit-Rate
            # Cache-Strategien anpassen
            new_ttl_multiplier = self.current_parameters["cache_ttl_multiplier"] * 1.3
            new_similarity_threshold = (
                self.current_parameters["semantic_similarity_threshold"] * 0.95
            )

            return OptimizationAction(
                action_id=f"cache_opt_{int(time.time())}",
                action_type="cache_optimization",
                component="caching_system",
                parameters={
                    "cache_ttl_multiplier": new_ttl_multiplier,
                    "semantic_similarity_threshold": new_similarity_threshold,
                },
                expected_impact=0.25,
                confidence=0.8,
                execution_time=0.05,
                rollback_data=dict(self.current_parameters),
            )

        return None

    def _optimize_quality_scores(
        self, anomaly: Dict[str, Any], metrics_collector: MetricsCollector
    ) -> Optional[OptimizationAction]:
        """Optimiert Qualitäts-Scores"""

        metric_name = anomaly.get("metric", "")

        if metric_name == "confidence_score":
            # Erhöhe Retrieval-Depth und Neural-Fusion-Weight
            new_retrieval_k = min(10, self.current_parameters["retrieval_k"] + 2)
            new_neural_weight = min(
                0.6, self.current_parameters["neural_fusion_weight"] + 0.1
            )

            return OptimizationAction(
                action_id=f"quality_opt_{int(time.time())}",
                action_type="model_adjustment",
                component="answer_generator",
                parameters={
                    "retrieval_k": new_retrieval_k,
                    "neural_fusion_weight": new_neural_weight,
                    "confidence_threshold": self.current_parameters[
                        "confidence_threshold"
                    ]
                    + 0.05,
                },
                expected_impact=0.15,
                confidence=0.7,
                execution_time=0.2,
                rollback_data=dict(self.current_parameters),
            )

        return None

    def _optimize_resource_usage(
        self, anomaly: Dict[str, Any], metrics_collector: MetricsCollector
    ) -> Optional[OptimizationAction]:
        """Optimiert Resource-Usage"""

        # Reduziere aufwändige Operationen
        new_retrieval_k = max(3, self.current_parameters["retrieval_k"] - 1)
        new_neural_weight = max(
            0.1, self.current_parameters["neural_fusion_weight"] - 0.1
        )

        return OptimizationAction(
            action_id=f"resource_opt_{int(time.time())}",
            action_type="resource_optimization",
            component="system",
            parameters={
                "retrieval_k": new_retrieval_k,
                "neural_fusion_weight": new_neural_weight,
                "max_query_length": min(
                    300, self.current_parameters["max_query_length"]
                ),
            },
            expected_impact=0.2,
            confidence=0.65,
            execution_time=0.05,
            rollback_data=dict(self.current_parameters),
        )

    def _prioritize_optimizations(
        self, optimizations: List[OptimizationAction]
    ) -> List[OptimizationAction]:
        """Priorisiert Optimierungen nach Impact und Confidence"""

        def priority_score(opt: OptimizationAction) -> float:
            return opt.expected_impact * opt.confidence

        # Sortiere nach Priority-Score
        optimizations.sort(key=priority_score, reverse=True)

        # Entferne Duplikate und widersprüchliche Optimierungen
        unique_optimizations = []
        seen_components = set()

        for opt in optimizations:
            if opt.component not in seen_components:
                unique_optimizations.append(opt)
                seen_components.add(opt.component)

        return unique_optimizations[:5]  # Max 5 gleichzeitige Optimierungen

    def execute_optimization(self, optimization: OptimizationAction) -> Dict[str, Any]:
        """Führt eine Optimierung aus"""

        start_time = time.time()

        try:
            # Backup der aktuellen Parameter
            optimization.rollback_data = dict(self.current_parameters)

            # Parameter anwenden
            for param, value in optimization.parameters.items():
                if param in self.current_parameters:
                    self.current_parameters[param] = value

            # Tracking
            self.active_optimizations[optimization.action_id] = optimization

            execution_time = time.time() - start_time

            result = {
                "success": True,
                "action_id": optimization.action_id,
                "execution_time": execution_time,
                "applied_parameters": optimization.parameters,
                "message": f"Optimierung {optimization.action_type} erfolgreich angewendet",
            }

            # In Historie speichern
            self.optimization_history.append(
                {
                    "timestamp": time.time(),
                    "action": asdict(optimization),
                    "result": result,
                }
            )

            return result

        except Exception as e:
            return {
                "success": False,
                "action_id": optimization.action_id,
                "error": str(e),
                "message": f"Fehler bei Optimierung: {e}",
            }

    def rollback_optimization(self, action_id: str) -> Dict[str, Any]:
        """Rollt eine Optimierung zurück"""

        if action_id not in self.active_optimizations:
            return {"success": False, "message": "Optimierung nicht gefunden"}

        optimization = self.active_optimizations[action_id]

        try:
            # Parameter zurücksetzen
            for param, value in optimization.rollback_data.items():
                self.current_parameters[param] = value

            # Aus aktiven Optimierungen entfernen
            del self.active_optimizations[action_id]

            return {
                "success": True,
                "action_id": action_id,
                "message": "Optimierung erfolgreich zurück gesetzt",
            }

        except Exception as e:
            return {"success": False, "action_id": action_id, "error": str(e)}


class AutoOptimizingSystemManager:
    """Haupt-Manager für auto-optimierendes System"""

    def __init__(self, db_path: str = "auto_optimization.db"):
        self.db_path = db_path

        # Komponenten
        self.metrics_collector = MetricsCollector(collection_interval=2.0)
        self.anomaly_detector = AnomalyDetector(sensitivity=2.0)
        self.optimization_engine = OptimizationEngine()

        # Betriebsstatus
        self.is_running = False
        self.optimization_thread: Optional[threading.Thread] = None

        # Konfiguration
        self.config = {
            "optimization_interval": 30,  # Sekunden
            "anomaly_check_interval": 10,  # Sekunden
            "auto_execute_optimizations": True,
            "max_concurrent_optimizations": 3,
            "rollback_on_degradation": True,
            "learning_rate": 0.1,
        }

        # Performance-Tracking
        self.system_performance_history: deque = deque(maxlen=1000)
        self.optimization_results: deque = deque(maxlen=500)

        self._initialize_database()

    def _initialize_database(self):
        """Initialisiert SQLite-Datenbank für persistente Speicherung"""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                value REAL NOT NULL,
                timestamp REAL NOT NULL,
                category TEXT,
                component TEXT,
                metadata TEXT
            )
        """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS optimization_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_id TEXT NOT NULL,
                action_type TEXT,
                component TEXT,
                parameters TEXT,
                expected_impact REAL,
                actual_impact REAL,
                execution_time REAL,
                timestamp REAL,
                success BOOLEAN
            )
        """
        )

        conn.commit()
        conn.close()

    def start_auto_optimization(self):
        """Startet automatische Optimierung"""

        if self.is_running:
            print("⚠️ Auto-Optimierung läuft bereits")
            return

        self.is_running = True

        # Starte Metrik-Sammlung
        self.metrics_collector.start_collection()

        # Starte Optimierungs-Thread
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop, daemon=True
        )
        self.optimization_thread.start()

        print("🚀 Auto-Optimierendes System gestartet")
        print(
            f"📊 Metrik-Sammlung: {self.metrics_collector.collection_interval}s Intervall"
        )
        print(
            f"🔍 Anomalie-Prüfung: {self.config['anomaly_check_interval']}s Intervall"
        )
        print(f"⚡ Optimierung: {self.config['optimization_interval']}s Intervall")

    def stop_auto_optimization(self):
        """Stoppt automatische Optimierung"""

        self.is_running = False

        # Stoppe Metrik-Sammlung
        self.metrics_collector.stop_collection()

        # Warte auf Optimierungs-Thread
        if self.optimization_thread:
            self.optimization_thread.join(timeout=5.0)

        print("🛑 Auto-Optimierendes System gestoppt")

    def _optimization_loop(self):
        """Haupt-Loop für automatische Optimierung"""

        last_anomaly_check = 0
        last_optimization = 0

        while self.is_running:
            try:
                current_time = time.time()

                # Anomalie-Prüfung
                if (
                    current_time - last_anomaly_check
                    >= self.config["anomaly_check_interval"]
                ):
                    self._check_for_anomalies()
                    last_anomaly_check = current_time

                # Optimierung
                if (
                    current_time - last_optimization
                    >= self.config["optimization_interval"]
                ):
                    self._perform_optimization_cycle()
                    last_optimization = current_time

                # Performance-Tracking
                self._track_system_performance()

                time.sleep(1.0)  # Basis-Takt

            except Exception as e:
                print(f"Fehler in Optimierungs-Loop: {e}")
                time.sleep(5.0)

    def _check_for_anomalies(self):
        """Prüft auf Anomalien und reagiert entsprechend"""

        anomalies = self.anomaly_detector.detect_anomalies(self.metrics_collector)

        if anomalies:
            print(f"🚨 {len(anomalies)} Anomalien erkannt:")
            for anomaly in anomalies:
                print(
                    f"   • {anomaly['metric']}: {anomaly['description']} (Severity: {anomaly['severity']:.2f})"
                )

            # Sofortige Optimierung bei kritischen Anomalien
            critical_anomalies = [a for a in anomalies if a.get("severity", 0) > 2.0]
            if critical_anomalies and self.config["auto_execute_optimizations"]:
                print("⚡ Kritische Anomalien - führe sofortige Optimierung durch")
                self._generate_and_execute_optimizations(critical_anomalies)

    def _perform_optimization_cycle(self):
        """Führt einen vollständigen Optimierungszyklus durch"""

        # Sammle alle Anomalien der letzten Periode
        anomalies = self.anomaly_detector.detect_anomalies(self.metrics_collector)

        if not anomalies:
            return

        # Generiere Optimierungen
        optimizations = self.optimization_engine.generate_optimizations(
            anomalies, self.metrics_collector
        )

        if not optimizations:
            return

        print(f"🔧 {len(optimizations)} Optimierungen generiert")

        # Führe Optimierungen aus (falls aktiviert)
        if self.config["auto_execute_optimizations"]:
            self._execute_optimizations(optimizations)
        else:
            print(
                "📋 Optimierungen generiert, aber automatische Ausführung ist deaktiviert"
            )

    def _generate_and_execute_optimizations(self, anomalies: List[Dict[str, Any]]):
        """Generiert und führt Optimierungen für spezifische Anomalien aus"""

        optimizations = self.optimization_engine.generate_optimizations(
            anomalies, self.metrics_collector
        )

        if optimizations:
            self._execute_optimizations(optimizations)

    def _execute_optimizations(self, optimizations: List[OptimizationAction]):
        """Führt Liste von Optimierungen aus"""

        successful_optimizations = []

        for optimization in optimizations[
            : self.config["max_concurrent_optimizations"]
        ]:
            print(
                f"⚡ Führe Optimierung aus: {optimization.action_type} für {optimization.component}"
            )

            result = self.optimization_engine.execute_optimization(optimization)

            if result["success"]:
                successful_optimizations.append(optimization)
                print(f"✅ Optimierung erfolgreich: {result['message']}")
            else:
                print(f"❌ Optimierung fehlgeschlagen: {result['message']}")

            # Kurze Pause zwischen Optimierungen
            time.sleep(0.5)

        # Überwache Auswirkungen
        if successful_optimizations:
            self._monitor_optimization_impact(successful_optimizations)

    def _monitor_optimization_impact(self, optimizations: List[OptimizationAction]):
        """Überwacht die Auswirkungen von Optimierungen"""

        # Stelle Timer für Impact-Monitoring ein
        def check_impact():
            time.sleep(60)  # Warte 1 Minute

            for optimization in optimizations:
                impact = self._measure_optimization_impact(optimization)

                if impact["improved"]:
                    print(
                        f"✅ Optimierung {optimization.action_id} zeigt positive Auswirkungen"
                    )
                else:
                    print(
                        f"⚠️ Optimierung {optimization.action_id} zeigt keine/negative Auswirkungen"
                    )

                    if self.config["rollback_on_degradation"]:
                        result = self.optimization_engine.rollback_optimization(
                            optimization.action_id
                        )
                        if result["success"]:
                            print(
                                f"🔄 Optimierung {optimization.action_id} zurückgesetzt"
                            )

        # Starte Impact-Monitoring in separatem Thread
        monitor_thread = threading.Thread(target=check_impact, daemon=True)
        monitor_thread.start()

    def _measure_optimization_impact(
        self, optimization: OptimizationAction
    ) -> Dict[str, Any]:
        """Misst die tatsächliche Auswirkung einer Optimierung"""

        # Vereinfachte Impact-Messung
        # (In echter Implementierung würde dies komplexere A/B-Tests beinhalten)

        current_performance = self._get_current_performance_score()

        # Simuliere Verbesserung basierend auf erwarteter Auswirkung
        expected_improvement = optimization.expected_impact
        actual_improvement = np.random.normal(
            expected_improvement, expected_improvement * 0.3
        )

        improved = actual_improvement > 0.05  # 5% Mindestverbesserung

        return {
            "improved": improved,
            "expected_impact": expected_improvement,
            "actual_impact": actual_improvement,
            "performance_score": current_performance,
        }

    def _track_system_performance(self):
        """Tracked die Gesamt-System-Performance"""

        performance_score = self._get_current_performance_score()

        self.system_performance_history.append(
            {
                "timestamp": time.time(),
                "performance_score": performance_score,
                "active_optimizations": len(
                    self.optimization_engine.active_optimizations
                ),
                "system_parameters": dict(self.optimization_engine.current_parameters),
            }
        )

    def _get_current_performance_score(self) -> float:
        """Berechnet aktuellen Performance-Score"""

        # Sammle wichtige Metriken
        response_time_stats = self.metrics_collector.get_metric_statistics(
            "response_time", window_minutes=5
        )
        confidence_stats = self.metrics_collector.get_metric_statistics(
            "confidence_score", window_minutes=5
        )
        cache_stats = self.metrics_collector.get_metric_statistics(
            "cache_hit_rate", window_minutes=5
        )

        # Berechne gewichteten Performance-Score
        score = 0.0
        weight_sum = 0.0

        if response_time_stats:
            # Niedrigere Response-Zeit = höherer Score
            rt_score = max(
                0, 1.0 - (response_time_stats["mean"] / 2.0)
            )  # Normalisiere auf 2s
            score += rt_score * 0.3
            weight_sum += 0.3

        if confidence_stats:
            score += confidence_stats["mean"] * 0.4
            weight_sum += 0.4

        if cache_stats:
            score += cache_stats["mean"] * 0.3
            weight_sum += 0.3

        return score / weight_sum if weight_sum > 0 else 0.5

    def get_system_status(self) -> Dict[str, Any]:
        """Gibt aktuellen System-Status zurück"""

        return {
            "is_running": self.is_running,
            "current_performance_score": self._get_current_performance_score(),
            "active_optimizations": len(self.optimization_engine.active_optimizations),
            "total_metrics_collected": len(self.metrics_collector.metrics_buffer),
            "anomalies_detected_total": len(self.anomaly_detector.anomaly_history),
            "optimizations_executed_total": len(
                self.optimization_engine.optimization_history
            ),
            "current_parameters": dict(self.optimization_engine.current_parameters),
            "recent_anomalies": list(self.anomaly_detector.anomaly_history)[-5:],
            "system_health": self._assess_system_health(),
        }

    def _assess_system_health(self) -> str:
        """Bewertet Gesamt-System-Gesundheit"""

        performance_score = self._get_current_performance_score()
        recent_anomalies = len(
            [
                a
                for a in self.anomaly_detector.anomaly_history
                if time.time() - a.get("timestamp", 0) < 300
            ]
        )  # Letzte 5 Minuten

        if performance_score > 0.8 and recent_anomalies == 0:
            return "excellent"
        elif performance_score > 0.6 and recent_anomalies <= 2:
            return "good"
        elif performance_score > 0.4 and recent_anomalies <= 5:
            return "fair"
        else:
            return "poor"


# === TESTING & DEMO ===


def demonstrate_auto_optimization():
    """Demonstriert das Auto-Optimizing System"""

    print("🎯 AUTO-OPTIMIZING SYSTEM MANAGER - DEMO")
    print("=" * 50)

    # Initialize Manager
    manager = AutoOptimizingSystemManager()

    print("🚀 Starte Auto-Optimierung...")
    manager.start_auto_optimization()

    try:
        # Lasse System für 30 Sekunden laufen
        print("⏱️ Lasse System 30 Sekunden laufen...")

        for i in range(6):
            time.sleep(5)
            status = manager.get_system_status()

            print(f"\n📊 Status nach {(i+1)*5}s:")
            print(f"   Performance-Score: {status['current_performance_score']:.3f}")
            print(f"   Aktive Optimierungen: {status['active_optimizations']}")
            print(f"   Gesammelte Metriken: {status['total_metrics_collected']}")
            print(f"   System-Gesundheit: {status['system_health']}")

            if status["recent_anomalies"]:
                print(f"   Aktuelle Anomalien: {len(status['recent_anomalies'])}")

        # Finale Status-Ausgabe
        print("\n🏁 FINALE SYSTEM-ANALYSE:")
        final_status = manager.get_system_status()

        print(f"📈 Performance-Score: {final_status['current_performance_score']:.3f}")
        print(
            f"🔧 Optimierungen ausgeführt: {final_status['optimizations_executed_total']}"
        )
        print(f"🚨 Anomalien erkannt: {final_status['anomalies_detected_total']}")
        print(f"💚 System-Gesundheit: {final_status['system_health']}")

        print("\n⚙️ Aktuelle Parameter:")
        for param, value in final_status["current_parameters"].items():
            print(f"   • {param}: {value}")

    finally:
        print("\n🛑 Stoppe Auto-Optimierung...")
        manager.stop_auto_optimization()
        print("✅ Demo beendet")


if __name__ == "__main__":
    demonstrate_auto_optimization()
