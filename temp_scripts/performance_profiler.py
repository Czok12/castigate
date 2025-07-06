"""
⚡ PERFORMANCE PROFILER & BOTTLENECK ANALYZER
==========================================

Detaillierte Performance-Analyse und Bottleneck-Erkennung für RAG-Systeme
"""

import gc
import statistics
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import psutil


@dataclass
class PerformanceSnapshot:
    """Momentaufnahme der System-Performance"""

    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_available: float
    thread_count: int
    active_processes: int

    # RAG-spezifische Metriken
    vectorstore_size: Optional[int] = None
    cache_hit_rate: Optional[float] = None
    llm_response_time: Optional[float] = None
    retrieval_time: Optional[float] = None


@dataclass
class ComponentProfile:
    """Performance-Profil einer Komponente"""

    component_name: str
    call_count: int = 0
    total_time: float = 0.0
    min_time: float = float("inf")
    max_time: float = 0.0
    avg_time: float = 0.0
    recent_times: deque = field(default_factory=lambda: deque(maxlen=100))
    error_count: int = 0
    last_error: Optional[str] = None
    memory_usage: List[float] = field(default_factory=list)

    def update(
        self,
        execution_time: float,
        memory_delta: float = 0.0,
        error: Optional[str] = None,
    ):
        """Aktualisiert Profil-Daten"""
        self.call_count += 1
        self.total_time += execution_time
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.avg_time = self.total_time / self.call_count
        self.recent_times.append(execution_time)

        if memory_delta != 0.0:
            self.memory_usage.append(memory_delta)

        if error:
            self.error_count += 1
            self.last_error = error


class PerformanceProfiler:
    """Detaillierter Performance-Profiler"""

    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.is_monitoring = False
        self.monitoring_thread = None

        # Performance-Daten
        self.system_snapshots: deque = deque(maxlen=1000)
        self.component_profiles: Dict[str, ComponentProfile] = {}

        # Bottleneck-Tracking
        self.bottlenecks = defaultdict(list)
        self.performance_alerts = []

        # Thresholds für Alerts
        self.thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "response_time": 5.0,
            "error_rate": 5.0,  # Prozent
        }

    def start_monitoring(self):
        """Startet kontinuierliches System-Monitoring"""
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop, daemon=True
            )
            self.monitoring_thread.start()

    def stop_monitoring(self):
        """Stoppt System-Monitoring"""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=2.0)

    def _monitoring_loop(self):
        """Haupt-Monitoring-Loop"""
        while self.is_monitoring:
            try:
                snapshot = self._take_system_snapshot()
                self.system_snapshots.append(snapshot)
                self._check_performance_alerts(snapshot)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.monitoring_interval)

    def _take_system_snapshot(self) -> PerformanceSnapshot:
        """Erstellt System-Performance-Snapshot"""

        # System-Metriken
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        memory_available = memory.available / (1024**3)  # GB

        # Process-Metriken
        current_process = psutil.Process()
        thread_count = current_process.num_threads()
        active_processes = len(psutil.pids())

        return PerformanceSnapshot(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            memory_available=memory_available,
            thread_count=thread_count,
            active_processes=active_processes,
        )

    def _check_performance_alerts(self, snapshot: PerformanceSnapshot):
        """Prüft auf Performance-Probleme"""

        alerts = []

        if snapshot.cpu_usage > self.thresholds["cpu_usage"]:
            alerts.append(
                {
                    "type": "HIGH_CPU",
                    "value": snapshot.cpu_usage,
                    "threshold": self.thresholds["cpu_usage"],
                    "timestamp": snapshot.timestamp,
                }
            )

        if snapshot.memory_usage > self.thresholds["memory_usage"]:
            alerts.append(
                {
                    "type": "HIGH_MEMORY",
                    "value": snapshot.memory_usage,
                    "threshold": self.thresholds["memory_usage"],
                    "timestamp": snapshot.timestamp,
                }
            )

        if snapshot.memory_available < 1.0:  # Weniger als 1GB verfügbar
            alerts.append(
                {
                    "type": "LOW_MEMORY_AVAILABLE",
                    "value": snapshot.memory_available,
                    "threshold": 1.0,
                    "timestamp": snapshot.timestamp,
                }
            )

        self.performance_alerts.extend(alerts)
        # Nur letzte 100 Alerts behalten
        self.performance_alerts = self.performance_alerts[-100:]

    def profile_function(self, component_name: str):
        """Decorator für Funktions-Profiling"""

        def decorator(func: Callable) -> Callable:
            def wrapper(*args, **kwargs):
                return self._profile_execution(component_name, func, *args, **kwargs)

            return wrapper

        return decorator

    def _profile_execution(
        self, component_name: str, func: Callable, *args, **kwargs
    ) -> Any:
        """Führt Funktion mit Profiling aus"""

        # Memory vor Ausführung
        gc.collect()  # Garbage Collection für saubere Messung
        memory_before = psutil.Process().memory_info().rss / (1024**2)  # MB

        start_time = time.time()
        error = None
        result = None

        try:
            result = func(*args, **kwargs)
        except Exception as e:
            error = str(e)
            raise
        finally:
            execution_time = time.time() - start_time

            # Memory nach Ausführung
            memory_after = psutil.Process().memory_info().rss / (1024**2)  # MB
            memory_delta = memory_after - memory_before

            # Profil aktualisieren
            if component_name not in self.component_profiles:
                self.component_profiles[component_name] = ComponentProfile(
                    component_name
                )

            self.component_profiles[component_name].update(
                execution_time, memory_delta, error
            )

            # Bottleneck-Erkennung
            self._detect_bottlenecks(component_name, execution_time)

        return result

    def _detect_bottlenecks(self, component_name: str, execution_time: float):
        """Erkennt Performance-Bottlenecks"""

        profile = self.component_profiles[component_name]

        # Slow-Response-Detection
        if execution_time > self.thresholds["response_time"]:
            self.bottlenecks["slow_responses"].append(
                {
                    "component": component_name,
                    "time": execution_time,
                    "timestamp": time.time(),
                }
            )

        # Performance-Regression-Detection
        if len(profile.recent_times) >= 10:
            recent_avg = statistics.mean(list(profile.recent_times)[-10:])
            overall_avg = profile.avg_time

            if recent_avg > overall_avg * 1.5:  # 50% Verschlechterung
                self.bottlenecks["performance_regression"].append(
                    {
                        "component": component_name,
                        "recent_avg": recent_avg,
                        "overall_avg": overall_avg,
                        "degradation": (recent_avg / overall_avg - 1) * 100,
                        "timestamp": time.time(),
                    }
                )

        # Memory-Leak-Detection
        if len(profile.memory_usage) >= 10:
            recent_memory = profile.memory_usage[-10:]
            if all(m > 0 for m in recent_memory):  # Kontinuierlicher Anstieg
                total_increase = sum(recent_memory)
                if total_increase > 100:  # > 100MB Anstieg
                    self.bottlenecks["memory_leaks"].append(
                        {
                            "component": component_name,
                            "memory_increase": total_increase,
                            "timestamp": time.time(),
                        }
                    )

    def get_performance_report(self) -> Dict:
        """Erstellt umfassenden Performance-Report"""

        # System-Übersicht
        system_overview = self._get_system_overview()

        # Component-Analyse
        component_analysis = self._get_component_analysis()

        # Bottleneck-Analyse
        bottleneck_analysis = self._get_bottleneck_analysis()

        # Optimierungsempfehlungen
        recommendations = self._generate_recommendations()

        return {
            "timestamp": time.time(),
            "system_overview": system_overview,
            "component_analysis": component_analysis,
            "bottleneck_analysis": bottleneck_analysis,
            "recommendations": recommendations,
            "alerts": self.performance_alerts[-10:],  # Letzte 10 Alerts
        }

    def _get_system_overview(self) -> Dict:
        """System-Performance-Übersicht"""

        if not self.system_snapshots:
            return {"status": "no_data"}

        recent_snapshots = list(self.system_snapshots)[-10:]  # Letzte 10

        return {
            "current_cpu": recent_snapshots[-1].cpu_usage,
            "avg_cpu": statistics.mean([s.cpu_usage for s in recent_snapshots]),
            "max_cpu": max([s.cpu_usage for s in recent_snapshots]),
            "current_memory": recent_snapshots[-1].memory_usage,
            "avg_memory": statistics.mean([s.memory_usage for s in recent_snapshots]),
            "max_memory": max([s.memory_usage for s in recent_snapshots]),
            "memory_available": recent_snapshots[-1].memory_available,
            "thread_count": recent_snapshots[-1].thread_count,
            "monitoring_duration": len(self.system_snapshots)
            * self.monitoring_interval
            / 60,  # Minuten
            "total_snapshots": len(self.system_snapshots),
        }

    def _get_component_analysis(self) -> Dict:
        """Komponenten-Performance-Analyse"""

        components = {}

        for name, profile in self.component_profiles.items():

            # Performance-Statistiken
            recent_times = list(profile.recent_times) if profile.recent_times else [0]
            error_rate = (profile.error_count / max(profile.call_count, 1)) * 100

            # Trend-Analyse
            trend = "stable"
            if len(recent_times) >= 5:
                first_half = recent_times[: len(recent_times) // 2]
                second_half = recent_times[len(recent_times) // 2 :]

                if len(first_half) > 0 and len(second_half) > 0:
                    first_avg = statistics.mean(first_half)
                    second_avg = statistics.mean(second_half)

                    if second_avg > first_avg * 1.2:
                        trend = "degrading"
                    elif second_avg < first_avg * 0.8:
                        trend = "improving"

            components[name] = {
                "call_count": profile.call_count,
                "total_time": round(profile.total_time, 3),
                "avg_time": round(profile.avg_time, 3),
                "min_time": round(profile.min_time, 3),
                "max_time": round(profile.max_time, 3),
                "recent_avg": round(statistics.mean(recent_times), 3),
                "error_count": profile.error_count,
                "error_rate": round(error_rate, 2),
                "last_error": profile.last_error,
                "memory_usage": {
                    "total": (
                        round(sum(profile.memory_usage), 2)
                        if profile.memory_usage
                        else 0
                    ),
                    "avg": (
                        round(statistics.mean(profile.memory_usage), 2)
                        if profile.memory_usage
                        else 0
                    ),
                    "max": (
                        round(max(profile.memory_usage), 2)
                        if profile.memory_usage
                        else 0
                    ),
                },
                "trend": trend,
            }

        # Sortierung nach Gesamt-Zeit
        sorted_components = dict(
            sorted(components.items(), key=lambda x: x[1]["total_time"], reverse=True)
        )

        return sorted_components

    def _get_bottleneck_analysis(self) -> Dict:
        """Bottleneck-Analyse"""

        analysis = {}

        for bottleneck_type, incidents in self.bottlenecks.items():
            if incidents:
                recent_incidents = [
                    i for i in incidents if time.time() - i["timestamp"] < 3600
                ]  # Letzte Stunde

                analysis[bottleneck_type] = {
                    "total_incidents": len(incidents),
                    "recent_incidents": len(recent_incidents),
                    "latest_incident": incidents[-1] if incidents else None,
                    "affected_components": list(
                        set([i["component"] for i in recent_incidents])
                    ),
                }

        return analysis

    def _generate_recommendations(self) -> List[Dict]:
        """Generiert Performance-Optimierungsempfehlungen"""

        recommendations = []

        # CPU-Optimierung
        if self.system_snapshots:
            recent_cpu = [s.cpu_usage for s in list(self.system_snapshots)[-10:]]
            avg_cpu = statistics.mean(recent_cpu)

            if avg_cpu > 70:
                recommendations.append(
                    {
                        "type": "cpu_optimization",
                        "priority": "high",
                        "description": f"CPU-Auslastung hoch ({avg_cpu:.1f}%)",
                        "suggestion": "Erwägen Sie Parallelisierung oder Caching für CPU-intensive Operationen",
                    }
                )

        # Memory-Optimierung
        for name, profile in self.component_profiles.items():
            if profile.memory_usage:
                avg_memory = statistics.mean(profile.memory_usage)
                if avg_memory > 50:  # > 50MB pro Call
                    recommendations.append(
                        {
                            "type": "memory_optimization",
                            "priority": "medium",
                            "description": f"Komponente '{name}' verbraucht viel Speicher ({avg_memory:.1f}MB)",
                            "suggestion": "Prüfen Sie Memory-Leaks oder optimieren Sie Datenstrukturen",
                        }
                    )

        # Performance-Optimierung
        for name, profile in self.component_profiles.items():
            if profile.avg_time > 2.0:  # > 2 Sekunden
                recommendations.append(
                    {
                        "type": "performance_optimization",
                        "priority": "high",
                        "description": f"Komponente '{name}' ist langsam ({profile.avg_time:.2f}s)",
                        "suggestion": "Implementieren Sie Caching oder optimieren Sie Algorithmen",
                    }
                )

        # Error-Rate-Optimierung
        for name, profile in self.component_profiles.items():
            error_rate = (profile.error_count / max(profile.call_count, 1)) * 100
            if error_rate > 5:  # > 5% Fehlerrate
                recommendations.append(
                    {
                        "type": "reliability_optimization",
                        "priority": "high",
                        "description": f"Komponente '{name}' hat hohe Fehlerrate ({error_rate:.1f}%)",
                        "suggestion": "Verbessern Sie Error-Handling und Retry-Mechanismen",
                    }
                )

        # Bottleneck-spezifische Empfehlungen
        if "slow_responses" in self.bottlenecks and self.bottlenecks["slow_responses"]:
            slow_components = set(
                [b["component"] for b in self.bottlenecks["slow_responses"]]
            )
            for component in slow_components:
                recommendations.append(
                    {
                        "type": "bottleneck_resolution",
                        "priority": "high",
                        "description": f"Komponente '{component}' verursacht langsame Antworten",
                        "suggestion": "Implementieren Sie Async-Processing oder Query-Optimierung",
                    }
                )

        return recommendations

    def get_real_time_metrics(self) -> Dict:
        """Liefert Echtzeit-Performance-Metriken"""

        if not self.system_snapshots:
            return {"status": "no_data"}

        latest = self.system_snapshots[-1]

        # Top-Performance-Components
        top_components = []
        for name, profile in self.component_profiles.items():
            if profile.call_count > 0:
                top_components.append(
                    {
                        "name": name,
                        "avg_time": profile.avg_time,
                        "call_count": profile.call_count,
                        "error_rate": (profile.error_count / profile.call_count) * 100,
                    }
                )

        top_components.sort(key=lambda x: x["avg_time"], reverse=True)

        return {
            "timestamp": latest.timestamp,
            "system": {
                "cpu_usage": latest.cpu_usage,
                "memory_usage": latest.memory_usage,
                "memory_available_gb": latest.memory_available,
                "thread_count": latest.thread_count,
            },
            "top_components": top_components[:5],
            "recent_alerts": len(
                [
                    a
                    for a in self.performance_alerts
                    if time.time() - a["timestamp"] < 300
                ]
            ),  # Letzte 5 Min
            "active_bottlenecks": len(
                [
                    b
                    for bottlenecks in self.bottlenecks.values()
                    for b in bottlenecks
                    if time.time() - b["timestamp"] < 3600
                ]
            ),  # Letzte Stunde
        }

    def reset_statistics(self):
        """Setzt alle Statistiken zurück"""
        self.system_snapshots.clear()
        self.component_profiles.clear()
        self.bottlenecks.clear()
        self.performance_alerts.clear()

    def export_performance_data(self, filepath: str):
        """Exportiert Performance-Daten"""

        export_data = {
            "export_timestamp": time.time(),
            "system_snapshots": [
                {
                    "timestamp": s.timestamp,
                    "cpu_usage": s.cpu_usage,
                    "memory_usage": s.memory_usage,
                    "memory_available": s.memory_available,
                    "thread_count": s.thread_count,
                }
                for s in self.system_snapshots
            ],
            "component_profiles": {
                name: {
                    "call_count": profile.call_count,
                    "total_time": profile.total_time,
                    "avg_time": profile.avg_time,
                    "min_time": profile.min_time,
                    "max_time": profile.max_time,
                    "error_count": profile.error_count,
                    "memory_usage": profile.memory_usage,
                }
                for name, profile in self.component_profiles.items()
            },
            "bottlenecks": dict(self.bottlenecks),
            "performance_alerts": self.performance_alerts,
        }

        import json

        with open(filepath, "w") as f:
            json.dump(export_data, f, indent=2, default=str)
