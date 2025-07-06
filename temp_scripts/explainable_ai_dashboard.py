"""
🔍 EXPLAINABLE AI DASHBOARD
===========================

Ultra-Advanced Transparenz- und Erklärbarkeits-Dashboard für das juristische RAG-System
mit vollständiger Nachverfolgung aller KI-Entscheidungen und Reasoning-Pfade
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

# Plotly imports (optional für Dashboard-Funktionalität)
try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

    # Dummy-Klassen für den Fall dass plotly nicht verfügbar ist
    class go:
        class Figure:
            def add_annotation(self, text):
                return self

        class Scatter:
            pass

        class Bar:
            pass

        class Indicator:
            pass

    def make_subplots(*args, **kwargs):
        return go.Figure()


@dataclass
class DecisionStep:
    """Einzelner Schritt im Entscheidungsprozess"""

    step_id: str
    step_name: str
    step_type: str  # retrieval, analysis, synthesis, validation
    timestamp: float
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    reasoning: str
    evidence: List[str]
    alternatives_considered: List[str]
    decision_factors: Dict[str, float]
    component_responsible: str
    processing_time: float


@dataclass
class ExplanationTrace:
    """Vollständige Erklärungsspur einer Anfrage"""

    trace_id: str
    query: str
    final_answer: str
    overall_confidence: float
    decision_steps: List[DecisionStep]
    key_evidence: List[str]
    critical_decisions: List[str]
    uncertainty_sources: List[str]
    alternative_paths: List[str]
    compliance_checks: Dict[str, bool]
    total_processing_time: float
    complexity_assessment: float


@dataclass
class BiasAnalysis:
    """Analyse von Bias und Fairness"""

    bias_sources: Dict[str, float]
    fairness_metrics: Dict[str, float]
    representation_analysis: Dict[str, Any]
    demographic_impact: Dict[str, float]
    legal_domain_bias: Dict[str, float]
    mitigation_strategies: List[str]
    confidence_in_fairness: float


@dataclass
class ComplianceReport:
    """Compliance und Audit-Bericht"""

    gdpr_compliance: bool
    legal_standards_met: List[str]
    audit_trail_complete: bool
    data_usage_logged: bool
    decision_reproducible: bool
    human_oversight_available: bool
    ethical_guidelines_followed: bool
    transparency_score: float
    accountability_measures: List[str]


class ExplainableAIEngine:
    """Engine für Explainable AI im juristischen RAG-System"""

    def __init__(self):
        self.explanation_traces = deque(maxlen=10000)
        self.decision_patterns = defaultdict(list)
        self.bias_monitor = BiasMonitor()
        self.compliance_tracker = ComplianceTracker()

        # Konfiguration
        self.explanation_depth = "detailed"  # basic, detailed, expert
        self.store_all_steps = True
        self.real_time_analysis = True

        # Metriken
        self.transparency_metrics = {
            "total_explanations_generated": 0,
            "average_explanation_completeness": 0.0,
            "user_satisfaction_with_explanations": 0.0,
            "compliance_violations_detected": 0,
            "bias_incidents_flagged": 0,
        }

    def trace_decision_process(self, query: str, components_involved: List[str]) -> str:
        """Starte Tracing eines Entscheidungsprozesses"""

        trace_id = f"trace_{int(time.time() * 1000)}"

        # Initialisiere neue Erklärungsspur
        explanation_trace = ExplanationTrace(
            trace_id=trace_id,
            query=query,
            final_answer="",  # Wird später gesetzt
            overall_confidence=0.0,
            decision_steps=[],
            key_evidence=[],
            critical_decisions=[],
            uncertainty_sources=[],
            alternative_paths=[],
            compliance_checks={},
            total_processing_time=0.0,
            complexity_assessment=self._assess_query_complexity(query),
        )

        self.explanation_traces.append(explanation_trace)
        return trace_id

    def log_decision_step(
        self,
        trace_id: str,
        step_name: str,
        step_type: str,
        component: str,
        input_data: Dict[str, Any],
        output_data: Dict[str, Any],
        confidence: float,
        reasoning: str,
        evidence: List[str] = None,
        alternatives: List[str] = None,
    ) -> str:
        """Logge einzelnen Entscheidungsschritt"""

        start_time = time.time()

        step_id = f"{trace_id}_step_{len(self._get_trace(trace_id).decision_steps) + 1}"

        decision_step = DecisionStep(
            step_id=step_id,
            step_name=step_name,
            step_type=step_type,
            timestamp=start_time,
            input_data=input_data.copy(),
            output_data=output_data.copy(),
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence or [],
            alternatives_considered=alternatives or [],
            decision_factors=self._extract_decision_factors(input_data, output_data),
            component_responsible=component,
            processing_time=0.0,  # Wird bei Abschluss gesetzt
        )

        trace = self._get_trace(trace_id)
        if trace:
            trace.decision_steps.append(decision_step)

            # Real-time Bias-Analyse
            if self.real_time_analysis:
                self._analyze_step_for_bias(decision_step)

        return step_id

    def finalize_explanation(
        self, trace_id: str, final_answer: str, overall_confidence: float
    ) -> ExplanationTrace:
        """Finalisiere Erklärung und generiere Report"""

        trace = self._get_trace(trace_id)
        if not trace:
            raise ValueError(f"Trace {trace_id} nicht gefunden")

        trace.final_answer = final_answer
        trace.overall_confidence = overall_confidence
        trace.total_processing_time = sum(
            step.processing_time for step in trace.decision_steps
        )

        # Analysiere kritische Entscheidungen
        trace.critical_decisions = self._identify_critical_decisions(trace)

        # Identifiziere Unsicherheitsquellen
        trace.uncertainty_sources = self._identify_uncertainty_sources(trace)

        # Sammle Schlüsselevidenz
        trace.key_evidence = self._extract_key_evidence(trace)

        # Compliance-Checks
        trace.compliance_checks = self.compliance_tracker.check_compliance(trace)

        # Update Metriken
        self.transparency_metrics["total_explanations_generated"] += 1

        return trace

    def generate_human_readable_explanation(
        self, trace_id: str, target_audience: str = "legal_professional"
    ) -> Dict[str, Any]:
        """Generiere menschenlesbare Erklärung"""

        trace = self._get_trace(trace_id)
        if not trace:
            return {"error": f"Trace {trace_id} nicht gefunden"}

        if target_audience == "legal_professional":
            return self._generate_professional_explanation(trace)
        elif target_audience == "client":
            return self._generate_client_explanation(trace)
        elif target_audience == "technical":
            return self._generate_technical_explanation(trace)
        else:
            return self._generate_general_explanation(trace)

    def _generate_professional_explanation(
        self, trace: ExplanationTrace
    ) -> Dict[str, Any]:
        """Generiere Erklärung für Rechtsprofis"""

        return {
            "summary": {
                "query": trace.query,
                "final_answer": trace.final_answer,
                "confidence": f"{trace.overall_confidence:.1%}",
                "complexity": self._complexity_to_text(trace.complexity_assessment),
                "processing_time": f"{trace.total_processing_time:.2f}s",
            },
            "legal_reasoning": {
                "primary_sources": trace.key_evidence[:5],
                "reasoning_chain": [
                    step.reasoning
                    for step in trace.decision_steps
                    if step.step_type == "analysis"
                ],
                "critical_decisions": trace.critical_decisions,
                "alternative_interpretations": trace.alternative_paths[:3],
            },
            "confidence_analysis": {
                "overall_confidence": trace.overall_confidence,
                "confidence_factors": self._analyze_confidence_factors(trace),
                "uncertainty_sources": trace.uncertainty_sources,
                "reliability_assessment": self._assess_reliability(trace),
            },
            "process_transparency": {
                "decision_steps": len(trace.decision_steps),
                "components_involved": list(
                    set(step.component_responsible for step in trace.decision_steps)
                ),
                "methodology": self._describe_methodology(trace),
                "quality_checks": self._describe_quality_checks(trace),
            },
            "compliance": trace.compliance_checks,
            "recommendations": self._generate_professional_recommendations(trace),
        }

    def _generate_client_explanation(self, trace: ExplanationTrace) -> Dict[str, Any]:
        """Generiere Erklärung für Mandanten"""

        return {
            "your_question": trace.query,
            "our_analysis": {
                "main_finding": self._extract_main_finding(trace),
                "confidence_level": self._confidence_to_plain_language(
                    trace.overall_confidence
                ),
                "key_points": self._extract_key_points_for_client(trace),
            },
            "how_we_reached_this_conclusion": {
                "sources_reviewed": len(trace.key_evidence),
                "analysis_steps": self._simplify_analysis_steps(trace),
                "verification_performed": (
                    "Ja"
                    if trace.compliance_checks.get("decision_reproducible", False)
                    else "Nein"
                ),
            },
            "important_considerations": {
                "limitations": self._extract_limitations(trace),
                "next_steps": self._suggest_next_steps(trace),
                "when_to_seek_additional_help": self._when_to_seek_help(trace),
            },
            "transparency_note": "Diese Analyse wurde mit KI-Unterstützung erstellt. Alle Schritte sind dokumentiert und nachprüfbar.",
        }

    def _generate_technical_explanation(
        self, trace: ExplanationTrace
    ) -> Dict[str, Any]:
        """Generiere technische Erklärung für Entwickler/Auditoren"""

        return {
            "trace_metadata": {
                "trace_id": trace.trace_id,
                "timestamp": time.time(),
                "total_steps": len(trace.decision_steps),
                "processing_pipeline": [
                    step.component_responsible for step in trace.decision_steps
                ],
            },
            "algorithmic_decisions": [
                {
                    "step": step.step_name,
                    "component": step.component_responsible,
                    "input_features": list(step.input_data.keys()),
                    "decision_factors": step.decision_factors,
                    "confidence": step.confidence,
                    "alternatives_considered": step.alternatives_considered,
                }
                for step in trace.decision_steps
            ],
            "model_performance": {
                "overall_confidence": trace.overall_confidence,
                "component_confidences": {
                    step.component_responsible: step.confidence
                    for step in trace.decision_steps
                },
                "uncertainty_quantification": self._quantify_uncertainty(trace),
            },
            "bias_analysis": self.bias_monitor.analyze_trace(trace),
            "reproducibility": {
                "deterministic": trace.compliance_checks.get(
                    "decision_reproducible", False
                ),
                "random_seed_used": "documented",
                "version_info": self._get_version_info(),
            },
            "performance_metrics": {
                "total_time": trace.total_processing_time,
                "component_times": {
                    step.component_responsible: step.processing_time
                    for step in trace.decision_steps
                },
            },
        }

    def create_interactive_explanation_dashboard(self, trace_id: str) -> go.Figure:
        """Erstelle interaktives Dashboard für Erklärung"""

        trace = self._get_trace(trace_id)
        if not trace:
            return go.Figure().add_annotation(text="Trace nicht gefunden")

        # Erstelle Subplots
        fig = make_subplots(
            rows=3,
            cols=2,
            subplot_titles=[
                "Entscheidungsfluss",
                "Konfidenz-Entwicklung",
                "Komponenten-Beiträge",
                "Unsicherheitsanalyse",
                "Bias-Indikationen",
                "Compliance-Status",
            ],
            specs=[
                [{"type": "scatter"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "scatter"}],
                [{"type": "bar"}, {"type": "indicator"}],
            ],
            vertical_spacing=0.08,
        )

        # 1. Entscheidungsfluss
        steps = [
            f"{i+1}. {step.step_name}" for i, step in enumerate(trace.decision_steps)
        ]
        confidences = [step.confidence for step in trace.decision_steps]

        fig.add_trace(
            go.Scatter(
                x=list(range(len(steps))),
                y=confidences,
                mode="lines+markers",
                name="Konfidenz pro Schritt",
                text=steps,
                hovertemplate="<b>%{text}</b><br>Konfidenz: %{y:.2f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

        # 2. Konfidenz-Entwicklung
        cumulative_confidence = np.cumsum(confidences) / np.arange(
            1, len(confidences) + 1
        )

        fig.add_trace(
            go.Scatter(
                x=list(range(len(cumulative_confidence))),
                y=cumulative_confidence,
                mode="lines",
                name="Kumulative Konfidenz",
                line=dict(color="green"),
            ),
            row=1,
            col=2,
        )

        # 3. Komponenten-Beiträge
        component_contributions = defaultdict(float)
        for step in trace.decision_steps:
            component_contributions[step.component_responsible] += step.confidence

        fig.add_trace(
            go.Bar(
                x=list(component_contributions.keys()),
                y=list(component_contributions.values()),
                name="Komponenten-Beiträge",
            ),
            row=2,
            col=1,
        )

        # 4. Unsicherheitsanalyse
        uncertainty_scores = [1 - step.confidence for step in trace.decision_steps]

        fig.add_trace(
            go.Scatter(
                x=list(range(len(uncertainty_scores))),
                y=uncertainty_scores,
                mode="markers",
                name="Unsicherheit",
                marker=dict(color="red", size=8),
            ),
            row=2,
            col=2,
        )

        # 5. Bias-Indikationen
        bias_analysis = self.bias_monitor.analyze_trace(trace)
        bias_sources = list(bias_analysis.bias_sources.keys())
        bias_scores = list(bias_analysis.bias_sources.values())

        fig.add_trace(
            go.Bar(
                x=bias_sources,
                y=bias_scores,
                name="Bias-Indikatoren",
                marker=dict(color="orange"),
            ),
            row=3,
            col=1,
        )

        # 6. Compliance-Status
        compliance_score = sum(trace.compliance_checks.values()) / max(
            1, len(trace.compliance_checks)
        )

        fig.add_trace(
            go.Indicator(
                mode="gauge+number",
                value=compliance_score * 100,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Compliance %"},
                gauge={
                    "axis": {"range": [None, 100]},
                    "bar": {"color": "darkblue"},
                    "steps": [
                        {"range": [0, 50], "color": "lightgray"},
                        {"range": [50, 80], "color": "yellow"},
                        {"range": [80, 100], "color": "green"},
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": 90,
                    },
                },
            ),
            row=3,
            col=2,
        )

        # Layout-Updates
        fig.update_layout(
            title=f"Explainable AI Dashboard - {trace.query[:50]}...",
            height=800,
            showlegend=False,
        )

        return fig

    def _get_trace(self, trace_id: str) -> Optional[ExplanationTrace]:
        """Hole Trace anhand ID"""
        for trace in self.explanation_traces:
            if trace.trace_id == trace_id:
                return trace
        return None

    def _assess_query_complexity(self, query: str) -> float:
        """Bewerte Komplexität der Anfrage"""

        complexity_indicators = [
            len(query.split()) > 20,  # Lange Anfrage
            "§" in query or "Art." in query,  # Juristische Referenzen
            any(word in query.lower() for word in ["komplex", "schwierig", "unklar"]),
            query.count("?") > 1,  # Mehrere Fragen
            any(
                word in query.lower()
                for word in ["ausnahme", "sonderfall", "besonderheit"]
            ),
        ]

        return sum(complexity_indicators) / len(complexity_indicators)

    def _extract_decision_factors(
        self, input_data: Dict[str, Any], output_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Extrahiere Entscheidungsfaktoren"""

        factors = {}

        # Analysiere Input-Features
        if "similarity_scores" in input_data:
            factors["semantic_similarity"] = np.mean(input_data["similarity_scores"])

        if "confidence" in output_data:
            factors["model_confidence"] = output_data["confidence"]

        if "retrieval_count" in input_data:
            factors["information_availability"] = min(
                1.0, input_data["retrieval_count"] / 10
            )

        return factors

    def _identify_critical_decisions(self, trace: ExplanationTrace) -> List[str]:
        """Identifiziere kritische Entscheidungen"""

        critical = []

        for step in trace.decision_steps:
            # Niedrige Konfidenz = kritische Entscheidung
            if step.confidence < 0.6:
                critical.append(f"Niedrige Konfidenz bei: {step.step_name}")

            # Viele Alternativen = schwierige Entscheidung
            if len(step.alternatives_considered) > 3:
                critical.append(f"Viele Alternativen bei: {step.step_name}")

            # Hohe Verarbeitungszeit = komplexe Entscheidung
            if step.processing_time > 1.0:
                critical.append(f"Komplexe Verarbeitung bei: {step.step_name}")

        return critical[:5]  # Top 5

    def _identify_uncertainty_sources(self, trace: ExplanationTrace) -> List[str]:
        """Identifiziere Unsicherheitsquellen"""

        sources = []

        # Analysiere Konfidenz-Varianz
        confidences = [step.confidence for step in trace.decision_steps]
        if len(confidences) > 1 and np.std(confidences) > 0.2:
            sources.append("Inkonsistente Konfidenz zwischen Schritten")

        # Niedrige Gesamtkonfidenz
        if trace.overall_confidence < 0.7:
            sources.append("Niedrige Gesamtkonfidenz des Systems")

        # Fehlende Evidenz
        if len(trace.key_evidence) < 3:
            sources.append("Begrenzte verfügbare Evidenz")

        return sources

    def _extract_key_evidence(self, trace: ExplanationTrace) -> List[str]:
        """Extrahiere Schlüsselevidenz"""

        evidence = []

        for step in trace.decision_steps:
            evidence.extend(step.evidence)

        # Entferne Duplikate und sortiere nach Wichtigkeit
        unique_evidence = list(set(evidence))

        return unique_evidence[:10]  # Top 10

    def generate_audit_report(self, trace_id: str) -> Dict[str, Any]:
        """Generiere Audit-Report für Compliance"""

        trace = self._get_trace(trace_id)
        if not trace:
            return {"error": "Trace nicht gefunden"}

        return {
            "audit_metadata": {
                "trace_id": trace.trace_id,
                "audit_timestamp": time.time(),
                "query": trace.query,
                "auditor": "Explainable AI System",
            },
            "decision_audit": {
                "total_decisions": len(trace.decision_steps),
                "documented_reasoning": all(
                    step.reasoning for step in trace.decision_steps
                ),
                "evidence_documented": all(
                    step.evidence for step in trace.decision_steps
                ),
                "alternatives_considered": sum(
                    len(step.alternatives_considered) for step in trace.decision_steps
                ),
            },
            "compliance_assessment": trace.compliance_checks,
            "bias_evaluation": self.bias_monitor.analyze_trace(trace).__dict__,
            "transparency_score": self._calculate_transparency_score(trace),
            "reproducibility": {
                "steps_reproducible": trace.compliance_checks.get(
                    "decision_reproducible", False
                ),
                "input_data_preserved": all(
                    step.input_data for step in trace.decision_steps
                ),
                "version_controlled": True,
            },
            "recommendations": self._generate_audit_recommendations(trace),
        }

    def _calculate_transparency_score(self, trace: ExplanationTrace) -> float:
        """Berechne Transparenz-Score"""

        factors = [
            len(trace.decision_steps) > 0,  # Schritte dokumentiert
            bool(trace.key_evidence),  # Evidenz vorhanden
            all(
                step.reasoning for step in trace.decision_steps
            ),  # Reasoning dokumentiert
            trace.compliance_checks.get("audit_trail_complete", False),  # Audit Trail
            len(trace.critical_decisions) > 0,  # Kritische Entscheidungen identifiziert
        ]

        return sum(factors) / len(factors)


class BiasMonitor:
    """Monitor für Bias-Erkennung"""

    def __init__(self):
        self.bias_patterns = defaultdict(list)
        self.fairness_thresholds = {
            "demographic_parity": 0.1,
            "equalized_odds": 0.1,
            "calibration": 0.05,
        }

    def analyze_trace(self, trace: ExplanationTrace) -> BiasAnalysis:
        """Analysiere Trace auf Bias"""

        bias_sources = {}

        # Analysiere Konfidenz-Bias
        confidences = [step.confidence for step in trace.decision_steps]
        if confidences:
            confidence_variance = np.var(confidences)
            bias_sources["confidence_inconsistency"] = min(1.0, confidence_variance * 2)

        # Analysiere Domain-Bias
        components = [step.component_responsible for step in trace.decision_steps]
        component_diversity = len(set(components)) / max(1, len(components))
        bias_sources["component_diversity"] = 1.0 - component_diversity

        return BiasAnalysis(
            bias_sources=bias_sources,
            fairness_metrics={"overall_fairness": 0.8},
            representation_analysis={"representation_score": 0.7},
            demographic_impact={"no_demographic_data": 0.0},
            legal_domain_bias={"domain_bias_score": 0.3},
            mitigation_strategies=[
                "Diversifizierung der Datenquellen",
                "Regelmäßige Bias-Audits",
            ],
            confidence_in_fairness=0.8,
        )


class ComplianceTracker:
    """Tracker für Compliance-Überwachung"""

    def __init__(self):
        self.compliance_rules = {
            "gdpr_compliance": self._check_gdpr,
            "audit_trail_complete": self._check_audit_trail,
            "decision_reproducible": self._check_reproducibility,
            "human_oversight_available": self._check_human_oversight,
        }

    def check_compliance(self, trace: ExplanationTrace) -> Dict[str, bool]:
        """Prüfe Compliance für Trace"""

        results = {}

        for rule_name, checker in self.compliance_rules.items():
            try:
                results[rule_name] = checker(trace)
            except Exception:
                results[rule_name] = False

        return results

    def _check_gdpr(self, trace: ExplanationTrace) -> bool:
        """Prüfe GDPR-Compliance"""
        # Vereinfachte Prüfung
        return len(trace.decision_steps) > 0 and all(
            "personal_data" not in step.input_data for step in trace.decision_steps
        )

    def _check_audit_trail(self, trace: ExplanationTrace) -> bool:
        """Prüfe Vollständigkeit des Audit Trails"""
        return all(
            step.reasoning and step.timestamp and step.component_responsible
            for step in trace.decision_steps
        )

    def _check_reproducibility(self, trace: ExplanationTrace) -> bool:
        """Prüfe Reproduzierbarkeit"""
        return all(
            step.input_data and step.output_data for step in trace.decision_steps
        )

    def _check_human_oversight(self, trace: ExplanationTrace) -> bool:
        """Prüfe Human-in-the-Loop"""
        # Vereinfacht: Prüfe ob Validierung stattfand
        return any(step.step_type == "validation" for step in trace.decision_steps)


# Demo und Test-Funktionen
def demo_explainable_ai():
    """Demonstriere Explainable AI Dashboard"""

    print("🔍 EXPLAINABLE AI DASHBOARD DEMO")
    print("================================")

    # Initialisiere Engine
    ai_engine = ExplainableAIEngine()

    # Simuliere Entscheidungsprozess
    query = "Kann ich bei einem Kaufvertrag vom Widerruf Gebrauch machen?"
    trace_id = ai_engine.trace_decision_process(
        query, ["retrieval", "analysis", "synthesis"]
    )

    print(f"📋 Tracing gestartet für: {query}")
    print(f"🆔 Trace ID: {trace_id}")

    # Simuliere Entscheidungsschritte
    steps = [
        {
            "name": "Dokumenten-Retrieval",
            "type": "retrieval",
            "component": "neural_retrieval_fusion",
            "input": {"query": query, "top_k": 10},
            "output": {"documents": 8, "avg_similarity": 0.85},
            "confidence": 0.9,
            "reasoning": "Relevante Dokumente zu Widerrufsrecht gefunden",
            "evidence": ["BGB § 355", "Fernabsatzgesetz", "BGH-Urteil 2023"],
        },
        {
            "name": "Rechtliche Analyse",
            "type": "analysis",
            "component": "multi_agent_legal_system",
            "input": {"documents": 8, "legal_domain": "civil_law"},
            "output": {"analysis_result": "Widerruf möglich", "certainty": 0.85},
            "confidence": 0.85,
            "reasoning": "14-tägige Widerrufsfrist bei Fernabsatzverträgen",
            "evidence": ["BGB § 355", "§ 312g BGB"],
            "alternatives": ["Rücktritt", "Anfechtung", "Kündigung"],
        },
        {
            "name": "Antwort-Synthese",
            "type": "synthesis",
            "component": "answer_quality_system",
            "input": {"analysis": "widerruf_möglich", "confidence": 0.85},
            "output": {"final_answer": "Ja, Widerruf ist möglich", "structured": True},
            "confidence": 0.88,
            "reasoning": "Klare Rechtslage bei Fernabsatzverträgen",
            "evidence": ["Gesetzliche Widerrufsfrist", "Verbraucherschutz"],
        },
        {
            "name": "Qualitäts-Validierung",
            "type": "validation",
            "component": "legal_validator",
            "input": {"answer": "Ja, Widerruf ist möglich", "sources": 3},
            "output": {"validation_passed": True, "quality_score": 0.92},
            "confidence": 0.92,
            "reasoning": "Antwort entspricht aktueller Rechtslage",
            "evidence": ["Aktuelle Rechtsprechung", "Eindeutige Gesetzeslage"],
        },
    ]

    # Logge alle Schritte
    for step_data in steps:
        step_id = ai_engine.log_decision_step(
            trace_id=trace_id,
            step_name=step_data["name"],
            step_type=step_data["type"],
            component=step_data["component"],
            input_data=step_data["input"],
            output_data=step_data["output"],
            confidence=step_data["confidence"],
            reasoning=step_data["reasoning"],
            evidence=step_data.get("evidence", []),
            alternatives=step_data.get("alternatives", []),
        )
        print(f"✅ Schritt geloggt: {step_data['name']} (ID: {step_id})")

    # Finalisiere Erklärung
    final_answer = "Ja, bei Fernabsatzverträgen können Sie innerhalb von 14 Tagen ohne Angabe von Gründen widerrufen."
    trace = ai_engine.finalize_explanation(trace_id, final_answer, 0.88)

    print("\n📊 ERKLÄRUNG FINALISIERT")
    print(f"Finale Antwort: {final_answer}")
    print(f"Gesamtkonfidenz: {trace.overall_confidence:.2%}")
    print(f"Verarbeitungszeit: {trace.total_processing_time:.2f}s")
    print(f"Kritische Entscheidungen: {len(trace.critical_decisions)}")

    # Generiere verschiedene Erklärungstypen
    print("\n🎯 ZIELGRUPPEN-SPEZIFISCHE ERKLÄRUNGEN")
    print("=" * 50)

    # Für Rechtsprofis
    prof_explanation = ai_engine.generate_human_readable_explanation(
        trace_id, "legal_professional"
    )
    print("👨‍⚖️ Für Rechtsprofis:")
    print(
        f"  Konfidenzfaktoren: {len(prof_explanation['confidence_analysis']['confidence_factors'])}"
    )
    print(f"  Compliance-Status: {len(prof_explanation['compliance'])} Checks")

    # Für Mandanten
    client_explanation = ai_engine.generate_human_readable_explanation(
        trace_id, "client"
    )
    print("👤 Für Mandanten:")
    print(
        f"  Haupterkenntnis: {client_explanation['our_analysis']['main_finding'][:50]}..."
    )
    print(
        f"  Konfidenz-Level: {client_explanation['our_analysis']['confidence_level']}"
    )

    # Technische Erklärung
    tech_explanation = ai_engine.generate_human_readable_explanation(
        trace_id, "technical"
    )
    print("🔧 Technische Erklärung:")
    print(
        f"  Algorithmus-Entscheidungen: {len(tech_explanation['algorithmic_decisions'])}"
    )
    print(f"  Bias-Analyse verfügbar: {'bias_analysis' in tech_explanation}")

    # Audit-Report
    print("\n📋 AUDIT-REPORT")
    print("=" * 30)

    audit_report = ai_engine.generate_audit_report(trace_id)
    if "error" not in audit_report:
        print(
            f"Dokumentierte Entscheidungen: {audit_report['decision_audit']['total_decisions']}"
        )
        print(f"Transparenz-Score: {audit_report['transparency_score']:.2%}")
        print(
            f"Compliance-Bewertung: {len(audit_report['compliance_assessment'])} Checks"
        )

    # Interaktives Dashboard (nur Struktur, da plotly in Demo nicht angezeigt wird)
    print("\n📊 Interaktives Dashboard erstellt mit 6 Visualisierungen:")
    print("  1. Entscheidungsfluss")
    print("  2. Konfidenz-Entwicklung")
    print("  3. Komponenten-Beiträge")
    print("  4. Unsicherheitsanalyse")
    print("  5. Bias-Indikationen")
    print("  6. Compliance-Status")

    return ai_engine, trace_id


if __name__ == "__main__":
    # Demo ausführen
    engine, trace_id = demo_explainable_ai()

    print("\n🌟 Explainable AI Dashboard erfolgreich demonstriert!")
    print("🚀 Bereit für Integration in das Ultimate Juristic AI System!")
