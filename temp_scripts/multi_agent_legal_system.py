"""
🤖 MULTI-AGENT LEGAL INTELLIGENCE SYSTEM
========================================

Ultra-Advanced Multi-Agent-System mit spezialisierten KI-Agenten für verschiedene Rechtsgebiete,
koordinierte Kollaboration und hierarchische Entscheidungsfindung
"""

import asyncio
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class LegalDomain(Enum):
    """Rechtliche Fachgebiete"""

    CRIMINAL_LAW = "strafrecht"
    CIVIL_LAW = "zivilrecht"
    CONSTITUTIONAL_LAW = "verfassungsrecht"
    ADMINISTRATIVE_LAW = "verwaltungsrecht"
    COMMERCIAL_LAW = "handelsrecht"
    TAX_LAW = "steuerrecht"
    LABOR_LAW = "arbeitsrecht"
    EUROPEAN_LAW = "europarecht"
    PROCEDURAL_LAW = "verfahrensrecht"
    GENERAL_LEGAL = "allgemein"


class AgentRole(Enum):
    """Agent-Rollen im System"""

    SPECIALIST = "specialist"  # Fachgebiet-Spezialist
    COORDINATOR = "coordinator"  # Koordiniert andere Agenten
    SYNTHESIZER = "synthesizer"  # Synthetisiert Ergebnisse
    VALIDATOR = "validator"  # Validiert juristische Korrektheit
    RESEARCHER = "researcher"  # Recherchiert Präzedenzfälle
    CITATION_EXPERT = "citation_expert"  # Zitationsexperte


@dataclass
class AgentCapability:
    """Fähigkeiten eines Agenten"""

    domain_expertise: Dict[LegalDomain, float]  # 0-1 Expertise-Level
    language_skills: List[str]
    specializations: List[str]
    processing_speed: float
    accuracy_rating: float
    collaboration_score: float
    learning_rate: float


@dataclass
class LegalQuery:
    """Juristische Anfrage mit Kontext"""

    query_id: str
    original_question: str
    processed_question: str
    legal_domain: Optional[LegalDomain]
    complexity_score: float
    urgency_level: int  # 1-5
    required_accuracy: float
    context_information: Dict[str, Any]
    previous_queries: List[str] = field(default_factory=list)
    user_preferences: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResponse:
    """Antwort eines einzelnen Agenten"""

    agent_id: str
    response_content: str
    confidence_score: float
    legal_citations: List[str]
    reasoning_steps: List[str]
    domain_relevance: float
    processing_time: float
    certainty_indicators: Dict[str, float]
    follow_up_questions: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class CollaborativeDecision:
    """Kollaborative Entscheidung des Agenten-Teams"""

    final_answer: str
    consensus_score: float
    contributing_agents: List[str]
    confidence_distribution: Dict[str, float]
    dissenting_opinions: List[str]
    synthesis_method: str
    validation_status: str
    citation_completeness: float
    decision_time: float


class LegalAgent:
    """Einzelner spezialisierter Rechts-Agent"""

    def __init__(self, agent_id: str, role: AgentRole, capabilities: AgentCapability):
        self.agent_id = agent_id
        self.role = role
        self.capabilities = capabilities
        self.performance_history = deque(maxlen=1000)
        self.collaboration_history = defaultdict(list)
        self.learning_metrics = {
            "queries_processed": 0,
            "average_accuracy": 0.0,
            "collaboration_success": 0.0,
            "domain_improvements": defaultdict(float),
        }

        # Agent-spezifische Konfiguration
        self.decision_threshold = 0.7
        self.collaboration_eagerness = 0.8
        self.specialization_focus = 0.9

    async def process_query(self, query: LegalQuery) -> AgentResponse:
        """Verarbeite juristische Anfrage"""

        start_time = time.time()

        # Bestimme Domain-Relevanz
        domain_relevance = self._calculate_domain_relevance(query)

        # Entscheide ob Agent zuständig ist
        if domain_relevance < 0.3 and self.role == AgentRole.SPECIALIST:
            return self._create_referral_response(query, domain_relevance, start_time)

        # Simuliere Agent-spezifische Verarbeitung
        response_content = await self._generate_response(query)
        confidence = self._calculate_confidence(query, response_content)
        citations = self._extract_citations(query, response_content)
        reasoning = self._generate_reasoning_steps(query)

        processing_time = time.time() - start_time

        response = AgentResponse(
            agent_id=self.agent_id,
            response_content=response_content,
            confidence_score=confidence,
            legal_citations=citations,
            reasoning_steps=reasoning,
            domain_relevance=domain_relevance,
            processing_time=processing_time,
            certainty_indicators=self._analyze_certainty(response_content),
            follow_up_questions=self._generate_follow_ups(query),
            warnings=self._check_legal_warnings(query, response_content),
        )

        # Update Performance-Metriken
        self._update_performance_metrics(query, response)

        return response

    def _calculate_domain_relevance(self, query: LegalQuery) -> float:
        """Berechne Relevanz der Anfrage für diesen Agenten"""

        if (
            query.legal_domain
            and query.legal_domain in self.capabilities.domain_expertise
        ):
            base_relevance = self.capabilities.domain_expertise[query.legal_domain]
        else:
            # Verwende durchschnittliche Expertise wenn Domain unbekannt
            base_relevance = np.mean(list(self.capabilities.domain_expertise.values()))

        # Justiere basierend auf Agent-Rolle
        if self.role == AgentRole.COORDINATOR:
            base_relevance = max(0.7, base_relevance)  # Koordinator ist immer relevant
        elif self.role == AgentRole.GENERAL_LEGAL:
            base_relevance = max(0.5, base_relevance)

        # Berücksichtige Komplexität
        complexity_factor = 1.0 - (query.complexity_score * 0.3)

        return min(1.0, base_relevance * complexity_factor)

    async def _generate_response(self, query: LegalQuery) -> str:
        """Generiere Agent-spezifische Antwort"""

        # Simuliere asynchrone Verarbeitung
        await asyncio.sleep(0.1 * np.random.rand())

        # Agent-Rolle-spezifische Response-Generierung
        if self.role == AgentRole.SPECIALIST:
            return self._generate_specialist_response(query)
        elif self.role == AgentRole.RESEARCHER:
            return self._generate_research_response(query)
        elif self.role == AgentRole.CITATION_EXPERT:
            return self._generate_citation_response(query)
        elif self.role == AgentRole.VALIDATOR:
            return self._generate_validation_response(query)
        else:
            return self._generate_general_response(query)

    def _generate_specialist_response(self, query: LegalQuery) -> str:
        """Spezialist-spezifische Antwort"""

        domain_name = query.legal_domain.value if query.legal_domain else "allgemein"

        response_templates = {
            LegalDomain.CRIMINAL_LAW: f"Aus strafrechtlicher Sicht zu '{query.processed_question}': [Spezialisierte strafrechtliche Analyse]",
            LegalDomain.CIVIL_LAW: f"Zivilrechtlich betrachtet zu '{query.processed_question}': [Detaillierte zivilrechtliche Bewertung]",
            LegalDomain.CONSTITUTIONAL_LAW: f"Verfassungsrechtliche Einordnung zu '{query.processed_question}': [Verfassungsrechtliche Prüfung]",
            LegalDomain.COMMERCIAL_LAW: f"Handelsrechtliche Analyse zu '{query.processed_question}': [Handelsrechtliche Bewertung]",
        }

        if query.legal_domain in response_templates:
            return response_templates[query.legal_domain]
        else:
            return f"Fachspezifische Analyse zu '{query.processed_question}' im Bereich {domain_name}: [Detaillierte juristische Bewertung]"

    def _generate_research_response(self, query: LegalQuery) -> str:
        """Recherche-spezifische Antwort"""
        return f"Rechtsprechungsrecherche zu '{query.processed_question}': [Präzedenzfälle und aktuelle Rechtsprechung]"

    def _generate_citation_response(self, query: LegalQuery) -> str:
        """Zitations-spezifische Antwort"""
        return f"Zitationsanalyse zu '{query.processed_question}': [Relevante Gesetze, Urteile und Literatur]"

    def _generate_validation_response(self, query: LegalQuery) -> str:
        """Validierungs-spezifische Antwort"""
        return f"Rechtliche Validierung zu '{query.processed_question}': [Korrektheitsprüfung und Qualitätsbewertung]"

    def _generate_general_response(self, query: LegalQuery) -> str:
        """Allgemeine Antwort"""
        return f"Juristische Einschätzung zu '{query.processed_question}': [Umfassende rechtliche Analyse]"

    def _calculate_confidence(self, query: LegalQuery, response: str) -> float:
        """Berechne Konfidenz der Antwort"""

        base_confidence = self.capabilities.accuracy_rating

        # Justiere basierend auf Domain-Expertise
        if query.legal_domain in self.capabilities.domain_expertise:
            domain_factor = self.capabilities.domain_expertise[query.legal_domain]
        else:
            domain_factor = 0.5

        # Berücksichtige Komplexität
        complexity_penalty = query.complexity_score * 0.2

        # Response-Länge als Qualitätsindikator
        length_factor = min(1.0, len(response) / 500)

        confidence = (
            base_confidence * domain_factor * (1 - complexity_penalty) * length_factor
        )

        return max(0.1, min(1.0, confidence))

    def _extract_citations(self, query: LegalQuery, response: str) -> List[str]:
        """Extrahiere rechtliche Zitationen"""

        # Simuliere Zitations-Extraktion basierend auf Agent-Fähigkeiten
        citation_count = int(3 * self.capabilities.accuracy_rating * np.random.rand())

        citations = []
        for i in range(citation_count):
            if query.legal_domain == LegalDomain.CRIMINAL_LAW:
                citations.append(f"StGB § {20 + i}, BGH NJW 2024, {100 + i}")
            elif query.legal_domain == LegalDomain.CIVIL_LAW:
                citations.append(f"BGB § {200 + i}, BGH NZM 2024, {150 + i}")
            else:
                citations.append(f"Gesetz § {50 + i}, Urteil 2024/{100 + i}")

        return citations

    def _generate_reasoning_steps(self, query: LegalQuery) -> List[str]:
        """Generiere Reasoning-Schritte"""

        steps = [
            f"1. Rechtliche Einordnung: {query.legal_domain.value if query.legal_domain else 'Allgemein'}",
            "2. Relevante Normen identifiziert",
            "3. Rechtsprechung analysiert",
            "4. Subsumtion durchgeführt",
            "5. Ergebnis formuliert",
        ]

        return steps

    def _analyze_certainty(self, response: str) -> Dict[str, float]:
        """Analysiere Gewissheits-Indikatoren"""

        return {
            "legal_certainty": 0.8 * np.random.rand(),
            "factual_certainty": 0.9 * np.random.rand(),
            "procedural_certainty": 0.7 * np.random.rand(),
            "precedent_strength": 0.85 * np.random.rand(),
        }

    def _generate_follow_ups(self, query: LegalQuery) -> List[str]:
        """Generiere Follow-up-Fragen"""

        return [
            "Welche spezifischen Umstände liegen vor?",
            "Sind Fristen zu beachten?",
            "Welche Rechtsmittel kommen in Betracht?",
        ]

    def _check_legal_warnings(self, query: LegalQuery, response: str) -> List[str]:
        """Prüfe auf rechtliche Warnhinweise"""

        warnings = []

        if query.complexity_score > 0.8:
            warnings.append(
                "⚠️ Hochkomplexer Rechtsfall - Anwaltliche Beratung empfohlen"
            )

        if query.urgency_level >= 4:
            warnings.append("🚨 Zeitkritisch - Sofortiges Handeln erforderlich")

        return warnings

    def _create_referral_response(
        self, query: LegalQuery, relevance: float, start_time: float
    ) -> AgentResponse:
        """Erstelle Verweis-Antwort für nicht-zuständige Anfragen"""

        processing_time = time.time() - start_time

        return AgentResponse(
            agent_id=self.agent_id,
            response_content=f"Diese Anfrage liegt außerhalb meines Fachgebiets (Relevanz: {relevance:.2f}). Weiterleitung an spezialisierten Agenten empfohlen.",
            confidence_score=0.1,
            legal_citations=[],
            reasoning_steps=["Fachgebiet-Relevanz geprüft", "Weiterleitung empfohlen"],
            domain_relevance=relevance,
            processing_time=processing_time,
            certainty_indicators={"referral_certainty": 0.9},
            follow_up_questions=[
                "An welchen spezialisierten Agenten soll weitergeleitet werden?"
            ],
            warnings=["⚠️ Nicht zuständig für diese Anfrage"],
        )

    def _update_performance_metrics(self, query: LegalQuery, response: AgentResponse):
        """Update Performance-Metriken des Agenten"""

        self.learning_metrics["queries_processed"] += 1

        # Rolling Average für Accuracy
        current_accuracy = self.learning_metrics["average_accuracy"]
        new_accuracy = response.confidence_score

        queries_count = self.learning_metrics["queries_processed"]
        self.learning_metrics["average_accuracy"] = (
            current_accuracy * (queries_count - 1) + new_accuracy
        ) / queries_count

        # Domain-spezifische Verbesserungen
        if query.legal_domain:
            domain_key = query.legal_domain.value
            self.learning_metrics["domain_improvements"][domain_key] += 0.001

    def get_performance_summary(self) -> Dict[str, Any]:
        """Hole Performance-Zusammenfassung des Agenten"""

        return {
            "agent_id": self.agent_id,
            "role": self.role.value,
            "capabilities": {
                "domain_expertise": {
                    d.value: score
                    for d, score in self.capabilities.domain_expertise.items()
                },
                "accuracy_rating": self.capabilities.accuracy_rating,
                "collaboration_score": self.capabilities.collaboration_score,
            },
            "performance_metrics": dict(self.learning_metrics),
            "processing_stats": {
                "queries_processed": self.learning_metrics["queries_processed"],
                "average_accuracy": self.learning_metrics["average_accuracy"],
            },
        }


class MultiAgentCoordinator:
    """Koordinator für Multi-Agent-System"""

    def __init__(self):
        self.agents: Dict[str, LegalAgent] = {}
        self.collaboration_matrix = defaultdict(lambda: defaultdict(float))
        self.query_history = deque(maxlen=1000)
        self.performance_tracker = defaultdict(list)

        # Koordinations-Parameter
        self.min_agents_per_query = 2
        self.max_agents_per_query = 5
        self.consensus_threshold = 0.7
        self.confidence_threshold = 0.6

        # Initialisiere Standard-Agenten
        self._initialize_default_agents()

    def _initialize_default_agents(self):
        """Initialisiere Standard-Agent-Team"""

        # Strafrechts-Spezialist
        criminal_capabilities = AgentCapability(
            domain_expertise={
                LegalDomain.CRIMINAL_LAW: 0.95,
                LegalDomain.PROCEDURAL_LAW: 0.8,
                LegalDomain.CONSTITUTIONAL_LAW: 0.6,
            },
            language_skills=["deutsch", "englisch"],
            specializations=["Wirtschaftsstrafrecht", "Verkehrsstrafrecht"],
            processing_speed=0.8,
            accuracy_rating=0.9,
            collaboration_score=0.85,
            learning_rate=0.1,
        )

        self.add_agent(
            "criminal_specialist", AgentRole.SPECIALIST, criminal_capabilities
        )

        # Zivilrechts-Spezialist
        civil_capabilities = AgentCapability(
            domain_expertise={
                LegalDomain.CIVIL_LAW: 0.95,
                LegalDomain.COMMERCIAL_LAW: 0.8,
                LegalDomain.PROCEDURAL_LAW: 0.7,
            },
            language_skills=["deutsch", "englisch"],
            specializations=["Vertragsrecht", "Schadensersatzrecht"],
            processing_speed=0.9,
            accuracy_rating=0.88,
            collaboration_score=0.9,
            learning_rate=0.08,
        )

        self.add_agent("civil_specialist", AgentRole.SPECIALIST, civil_capabilities)

        # Recherche-Agent
        research_capabilities = AgentCapability(
            domain_expertise={domain: 0.7 for domain in LegalDomain},
            language_skills=["deutsch", "englisch", "französisch"],
            specializations=["Rechtsprechungsrecherche", "Literaturanalyse"],
            processing_speed=0.95,
            accuracy_rating=0.85,
            collaboration_score=0.95,
            learning_rate=0.12,
        )

        self.add_agent("legal_researcher", AgentRole.RESEARCHER, research_capabilities)

        # Zitations-Experte
        citation_capabilities = AgentCapability(
            domain_expertise={domain: 0.6 for domain in LegalDomain},
            language_skills=["deutsch", "englisch"],
            specializations=["Zitationsanalyse", "Quellenvalidierung"],
            processing_speed=0.85,
            accuracy_rating=0.92,
            collaboration_score=0.8,
            learning_rate=0.05,
        )

        self.add_agent(
            "citation_expert", AgentRole.CITATION_EXPERT, citation_capabilities
        )

        # Validator
        validator_capabilities = AgentCapability(
            domain_expertise={domain: 0.8 for domain in LegalDomain},
            language_skills=["deutsch"],
            specializations=["Qualitätssicherung", "Rechtsprüfung"],
            processing_speed=0.7,
            accuracy_rating=0.95,
            collaboration_score=0.7,
            learning_rate=0.03,
        )

        self.add_agent("legal_validator", AgentRole.VALIDATOR, validator_capabilities)

    def add_agent(self, agent_id: str, role: AgentRole, capabilities: AgentCapability):
        """Füge neuen Agenten hinzu"""

        agent = LegalAgent(agent_id, role, capabilities)
        self.agents[agent_id] = agent

        print(f"✅ Agent '{agent_id}' ({role.value}) hinzugefügt")

    async def process_collaborative_query(
        self, query: LegalQuery
    ) -> CollaborativeDecision:
        """Verarbeite Anfrage kollaborativ mit mehreren Agenten"""

        start_time = time.time()

        # Wähle relevante Agenten aus
        selected_agents = self._select_agents_for_query(query)

        print(
            f"🤖 {len(selected_agents)} Agenten ausgewählt für Query: {query.query_id}"
        )

        # Sammle Antworten aller Agenten
        agent_responses = []
        tasks = []

        for agent_id in selected_agents:
            agent = self.agents[agent_id]
            task = agent.process_query(query)
            tasks.append(task)

        # Warte auf alle Antworten
        responses = await asyncio.gather(*tasks)
        agent_responses.extend(responses)

        # Synthetisiere finale Antwort
        collaborative_decision = self._synthesize_responses(query, agent_responses)

        # Update Kollaborations-Metriken
        self._update_collaboration_metrics(selected_agents, collaborative_decision)

        collaborative_decision.decision_time = time.time() - start_time

        self.query_history.append((query, collaborative_decision))

        return collaborative_decision

    def _select_agents_for_query(self, query: LegalQuery) -> List[str]:
        """Wähle optimale Agenten für Anfrage aus"""

        agent_scores = {}

        for agent_id, agent in self.agents.items():
            # Berechne Relevanz-Score
            relevance = agent._calculate_domain_relevance(query)

            # Berücksichtige Agent-Rolle
            role_bonus = 0
            if agent.role == AgentRole.SPECIALIST and query.complexity_score > 0.7:
                role_bonus = 0.2
            elif agent.role == AgentRole.RESEARCHER and query.complexity_score > 0.8:
                role_bonus = 0.3
            elif agent.role == AgentRole.VALIDATOR:
                role_bonus = 0.1  # Validator ist immer hilfreich

            # Berücksichtige Performance-Historie
            performance_bonus = agent.learning_metrics["average_accuracy"] * 0.1

            final_score = relevance + role_bonus + performance_bonus
            agent_scores[agent_id] = final_score

        # Sortiere nach Score und wähle Top-Agenten
        sorted_agents = sorted(agent_scores.items(), key=lambda x: x[1], reverse=True)

        # Wähle Anzahl basierend auf Komplexität
        if query.complexity_score > 0.8:
            num_agents = min(self.max_agents_per_query, len(sorted_agents))
        elif query.complexity_score > 0.5:
            num_agents = min(4, len(sorted_agents))
        else:
            num_agents = min(self.min_agents_per_query, len(sorted_agents))

        selected = [
            agent_id for agent_id, score in sorted_agents[:num_agents] if score > 0.3
        ]

        # Stelle sicher, dass mindestens ein Agent ausgewählt ist
        if not selected and sorted_agents:
            selected = [sorted_agents[0][0]]

        return selected

    def _synthesize_responses(
        self, query: LegalQuery, responses: List[AgentResponse]
    ) -> CollaborativeDecision:
        """Synthetisiere Agent-Antworten zu finaler Entscheidung"""

        if not responses:
            return CollaborativeDecision(
                final_answer="Keine Agenten-Antworten verfügbar",
                consensus_score=0.0,
                contributing_agents=[],
                confidence_distribution={},
                dissenting_opinions=[],
                synthesis_method="none",
                validation_status="failed",
                citation_completeness=0.0,
                decision_time=0.0,
            )

        # Gewichtete Consensus-Bildung
        weighted_responses = self._weight_responses_by_confidence(responses)

        # Synthetisiere finale Antwort
        final_answer = self._create_synthesized_answer(weighted_responses)

        # Berechne Consensus-Score
        consensus_score = self._calculate_consensus_score(responses)

        # Sammle alle Zitationen
        all_citations = []
        for response in responses:
            all_citations.extend(response.legal_citations)

        citation_completeness = min(
            1.0, len(all_citations) / 5
        )  # Erwartet ~5 Zitationen

        # Identifiziere abweichende Meinungen
        dissenting_opinions = self._identify_dissenting_opinions(responses)

        # Validierungsstatus
        validation_status = (
            "validated"
            if any(r.agent_id.endswith("validator") for r in responses)
            else "pending"
        )

        return CollaborativeDecision(
            final_answer=final_answer,
            consensus_score=consensus_score,
            contributing_agents=[r.agent_id for r in responses],
            confidence_distribution={r.agent_id: r.confidence_score for r in responses},
            dissenting_opinions=dissenting_opinions,
            synthesis_method="weighted_consensus",
            validation_status=validation_status,
            citation_completeness=citation_completeness,
            decision_time=0.0,  # Wird später gesetzt
        )

    def _weight_responses_by_confidence(
        self, responses: List[AgentResponse]
    ) -> List[Tuple[AgentResponse, float]]:
        """Gewichte Antworten nach Konfidenz und Agent-Qualität"""

        weighted = []

        for response in responses:
            agent = self.agents[response.agent_id]

            # Basis-Gewicht = Konfidenz * Agent-Accuracy
            base_weight = response.confidence_score * agent.capabilities.accuracy_rating

            # Bonus für hohe Domain-Relevanz
            domain_bonus = response.domain_relevance * 0.5

            # Bonus für Validator-Rolle
            role_bonus = 0.3 if agent.role == AgentRole.VALIDATOR else 0.0

            final_weight = base_weight + domain_bonus + role_bonus
            weighted.append((response, final_weight))

        return weighted

    def _create_synthesized_answer(
        self, weighted_responses: List[Tuple[AgentResponse, float]]
    ) -> str:
        """Erstelle synthetisierte finale Antwort"""

        if not weighted_responses:
            return "Keine verwertbaren Antworten vorhanden."

        # Sortiere nach Gewichtung
        sorted_responses = sorted(weighted_responses, key=lambda x: x[1], reverse=True)

        # Kombiniere Top-Antworten
        synthesis_parts = []

        synthesis_parts.append("📋 KOLLABORATIVE JURISTISCHE ANALYSE")
        synthesis_parts.append("=" * 50)

        for i, (response, weight) in enumerate(sorted_responses[:3]):  # Top 3
            agent = self.agents[response.agent_id]
            role_name = agent.role.value.title()

            synthesis_parts.append(f"\n🔹 {role_name} ({response.agent_id}):")
            synthesis_parts.append(
                f"   Konfidenz: {response.confidence_score:.2f} | Gewichtung: {weight:.2f}"
            )
            synthesis_parts.append(f"   {response.response_content}")

            if response.legal_citations:
                synthesis_parts.append(
                    f"   📚 Zitationen: {', '.join(response.legal_citations)}"
                )

        # Konsens-Statement
        consensus_score = self._calculate_consensus_score(
            [r[0] for r in sorted_responses]
        )
        if consensus_score > 0.8:
            synthesis_parts.append(f"\n✅ HOHER KONSENS (Score: {consensus_score:.2f})")
        elif consensus_score > 0.6:
            synthesis_parts.append(
                f"\n⚠️ MODERATER KONSENS (Score: {consensus_score:.2f})"
            )
        else:
            synthesis_parts.append(
                f"\n❌ NIEDIGER KONSENS (Score: {consensus_score:.2f})"
            )

        return "\n".join(synthesis_parts)

    def _calculate_consensus_score(self, responses: List[AgentResponse]) -> float:
        """Berechne Konsens-Score zwischen Agenten"""

        if len(responses) < 2:
            return 1.0

        # Verwende Konfidenz-Scores als Proxy für Übereinstimmung
        confidences = [r.confidence_score for r in responses]

        # Berechne Standardabweichung (niedrige Abweichung = hoher Konsens)
        std_dev = np.std(confidences)
        max_possible_std = 0.5  # Normalisierungsfaktor

        consensus_score = 1.0 - min(1.0, std_dev / max_possible_std)

        # Bonus für allgemein hohe Konfidenz
        avg_confidence = np.mean(confidences)
        consensus_score = (consensus_score + avg_confidence) / 2

        return consensus_score

    def _identify_dissenting_opinions(
        self, responses: List[AgentResponse]
    ) -> List[str]:
        """Identifiziere abweichende Meinungen"""

        dissenting = []

        if len(responses) < 2:
            return dissenting

        confidences = [r.confidence_score for r in responses]
        avg_confidence = np.mean(confidences)

        for response in responses:
            # Markiere als abweichend wenn deutlich unter Durchschnitt
            if response.confidence_score < avg_confidence * 0.7:
                agent = self.agents[response.agent_id]
                dissenting.append(
                    f"{agent.role.value} ({response.agent_id}): "
                    f"Niedrige Konfidenz ({response.confidence_score:.2f}) - {response.response_content[:100]}..."
                )

        return dissenting

    def _update_collaboration_metrics(
        self, agent_ids: List[str], decision: CollaborativeDecision
    ):
        """Update Kollaborations-Metriken"""

        # Update paarweise Kollaborations-Scores
        for i, agent1 in enumerate(agent_ids):
            for agent2 in agent_ids[i + 1 :]:
                current_score = self.collaboration_matrix[agent1][agent2]
                new_score = decision.consensus_score

                # Rolling Average
                self.collaboration_matrix[agent1][agent2] = (
                    current_score * 0.9 + new_score * 0.1
                )
                self.collaboration_matrix[agent2][agent1] = self.collaboration_matrix[
                    agent1
                ][agent2]

        # Update Performance-Tracker
        for agent_id in agent_ids:
            self.performance_tracker[agent_id].append(
                {
                    "timestamp": time.time(),
                    "consensus_contribution": decision.consensus_score,
                    "query_complexity": 0.5,  # Placeholder
                }
            )

    def get_system_performance_report(self) -> Dict[str, Any]:
        """Generiere System-Performance-Report"""

        # Agent-Performance
        agent_reports = {}
        for agent_id, agent in self.agents.items():
            agent_reports[agent_id] = agent.get_performance_summary()

        # Kollaborations-Analyse
        collaboration_analysis = {}
        for agent1 in self.agents:
            collaboration_analysis[agent1] = {}
            for agent2 in self.agents:
                if agent1 != agent2:
                    score = self.collaboration_matrix[agent1][agent2]
                    collaboration_analysis[agent1][agent2] = round(score, 3)

        # System-Metriken
        total_queries = len(self.query_history)
        if total_queries > 0:
            avg_consensus = np.mean(
                [decision.consensus_score for _, decision in self.query_history]
            )
            avg_decision_time = np.mean(
                [decision.decision_time for _, decision in self.query_history]
            )
        else:
            avg_consensus = 0.0
            avg_decision_time = 0.0

        return {
            "system_overview": {
                "total_agents": len(self.agents),
                "total_queries_processed": total_queries,
                "average_consensus_score": round(avg_consensus, 3),
                "average_decision_time": round(avg_decision_time, 3),
            },
            "agent_performance": agent_reports,
            "collaboration_matrix": collaboration_analysis,
            "recent_performance": {
                "last_10_queries": [
                    {
                        "query_id": query.query_id,
                        "consensus_score": decision.consensus_score,
                        "contributing_agents": len(decision.contributing_agents),
                        "validation_status": decision.validation_status,
                    }
                    for query, decision in list(self.query_history)[-10:]
                ]
            },
        }


# Demo und Test-Funktionen
async def demo_multi_agent_system():
    """Demonstriere Multi-Agent Legal Intelligence System"""

    print("🤖 MULTI-AGENT LEGAL INTELLIGENCE DEMO")
    print("======================================")

    # Initialisiere System
    coordinator = MultiAgentCoordinator()

    print(f"✅ System initialisiert mit {len(coordinator.agents)} Agenten")

    # Teste verschiedene Anfrage-Typen
    test_queries = [
        LegalQuery(
            query_id="Q001",
            original_question="Kann ich bei einem Vertragsbruch Schadensersatz fordern?",
            processed_question="Schadensersatzanspruch bei Vertragsbruch",
            legal_domain=LegalDomain.CIVIL_LAW,
            complexity_score=0.6,
            urgency_level=2,
            required_accuracy=0.8,
            context_information={},
        ),
        LegalQuery(
            query_id="Q002",
            original_question="Welche Strafe droht bei Diebstahl?",
            processed_question="Strafmaß für Diebstahl",
            legal_domain=LegalDomain.CRIMINAL_LAW,
            complexity_score=0.4,
            urgency_level=3,
            required_accuracy=0.9,
            context_information={},
        ),
        LegalQuery(
            query_id="Q003",
            original_question="Ist die neue EU-Verordnung verfassungskonform?",
            processed_question="Verfassungskonformität EU-Verordnung",
            legal_domain=LegalDomain.CONSTITUTIONAL_LAW,
            complexity_score=0.9,
            urgency_level=1,
            required_accuracy=0.95,
            context_information={},
        ),
    ]

    print(f"\n🚀 Verarbeite {len(test_queries)} Test-Anfragen...")

    # Verarbeite Anfragen
    for query in test_queries:
        print(f"\n📋 Verarbeite Query: {query.query_id}")
        print(f"   Frage: {query.original_question}")
        print(f"   Domain: {query.legal_domain.value}")
        print(f"   Komplexität: {query.complexity_score:.2f}")

        decision = await coordinator.process_collaborative_query(query)

        print("✅ Kollaborative Entscheidung getroffen:")
        print(f"   Konsens-Score: {decision.consensus_score:.3f}")
        print(f"   Beteiligte Agenten: {len(decision.contributing_agents)}")
        print(f"   Validierungsstatus: {decision.validation_status}")
        print(f"   Entscheidungszeit: {decision.decision_time:.2f}s")

        if decision.dissenting_opinions:
            print(f"   ⚠️ Abweichende Meinungen: {len(decision.dissenting_opinions)}")

    # Performance-Report
    print("\n📈 MULTI-AGENT PERFORMANCE REPORT")
    print("=" * 50)

    report = coordinator.get_system_performance_report()

    system_overview = report["system_overview"]
    print(f"Gesamte Agenten: {system_overview['total_agents']}")
    print(f"Verarbeitete Anfragen: {system_overview['total_queries_processed']}")
    print(f"Ø Konsens-Score: {system_overview['average_consensus_score']:.3f}")
    print(f"Ø Entscheidungszeit: {system_overview['average_decision_time']:.3f}s")

    print("\n🤖 AGENT-PERFORMANCE:")
    for agent_id, performance in report["agent_performance"].items():
        role = performance["role"]
        accuracy = performance["performance_metrics"]["average_accuracy"]
        queries = performance["performance_metrics"]["queries_processed"]
        print(f"  {agent_id} ({role}): {queries} Anfragen, Ø Accuracy: {accuracy:.3f}")

    return coordinator


if __name__ == "__main__":
    # Demo ausführen
    coordinator = asyncio.run(demo_multi_agent_system())

    print("\n🌟 Multi-Agent Legal Intelligence System erfolgreich demonstriert!")
    print("🚀 Bereit für Integration in das Ultimate Juristic AI System!")
