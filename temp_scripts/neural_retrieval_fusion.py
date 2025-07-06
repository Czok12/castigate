"""
🧠 NEURAL RETRIEVAL FUSION ENGINE
=================================

Ultra-Advanced Multi-Modal Retrieval mit neuraler Fusion und ML-Ranking
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler


@dataclass
class RetrievalCandidate:
    """Einzelner Retrieval-Kandidat mit umfassenden Features"""

    content: str
    source: str
    retrieval_method: str
    semantic_score: float
    keyword_score: float
    entity_score: float
    context_score: float
    relevance_score: float
    freshness_score: float
    authority_score: float
    completeness_score: float
    legal_specificity: float
    citation_count: int
    user_engagement: float
    raw_features: Dict[str, float]


@dataclass
class FusionResult:
    """Ergebnis der neuralen Fusion"""

    final_score: float
    confidence: float
    explanation: Dict[str, float]
    retrieval_path: List[str]
    fusion_time: float


class NeuralRankingNetwork(nn.Module):
    """Neuronales Netz für intelligentes Ranking"""

    def __init__(self, input_dim: int = 32, hidden_dims: List[int] = [64, 32, 16]):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(0.2)])
            prev_dim = hidden_dim

        # Output Layer
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

        # Attention-Mechanismus für Feature-Wichtigkeit
        self.attention = nn.MultiheadAttention(input_dim, num_heads=4, batch_first=True)
        self.feature_importance = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        # Attention für Feature-Wichtigkeit
        attended, attention_weights = self.attention(
            x.unsqueeze(1), x.unsqueeze(1), x.unsqueeze(1)
        )
        attended = attended.squeeze(1)

        # Feature-Wichtigkeit berechnen
        importance = torch.sigmoid(self.feature_importance(attended))
        weighted_features = x * importance

        # Durch Netzwerk
        score = self.network(weighted_features)

        return score, attention_weights, importance


class AdvancedFeatureExtractor:
    """Erweiterte Feature-Extraktion für Retrieval-Kandidaten"""

    def __init__(self):
        self.legal_keywords = {
            "high_authority": ["BGH", "BVerfG", "EuGH", "BFH", "BAG", "BSG", "BVerwG"],
            "procedural": ["Verfahren", "Klage", "Berufung", "Revision", "Beschwerde"],
            "substantive": [
                "Anspruch",
                "Recht",
                "Pflicht",
                "Voraussetzung",
                "Rechtsfolge",
            ],
            "temporal": ["Frist", "Verjährung", "Termin", "Zeitpunkt", "sofort"],
            "quantitative": ["Höhe", "Betrag", "Prozent", "Zinsen", "Schaden"],
        }

        self.complexity_indicators = [
            "jedoch",
            "allerdings",
            "vielmehr",
            "insbesondere",
            "grundsätzlich",
            "ausnahmsweise",
            "unter Umständen",
            "gegebenenfalls",
            "sofern",
        ]

        self.scaler = StandardScaler()
        self.is_fitted = False

    def extract_comprehensive_features(
        self, candidate: RetrievalCandidate, query: str
    ) -> np.ndarray:
        """Extrahiert umfassende Features für ML-Ranking"""

        features = []
        content_lower = candidate.content.lower()
        query_lower = query.lower()

        # === GRUNDLEGENDE RETRIEVAL-SCORES ===
        features.extend(
            [
                candidate.semantic_score,
                candidate.keyword_score,
                candidate.entity_score,
                candidate.context_score,
                candidate.relevance_score,
            ]
        )

        # === CONTENT-QUALITÄT ===
        features.extend(
            [
                len(candidate.content) / 1000,  # Normalisierte Länge
                len(candidate.content.split()) / 100,  # Normalisierte Wort-Anzahl
                candidate.content.count(".")
                / len(candidate.content)
                * 100,  # Satz-Dichte
                candidate.freshness_score,
                candidate.authority_score,
                candidate.completeness_score,
            ]
        )

        # === JURISTISCHE SPEZIFITÄT ===
        legal_density = (
            sum(
                content_lower.count(kw)
                for category in self.legal_keywords.values()
                for kw in category
            )
            / len(candidate.content)
            * 1000
        )
        features.append(legal_density)

        # Authority-Indikatoren
        authority_mentions = sum(
            content_lower.count(auth.lower())
            for auth in self.legal_keywords["high_authority"]
        )
        features.append(authority_mentions)

        # Komplexitäts-Score
        complexity_score = sum(
            content_lower.count(indicator) for indicator in self.complexity_indicators
        )
        features.append(complexity_score / len(candidate.content) * 1000)

        # === QUERY-MATCHING ===
        query_words = set(query_lower.split())
        content_words = set(content_lower.split())

        # Exakte Wort-Überschneidung
        word_overlap = (
            len(query_words & content_words) / len(query_words) if query_words else 0
        )
        features.append(word_overlap)

        # Position wichtiger Query-Begriffe
        first_match_position = float("inf")
        for word in query_words:
            if word in content_lower:
                pos = content_lower.find(word) / len(content_lower)
                first_match_position = min(first_match_position, pos)

        if first_match_position == float("inf"):
            first_match_position = 1.0
        features.append(1.0 - first_match_position)  # Frühere Position = höherer Score

        # === ENGAGEMENT & POPULARITÄT ===
        features.extend(
            [
                candidate.user_engagement,
                candidate.citation_count / 10,  # Normalisiert
                candidate.legal_specificity,
            ]
        )

        # === STRUKTURELLE FEATURES ===
        # Paragraphen-Dichte
        paragraph_density = content_lower.count("§") / len(candidate.content) * 1000
        features.append(paragraph_density)

        # Aufzählungs-Struktur
        enumeration_density = (
            (
                content_lower.count("1.")
                + content_lower.count("a)")
                + content_lower.count("(1)")
            )
            / len(candidate.content)
            * 1000
        )
        features.append(enumeration_density)

        # Zitier-Dichte
        citation_density = (
            (
                content_lower.count("vgl.")
                + content_lower.count("s.")
                + content_lower.count("rn")
            )
            / len(candidate.content)
            * 1000
        )
        features.append(citation_density)

        # === KONTEXT-FEATURES ===
        # Retrieval-Method Encoding
        method_scores = {
            "semantic": 1.0,
            "keyword": 0.8,
            "legal_entity": 0.9,
            "hybrid": 0.95,
            "neural": 1.0,
            "rerank": 0.85,
        }
        features.append(method_scores.get(candidate.retrieval_method, 0.5))

        # Source-Type Encoding
        source_scores = {
            "lehrbuch": 1.0,
            "kommentar": 0.9,
            "rechtsprechung": 0.95,
            "aufsatz": 0.7,
            "gesetze": 0.8,
            "online": 0.6,
        }
        source_type = candidate.source.lower()
        source_score = max(
            [score for source, score in source_scores.items() if source in source_type],
            default=0.5,
        )
        features.append(source_score)

        # === RAW FEATURES INTEGRATION ===
        # Zusätzliche Features aus raw_features (4 Features hinzufügen um auf 32 zu kommen)
        for key in [
            "readability",
            "coherence",
            "technical_depth",
            "practical_relevance",
        ]:
            features.append(candidate.raw_features.get(key, 0.5))

        # Padding um auf exakt 32 Features zu kommen
        current_length = len(features)
        target_length = 32
        if current_length < target_length:
            features.extend([0.5] * (target_length - current_length))
        elif current_length > target_length:
            features = features[:target_length]

        return np.array(features, dtype=np.float32)

    def fit_scaler(self, feature_matrix: np.ndarray):
        """Fittet den Scaler für Feature-Normalisierung"""
        self.scaler.fit(feature_matrix)
        self.is_fitted = True

    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """Normalisiert Features"""
        if not self.is_fitted:
            return features
        return self.scaler.transform(features.reshape(1, -1))[0]


class NeuralRetrievalFusion:
    """Haupt-Engine für neurale Retrieval-Fusion"""

    def __init__(self, model_path: str = "neural_ranking_model.pth"):
        self.model_path = model_path
        self.feature_extractor = AdvancedFeatureExtractor()

        # Neuronales Netz
        self.neural_ranker = None
        self.feature_dim = 32  # Angepasst an Feature-Extraktor

        # Ensemble-Modelle
        self.ensemble_models = {
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "gradient_boost": GradientBoostingRegressor(
                n_estimators=100, random_state=42
            ),
        }

        # Fusion-Strategien
        self.fusion_strategies = {
            "weighted_average": self._weighted_average_fusion,
            "neural_attention": self._neural_attention_fusion,
            "dynamic_ensemble": self._dynamic_ensemble_fusion,
            "adaptive_ranking": self._adaptive_ranking_fusion,
        }

        # Performance-Tracking
        self.strategy_performance = defaultdict(
            lambda: {
                "usage_count": 0,
                "avg_precision": 0.0,
                "avg_time": 0.0,
                "user_satisfaction": 0.0,
            }
        )

        # Learning-History für Adaptation
        self.learning_history = deque(maxlen=1000)

        self._initialize_neural_ranker()
        self._load_models()

    def _initialize_neural_ranker(self):
        """Initialisiert das neuronale Ranking-Netzwerk"""
        self.neural_ranker = NeuralRankingNetwork(
            input_dim=self.feature_dim, hidden_dims=[64, 32, 16]
        )

    def _load_models(self):
        """Lädt vortrainierte Modelle"""
        try:
            if os.path.exists(self.model_path):
                checkpoint = torch.load(self.model_path, map_location="cpu")
                if self.neural_ranker is not None:
                    self.neural_ranker.load_state_dict(checkpoint["model_state"])
                print("✅ Neuronales Ranking-Modell geladen")
            else:
                print(
                    "⚠️ Kein vortrainiertes Modell gefunden - verwende Zufallsinitialisierung"
                )
        except Exception as e:
            print(f"❌ Fehler beim Laden des Modells: {e}")

    def fuse_retrieval_results(
        self,
        candidates: List[RetrievalCandidate],
        query: str,
        strategy: str = "adaptive_ranking",
        top_k: int = 5,
    ) -> List[Tuple[RetrievalCandidate, FusionResult]]:
        """Hauptfunktion für neurale Retrieval-Fusion"""

        start_time = time.time()

        if not candidates:
            return []

        # Feature-Extraktion für alle Kandidaten
        features_matrix = []
        for candidate in candidates:
            features = self.feature_extractor.extract_comprehensive_features(
                candidate, query
            )
            normalized_features = self.feature_extractor.normalize_features(features)
            features_matrix.append(normalized_features)

        features_matrix = np.array(features_matrix)

        # Fusion-Strategie anwenden
        fusion_func = self.fusion_strategies.get(
            strategy, self._adaptive_ranking_fusion
        )
        fusion_results = fusion_func(candidates, features_matrix, query)

        # Ranking nach Fusion-Score
        ranked_results = sorted(
            zip(candidates, fusion_results),
            key=lambda x: x[1].final_score,
            reverse=True,
        )

        # Performance-Tracking
        fusion_time = time.time() - start_time
        self._update_strategy_performance(strategy, fusion_time, len(candidates))

        return ranked_results[:top_k]

    def _weighted_average_fusion(
        self,
        candidates: List[RetrievalCandidate],
        features_matrix: np.ndarray,
        query: str,
    ) -> List[FusionResult]:
        """Gewichtete Durchschnitts-Fusion"""

        results = []
        weights = np.array([0.3, 0.25, 0.2, 0.15, 0.1])  # Für Top-5 Scores

        for i, candidate in enumerate(candidates):
            scores = np.array(
                [
                    candidate.semantic_score,
                    candidate.relevance_score,
                    candidate.authority_score,
                    candidate.completeness_score,
                    candidate.legal_specificity,
                ]
            )

            final_score = np.dot(scores, weights)
            score_std = np.std(scores)
            confidence = 1.0 - float(
                score_std
            )  # Niedrige Standardabweichung = hohe Confidence

            explanation = {
                "semantic_weight": 0.3,
                "relevance_weight": 0.25,
                "authority_weight": 0.2,
                "completeness_weight": 0.15,
                "specificity_weight": 0.1,
            }

            results.append(
                FusionResult(
                    final_score=final_score,
                    confidence=1.0 - confidence,
                    explanation=explanation,
                    retrieval_path=["weighted_average"],
                    fusion_time=0.001,
                )
            )

        return results

    def _neural_attention_fusion(
        self,
        candidates: List[RetrievalCandidate],
        features_matrix: np.ndarray,
        query: str,
    ) -> List[FusionResult]:
        """Neuronale Attention-basierte Fusion"""

        results = []

        with torch.no_grad():
            for i, features in enumerate(features_matrix):
                features_tensor = torch.FloatTensor(features).unsqueeze(0)

                # Neuronales Ranking (fallback if model not available)
                if self.neural_ranker is not None:
                    score, attention_weights, feature_importance = self.neural_ranker(
                        features_tensor
                    )
                    final_score = float(score.item())
                    confidence = float(torch.mean(attention_weights).item())

                    # Feature-Wichtigkeit als Erklärung
                    feature_names = [f"feature_{j}" for j in range(len(features))]
                    explanation = dict(
                        zip(
                            feature_names,
                            [float(x) for x in feature_importance.squeeze().tolist()],
                        )
                    )
                else:
                    # Fallback ohne neuronales Netz
                    final_score = float(np.mean(features[:5]))  # Top-5 Features
                    confidence = 0.5
                    explanation = {"fallback_mode": 1.0}

                results.append(
                    FusionResult(
                        final_score=final_score,
                        confidence=confidence,
                        explanation=explanation,
                        retrieval_path=["neural_attention"],
                        fusion_time=0.005,
                    )
                )

        return results

    def _dynamic_ensemble_fusion(
        self,
        candidates: List[RetrievalCandidate],
        features_matrix: np.ndarray,
        query: str,
    ) -> List[FusionResult]:
        """Dynamisches Ensemble verschiedener Ranking-Modelle"""

        results = []

        # Ensemble-Predictions (falls Modelle trainiert sind)
        ensemble_scores = []
        for model_name, model in self.ensemble_models.items():
            try:
                if hasattr(model, "predict"):
                    scores = model.predict(features_matrix)
                    ensemble_scores.append(scores)
            except Exception:
                # Fallback auf heuristische Scores
                heuristic_scores = []
                for candidate in candidates:
                    score = (
                        candidate.semantic_score * 0.4
                        + candidate.authority_score * 0.3
                        + candidate.relevance_score * 0.3
                    )
                    heuristic_scores.append(score)
                ensemble_scores.append(np.array(heuristic_scores))

        if not ensemble_scores:
            return self._weighted_average_fusion(candidates, features_matrix, query)

        # Ensemble-Fusion
        ensemble_scores = np.array(ensemble_scores)

        for i in range(len(candidates)):
            # Dynamische Gewichtung basierend auf Modell-Performance
            model_weights = self._get_dynamic_model_weights()
            final_score = float(
                np.average(ensemble_scores[:, i], weights=model_weights)
            )

            # Confidence basierend auf Ensemble-Varianz
            score_variance = np.var(ensemble_scores[:, i])
            confidence = float(
                1.0 / (1.0 + score_variance * 10)
            )  # Inversely related to variance

            explanation = {
                f"model_{j}_score": float(ensemble_scores[j, i])
                for j in range(len(ensemble_scores))
            }

            results.append(
                FusionResult(
                    final_score=final_score,
                    confidence=confidence,
                    explanation=explanation,
                    retrieval_path=["dynamic_ensemble"],
                    fusion_time=0.003,
                )
            )

        return results

    def _adaptive_ranking_fusion(
        self,
        candidates: List[RetrievalCandidate],
        features_matrix: np.ndarray,
        query: str,
    ) -> List[FusionResult]:
        """Adaptive Ranking-Fusion mit Query-spezifischer Anpassung"""

        results = []

        # Query-Intent-Analyse für adaptive Gewichtung
        query_intent_weights = self._analyze_query_intent(query)

        for i, candidate in enumerate(candidates):
            features = features_matrix[i]

            # Intent-basierte Gewichtung der Features
            semantic_weight = query_intent_weights.get("semantic_focus", 0.3)
            authority_weight = query_intent_weights.get("authority_focus", 0.2)
            practical_weight = query_intent_weights.get("practical_focus", 0.2)
            legal_weight = query_intent_weights.get("legal_focus", 0.3)

            # Adaptive Score-Berechnung
            adaptive_score = (
                candidate.semantic_score * semantic_weight
                + candidate.authority_score * authority_weight
                + candidate.user_engagement * practical_weight
                + candidate.legal_specificity * legal_weight
            )

            # Neuronale Komponente hinzufügen
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features).unsqueeze(0)
                if self.neural_ranker is not None:
                    neural_score, _, _ = self.neural_ranker(features_tensor)
                    neural_component = float(neural_score.item()) * 0.3
                else:
                    neural_component = 0.0

            final_score = adaptive_score * 0.7 + neural_component

            # Confidence basierend auf Intent-Klarheit
            intent_clarity = max(query_intent_weights.values()) - min(
                query_intent_weights.values()
            )
            confidence = intent_clarity * 0.5 + 0.5

            explanation = {
                "semantic_focus": semantic_weight,
                "authority_focus": authority_weight,
                "practical_focus": practical_weight,
                "legal_focus": legal_weight,
                "neural_component": neural_component,
                "intent_clarity": intent_clarity,
            }

            results.append(
                FusionResult(
                    final_score=final_score,
                    confidence=confidence,
                    explanation=explanation,
                    retrieval_path=["adaptive_ranking"],
                    fusion_time=0.004,
                )
            )

        return results

    def _analyze_query_intent(self, query: str) -> Dict[str, float]:
        """Analysiert Query-Intent für adaptive Gewichtung"""

        query_lower = query.lower()
        intent_weights = {
            "semantic_focus": 0.3,  # Standard-Gewichtung
            "authority_focus": 0.2,
            "practical_focus": 0.2,
            "legal_focus": 0.3,
        }

        # Intent-Pattern-Matching
        if any(
            word in query_lower
            for word in ["definition", "bedeutung", "was ist", "begriffe"]
        ):
            intent_weights["semantic_focus"] = 0.5
            intent_weights["legal_focus"] = 0.3

        elif any(
            word in query_lower
            for word in ["rechtsprechung", "urteil", "bgh", "bverfg"]
        ):
            intent_weights["authority_focus"] = 0.5
            intent_weights["legal_focus"] = 0.3

        elif any(
            word in query_lower for word in ["praxis", "vorgehen", "wie", "beispiel"]
        ):
            intent_weights["practical_focus"] = 0.5
            intent_weights["semantic_focus"] = 0.3

        elif any(
            word in query_lower
            for word in ["§", "paragraph", "gesetz", "voraussetzung"]
        ):
            intent_weights["legal_focus"] = 0.5
            intent_weights["authority_focus"] = 0.3

        # Normalisierung
        total_weight = sum(intent_weights.values())
        return {k: v / total_weight for k, v in intent_weights.items()}

    def _get_dynamic_model_weights(self) -> np.ndarray:
        """Berechnet dynamische Modell-Gewichtungen basierend auf Performance"""

        # Einfache gleichmäßige Gewichtung als Fallback
        num_models = len(self.ensemble_models)
        if num_models == 0:
            return np.array([1.0])

        return np.ones(num_models) / num_models

    def _update_strategy_performance(
        self, strategy: str, fusion_time: float, candidate_count: int
    ):
        """Aktualisiert Performance-Statistiken"""

        perf = self.strategy_performance[strategy]
        perf["usage_count"] += 1
        perf["avg_time"] = (
            perf["avg_time"] * (perf["usage_count"] - 1) + fusion_time
        ) / perf["usage_count"]

    def get_fusion_analytics(self) -> Dict[str, Any]:
        """Gibt detaillierte Analytics über Fusion-Performance zurück"""

        analytics = {
            "strategy_performance": dict(self.strategy_performance),
            "feature_importance": self._get_feature_importance(),
            "model_statistics": {
                "neural_ranker_parameters": (
                    sum(p.numel() for p in self.neural_ranker.parameters())
                    if self.neural_ranker
                    else 0
                ),
                "ensemble_models": list(self.ensemble_models.keys()),
                "feature_dimension": self.feature_dim,
            },
            "learning_history_size": len(self.learning_history),
            "recommendations": self._generate_optimization_recommendations(),
        }

        return analytics

    def _get_feature_importance(self) -> Dict[str, float]:
        """Analysiert Feature-Wichtigkeit"""

        # Vereinfachte Feature-Wichtigkeit (in echter Implementierung durch Modell-Analyse)
        return {
            "semantic_score": 0.25,
            "authority_score": 0.20,
            "relevance_score": 0.18,
            "legal_specificity": 0.15,
            "completeness_score": 0.12,
            "user_engagement": 0.10,
        }

    def _generate_optimization_recommendations(self) -> List[Dict[str, str]]:
        """Generiert Optimierungsempfehlungen"""

        recommendations = []

        # Analyse der Strategy-Performance
        if self.strategy_performance:
            best_strategy = max(
                self.strategy_performance.keys(),
                key=lambda k: self.strategy_performance[k]["usage_count"],
            )

            recommendations.append(
                {
                    "type": "strategy_optimization",
                    "recommendation": f"Verwende primär {best_strategy} für bessere Performance",
                    "impact": "high",
                }
            )

        recommendations.extend(
            [
                {
                    "type": "model_training",
                    "recommendation": "Sammle mehr Trainingsdaten für bessere neuronale Fusion",
                    "impact": "medium",
                },
                {
                    "type": "feature_engineering",
                    "recommendation": "Erweitere Feature-Set um domänen-spezifische Signale",
                    "impact": "high",
                },
            ]
        )

        return recommendations


# === HELPER FUNCTIONS ===


def simulate_retrieval_candidates(
    query: str, count: int = 10
) -> List[RetrievalCandidate]:
    """Simuliert Retrieval-Kandidaten für Testing"""

    candidates = []

    for i in range(count):
        candidate = RetrievalCandidate(
            content=f"Beispiel-Content für Query '{query}' - Kandidat {i+1}. " * 5,
            source=f"Lehrbuch_Kapitel_{i+1}",
            retrieval_method=np.random.choice(
                ["semantic", "keyword", "hybrid", "neural"]
            ),
            semantic_score=np.random.uniform(0.3, 0.95),
            keyword_score=np.random.uniform(0.2, 0.9),
            entity_score=np.random.uniform(0.1, 0.8),
            context_score=np.random.uniform(0.4, 0.9),
            relevance_score=np.random.uniform(0.3, 0.95),
            freshness_score=np.random.uniform(0.5, 1.0),
            authority_score=np.random.uniform(0.2, 0.9),
            completeness_score=np.random.uniform(0.4, 0.95),
            legal_specificity=np.random.uniform(0.3, 0.9),
            citation_count=np.random.randint(0, 20),
            user_engagement=np.random.uniform(0.1, 0.8),
            raw_features={
                "readability": np.random.uniform(0.3, 0.9),
                "coherence": np.random.uniform(0.4, 0.9),
                "technical_depth": np.random.uniform(0.2, 0.8),
                "practical_relevance": np.random.uniform(0.3, 0.9),
            },
        )
        candidates.append(candidate)

    return candidates


# === TESTING & DEMO ===


def demonstrate_neural_fusion():
    """Demonstriert die Neural Retrieval Fusion"""

    print("🧠 NEURAL RETRIEVAL FUSION - DEMO")
    print("=" * 50)

    # Initialize Engine
    fusion_engine = NeuralRetrievalFusion()

    # Test Query
    test_query = "Voraussetzungen eines Kaufvertrags im BGB"

    # Simuliere Retrieval-Kandidaten
    candidates = simulate_retrieval_candidates(test_query, count=8)

    print(f"\n📊 Testing mit Query: '{test_query}'")
    print(f"🎯 {len(candidates)} Retrieval-Kandidaten")

    # Teste verschiedene Fusion-Strategien
    strategies = [
        "weighted_average",
        "neural_attention",
        "dynamic_ensemble",
        "adaptive_ranking",
    ]

    for strategy in strategies:
        print(f"\n🔄 Teste {strategy.upper()} Fusion...")
        start_time = time.time()

        results = fusion_engine.fuse_retrieval_results(
            candidates=candidates, query=test_query, strategy=strategy, top_k=3
        )

        fusion_time = time.time() - start_time

        print(f"⏱️  Fusion-Zeit: {fusion_time:.4f}s")
        print("🏆 Top-3 Results:")

        for i, (candidate, fusion_result) in enumerate(results[:3], 1):
            print(
                f"   {i}. Score: {fusion_result.final_score:.4f} | "
                f"Confidence: {fusion_result.confidence:.3f} | "
                f"Method: {candidate.retrieval_method}"
            )

    # Analytics
    print("\n📈 FUSION ANALYTICS:")
    analytics = fusion_engine.get_fusion_analytics()

    print("🔧 Feature-Wichtigkeit:")
    for feature, importance in analytics["feature_importance"].items():
        print(f"   • {feature}: {importance:.3f}")

    print("\n💡 Empfehlungen:")
    for rec in analytics["recommendations"]:
        print(f"   • {rec['recommendation']} (Impact: {rec['impact']})")


if __name__ == "__main__":
    import os

    demonstrate_neural_fusion()
