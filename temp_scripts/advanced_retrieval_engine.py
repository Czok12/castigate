"""
🎯 ERWEITERTE RETRIEVAL-ENGINE
============================

Multi-Stage-Retrieval, Query-Expansion und intelligente Ranking-Algorithmen
"""

import re
import time
from typing import Dict, List, Optional

import numpy as np

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class QueryProcessor:
    """Erweiterte Query-Verarbeitung und -Expansion"""

    def __init__(self):
        self.legal_synonyms = {
            "vertrag": ["vereinbarung", "kontrakt", "abkommen"],
            "schadenersatz": ["schadensersatz", "entschädigung", "kompensation"],
            "eigentum": ["besitz", "eigentumsrecht", "vermögen"],
            "haftung": ["verantwortlichkeit", "schuld", "pflicht"],
            "recht": ["berechtigung", "anspruch", "befugnis"],
            "pflicht": ["verpflichtung", "obliegenheit", "schuld"],
        }

        self.legal_abbreviations = {
            "bgb": "bürgerliches gesetzbuch",
            "stgb": "strafgesetzbuch",
            "gg": "grundgesetz",
            "zpo": "zivilprozessordnung",
            "stpo": "strafprozessordnung",
            "hgb": "handelsgesetzbuch",
            "agg": "allgemeines gleichbehandlungsgesetz",
        }

        # Juristische Gewichtungswörter
        self.high_weight_terms = [
            "voraussetzung",
            "tatbestand",
            "rechtsfolge",
            "anspruch",
            "berechtigung",
            "verpflichtung",
            "schutzbereich",
            "eingriff",
            "rechtfertigung",
        ]

    def expand_query(self, query: str) -> Dict[str, List[str]]:
        """Erweitert Query um Synonyme und juristische Begriffe"""

        original_terms = query.lower().split()
        expanded_terms = set(original_terms)
        high_weight_found = []

        # Abkürzungen expandieren
        for term in original_terms:
            if term in self.legal_abbreviations:
                expanded_terms.add(self.legal_abbreviations[term])

        # Synonyme hinzufügen
        for term in original_terms:
            if term in self.legal_synonyms:
                expanded_terms.update(self.legal_synonyms[term])

        # Hohe Gewichtung für wichtige juristische Begriffe
        for term in original_terms:
            if term in self.high_weight_terms:
                high_weight_found.append(term)

        return {
            "original": original_terms,
            "expanded": list(expanded_terms),
            "high_weight": high_weight_found,
        }

    def extract_legal_entities(self, query: str) -> Dict[str, List[str]]:
        """Extrahiert juristische Entitäten aus der Query"""

        patterns = {
            "paragraphs": r"§\s*(\d+[a-z]?)",
            "articles": r"Art\.\s*(\d+[a-z]?)",
            "courts": r"(BGH|BVerfG|BFH|BAG|BSG|BVerwG|OLG|LG|AG|VG|OVG)",
            "laws": r"(BGB|StGB|GG|ZPO|StPO|HGB|AO|VAG|InsO|AktG|GmbHG)",
            "decisions": r"(\d{1,2}\.\d{1,2}\.\d{4})",
        }

        entities = {}
        for entity_type, pattern in patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            entities[entity_type] = matches

        return entities


class MultiStageRetriever:
    """Multi-Stage-Retrieval für höchste Präzision"""

    def __init__(self, vectorstore, embedding_model):
        self.vectorstore = vectorstore
        self.embedding_model = embedding_model
        self.query_processor = QueryProcessor()

        # TF-IDF für Keyword-Matching (nur wenn sklearn verfügbar)
        if SKLEARN_AVAILABLE:
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                stop_words=None,  # Keine Stop-Words für juristische Texte
            )
        else:
            self.tfidf_vectorizer = None

        self.is_tfidf_fitted = False

    def fit_tfidf(self, documents: List[str]):
        """Trainiert TF-IDF auf Dokumenten-Corpus"""
        if not self.is_tfidf_fitted and self.tfidf_vectorizer is not None:
            self.tfidf_vectorizer.fit(documents)
            self.is_tfidf_fitted = True

    def multi_stage_retrieve(
        self,
        query: str,
        k: int = 4,
        stages: List[str] = ["semantic", "keyword", "legal_entity", "rerank"],
    ) -> List[Dict]:
        """Mehrstufiges Retrieval mit verschiedenen Strategien"""

        start_time = time.time()

        # Query-Verarbeitung
        query_info = self.query_processor.expand_query(query)
        legal_entities = self.query_processor.extract_legal_entities(query)

        stage_results = {}

        # Stage 1: Semantic Retrieval
        if "semantic" in stages:
            semantic_docs = self._semantic_retrieval(query, k * 2)
            stage_results["semantic"] = semantic_docs

        # Stage 2: Keyword-based Retrieval
        if "keyword" in stages and self.is_tfidf_fitted and SKLEARN_AVAILABLE:
            keyword_docs = self._keyword_retrieval(query_info["expanded"], k)
            stage_results["keyword"] = keyword_docs

        # Stage 3: Legal Entity Retrieval
        if "legal_entity" in stages and any(legal_entities.values()):
            entity_docs = self._legal_entity_retrieval(legal_entities, k)
            stage_results["legal_entity"] = entity_docs

        # Stage 4: Reranking & Fusion
        if "rerank" in stages:
            final_docs = self._rerank_and_fuse(
                stage_results, query, query_info, legal_entities, k
            )
        else:
            # Einfache Fusion ohne Reranking
            final_docs = self._simple_fusion(stage_results, k)

        retrieval_time = time.time() - start_time

        # Finale Bewertung hinzufügen
        for doc in final_docs:
            doc["retrieval_time"] = retrieval_time
            doc["stages_used"] = list(stage_results.keys())

        return final_docs

    def _semantic_retrieval(self, query: str, k: int) -> List[Dict]:
        """Standard Semantic Search"""

        docs = self.vectorstore.similarity_search_with_score(query, k=k)

        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "similarity_score": float(score),
                "retrieval_method": "semantic",
            }
            for doc, score in docs
        ]

    def _keyword_retrieval(self, expanded_terms: List[str], k: int) -> List[Dict]:
        """TF-IDF basierte Keyword-Suche"""

        if not SKLEARN_AVAILABLE or self.tfidf_vectorizer is None:
            return []

        # Alle Dokumente für TF-IDF
        all_docs = self.vectorstore.similarity_search(
            "", k=1000
        )  # Holt alle verfügbaren
        doc_texts = [doc.page_content for doc in all_docs]

        if not doc_texts:
            return []

        # TF-IDF Query-Vektor
        query_text = " ".join(expanded_terms)
        doc_tfidf = self.tfidf_vectorizer.transform(doc_texts)
        query_tfidf = self.tfidf_vectorizer.transform([query_text])

        # Similarity berechnen
        similarities = cosine_similarity(query_tfidf, doc_tfidf).flatten()

        # Top-k auswählen
        top_indices = np.argsort(similarities)[::-1][:k]

        return [
            {
                "content": all_docs[i].page_content,
                "metadata": all_docs[i].metadata,
                "tfidf_score": float(similarities[i]),
                "retrieval_method": "keyword",
            }
            for i in top_indices
            if similarities[i] > 0.01  # Mindest-Relevanz
        ]

    def _legal_entity_retrieval(
        self, entities: Dict[str, List[str]], k: int
    ) -> List[Dict]:
        """Retrieval basierend auf juristischen Entitäten"""

        entity_results = []

        for entity_type, entity_list in entities.items():
            if not entity_list:
                continue

            # Suche nach spezifischen Entitäten
            for entity in entity_list:
                entity_query = f"{entity_type.rstrip('s')} {entity}"  # "paragraph 123"
                docs = self.vectorstore.similarity_search(entity_query, k=k // 2)

                for doc in docs:
                    entity_results.append(
                        {
                            "content": doc.page_content,
                            "metadata": doc.metadata,
                            "entity_type": entity_type,
                            "entity_value": entity,
                            "retrieval_method": "legal_entity",
                        }
                    )

        return entity_results[:k]

    def _rerank_and_fuse(
        self,
        stage_results: Dict[str, List[Dict]],
        query: str,
        query_info: Dict,
        legal_entities: Dict,
        k: int,
    ) -> List[Dict]:
        """Intelligente Fusion und Reranking aller Stage-Ergebnisse"""

        all_docs = []

        # Sammle alle Dokumente mit Scores
        for stage, docs in stage_results.items():
            for doc in docs:
                doc["stage"] = stage
                all_docs.append(doc)

        # Deduplizierung basierend auf Content
        unique_docs = self._deduplicate_by_content(all_docs)

        # Advanced Scoring für jedes Dokument
        for doc in unique_docs:
            doc["final_score"] = self._calculate_advanced_score(
                doc, query, query_info, legal_entities
            )

        # Sortiere nach finalem Score
        unique_docs.sort(key=lambda x: x["final_score"], reverse=True)

        return unique_docs[:k]

    def _calculate_advanced_score(
        self, doc: Dict, query: str, query_info: Dict, legal_entities: Dict
    ) -> float:
        """Berechnet erweiterten Relevanz-Score"""

        content = doc["content"].lower()
        base_score = 0.0

        # 1. Stage-spezifische Scores
        if doc["retrieval_method"] == "semantic" and "similarity_score" in doc:
            base_score += (
                1 - doc["similarity_score"]
            ) * 0.4  # Semantic distance zu similarity
        elif doc["retrieval_method"] == "keyword" and "tfidf_score" in doc:
            base_score += doc["tfidf_score"] * 0.3
        elif doc["retrieval_method"] == "legal_entity":
            base_score += 0.5  # Hoher Basis-Score für Entity-Matches

        # 2. Query-Term-Matching
        original_terms = query_info["original"]
        expanded_terms = query_info["expanded"]

        # Original-Terms höher gewichtet
        original_matches = sum(1 for term in original_terms if term in content)
        expanded_matches = sum(1 for term in expanded_terms if term in content)

        term_score = (original_matches * 0.1) + (expanded_matches * 0.05)
        base_score += term_score

        # 3. High-Weight-Terms Bonus
        high_weight_matches = sum(
            1 for term in query_info["high_weight"] if term in content
        )
        base_score += high_weight_matches * 0.15

        # 4. Legal Entity Bonus
        entity_bonus = 0.0
        for entity_type, entity_list in legal_entities.items():
            for entity in entity_list:
                if entity.lower() in content:
                    entity_bonus += 0.1

        base_score += min(entity_bonus, 0.3)  # Cap bei 30%

        # 5. Content-Qualität
        quality_score = self._assess_content_quality(content)
        base_score += quality_score * 0.1

        # 6. Multi-Stage-Bonus (wenn in mehreren Stages gefunden)
        if hasattr(doc, "stage_count") and doc.get("stage_count"):
            base_score += doc["stage_count"] * 0.05

        return min(base_score, 1.0)  # Cap bei 1.0

    def _assess_content_quality(self, content: str) -> float:
        """Bewertet die Qualität des Inhalts"""

        # Länge (optimal um 800-1200 Zeichen)
        length_score = 1.0 - abs(len(content) - 1000) / 1000
        length_score = max(0, min(1, length_score))

        # Strukturiertheit (Satzzeichen, Absätze)
        structure_indicators = (
            content.count(".") + content.count("\n") + content.count(":")
        )
        structure_score = min(structure_indicators / 10, 1.0)

        # Juristische Komplexität (mehr Fachbegriffe = höhere Qualität)
        complex_terms = ["voraussetzung", "tatbestand", "rechtsfolge", "anspruch"]
        complexity_score = sum(1 for term in complex_terms if term in content) / len(
            complex_terms
        )

        return (length_score + structure_score + complexity_score) / 3

    def _deduplicate_by_content(self, docs: List[Dict]) -> List[Dict]:
        """Entfernt Duplikate basierend auf Content-Ähnlichkeit"""

        unique_docs = []
        seen_contents: List[str] = []

        for doc in docs:
            content_sample = doc["content"][:200].lower()  # Erste 200 Zeichen

            # Prüfe Ähnlichkeit zu bereits gesehenen Contents
            is_duplicate = False
            for seen in seen_contents:
                similarity = self._simple_similarity(content_sample, seen)
                if similarity > 0.8:  # 80% Ähnlichkeit = Duplikat
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_docs.append(doc)
                seen_contents.append(content_sample)

        return unique_docs

    def _simple_similarity(self, text1: str, text2: str) -> float:
        """Einfache Textähnlichkeit basierend auf gemeinsamen Wörtern"""

        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def _simple_fusion(
        self, stage_results: Dict[str, List[Dict]], k: int
    ) -> List[Dict]:
        """Einfache Fusion ohne Reranking"""

        all_docs = []
        for docs in stage_results.values():
            all_docs.extend(docs)

        # Einfache Deduplizierung
        unique_docs = self._deduplicate_by_content(all_docs)

        return unique_docs[:k]


class RetrievalAnalyzer:
    """Analysiert und optimiert Retrieval-Performance"""

    def __init__(self):
        self.retrieval_history = []

    def analyze_retrieval_quality(
        self,
        query: str,
        retrieved_docs: List[Dict],
        user_feedback: Optional[Dict] = None,
    ) -> Dict:
        """Analysiert Qualität der Retrieval-Ergebnisse"""

        analysis = {
            "query": query,
            "num_docs": len(retrieved_docs),
            "avg_score": np.mean([doc.get("final_score", 0) for doc in retrieved_docs]),
            "score_distribution": self._calculate_score_distribution(retrieved_docs),
            "coverage_analysis": self._analyze_query_coverage(query, retrieved_docs),
            "diversity_score": self._calculate_diversity(retrieved_docs),
            "timestamp": time.time(),
        }

        if user_feedback:
            analysis["user_feedback"] = user_feedback

        self.retrieval_history.append(analysis)

        return analysis

    def _calculate_score_distribution(self, docs: List[Dict]) -> Dict:
        """Berechnet Score-Verteilung"""

        scores = [doc.get("final_score", 0) for doc in docs]

        if not scores:
            return {"min": 0, "max": 0, "mean": 0, "std": 0}

        return {
            "min": float(np.min(scores)),
            "max": float(np.max(scores)),
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores)),
            "high_quality_ratio": sum(1 for s in scores if s > 0.7) / len(scores),
        }

    def _analyze_query_coverage(self, query: str, docs: List[Dict]) -> Dict:
        """Analysiert wie gut die Query-Begriffe abgedeckt sind"""

        query_terms = set(query.lower().split())

        coverage_per_doc = []
        for doc in docs:
            content_terms = set(doc["content"].lower().split())
            coverage = len(query_terms.intersection(content_terms)) / len(query_terms)
            coverage_per_doc.append(coverage)

        return {
            "avg_coverage": float(np.mean(coverage_per_doc)) if coverage_per_doc else 0,
            "max_coverage": float(np.max(coverage_per_doc)) if coverage_per_doc else 0,
            "coverage_scores": coverage_per_doc,
        }

    def _calculate_diversity(self, docs: List[Dict]) -> float:
        """Berechnet Diversität der Ergebnisse"""

        if len(docs) < 2:
            return 1.0

        # Einfache Diversität basierend auf Wort-Überschneidung
        diversities = []

        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                similarity = self._simple_content_similarity(
                    docs[i]["content"], docs[j]["content"]
                )
                diversity = 1 - similarity
                diversities.append(diversity)

        return float(np.mean(diversities)) if diversities else 1.0

    def _simple_content_similarity(self, content1: str, content2: str) -> float:
        """Einfache Content-Ähnlichkeit"""

        words1 = set(content1.lower().split())
        words2 = set(content2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        union = words1.union(words2)

        return len(intersection) / len(union)

    def get_performance_insights(self) -> Dict:
        """Gibt Performance-Insights basierend auf Historie zurück"""

        if not self.retrieval_history:
            return {"message": "Keine Retrieval-Historie verfügbar"}

        recent_analyses = self.retrieval_history[-10:]  # Letzte 10

        avg_scores = [a["avg_score"] for a in recent_analyses]
        diversity_scores = [a["diversity_score"] for a in recent_analyses]

        return {
            "total_queries": len(self.retrieval_history),
            "recent_performance": {
                "avg_relevance": float(np.mean(avg_scores)),
                "avg_diversity": float(np.mean(diversity_scores)),
                "performance_trend": (
                    "improving" if avg_scores[-1] > avg_scores[0] else "stable"
                ),
            },
            "recommendations": self._generate_recommendations(recent_analyses),
        }

    def _generate_recommendations(self, analyses: List[Dict]) -> List[str]:
        """Generiert Verbesserungsempfehlungen"""

        recommendations = []

        avg_relevance = np.mean([a["avg_score"] for a in analyses])
        avg_diversity = np.mean([a["diversity_score"] for a in analyses])

        if avg_relevance < 0.5:
            recommendations.append(
                "Retrieval-Relevanz ist niedrig - erwäge Query-Expansion oder andere Embedding-Modelle"
            )

        if avg_diversity < 0.3:
            recommendations.append(
                "Ergebnisse sind zu ähnlich - implementiere Diversitäts-basiertes Reranking"
            )

        high_quality_ratios = [
            a["score_distribution"]["high_quality_ratio"] for a in analyses
        ]
        avg_high_quality = np.mean(high_quality_ratios)

        if avg_high_quality < 0.3:
            recommendations.append(
                "Zu wenige hochqualitative Ergebnisse - optimiere Scoring-Algorithmus"
            )

        return recommendations
