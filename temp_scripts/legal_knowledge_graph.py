"""
🕸️ LEGAL KNOWLEDGE GRAPH SYSTEM
===============================

Aufbau und Navigation eines juristischen Wissensgraphen für bessere Kontextualisierung
"""

import pickle
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Optional dependencies
try:
    import networkx as nx

    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    nx = None

try:
    import spacy

    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False
    spacy = None


@dataclass
class LegalEntity:
    """Juristische Entität im Knowledge Graph"""

    entity_id: str
    entity_type: str  # "paragraph", "article", "case", "concept", "institution"
    name: str
    full_text: str
    aliases: List[str]
    metadata: Dict
    importance_score: float


@dataclass
class LegalRelation:
    """Beziehung zwischen juristischen Entitäten"""

    source_id: str
    target_id: str
    relation_type: str  # "cites", "modifies", "defines", "relates_to", "contradicts"
    strength: float
    context: str
    evidence_count: int


class LegalEntityExtractor:
    """Extrahiert juristische Entitäten aus Texten"""

    def __init__(self):
        # NLP-Model laden
        try:
            self.nlp = spacy.load("de_core_news_sm")
        except OSError:
            print(
                "Spacy-Model nicht gefunden. Installieren Sie: python -m spacy download de_core_news_sm"
            )
            self.nlp = None

        # Regex-Patterns für juristische Entitäten
        self.legal_patterns = {
            "paragraph": r"§\s*(\d+[a-z]?(?:\s+Abs\.?\s+\d+)?(?:\s+S\.?\s+\d+)?)",
            "article": r"Art\.?\s*(\d+[a-z]?(?:\s+Abs\.?\s+\d+)?(?:\s+S\.?\s+\d+)?)",
            "bgb": r"§\s*(\d+)\s+BGB",
            "stgb": r"§\s*(\d+)\s+StGB",
            "gg": r"Art\.?\s*(\d+)\s+GG",
            "court_decision": r"(BGH|BVerfG|BFH|BVerwG)\s+(?:Urt\.|Beschl\.?)\s+v\.\s+(\d{1,2}\.\d{1,2}\.\d{4})",
            "legal_concept": r"\b([A-ZÄÖÜ][a-zäöüß]*(?:pflicht|recht|anspruch|haftung|schutz|verbot))\b",
        }

        # Juristische Kernbegriffe
        self.legal_concepts = {
            "vertrag",
            "eigentum",
            "besitz",
            "haftung",
            "schuld",
            "anspruch",
            "rechtsfähigkeit",
            "geschäftsfähigkeit",
            "willenserklärung",
            "rechtsgeschäft",
            "nichtigkeit",
            "anfechtung",
            "verjährung",
            "pfandrecht",
            "hypothek",
            "grundschuld",
            "miete",
            "pacht",
            "kaufvertrag",
            "werkvertrag",
            "dienstvertrag",
            "gesellschaft",
            "vollmacht",
            "stellvertretung",
            "bürgschaft",
            "delikt",
        }

    def extract_entities(self, text: str, source_metadata: Dict) -> List[LegalEntity]:
        """Extrahiert juristische Entitäten aus Text"""

        entities = []

        # 1. Pattern-basierte Extraktion
        for entity_type, pattern in self.legal_patterns.items():
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = LegalEntity(
                    entity_id=f"{entity_type}_{match.group(1).replace(' ', '_')}",
                    entity_type=entity_type,
                    name=match.group(0),
                    full_text=self._extract_context(text, match.start(), match.end()),
                    aliases=[match.group(0), match.group(1)],
                    metadata={
                        "source": source_metadata,
                        "position": match.start(),
                        "pattern_match": True,
                    },
                    importance_score=self._calculate_importance(
                        match.group(0), entity_type
                    ),
                )
                entities.append(entity)

        # 2. NLP-basierte Extraktion (falls verfügbar)
        if self.nlp:
            nlp_entities = self._extract_nlp_entities(text, source_metadata)
            entities.extend(nlp_entities)

        # 3. Konzept-basierte Extraktion
        concept_entities = self._extract_legal_concepts(text, source_metadata)
        entities.extend(concept_entities)

        return entities

    def _extract_context(
        self, text: str, start: int, end: int, context_window: int = 200
    ) -> str:
        """Extrahiert Kontext um gefundene Entität"""

        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)

        return text[context_start:context_end].strip()

    def _calculate_importance(self, entity_text: str, entity_type: str) -> float:
        """Berechnet Wichtigkeits-Score für Entität"""

        base_scores = {
            "paragraph": 0.8,
            "article": 0.9,
            "bgb": 0.85,
            "stgb": 0.85,
            "gg": 0.95,
            "court_decision": 0.9,
            "legal_concept": 0.6,
        }

        base_score = base_scores.get(entity_type, 0.5)

        # Boost für häufig zitierte Paragraphen
        if entity_type in ["paragraph", "bgb", "stgb"]:
            number_match = re.search(r"\d+", entity_text)
            if number_match:
                paragraph_num = int(number_match.group())
                # Wichtige Paragraphen (häufig zitiert)
                important_paragraphs = {
                    # BGB
                    1,
                    7,
                    12,
                    90,
                    104,
                    105,
                    119,
                    123,
                    134,
                    138,
                    145,
                    146,
                    147,
                    241,
                    242,
                    249,
                    250,
                    280,
                    311,
                    323,
                    325,
                    398,
                    433,
                    437,
                    439,
                    459,
                    535,
                    611,
                    631,
                    705,
                    812,
                    823,
                    831,
                    903,
                    929,
                    1004,
                    1922,
                    1924,
                    1937,
                    2018,
                    2032,
                    2078,
                    2087,
                    2303,
                }
                if paragraph_num in important_paragraphs:
                    base_score += 0.1

        return min(base_score, 1.0)

    def _extract_nlp_entities(
        self, text: str, source_metadata: Dict
    ) -> List[LegalEntity]:
        """Extrahiert Entitäten mit NLP"""

        entities = []
        doc = self.nlp(text)

        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "GPE"]:  # Organisationen, Personen, Orte
                entity = LegalEntity(
                    entity_id=f"nlp_{ent.label_.lower()}_{ent.text.replace(' ', '_')}",
                    entity_type=f"nlp_{ent.label_.lower()}",
                    name=ent.text,
                    full_text=self._extract_context(text, ent.start_char, ent.end_char),
                    aliases=[ent.text],
                    metadata={
                        "source": source_metadata,
                        "nlp_label": ent.label_,
                        "confidence": ent._.get("confidence", 0.8),
                    },
                    importance_score=0.4,
                )
                entities.append(entity)

        return entities

    def _extract_legal_concepts(
        self, text: str, source_metadata: Dict
    ) -> List[LegalEntity]:
        """Extrahiert juristische Kernkonzepte"""

        entities = []
        text_lower = text.lower()

        for concept in self.legal_concepts:
            if concept in text_lower:
                positions = [
                    m.start()
                    for m in re.finditer(r"\b" + re.escape(concept) + r"\b", text_lower)
                ]

                for pos in positions:
                    entity = LegalEntity(
                        entity_id=f"concept_{concept}",
                        entity_type="concept",
                        name=concept.title(),
                        full_text=self._extract_context(text, pos, pos + len(concept)),
                        aliases=[concept, concept.title()],
                        metadata={
                            "source": source_metadata,
                            "concept_category": "legal_principle",
                        },
                        importance_score=0.7,
                    )
                    entities.append(entity)
                    break  # Nur einmal pro Konzept pro Dokument

        return entities


class LegalRelationDetector:
    """Erkennt Beziehungen zwischen juristischen Entitäten"""

    def __init__(self):
        # Relation-Pattern
        self.relation_patterns = {
            "cites": [
                r"nach\s+({entity})",
                r"gemäß\s+({entity})",
                r"entsprechend\s+({entity})",
                r"im\s+Sinne\s+(?:des|der)\s+({entity})",
            ],
            "modifies": [
                r"ändert\s+({entity})",
                r"ergänzt\s+({entity})",
                r"ersetzt\s+({entity})",
            ],
            "defines": [
                r"({entity})\s+(?:ist|bedeutet|bezeichnet)",
                r"unter\s+({entity})\s+versteht\s+man",
            ],
            "relates_to": [
                r"im\s+Zusammenhang\s+mit\s+({entity})",
                r"in\s+Verbindung\s+mit\s+({entity})",
                r"bezüglich\s+({entity})",
            ],
        }

    def detect_relations(
        self, text: str, entities: List[LegalEntity]
    ) -> List[LegalRelation]:
        """Erkennt Beziehungen zwischen Entitäten im Text"""

        relations = []

        # Entität-Positionen im Text finden
        entity_positions = {}
        for entity in entities:
            positions = []
            for alias in entity.aliases:
                for match in re.finditer(re.escape(alias), text, re.IGNORECASE):
                    positions.append((match.start(), match.end()))

            if positions:
                entity_positions[entity.entity_id] = {
                    "entity": entity,
                    "positions": positions,
                }

        # Pattern-basierte Relation-Erkennung
        for relation_type, patterns in self.relation_patterns.items():
            for pattern in patterns:
                for entity_id, entity_data in entity_positions.items():
                    entity = entity_data["entity"]

                    # Pattern mit Entität-Namen anwenden
                    for alias in entity.aliases:
                        filled_pattern = pattern.replace("{entity}", re.escape(alias))

                        for match in re.finditer(filled_pattern, text, re.IGNORECASE):
                            # Suche nach anderen Entitäten in der Nähe
                            nearby_entities = self._find_nearby_entities(
                                match.start(), match.end(), entity_positions, entity_id
                            )

                            for nearby_entity_id, distance in nearby_entities:
                                strength = max(
                                    0.1, 1.0 - (distance / 500)
                                )  # Nähe-basierte Stärke

                                relation = LegalRelation(
                                    source_id=nearby_entity_id,
                                    target_id=entity_id,
                                    relation_type=relation_type,
                                    strength=strength,
                                    context=self._extract_context(
                                        text, match.start(), match.end()
                                    ),
                                    evidence_count=1,
                                )
                                relations.append(relation)

        # Co-Occurrence basierte Relationen
        co_occurrence_relations = self._detect_co_occurrence_relations(
            entity_positions, text
        )
        relations.extend(co_occurrence_relations)

        return relations

    def _find_nearby_entities(
        self,
        start: int,
        end: int,
        entity_positions: Dict,
        exclude_entity_id: str,
        max_distance: int = 500,
    ) -> List[Tuple[str, int]]:
        """Findet Entitäten in der Nähe einer Position"""

        nearby = []

        for entity_id, entity_data in entity_positions.items():
            if entity_id == exclude_entity_id:
                continue

            for pos_start, pos_end in entity_data["positions"]:
                # Minimale Distanz berechnen
                if pos_end < start:
                    distance = start - pos_end
                elif pos_start > end:
                    distance = pos_start - end
                else:
                    distance = 0  # Überlappung

                if distance <= max_distance:
                    nearby.append((entity_id, distance))

        return nearby

    def _detect_co_occurrence_relations(
        self, entity_positions: Dict, text: str, window_size: int = 1000
    ) -> List[LegalRelation]:
        """Erkennt Co-Occurrence-basierte Relationen"""

        relations = []
        entity_ids = list(entity_positions.keys())

        for i, entity_id_1 in enumerate(entity_ids):
            for entity_id_2 in entity_ids[i + 1 :]:

                entity_1 = entity_positions[entity_id_1]["entity"]
                entity_2 = entity_positions[entity_id_2]["entity"]

                # Co-Occurrences zählen
                co_occurrences = 0
                total_contexts = []

                for pos1_start, pos1_end in entity_positions[entity_id_1]["positions"]:
                    for pos2_start, pos2_end in entity_positions[entity_id_2][
                        "positions"
                    ]:

                        distance = min(
                            abs(pos1_start - pos2_start),
                            abs(pos1_end - pos2_end),
                            abs(pos1_start - pos2_end),
                            abs(pos1_end - pos2_start),
                        )

                        if distance <= window_size:
                            co_occurrences += 1
                            context_start = min(pos1_start, pos2_start)
                            context_end = max(pos1_end, pos2_end)
                            context = text[context_start:context_end]
                            total_contexts.append(context)

                if co_occurrences > 0:
                    strength = min(0.8, co_occurrences * 0.2)

                    relation = LegalRelation(
                        source_id=entity_id_1,
                        target_id=entity_id_2,
                        relation_type="co_occurs",
                        strength=strength,
                        context=" | ".join(total_contexts[:3]),  # Max 3 Kontexte
                        evidence_count=co_occurrences,
                    )
                    relations.append(relation)

        return relations

    def _extract_context(
        self, text: str, start: int, end: int, window: int = 100
    ) -> str:
        """Extrahiert Kontext um Position"""

        context_start = max(0, start - window)
        context_end = min(len(text), end + window)

        return text[context_start:context_end].strip()


class LegalKnowledgeGraph:
    """Haupt-Knowledge-Graph-System"""

    def __init__(self, graph_path: str = "legal_knowledge_graph.pkl"):
        self.graph_path = graph_path
        self.graph = nx.DiGraph()
        self.entities = {}  # entity_id -> LegalEntity
        self.relations = {}  # relation_id -> LegalRelation

        self.entity_extractor = LegalEntityExtractor()
        self.relation_detector = LegalRelationDetector()

        # Statistiken
        self.stats = {
            "total_entities": 0,
            "total_relations": 0,
            "entity_types": Counter(),
            "relation_types": Counter(),
        }

        self._load_graph()

    def add_document(self, text: str, metadata: Dict) -> Dict:
        """Fügt Dokument zum Knowledge Graph hinzu"""

        # Entitäten extrahieren
        entities = self.entity_extractor.extract_entities(text, metadata)

        # Relationen erkennen
        relations = self.relation_detector.detect_relations(text, entities)

        # Zum Graph hinzufügen
        new_entities = 0
        new_relations = 0

        for entity in entities:
            if entity.entity_id not in self.entities:
                self.entities[entity.entity_id] = entity
                self.graph.add_node(entity.entity_id, **entity.__dict__)
                new_entities += 1
                self.stats["entity_types"][entity.entity_type] += 1
            else:
                # Bestehende Entität aktualisieren
                existing = self.entities[entity.entity_id]
                existing.importance_score = max(
                    existing.importance_score, entity.importance_score
                )

        for relation in relations:
            relation_id = (
                f"{relation.source_id}__{relation.relation_type}__{relation.target_id}"
            )

            if relation_id not in self.relations:
                self.relations[relation_id] = relation
                self.graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    relation_type=relation.relation_type,
                    strength=relation.strength,
                    context=relation.context,
                    evidence_count=relation.evidence_count,
                )
                new_relations += 1
                self.stats["relation_types"][relation.relation_type] += 1
            else:
                # Bestehende Relation verstärken
                existing = self.relations[relation_id]
                existing.evidence_count += relation.evidence_count
                existing.strength = min(
                    1.0, existing.strength + relation.strength * 0.1
                )

        self.stats["total_entities"] = len(self.entities)
        self.stats["total_relations"] = len(self.relations)

        return {
            "new_entities": new_entities,
            "new_relations": new_relations,
            "total_entities": self.stats["total_entities"],
            "total_relations": self.stats["total_relations"],
        }

    def find_related_entities(
        self, entity_id: str, max_depth: int = 2, min_strength: float = 0.3
    ) -> List[Dict]:
        """Findet verwandte Entitäten im Graph"""

        if entity_id not in self.graph:
            return []

        related = []
        visited = set()

        def explore(current_id: str, depth: int, path_strength: float):
            if depth > max_depth or current_id in visited:
                return

            visited.add(current_id)

            # Nachbarn erkunden
            for neighbor in self.graph.neighbors(current_id):
                edge_data = self.graph.get_edge_data(current_id, neighbor)
                strength = edge_data.get("strength", 0.5)
                combined_strength = path_strength * strength

                if combined_strength >= min_strength:
                    if neighbor in self.entities:
                        related.append(
                            {
                                "entity": self.entities[neighbor],
                                "relation_type": edge_data.get(
                                    "relation_type", "related"
                                ),
                                "strength": combined_strength,
                                "depth": depth,
                                "path": f"{current_id} -> {neighbor}",
                            }
                        )

                    explore(neighbor, depth + 1, combined_strength)

        explore(entity_id, 0, 1.0)

        # Nach Stärke sortieren
        related.sort(key=lambda x: x["strength"], reverse=True)

        return related

    def search_entities(
        self,
        query: str,
        entity_types: Optional[List[str]] = None,
        min_importance: float = 0.0,
    ) -> List[LegalEntity]:
        """Sucht Entitäten im Graph"""

        query_lower = query.lower()
        matches = []

        for entity_id, entity in self.entities.items():
            # Typ-Filter
            if entity_types and entity.entity_type not in entity_types:
                continue

            # Importance-Filter
            if entity.importance_score < min_importance:
                continue

            # Text-Matching
            score = 0

            # Exakter Name-Match
            if query_lower == entity.name.lower():
                score = 1.0
            # Name enthält Query
            elif query_lower in entity.name.lower():
                score = 0.8
            # Alias-Match
            elif any(query_lower in alias.lower() for alias in entity.aliases):
                score = 0.6
            # Full-Text-Match
            elif query_lower in entity.full_text.lower():
                score = 0.4

            if score > 0:
                matches.append((entity, score))

        # Nach Score sortieren
        matches.sort(key=lambda x: x[1], reverse=True)

        return [entity for entity, score in matches]

    def get_entity_context(self, entity_id: str, context_depth: int = 1) -> Dict:
        """Liefert erweiterten Kontext für Entität"""

        if entity_id not in self.entities:
            return {}

        entity = self.entities[entity_id]
        related_entities = self.find_related_entities(
            entity_id, max_depth=context_depth
        )

        # Eingehende und ausgehende Relationen
        incoming_relations = []
        outgoing_relations = []

        for relation_id, relation in self.relations.items():
            if relation.target_id == entity_id:
                incoming_relations.append(relation)
            elif relation.source_id == entity_id:
                outgoing_relations.append(relation)

        return {
            "entity": entity,
            "related_entities": related_entities[:10],  # Top 10
            "incoming_relations": incoming_relations,
            "outgoing_relations": outgoing_relations,
            "centrality": self._calculate_centrality(entity_id),
            "importance_ranking": self._get_importance_ranking(entity_id),
        }

    def _calculate_centrality(self, entity_id: str) -> Dict:
        """Berechnet Zentralitäts-Metriken"""

        try:
            return {
                "degree": self.graph.degree(entity_id),
                "in_degree": self.graph.in_degree(entity_id),
                "out_degree": self.graph.out_degree(entity_id),
                "betweenness": nx.betweenness_centrality(self.graph).get(entity_id, 0),
                "pagerank": nx.pagerank(self.graph).get(entity_id, 0),
            }
        except:
            return {
                "degree": 0,
                "in_degree": 0,
                "out_degree": 0,
                "betweenness": 0,
                "pagerank": 0,
            }

    def _get_importance_ranking(self, entity_id: str) -> int:
        """Liefert Importance-Ranking der Entität"""

        sorted_entities = sorted(
            self.entities.items(), key=lambda x: x[1].importance_score, reverse=True
        )

        for rank, (eid, _) in enumerate(sorted_entities, 1):
            if eid == entity_id:
                return rank

        return len(sorted_entities)

    def enhance_retrieval_context(
        self, retrieved_docs: List[Dict], query: str
    ) -> List[Dict]:
        """Erweitert Retrieval-Ergebnisse mit Knowledge Graph Context"""

        enhanced_docs = []

        for doc in retrieved_docs:
            enhanced_doc = doc.copy()
            doc_content = doc.get("content", "")

            # Entitäten im Dokument finden
            temp_entities = self.entity_extractor.extract_entities(
                doc_content, {"temp": True}
            )

            # Bestehende Entitäten im Graph suchen
            found_entities = []
            for temp_entity in temp_entities:
                matches = self.search_entities(
                    temp_entity.name, [temp_entity.entity_type], min_importance=0.5
                )
                found_entities.extend(matches[:2])  # Top 2 matches

            # Knowledge Graph Context hinzufügen
            kg_context = {
                "entities_found": len(found_entities),
                "entity_details": [],
                "related_concepts": [],
                "legal_connections": [],
            }

            for entity in found_entities[:3]:  # Top 3 Entitäten
                context = self.get_entity_context(entity.entity_id, context_depth=1)

                kg_context["entity_details"].append(
                    {
                        "name": entity.name,
                        "type": entity.entity_type,
                        "importance": entity.importance_score,
                        "related_count": len(context.get("related_entities", [])),
                    }
                )

                # Verwandte Konzepte sammeln
                for related in context.get("related_entities", [])[:2]:
                    related_entity = related["entity"]
                    if related_entity.entity_type == "concept":
                        kg_context["related_concepts"].append(related_entity.name)

            enhanced_doc["knowledge_graph_context"] = kg_context
            enhanced_docs.append(enhanced_doc)

        return enhanced_docs

    def get_graph_statistics(self) -> Dict:
        """Liefert Graph-Statistiken"""

        return {
            "total_entities": self.stats["total_entities"],
            "total_relations": self.stats["total_relations"],
            "entity_types": dict(self.stats["entity_types"]),
            "relation_types": dict(self.stats["relation_types"]),
            "graph_density": nx.density(self.graph),
            "connected_components": nx.number_weakly_connected_components(self.graph),
            "average_degree": sum(dict(self.graph.degree()).values())
            / max(len(self.graph), 1),
        }

    def _save_graph(self):
        """Speichert Knowledge Graph"""

        graph_data = {
            "entities": self.entities,
            "relations": self.relations,
            "graph": self.graph,
            "stats": self.stats,
        }

        with open(self.graph_path, "wb") as f:
            pickle.dump(graph_data, f)

    def _load_graph(self):
        """Lädt Knowledge Graph"""

        try:
            with open(self.graph_path, "rb") as f:
                graph_data = pickle.load(f)

                self.entities = graph_data.get("entities", {})
                self.relations = graph_data.get("relations", {})
                self.graph = graph_data.get("graph", nx.DiGraph())
                self.stats = graph_data.get(
                    "stats",
                    {
                        "total_entities": 0,
                        "total_relations": 0,
                        "entity_types": Counter(),
                        "relation_types": Counter(),
                    },
                )
        except FileNotFoundError:
            pass  # Neuer Graph

    def save(self):
        """Öffentliche Save-Methode"""
        self._save_graph()
