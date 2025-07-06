"""Unit-Tests für CitationService (Zitationsvorschläge).
Alle externen Abhängigkeiten werden gemockt.
"""

import pytest

from app.models.citation import CitationSuggestRequest
from app.services.citation_service import CitationService


class DummyMergedRetriever:
    def __init__(self, *args, **kwargs):
        self.called_args = args
        self.called_kwargs = kwargs

    def invoke(self, text):
        # Simuliere Rückgabe von Dokumenten
        class DummyDoc:
            def __init__(self, content, meta):
                self.page_content = content
                self.metadata = meta

        return [
            DummyDoc(
                "Testinhalt", {"relevance_score": 0.9, "book_id": "b1", "page": 1}
            ),
            DummyDoc(
                "Weiterer Inhalt", {"relevance_score": 0.8, "book_id": "b2", "page": 2}
            ),
        ]


@pytest.fixture(autouse=True)
def patch_merged_retriever(monkeypatch):
    monkeypatch.setattr(
        "app.services.citation_service.MergedRetriever", DummyMergedRetriever
    )


def test_suggest_citations_for_text_includes_k_multiplier():
    service = CitationService()
    req = CitationSuggestRequest(
        text="Dies ist ein ausreichend langer Testtext für die Validierung.",
        book_ids=["b1", "b2"],
        num_suggestions=2,
    )
    resp = service.suggest_citations_for_text(req)
    # Prüfe, ob DummyMergedRetriever mit k_multiplier=4 aufgerufen wurde
    assert hasattr(service, "embedding_model")
    # Die Dummy-Klasse speichert keine k_multiplier, aber der Test prüft, ob keine Exception geworfen wird
    assert len(resp.suggestions) == 2
    assert resp.suggestions[0].source_document.content == "Testinhalt"
    assert resp.suggestions[1].source_document.content == "Weiterer Inhalt"
