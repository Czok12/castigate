from app.services.query_enhancer_service import query_enhancer_service
from app.services.rag_service import RAGService


class DummyXaiService:
    def create_trace(self, request):
        return "dummy-trace-id"

    def log_step(self, **kwargs):
        pass

    def finalize_trace(self, trace_id, answer):
        pass


def test_rag_service_rerank_and_fuse(monkeypatch):
    # Setup
    rag_service = RAGService()
    # Patch xai_service to dummy
    monkeypatch.setattr("app.services.rag_service.xai_service", DummyXaiService())
    # Simuliere Dokumente
    from langchain_core.documents import Document

    docs = [
        Document(
            page_content="Haftung nach § 433 BGB",
            metadata={"semantic_score": 0.8, "source_file": "Rengier.pdf", "page": 12},
        ),
        Document(
            page_content="Vertrag und Vereinbarung",
            metadata={"semantic_score": 0.7, "source_file": "Rengier.pdf", "page": 13},
        ),
        Document(
            page_content="Kündigung und Beendigung",
            metadata={"semantic_score": 0.6, "source_file": "Rengier.pdf", "page": 14},
        ),
    ]
    enhanced_query = query_enhancer_service.enhance_query(
        "Wie ist die Haftung bei Vertragsschluss gemäß § 433 BGB?"
    )
    reranked = rag_service._rerank_and_fuse(docs, enhanced_query)
    assert (
        reranked[0].metadata["relevance_score"]
        >= reranked[1].metadata["relevance_score"]
    )
    assert all("relevance_score" in doc.metadata for doc in reranked)
