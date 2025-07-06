# castigatio_backend/app/services/xai_service.py
import uuid
from collections import deque
from datetime import datetime
from typing import Dict, Optional

from app.models.rag import QueryRequest
from app.models.xai import DecisionStep, ExplanationTrace


class XAIService:
    """
    Service zur Nachverfolgung und Speicherung von RAG-Entscheidungsprozessen.
    """

    def __init__(self, max_traces: int = 100):
        # Wir speichern die Traces vorerst im Speicher.
        # Für eine persistente Lösung könnte hier eine Datenbank (z.B. MongoDB, JSON-Datei) verwendet werden.
        self.traces: deque[ExplanationTrace] = deque(maxlen=max_traces)
        self.traces_by_id: Dict[str, ExplanationTrace] = {}

    def create_trace(self, request: QueryRequest) -> str:
        """Erstellt einen neuen, leeren Trace für eine Anfrage und gibt die Trace-ID zurück."""
        trace_id = f"trace-{uuid.uuid4()}"

        # Leeren Trace vorläufig speichern
        placeholder_trace = ExplanationTrace(
            trace_id=trace_id,
            request=request,
            final_answer="",
            steps=[],
            total_duration_ms=0.0,
        )
        self.traces_by_id[trace_id] = placeholder_trace

        return trace_id

    def log_step(
        self,
        trace_id: str,
        step_name: str,
        input_data: Dict,
        output_data: Dict,
        start_time: datetime,
        end_time: datetime,
    ):
        """Fügt einen abgeschlossenen Schritt zum Trace hinzu."""
        if trace_id not in self.traces_by_id:
            return

        duration = (end_time - start_time).total_seconds() * 1000
        step = DecisionStep(
            step_name=step_name,
            input=input_data,
            output=output_data,
            start_time=start_time,
            end_time=end_time,
            duration_ms=round(duration, 2),
        )
        self.traces_by_id[trace_id].steps.append(step)

    def finalize_trace(self, trace_id: str, final_answer: str):
        """Schließt einen Trace ab, berechnet die Gesamtdauer und fügt ihn zur Queue hinzu."""
        if trace_id not in self.traces_by_id:
            return

        trace = self.traces_by_id[trace_id]
        trace.final_answer = final_answer

        if trace.steps:
            total_duration = (
                trace.steps[-1].end_time - trace.steps[0].start_time
            ).total_seconds() * 1000
            trace.total_duration_ms = round(total_duration, 2)

        # In die Deque für den Zugriff auf die letzten N Traces einfügen
        self.traces.append(trace)

        # Bereinigung, falls das Dict zu groß wird (optional)
        if len(self.traces_by_id) > (self.traces.maxlen or 100) * 1.2:
            oldest_trace_id = self.traces[0].trace_id
            if oldest_trace_id in self.traces_by_id:
                del self.traces_by_id[oldest_trace_id]

    def get_trace_by_id(self, trace_id: str) -> Optional[ExplanationTrace]:
        """Ruft einen spezifischen Trace anhand seiner ID ab."""
        return self.traces_by_id.get(trace_id)


xai_service = XAIService()
