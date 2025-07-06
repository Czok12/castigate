import React, { useState } from "react";

export type QueryRequest = {
    question: string;
    book_ids?: string[];
    context_size?: number;
};

export type QueryResponse = {
    answer: string;
    sources: Array<{
        book_id: string;
        book_title: string;
        chunk_id: string;
        content: string;
        relevance_score: number;
    }>;
    query_id: string;
    timestamp: string;
};

const QueryView: React.FC = () => {
    const [question, setQuestion] = useState("");
    const [response, setResponse] = useState<QueryResponse | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (e: React.FormEvent) => {
        e.preventDefault();
        if (!question.trim()) return;

        setLoading(true);
        setError(null);
        setResponse(null);

        try {
            const requestBody: QueryRequest = {
                question: question.trim(),
                context_size: 5,
            };

            const res = await fetch("http://127.0.0.1:8000/api/v1/query", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify(requestBody),
            });

            if (!res.ok) {
                throw new Error(`HTTP ${res.status}: ${res.statusText}`);
            }

            const data: QueryResponse = await res.json();
            setResponse(data);
        } catch (err: any) {
            setError(err.message || "Unbekannter Fehler beim Verarbeiten der Anfrage");
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto space-y-6">
            {/* Query Form */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
                <h2 className="text-2xl font-semibold mb-4 text-blue-400">Stelle eine Frage</h2>
                <form onSubmit={handleSubmit} className="space-y-4">
                    <div>
                        <label htmlFor="question" className="block text-sm font-medium text-gray-300 mb-2">
                            Deine Frage zum Strafrecht:
                        </label>
                        <textarea
                            id="question"
                            value={question}
                            onChange={(e) => setQuestion(e.target.value)}
                            placeholder="z.B. Was ist der Unterschied zwischen Mord und Totschlag?"
                            className="w-full px-4 py-3 bg-gray-700/50 border border-gray-600 rounded-lg text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
                            rows={4}
                            disabled={loading}
                        />
                    </div>
                    <button
                        type="submit"
                        disabled={loading || !question.trim()}
                        className="w-full px-6 py-3 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors duration-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900"
                    >
                        {loading ? (
                            <div className="flex items-center justify-center gap-2">
                                <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
                                Verarbeite Anfrage...
                            </div>
                        ) : (
                            "Frage stellen"
                        )}
                    </button>
                </form>
            </div>

            {/* Error Display */}
            {error && (
                <div className="bg-red-900/50 border border-red-700 rounded-xl p-4">
                    <h3 className="text-red-400 font-medium mb-2">Fehler</h3>
                    <p className="text-red-300">{error}</p>
                </div>
            )}

            {/* Response Display */}
            {response && (
                <div className="space-y-6">
                    {/* Answer */}
                    <div className="bg-green-900/20 border border-green-700/50 rounded-xl p-6">
                        <h3 className="text-green-400 font-semibold text-lg mb-4">Antwort</h3>
                        <div className="prose prose-invert max-w-none">
                            <p className="text-gray-200 leading-relaxed whitespace-pre-wrap">
                                {response.answer}
                            </p>
                        </div>
                    </div>

                    {/* Sources */}
                    {response.sources && response.sources.length > 0 && (
                        <div className="bg-blue-900/20 border border-blue-700/50 rounded-xl p-6">
                            <h3 className="text-blue-400 font-semibold text-lg mb-4">
                                Quellen ({response.sources.length})
                            </h3>
                            <div className="space-y-4">
                                {response.sources.map((source, sourceIndex) => (
                                    <div
                                        key={`${source.book_id}-${source.chunk_id}-${sourceIndex}`}
                                        className="bg-gray-800/50 rounded-lg p-4 border border-gray-700/30"
                                    >
                                        <div className="flex justify-between items-start mb-2">
                                            <h4 className="font-medium text-blue-300">
                                                {source.book_title}
                                            </h4>
                                            <span className="text-xs bg-blue-600/20 text-blue-300 px-2 py-1 rounded">
                                                Relevanz: {(source.relevance_score * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                        <p className="text-gray-300 text-sm leading-relaxed">
                                            {source.content}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}

                    {/* Query Metadata */}
                    <div className="text-xs text-gray-500">
                        Query ID: {response.query_id} | Zeitstempel: {new Date(response.timestamp).toLocaleString('de-DE')}
                    </div>
                </div>
            )}
        </div>
    );
};

export default QueryView;
