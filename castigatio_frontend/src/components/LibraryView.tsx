import React, { useEffect, useState } from "react";

export type Book = {
    id: string;
    titel: string;
    autor: string;
    jahr: number;
    chunk_anzahl: number;
};

const LibraryView: React.FC = () => {
    const [books, setBooks] = useState<Book[]>([]);
    const [loading, setLoading] = useState<boolean>(true);
    const [error, setError] = useState<string | null>(null);
    const [processingBooks, setProcessingBooks] = useState<Set<string>>(new Set());

    const fetchBooks = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch("http://127.0.0.1:8000/api/v1/books");
            if (!response.ok) {
                throw new Error(`Fehler: ${response.status}`);
            }
            const data = await response.json();
            setBooks(data);
        } catch (err: any) {
            setError(err.message || "Unbekannter Fehler");
        } finally {
            setLoading(false);
        }
    };

    const handleIngest = async (bookId: string) => {
        setProcessingBooks(prev => new Set(prev).add(bookId));
        try {
            const response = await fetch(`http://127.0.0.1:8000/api/v1/books/${bookId}/ingest`, {
                method: 'POST',
            });

            if (!response.ok) {
                throw new Error(`Ingestion fehlgeschlagen: ${response.status}`);
            }

            // Refresh books list after successful ingestion
            await fetchBooks();
        } catch (err: any) {
            setError(err.message || "Fehler beim Verarbeiten des Buchs");
        } finally {
            setProcessingBooks(prev => {
                const newSet = new Set(prev);
                newSet.delete(bookId);
                return newSet;
            });
        }
    };

    const handleDelete = async (bookId: string) => {
        if (!confirm("Sind Sie sicher, dass Sie dieses Buch löschen möchten?")) {
            return;
        }

        try {
            const response = await fetch(`http://127.0.0.1:8000/api/v1/books/${bookId}`, {
                method: 'DELETE',
            });

            if (!response.ok) {
                throw new Error(`Löschen fehlgeschlagen: ${response.status}`);
            }

            // Refresh books list after successful deletion
            await fetchBooks();
        } catch (err: any) {
            setError(err.message || "Fehler beim Löschen des Buchs");
        }
    };

    useEffect(() => {
        fetchBooks();
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="flex items-center gap-3">
                    <div className="w-6 h-6 border-2 border-blue-500/20 border-t-blue-500 rounded-full animate-spin"></div>
                    <span className="text-gray-300 text-lg">Lade Bibliothek...</span>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="text-center">
                    <div className="text-red-400 text-lg mb-2">⚠️ Fehler beim Laden</div>
                    <p className="text-gray-400">{error}</p>
                    <button
                        onClick={fetchBooks}
                        className="mt-4 px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                    >
                        Erneut versuchen
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            <div className="flex items-center justify-between">
                <h2 className="text-2xl font-semibold text-blue-400">Bibliothek</h2>
                <button
                    onClick={fetchBooks}
                    className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors text-sm"
                >
                    🔄 Aktualisieren
                </button>
            </div>

            {books.length === 0 ? (
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-8 border border-gray-700/50 text-center">
                    <div className="text-6xl mb-4">📚</div>
                    <h3 className="text-xl font-medium text-gray-300 mb-2">Keine Bücher gefunden</h3>
                    <p className="text-gray-400">
                        Fügen Sie PDFs zum Backend hinzu, um sie hier zu sehen.
                    </p>
                </div>
            ) : (
                <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl border border-gray-700/50 overflow-hidden">
                    <div className="overflow-x-auto">
                        <table className="min-w-full">
                            <thead className="bg-gray-900/50">
                                <tr>
                                    <th className="px-6 py-4 text-left text-gray-300 font-semibold text-sm uppercase tracking-wider">
                                        Autor
                                    </th>
                                    <th className="px-6 py-4 text-left text-gray-300 font-semibold text-sm uppercase tracking-wider">
                                        Titel
                                    </th>
                                    <th className="px-6 py-4 text-left text-gray-300 font-semibold text-sm uppercase tracking-wider">
                                        Jahr
                                    </th>
                                    <th className="px-6 py-4 text-left text-gray-300 font-semibold text-sm uppercase tracking-wider">
                                        Status
                                    </th>
                                    <th className="px-6 py-4 text-left text-gray-300 font-semibold text-sm uppercase tracking-wider">
                                        Aktionen
                                    </th>
                                </tr>
                            </thead>
                            <tbody className="divide-y divide-gray-700/50">
                                {books.map((book) => (
                                    <tr
                                        key={book.id}
                                        className="hover:bg-gray-700/30 transition-colors"
                                    >
                                        <td className="px-6 py-4 text-gray-200 font-medium">
                                            {book.autor}
                                        </td>
                                        <td className="px-6 py-4 text-gray-200">
                                            {book.titel}
                                        </td>
                                        <td className="px-6 py-4 text-gray-200">
                                            {book.jahr}
                                        </td>
                                        <td className="px-6 py-4">
                                            {book.chunk_anzahl > 0 ? (
                                                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-green-900/30 text-green-400 border border-green-700/50">
                                                    ✓ Indiziert ({book.chunk_anzahl} Chunks)
                                                </span>
                                            ) : (
                                                <span className="inline-flex items-center px-3 py-1 rounded-full text-xs font-medium bg-yellow-900/30 text-yellow-400 border border-yellow-700/50">
                                                    ⏳ Nicht indiziert
                                                </span>
                                            )}
                                        </td>
                                        <td className="px-6 py-4">
                                            <div className="flex gap-2">
                                                <button
                                                    onClick={() => handleIngest(book.id)}
                                                    disabled={processingBooks.has(book.id)}
                                                    className="px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white text-sm font-medium rounded transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900"
                                                >
                                                    {processingBooks.has(book.id) ? (
                                                        <div className="flex items-center gap-2">
                                                            <div className="w-3 h-3 border border-white/20 border-t-white rounded-full animate-spin"></div>
                                                            Processing...
                                                        </div>
                                                    ) : (
                                                        "📥 Ingest"
                                                    )}
                                                </button>
                                                <button
                                                    onClick={() => handleDelete(book.id)}
                                                    className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm font-medium rounded transition-colors focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 focus:ring-offset-gray-900"
                                                >
                                                    🗑️ Löschen
                                                </button>
                                            </div>
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}
        </div>
    );
};

export default LibraryView;
