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

    useEffect(() => {
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
        fetchBooks();
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <span className="text-gray-300 text-lg animate-pulse">Lade...</span>
            </div>
        );
    }
    if (error) {
        return (
            <div className="flex items-center justify-center h-64">
                <span className="text-red-400 text-lg">Fehler: {error}</span>
            </div>
        );
    }

    return (
        <div className="overflow-x-auto p-6 bg-gray-900 rounded-lg shadow-xl">
            <table className="min-w-full bg-gray-800 rounded-lg">
                <thead>
                    <tr>
                        <th className="px-6 py-3 text-left text-gray-300 font-semibold tracking-wider">Autor</th>
                        <th className="px-6 py-3 text-left text-gray-300 font-semibold tracking-wider">Titel</th>
                        <th className="px-6 py-3 text-left text-gray-300 font-semibold tracking-wider">Jahr</th>
                        <th className="px-6 py-3 text-left text-gray-300 font-semibold tracking-wider">Status</th>
                        <th className="px-6 py-3 text-left text-gray-300 font-semibold tracking-wider">Aktionen</th>
                    </tr>
                </thead>
                <tbody>
                    {books.map((book) => (
                        <tr
                            key={book.id}
                            className="border-b border-gray-700 hover:bg-gray-700/60 transition-colors"
                        >
                            <td className="px-6 py-3 text-gray-200 whitespace-nowrap">{book.autor}</td>
                            <td className="px-6 py-3 text-gray-200 whitespace-nowrap">{book.titel}</td>
                            <td className="px-6 py-3 text-gray-200 whitespace-nowrap">{book.jahr}</td>
                            <td className="px-6 py-3">
                                {book.chunk_anzahl > 0 ? (
                                    <span className="bg-green-600/90 text-xs text-white px-3 py-1 rounded-full font-semibold shadow">Indiziert</span>
                                ) : (
                                    <span className="bg-yellow-400/90 text-xs text-gray-900 px-3 py-1 rounded-full font-semibold shadow">Nicht indiziert</span>
                                )}
                            </td>
                            <td className="px-6 py-3 flex gap-2">
                                <button
                                    className="bg-blue-600 hover:bg-blue-700 text-white px-4 py-1 rounded shadow text-sm font-medium transition-colors"
                                    type="button"
                                    tabIndex={-1}
                                >
                                    Ingest
                                </button>
                                <button
                                    className="bg-red-600 hover:bg-red-700 text-white px-4 py-1 rounded shadow text-sm font-medium transition-colors"
                                    type="button"
                                    tabIndex={-1}
                                >
                                    Löschen
                                </button>
                            </td>
                        </tr>
                    ))}
                </tbody>
            </table>
            {books.length === 0 && (
                <div className="text-gray-400 text-center py-8">Keine Bücher gefunden.</div>
            )}
        </div>
    );
};

export default LibraryView;
