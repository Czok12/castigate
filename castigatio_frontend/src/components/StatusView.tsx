import React, { useEffect, useState } from "react";

export type SystemStatus = {
    status: "healthy" | "degraded" | "unhealthy";
    message: string;
    timestamp: string;
    components: {
        database: "healthy" | "unhealthy";
        vector_store: "healthy" | "unhealthy";
        llm: "healthy" | "unhealthy";
    };
};

const StatusView: React.FC = () => {
    const [status, setStatus] = useState<SystemStatus | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    const fetchStatus = async () => {
        setLoading(true);
        setError(null);
        try {
            const response = await fetch("http://127.0.0.1:8000/api/v1/status");
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            const data: SystemStatus = await response.json();
            setStatus(data);
        } catch (err: any) {
            setError(err.message || "Fehler beim Laden des Systemstatus");
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchStatus();
        // Auto-refresh every 30 seconds
        const interval = setInterval(fetchStatus, 30000);
        return () => clearInterval(interval);
    }, []);

    if (loading) {
        return (
            <div className="flex items-center justify-center h-64">
                <div className="flex items-center gap-3">
                    <div className="w-6 h-6 border-2 border-blue-500/20 border-t-blue-500 rounded-full animate-spin"></div>
                    <span className="text-gray-300 text-lg">Lade Systemstatus...</span>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="max-w-4xl mx-auto">
                <div className="bg-red-900/50 border border-red-700 rounded-xl p-6">
                    <h3 className="text-red-400 font-medium mb-2">⚠️ Verbindungsfehler</h3>
                    <p className="text-red-300 mb-4">{error}</p>
                    <button
                        onClick={fetchStatus}
                        className="px-4 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg transition-colors"
                    >
                        Erneut versuchen
                    </button>
                </div>
            </div>
        );
    }

    if (!status) {
        return null;
    }

    const getStatusColor = (componentStatus: string) => {
        switch (componentStatus) {
            case "healthy":
                return "text-green-400 bg-green-900/30 border-green-700";
            case "degraded":
                return "text-yellow-400 bg-yellow-900/30 border-yellow-700";
            case "unhealthy":
                return "text-red-400 bg-red-900/30 border-red-700";
            default:
                return "text-gray-400 bg-gray-900/30 border-gray-700";
        }
    };

    const getStatusIcon = (componentStatus: string) => {
        switch (componentStatus) {
            case "healthy":
                return "✅";
            case "degraded":
                return "⚠️";
            case "unhealthy":
                return "❌";
            default:
                return "❓";
        }
    };

    return (
        <div className="max-w-4xl mx-auto space-y-6">
            {/* Overall Status */}
            <div className={`rounded-xl p-6 border ${getStatusColor(status.status)}`}>
                <div className="flex items-center justify-between mb-4">
                    <h2 className="text-2xl font-semibold flex items-center gap-3">
                        {getStatusIcon(status.status)} Systemstatus
                    </h2>
                    <button
                        onClick={fetchStatus}
                        className="px-3 py-1 bg-gray-700 hover:bg-gray-600 text-white rounded transition-colors text-sm"
                    >
                        🔄 Aktualisieren
                    </button>
                </div>
                <p className="text-lg mb-2">{status.message}</p>
                <p className="text-sm opacity-75">
                    Letzte Aktualisierung: {new Date(status.timestamp).toLocaleString('de-DE')}
                </p>
            </div>

            {/* Component Status */}
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-6 border border-gray-700/50">
                <h3 className="text-xl font-semibold text-blue-400 mb-4">Komponentenstatus</h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    {/* Database */}
                    <div className={`rounded-lg p-4 border ${getStatusColor(status.components.database)}`}>
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-2xl">{getStatusIcon(status.components.database)}</span>
                            <h4 className="font-medium">Datenbank</h4>
                        </div>
                        <p className="text-sm opacity-75 capitalize">{status.components.database}</p>
                    </div>

                    {/* Vector Store */}
                    <div className={`rounded-lg p-4 border ${getStatusColor(status.components.vector_store)}`}>
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-2xl">{getStatusIcon(status.components.vector_store)}</span>
                            <h4 className="font-medium">Vector Store</h4>
                        </div>
                        <p className="text-sm opacity-75 capitalize">{status.components.vector_store}</p>
                    </div>

                    {/* LLM */}
                    <div className={`rounded-lg p-4 border ${getStatusColor(status.components.llm)}`}>
                        <div className="flex items-center gap-2 mb-2">
                            <span className="text-2xl">{getStatusIcon(status.components.llm)}</span>
                            <h4 className="font-medium">LLM</h4>
                        </div>
                        <p className="text-sm opacity-75 capitalize">{status.components.llm}</p>
                    </div>
                </div>
            </div>

            {/* System Information */}
            <div className="bg-blue-900/20 border border-blue-700/50 rounded-xl p-6">
                <h3 className="text-blue-400 font-semibold text-lg mb-4">ℹ️ Systeminformationen</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
                    <div>
                        <p className="text-gray-400">Backend URL:</p>
                        <p className="text-gray-200 font-mono">http://127.0.0.1:8000</p>
                    </div>
                    <div>
                        <p className="text-gray-400">API Version:</p>
                        <p className="text-gray-200">v1</p>
                    </div>
                    <div>
                        <p className="text-gray-400">Frontend Version:</p>
                        <p className="text-gray-200">0.1.0</p>
                    </div>
                    <div>
                        <p className="text-gray-400">Auto-Refresh:</p>
                        <p className="text-gray-200">30 Sekunden</p>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default StatusView;
