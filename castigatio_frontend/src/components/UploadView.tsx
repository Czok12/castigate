import React, { useState } from "react";

const UploadView: React.FC = () => {
    const [files, setFiles] = useState<FileList | null>(null);
    const [uploading, setUploading] = useState(false);
    const [message, setMessage] = useState<string | null>(null);
    const [messageType, setMessageType] = useState<"success" | "error" | null>(null);

    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        setFiles(e.target.files);
        setMessage(null);
    };

    const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
        const droppedFiles = e.dataTransfer.files;
        setFiles(droppedFiles);
        setMessage(null);
    };

    const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
        e.preventDefault();
    };

    const uploadFiles = async () => {
        if (!files || files.length === 0) {
            setMessage("Bitte wählen Sie mindestens eine Datei aus.");
            setMessageType("error");
            return;
        }

        setUploading(true);
        setMessage(null);

        try {
            // Note: This is a placeholder - the actual implementation would need
            // to use Tauri's file system APIs to copy files to the backend's pdf_bibliothek folder

            for (let i = 0; i < files.length; i++) {
                const file = files[i];

                if (!file.name.toLowerCase().endsWith('.pdf')) {
                    throw new Error(`${file.name} ist keine PDF-Datei.`);
                }

                // For now, this is just a simulation
                // In a real implementation, you would:
                // 1. Use Tauri's dialog API to show a save dialog
                // 2. Use Tauri's fs API to copy the file to the backend directory
                // 3. Possibly trigger the backend to refresh its book list

                console.log(`Uploading ${file.name} (${file.size} bytes)`);
            }

            setMessage(`${files.length} Datei(en) erfolgreich hochgeladen!`);
            setMessageType("success");
            setFiles(null);

            // Reset the file input
            const fileInput = document.getElementById('file-input') as HTMLInputElement;
            if (fileInput) {
                fileInput.value = '';
            }

        } catch (error: any) {
            setMessage(error.message || "Fehler beim Hochladen der Dateien.");
            setMessageType("error");
        } finally {
            setUploading(false);
        }
    };

    return (
        <div className="max-w-4xl mx-auto space-y-6">
            <div className="bg-gray-800/50 backdrop-blur-sm rounded-xl p-8 border border-gray-700/50">
                <h2 className="text-2xl font-semibold mb-6 text-blue-400">PDF-Dateien hochladen</h2>

                {/* Drop Zone */}
                <div
                    onDrop={handleDrop}
                    onDragOver={handleDragOver}
                    className="border-2 border-dashed border-gray-600 rounded-lg p-8 text-center hover:border-gray-500 transition-colors"
                >
                    <div className="space-y-4">
                        <div className="text-6xl text-gray-500">📁</div>
                        <div>
                            <p className="text-lg text-gray-300 mb-2">
                                Ziehen Sie PDF-Dateien hierher oder klicken Sie zum Auswählen
                            </p>
                            <input
                                id="file-input"
                                type="file"
                                multiple
                                accept=".pdf"
                                onChange={handleFileChange}
                                className="hidden"
                            />
                            <button
                                onClick={() => document.getElementById('file-input')?.click()}
                                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors"
                            >
                                Dateien auswählen
                            </button>
                        </div>
                    </div>
                </div>

                {/* Selected Files */}
                {files && files.length > 0 && (
                    <div className="mt-6">
                        <h3 className="text-lg font-medium text-gray-300 mb-3">
                            Ausgewählte Dateien ({files.length})
                        </h3>
                        <div className="space-y-2">
                            {Array.from(files).map((file, index) => (
                                <div
                                    key={index}
                                    className="flex items-center justify-between bg-gray-700/50 rounded-lg p-3"
                                >
                                    <div className="flex items-center gap-3">
                                        <span className="text-xl">📄</span>
                                        <div>
                                            <p className="text-gray-200 font-medium">{file.name}</p>
                                            <p className="text-gray-400 text-sm">
                                                {(file.size / 1024 / 1024).toFixed(2)} MB
                                            </p>
                                        </div>
                                    </div>
                                    <div className="text-green-400 text-sm">
                                        {file.name.toLowerCase().endsWith('.pdf') ? '✓ PDF' : '⚠️ Nicht PDF'}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {/* Upload Button */}
                {files && files.length > 0 && (
                    <div className="mt-6">
                        <button
                            onClick={uploadFiles}
                            disabled={uploading}
                            className="w-full px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-medium rounded-lg transition-colors"
                        >
                            {uploading ? (
                                <div className="flex items-center justify-center gap-2">
                                    <div className="w-5 h-5 border-2 border-white/20 border-t-white rounded-full animate-spin"></div>
                                    Hochladen...
                                </div>
                            ) : (
                                `${files.length} Datei(en) hochladen`
                            )}
                        </button>
                    </div>
                )}

                {/* Message */}
                {message && (
                    <div className={`mt-6 p-4 rounded-lg ${messageType === 'success'
                            ? 'bg-green-900/50 border border-green-700 text-green-400'
                            : 'bg-red-900/50 border border-red-700 text-red-400'
                        }`}>
                        <p>{message}</p>
                    </div>
                )}
            </div>

            {/* Instructions */}
            <div className="bg-blue-900/20 border border-blue-700/50 rounded-xl p-6">
                <h3 className="text-blue-400 font-semibold text-lg mb-4">Hinweise</h3>
                <ul className="text-gray-300 space-y-2 text-sm">
                    <li>• Nur PDF-Dateien werden unterstützt</li>
                    <li>• Die Dateien werden in das Backend-Verzeichnis kopiert</li>
                    <li>• Nach dem Upload können Sie die Bücher in der Bibliothek verwalten</li>
                    <li>• Verwenden Sie die Ingest-Funktion, um die PDFs zu indexieren</li>
                </ul>
            </div>

            {/* Development Notice */}
            <div className="bg-yellow-900/20 border border-yellow-700/50 rounded-xl p-6">
                <h3 className="text-yellow-400 font-semibold text-lg mb-2">🚧 Entwicklungshinweis</h3>
                <p className="text-gray-300 text-sm">
                    Das Upload-Feature befindet sich noch in der Entwicklung.
                    Derzeit müssen PDF-Dateien manuell in den Ordner <code className="bg-gray-700 px-1 rounded">castigatio_backend/data/pdf_bibliothek/</code> kopiert werden.
                </p>
            </div>
        </div>
    );
};

export default UploadView;
