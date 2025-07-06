
import { useState } from "react";
import "./App.css";
import LibraryView from "./components/LibraryView";
import QueryView from "./components/QueryView";
import UploadView from "./components/UploadView";
import StatusView from "./components/StatusView";

type ViewType = "library" | "query" | "upload" | "status";

function App() {
  const [currentView, setCurrentView] = useState<ViewType>("library");

  const renderCurrentView = () => {
    switch (currentView) {
      case "library":
        return <LibraryView />;
      case "query":
        return <QueryView />;
      case "upload":
        return <UploadView />;
      case "status":
        return <StatusView />;
      default:
        return <LibraryView />;
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-gray-900 to-slate-800 text-white">
      <header className="bg-black/20 backdrop-blur-sm border-b border-gray-700/50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="py-6">
            <h1 className="text-4xl font-bold flex items-center gap-3 mb-2">
              <span role="img" aria-label="Justitia" className="text-5xl">🏛️</span>
              <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                Castigatio
              </span>
            </h1>
            <p className="text-gray-300 text-lg">Deine private Wissensdatenbank für Strafrecht</p>
          </div>

          {/* Navigation */}
          <nav className="flex space-x-8 pb-4">
            <button
              onClick={() => setCurrentView("library")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${currentView === "library"
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-gray-300 hover:text-white hover:bg-gray-700/50"
                }`}
            >
              📚 Bibliothek
            </button>
            <button
              onClick={() => setCurrentView("query")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${currentView === "query"
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-gray-300 hover:text-white hover:bg-gray-700/50"
                }`}
            >
              💬 Fragen stellen
            </button>
            <button
              onClick={() => setCurrentView("upload")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${currentView === "upload"
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-gray-300 hover:text-white hover:bg-gray-700/50"
                }`}
            >
              📤 Upload
            </button>
            <button
              onClick={() => setCurrentView("status")}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg transition-all duration-200 ${currentView === "status"
                  ? "bg-blue-600 text-white shadow-lg"
                  : "text-gray-300 hover:text-white hover:bg-gray-700/50"
                }`}
            >
              🔍 Status
            </button>
          </nav>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {renderCurrentView()}
      </main>
    </div>
  );
}

export default App;
