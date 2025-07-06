import os

import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings  # Korrekter Import
from langchain_community.vectorstores import FAISS
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama.llms import OllamaLLM

# --- Konfiguration ---
DB_PATH = "faiss_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"

# --- Lade-Funktionen (mit Caching für bessere Performance) ---


@st.cache_resource
def load_retriever():
    """Lädt nur den Retriever, der für beide Funktionen benötigt wird."""
    if not os.path.exists(DB_PATH):
        return None  # Gibt None zurück, wenn die DB nicht existiert

    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "mps"},  # Für deinen Mac M3 optimiert
    )
    vectorstore = FAISS.load_local(
        DB_PATH, embedding_model, allow_dangerous_deserialization=True
    )
    # search_kwargs={'k': 4} -> Finde die 4 relevantesten Chunks. Ein guter Kompromiss.
    return vectorstore.as_retriever(search_kwargs={"k": 4})


@st.cache_resource
def load_llm():
    """Lädt das Sprachmodell separat."""
    return OllamaLLM(model="llama3.2")  # Oder welches Modell du auch immer nutzt


# --- Haupt-Logik der App ---

st.set_page_config(page_title="Jura-Assistent Pro", layout="wide")
st.title("📚 Jura-Assistent Pro")
st.markdown("---")

# Lade die Komponenten und prüfe, ob die Datenbank existiert
retriever = load_retriever()

if retriever is None:
    st.error(f"Fehler: Die Vektordatenbank unter '{DB_PATH}' wurde nicht gefunden.")
    st.warning(
        "Bitte führe zuerst das Skript 'ingest.py' aus, um die Datenbank für dein Lehrbuch zu erstellen."
    )
    st.stop()  # Hält die Ausführung der App an, wenn die DB fehlt

# Erstelle Tabs für die verschiedenen Modi
tab1, tab2 = st.tabs(["💬 RAG-Assistent (Frage & Antwort)", "🔍 Quellen-Finder"])

# --- Tab 1: Der RAG-Assistent ---
with tab1:
    st.header("Stelle eine Frage an dein Lehrbuch")
    st.markdown(
        "Dieses Tool nutzt dein Lehrbuch, um präzise Antworten auf deine Fragen zu generieren."
    )

    # Lade das LLM nur, wenn dieser Tab aktiv ist
    llm = load_llm()

    # Definiere die Vorlage für den Prompt (der verbesserte Jura-Prompt)
    prompt_template = """
    Du bist ein hochqualifizierter juristischer Tutor. Deine Aufgabe ist es, die Frage des Studenten präzise und didaktisch aufzubereiten.
    Deine Antwort muss sich strikt und ausschließlich auf die Informationen aus dem bereitgestellten 'Kontext' (Auszüge aus einem Lehrbuch/Gesetz) stützen. Erfinde keine Fakten oder Paragraphen.

    Anweisungen für die Antwort:
    1. Beginne mit einer klaren, direkten Antwort auf die Frage.
    2. Gliedere deine Antwort in logische Abschnitte oder Aufzählungspunkte, um die Lesbarkeit zu erhöhen.
    3. Wenn im Kontext Paragraphen (§) oder Artikel (Art.) erwähnt werden, die für die Antwort relevant sind, nenne diese explizit in deiner Antwort (z.B. "gemäß § 433 BGB...").
    4. Wenn die Informationen im Kontext nicht ausreichen, um die Frage zu beantworten, antworte ausschließlich mit: "Die benötigten Informationen zur Beantwortung dieser Frage sind in den vorliegenden Textauszügen nicht enthalten."

    Kontext:
    {context}

    Frage:
    {question}

    Deine strukturierte Antwort:
    """
    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Baue die RAG-Kette
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # UI für den RAG-Chat
    question = st.text_input(
        "Deine Frage:",
        placeholder="z.B. Was sind die Voraussetzungen der GoA?",
        key="rag_question",
    )

    if question:
        with st.spinner("Ich denke nach und durchsuche die Dokumente..."):
            try:
                answer = rag_chain.invoke(question)
                st.success("Antwort generiert:")
                st.markdown(answer)

                # Bonus: Zeige die relevanten Quellen an
                with st.expander("📖 Verwendete Textstellen (mit Seitenangaben)"):
                    retrieved_docs = retriever.invoke(question)
                    for i, doc in enumerate(retrieved_docs):
                        source = doc.metadata.get("source", "Unbekannt")
                        page = doc.metadata.get("page", "?")
                        chunk_id = doc.metadata.get("chunk_id", "")

                        st.write(f"**Quelle {i+1}:** {source}, Seite {page}")
                        if chunk_id:
                            st.caption(f"Chunk-ID: {chunk_id}")
                        st.info(doc.page_content)
                        st.write("---")
            except Exception as e:
                st.error(f"Ein Fehler ist aufgetreten: {e}")

# --- Tab 2: Der Quellen-Finder ---
with tab2:
    st.header("Finde passende Zitate/Quellen für deinen Text")
    st.markdown(
        "Füge einen von dir verfassten Text ein. Das Tool durchsucht deine Datenbank und zeigt dir die inhaltlich ähnlichsten Passagen als mögliche Quellen an."
    )

    # UI für den Quellen-Finder
    user_text = st.text_area(
        "Dein Text:",
        height=200,
        placeholder="Schreibe hier einen Absatz aus deiner Hausarbeit oder Zusammenfassung...",
        key="source_finder_text",
    )

    if user_text:
        with st.spinner("Suche nach passenden Quellen in der Datenbank..."):
            try:
                # Hier nutzen wir NUR den Retriever, kein LLM!
                retrieved_docs = retriever.invoke(user_text)

                st.success(
                    f"Die {len(retrieved_docs)} relevantesten Quellen wurden gefunden:"
                )

                for i, doc in enumerate(retrieved_docs):
                    # Metadaten für bessere Quellenangaben
                    source = doc.metadata.get("source", "Unbekannt")
                    page = doc.metadata.get("page", "?")

                    st.write("---")
                    st.write(f"**📍 Treffer {i+1}**")
                    st.write(f"**Quelle:** {source} | **Seite:** {page}")
                    # Wir heben den gefundenen Text hervor
                    st.info(doc.page_content)

            except Exception as e:
                st.error(f"Ein Fehler bei der Suche ist aufgetreten: {e}")
