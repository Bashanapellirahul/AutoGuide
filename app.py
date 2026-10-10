"""AutoGuide's source-backed Streamlit chat interface."""
import os
from datetime import datetime
import streamlit as st
from rag.config import PROCESSED_DIR, CHUNKS_DIR
from rag.chatbot import AutoGuideChatbot
from rag.evidence import EvidenceChecker
from rag.retrieval import load_chunks

st.set_page_config(page_title="AutoGuide AI", page_icon="🚗", layout="centered")
st.title("🚗 AutoGuide AI")
st.subheader("Your Vehicle Safety Assistant")
st.write("Explore vehicle recalls and safety features with links to official source information.")


@st.cache_resource
def get_checker(signature, semantic):
    return EvidenceChecker(enable_dense=semantic, rerank=semantic)


def corpus_signature():
    paths = sorted(CHUNKS_DIR.glob("*.json*")) or sorted(PROCESSED_DIR.rglob("*.json"))
    return tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in paths)


def show_sources(sources):
    if not sources:
        return
    with st.expander("Sources", expanded=True):
        for source in sources:
            st.markdown(f"**[{source['reference']}] {source['title']}**")
            st.link_button("Open official source", source["url"])
            date = source.get("publication_date") or source.get("retrieved_at")
            if date:
                st.caption(f"Source date: {date}")


with st.sidebar:
    st.header("Knowledge base")
    chunks = load_chunks()
    documents = {c.get("document_id", c["id"]) for c in chunks}
    st.metric("Documents", len(documents))
    st.caption(f"{len(chunks)} searchable chunks")
    if st.button("Prepare knowledge base", type="primary", help="Fetch recall data and NHTSA safety-feature publications, then create chunks."):
        try:
            with st.spinner("Collecting official sources and preparing documents…"):
                from ingestion.api_collector import NHTSAAPICollector
                from ingestion.api_normalizer import process_api_documents
                from ingestion.feature_collector import collect_features
                from rag.chunking import build_chunks
                recalls = NHTSAAPICollector().collect()
                process_api_documents()
                summary = collect_features()
                build_chunks()
                st.cache_resource.clear()
                st.session_state["preparation_message"] = f"Knowledge base prepared. Feature publications collected: {summary['saved']}."
                st.session_state["preparation_failures"] = summary["failures"] + recalls["failures"]
            st.rerun()
        except Exception as error:
            st.error(f"Preparation failed ({type(error).__name__}): {error}")
    if st.session_state.get("preparation_message"):
        st.success(st.session_state["preparation_message"])
        if st.session_state.get("preparation_failures"):
            st.warning("Some sources could not be refreshed. Available sources remain searchable; check their dates.")
    with st.expander("Add another vehicle"):
        with st.form("vehicle_lookup"):
            lookup_make = st.text_input("Make", placeholder="Honda")
            lookup_model = st.text_input("Model", placeholder="Accord")
            lookup_year = st.number_input("Model year", min_value=1960, max_value=datetime.now().year + 1, value=2020)
            lookup = st.form_submit_button("Fetch vehicle recalls")
        if lookup:
            if not lookup_make.strip() or not lookup_model.strip():
                st.error("Enter both make and model.")
            else:
                try:
                    with st.spinner("Fetching vehicle recalls…"):
                        from ingestion.api_collector import NHTSAAPICollector
                        from ingestion.api_normalizer import process_api_documents
                        from rag.chunking import build_chunks
                        collector = NHTSAAPICollector()
                        result = collector.fetch_recalls(lookup_make.strip(), lookup_model.strip(), int(lookup_year))
                        if result is None:
                            st.error("The recall API could not be reached. Try again later.")
                        else:
                            count = len(result["data"].get("results", []))
                            collector.save_recalls(result, lookup_make.strip(), lookup_model.strip(), int(lookup_year))
                            process_api_documents()
                            if count or chunks:
                                build_chunks()
                            st.cache_resource.clear()
                            st.session_state["preparation_message"] = f"The API returned {count} model-year recall records for {lookup_year} {lookup_make} {lookup_model}. This does not verify a VIN."
                            st.session_state["preparation_failures"] = []
                            st.rerun()
                except Exception as error:
                    st.error(f"Vehicle collection failed ({type(error).__name__}): {error}")
    st.divider()
    vehicles = sorted({(str(v.get("make", "")), str(v.get("model", "")), str(v.get("year", "")))
                       for c in chunks if (v := c.get("vehicle"))})
    options = ["All vehicles"] + [f"{year} {make} {model}" for make, model, year in vehicles]
    selected = st.selectbox("Vehicle for recall questions", options)
    vehicle_filter = None
    if selected != "All vehicles":
        make, model, year = vehicles[options.index(selected) - 1]
        vehicle_filter = {"make": make, "model": model, "year": year}
    semantic = st.checkbox("Enable semantic search and reranking", value=True,
                           help="Downloads local models on first use. Keyword search works immediately.")
    mode = st.radio("Answer style", ["Source passages", "AI summary"])
    api_key = None
    if mode == "AI summary" and not os.getenv("OPENAI_API_KEY"):
        api_key = st.text_input("OpenAI API key", type="password", help="Used for this session only.")
        st.caption("AI summaries require an API key. Source passages work without one.")
    if st.button("Clear conversation"):
        st.session_state["messages"] = []
        st.rerun()
    st.caption("Model-year recall records cannot confirm open recalls for an individual VIN.")
    st.link_button("Check a VIN with NHTSA", "https://www.nhtsa.gov/recalls")

if not chunks:
    st.info("Select Prepare knowledge base in the sidebar to get started.")
else:
    st.caption("Try: What is automatic emergency braking? · What recalls affect a 2020 Ford F-150?")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        show_sources(message.get("sources", []))

question = st.chat_input("Ask a vehicle safety question…")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching official sources…"):
                checker = get_checker(corpus_signature(), semantic)
                chatbot = AutoGuideChatbot(checker=checker, api_key=api_key, use_llm=mode == "AI summary")
                result = chatbot.answer(question, vehicle_filter=vehicle_filter)
            st.markdown(result["answer"])
            if result.get("warning"):
                st.warning(result["warning"])
            show_sources(result["sources"])
            st.session_state.messages.append({"role": "assistant", "content": result["answer"], "sources": result["sources"]})
        except ValueError as error:
            st.error(str(error))
        except Exception:
            import logging
            logging.getLogger(__name__).exception("Question failed")
            st.error("The question could not be processed. Try Source passages with semantic search disabled, or check the terminal for details.")
