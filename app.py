import streamlit as st

from financial_agent.agent import FinancialResearchAgent
from financial_agent.ingest import ingest_sec_filings

st.set_page_config(
    page_title="Financial RAG agent",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Financial Research Agent with RAG")
st.write("Ask questions about SEC filings using Groq, Gemini embeddings, and Chroma.")

with st.sidebar:
    st.header("Ingest SEC filings")

    ticker = st.text_input("Ticker", value="AAPL")
    forms = st.multiselect(
        "Forms",
        options=["10-K", "10-Q", "8-K"],
        default=["10-K", "10-Q"],
    )

    limit = st.slider("Number of filings", min_value=1, max_value=10, value=4)
    reset = st.checkbox("Reset vector database before ingesting")

    if st.button("Ingest filings"):
        try:
            with st.spinner("Downloading, chunking, embedding, and indexing..."):
                ingest_sec_filings(
                    ticker=ticker.strip().upper(),
                    forms=forms,
                    limit=limit,
                    reset=reset,
                )

            st.success("Ingestion complete.")

        except ValueError as exc:
            st.error(str(exc))

        except Exception as exc:
            st.error(f"Unexpected error: {exc}")

st.header("Ask a question")

question = st.text_area(
    "Question",
    value="What are Apple's main risk factors according to the latest filings?",
    height=100,
)

if st.button("Ask agent"):
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        agent = FinancialResearchAgent()

        with st.spinner("Thinking and retrieving relevant filing chunks..."):
            answer = agent.answer(question)

        st.markdown(answer)
