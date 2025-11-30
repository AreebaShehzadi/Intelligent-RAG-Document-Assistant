import os

# ---- OFFLINE MODE  ---
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import streamlit as st
from typing import List, TypedDict

# --- Libraries ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- LangGraph Imports ---
from langgraph.graph import END, StateGraph

# --- Page Setup ---
st.set_page_config(page_title="Advanced AI Agent", layout="wide")
st.title("🤖 Intelligent RAG Document Assistant")

# --- UI Update (Llama2 Reflected Here) ---
st.markdown(
    """
    <style>
    .reportview-container { margin-top: -2em; }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    </style>
    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; border-left: 5px solid #ff4b4b;'>
        <p style='font-size: 14px; margin: 0;'>
            <b>🚀 100% OFFLINE MODE:</b> Running locally with 
            <code style='color: #ff4b4b'>tinyllama</code> and 
            <code style='color: #ff4b4b'>Local Embeddings</code>. 
            No Internet Required.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Sidebar ---
with st.sidebar:
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    vector_store = None
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Agent is analyzing the document ..."):
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)

            # --- EMBEDDINGS FOR OFFLINE ---
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            vector_store = FAISS.from_documents(splits, embeddings)
            st.success("Analysis Complete! Ready to Chat.")


# --- LangGraph Setup ---

class GraphState(TypedDict):
    question: str
    context: List[Document]
    answer: str


# LLM (
llm = ChatOllama(model="tinyllama", temperature=0)


def retrieve(state):
    print("--- RETRIEVING ---")
    question = state["question"]
    retriever = vector_store.as_retriever()
    documents = retriever.invoke(question)
    return {"context": documents, "question": question}


def generate(state):
    print("--- GENERATING ---")
    question = state["question"]
    documents = state["context"]

    prompt = ChatPromptTemplate.from_template(
        """
        You are an intelligent document assistant. Your task is to answer the user's question based ONLY on the provided context.

        **CRITICAL INSTRUCTION: Your final answer MUST be a fluent, professional response written ENTIRELY in English.**

        You must read and understand the user's question, even if it is provided in a different language like Roman Urdu (Hinglish), but the response must be strictly in English.

        Always use the following context to answer the question. 
        If you don't know the answer based on the context, just say that you don't know.

        Context: {context}

        Question: {question}
        Answer:
        """
    )

    chain = prompt | llm
    response = chain.invoke({"context": documents, "question": question})
    return {"answer": response.content}

def build_graph():
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


# --- Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask something about the PDF/Document...")

if user_query and vector_store:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Thinking..."):
        try:
            app = build_graph()
            inputs = {"question": user_query}
            result = app.invoke(inputs)
            final_answer = result["answer"]

            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            with st.chat_message("assistant"):
                st.markdown(final_answer)

        except Exception as e:
            st.error(f"Error: {e}")

elif user_query and not vector_store:
    st.info("Please upload a PDF first!")


#ollama run tinyllama
#streamlit run app.py