import streamlit as st
import os
from typing import List, TypedDict

# --- Libraries ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# --- LangGraph Imports (The Advanced Part) ---
from langgraph.graph import END, StateGraph

# --- 1. Page Setup ---
st.set_page_config(page_title="Advanced AI Agent", layout="wide")
st.title("🤖Intelligent RAG Document Assistantt")
#st.caption("Powered by: LangGraph,Phi3 & Python 3.10")
st.markdown(
    """
    <style>
    .reportview-container {
        margin-top: -2em;
    }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    </style>
    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; border-left: 5px solid #ff4b4b;'>
        <p style='font-size: 14px; margin: 0;'>
            <b>🚀 Powered by Agentic RAG:</b> An autonomous AI agent built with 
            <code style='color: #ff4b4b'>LangGraph</code> and 
            <code style='color: #ff4b4b'>Local LLMs</code>. 
            100% Offline & Private.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- 2. Sidebar & File Processing ---
with st.sidebar:
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    vector_store = None
    if uploaded_file:
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Processing PDF
        with st.spinner("Agent is analyzing the document..."):
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)

            # Embeddings
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(splits, embeddings)
            st.success("Analysis Complete! Agent is ready.")


# --- 3. LangGraph Setup (The Brain) ---

# A. State Define karna (Bot ki Diary)
class GraphState(TypedDict):
    question: str
    context: List[Document]
    answer: str


# B. LLM Initialize
llm = ChatOllama(model="phi3", temperature=0)  # Ya "llama3" agar wo installed hai


# C. Nodes Define karna (Workers)

def retrieve(state):
    """
    Step 1: Document mein se data dhoondo
    """
    print("--- RETRIEVING ---")
    question = state["question"]
    # Retriever
    retriever = vector_store.as_retriever()
    documents = retriever.invoke(question)
    return {"context": documents, "question": question}


def generate(state):
    """
    Step 2: Jawab banao
    """
    print("--- GENERATING ---")
    question = state["question"]
    documents = state["context"]

    # Prompt Template
    prompt = ChatPromptTemplate.from_template(
        """You are an intelligent assistant. Use the following context to answer the question. 
        If you don't know the answer, just say that you don't know.

        Context: {context}

        Question: {question}
        Answer:"""
    )

    # Chain (Prompt -> LLM)
    chain = prompt | llm

    # Jawab generate karna
    response = chain.invoke({"context": documents, "question": question})
    return {"answer": response.content}


# D. Graph Banana (Workflow Jorna)
def build_graph():
    workflow = StateGraph(GraphState)

    # Nodes add karna
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    # Rasta banana (Edges)
    # Start -> Retrieve -> Generate -> End
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile (Start karna)
    return workflow.compile()


# --- 4. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_query = st.chat_input("Ask something about the PDF...")

if user_query and vector_store:  # Sirf tab chalay jab PDF upload ho
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Running Agent Workflow..."):
        try:
            # 1. Graph banaya
            app = build_graph()

            # 2. Graph ko chalaya (Invoke)
            inputs = {"question": user_query}
            result = app.invoke(inputs)

            final_answer = result["answer"]

            # 3. Jawab show kiya
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            with st.chat_message("assistant"):
                st.markdown(final_answer)

        except Exception as e:
            st.error(f"Error in Graph: {e}")

elif user_query and not vector_store:
    st.info("Please upload a PDF first!")



#python -m streamlit run app.py
#ollama run phi3