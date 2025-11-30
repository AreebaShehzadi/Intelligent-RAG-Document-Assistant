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
st.title("🤖 Intelligent RAG Document Assistant")
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
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Processing PDF
        with st.spinner("Agent is analyzing the document..."):
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()

            # Splitting text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)

            # Creating Embeddings and Vector Store
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vector_store = FAISS.from_documents(splits, embeddings)
            st.success("Analysis Complete! Agent is ready.")


# --- 3. LangGraph Setup (The Brain) ---

# A. State Definition (The Memory of the Bot)
class GraphState(TypedDict):
    question: str
    context: List[Document]
    answer: str


# B. LLM Initialization
# Using 'tinyllama' for better performance on local machines
llm = ChatOllama(model="tinyllama", temperature=0)


# C. Define Nodes (The Workers)

def retrieve(state):
    """
    Step 1: Retrieve relevant documents from the vector store based on the question.
    """
    print("--- RETRIEVING ---")
    question = state["question"]

    # Retrieve documents
    retriever = vector_store.as_retriever()
    documents = retriever.invoke(question)
    return {"context": documents, "question": question}


def generate(state):
    """
    Step 2: Generate an answer using the LLM and the retrieved context.
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

    # Create Chain (Prompt -> LLM)
    chain = prompt | llm

    # Generate Response
    response = chain.invoke({"context": documents, "question": question})
    return {"answer": response.content}


# D. Build Graph (Connecting the Workflow)
def build_graph():
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    # Define Edges (The Flow)
    # Start -> Retrieve -> Generate -> End
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    # Compile the Graph
    return workflow.compile()


# --- 4. Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
user_query = st.chat_input("Ask something about the PDF...")

if user_query and vector_store:  # Only run if a PDF is uploaded
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.spinner("Running Agent Workflow..."):
        try:
            # 1. Build the Graph
            app = build_graph()

            # 2. Execute the Graph
            inputs = {"question": user_query}
            result = app.invoke(inputs)

            final_answer = result["answer"]

            # 3. Display the Assistant's Response
            st.session_state.messages.append({"role": "assistant", "content": final_answer})
            with st.chat_message("assistant"):
                st.markdown(final_answer)

        except Exception as e:
            st.error(f"Error in Graph Execution: {e}")

elif user_query and not vector_store:
    st.info("Please upload a PDF first to start the chat!")

#python -m streamlit run app.py
#ollama run phi3