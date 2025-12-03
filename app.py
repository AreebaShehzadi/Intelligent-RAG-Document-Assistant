import os
import time

# --- 1. FORCE OFFLINE MODE ---
# These settings prevent the library from attempting to connect to the internet,
# ensuring the app runs completely offline without errors.
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

# --- UI Customization & CSS ---
st.markdown(
    """
    <style>
    /* Hide default Streamlit elements for a cleaner look */
    .reportview-container { margin-top: -2em; }
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}

    /* --- WHATSAPP STYLE TYPING ANIMATION CSS --- */
    .typing-indicator {
        display: inline-flex;
        align-items: center;
        column-gap: 6px;
    }
    .dot {
        height: 10px;
        width: 10px;
        background-color: #888;
        border-radius: 50%;
        animation: blink 1.4s infinite both;
    }
    .dot:nth-child(1) { animation-delay: 0s; }
    .dot:nth-child(2) { animation-delay: 0.2s; }
    .dot:nth-child(3) { animation-delay: 0.4s; }

    @keyframes blink {
        0% { opacity: 0.1; transform: scale(0.8); }
        20% { opacity: 1; transform: scale(1); }
        100% { opacity: 0.1; transform: scale(0.8); }
    }
    </style>

    <div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #ff4b4b;'>
        <p style='font-size: 14px; margin: 0;'>
            <b>🚀 100% OFFLINE & PRIVATE:</b> Powered by 
            <code style='color: #ff4b4b'>Qwen 2.5 (1.5B)</code>. 
            Running locally for maximum privacy and speed.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Sidebar: File Upload & Processing ---
with st.sidebar:
    st.header("Knowledge Base")
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")

    vector_store = None
    if uploaded_file:
        # Save the uploaded file temporarily
        with open("temp.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())

        with st.spinner("Agent is analyzing the document..."):
            # Load and split the PDF
            loader = PyPDFLoader("temp.pdf")
            docs = loader.load()

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            splits = text_splitter.split_documents(docs)

            # --- EMBEDDINGS (Offline Configuration) ---
            # Using HuggingFace embeddings running on CPU
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'}
            )

            # Create Vector Store
            vector_store = FAISS.from_documents(splits, embeddings)
            st.success("Analysis Complete! Ready to Chat.")


# --- LangGraph Workflow Setup ---

# Define the state of the graph
class GraphState(TypedDict):
    question: str
    context: List[Document]
    answer: str


# --- LLM INITIALIZATION ---
# Using Qwen 2.5 (1.5B) running locally via Ollama
llm = ChatOllama(model="qwen2.5:1.5b", temperature=0)


def retrieve(state):
    """
    Retrieve relevant documents from the vector store based on the user's question.
    """
    print("--- RETRIEVING ---")
    question = state["question"]
    retriever = vector_store.as_retriever()
    documents = retriever.invoke(question)
    return {"context": documents, "question": question}


def generate(state):
    """
    Generate an answer using the LLM and the retrieved context.
    """
    print("--- GENERATING ---")
    question = state["question"]
    documents = state["context"]

    # --- PROMPT TEMPLATE ---
    prompt = ChatPromptTemplate.from_template(
        """
        You are an intelligent assistant named 'Reeba's AI Agent'. 
        You are helpful, kind, and professional.

        INSTRUCTIONS:
        1. Answer the user's question based ONLY on the provided Context.
        2. Your answer must be in clear and concise English.
        3. If the answer is not available in the context, politely say that you don't know.

        Context: {context}

        Question: {question}
        Answer:
        """
    )

    chain = prompt | llm
    response = chain.invoke({"context": documents, "question": question})
    return {"answer": response.content}


def build_graph():
    """
    Compile the LangGraph workflow.
    """
    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)
    return workflow.compile()


# --- Chat Interface ---

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle User Input
user_query = st.chat_input("Ask something about the document...")

if user_query and vector_store:
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 2. Display Assistant Response with Animation
    with st.chat_message("assistant"):
        # Create a placeholder for the typing animation
        response_placeholder = st.empty()

        # Inject the custom HTML for the "Bouncing Dots" animation
        response_placeholder.markdown(
            """
            <div class="typing-indicator">
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
            </div>
            """,
            unsafe_allow_html=True
        )

        try:
            # 3. Execute the Graph (Backend Processing)
            app = build_graph()
            inputs = {"question": user_query}
            result = app.invoke(inputs)
            final_answer = result["answer"]

            # 4. Remove Animation & Show Final Answer
            response_placeholder.empty()  # Clear the dots
            st.markdown(final_answer)  # Display the text

            # Save assistant response to history
            st.session_state.messages.append({"role": "assistant", "content": final_answer})

        except Exception as e:
            response_placeholder.empty()
            st.error(f"Error: {e}")

elif user_query and not vector_store:
    st.info("Please upload a PDF first!")



#ollama run qwen2.5:1.5b
#streamlit run app.py
