# Import necessary libraries
import streamlit as st 
import fitz  # PyMuPDF for reading PDF files
import time
import os
import faiss  # Facebook AI Similarity Search for vector indexing
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama.chat_models import ChatOllama

# Fixing torch.classes path to avoid import errors in some environments
torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

# Define constants
MODEL = "phi4"

# Initialize RAG chain in session state if not already initialized
if "rag_chain" not in st.session_state:
    # Load the chat model from Ollama
    st.session_state.llm = ChatOllama(
                            model= MODEL,
                            temperature= 0.1,
                            num_ctx=16384
                        )
    # Default template prompting the user to upload PDFs
    st.session_state.template = """Answer the question to the best of your knowledge in a succinct manner. REQUEST THE USER TO UPLOAD PDFs for more context on the topic which will help you better assist the user.
                       
                          Question: {question}
                       """
    # Create a LangChain prompt template from the template string
    st.session_state.prompt = ChatPromptTemplate.from_template(st.session_state.template)
    # Define the RAG chain with a passthrough + prompt + model + output parser
    st.session_state.rag_chain = (
                                    RunnableLambda(lambda question: {"question": question})
                                    | st.session_state.prompt
                                    | st.session_state.llm
                                    | StrOutputParser()
                                )

# Initialize FAISS vector DB and embedding model if not already initialized
if "vector_db" not in st.session_state:
    st.session_state.text_id_map = {}  # Mapping of text chunk IDs to content
    # Load sentence-transformer model for embeddings
    st.session_state.embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1") 
    # Get dimension of embeddings
    sample_encoding = st.session_state.embedding_model.encode(["example sentence"])[0]
    dimension = sample_encoding.shape[0]
    # Initialize FAISS index for inner product search
    st.session_state.vector_db = faiss.IndexFlatIP(dimension)

# Extract text from a PDF file using PyMuPDF
def extract_text(file):
    doc = fitz.open(stream=file, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Split text into overlapping chunks
def split_text(text, chunk_size = 500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

# Normalize vectors to unit length (for cosine similarity in FAISS)
def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

# Process uploaded PDF files and update vector DB
def process_pdf(*pdf_files):
    sample_encoding = st.session_state.embedding_model.encode(["example sentence"])[0]
    dimension = sample_encoding.shape[0]
    # Re-initialize the FAISS index
    st.session_state.vector_db = faiss.IndexFlatIP(dimension)
    text_chunks = []
    st.session_state.text_id_map = {}
    if len(pdf_files[0]) != 0:
        for file in pdf_files[0]:
            text = extract_text(file)
            text_chunks += split_text(text)
    # Map chunk index to text
    st.session_state.text_id_map = {i: chunk for i, chunk in enumerate(text_chunks)}
    # Generate and normalize embeddings
    embeddings = normalize(np.array(st.session_state.embedding_model.encode(text_chunks, show_progress_bar=True)).astype('float32'))
    # Add to FAISS index
    st.session_state.vector_db.add(embeddings)
    # Update template to use context from PDFs
    st.session_state.template = """ Answer the question succinctly based ONLY on the following context:
                            {context}

                            Question: {question}
                       """
    st.session_state.prompt = ChatPromptTemplate.from_template(st.session_state.template)
    # Define retriever function and updated RAG chain
    st.session_state.retriever = RunnableLambda(lambda question: {"context": get_similar_chunks(question), "question": question})
    st.session_state.rag_chain = (
        st.session_state.retriever
        | st.session_state.prompt
        | st.session_state.llm
        | StrOutputParser()
    )

# Retrieve top 3 most similar chunks to a question using FAISS
def get_similar_chunks(question):
    # Encode and normalize the query
    query_embedding = normalize(np.array(st.session_state.embedding_model.encode([question], show_progress_bar=True)).astype('float32'))
    # Perform search in FAISS index
    D, I = st.session_state.vector_db.search(query_embedding, k=3)
    # Get corresponding chunks
    top_chunks = [st.session_state.text_id_map[i] for i in I[0]]
    return "\n\n".join(top_chunks)

# Function to simulate typing effect for assistant responses
def response_generator(prompt):
    for word in prompt.split():
        yield word + " "
        time.sleep(0.05)

# App title in Streamlit
app_title = st.title("Chat with PDFs 📑", anchor=False)

# Initialize message history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages in the chat
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Main chat input field
if question:= st.chat_input("Upload the pdf documents you want to chat with and start asking questions."):
    result = ""
    # Append user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        # Generate response from RAG chain
        result = st.session_state.rag_chain.invoke(question)
        # Simulate typing animation
        st.write_stream(response_generator(result))
    # Append assistant message
    st.session_state.messages.append({"role": "assistant", "content": result})

# Sidebar for uploading PDFs
with st.sidebar:
    st.subheader("Your Documents")
    st.write("Upload your PDF here and click on 'Process'")
    # File uploader accepts multiple PDF files
    uploaded_files = st.file_uploader(
        "Choose a PDF file", accept_multiple_files=True, type="pdf"
    )
    # Button to trigger PDF processing
    st.button(label="Process", on_click=process_pdf, args=(uploaded_files,))
