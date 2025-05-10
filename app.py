import streamlit as st 
import fitz
import time
import os
import faiss
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_ollama.chat_models import ChatOllama

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "phi4"

if "rag_chain" not in st.session_state:
    st.session_state.llm = ChatOllama(
                            model= "phi4",
                            temperature= 0.1,
                            num_ctx=16384
                        )
    st.session_state.template = """Answer the question to the best of your knowledge in a succinct manner. REQUEST THE USER TO UPLOAD PDFs for more context on the topic which will help you better assist the user.
                       
                          Question: {question}
                       """
    st.session_state.prompt = ChatPromptTemplate.from_template(st.session_state.template)
    st.session_state.rag_chain = (
                                    RunnableLambda(lambda question: {"question": question})
                                    | st.session_state.prompt
                                    | st.session_state.llm
                                    | StrOutputParser()
                                )

if "vector_db" not in st.session_state:
    st.session_state.text_id_map = {}
    st.session_state.embedding_model = SentenceTransformer("multi-qa-mpnet-base-dot-v1") 
    sample_encoding = st.session_state.embedding_model.encode(["example sentence"])[0]
    dimension = sample_encoding.shape[0]
    st.session_state.vector_db = faiss.IndexFlatIP(dimension)

def extract_text(file):
    doc = fitz.open(stream=file, filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def split_text(text, chunk_size = 500, overlap=50):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def normalize(vecs):
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

def process_pdf(*pdf_files):
    sample_encoding = st.session_state.embedding_model.encode(["example sentence"])[0]
    dimension = sample_encoding.shape[0]
    st.session_state.vector_db = faiss.IndexFlatIP(dimension)
    text_chunks = []
    st.session_state.text_id_map = {}
    if len(pdf_files[0]) != 0:
        for file in pdf_files[0]:
            text = extract_text(file)
            text_chunks += split_text(text)
    st.session_state.text_id_map = {i: chunk for i, chunk in enumerate(text_chunks)}
    embeddings = normalize(np.array(st.session_state.embedding_model.encode(text_chunks, show_progress_bar=True)).astype('float32'))
    st.session_state.vector_db.add(embeddings)
    st.session_state.template = """ Answer the question succinctly based ONLY on the following context:
                            {context}

                            Question: {question}
                       """
    st.session_state.prompt = ChatPromptTemplate.from_template(st.session_state.template)
    st.session_state.retriever = RunnableLambda(lambda question: {"context": get_similar_chunks(question), "question": question})
    st.session_state.rag_chain = (
        st.session_state.retriever
        | st.session_state.prompt
        | st.session_state.llm
        | StrOutputParser()
    )

def get_similar_chunks(question):
    query_embedding = normalize(np.array(st.session_state.embedding_model.encode([question], show_progress_bar=True)).astype('float32'))
    D, I = st.session_state.vector_db.search(query_embedding, k=3)
    top_chunks = [st.session_state.text_id_map[i] for i in I[0]]
    return "\n\n".join(top_chunks)

def response_generator(prompt):
    for word in prompt.split():
        yield word + " "
        time.sleep(0.05)

app_title = st.title("Chat with PDFs 📑", anchor=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if question:= st.chat_input("Upload the pdf documents you want to chat with and start asking questions."):
    result = ""
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    
    with st.chat_message("assistant"):
        result = st.session_state.rag_chain.invoke(question)
        st.write_stream(response_generator(result))
    st.session_state.messages.append({"role": "assistant", "content": result})

with st.sidebar:
    st.subheader("Your Documents")
    st.write("Upload your PDF here and click on 'Process'")
    uploaded_files = st.file_uploader(
        "Choose a PDF file", accept_multiple_files=True, type="pdf"
    )
    st.button(label="Process", on_click=process_pdf, args=(uploaded_files,))
