# Chat with PDFs 📑 - A Demo of Retrieval-Augmented Generation (RAG)

This is a demonstration of a Retrieval-Augmented Generation (RAG) application that allows users to interact with uploaded PDFs through a chat interface. By leveraging the principles of RAG, this app uses FAISS for efficient similarity search and a transformer-based model from Ollama to generate contextually relevant responses based on the content of the uploaded PDFs.

## Features
- **Chat with PDFs:** Upload your PDFs and ask questions about their content. The app retrieves the most relevant sections and provides answers based on the document.
- **RAG (Retrieval-Augmented Generation):** Combines retrieval (finding relevant document chunks) and generation (producing contextually relevant answers).
- **Real-time Response Generation:** Interactive chat interface powered by LangChain and Ollama.
- **Multi-Model Support:** Easily switch between different Ollama models for varied responses.
  
## Tech Stack
- **Ollama:** Used for large language models (LLMs) to generate answers based on document context.
- **LangChain:** For orchestrating the RAG pipeline (prompt management, chaining LLMs, and output parsing).
- **Streamlit:** For creating the chat interface and managing user interaction.
- **FAISS:** For efficient vector search in large collections of text data.
- **Sentence Transformers:** For generating high-quality document embeddings.
  
## Setup Instructions

### Requirements
1. Python 3.10+
2. Install the necessary Python dependencies:
   ```bash
   pip install streamlit fitz faiss-cpu sentence-transformers langchain ollama
    ```

### Running the App Locally

1. Clone the repository or download the project files.
2. Install the dependencies listed above.
3. Start the Streamlit app:

   ```bash
   streamlit run app.py
   ```
4. Once the app starts, it will open in the default web browser.

### Switching Models

This app supports multiple Ollama models. To switch models:

1. Modify the constant `MODEL` in the code to the desired model (e.g., `"phi4"`).
2. Pull the model using the following command:

   ```bash
   ollama pull <model_name>
   ```
3. No need to restart the app. The model change will be applied immediately.

### Upload PDFs and Ask Questions

Once the app is running, you can upload your PDF files in the sidebar and start chatting with them. The system will extract text from the PDF and use that content to generate answers to your questions.

## How It Works (RAG Pipeline)

The core of this app is a Retrieval-Augmented Generation (RAG) pipeline. Here's how it works:

1. **PDF Parsing:** The app extracts text from uploaded PDFs using PyMuPDF.
2. **Text Chunking:** The extracted text is split into smaller, overlapping chunks to improve retrieval accuracy.
3. **Embedding Generation:** Each chunk is encoded into a vector representation using the SentenceTransformer model.
4. **FAISS Indexing:** The vectors are indexed in FAISS to enable fast similarity search.
5. **Question Processing:** When the user asks a question, the system encodes the question into a vector, searches the FAISS index for the most similar text chunks, and uses those chunks as context.
6. **Answer Generation:** The retrieved context is passed to the Ollama model, which generates a response based on the provided context.

## Future Improvements

* **Multi-threaded Conversations:** Allow users to manage multiple threads, enabling them to chat with different PDFs or contexts simultaneously.
* **Support for Other File Formats:** Extend support for additional document types (e.g., DOCX, TXT, HTML) for more flexibility in file uploads.
* **Model Selection UI:** Implement an interface to select the model interactively, making it easier for users to switch between different Ollama models without modifying the code.

