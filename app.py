import streamlit as st 
import random
import time

def response_generator():
    response = random.choice(
        [
            "Hello there! How can I assist you today?",
            "Hi, human! Is there anything I can help you with?",
            "Do you need help?",
        ]
    )

    for word in response.split():
        yield word + " "
        time.sleep(0.05)

app_title = st.title("Chat with PDFs 📑", anchor=False)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

with st.chat_message("assistant"):
    response = st.write_stream(response_generator())

st.session_state.messages.append({"role": "assistant", "content": response})

with st.sidebar:
    st.subheader("Your Documents")
    st.write("Upload your PDF here and click on 'Process'")
    uploaded_files = st.file_uploader(
        "Choose a PDF file", accept_multiple_files=True
    )
    st.button(label="Process")