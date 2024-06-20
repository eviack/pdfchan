from langchain.text_splitter import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain

import streamlit as st
import os
from PyPDF2 import PdfReader
from io import StringIO

from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def read_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            text += page.extract_text()

    return text


def to_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=8000,
        chunk_overlap=1000, 
        length_function = len
        )
    chunks = text_splitter.split_text(raw_text)

    return chunks

def dump_vectorbases(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_dump = FAISS.from_texts(chunks, embeddings)

    return vector_dump


def get_conversational_chain(vector_dump):
    model = ChatGoogleGenerativeAI(model="models/gemini-1.5-pro-latest", temperature=0.3)
    mem = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm = model,
        retriever=vector_dump.as_retriever(),
        memory = mem
    )

    return conversational_chain


def generate_chat_history_text(chat_history):
    output = StringIO()
    for message in chat_history:
        output.write(f"user: {message['user']}\n")
        output.write(f"assistant: {message['assistant']}\n")
        output.write("\n") 
    return output.getvalue()


def display_chat_history():
    for entry in st.session_state.chat_history:
        if entry.get('user'):
            with st.chat_message("user"):
                st.write(entry['user'])
        if entry.get('assistant'):
            with st.chat_message("assistant"):
                st.markdown(entry['assistant'], unsafe_allow_html=True)
  

def main():
    st.title("PQuery")
    st.caption("Chat with your documents!")

    if "vector_dump" not in st.session_state:
        st.session_state.vector_dump = None
    if "conversational_chain" not in st.session_state:
        st.session_state.conversational_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []


    with st.sidebar:
        pdf_load = st.sidebar.file_uploader("Upload your pdfs here", accept_multiple_files=True)

        if st.sidebar.button("Process") and pdf_load:
            raw_text = read_pdfs(pdf_load)
            chunks = to_chunks(raw_text)
            vector_dump = dump_vectorbases(chunks)
            st.session_state.vector_dump = vector_dump
            st.session_state.conversational_chain = get_conversational_chain(vector_dump)
            st.sidebar.success("PDF processed successfully !")
            

    if st.session_state.conversational_chain:
        with st.chat_message("assistant"):
            st.markdown("Your documents have been *uploaded* and *processed*. You can now ask me questions about them !")

        if user_input := st.chat_input("Ask your doubts away!"):
            with st.spinner('Processing...'):
                response = st.session_state.conversational_chain({
                    'question': user_input,
                    'chat_history': st.session_state.chat_history 
                })
                st.session_state.chat_history.append({"user": user_input, "assistant": response['answer']})
                
                display_chat_history() 


    if st.session_state.chat_history:
        st.sidebar.markdown("Save the raw text of your chat as .txt")

        chat_history_text = generate_chat_history_text(st.session_state.chat_history)
        button = st.sidebar.download_button(label="Download Chat History", data=chat_history_text, file_name='chat_history.txt', mime='text/plain')

        if button:
            st.sidebar.success("Chat history saved successfully !")
            display_chat_history()
           


if __name__=="__main__":
    main()

