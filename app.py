import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import asyncio
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

# Load environment variables (your Google API Key)
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

async def my_async_function():
    await asyncio.sleep(1)
    st.write("Async function executed!")

# Ensure an event loop exists for the current thread
try:
    loop = asyncio.get_event_loop()
except RuntimeError:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

# Run the async function
if st.button("Run Async Function"):
    loop.run_until_complete(my_async_function())


def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    """
    Splits the given text into smaller chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Generates embeddings for text chunks and stores them in a FAISS vector store.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    # You can optionally save the FAISS index locally for faster loading
    # vector_store.save_local("faiss_index")
    return vector_store

def get_conversational_chain():
    """
    Defines the conversational RAG chain using Gemini.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details.
    If the answer is not available in the provided context, just say "Answer is not available in the context",
    don't provide the wrong answer.

    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question, vector_store):
    """
    Processes user questions, performs similarity search, and gets an answer from the LLM.
    """
    if not vector_store:
        st.error("Please process your PDF documents first.")
        return

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # If you saved your FAISS index, you can load it here:
    # new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    # docs = new_db.similarity_search(user_question)

    # Otherwise, use the in-memory vector_store
    docs = vector_store.similarity_search(user_question)


    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )
    return response["output_text"]

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":robot:")
    st.header("Chat with Multiple PDFs :robot:")

    user_question = st.text_input("Ask a question about your documents:")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_question:
        # Add user question to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                if st.session_state.vector_store:
                    response_text = user_input(user_question, st.session_state.vector_store)
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                else:
                    st.warning("Please upload and process PDF documents first.")
                    st.session_state.messages.append({"role": "assistant", "content": "Please upload and process PDF documents first."})


    with st.sidebar:
        st.title("Your Documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF files here and click on 'Process'",
            accept_multiple_files=True
        )
        if st.button("Process"):
            if pdf_docs:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    st.session_state.vector_store = get_vector_store(text_chunks)
                st.success("PDFs processed successfully!")
                st.session_state.messages = [{"role": "assistant", "content": "PDFs processed! Ask me a question about them."}]
                 # Rerun to update chat history

if __name__ == "__main__":
    main()