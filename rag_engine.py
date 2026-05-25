# -*- coding: utf-8 -*-
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

load_dotenv()

def build_rag_chain(pdf_path: str):
    # Load PDF
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} page(s)")

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks")

    # Free HuggingFace embeddings (no API key needed)
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    vector_store = FAISS.from_documents(chunks, embeddings)
    print("Embeddings stored in FAISS")

    # Retriever
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # Prompt
    prompt = PromptTemplate.from_template("""
Answer the question based only on the context below.
If you dont know, say you dont know.

Context: {context}

Question: {question}

Answer:""")

    # Free Groq LLM
    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0
    
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # Chain
    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    print("RAG chain is ready!")
    return chain, retriever


def ask_question(chain_tuple, question: str) -> dict:
    chain, retriever = chain_tuple
    answer = chain.invoke(question)
    source_docs = retriever.invoke(question)
    return {
        "answer": answer,
        "sources": [doc.page_content[:200] for doc in source_docs]
    }