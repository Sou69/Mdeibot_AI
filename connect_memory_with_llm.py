import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFaceHub
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

def load_llm():
    import streamlit as st
    hf_token = st.secrets.get("HF_TOKEN") or os.environ.get("HF_TOKEN")
    os.environ["HUGGINGFACEHUB_API_TOKEN"] = hf_token
    return HuggingFaceHub(
        repo_id="HuggingFaceH4/zephyr-7b-beta",
        huggingfacehub_api_token=hf_token,
        model_kwargs={"temperature": 0.3, "max_new_tokens": 256}
    )

custom_prompt_template = """You are a helpful medical assistant. Use the context to answer the question.
If you don't know the answer, say you don't know.

Context: {context}
Question: {question}
Answer:"""

def set_custom_prompt(custom_prompt_template):
    return PromptTemplate(
        template=custom_prompt_template,
        input_variables=["context", "question"]
    )

class QAChainWrapper:
    def __init__(self, llm, db, prompt):
        self.retriever = db.as_retriever(search_kwargs={"k": 3})
        self.chain = prompt | llm | StrOutputParser()

    def invoke(self, inputs):
        question = inputs.get("query", inputs.get("question", ""))
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        result = self.chain.invoke({"context": context, "question": question})
        return {"result": result, "source_documents": docs}

def create_qa_chain():
    DB_FAISS_PATH = "vectorstore/db_faiss"
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"Vectorstore not found at {DB_FAISS_PATH}")
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local(
        DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
    )
    return QAChainWrapper(
        llm=load_llm(),
        db=db,
        prompt=set_custom_prompt(custom_prompt_template)
    )
