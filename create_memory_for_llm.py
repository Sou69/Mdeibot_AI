import os
import sys

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

#step-1 Load raw pdf

DATA_PATH="data/"
def load_pdf_files(data):
    if not os.path.exists(data):
        os.makedirs(data)
        print(f"Created directory: {data}. Please add your PDF files there.")
        return []
    loader = DirectoryLoader(data,
                            glob='*.pdf',
                            loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents

documents=load_pdf_files(data=DATA_PATH)
print("length of pdf pages:", len(documents))

if not documents:
    print("No documents found. Exiting.")
    sys.exit()
#step-2 Create Chunks

def create_chunks(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks
text_chunks=create_chunks(extracted_data=documents)
print("Length of text chunks", len(text_chunks))

#step-3 Create Vector Embedding

def get_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return embedding_model

embedding_model = get_embedding_model()

#step-4 store embedding in FAISS

DB_FAISS_PATH = "vectorstore/db_faiss"
db = FAISS.from_documents(text_chunks, embedding_model)
db.save_local(DB_FAISS_PATH)
