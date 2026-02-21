import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFacePipeline
from langchain_core.output_parsers import StrOutputParser

# Load .env from project root (which is parent of backend directory)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

def load_llm():
    from langchain_huggingface import ChatHuggingFace
    llm = HuggingFaceEndpoint(
        repo_id="mistralai/Mistral-7B-Instruct-v0.2",
        huggingfacehub_api_token=os.environ.get("HF_TOKEN"),
        max_new_tokens=150,
        temperature=0.3,
    )
    return ChatHuggingFace(llm=llm)

custom_prompt_template = """<|system|>
You are a helpful medical assistant. Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer.
Dont provide anything out of the given context.

Context: {context}</s>
<|user|>
{question}</s>
<|assistant|>
"""

def set_custom_prompt(custom_prompt_template):
    prompt = PromptTemplate(
        template=custom_prompt_template, input_variables=["context", "question"]
    )
    return prompt

class QAChainWrapper:
    def __init__(self, llm, db, prompt):
        self.llm = llm
        self.retriever = db.as_retriever(search_kwargs={"k": 3})
        self.prompt = prompt
        self.chain = self.prompt | self.llm | StrOutputParser()

    def invoke(self, inputs):
        question = inputs.get("query", inputs.get("question", ""))
        docs = self.retriever.invoke(question)
        context = "\n\n".join(doc.page_content for doc in docs)
        result = self.chain.invoke({"context": context, "question": question})
        return {"result": result, "source_documents": docs}

def create_qa_chain():
    DB_FAISS_PATH = "vectorstore/db_faiss"
    print(f"Loading FAISS from: {DB_FAISS_PATH}")
    
    if not os.path.exists(DB_FAISS_PATH):
        raise FileNotFoundError(f"Vectorstore not found at {DB_FAISS_PATH}. Please run 'create_memory_for_llm.py' first.")

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embeddings loaded")

    db = FAISS.load_local(
        DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True
    )
    print("FAISS database loaded")

    print("Creating QA chain...")
    qa_chain = QAChainWrapper(
        llm=load_llm(),
        db=db,
        prompt=set_custom_prompt(custom_prompt_template)
    )
    print("QA chain created successfully")
    return qa_chain

if __name__ == "__main__":
    qa_chain = create_qa_chain()
    try:
        user_query = sys.argv[1] if len(sys.argv) > 1 else input("Write query here: ")
    except EOFError:
        raise SystemExit(
            "No input available. Run with a question argument, e.g.\n"
            "  python connect_memory_with_llm.py \"What is diabetes?\""
        )

    response = qa_chain.invoke({"query": user_query})

    stdout_encoding = sys.stdout.encoding or "utf-8"

    def _safe_printable(text: object) -> str:
        s = str(text)
        return s.encode(stdout_encoding, errors="replace").decode(stdout_encoding, errors="replace")

    print("RESULT:", _safe_printable(response["result"]))
    print("SOURCE DOCUMENTS:", [_safe_printable(doc.page_content) for doc in response["source_documents"]])
