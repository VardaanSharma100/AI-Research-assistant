import os
import time
import tempfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from typing import Dict, List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper,ArxivAPIWrapper

load_dotenv()
groq_api_key=os.getenv("GROQ_API_KEY")
groq_model=os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
if not groq_api_key:
    raise RuntimeError("GROQ_API_KEY not found. Add it to your .env file.")

app=FastAPI(title="AI Research Assistant API",version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pdf_retrievers: Dict[str, object]={}
session_store: Dict[str, ChatMessageHistory]={}

embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
splitter=RecursiveCharacterTextSplitter(chunk_size=768,chunk_overlap=150)


class AskRequest(BaseModel):
    session_id: str="default_session"
    question: str


class QueryRequest(BaseModel):
    query: str

def invoke_with_retry(runnable,payload,config=None,max_retries=3,base_delay=2,timeout=25):
    for attempt in range(max_retries):
        executor=None
        future=None
        try:
            executor=ThreadPoolExecutor(max_workers=1)
            if config is None:
                future=executor.submit(runnable.invoke,payload)
            else:
                future=executor.submit(runnable.invoke,payload,config=config)
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            if future is not None:
                future.cancel()
            if executor is not None:
                executor.shutdown(wait=False,cancel_futures=True)
            if attempt < max_retries-1:
                time.sleep(base_delay*(2**attempt))
                continue
            raise Exception("Request timed out. Please try a shorter or more specific query.")
        except Exception as e:
            if "429" in str(e) and attempt < max_retries-1:
                if executor is not None:
                    executor.shutdown(wait=False,cancel_futures=True)
                time.sleep(base_delay*(2**attempt))
                continue
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=False,cancel_futures=True)

def get_session_history(session_id:str)->BaseChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id]=ChatMessageHistory()
    return session_store[session_id]


def create_retriever_from_files(files: List[UploadFile]):
    documents=[]
    for uploaded_file in files:
        suffix=os.path.splitext(uploaded_file.filename or "upload.pdf")[1] or ".pdf"
        with tempfile.NamedTemporaryFile(delete=False,suffix=suffix) as tempf:
            tempf.write(uploaded_file.file.read())
            temp_path=tempf.name
        try:
            loader=PyPDFLoader(temp_path)
            docs=loader.load()
            documents.extend(docs)
        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass

    if not documents:
        raise HTTPException(status_code=400,detail="No readable PDF content found.")

    splits=splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
    return vectorstore.as_retriever()

def build_pdf_chain(session_id:str):
    retriever=pdf_retrievers.get(session_id)
    if retriever is None:
        raise HTTPException(status_code=400,detail="Upload PDF files first for this session.")

    contextualize_q_system_prompt=(
        "Given a chat history and the latest user question"
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt=ChatPromptTemplate.from_messages(
        [
            ("system",contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human",'{input}')
        ]
    )

    llm=ChatGroq(model=groq_model,api_key=groq_api_key)
    history_aware_retriever=create_history_aware_retriever(llm,retriever,contextualize_q_prompt)

    system_prompt=(
        "You are an Research assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you do not know the answer, say that you "
        "do not know. Keep the answer detailed.\n\n"
        "Context:{context}"
    )
    qa_prompt=ChatPromptTemplate.from_messages(
        [
            ('system',system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human","{input}")
        ]
    )
    question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
    rag_chain=create_retrieval_chain(history_aware_retriever,question_answer_chain)
    return RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key='input',
        history_messages_key='chat_history',
        output_messages_key='answer'
    )


@app.get("/health")
def health():
    return {"status":"ok"}


@app.post("/api/pdf/upload")
def upload_pdf(session_id: str = Form("default_session"),files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400,detail="No files uploaded.")
    retriever=create_retriever_from_files(files)
    pdf_retrievers[session_id]=retriever
    return {"message":"PDFs processed successfully.","session_id":session_id,"file_count":len(files)}


@app.post("/api/pdf/ask")
def ask_pdf(payload: AskRequest):
    chain=build_pdf_chain(payload.session_id)
    response=chain.invoke(
        {"input":payload.question},
        config={'configurable':{'session_id':payload.session_id}}
    )
    return {"answer":response.get("answer","")}


@app.post("/api/arxiv/search")
def arxiv_search(payload: QueryRequest):
    api_wrapper_arxiv=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=700,load_max_docs=1)
    arxiv=ArxivQueryRun(api_wrapper=api_wrapper_arxiv)
    try:
        result=invoke_with_retry(arxiv,payload.query,max_retries=2,timeout=20)
        return {"result":result}
    except Exception as e:
        message=str(e)
        if "429" in message:
            raise HTTPException(status_code=429,detail="Arxiv rate limit reached. Try again shortly.")
        if "timed out" in message.lower():
            raise HTTPException(status_code=504,detail="Arxiv request timed out.")
        raise HTTPException(status_code=500,detail=f"Arxiv search failed: {message}")


@app.post("/api/wikipedia/search")
def wikipedia_search(payload: QueryRequest):
    api_wrapper_wiki=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=700)
    wiki=WikipediaQueryRun(api_wrapper=api_wrapper_wiki)
    try:
        result=invoke_with_retry(wiki,payload.query,max_retries=2,timeout=18)
        return {"result":result}
    except Exception as e:
        message=str(e)
        if "429" in message:
            raise HTTPException(status_code=429,detail="Wikipedia rate limit reached. Try again shortly.")
        if "timed out" in message.lower():
            raise HTTPException(status_code=504,detail="Wikipedia request timed out.")
        raise HTTPException(status_code=500,detail=f"Wikipedia search failed: {message}")
