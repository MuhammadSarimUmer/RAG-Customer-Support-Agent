import shutil
import uuid
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import HumanMessage, AIMessage
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.types import Command
from core.settings import settings
from services.loader import load_file_content
router = APIRouter(prefix="/rag", tags=["RAG"])
class ThreadSession(SQLModel, table=True):
    id: str = Field(primary_key=True)          
    created_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = Field(default="active")       
    user_email: Optional[str] = None

engine = create_engine(f"sqlite:///data/threads.db")
SQLModel.metadata.create_all(engine)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"token": settings.hf_token},
)
class ChatRequest(BaseModel):
    message: str
    thread_id: Optional[str] = None  

class ResumeRequest(BaseModel):
    thread_id: str
    confirmed: bool
    email: Optional[str] = None
@router.post("/upload")
async def upload_file(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No file uploaded or file is empty")
    all_docs = []
    try:
        for file in files:
            docs = await load_file_content(file)
            all_docs.extend(docs)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file: {e}")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(all_docs)

    Chroma.from_documents(
        split_docs,
        embedding=embedding_model,              
        persist_directory=settings.chroma_persist_directory,
    )
    return {
        "message": f"Successfully uploaded and processed {len(files)} file(s). "
                   f"Total chunks created: {len(split_docs)}."
    }

@router.post("/chat")
async def chat(payload: ChatRequest, request: Request):
    chatbot = request.app.state.chatbot
    thread_id = payload.thread_id or str(uuid.uuid4())
    if not payload.thread_id:
        with Session(engine) as session:
            session.add(ThreadSession(id=thread_id))
            session.commit()

    config = {"configurable": {"thread_id": thread_id}}

    try:
        result = await chatbot.ainvoke(
            {"query": [HumanMessage(content=payload.message)]},
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chat: {e}")
    if "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"][0].value
        return {
            "thread_id": thread_id,
            "status": "needs_confirmation",
            "data": interrupt_payload,
        }

    return {
        "thread_id": thread_id,
        "status": "done",
        "message": result["query"][-1].content,
    }
@router.post("/chat/resume")
async def resume(payload: ResumeRequest, request: Request):
    chatbot = request.app.state.chatbot
    config = {"configurable": {"thread_id": payload.thread_id}}

    with Session(engine) as session:
        thread = session.get(ThreadSession, payload.thread_id)
        if not thread:
            raise HTTPException(status_code=404, detail="Unknown thread_id")

    try:
        result = await chatbot.ainvoke(
            Command(resume={"confirmed": payload.confirmed, "email": payload.email}),
            config=config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error resuming chat: {e}")

    with Session(engine) as session:
        thread = session.get(ThreadSession, payload.thread_id)
        thread.status = "escalated" if payload.confirmed else "resolved"
        if payload.email:
            thread.user_email = payload.email
        session.add(thread)
        session.commit()

    return {
        "thread_id": payload.thread_id,
        "status": "done",
        "message": result["query"][-1].content,
    }
@router.get("/chat/{thread_id}/history")
async def chat_history(thread_id: str, request: Request):
    chatbot = request.app.state.chatbot
    config = {"configurable": {"thread_id": thread_id}}
    state = await chatbot.aget_state(config)

    if not state or not state.values:
        raise HTTPException(status_code=404, detail="No history found for this thread_id")

    messages = state.values.get("query", [])
    return {
        "thread_id": thread_id,
        "messages": [
            {"role": "user" if isinstance(m, HumanMessage) else "assistant", "content": m.content}
            for m in messages
        ],
    }


@router.delete("/collection")
async def destroy_collection():
    persist_dir = settings.chroma_persist_directory
    try:
        shutil.rmtree(persist_dir, ignore_errors=False)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="No Chroma collection found to delete")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting collection: {e}")

    return {"message": "Chroma collection deleted successfully."}