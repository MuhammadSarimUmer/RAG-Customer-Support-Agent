from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from core.settings import settings
from services.graph import graph          
from routers import rag as rag_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with AsyncSqliteSaver.from_conn_string(
        f"data/checkpoints.db"
    ) as checkpointer:
        app.state.chatbot = graph.compile(checkpointer=checkpointer)
        yield



app = FastAPI(title="RAG Customer Support Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(rag_router.router)


@app.get("/health")
async def health():
    return {"status": "ok"}