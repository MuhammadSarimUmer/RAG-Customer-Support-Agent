from langgraph.graph import StateGraph, START,END
from typing import TypedDict, List, Optional,Annotated
from langgraph.graph.message import add_messages
from langchain.messages import SystemMessage,HumanMessage,AIMessage
from langchain_core.messages import BaseMessage
from langchain_chroma import Chroma
from core.settings import settings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from schemas.fastapi_models import SearchRequest
from core.llms import GenerationModel
import asyncio
from langchain_core.messages import AIMessage
from services.email_tool import send_email
from langgraph.types import interrupt
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver




class RAGState(TypedDict):
    query: Annotated[list[BaseMessage], add_messages]
    confidence: Optional[float]
    context: Optional[str]


async def search_node(state: RAGState) -> RAGState:
    model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token": settings.hf_token}
    )
    reranker = CrossEncoder("mixedbread-ai/mxbai-rerank-xsmall-v1")

    query_text = state['query'][-1].content
    db = Chroma(persist_directory=settings.chroma_persist_directory, embeddings=model)
    semantic_results = await asyncio.to_thread(
        db.similarity_search, query_text, k=10
    )
    
    if not semantic_results:
        return {
            'context': "",
            'confidence': 0.0
        }
        
    pairs = [[query_text, doc.page_content] for doc in semantic_results]
    scores = await asyncio.to_thread(reranker.predict, pairs)
    scored_docs = sorted(zip(scores, semantic_results), key=lambda x: x[0], reverse=True)
    highest_score = float(scored_docs[0][0])
    top_docs = [doc for score, doc in scored_docs[:4]]
    final_context_string = "\n\n---\n\n".join([doc.page_content for doc in top_docs])
    
    return {
        'context': final_context_string,
        'confidence': highest_score
    }
 
def router_node(state:RAGState):
    if state['confidence'] > 0.45:
        return 'generation_node'
    else:
        return 'email_node'
    
async def generation_node(state:RAGState):
    context = state['context']
    query = state['query'][-1].content
    messages = [
        SystemMessage(content=f"You are a helpful customer support agent. Use the context to answer the user's question. If the context does not contain relevant information, respond with 'I don't know'"),
        HumanMessage(content=f"answer the following the query:\n{query}\n\n strictly use the following context:\n{context}")
    ]
    response = await GenerationModel.ainvoke(messages)
    return{
        'query': [response],
    }

async def email_node(state: RAGState):
    query = state['query'][-1].content

    user_response = interrupt({
        "type": "confirm_escalation",
        "message": "I wasn't confident in my answer. Want me to escalate this to our support team? If so, share your email.",
        "query": query
    })

    if not user_response.get("confirmed"):
        return {
            'query': [AIMessage(content="No problem, I won't escalate. Let me know if there's anything else I can help with.")]
        }

    user_email = user_response.get("email", "unknown user")
    email_subject = f"Escalated Customer Query from {user_email}"
    email_body = (
        f"Dear Support Team,\n\n"
        f"We have received a customer query that requires your attention:\n\n"
        f"{query}\n\n"
        f"Customer email: {user_email}\n\n"
        f"Please address this query at your earliest convenience.\n\n"
        f"Best regards,\nCustomer Support Agent"
    )

    result = await send_email.ainvoke({
        "subject": email_subject,
        "body": email_body,
    })

    return {
        'user_email': user_email,
        'query': [AIMessage(content=f"I've escalated your query to our support team. {result}")]
    }

graph = StateGraph(RAGState)

graph.add_node('search_node',search_node)
graph.add_node('generation_node',generation_node)
graph.add_node('email_node',email_node)

graph.add_edge(START,'search_node')
graph.add_conditional_edges('search_node',router_node)
graph.add_edge('generation_node',END)
graph.add_edge('email_node',END)

