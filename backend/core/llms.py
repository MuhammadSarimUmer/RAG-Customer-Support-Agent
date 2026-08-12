from langchain_groq import ChatGroq
from core.settings import settings
from services.email_tool import send_email

GenerationModel = ChatGroq(
    model='openai/gpt-oss-120b',
    temperature=0.5,
    api_key=settings.groq_api_key
)
tools = [send_email]

ToolsModel = GenerationModel.bind_tools(tools)



