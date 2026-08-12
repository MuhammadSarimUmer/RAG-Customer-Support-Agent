from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    groq_api_key: str = Field(...,description="API key for Groq API")
    hf_token: str = Field(...,description="Hugging Face API token")
    chroma_persist_directory: str = Field(...,description="Directory to persist Chroma database")
    brevo_key: str = Field(...,description="API key for Brevo email service")
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()

    